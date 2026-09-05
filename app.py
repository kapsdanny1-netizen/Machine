#!/usr/bin/env python3
"""Lightweight web dashboard + inference playground for the kick detector.

Serves a single-page dashboard (metrics, McNemar, plots) and a live
inference API over the trained models:

    GET  /                     dashboard page
    GET  /api/meta             models, dataset range, metric tables
    GET  /plots/<name>.png     whitelisted result plots
    POST /api/predict          {model, center, window, perturb} -> probabilities

Run:  python app.py [--port 8000]      (binds 0.0.0.0)
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: E402
from src.data_io import derive_label, load_dataset  # noqa: E402
from src.features import derive_features, feature_matrix  # noqa: E402
from src.models import make_windows  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
for _name in ("src.features", "src.data_io"):
    logging.getLogger(_name).setLevel(logging.WARNING)  # quiet per-request logs
log = logging.getLogger("app")

# --------------------------------------------------------------------- state
CLASSICAL: dict[str, dict] = {}
LSTM_MODEL = None
DF_RAW: pd.DataFrame | None = None
Y_ALL: np.ndarray | None = None
META: dict = {}
PLOTS = [
    "kick_timeseries.png",
    "probability_timeline.png",
    "confusion_matrices.png",
    "roc_curves.png",
    "importance_random_forest.png",
    "importance_xgboost.png",
]
MODEL_CHOICES = [
    ("random_forest", "Random Forest"),
    ("xgboost", "XGBoost"),
    ("decision_tree", "Decision Tree"),
    ("svm", "SVM (RBF)"),
    ("knn", "KNN"),
    ("lstm", "LSTM"),
]
MAX_WINDOW = 201
PERTURB_LIMITS = {"fout": 0.5, "fpress": 2000.0, "atvol": 60.0}


def load_everything() -> None:
    """Load dataset, result artifacts and all six trained models once."""
    global DF_RAW, Y_ALL, LSTM_MODEL, META
    labeled = derive_label(load_dataset())
    Y_ALL = labeled[config.TARGET_COLUMN].to_numpy(dtype=int)
    DF_RAW = labeled.drop(columns=[config.TARGET_COLUMN]).reset_index(drop=True)

    for name in ("decision_tree", "random_forest", "svm", "knn", "xgboost"):
        with open(config.MODELS_DIR / f"best_{name}.pkl", "rb") as fh:
            CLASSICAL[name] = pickle.load(fh)

    import tensorflow as tf

    LSTM_MODEL = tf.keras.models.load_model(config.MODELS_DIR / "lstm.keras")

    metrics = pd.read_csv(config.RESULTS_DIR / "metrics.csv")
    mcnemar = pd.read_csv(config.RESULTS_DIR / "mcnemar.csv")
    onset_idx = int(np.argmax(Y_ALL == 1))
    META = {
        "models": [{"id": i, "name": n} for i, n in MODEL_CHOICES],
        "n_rows": int(len(DF_RAW)),
        "t_min": float(DF_RAW["WellDepth"].min()),
        "t_max": float(DF_RAW["WellDepth"].max()),
        "onset_index": onset_idx,
        "onset_time": float(DF_RAW["WellDepth"].iloc[onset_idx]),
        "md5": config.DATASET_MD5,
        "metrics": metrics.where(pd.notna(metrics), None).to_dict("records"),
        "mcnemar": mcnemar.where(pd.notna(mcnemar), None).to_dict("records"),
        "plots": PLOTS,
        "perturb_limits": PERTURB_LIMITS,
    }
    log.info("Loaded %d rows, 5 classical models + LSTM. Kick onset row %d (t=%.2f).",
             len(DF_RAW), onset_idx, META["onset_time"])


# ------------------------------------------------------------ ML prediction
def _scaler():
    return CLASSICAL["random_forest"]["scaler"]


def _scaled(rows: pd.DataFrame) -> np.ndarray:
    feats = feature_matrix(derive_features(rows)).to_numpy(dtype=float)
    return _scaler().transform(feats)


def _apply_perturb(rows: pd.DataFrame, perturb: dict) -> pd.DataFrame:
    out = rows.copy()
    if perturb.get("fout"):
        out["FOut"] = out["FOut"] + float(perturb["fout"])
    if perturb.get("fpress"):
        out["FPress"] = out["FPress"] + float(perturb["fpress"])
    if perturb.get("atvol"):
        out["ATVolume"] = out["ATVolume"] + float(perturb["atvol"])
    return out


def _lstm_proba_windows(scaled_block: np.ndarray) -> np.ndarray:
    """One P(kick) per row of ``scaled_block`` (each window ends at that row)."""
    idx = np.arange(len(scaled_block))
    Xw, _ = make_windows(scaled_block, np.zeros(len(scaled_block), dtype=int),
                         idx, stride=1)
    return LSTM_MODEL.predict(Xw, verbose=0).ravel()


def _proba_single(model_id: str, center: int, row: pd.DataFrame) -> float:
    """P(kick) from one model for a single (possibly perturbed) row."""
    if model_id == "lstm":
        start = max(0, center - config.LSTM_WINDOW + 1)
        block = pd.concat([DF_RAW.iloc[start:center], row])
        return float(_lstm_proba_windows(_scaled(block))[-1])
    blob = CLASSICAL[model_id]
    return float(blob["estimator"].predict_proba(_scaled(row))[:, 1][0])


def predict(payload: dict) -> dict:
    """Window probabilities + model consensus around a center row."""
    model_id = payload.get("model", "random_forest")
    if model_id not in CLASSICAL and model_id != "lstm":
        raise ValueError(f"unknown model: {model_id}")
    n = len(DF_RAW)
    center = max(0, min(n - 1, int(payload.get("center", META["onset_index"]))))
    window = int(payload.get("window", 101))
    window = min(MAX_WINDOW, max(5, window if window % 2 == 1 else window + 1))
    perturb = payload.get("perturb") or {}
    perturb = {k: max(-lim, min(lim, float(perturb.get(k, 0.0) or 0.0)))
               for k, lim in PERTURB_LIMITS.items()}

    half = window // 2
    start, end = max(0, center - half), min(n, center + half + 1)
    rows = DF_RAW.iloc[start:end]
    pert_rows = _apply_perturb(rows, perturb)

    p_center_base = _proba_single(model_id, center,
                                  DF_RAW.iloc[center:center + 1])

    if model_id == "lstm":
        ctx = DF_RAW.iloc[max(0, start - config.LSTM_WINDOW + 1):start]
        block = pd.concat([ctx, pert_rows])
        p_all = _lstm_proba_windows(_scaled(block))[len(ctx):]
    else:
        p_all = CLASSICAL[model_id]["estimator"].predict_proba(_scaled(pert_rows))[:, 1]

    pert_center_row = _apply_perturb(DF_RAW.iloc[center:center + 1], perturb)
    consensus = [{"model": nm, "p": round(_proba_single(mid, center, pert_center_row), 4)}
                 for mid, nm in MODEL_CHOICES]

    local_center = center - start
    return {
        "model": model_id,
        "center": center,
        "time": float(DF_RAW["WellDepth"].iloc[center]),
        "times": [round(float(t), 2) for t in pert_rows["WellDepth"]],
        "proba": [round(float(p), 4) for p in p_all],
        "actual": [int(v) for v in Y_ALL[start:end]],
        "p_center_base": round(p_center_base, 4),
        "p_center": round(float(p_all[local_center]), 4),
        "perturb": perturb,
        "consensus": consensus,
        "channels": {c: round(float(pert_center_row.iloc[0][c]), 4) for c in
                     ("FPress", "FIn", "FOut", "ATVolume", "MRFlow", "RoPen")},
    }


# -------------------------------------------------------------- HTTP layer
PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gas-Kick Detection — Live Dashboard</title>
<style>
 :root{--bg:#0d1b2a;--card:#13263b;--ink:#e8f1f8;--mut:#8fa8bf;--acc:#4cc9f0;
      --ok:#4ade80;--bad:#f87171;--line:#22405f}
 *{box-sizing:border-box} body{margin:0;font:14px/1.5 system-ui,sans-serif;
   background:var(--bg);color:var(--ink)}
 header{padding:18px 26px;border-bottom:1px solid var(--line);display:flex;
   gap:14px;align-items:baseline;flex-wrap:wrap}
 header h1{margin:0;font-size:19px} header .sub{color:var(--mut)}
 .badge{background:#1d3a57;border:1px solid var(--acc);color:var(--acc);
   padding:2px 10px;border-radius:12px;font-size:12px}
 main{max-width:1180px;margin:0 auto;padding:20px 26px 60px;display:grid;
   gap:20px}
 section{background:var(--card);border:1px solid var(--line);border-radius:12px;
   padding:16px 18px}
 h2{margin:0 0 10px;font-size:15px;color:var(--acc);text-transform:uppercase;
   letter-spacing:.08em}
 table{border-collapse:collapse;width:100%;font-size:13px}
 th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right}
 th:first-child,td:first-child{text-align:left}
 .win td{background:#12351f}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));
   gap:14px} .grid img{width:100%;border-radius:8px;border:1px solid var(--line);
   background:#fff}
 .ctrls{display:flex;gap:18px;flex-wrap:wrap;align-items:end;margin-bottom:12px}
 .ctrl label{display:block;color:var(--mut);font-size:12px;margin-bottom:4px}
 select,input[type=range]{accent-color:var(--acc)}
 select{background:#0e2236;color:var(--ink);border:1px solid var(--line);
   padding:6px 10px;border-radius:8px}
 .val{font-variant-numeric:tabular-nums;color:var(--acc)}
 .kpi{display:flex;gap:26px;flex-wrap:wrap;margin:6px 0 12px}
 .kpi div b{display:block;font-size:22px} .kpi div span{color:var(--mut);
   font-size:12px}
 .kick{color:var(--bad);font-weight:700} .nokick{color:var(--ok);font-weight:700}
 svg text{fill:var(--mut);font-size:10px}
 .foot{color:var(--mut);font-size:12px;margin-top:8px}
 .err{color:var(--bad)}
</style></head><body>
<header>
 <h1>Gas-Kick Detection — Live Preview</h1>
 <span class="sub">DataDRILL · Kick_Detection.csv</span>
 <span class="badge" id="md5">md5 …</span>
 <span class="sub" id="rows"></span>
</header>
<main>

<section>
 <h2>Model leaderboard (held-out test partition)</h2>
 <div class="kpi">
  <div><b id="k-winner">…</b><span>recommended model</span></div>
  <div><b id="k-recall">…</b><span>winner recall</span></div>
  <div><b id="k-onset">…</b><span>true kick onset, s</span></div>
 </div>
 <div style="overflow:auto"><table id="metrics"></table></div>
 <details style="margin-top:10px"><summary style="cursor:pointer;color:var(--mut)">
   Pairwise McNemar tests (15 pairs)</summary>
  <div style="overflow:auto;margin-top:8px"><table id="mcnemar"></table></div>
 </details>
</section>

<section>
 <h2>Live inference playground</h2>
 <div class="ctrls">
  <div class="ctrl"><label>Model</label>
   <select id="model"></select></div>
  <div class="ctrl"><label>Row window</label>
   <select id="window"><option>25</option><option selected>101</option>
    <option>201</option></select></div>
  <div class="ctrl"><label>Simulation time · <span class="val" id="tval"></span> s
   </label><input type="range" id="center" style="width:340px"></div>
  <div class="ctrl"><label>What-if Δ Flow-Out (<span class="val" id="v-fout">0</span>)</label>
   <input type="range" id="p-fout" min="-0.5" max="0.5" step="0.02" value="0"></div>
  <div class="ctrl"><label>Δ Formation press. (<span class="val" id="v-fpress">0</span>)</label>
   <input type="range" id="p-fpress" min="-2000" max="2000" step="25" value="0"></div>
  <div class="ctrl"><label>Δ Pit volume (<span class="val" id="v-atvol">0</span>)</label>
   <input type="range" id="p-atvol" min="-60" max="60" step="1" value="0"></div>
  <div class="ctrl"><label>&nbsp;</label><button id="reset">Reset</button></div>
 </div>
 <div class="kpi">
  <div><b class="val" id="p-base">…</b><span>P(kick) at row — unperturbed</span></div>
  <div><b class="val" id="p-now">…</b><span>P(kick) at row — current</span></div>
  <div><b id="verdict">…</b><span>verdict</span></div>
 </div>
 <div id="chart"></div>
 <div class="foot">Chart: P(kick) across the selected window (red = actual kick
  rows, dashed = 0.5 threshold, line = selected row). Perturbations apply to
  every row in the window — watch the model react around the onset.</div>
 <h2 style="margin-top:16px">Model consensus at selected row</h2>
 <div style="overflow:auto"><table id="consensus"></table></div>
 <div class="foot" id="channels"></div>
</section>

<section>
 <h2>Result plots</h2>
 <div class="grid" id="plots"></div>
</section>

</main>
<script>
const $=id=>document.getElementById(id);
let META=null;

async function init(){
  META=await (await fetch('api/meta')).json();
  $('md5').textContent='md5 verified · '+META.md5.slice(0,8)+'…';
  $('rows').textContent=META.n_rows+' rows · '+META.t_min.toFixed(2)+'–'+
    META.t_max.toFixed(2)+' s';
  $('k-onset').textContent=META.onset_time.toFixed(2);
  const win=META.metrics.find(r=>r.Model==='Random Forest');
  $('k-winner').textContent=win.Model; $('k-recall').textContent=win.Recall.toFixed(4);
  $('metrics').innerHTML=hdr(META.metrics)+META.metrics.map(r=>
    `<tr${r.Model===win.Model?' class="win":""}><td>${r.Model}</td>`+
    ['Accuracy','Precision','Recall','F1','ROC-AUC','CV Recall (train)'].map(k=>
    `<td>${r[k]==null?'—':r[k]}</td>`).join('')+'</tr>').join('');
  const mc=META.mcnemar;
  $('mcnemar').innerHTML=hdr(mc)+mc.map(r=>`<tr><td>${r.model_a}</td><td>${
    r.model_b}</td><td>${r.statistic}</td><td>${r.p_value}</td><td>${
    r['significant (p<0.05)']?'yes':'no'}</td></tr>`).join('');
  $('model').innerHTML=META.models.map(m=>
    `<option value="${m.id}"${m.id==='random_forest'?' selected':''}>${m.name}</option>`).join('');
  const c=$('center'); c.min=0; c.max=META.n_rows-1; c.value=META.onset_index;
  $('plots').innerHTML=META.plots.map(p=>
    `<img src="plots/${p}" alt="${p}" loading="lazy">`).join('');
  ['model','window','center','p-fout','p-fpress','p-atvol'].forEach(id=>
    $(id).addEventListener('input',run));
  $('reset').addEventListener('click',()=>{['p-fout','p-fpress','p-atvol']
    .forEach(id=>$(id).value=0);run();});
  run();
}
const hdr=rows=>'<tr>'+Object.keys(rows[0]).map(k=>`<th>${k}</th>`).join('')+'</tr>';

async function run(){
  const body={model:$('model').value,center:+$('center').value,
    window:+$('window').value,perturb:{fout:+$('p-fout').value,
    fpress:+$('p-fpress').value,atvol:+$('p-atvol').value}};
  $('v-fout').textContent=body.perturb.fout.toFixed(2);
  $('v-fpress').textContent=body.perturb.fpress.toFixed(0);
  $('v-atvol').textContent=body.perturb.atvol.toFixed(0);
  try{
    const r=await (await fetch('api/predict',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
    if(r.error){$('verdict').innerHTML='<span class="err">'+r.error+'</span>';return;}
    $('tval').textContent=r.time.toFixed(2);
    $('p-base').textContent=r.p_center_base.toFixed(3);
    $('p-now').textContent=r.p_center.toFixed(3);
    $('verdict').innerHTML=r.p_center>=0.5?'<span class="kick">KICK</span>'
      :'<span class="nokick">no kick</span>';
    draw(r);
    $('consensus').innerHTML='<tr><th>Model</th><th>P(kick)</th></tr>'+
      r.consensus.map(c=>`<tr><td>${c.model}</td><td>${c.p}</td></tr>`).join('');
    $('channels').textContent='Row channels — FPress '+r.channels.FPress+
      ' psi · FIn '+r.channels.FIn+' · FOut '+r.channels.FOut+
      ' · ATVolume '+r.channels.ATVolume+' bbl · MRFlow '+r.channels.MRFlow+
      ' · RoPen '+r.channels.RoPen;
  }catch(e){$('verdict').innerHTML='<span class="err">'+e+'</span>';}
}

function draw(r){
  const W=1080,H=260,P={l:44,r:14,t:12,b:26};
  const ts=r.times,ps=r.proba;
  const x=t=>P.l+(t-ts[0])/Math.max(1e-9,ts[ts.length-1]-ts[0])*(W-P.l-P.r);
  const y=p=>H-P.b-p*(H-P.t-P.b);
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  s+=`<rect x="${P.l}" y="${P.t}" width="${W-P.l-P.r}" height="${H-P.t-P.b}"
      fill="none" stroke="#22405f"/>`;
  s+=`<line x1="${P.l}" x2="${W-P.r}" y1="${y(.5)}" y2="${y(.5)}"
      stroke="#8fa8bf" stroke-dasharray="4 4"/>`;
  const ci=Math.floor(ts.length/2);
  s+=`<line x1="${x(ts[ci])}" x2="${x(ts[ci])}" y1="${P.t}" y2="${H-P.b}"
      stroke="#4cc9f0" stroke-width="1.5"/>`;
  for(let i=0;i<ts.length;i++) if(r.actual[i])
    s+=`<circle cx="${x(ts[i])}" cy="${H-P.b-3}" r="1.8" fill="#f87171"/>`;
  s+=`<polyline fill="none" stroke="#4cc9f0" stroke-width="2" points="${
      ts.map((t,i)=>x(t).toFixed(1)+','+y(ps[i]).toFixed(1)).join(' ')}"/>`;
  s+=`<text x="${P.l}" y="${P.t+8}">1.0</text><text x="${P.l}" y="${y(.5)-3}">0.5</text>
      <text x="${P.l}" y="${H-P.b-3}">0.0</text>
      <text x="${P.l}" y="${H-6}">${ts[0]}</text>
      <text x="${W-P.r-40}" y="${H-6}">${ts[ts.length-1]}</text>`;
  s+='</svg>'; $('chart').innerHTML=s;
}
init();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet default access log
        log.debug(fmt, *args)

    # -- helpers -----------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    # -- routes ------------------------------------------------------------
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/api/meta":
            self._json(META)
        elif path.startswith("/plots/"):
            name = Path(path).name
            if name not in PLOTS:
                return self._json({"error": "unknown plot"}, 404)
            blob = (config.RESULTS_DIR / name).read_bytes()
            self._send(200, blob, "image/png")
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            return self._json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._json(predict(payload))
        except Exception as exc:  # surface actionable errors to the client
            log.error("predict failed:\n%s", traceback.format_exc())
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    load_everything()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info("Dashboard listening on http://%s:%d", args.host, args.port)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
