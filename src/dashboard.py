"""Read-only local dashboard: photos, boxes, predictions, scores, pipeline status.

Run from the repo root: .venv/bin/python src/dashboard.py  ->  http://localhost:8501
Re-reads files on every request. Never writes data, never calls a model or an API.
"""
import csv
import io
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LOG_DIR = Path("runs/judge_logs")
DETECT_SIDE = 2400  # detect.py resizes so the longest side is 2400px
VIEW_SIDE = 1200
IMG_EXT = {".heic", ".jpg", ".jpeg", ".png"}

sys.path.insert(0, str(ROOT / "src"))
try:
    import score as scorer  # noqa: E402
except Exception as e:  # missing deps should not take the dashboard down
    scorer, SCORER_ERROR = None, repr(e)

app = FastAPI(title="bottle-vision dashboard")
_jpeg_cache: dict[tuple[str, int, float], bytes] = {}


def read_csv(path: Path) -> list[dict]:
    """Rows of a CSV, tolerating a missing file or a half-written last line."""
    try:
        with path.open(newline="") as f:
            return [r for r in csv.DictReader(f) if r and None not in r and None not in r.values()]
    except (FileNotFoundError, csv.Error, UnicodeDecodeError):
        return []


def photos() -> list[Path]:
    raw = DATA / "raw"
    return sorted(p for p in raw.iterdir() if p.suffix.lower() in IMG_EXT) if raw.exists() else []


def photo_path(name: str) -> Path:
    for p in photos():
        if p.name == name or p.stem == name:
            return p
    raise HTTPException(404, f"no photo {name}")


def inventory() -> list[dict]:
    return read_csv(DATA / "labelled" / "inventory.csv")


def prediction_files() -> list[Path]:
    return sorted(DATA.glob("predictions*.csv"))


def parse_bbox(s: str):
    try:
        b = json.loads(s)
        return [float(v) for v in b] if isinstance(b, list) and len(b) == 4 else None
    except (ValueError, TypeError):
        return None


@app.get("/photo/{name}")
def photo_jpeg(name: str, side: int = VIEW_SIDE):
    path = photo_path(name)
    side = max(100, min(side, VIEW_SIDE))
    key = (path.name, side, path.stat().st_mtime)
    if key not in _jpeg_cache:
        img = Image.open(path).convert("RGB")
        img.thumbnail((side, side))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        _jpeg_cache[key] = buf.getvalue()
    return Response(_jpeg_cache[key], media_type="image/jpeg")


@app.get("/api/photos")
def api_photos():
    out = []
    for p in photos():
        try:
            w, h = Image.open(p).size
        except Exception:
            w = h = None
        out.append({"name": p.name, "stem": p.stem, "width": w, "height": h})
    return out


@app.get("/api/inventory")
def api_inventory():
    return inventory()


@app.get("/api/photo/{name}")
def api_photo(name: str):
    path = photo_path(name)
    w, h = Image.open(path).size
    scale = max(w, h) / DETECT_SIDE
    labels = {(r["photo"], r["crop"]): r for r in read_csv(DATA / "crops" / "labels.csv")}
    detector = []
    for b in read_csv(DATA / "crops" / "boxes.csv"):
        if b["photo"] != path.stem:
            continue
        try:
            x1, y1, x2, y2 = (float(b[k]) * scale for k in ("x1", "y1", "x2", "y2"))
        except ValueError:
            continue
        lab = labels.get((b["photo"], b["crop"]))
        item = int(lab["item_id"]) if lab and lab["item_id"].lstrip("-").isdigit() else None
        detector.append({"crop": b["crop"], "score": b["score"], "bbox": [x1, y1, x2 - x1, y2 - y1], "item_id": item})

    sources = {}
    for f in prediction_files():
        runs = defaultdict(list)
        for r in read_csv(f):
            if r.get("image") in (path.name, path.stem):
                runs[r.get("run", "1")].append(r | {"bbox": parse_bbox(r.get("bbox", ""))})
        sources[f.stem] = dict(sorted(runs.items()))
    return {"name": path.name, "width": w, "height": h, "detector": detector, "sources": sources}


@app.get("/api/scores")
def api_scores():
    if scorer is None:
        return {"error": f"could not import src/score.py: {SCORER_ERROR}", "files": []}
    inv = inventory()
    files = []
    for f in prediction_files():
        by_photo = defaultdict(list)
        for r in read_csv(f):
            if r.get("image"):
                by_photo[(r.get("run", "1"), r["image"])].append(r)
        if not by_photo or not inv:
            files.append({"file": f.name, "error": "no rows" if inv else "no inventory"})
            continue
        try:
            scores = {k: scorer.score_photo(v, inv) for k, v in by_photo.items()}
        except Exception as e:
            files.append({"file": f.name, "error": repr(e)})
            continue
        metrics = list(next(iter(scores.values())))
        runs = defaultdict(list)
        for (run, _), s in scores.items():
            runs[run].append(s)
        # same aggregation as score.py: mean per run, then mean across runs
        means = {m: statistics.mean(statistics.mean(s[m] for s in ph) for ph in runs.values()) for m in metrics}
        files.append({"file": f.name, "runs": len(runs), "photos": len({img for _, img in scores}),
                      "rows": sum(len(v) for v in by_photo.values()), "means": means})
    return {"files": files}


def count_files(d: Path, pattern: str = "*") -> int | None:
    return sum(1 for p in d.rglob(pattern) if p.is_file() and not p.name.startswith(".")) if d.exists() else None


@app.get("/api/status")
def api_status():
    preds = DATA / "predictions.csv"
    logs = []
    if LOG_DIR.exists():
        for p in sorted(LOG_DIR.glob("*.log")):
            try:
                lines = p.read_text(errors="replace").splitlines()
            except OSError:
                lines = []
            logs.append({"name": p.name, "lines": len(lines), "tail": "\n".join(lines[-20:])})
    return {
        "counts": {
            "data/raw photos": len(photos()),
            "data/crops images": count_files(DATA / "crops", "*.jpg"),
            "data/catalogue/img files": count_files(DATA / "catalogue" / "img"),
            "data/predictions.csv rows": len(read_csv(preds)) if preds.exists() else None,
        },
        "prediction_files": [{"file": f.name, "rows": len(read_csv(f))} for f in prediction_files()],
        "logs": logs,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>bottle-vision</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f7f7f5;color:#222;margin:0;padding:16px 24px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:24px 0 8px}
.muted{color:#777;font-size:13px}
nav a{margin-right:16px;color:#246;cursor:pointer;text-decoration:none;font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}
.card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:6px;cursor:pointer;font-size:13px}
.card img{width:100%;display:block;border-radius:4px}
table{border-collapse:collapse;background:#fff;font-size:13px}
th,td{border:1px solid #ddd;padding:4px 8px;text-align:left;vertical-align:top}th{background:#eee}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.detail{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}
.viewer{flex:1 1 600px;min-width:300px}.viewer svg{width:100%;height:auto;background:#fff;border:1px solid #ddd}
.side{flex:1 1 420px;min-width:300px;overflow-x:auto}
.controls{margin:8px 0;display:flex;gap:12px;flex-wrap:wrap;align-items:center;font-size:14px}
.legend span{display:inline-block;margin:2px 8px 2px 0;font-size:12px}
.sw{display:inline-block;width:10px;height:10px;margin-right:4px;vertical-align:middle}
pre{background:#fff;border:1px solid #ddd;padding:8px;font-size:12px;overflow-x:auto;max-height:300px}
.err{color:#a00}
</style></head><body>
<h1>bottle-vision</h1><div class="muted">Read-only. Every view re-reads the files. <span id="when"></span></div>
<nav style="margin:12px 0"><a onclick="show('photos')">Photos</a><a onclick="show('scores')">Scores</a><a onclick="show('status')">Pipeline status</a></nav>
<div id="main"></div>
<script>
const COLORS=['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4','#42d4f4','#f032e6','#9a6324','#469990','#800000','#808000','#000075'];
const itemColor=id=>id>0?COLORS[(id-1)%COLORS.length]:'#999';
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const $=id=>document.getElementById(id);
const get=u=>fetch(u,{cache:'no-store'}).then(r=>r.ok?r.json():Promise.reject(r.status+' '+u));
let INV=[],DETAIL=null,state={source:'detector',run:null};

function show(view,arg){
  location.hash=view+(arg?'/'+arg:'');
  $('when').textContent='Loaded '+new Date().toLocaleTimeString();
  ({photos:viewPhotos,photo:viewPhoto,scores:viewScores,status:viewStatus}[view]||viewPhotos)(arg).catch(e=>$('main').innerHTML='<p class=err>'+esc(e)+'</p>');
}
async function viewPhotos(){
  const ps=await get('/api/photos');
  $('main').innerHTML=ps.length?'<div class=grid>'+ps.map(p=>`<div class=card onclick="show('photo','${esc(p.name)}')"><img loading=lazy src="/photo/${encodeURIComponent(p.name)}?side=400"><div>${esc(p.name)}</div><div class=muted>${p.width}x${p.height}</div></div>`).join('')+'</div>':'<p>No photos in data/raw.</p>';
}
async function viewPhoto(name){
  [DETAIL,INV]=await Promise.all([get('/api/photo/'+encodeURIComponent(name)),get('/api/inventory')]);
  const srcs=['detector',...Object.keys(DETAIL.sources)];
  if(!srcs.includes(state.source))state.source='detector';
  $('main').innerHTML=`<div class=controls><a onclick="show('photos')" style="cursor:pointer;color:#246">&larr; all photos</a><b>${esc(DETAIL.name)}</b>
  <label>Boxes <select id=src onchange="state.source=this.value;state.run=null;renderDetail()">${srcs.map(s=>`<option ${s==state.source?'selected':''} value="${esc(s)}">${s=='detector'?'local detector (hand labels)':esc(s)}</option>`).join('')}</select></label>
  <label id=runwrap>Run <select id=run onchange="state.run=this.value;renderDetail()"></select></label></div>
  <div class=detail><div class=viewer id=viewer></div><div class=side id=side></div></div>`;
  renderDetail();
}
function renderDetail(){
  const d=DETAIL,W=d.width,H=d.height,lw=Math.max(W,H)/300;
  let boxes=[],table='';
  if(state.source=='detector'){
    $('runwrap').style.display='none';
    boxes=d.detector.map(b=>({bbox:b.bbox,color:itemColor(b.item_id),dash:b.item_id==-1,label:b.item_id>0?b.item_id:(b.item_id==0?'x':'?')}));
    const name=id=>{const i=INV.find(r=>+r.id==id);return id==0?'clutter':id==-1?'fragment':id==null?'unlabelled':i?`${i.brand} ${i.product}`:'item '+id};
    table=`<h2>Detector crops (${d.detector.length})</h2><table><tr><th>crop</th><th>score</th><th>label</th></tr>${d.detector.map(b=>`<tr><td class=n>${b.crop}</td><td class=n>${b.score}</td><td><span class=sw style="background:${itemColor(b.item_id)}"></span>${esc(name(b.item_id))}</td></tr>`).join('')}</table>`;
  }else{
    const runs=d.sources[state.source]||{},keys=Object.keys(runs);
    if(!keys.includes(state.run))state.run=keys[0]||null;
    $('runwrap').style.display=keys.length>1?'':'none';
    $('run').innerHTML=keys.map(k=>`<option ${k==state.run?'selected':''}>${esc(k)}</option>`).join('');
    const rows=runs[state.run]||[];
    boxes=rows.filter(r=>r.bbox).map((r,i)=>({bbox:r.bbox,color:COLORS[i%COLORS.length],label:r.position}));
    table=`<h2>Predicted (${rows.length})</h2>`+(rows.length?`<table><tr><th>#</th><th>brand</th><th>product</th><th>size</th><th>unit</th><th>fill_ml</th><th>id_source</th></tr>${rows.map((r,i)=>`<tr><td><span class=sw style="background:${COLORS[i%COLORS.length]}"></span>${esc(r.position)}</td><td>${esc(r.brand)}</td><td>${esc(r.product)}</td><td class=n>${esc(r.size)}</td><td>${esc(r.unit)}</td><td class=n>${esc(r.fill_ml)}</td><td>${esc(r.id_source)}</td></tr>`).join('')}</table>`:'<p class=muted>No rows for this photo.</p>');
  }
  table+=`<h2>Ground truth (${INV.length})</h2><table><tr><th>id</th><th>brand</th><th>product</th><th>size</th><th>unit</th></tr>${INV.map(r=>`<tr><td><span class=sw style="background:${itemColor(+r.id)}"></span>${esc(r.id)}</td><td>${esc(r.brand)}</td><td>${esc(r.product)}</td><td class=n>${esc(r.size)}</td><td>${esc(r.unit)}</td></tr>`).join('')}</table>`;
  const fs=Math.max(W,H)/60;
  $('viewer').innerHTML=`<svg viewBox="0 0 ${W} ${H}"><image href="/photo/${encodeURIComponent(d.name)}" width="${W}" height="${H}"/>${boxes.map(b=>{const[x,y,w,h]=b.bbox;return `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="none" stroke="${b.color}" stroke-width="${lw}" ${b.dash?`stroke-dasharray="${lw*3}"`:''}/><text x="${x+lw}" y="${y+fs}" font-size="${fs}" fill="${b.color}" stroke="#fff" stroke-width="${fs/12}" paint-order="stroke" font-weight="bold">${esc(b.label)}</text>`}).join('')}</svg>`;
  $('side').innerHTML=table;
}
async function viewScores(){
  const s=await get('/api/scores');
  if(s.error){$('main').innerHTML='<p class=err>'+esc(s.error)+'</p>';return}
  const ms=[...new Set(s.files.flatMap(f=>Object.keys(f.means||{})))];
  $('main').innerHTML=`<h2>Scores (src/score.py logic, mean per run then across runs)</h2><table><tr><th>file</th><th>runs</th><th>photos</th><th>rows</th>${ms.map(m=>`<th>${m}</th>`).join('')}</tr>${s.files.map(f=>f.error?`<tr><td>${esc(f.file)}</td><td colspan=${3+ms.length} class=err>${esc(f.error)}</td></tr>`:`<tr><td>${esc(f.file)}</td><td class=n>${f.runs}</td><td class=n>${f.photos}</td><td class=n>${f.rows}</td>${ms.map(m=>`<td class=n>${f.means[m]?.toFixed(3)??''}</td>`).join('')}</tr>`).join('')}</table>
  <p class=muted>Local files: precision is always 1.0 and size/category just repeat ID accuracy (see CLAUDE.md).</p>`;
}
async function viewStatus(){
  const s=await get('/api/status');
  $('main').innerHTML=`<h2>Counts</h2><table>${Object.entries(s.counts).map(([k,v])=>`<tr><td>${esc(k)}</td><td class=n>${v??'<span class=muted>missing</span>'}</td></tr>`).join('')}${s.prediction_files.map(f=>`<tr><td>data/${esc(f.file)} rows</td><td class=n>${f.rows}</td></tr>`).join('')}</table>
  <h2>Logs</h2>${s.logs.length?s.logs.map(l=>`<div><b>${esc(l.name)}</b> <span class=muted>${l.lines} lines, last 20</span><pre>${esc(l.tail)}</pre></div>`).join(''):'<p class=muted>No logs.</p>'}`;
}
const [v,...a]=location.hash.slice(1).split('/');show(v||'photos',decodeURIComponent(a.join('/'))||undefined);
</script></body></html>"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8501)
