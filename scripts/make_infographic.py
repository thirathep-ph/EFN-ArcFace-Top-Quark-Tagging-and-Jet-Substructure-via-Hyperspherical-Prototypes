"""Build infographic v3 HTML (self-contained, base64 real figures) for PNG export.

Usage: .\\.venv\\Scripts\\python.exe scripts/make_infographic_v3.py
Output: docs/infographic.html  (then export with concept-to-image render script)
"""
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "figures"

PANELS = [
    ("final_roc_curves.png", "(a) ROC — AUC 0.974, accuracy 91.84% [91.76, 91.92], rejection 124x (RF leads: 92.37%)"),
    ("dglap_fit.png", "(b) zg fits — QCD-Core -1.242 / QCD-Edge -1.022 / Top-Core -0.493 (Top-Edge excluded)"),
    ("lund_status_tp.png", "(c) True-Top Lund plane — clear 3-prong structure"),
    ("hypersphere_3d.png", "(d) Geometry — embeddings collapse to a 2-D disk, prototypes 43.6 deg apart"),
]

STEPS = [
    ("Constituents", "(E, px, py, pz) — up to 200 / jet"),
    ("EFN backbone", "Phi [2-128-128-128], energy pool, rho [128-128-64] — 59,072 params"),
    ("Hypersphere", "L2 norm onto S^63 — ArcFace margin m=0.5, scale s=16"),
    ("Prototypes", "nearest of 2 decides: Top / QCD — the rule IS the explanation"),
]

TOOLS = [
    ("FastJet", "jet finding"),
    ("PyTorch", "models + training"),
    ("scikit-learn", "clustering + RF"),
    ("awkward", "data IO"),
    ("matplotlib", "figures"),
    ("PySR", "distillation"),
]


def b64(name):
    return base64.b64encode((FIG / name).read_bytes()).decode()


panels_html = "\n".join(
    f"""<div class="panel"><img src="data:image/png;base64,{b64(n)}"><div class="cap">{c}</div></div>"""
    for n, c in PANELS
)
steps_html = "\n".join(
    f"""<div class="step"><div class="n">{i + 1}</div><div><b>{t}</b><span>{s}</span></div></div>"""
    for i, (t, s) in enumerate(STEPS)
)
tools_html = "\n".join(f"""<div class="tool"><b>{t}</b><span>{s}</span></div>""" for t, s in TOOLS)

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<style>
*,*::before,*::after{{margin:0;padding:0;box-sizing:border-box}}
:root{{--bg:#f8f6f1;--ink:#16161f;--teal:#0e7490;--brick:#b3341f;--mut:#5b5b66;--line:#d8d3c8;
--f-d:'Georgia','Times New Roman',serif;--f-b:'Helvetica Neue','Arial',sans-serif;--f-m:'Courier New','Consolas',monospace}}
body{{background:#e5e5e5;display:flex;justify-content:center;align-items:center;min-height:100vh}}
.canvas{{width:1600px;height:1000px;background:var(--bg);position:relative;overflow:hidden;display:flex;color:var(--ink);font-family:var(--f-b)}}
.left{{width:560px;padding:44px 40px 32px 48px;display:flex;flex-direction:column;border-right:2px solid var(--line)}}
.kick{{font-family:var(--f-m);font-size:13px;letter-spacing:3px;color:var(--teal);font-weight:bold}}
h1{{font-family:var(--f-d);font-size:46px;line-height:1.08;margin:10px 0 8px;font-weight:bold}}
.sub{{font-size:15px;color:var(--mut);line-height:1.45}}
.flow{{margin-top:18px;display:flex;flex-direction:column}}
.step{{display:flex;gap:12px;align-items:flex-start;padding:9px 0;border-top:1px solid var(--line)}}
.step .n{{flex:0 0 26px;height:26px;border-radius:50%;background:var(--ink);color:#fff;font-family:var(--f-m);font-size:14px;font-weight:bold;display:flex;align-items:center;justify-content:center;margin-top:1px}}
.step b{{font-size:15px;display:block}}
.step span{{font-family:var(--f-m);font-size:12px;color:var(--mut);display:block;margin-top:2px}}
.hero{{margin-top:16px;background:var(--ink);color:#fff;padding:14px 18px}}
.hero .v{{font-family:var(--f-m);font-size:40px;font-weight:bold}}
.hero .v small{{font-size:16px;font-weight:normal}}
.hero .v2{{font-family:var(--f-m);font-size:17px;margin-top:4px}}
.hero .l{{font-size:12.5px;color:#c9c9d2;margin-top:2px}}
.tools{{margin-top:14px;display:flex;flex-wrap:wrap;gap:7px}}
.tool{{border:1px solid var(--line);background:#fff;padding:5px 9px;font-size:12px}}
.tool b{{font-family:var(--f-m)}}
.tool span{{color:var(--mut);margin-left:6px}}
.foot{{margin-top:auto;font-size:11.5px;color:var(--mut);line-height:1.5;border-top:1px solid var(--line);padding-top:10px}}
.right{{flex:1;padding:28px 32px;display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:18px}}
.panel{{background:#fff;border:1px solid var(--line);display:flex;flex-direction:column;overflow:hidden}}
.panel img{{width:100%;height:352px;object-fit:contain;background:#fff}}
.panel .cap{{border-top:1px solid var(--line);padding:8px 12px;font-size:13px;line-height:1.4}}
.panel .cap b{{font-family:var(--f-m)}}
.url{{position:absolute;right:34px;bottom:10px;font-family:var(--f-m);font-size:11px;color:var(--mut)}}
</style></head><body><div class="canvas">
<div class="left">
<div class="kick">EFN-ARCFACE &middot; TOP QUARK TAGGING</div>
<h1>Nearest prototype on a hypersphere.</h1>
<div class="sub">An IRC-safe Energy Flow Network plus an ArcFace head &mdash; the binary decision rule itself is the explanation. Every panel on the right is a real result figure.</div>
<div class="flow">{steps_html}</div>
<div class="hero"><div class="v">91.84% <small>[91.76, 91.92]</small></div><div class="v2">AUC 0.974 &middot; rejection 124&times; &middot; 404k test jets</div><div class="l">random forest still leads at 92.37% &mdash; stated openly</div></div>
<div class="tools">{tools_html}</div>
<div class="foot">Simulation-only (Pythia 8, no detector) &middot; Top-Edge excluded &middot; null controls + single-seed shuffle disclosed &middot; all 48 numbers machine-checked vs tracked artifacts</div>
</div>
<div class="right">{panels_html}</div>
<div class="url">github.com/thirathep-ph/EFN-ArcFace-Top-Quark-Tagging-and-Jet-Substructure-via-Hyperspherical-Prototypes</div>
</div></body></html>"""

out = ROOT / "docs" / "infographic.html"
out.write_text(html, encoding="utf-8")
print("wrote", out, round(len(html) / 1024), "KB")
