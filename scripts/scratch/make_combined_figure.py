"""Make combined POC figure for paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
exp = _root / "experiments" / "poc_injection_scan.png"
mass = _root / "experiments" / "poc_mass_decoupled.png"

img_exp = mpimg.imread(str(exp))
img_mass = mpimg.imread(str(mass))

h_exp, w_exp = img_exp.shape[:2]
h_mass, w_mass = img_mass.shape[:2]

fig_w = 10
fig_h_top = fig_w * h_exp / w_exp
fig_h_bot = fig_w * h_mass / w_mass
fig_h = fig_h_top + fig_h_bot + 0.3

fig = plt.figure(figsize=(fig_w, fig_h))

ax_top = fig.add_axes([0, (fig_h_bot + 0.3) / fig_h, 1, fig_h_top / fig_h])
ax_top.imshow(img_exp)
ax_top.axis('off')

ax_bot = fig.add_axes([0, 0, 1, fig_h_bot / fig_h])
ax_bot.imshow(img_mass)
ax_bot.axis('off')

ax_top.text(-0.02, 1.02, '(a)', transform=ax_top.transAxes,
            fontsize=14, fontweight='bold', va='bottom')
ax_bot.text(-0.02, 1.02, '(b)', transform=ax_bot.transAxes,
            fontsize=14, fontweight='bold', va='bottom')

out = _root / "paper" / "figures" / "poc_combined.png"
fig.savefig(str(out), dpi=200, bbox_inches='tight', pad_inches=0.05)
plt.close()
print(f"Saved: {out}")
