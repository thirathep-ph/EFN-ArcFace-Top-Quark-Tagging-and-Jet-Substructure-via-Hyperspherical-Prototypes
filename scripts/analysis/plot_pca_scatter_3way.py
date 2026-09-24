import pathlib, json, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.utils.device import get_device
from arcefn.utils.paths import DATA_DIR
from arcefn.utils.model_loading import load_model_from_dir
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arcefn.utils.figure_style import apply_style, panel_label, add_grid, COLORS

MAX_EVENTS = 5000
MODEL_DIRS = {
    "ArcFace": "experiments/robustness_s16_m05",
    "CosLinear": "experiments/baselines_coslinear_seed_42",
    "Linear": "experiments/baselines_linear_seed_42",
}
LABELS = [r"ArcFace ($S^{63},\,m=0.5$)", r"CosLinear ($S^{63},\,m=0$)", r"Linear (Euclid.)"]

events_all, labels_all, _ = load_awkward(str(DATA_DIR / 'test.h5'), max_events=MAX_EVENTS, lazy=False)
labels_np = labels_all[:MAX_EVENTS]

apply_style()
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=False, sharey=False)
fig.subplots_adjust(wspace=0.28, left=0.06, right=0.98, top=0.86, bottom=0.18)

for col, (name, mdir) in enumerate(MODEL_DIRS.items()):
    device, _ = get_device()
    model = load_model_from_dir(pathlib.Path(mdir), device)
    model.eval()
    events, labels, _ = load_awkward(str(DATA_DIR / 'test.h5'), max_events=MAX_EVENTS, lazy=False)
    ds = JetTaggingDataset(events, labels, _)
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    embs = []
    with torch.no_grad():
        for x, y, w, mask, _ in loader:
            x, mask = x.to(device), mask.to(device)
            out = model(x, mask=mask)
            e = out[-1]
            e = F.normalize(e, p=2, dim=1)
            embs.append(e.cpu().numpy())
    embs = np.concatenate(embs, axis=0)
    pca = PCA(n_components=2)
    proj = pca.fit_transform(embs)
    var = pca.explained_variance_ratio_

    ax = axes[col]
    m_qcd = labels_np == 0
    m_top = labels_np == 1
    ax.scatter(proj[m_qcd, 0], proj[m_qcd, 1], s=7, alpha=0.45, c=COLORS["qcd"], edgecolors="none", rasterized=True)
    ax.scatter(proj[m_top, 0], proj[m_top, 1], s=7, alpha=0.45, c=COLORS["top"], edgecolors="none", rasterized=True)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    add_grid(ax)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"({chr(97+col)}) {LABELS[col]}\nPC1 {var[0]*100:.1f}% + PC2 {var[1]*100:.1f}% = {var.sum()*100:.1f}%", fontsize=7, pad=6)

# single legend outside
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["qcd"], markersize=6, alpha=0.8),
           Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["top"], markersize=6, alpha=0.8)]
fig.legend(handles, ["QCD", "Top"], loc="upper center", ncol=2, fontsize=8, frameon=True, bbox_to_anchor=(0.5, 1.02))

out = pathlib.Path("experiments/pca_scatter_3way.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"saved {out} {out.stat().st_size}")
plt.close(fig)
