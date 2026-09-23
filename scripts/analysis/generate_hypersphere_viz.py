"""
Generate hypersphere visualization for EFN+ArcFace embeddings.

For 2D embeddings: plots raw 2D embeddings with class/subclass coloring.
For >2D embeddings: projects to S^2 via top-3 SVD (as before).

Saves: ``MODEL_DIR / 'hypersphere_3d.png'``
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import torch
import os

from arcefn.utils.device import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Generate hypersphere visualization")
    parser.add_argument("--model_dir", type=str, default="experiments/robustness_s16_m05",
                        help="Directory containing model checkpoint and embeddings")
    parser.add_argument("--embedding_dim", type=int, default=None,
                        help="Embedding dimension (unused, kept for consistency)")
    return parser.parse_args()


N_POINTS = 50000
SEED = 42

SUBCLASS_NAMES = {0: "QCD core", 1: "QCD edge", 2: "Top core", 3: "Top edge"}
SUBCLASS_COLORS = {0: "#377eb8", 1: "#9ecae1", 2: "#e41a1c", 3: "#fdae6b"}
CLASS_COLORS = {0: "#377eb8", 1: "#e41a1c"}
CLASS_NAMES = {0: "QCD", 1: "Top"}


def slerp(p0, p1, n=300):
    t = np.linspace(0, 1, n)[:, None]
    om = np.arccos(np.clip(p0 @ p1, -1.0, 1.0))
    return (np.sin((1 - t) * om) * p0 + np.sin(t * om) * p1) / np.sin(om)


def panel_title(ax, letter, title):
    ax.set_title(r"$\mathbf{(%s)}$ %s" % (letter, title), loc="left",
                 fontsize=10.5, pad=6)


def main():
    args = parse_args()
    
    model_dir = args.model_dir
    embeddings_path = os.path.join(model_dir, "embeddings.npz")
    checkpoint_path = os.path.join(model_dir, "checkpoint.pt")
    subclass_labels_path = os.path.join(os.path.dirname(model_dir), "physics_sr", "subclass_labels.npz")
    out_path = os.path.join(model_dir, "hypersphere_3d.png")
    
    N_POINTS = 50000
    SEED = 42

    SUBCLASS_NAMES = {0: "QCD core", 1: "QCD edge", 2: "Top core", 3: "Top edge"}
    SUBCLASS_COLORS = {0: "#377eb8", 1: "#9ecae1", 2: "#e41a1c", 3: "#fdae6b"}
    CLASS_COLORS = {0: "#377eb8", 1: "#e41a1c"}
    CLASS_NAMES = {0: "QCD", 1: "Top"}

    emb_data = np.load(embeddings_path)
    emb = emb_data["embeddings"]
    labels = emb_data["labels"]
    sub_cls = np.load(subclass_labels_path)
    sub = sub_cls["sub"] + 2 * sub_cls["cls"]

    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)

    device, _ = get_device()
    sd = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "arcface_head.class_centers" not in sd:
        sd = sd.get("model_state_dict", sd)
    centers = sd["arcface_head.class_centers"].cpu().numpy()
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-10

    rng = np.random.default_rng(SEED)
    n_per_class = N_POINTS // 2
    idx = []
    for cls in [0, 1]:
        cls_idx = np.where(labels == cls)[0]
        idx.append(rng.choice(cls_idx, n_per_class, replace=False))
    idx = np.concatenate(idx)
    idx.sort()

    sv_full = np.linalg.svd(emb[idx], full_matrices=False, compute_uv=False)
    sv_ratio = sv_full[:6] / sv_full[0]
    var_frac = np.cumsum(sv_full**2) / np.sum(sv_full**2)

    # Handle 2D vs higher-dimensional embeddings
    if emb.shape[1] == 2:
        # For 2D embeddings, just use the raw 2D coordinates
        pts = emb[idx]
        c3 = centers
        plot_2d = True
    else:
        # For higher dimensions, project to 3D via SVD
        _, _, vt = np.linalg.svd(emb[idx], full_matrices=False)
        v3 = vt[:3].T
        pts = emb[idx] @ v3
        pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-10
        c3 = centers @ v3
        c3 /= np.linalg.norm(c3, axis=1, keepdims=True) + 1e-10
        plot_2d = False

    lab = labels[idx]
    sub_ = sub[idx]

    geo_rad = np.arccos(np.clip(c3[0] @ c3[1], -1.0, 1.0))
    geo_deg = np.degrees(geo_rad)

    if not plot_2d:
        arc = slerp(c3[0], c3[1])

    from arcefn.utils.figure_style import apply_style
    apply_style()  # same rcParams as below (DejaVu Serif, 10, lw 0.6); unified theme entry point

    if plot_2d:
        # 2D plot: single panel with class colors
        fig, ax = plt.subplots(figsize=(6, 6))
        
        for cls, color in CLASS_COLORS.items():
            m = lab == cls
            ax.scatter(pts[m, 0], pts[m, 1], s=1, alpha=0.4, c=color, rasterized=True)
        ax.scatter(c3[:, 0], c3[:, 1], s=60, c="k", zorder=6)
        
        # Draw circle
        circle = plt.Circle((0, 0), 1, fill=False, color="0.45", alpha=0.5, linewidth=0.5)
        ax.add_artist(circle)
        
        # Angle annotation
        mid = (c3[0] + c3[1]) / 2
        ax.text(mid[0], mid[1] + 0.15, f"${geo_deg:.1f}^\\circ$", fontsize=9, ha="center", va="bottom")
        
        ax.set_aspect('equal')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_title("2D Hypersphere Embeddings (Class)")
        
        # Legend
        cls_handles = [Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=c, markersize=6,
                              label=CLASS_NAMES[i])
                       for i, c in CLASS_COLORS.items()]
        proto_handle = Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="k", markersize=6,
                              label="Prototypes")
        cls_handles.append(proto_handle)
        ax.legend(handles=cls_handles, fontsize=8, frameon=True,
                  edgecolor="0.45", facecolor="white", framealpha=0.92,
                  ncol=2, columnspacing=1.2, handletextpad=0.4,
                  borderpad=0.4, loc="upper right",
                  borderaxespad=0)
        panel_title(ax, "a", "2D Embeddings (Class)")
        
        # Add singular value spectrum as inset
        axins = ax.inset_axes((0.55, 0.55, 0.4, 0.4))
        comps = np.arange(1, min(7, len(sv_ratio)+1))
        bars = axins.bar(comps, sv_ratio[:len(comps)], width=0.6,
                         color=["#377eb8", "#377eb8", "#b0b0b0", "#b0b0b0",
                                "#cccccc", "#cccccc"][:len(comps)],
                         edgecolor="white", linewidth=0.5)
        axins.set_xlabel("Component", fontsize=7)
        axins.set_ylabel(r"$\sigma_i / \sigma_1$", fontsize=7)
        axins.set_xticks(comps)
        axins.set_ylim(0, 1.1)
        axins.spines["top"].set_visible(False)
        axins.spines["right"].set_visible(False)
        
        out = os.path.join(model_dir, "hypersphere_3d.png")
        fig.savefig(str(out), dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    else:
        # Original 3D visualization for >2D
        def draw_sphere(ax, r=1.0, alpha=0.18, color="0.45"):
            u = np.linspace(0, 2 * np.pi, 60)
            v = np.linspace(0, np.pi, 30)
            x = r * np.outer(np.cos(u), np.sin(v))
            y = r * np.outer(np.sin(u), np.sin(v))
            z = r * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=0.35,
                              rstride=4, cstride=4)

        arc = slerp(c3[0], c3[1])

        plt.rcParams.update({"font.family": "DejaVu Serif",
                             "font.size": 10,
                             "axes.linewidth": 0.6})

        fig = plt.figure(figsize=(9.5, 4.8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1],
                              wspace=0.18, left=0.02, right=0.98,
                              top=0.92, bottom=0.02)
        ax1 = fig.add_subplot(gs[0], projection="3d")
        ax2 = fig.add_subplot(gs[1], projection="3d")

        for ax in (ax1, ax2):
            draw_sphere(ax)
            ax.set_box_aspect([1, 1, 1])
            ax.view_init(elev=40, azim=-53)
            ax.set_axis_off()

        proto_handle = Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="k", markersize=6,
                              label="Prototypes")

        for cls, color in CLASS_COLORS.items():
            m = lab == cls
            ax1.scatter(pts[m, 0], pts[m, 1], pts[m, 2], s=1, alpha=0.4,
                        c=color, rasterized=True)
        ax1.scatter(c3[:, 0], c3[:, 1], c3[:, 2], s=60, c="k", zorder=6)
        ax1.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="0.25",
                 linewidth=1.4, alpha=0.9, linestyle=(0, (4, 2)))

        mid = arc[150]
        ax1.text(mid[0], mid[1], mid[2] + 0.17,
                 f"${geo_deg:.1f}^\\circ$", fontsize=9, ha="center", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9))

        cls_handles = [Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=c, markersize=6,
                              label=CLASS_NAMES[i])
                       for i, c in CLASS_COLORS.items()]
        cls_handles.append(proto_handle)
        ax1.legend(handles=cls_handles, fontsize=8, frameon=True,
                   edgecolor="0.45", facecolor="white", framealpha=0.92,
                   ncol=3, columnspacing=1.2, handletextpad=0.4,
                   borderpad=0.4, loc="upper right",
                   borderaxespad=0)
        panel_title(ax1, "a", "QCD and top jets")

        for s in range(4):
            m = sub_ == s
            ax2.scatter(pts[m, 0], pts[m, 1], pts[m, 2], s=1, alpha=0.4,
                        c=SUBCLASS_COLORS[s], rasterized=True)
        ax2.scatter(c3[:, 0], c3[:, 1], c3[:, 2], s=60, c="k", zorder=6)
        ax2.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="0.25",
                 linewidth=1.4, alpha=0.9, linestyle=(0, (4, 2)))
        # duplicate angle label for clarity (both panels show geodesic)
        mid2 = arc[150]
        ax2.text(mid2[0], mid2[1], mid2[2] + 0.17, f"${geo_deg:.1f}^\\circ$", fontsize=9,
                 ha="center", va="bottom", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9))

        sub_handles = [Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=SUBCLASS_COLORS[s], markersize=6,
                              label=f"{SUBCLASS_NAMES[s]} ({(sub_==s).sum()/len(sub_)*100:.0f}%)")
                       for s in range(4)]
        sub_handles.append(proto_handle)
        ax2.legend(handles=sub_handles, fontsize=7.5, frameon=True,
                   edgecolor="0.45", facecolor="white", framealpha=0.92,
                   ncol=3, columnspacing=1.0, handletextpad=0.4,
                   borderpad=0.4, loc="upper right",
                   borderaxespad=0)
        panel_title(ax2, "b", "Spectral subclasses")

        out = os.path.join(model_dir, "hypersphere_3d.png")
        fig.savefig(str(out), dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    print(f"Saved: {out}")
    print(f"Singular value ratios: {sv_ratio[:6].round(4)}")
    print(f"Top-2 variance fraction: {var_frac[1]*100:.1f}%")
    print(f"Geodesic angle: {geo_deg:.2f}°")
    print(f"Mean embedding norm: {np.linalg.norm(emb[idx], axis=1).mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hypersphere visualization")
    parser.add_argument("--model_dir", type=str, default="experiments/robustness_s16_m05",
                        help="Directory containing model checkpoint and embeddings")
    parser.add_argument("--embedding_dim", type=int, default=None,
                        help="Embedding dimension (unused, kept for consistency)")
    args = parser.parse_args()
    
    model_dir = args.model_dir
    embeddings_path = os.path.join(model_dir, "embeddings.npz")
    checkpoint_path = os.path.join(model_dir, "checkpoint.pt")
    subclass_labels_path = os.path.join(os.path.dirname(model_dir), "physics_sr", "subclass_labels.npz")
    
    main()