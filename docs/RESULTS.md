# EFN–ArcFace Top Tagging — Results

## Abstract

Top-quark vs QCD jet classification with a geometrically readable decision rule: an IRC-safe Energy Flow Network backbone (59,072 params) plus an ArcFace prototype head, so classifying = nearest prototype on the unit hypersphere S⁶³. Tested on 404k jets: **91.84% accuracy [91.76, 91.92], AUC 0.974, 124× rejection**. Unsupervised clustering on the manifold recovers 4 subclasses (Core/Edge × Top/QCD), 3 of 4 validated against QCD predictions (splitting functions, angular scaling, Lund topology). A feature-based random forest still leads (92.37%) — stated openly, not hidden.

<details>
<summary>Pipeline figure</summary>

![Architecture](figures/architecture_schematic.png)
EFN ([2→128→128→128] Φ, energy-weighted sum pool, [128→128→64] ρ) → L2-normalized 64-D embedding on S⁶³ → ArcFace prototypes (m=0.5, s=16, seed 42). Trained on 1.2M jets.

</details>

## 1. Methodology

Jets → centered (Δη, Δφ) + energy weights → EFN → 64-D embedding → nearest of 2 learned prototypes. Subclasses found by spectral clustering per predicted class (eigengap picks k=2).

<details>
<summary>Method figures</summary>

![Class QCD](figures/deta_dphi_class_qcd_rep.png)
Representative QCD energy-flow map (single deposit).

![Class Top](figures/deta_dphi_class_top_rep.png)
Representative Top energy-flow map (3 deposits: b + W→qq̄).

</details>

## 2. Classification performance

**Takeaway:** competitive with plain EFN baselines; RF on hand-crafted features leads — the contribution is the readable representation, not accuracy.

<details>
<summary>Table + figures</summary>

| Model | Accuracy (404k test) | ROC AUC | Rejection@50%TPR |
|:------|:--------------------:|:-------:|:----------------:|
| EFN + ArcFace (s=16, m=0.5) | **91.84% [91.76, 91.92]** | 0.974 | **124×** |
| EFN + Linear (3-seed mean) | 91.94% ±0.01 | 0.974 | 122× |
| EFN + CosLinear (3-seed mean) | 92.00% ±0.03 | 0.975 | 132× |
| RF (10 substructure features) | 92.37% | 0.977 | 150× |

Scale check (3 seeds): s=22 → 91.81% mean (91.82/91.80/91.82) vs s=16 91.84% — within noise; s=16 kept for stability. McNemar ArcFace-vs-CosLinear: χ²=32.3, p=1.3e-8 (same test set, so the gap is model, not data luck).

![ROC](figures/final_roc_curves.png)
ROC curves: EFN+ArcFace AUC 0.974; RF leads at 92.37% accuracy.

![Rejection](figures/final_rejection_rates.png)
Rejection at 50% signal efficiency: ArcFace 124×, RF 150×, symbolic c=6 distillate 33×.

![POC](figures/poc_combined.png)
Proof-of-concept: score calibration and injection-scan stability.

![Mass vs score](figures/mass_vs_score_scatter.png)
Model score vs jet mass — gradual transition, not a sharp mass cutoff; misclassified events sit in the mass-overlap region (physical ambiguity, not just model error).

</details>

## 3. Subclass discovery & physics validation

**Takeaway:** 4 subclasses found, 3 of 4 validated against QCD; Top-Edge excluded (non-identifiable fit). Predicted-vs-truth partitions agree (ARI 0.79).

<details>
<summary>Table + figures</summary>

| Subclass | Mass | τ32 | Reading |
|:---------|:----:|:---:|:--------|
| QCD-Core | ~62 GeV | — | ordinary single-prong |
| QCD-Edge | ~134 GeV | — | hard gluon mimicking top (holds 99% of false negatives) |
| Top-Core | ~181 GeV | ~0.63 | boundary mixture (absorbs 93% of false positives) |
| Top-Edge | ~174 GeV | ~0.46 | diffuse tail, fit non-identifiable — excluded |

DGLAP power-law fits A·z^α(1−z)^β: QCD-Core α=−1.242, QCD-Edge −1.022, Top-Core −0.493 (ordering matches soft-gluon hierarchy; quark jets share the same 1/zg tail modulo color factor). Angular: subclass means follow θg ≈ 2m/pT (narrow-pT readout, stated limitation). One mass-matched bin keeps the Core–Edge gap (suggestive, not fully clean).

![Eigengap](figures/eigengap_spectrum.png)
Eigenvalue spectra + eigengaps: largest gap at k=2 for both QCD and Top partitions.

![Dendrogram](figures/hc_dendrogram.png)
Ward hierarchical clustering dendrogram supporting the 2-way split.

![Composition](figures/subclass_composition_heatmap.png)
Subclass composition across prediction categories (TP/FP/FN/TN).

![Subclass physics](figures/subclass_physics_distributions.png)
Mass / τ32 / zg distributions per subclass: the four groups separate physically.

![DGLAP fits](figures/dglap_fit.png)
Power-law fits: QCD-Core α=−1.242, QCD-Edge −1.022, Top-Core −0.493, Top-Edge +1.105 (excluded).

![Angular](figures/angular_collimation.png)
Subclass means follow θg ≈ 2m/pT scaling (narrow pT window: readout of mass, stated as limitation).

![Lund TP](figures/lund_status_tp.png)
True-Top Lund plane: clear 3-prong structure.

![Lund TN](figures/lund_status_tn.png)
True-QCD Lund plane: single-prong radiation pattern.

![Lund FP](figures/lund_status_fp.png)
False positives: top-like QCD (high-mass tail).

![Lund FN](figures/lund_status_fn.png)
False negatives: QCD-like tops in the overlap region.

![Lund QCD-Core](figures/lund_subclass_qcd_core.png)
QCD-Core Lund plane.

![Lund QCD-Edge TN](figures/lund_subclass_qcd_edge_tn.png)
QCD-Edge (true-negative slice) Lund plane.

![Lund QCD-Edge FP](figures/lund_subclass_qcd_edge_fp.png)
QCD-Edge (false-positive slice) Lund plane.

![Lund Top-Core TP](figures/lund_subclass_top_core_tp.png)
Top-Core (true-positive slice) Lund plane.

![Lund Top-Core FN](figures/lund_subclass_top_core_fn.png)
Top-Core (false-negative slice) Lund plane.

![Lund Top-Edge](figures/lund_subclass_top_edge.png)
Top-Edge Lund plane (diffuse tail, excluded from physics reading).

![QCD-Core avg](figures/deta_dphi_subclass_qcd_core_avg.png)
QCD-Core average energy-flow map.

![QCD-Core rep](figures/deta_dphi_subclass_qcd_core_rep.png)
QCD-Core representative map.

![QCD-Edge avg](figures/deta_dphi_subclass_qcd_edge_avg.png)
QCD-Edge average map.

![QCD-Edge rep](figures/deta_dphi_subclass_qcd_edge_rep.png)
QCD-Edge representative map.

![Top-Core avg](figures/deta_dphi_subclass_top_core_avg.png)
Top-Core average map.

![Top-Core rep](figures/deta_dphi_subclass_top_core_rep.png)
Top-Core representative map.

![Top-Edge avg](figures/deta_dphi_subclass_top_edge_avg.png)
Top-Edge average map.

![Top-Edge rep](figures/deta_dphi_subclass_top_edge_rep.png)
Top-Edge representative map.

</details>

## 4. Geometry & supporting evidence

**Takeaway:** embeddings collapse to 2-D (TwoNN 2.21, PCA 83.6+16.4); prototypes 43.6° apart; hypersphere alone (CosLinear) keeps balanced subclasses, margin m=0.5 resolves Top Core vs Edge. Null controls (noise, untrained) also give k*=2 — so k*=2 alone proves nothing; the physics case rests on §3.

<details>
<summary>Figures</summary>

![Hypersphere](figures/hypersphere_3d.png)
Embeddings collapse onto a 2-D disk spanned by the two prototypes.

![PCA](figures/pca_scatter_3way.png)
PCA scatter: PC1 83.6% + PC2 16.4% ≈ 100% (2-D collapse).

![Embedding geometry](figures/embedding_geometry_3way.png)
Embedding geometry across ArcFace / CosLinear / Linear heads.

![Head geometry](figures/head_geometry_3way.png)
Head comparison: hypersphere alone (CosLinear) keeps balanced subclasses; margin m=0.5 specifically resolves Top Core vs Edge.

![Dimensionality](figures/dimensionality_3way.png)
TwoNN + participation ratio per head: ArcFace TwoNN 2.21 vs 4.98/5.19 controls.

</details>

## 5. Symbolic distillation

**Takeaway:** PySR recovers tanh(m/pT) − 0.348·τ32 — the model rediscovers mass+3-prongness from raw inputs. Leading-physics distillate only (rejection 33× vs 124×), not a replacement.

<details>
<summary>Figure</summary>

![Pareto](figures/pareto_front_sr.png)
Pareto front: compact c=6 form (leading-physics distillate, not a replacement — rejection 33× vs 124× full model).

</details>

## 6. Discussion

- **What the prototypes learned:** high-mass/low-τ32 (Top) vs low-mass/high-τ32 (QCD) — checked, not assumed.
- **Why the hypersphere matters:** 3-head comparison isolates it — CosLinear keeps balanced subclasses without any margin; Linear collapses.
- **Limitations (stated openly):** simulation-only (Pythia 8), binary task, no detector effects, Top-Edge excluded, single-seed label-shuffle residual, QCD mass-confounding stated, narrow pT window, OOD scan is hypothesis-generating only.
- **Future work:** cross-generator check, multi-seed shuffle, full mass-matched table, multi-class.

## 7. Conclusion

A 59k-parameter tagger whose binary rule is nearest-prototype on S⁶³ reaches 91.84% with a 2-D, QCD-checked latent geometry: 4 subclasses found, 3 validated, controls run, limits stated. The RF still wins on accuracy — and says so in the abstract.

## Appendix A — dataset EDA

<details>
<summary>Figures</summary>

![EDA overview](figures/eda_figA1_dataset_overview.png)
Dataset overview: class balance, kinematics ranges.

![EDA kinematics](figures/eda_figA2_kinematics.png)
Kinematic distributions (pT, mass, multiplicity).

![EDA substructure](figures/eda_figA3_substructure.png)
Substructure observables (τ21, τ32, zg, θg, nSD).

![EDA correlation](figures/eda_figA4_correlation.png)
Observable correlation matrix (mass confounding visible here first).

</details>

## Appendix B — out-of-distribution (JetNet30, hypothesis-generating)

<details>
<summary>Figure</summary>

![OOD](figures/poc_ood_similarity.png)
Cosine similarity to prototypes per JetNet class: W/Z near QCD prototype; JetNet top maps to neither (30 vs ≤200 constituents shift — expected, honest low confidence).

</details>

## Reproduce

```bash
pip install -e . && python scripts/download_data.py   # 3.4 GB dataset
python scripts/train_arcface.py --scale 16 --seed 42  # 50 epochs
```
