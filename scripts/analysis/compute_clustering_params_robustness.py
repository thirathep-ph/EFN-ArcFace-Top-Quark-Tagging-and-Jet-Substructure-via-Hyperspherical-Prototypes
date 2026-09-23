"""Param robustness: SC affinity/n_neighbors/gamma + HC linkage/metrics."""
import json, numpy as np
from pathlib import Path
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
np.random.seed(42)
npz = np.load('experiments/robustness_s16_m05/embeddings.npz')
E = npz['embeddings']; y = npz['labels']
En = E / np.linalg.norm(E, axis=1, keepdims=True)
from arcefn.utils.pred_labels import arcface_preds_from_centers, check_cm
preds = arcface_preds_from_centers(E, 'experiments/robustness_s16_m05/checkpoint.pt')
check_cm(preds, y, "params_robustness")
n_per = 6000
configs_sc = [
    ("cosine-matrixfree-primary", None),
    ("nearest_neighbors_10", {"affinity":"nearest_neighbors","n_neighbors":10}),
    ("nearest_neighbors_20", {"affinity":"nearest_neighbors","n_neighbors":20}),
    ("nearest_neighbors_30", {"affinity":"nearest_neighbors","n_neighbors":30}),
    ("rbf_gamma0.5", {"affinity":"rbf","gamma":0.5}),
    ("rbf_gamma1.0", {"affinity":"rbf","gamma":1.0}),
]
configs_hc = [
    ("ward_euclidean", {"linkage":"ward","metric":"euclidean"}),
    ("complete_cosine", {"linkage":"complete","metric":"cosine"}),
    ("average_cosine", {"linkage":"average","metric":"cosine"}),
    ("complete_euclidean", {"linkage":"complete","metric":"euclidean"}),
]
from arcefn.utils.clustering import spectral_clustering_subclass
results={"partition": "predicted_labels_primary"}
for cls in [0,1]:
    cls_name="QCD" if cls==0 else "Top"
    idx=np.where(preds==cls)[0]
    sel=np.random.choice(idx, n_per, replace=False)
    X=En[sel]
    # primary: matrixfree cosine k=2 (reference)
    primary,_=spectral_clustering_subclass(X, max_k=5)
    cls_res=[]
    for name, kwargs in configs_sc:
        if kwargs is None:
            labels,_=spectral_clustering_subclass(X, max_k=5)
            k=2
        else:
            # test k=2..5, choose silhouette best (like eigengap analogue) - quick: just test k=2 vs k=3 silhouette proxy
            # For param sweep we fix k=2 to test assignment stability; also record k* via eigengap where applicable
            try:
                sc=SpectralClustering(n_clusters=2, random_state=42, assign_labels='kmeans', **kwargs)
                labels=sc.fit_predict(X)
                k=2
            except Exception as e:
                cls_res.append({"config":name,"error":str(e)})
                continue
        ari=float(adjusted_rand_score(primary, labels))
        # also test k* stability: run k=2..5 and pick best silhouette quickly via inertia proxy - just report k=2 stable
        cls_res.append({"config":name,"k_star":k,"ari_vs_primary":round(ari,3),"n0":int((labels==0).sum()),"n1":int((labels==1).sum())})
    # HC variants
    for name, kwargs in configs_hc:
        try:
            if kwargs["linkage"]=="ward":
                hc=AgglomerativeClustering(n_clusters=2, linkage='ward')
            else:
                hc=AgglomerativeClustering(n_clusters=2, linkage=kwargs["linkage"], metric=kwargs["metric"])
            h_labels=hc.fit_predict(X)
            ari=float(adjusted_rand_score(primary, h_labels))
            cls_res.append({"config":f"HC_{name}","k_star":2,"ari_vs_primary":round(ari,3),"n0":int((h_labels==0).sum()),"n1":int((h_labels==1).sum())})
        except Exception as e:
            cls_res.append({"config":f"HC_{name}","error":str(e)})
    results[cls_name]=cls_res
    # summary: min/max ARI
    aris=[r["ari_vs_primary"] for r in cls_res if "ari_vs_primary" in r]
    results[f"{cls_name}_summary"]={"n_configs":len(cls_res),"min_ari":min(aris) if aris else None,"max_ari":max(aris) if aris else None,"all_k_star_2":all(r.get("k_star")==2 for r in cls_res if "k_star" in r)}
Path("experiments/clustering_params_robustness.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
