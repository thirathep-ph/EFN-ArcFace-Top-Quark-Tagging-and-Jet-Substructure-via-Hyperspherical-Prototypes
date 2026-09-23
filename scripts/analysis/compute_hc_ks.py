"""KS / Cohen d per Ward k — model-independent physics distinctness"""
import json, numpy as np
from pathlib import Path
from scipy.stats import ks_2samp
from arcefn.data.loader import load_awkward
from arcefn.utils.physics import compute_features
from arcefn.utils.paths import DATA_DIR
from sklearn.cluster import AgglomerativeClustering

def cohen_d(a,b): 
    m1,m2=np.mean(a),np.mean(b)
    s=np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2+1e-12)
    return float((m1-m2)/s) if s>1e-12 else 0.0

np.random.seed(42)
E=np.load('experiments/robustness_s16_m05/embeddings.npz')
En=E['embeddings']; y=E['labels']
En=En/np.linalg.norm(En,axis=1,keepdims=True)
from arcefn.utils.pred_labels import arcface_preds_from_centers, check_cm
preds_all=arcface_preds_from_centers(E['embeddings'], 'experiments/robustness_s16_m05/checkpoint.pt')
check_cm(preds_all, y, "hc_ks")
NEED=50000
events,_,_=load_awkward(str(DATA_DIR/'test.h5'), max_events=NEED, lazy=False)
N=min(len(En), len(events))
En=En[:N]; y=y[:N]; events=events[:N]
preds=preds_all[:N]
feat=compute_features(events)
names=["Mass","mSD","Mult","nSD","d12","d23","Tau21","Tau32","zg","theta_g"]

res={}
for cls in [0,1]:
    cls_name="QCD" if cls==0 else "Top"
    idx=np.where(preds==cls)[0]
    sel=np.random.choice(idx, 12000, replace=False)
    X=En[sel]; fs=feat[sel]
    for k in [3,4,5,6]:
        hc=AgglomerativeClustering(n_clusters=k, linkage='ward')
        lab=hc.fit_predict(X)
        # pairwise KS per feature: report min p across features and best separating feature
        table=[]
        for i in range(k):
            for j in range(i+1,k):
                vi=fs[lab==i]; vj=fs[lab==j]
                # for each feature compute KS p and |d|
                best=None
                for f in range(10):
                    ks,p=ks_2samp(vi[:,f], vj[:,f], alternative='two-sided')
                    d=abs(cohen_d(vi[:,f], vj[:,f]))
                    if best is None or p<best['p']:
                        best={"feat":names[f],"ks":round(float(ks),3),"p":float(p),"d":round(d,2)}
                table.append({"pair":f"{i}-{j}","n":f"{int((lab==i).sum())} vs {int((lab==j).sum())}","best_feat":best['feat'],"ks":best['ks'],"d":best['d'],"p":best['p']})
        res[f"{cls_name}_k{k}"]=table
        print(f"{cls_name} k={k}")
        for t in table: print(f"  {t['pair']} {t['n']} -> {t['best_feat']} ks={t['ks']} d={t['d']} p={t['p']:.1e}")

Path('experiments/hc_ks_results.json').write_text(json.dumps(res,indent=2))
print("saved hc_ks_results.json")
