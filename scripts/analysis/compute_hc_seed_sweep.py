"""HC seed sweep 42/123/7, Ward k=2..6, 12k/class, bootstrap mass/tau"""
import json, numpy as np
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from arcefn.data.loader import load_awkward
from arcefn.utils.physics import compute_features
from arcefn.utils.paths import DATA_DIR
from scipy.optimize import curve_fit

def ap(zg,A,a,b): return A*np.power(np.clip(zg,1e-10,1-1e-10),a)*np.power(np.clip(1-zg,1e-10,1-1e-10),b)
def fit_zg(v):
    v=v[(v>0.1)&(v<0.5)]
    if len(v)<20: return None
    h,edges=np.histogram(v,bins=30,density=True)
    c=(edges[:-1]+edges[1:])/2
    m=h>0; c=c[m]; h=h[m]
    if len(c)<5: return None
    try:
        popt,_=curve_fit(ap,c,h,p0=[1,-0.5,0.5],bounds=([0,-5,-5],[100,10,10]),maxfev=6000)
        pred=ap(c,*popt)
        r2=1-np.sum((h-pred)**2)/(np.sum((h-h.mean())**2)+1e-10)
        return float(popt[1]), round(float(r2),3)
    except: return None

NEED=50000
E=np.load('experiments/robustness_s16_m05/embeddings.npz')
En=E['embeddings']; y=E['labels']
En=En/np.linalg.norm(En,axis=1,keepdims=True)
from arcefn.utils.pred_labels import arcface_preds_from_centers, check_cm
preds_all=arcface_preds_from_centers(E['embeddings'], 'experiments/robustness_s16_m05/checkpoint.pt')
check_cm(preds_all, y, "hc_seed_sweep")
events,_,_=load_awkward(str(DATA_DIR/'test.h5'), max_events=NEED, lazy=False)
N=min(len(En),len(events))
En=En[:N]; y=y[:N]; events=events[:N]
preds=preds_all[:N]
print("compute features once...")
feat=compute_features(events)

out={}
for seed in [42,123,7]:
    np.random.seed(seed)
    print(f"\n== seed {seed} ==")
    for cls in [0,1]:
        cls_name="QCD" if cls==0 else "Top"
        idx=np.where(preds==cls)[0]
        sel=np.random.choice(idx, 12000, replace=False)
        X=En[sel]; fs=feat[sel]
        for k in [2,3,4,5,6]:
            hc=AgglomerativeClustering(n_clusters=k, linkage='ward')
            lab=hc.fit_predict(X)
            cnts=[int((lab==c).sum()) for c in range(k)]
            # summarize by mass for QCD, tau32 for Top
            if cls==0:
                vals=[float(np.mean(fs[lab==c,0])) for c in range(k)]  # mass
                alphas=[]
                for c in range(k):
                    f=fit_zg(fs[lab==c,8])
                    alphas.append(f[0] if f else None)
                print(f" seed{seed} {cls_name} k{k} mass { [round(v,1) for v in sorted(vals)] } alpha {[round(a,2) if a else None for a in alphas]} cnts {cnts}")
            else:
                vals=[float(np.mean(fs[lab==c,7])) for c in range(k)]  # tau32
                print(f" seed{seed} {cls_name} k{k} tau32 {[round(v,3) for v in sorted(vals)]} cnts {cnts}")
            key=f"seed{seed}_{cls_name}_k{k}"
            out[key]={"counts":cnts,"mass_means": [round(float(np.mean(fs[lab==c,0])),1) for c in range(k)] if cls==0 else None,
                      "tau32_means":[round(float(np.mean(fs[lab==c,7])),3) for c in range(k)],
                      "zg_alpha":[fit_zg(fs[lab==c,8])[0] if fit_zg(fs[lab==c,8]) else None for c in range(k)]}

Path('experiments/hc_seed_sweep.json').write_text(json.dumps(out,indent=2))
print("saved hc_seed_sweep.json")
