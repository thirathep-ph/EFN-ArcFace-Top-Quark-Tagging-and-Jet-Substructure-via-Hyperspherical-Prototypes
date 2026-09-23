"""HC deep levels 2..5: ward physics per split"""
import json, numpy as np
from pathlib import Path
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
import awkward as ak
from arcefn.utils.physics import compute_features
from arcefn.data.loader import load_awkward
from arcefn.utils.paths import DATA_DIR
from scipy.optimize import curve_fit

def ap(zg,A,a,b): return A*np.power(np.clip(zg,1e-10,1-1e-10),a)*np.power(np.clip(1-zg,1e-10,1-1e-10),b)
def fit_zg(v):
    v=v[(v>0.1)&(v<0.5)]
    if len(v)<10: return None
    h,edges=np.histogram(v,bins=30,density=True)
    c=(edges[:-1]+edges[1:])/2
    m=h>0; c=c[m]; h=h[m]
    if len(c)<3: return None
    try:
        popt,_=curve_fit(ap,c,h,p0=[1,-0.5,0.5],bounds=([0,-5,-5],[100,10,10]),maxfev=8000)
        pred=ap(c,*popt)
        r2=1-np.sum((h-pred)**2)/(np.sum((h-h.mean())**2)+1e-10)
        return {"alpha":float(popt[1]),"beta":float(popt[2]),"r2":float(r2),"n":int(len(v))}
    except Exception: return None

np.random.seed(42)
E=np.load('experiments/robustness_s16_m05/embeddings.npz')
En=E['embeddings']; y=E['labels']
En=En/np.linalg.norm(En,axis=1,keepdims=True)
# load test events for physics (need 16k corresponding indices)
print("load test H5...")
NEED=20000
events,labels,weights=load_awkward(str(DATA_DIR/'test.h5'), max_events=NEED, lazy=False)
N=min(len(En), len(events), NEED)
En=En[:N]; y=y[:N]
feat=compute_features(events[:N])
# feat cols: 0 mass,7 tau32, 8 zg
mass=feat[:,0]; tau32=feat[:,7]; zg=feat[:,8]

results={}
for cls in [0,1]:
    cls_name="QCD" if cls==0 else "Top"
    idx=np.where(y==cls)[0]
    # use same 8000 as before
    sel=np.random.choice(idx, 8000, replace=False)
    X=En[sel]
    # linkage distances for gap
    Z=linkage(X[:800],method='ward')
    # sort distances descending to estimate gap k
    # not needed
    for k in [2,3,4,5]:
        hc=AgglomerativeClustering(n_clusters=k, linkage='ward')
        lab=hc.fit_predict(X)
        cnts=[int((lab==c).sum()) for c in range(k)]
        # physics per cluster
        per=[]
        for c in range(k):
            msk=sel[lab==c]
            vmass=float(np.mean(mass[msk])); vtau=float(np.mean(tau32[msk]))
            f=fit_zg(zg[msk])
            per.append({"counts":int(len(msk)),"mass_mean":round(vmass,1),"tau32_mean":round(vtau,3),"zg_fit":f})
        results[f"{cls_name}_k{k}"]={"counts":cnts,"clusters":per}
        print(f"{cls_name} k={k} counts {cnts} ->",[(p['mass_mean'],p['tau32_mean'],p['zg_fit']['alpha'] if p['zg_fit'] else None) for p in per])

Path('experiments/hc_deep_results.json').write_text(json.dumps(results,indent=2))
print("saved experiments/hc_deep_results.json")
