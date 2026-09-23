"""HC deep full physics: all 10 FastJet features + Lund + zg per Ward k=2..5"""
import json, numpy as np
from pathlib import Path
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
from arcefn.utils.physics import compute_features, get_lund_coordinates
from arcefn.data.loader import load_awkward
from arcefn.utils.paths import DATA_DIR
from scipy.optimize import curve_fit
import awkward as ak

def ap(zg,A,a,b): return A*np.power(np.clip(zg,1e-10,1-1e-10),a)*np.power(np.clip(1-zg,1e-10,1-1e-10),b)
def fit_zg(v):
    v=v[(v>0.1)&(v<0.5)]
    if len(v)<20: return None
    h,edges=np.histogram(v,bins=30,density=True)
    c=(edges[:-1]+edges[1:])/2
    m=h>0; c=c[m]; h=h[m]
    if len(c)<5: return None
    try:
        popt,_=curve_fit(ap,c,h,p0=[1,-0.5,0.5],bounds=([0,-5,-5],[100,10,10]),maxfev=8000)
        pred=ap(c,*popt)
        r2=1-np.sum((h-pred)**2)/(np.sum((h-h.mean())**2)+1e-10)
        return {"alpha":round(float(popt[1]),3),"beta":round(float(popt[2]),3),"r2":round(float(r2),3),"n":int(len(v))}
    except Exception: return None

def bootstrap_mean_ci(v, n_boot=1000, seed=0):
    rng=np.random.default_rng(seed)
    n=len(v)
    if n<20: return None
    boots=[float(np.mean(rng.choice(v, size=n, replace=True))) for _ in range(n_boot)]
    return [round(float(np.percentile(boots,2.5)),2), round(float(np.percentile(boots,97.5)),2)]

def bootstrap_zg_ci(v, n_boot=500, seed=0):
    rng=np.random.default_rng(seed)
    v0=v[(v>0.1)&(v<0.5)]
    if len(v0)<30: return None
    alphas=[]
    for _ in range(n_boot):
        s=rng.choice(v0, size=len(v0), replace=True)
        f=fit_zg(s)
        if f: alphas.append(f["alpha"])
    if len(alphas)<100: return None
    return [round(float(np.percentile(alphas,2.5)),2), round(float(np.percentile(alphas,97.5)),2)]

np.random.seed(42)
E=np.load('experiments/robustness_s16_m05/embeddings.npz')
En=E['embeddings']; y=E['labels']
En=En/np.linalg.norm(En,axis=1,keepdims=True)
from arcefn.utils.pred_labels import arcface_preds_from_centers, check_cm
preds_all=arcface_preds_from_centers(E['embeddings'], 'experiments/robustness_s16_m05/checkpoint.pt')
check_cm(preds_all, y, "hc_deep_full")
NEED=50000
print(f"load {NEED} test...")
events,labels,_=load_awkward(str(DATA_DIR/'test.h5'), max_events=NEED, lazy=False)
N=min(len(En), len(events), NEED)
En=En[:N]; y=y[:N]
preds=preds_all[:N]
events=events[:N]
print("compute 10 features...")
feat=compute_features(events) # (N,10) [Mass,mSD,Mult,nSD,d12,d23,Tau21,Tau32,zg,theta_g]
names=["Mass","mSD","Mult","nSD","d12","d23","Tau21","Tau32","zg","theta_g"]
# Lund coords aggregated per jet: need per-jet lund? get_lund per dataset returns flattened; compute per-cluster later via events subset
from arcefn.utils.physics import get_lund_coordinates
# for quick Lund mean we use per-jet theta_g already in feat; skip heavy Lund per cluster for now

results={}
for cls in [0,1]:
    cls_name="QCD" if cls==0 else "Top"
    idx=np.where(preds==cls)[0]
    n_per=min(12000, len(idx))
    sel=np.random.choice(idx, n_per, replace=False)
    X=En[sel]
    feat_sel=feat[sel]
    print(f"\n=== {cls_name} n={n_per} ===")
    for k in [2,3,4,5,6]:
        hc=AgglomerativeClustering(n_clusters=k, linkage='ward')
        lab=hc.fit_predict(X)
        cnts=[int((lab==c).sum()) for c in range(k)]
        print(f"k={k} {cnts}")
        clusters=[]
        for c in range(k):
            msk=lab==c
            f=feat_sel[msk]
            means={names[i]:round(float(f[:,i].mean()),3) for i in range(10)}
            stds={"Mass_std":round(float(f[:,0].std()),1),"Tau32_std":round(float(f[:,7].std()),3)}
            zgfit=fit_zg(f[:,8])
            mass_ci=bootstrap_mean_ci(f[:,0], n_boot=1000, seed=42+c*100+k)
            tau32_ci=bootstrap_mean_ci(f[:,7], n_boot=1000, seed=142+c*100+k)
            zg_alpha_ci=bootstrap_zg_ci(f[:,8], n_boot=500, seed=242+c*100+k)
            clusters.append({"n":int(msk.sum()),"means":means,"stds":stds,"zg_fit":zgfit,"mass_ci95":mass_ci,"tau32_ci95":tau32_ci,"zg_alpha_ci95":zg_alpha_ci})
            print(f"  c{c}: n={msk.sum()} mass={means['Mass']}{mass_ci} tau32={means['Tau32']}{tau32_ci} zg_alpha={zgfit['alpha'] if zgfit else None}{zg_alpha_ci} R2={zgfit['r2'] if zgfit else None}")
        results[f"{cls_name}_k{k}"]={"counts":cnts,"clusters":clusters,"n_per_class":n_per}

Path('experiments/hc_deep_full.json').write_text(json.dumps(results,indent=2))
print("saved experiments/hc_deep_full.json")
