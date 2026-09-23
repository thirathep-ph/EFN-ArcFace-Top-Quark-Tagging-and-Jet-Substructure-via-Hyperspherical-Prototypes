import json, numpy as np, torch
from torch.utils.data import DataLoader
from scipy.optimize import curve_fit
from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.utils.device import get_device
from arcefn.utils.paths import DATA_DIR, EXPERIMENTS
from arcefn.utils.model_loading import load_model_from_dir
from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.physics import compute_features
from pathlib import Path
from collections import Counter
import time, warnings; warnings.filterwarnings('ignore')

def dglap(z,A,alpha,beta): return A*np.clip(z,1e-10,0.999)**alpha*np.clip(1-z,1e-10,0.999)**beta
def fit(zg):
    v=zg[(zg>0.1)&(zg<0.5)]
    if len(v)<50: return None
    hist,edges=np.histogram(v,bins=30,density=True)
    bc=(edges[:-1]+edges[1:])/2
    try:
        popt,_=curve_fit(dglap,bc,hist,p0=[1,-1,0],bounds=[[0,-5,-5],[100,10,10]],maxfev=10000)
        res=hist-dglap(bc,*popt)
        r2=1-np.sum(res**2)/(np.sum((hist-np.mean(hist))**2)+1e-12)
        return float(popt[1]),float(popt[2]),float(r2)
    except: return None

def assign(emb,labels):
    sub=np.full(len(labels),-1,int)
    for cls,off in [(0,0),(1,2)]:
        m=labels==cls
        sl,k=spectral_clustering_subclass(emb[m],max_k=5)
        if k>=2:
            cnt=Counter(sl); order=sorted(cnt,key=lambda c:-cnt[c]); mp={order[0]:0,order[1]:1}
            sl=np.array([mp.get(l,0) for l in sl])%2
        sub[m]=sl+off
    return sub

print("loading model and data 404k")
device,_=get_device()
m=load_model_from_dir(Path("experiments/robustness_s16_m05"),device); m.eval()
events,labels,_=load_awkward(str(DATA_DIR/'test.h5'), max_events=404000, lazy=False)
ds=JetTaggingDataset(events,labels,_)
loader=DataLoader(ds,batch_size=2048,shuffle=False)
embs=[]
with torch.no_grad():
    for x,y,w,mask,_ in loader:
        x,mask=x.to(device),mask.to(device)
        out=m(x,mask=mask)
        embs.append(out[-1].cpu().numpy())
emb=np.vstack(embs)
# normalize for clustering
import torch.nn.functional as F
emb_norm = embs if False else np.array([e/np.linalg.norm(e) for e in emb])
# actually embs already from model is 64-d not normed; normalize
emb_norm = emb/np.linalg.norm(emb,axis=1,keepdims=True)
# predicted labels via prototype
import torch as th
centers = m.arcface_head.class_centers.detach().cpu().numpy() if hasattr(m,'arcface_head') else m.class_centers.detach().cpu().numpy()
# compute cos
cos = emb_norm @ centers.T
pred = np.argmax(cos,axis=1)
sub = assign(emb_norm, pred)
# zg once
print("computing zg 404k")
zg=np.full(len(events),-1.0)
for i in range(0,len(events),5000):
    feats=compute_features(events[i:i+5000])
    zg[i:i+5000]=feats[:,8]
res={}
for sc in range(4):
    mask=sub==sc
    v=zg[mask]
    v=v[(v>0.1)&(v<0.5)]
    print(f"sc {sc} N {mask.sum()} fitted {len(v)}")
    # bootstrap 500
    alphas=[]
    n=len(v)
    rng=np.random.default_rng(42)
    for b in range(500):
        sample=rng.choice(v,n,replace=True)
        hist,edges=np.histogram(sample,bins=30,density=True)
        bc=(edges[:-1]+edges[1:])/2
        try:
            popt,_=curve_fit(dglap,bc,hist,p0=[1,-1,0],bounds=[[0,-5,-5],[100,10,10]],maxfev=5000)
            alphas.append(float(popt[1]))
        except: pass
    alphas=np.array(alphas)
    lo,hi=np.percentile(alphas,[2.5,97.5])
    mean=np.mean(alphas)
    print(f"sc {sc} alpha mean {mean:.3f} [{lo:.3f},{hi:.3f}]")
    res[str(sc)]={"alpha_mean":float(mean),"alpha_ci95":[float(lo),float(hi)],"n":int(mask.sum()),"n_fit":int(len(v))}
Path("experiments/bootstrap_pred_alpha.json").write_text(json.dumps(res,indent=2))
print("saved")
