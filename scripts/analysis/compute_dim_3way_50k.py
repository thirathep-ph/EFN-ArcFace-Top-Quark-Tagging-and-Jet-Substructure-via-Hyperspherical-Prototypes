import numpy as np, torch, json
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.utils.device import get_device
from arcefn.utils.paths import DATA_DIR
from arcefn.utils.model_loading import load_model_from_dir
from pathlib import Path
import torch.nn.functional as F

def twonn(E):
    nbrs=NearestNeighbors(n_neighbors=3, metric='cosine').fit(E)
    d,_=nbrs.kneighbors(E)
    r1=d[:,1]+1e-10; r2=d[:,2]+1e-10
    return float(len(E)/np.sum(np.log(r2/r1)))
def pr_and_pca(E):
    pca=PCA().fit(E)
    ev=pca.explained_variance_
    evr=pca.explained_variance_ratio_
    pr=float((ev.sum()**2)/(ev**2).sum())
    return pr, evr[:6].tolist(), float(evr[:2].sum())

def get_emb(model_dir, max_events=50000):
    device,_=get_device()
    m=load_model_from_dir(Path(model_dir), device)
    m.eval()
    events,labels,_=load_awkward(str(DATA_DIR/'test.h5'), max_events=max_events, lazy=False)
    ds=JetTaggingDataset(events,labels,_)
    loader=DataLoader(ds,batch_size=1024,shuffle=False)
    embs=[]
    with torch.no_grad():
        for x,y,w,mask,_ in loader:
            x,mask=x.to(device),mask.to(device)
            out=m(x,mask=mask)
            e=out[-1]
            e=F.normalize(e,p=2,dim=1)
            embs.append(e.cpu().numpy())
    return np.concatenate(embs)

res={}
for name,dir in [("ArcFace","experiments/robustness_s16_m05"),("CosLinear","experiments/baselines_coslinear_seed_42"),("Linear","experiments/baselines_linear_seed_42")]:
    E=get_emb(dir, max_events=50000)
    pr, evr6, cum2 = pr_and_pca(E)
    d=twonn(E)
    res[name]={"twonn":round(d,2),"pr":round(pr,2),"pca_top6":[round(v,4) for v in evr6],"pca_cum2":round(cum2,4),"n":len(E)}
    print(name, res[name])

Path("experiments/dim_3way_50k.json").write_text(json.dumps(res,indent=2))
print("saved")
