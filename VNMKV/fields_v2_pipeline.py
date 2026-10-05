# v2 pipeline: Otsu land cover -> watershed parcels -> SAM (agribound refinement rules) -> agribound postprocess/evaluate

# ===== otsu2.py =====
import numpy as np, cv2
from skimage.filters import threshold_otsu, threshold_multiotsu
bgr=cv2.imread('/mnt/user-data/uploads/1791226972075_image.png'); img=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
lab=cv2.cvtColor(img,cv2.COLOR_RGB2LAB).astype(np.float32)
lab=cv2.bilateralFilter(lab,9,20,7)
L,a,b=lab[...,0],lab[...,1]-128,lab[...,2]-128
# trees: green-ish & dark -> index = -a + (low L); Otsu on a* alone first
ta=threshold_otsu(a); print('otsu a*',ta, 'pct a<ta',(a<ta).mean())
# greenness relative to soil: VARI-like on Lab: g = -a - 0.5*(L-L.mean())/L.std()*... keep simple
gi=-a*1.0 - 0.15*L
tg=threshold_otsu(gi); trees=gi>tg; print('otsu tree idx',tg,'tree %',trees.mean()*100)
# purple ploughed soil: low b* (blue-ish) ; dry fallow: high b*
tb=threshold_otsu(b[~trees]); print('otsu b*',tb)
tL=threshold_otsu(L[~trees]); print('otsu L',tL)
cls=np.full(L.shape,1)          # 1 fallow/dry
cls[(b<tb)&~trees]=0            # 0 ploughed dark soil
bright=(L>threshold_otsu(L[(~trees)&(b>=tb)]) )&(~trees)&(np.hypot(a,b)<np.percentile(np.hypot(a,b),25))
cls[bright]=3                   # built/roads (bright, low chroma)
cls[trees]=2
np.savez('otsu2.npz',cls=cls,L=L,a=a,b=b)
viz=np.zeros_like(img); viz[cls==0]=[80,40,60]; viz[cls==1]=[220,180,110]; viz[cls==2]=[20,150,40]; viz[cls==3]=[255,255,255]
print({k:round((cls==k).mean()*100,1) for k in range(4)})
cv2.imwrite('otsu2.png',cv2.cvtColor(np.hstack([img,viz]),cv2.COLOR_RGB2BGR))

# ===== parcels.py =====
# Otsu land cover -> boundary-aware watershed parcels (used as SAM prompts + gap filler)
import numpy as np, cv2
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
d=np.load('otsu2.npz'); cls=d['cls']; L=d['L']
soil=(cls==0)|(cls==1)
# bunds/tracks: bright thin lines (white top-hat) and dark thin lines (black-hat), Otsu inside soil
k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
wt=cv2.morphologyEx(L,cv2.MORPH_TOPHAT,k); bt=cv2.morphologyEx(L,cv2.MORPH_BLACKHAT,k)
lines=(wt>threshold_otsu(wt[soil]))|(bt>threshold_otsu(bt[soil]))
lines=cv2.morphologyEx(lines.astype(np.uint8),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))).astype(bool)
# class change between ploughed/fallow is also a boundary
cb=cv2.morphologyEx(cls.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3)))>0
barrier=lines|cb|~soil
inside=soil&~barrier
inside=cv2.morphologyEx(inside.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3))).astype(bool)
dt=cv2.distanceTransform(inside.astype(np.uint8),cv2.DIST_L2,5)
pk=peak_local_max(dt,min_distance=18,threshold_abs=6,labels=cv2.connectedComponents(inside.astype(np.uint8))[1])
mk=np.zeros(dt.shape,np.int32)
for i,(y,x) in enumerate(pk,1): mk[y,x]=i
ws=watershed(-dt,mk,mask=soil)
# drop tiny
ids,cnt=np.unique(ws,return_counts=True)
for i,c in zip(ids,cnt):
    if i and c<400: ws[ws==i]=0
seeds=[(int(x),int(y),int(ws[y,x])) for y,x in pk if ws[y,x]>0]
print('peaks',len(pk),'parcels',len(np.unique(ws))-1)
np.savez('parcels.npz',ws=ws,seeds=np.array(seeds),lines=lines)
img=cv2.imread('/mnt/user-data/uploads/1791226972075_image.png')
rng=np.random.default_rng(3); col=rng.integers(40,255,(ws.max()+1,3)); col[0]=0
ov=img.copy(); m=ws>0; ov[m]=(0.5*img[m]+0.5*col[ws[m]]).astype(np.uint8)
for x,y,_ in seeds: cv2.circle(ov,(x,y),3,(255,255,255),-1)
cv2.imwrite('parcels.png',ov)

# ===== refine_ab.py =====
"""Delineation (Otsu land cover + watershed) -> SAM box refinement following agribound 1.0.1
samgeo_engine rules (15% pad, >=64 px box, single mask, clip to padded box, largest part,
'trim' overlaps in score order, min coverage 0.5) -> agribound.postprocess merge/filter -> agribound.evaluate.
Mask binarisation: Otsu threshold on SAM logits inside each padded box (instead of fixed 0)."""
import json, numpy as np, cv2, onnxruntime as ort, geopandas as gpd, rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from skimage.filters import threshold_otsu
from agribound.postprocess.merge import merge_polygons
from agribound.postprocess.filter import filter_polygons
import agribound
up='/mnt/user-data/uploads/'
with rasterio.open('/mnt/user-data/outputs/image1_georef_EPSG3857.tif') as r: T1=r.transform
H,W=889,867
emb=np.load('emb.npy'); sc=float(np.load('meta.npy')[2])
dec=ort.InferenceSession('/home/claude/sam/segment_anything_vit_b_decoder.onnx')
ws=np.load('parcels.npz')['ws']; cls=np.load('otsu2.npz')['cls']
PAD,MINPX,MINCOV=0.15,64,0.5
def sam_box(x0,y0,x1,y1):
    pc=np.array([[[x0,y0],[x1,y1]]],np.float32)*sc
    m,iou,_=dec.run(None,{'image_embeddings':emb,'point_coords':pc,'point_labels':np.array([[2,3]],np.float32),
      'mask_input':np.zeros((1,1,256,256),np.float32),'has_mask_input':np.zeros(1,np.float32),'orig_im_size':np.array([H,W],np.float32)})
    return m[0,0],float(iou[0,0])     # index 0 = single-mask output (multimask_output=False)
recs=[]; otsu_t=[]
for pid in np.unique(ws)[1:]:
    pm=ws==pid; ys,xs=np.nonzero(pm)
    tree=(cls[pm]==2).mean(); built=(cls[pm]==3).mean()
    x0,x1,y0,y1=xs.min(),xs.max()+1,ys.min(),ys.max()+1
    bw,bh=x1-x0,y1-y0; px,py=PAD*bw,PAD*bh
    X0,Y0,X1,Y1=[int(v) for v in (max(0,np.floor(x0-px)),max(0,np.floor(y0-py)),min(W,np.ceil(x1+px)),min(H,np.ceil(y1+py)))]
    rec=dict(pid=int(pid),inp=pm,score=np.nan,refined=False,mask=None)
    if (X1-X0)>=MINPX and (Y1-Y0)>=MINPX:
        lg,s=sam_box(x0,y0,x1,y1); win=lg[Y0:Y1,X0:X1]
        t=float(np.clip(threshold_otsu(np.clip(win,-20,20)),-2,2)); otsu_t.append(t)
        mk=np.zeros((H,W),bool); mk[Y0:Y1,X0:X1]=win>t
        n,l,st,_=cv2.connectedComponentsWithStats(mk.astype(np.uint8))
        if n>1: mk=l==(1+np.argmax(st[1:,4]))   # largest part
        rec.update(score=s,mask=mk)
    rec['tree']=tree; rec['built']=built; recs.append(rec)
print('parcels',len(recs),'SAM-prompted',sum(r['mask'] is not None for r in recs),'Otsu logit thr median/min/max',np.round([np.median(otsu_t),min(otsu_t),max(otsu_t)],2))

# --- group over-segmented parcels: two parcels whose raw SAM masks each cover >=70% of the other parcel
# (or raw-mask IoU>0.6) are one field; re-prompt SAM with the group's box ---
P=[r for r in recs if r['mask'] is not None]
def sam_pt(x,y):
    pc=np.array([[[x,y],[0,0]]],np.float32)*[[[sc,sc],[0,0]]]
    m,iou,_=dec.run(None,{'image_embeddings':emb,'point_coords':pc.astype(np.float32),'point_labels':np.array([[1,-1]],np.float32),
      'mask_input':np.zeros((1,1,256,256),np.float32),'has_mask_input':np.zeros(1,np.float32),'orig_im_size':np.array([H,W],np.float32)})
    k=1+int(np.argmax(iou[0,1:])); lg=m[0,k]; return lg>0
for r in P:   # point prompt at the parcel's interior-most pixel (distance-transform max)
    dt=cv2.distanceTransform(r['inp'].astype(np.uint8),cv2.DIST_L2,5); y,x=np.unravel_index(dt.argmax(),dt.shape)
    pm_=sam_pt(x,y); r['ptmask']=pm_ if pm_.sum()<0.25*H*W else r['inp']
par=list(range(len(P)))
def f(i):
    while par[i]!=i: par[i]=par[par[i]]; i=par[i]
    return i
for i in range(len(P)):
    for j in range(i+1,len(P)):
        a,b=P[i],P[j]
        ca=(a['ptmask']&b['inp']).sum()/b['inp'].sum(); cb=(b['ptmask']&a['inp']).sum()/a['inp'].sum()
        iou=(a['ptmask']&b['ptmask']).sum()/max((a['ptmask']|b['ptmask']).sum(),1)
        if ca>=0.7 and cb>=0.7 and iou>0.6: par[f(i)]=f(j)
groups={}
for i in range(len(P)): groups.setdefault(f(i),[]).append(P[i])
newrecs=[r for r in recs if r['mask'] is None]
for g in groups.values():
    if len(g)==1: newrecs.append(g[0]); continue
    pm=np.any([r['inp'] for r in g],axis=0); ys,xs=np.nonzero(pm)
    x0,x1,y0,y1=xs.min(),xs.max()+1,ys.min(),ys.max()+1; bw,bh=x1-x0,y1-y0
    X0,Y0,X1,Y1=[int(v) for v in (max(0,np.floor(x0-PAD*bw)),max(0,np.floor(y0-PAD*bh)),min(W,np.ceil(x1+PAD*bw)),min(H,np.ceil(y1+PAD*bh)))]
    lg,s=sam_box(x0,y0,x1,y1); win=lg[Y0:Y1,X0:X1]; t=float(np.clip(threshold_otsu(np.clip(win,-20,20)),-2,2))
    mk=np.zeros((H,W),bool); mk[Y0:Y1,X0:X1]=win>t
    n,l,st,_=cv2.connectedComponentsWithStats(mk.astype(np.uint8))
    if n>1: mk=l==(1+np.argmax(st[1:,4]))
    newrecs.append(dict(pid=g[0]['pid'],inp=pm,score=s,refined=False,mask=mk,tree=(cls[pm]==2).mean(),built=(cls[pm]==3).mean(),merged=len(g)))
print('groups merged:',sum(len(g)>1 for g in groups.values()),'parcels absorbed:',sum(len(g) for g in groups.values() if len(g)>1))
recs=newrecs
# 'trim' overlaps in score order + coverage test
inputs=np.zeros((H,W),np.int32)
for r in recs: inputs[r['inp']]=r['pid']
claimed=np.zeros((H,W),bool); nlow=0
for r in sorted([r for r in recs if r['mask'] is not None],key=lambda r:-r['score']):
    mk=r['mask'] & ((inputs==0)|(inputs==r['pid'])) & ~claimed
    n,l,st,_=cv2.connectedComponentsWithStats(mk.astype(np.uint8))
    if n>1: mk=l==(1+np.argmax(st[1:,4]))
    cov=(mk&r['inp']).sum()/r['inp'].sum()
    if n>1 and cov>=MINCOV:
        r['geom_px']=mk; r['refined']=True; claimed|=mk&~r['inp']
    else: nlow+=1
print('refined',sum(r['refined'] for r in recs),'low-coverage/failed',nlow)
rows=[]
for r in recs:
    m=r.get('geom_px') if r['refined'] else r['inp']
    gs=[shape(g) for g,v in shapes(m.astype(np.uint8),mask=m,transform=T1)]
    if not gs: continue
    g=max(gs,key=lambda p:p.area).buffer(0)
    rows.append({'pid':r['pid'],'sam_refined':r['refined'],'sam_score':r['score'],'tree_frac':round(r['tree'],2),'built_frac':round(r['built'],2),'geometry':g})
gdf=gpd.GeoDataFrame(rows,crs=3857).to_crs(32643)
# local land-cover screen (stand-in for agribound's GEE LULC crop filter): drop tree/built-dominated polygons
gdf=gdf[(gdf.tree_frac<0.5)&(gdf.built_frac<0.3)]
gdf['geometry']=gdf.geometry.simplify(0.75).buffer(0)
gdf=merge_polygons(gdf,iou_threshold=0.3,containment_threshold=0.8)
gdf=filter_polygons(gdf,min_area_m2=500,remove_holes_below_m2=200)
gdf['area_ha']=(gdf.area/1e4).round(3)
print('final fields',len(gdf))
ref=gpd.read_file(up+'2_fields.geojson'); ref=ref[ref['__gm_id']!='feature-5'].to_crs(32643)
ev=agribound.evaluate(gdf,ref,iou_threshold=0.5)
print({k:v for k,v in ev.items() if not hasattr(v,'__len__') or isinstance(v,str)})
gdf.to_crs(4326).to_file('/mnt/user-data/outputs/fields_agribound_style_wgs84.geojson',driver='GeoJSON')
gdf.to_pickle('fields.pkl'); json.dump({k:(v if isinstance(v,(int,float,str,type(None))) else str(v)) for k,v in ev.items()},open('/mnt/user-data/outputs/evaluation_vs_drawn_fields.json','w'),indent=1)

# ===== viz2.py =====
import pandas as pd, numpy as np, cv2, rasterio, geopandas as gpd
g=pd.read_pickle('fields.pkl').to_crs(3857)
with rasterio.open('/mnt/user-data/outputs/image1_georef_EPSG3857.tif') as r: inv=~r.transform
img=cv2.imread('/mnt/user-data/uploads/1791226972075_image.png'); ov=img.copy(); fill=img.copy()
rng=np.random.default_rng(7)
for _,row in g.iterrows():
    gm=row.geometry; geoms=list(gm.geoms) if gm.geom_type=='MultiPolygon' else [gm]
    c=tuple(int(v) for v in rng.integers(60,255,3))
    for p in geoms:
        pts=np.array([inv*xy for xy in p.exterior.coords],np.int32); cv2.fillPoly(fill,[pts],c)
        cv2.polylines(ov,[pts],True,(255,255,255) if row.sam_refined else (0,255,255),2)
ov=cv2.addWeighted(fill,0.4,ov,0.6,0)
ref=gpd.read_file('/mnt/user-data/uploads/2_fields.geojson').to_crs(3857); ref=ref[ref['__gm_id']!='feature-5']
for gm in ref.geometry:
    for p in (gm.geoms if gm.geom_type=='MultiPolygon' else [gm]):
        cv2.polylines(ov,[np.array([inv*xy for xy in p.exterior.coords],np.int32)],True,(0,0,255),2)
cv2.imwrite('/mnt/user-data/outputs/fields_v2_preview.png',np.hstack([img,ov]))
