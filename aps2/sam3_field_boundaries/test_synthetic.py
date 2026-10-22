# Synthetic end-to-end test: stub replaces SAM 3 (no GPU, no weights). Run: python test_synthetic.py
import sys, json, os, shutil, types
import numpy as np, rasterio
from rasterio.transform import from_origin
from rasterio.features import shapes
from rasterio.warp import transform_geom
from scipy import ndimage as ndi
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sam3_field_boundaries as P
# stub compares raw colours, so use an identity stretch for this test (real run keeps the 2-98 % stretch)
P.stretch_params = lambda src: (np.zeros((3,1,1),np.float32), np.full((3,1,1),255.0,np.float32))

S = os.path.join(os.environ.get("TMPDIR", "/tmp"), "sam3_synth_test"); shutil.rmtree(S, ignore_errors=True); os.makedirs(S)
rng = np.random.default_rng(0)
H = W = 3000
# field grid with irregular cuts (40-70 m fields, 0.5 m px)
def cuts(n):
    c = [0]
    while c[-1] < n:
        c.append(min(n, c[-1] + int(rng.integers(90, 160))))
    return c
ycut, xcut = cuts(H), cuts(W)
lab = np.zeros((H, W), np.int32); bund = np.zeros((H, W), bool)
fid = 0
for i in range(len(ycut)-1):
    for j in range(len(xcut)-1):
        fid += 1
        lab[ycut[i]:ycut[i+1], xcut[j]:xcut[j+1]] = fid
for yc in ycut[1:-1]:
    bund[yc-1:yc+1, :] = True
for xc in xcut[1:-1]:
    bund[:, xc-1:xc+1] = True
rgb = np.zeros((H, W, 3), np.float32)
fieldcol = np.array([120, 150, 80], np.float32)
br = rng.normal(0, 6, size=fid+1).astype(np.float32)
inside = lab > 0
rgb[inside] = fieldcol + br[lab[inside]][:, None]
rgb[bund] = [200, 200, 190]
tree = np.zeros((H, W), bool)
for _ in range(400):
    y, x = rng.integers(10, H-10, 2)
    yy, xx = np.ogrid[:H, :W]
    d = (yy-y)**2 + (xx-x)**2 <= 16
    tree |= d & inside & ~bund
rgb[tree] = [30, 80, 30]
road = np.zeros((H, W), bool); road[1490:1510, :] = True
rgb[road] = [90, 90, 90]
rgb = np.clip(rgb, 0, 255).astype(np.uint8)
tf = from_origin(400000, 2301500, 0.5, 0.5)
ortho = os.path.join(S, "ortho.tif")
with rasterio.open(ortho, "w", driver="GTiff", height=H, width=W, count=3, dtype="uint8", crs="EPSG:32643", transform=tf) as d:
    d.write(np.transpose(rgb, (2, 0, 1)))
# ground-truth field polygons (WGS84)
gt = []
for g, v in shapes(lab, mask=lab > 0, transform=tf):
    gt.append({"type": "Feature", "properties": {}, "geometry": transform_geom("EPSG:32643", "EPSG:4326", g)})
json.dump({"type": "FeatureCollection", "features": gt}, open(os.path.join(S, "gt.geojson"), "w"))
print("GT fields:", len(gt))

class Stub:
    """SAM-3-like: returns instance masks (N,1,h,w) + scores; colour-based 'concepts' with 1% boundary noise."""
    COL = {"agricultural field": ([120,150,80], 60), "field bund": ([200,200,190], 40),
           "tree": ([30,80,30], 40), "road": ([90,90,90], 30)}
    def __init__(self): self.r = np.random.default_rng(1)
    def set_image(self, img): return {"img": np.asarray(img).astype(np.float32)}
    def set_text_prompt(self, prompt, state):
        col, tol = self.COL[prompt]
        m = np.abs(state["img"] - np.array(col)).sum(-1) < tol
        flip = self.r.random(m.shape) < 0.01
        m = m ^ flip
        l, n = ndi.label(m)
        masks, scores = [], []
        for k in range(1, n+1):
            mk = l == k
            if mk.sum() < 20: continue
            masks.append(mk); scores.append(float(self.r.uniform(0.6, 0.95)))
        if not masks:
            return {"masks": np.zeros((0,1,m.shape[0],m.shape[1]),bool), "scores": np.zeros(0)}
        return {"masks": np.stack(masks)[:,None], "scores": np.array(scores)}

def args_infer(out, win, ov):
    return types.SimpleNamespace(ortho=ortho, out=out, win=win, overlap=ov, stride=win-ov, device="cpu",
        confidence_floor=0.0, p_field="agricultural field", p_bund="field bund", p_tree="tree", p_road="road")
def args_post(out, block, halo):
    return types.SimpleNamespace(out=out, block=block, halo=halo, t_field=0.5, t_bund=0.4, t_tree=0.5, t_road=0.5,
        min_area_m2=150.0, min_bund_m2=2.0, min_tree_m2=4.0, min_road_m2=20.0, simplify_m=0.6)

out1 = os.path.join(S, "run_win1000"); P.infer(args_infer(out1, 1000, 200), proc=Stub())
P.postprocess(args_post(out1, 4000, 200))
out2 = os.path.join(S, "run_small_blocks")
shutil.copytree(out1, out2, ignore=shutil.ignore_patterns("*.geojson","run_summary.json"))
P.postprocess(args_post(out2, 700, 150))
for name, o in [("single block", out1), ("700px blocks", out2)]:
    r = P.evaluate(types.SimpleNamespace(pred=os.path.join(o,"fields.geojson"), ref=os.path.join(S,"gt.geojson"), crs="EPSG:32643", iou=0.5))
    print(name, json.dumps(r))
# seam check: probability map should have no tile-border artefacts (compare to truth field mask)
with rasterio.open(os.path.join(out1, "prob_field.tif")) as src:
    pf = src.read(1)
truth = (lab > 0) & ~bund & ~tree & ~road
print("prob_field mean in-field", round(float(pf[truth].mean()),3), "bund+road", round(float(pf[bund|road].mean()),3))
