#!/usr/bin/env python3
"""
SAM 3 field-boundary pipeline for 0.5 m drone orthomosaics (Indian smallholdings).

INPUT   one RGB orthomosaic (GeoTIFF / COG, projected CRS, 0.5 m).
        Only this raster is read. Drone capture footprints, capture points and
        CropView photo zips are NOT read, so flight artefacts cannot shape the result.

OUTPUT  <out>/prob_{field,bund,tree,road}.tif   soft probabilities 0-1, float32 (not in git)
        <out>/fields.geojson                    cleaned field polygons, WGS84
        <out>/bunds.geojson, trees.geojson, roads.geojson   class polygons, WGS84
        <out>/run_summary.json

STAGES
  infer        sliding windows (WIN px, STRIDE px). Each window is read from disk, so
               the scene never sits in RAM. SAM 3 is prompted per class. All instances
               are kept and each class gets a soft map  P(p) = max_i score_i * mask_i(p).
               Window maps are blended with an edge-ramp weight, so tile seams vanish.
               Progress is checkpointed, so a crashed run resumes.
  postprocess  Block-wise (BLOCK px + HALO px halo). Field = P_field >= t, minus trees
               and roads. Bunds split touching fields (watershed on soft field prob, so
               parcel edges follow the soft valleys). Each polygon is kept only in the
               block that contains its centroid, so there are no duplicates at block seams.
               Then: min-area filter, edge simplification (topology preserved), WGS84.
  eval         One-to-one IoU matching against reference polygons (e.g. survey plots).

RUN (GPU host, e.g. RuDRA server)
  pip install sam3 rasterio shapely scipy scikit-image pillow
  huggingface-cli login            # SAM 3 weights are gated on Hugging Face
  python sam3_field_boundaries.py infer --ortho /data/orthomosaic_0p5m.tif --out ./sam3_fields
  python sam3_field_boundaries.py postprocess --out ./sam3_fields
  python sam3_field_boundaries.py eval --pred ./sam3_fields/fields.geojson \
      --ref ../withSurveyNo_fieldsOnly_wgs84.geojson
"""
import argparse
import json
import os
import time

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_geom
from rasterio.windows import Window, transform as window_transform
from scipy import ndimage as ndi
from shapely.geometry import shape, mapping
from shapely.strtree import STRtree
from skimage.segmentation import watershed
from skimage.transform import resize as sk_resize

CLASSES = ("field", "bund", "tree", "road")

# Short noun phrases work best for SAM 3. Check "bund" wording on a sample tile first.
DEFAULT_PROMPTS = {
    "field": "agricultural field",
    "bund": "field bund",
    "tree": "tree",
    "road": "road",
}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def tile_starts(n, win, stride):
    """Start offsets that cover [0, n) with windows of size win, last one flush with the edge."""
    if n <= win:
        return [0]
    starts = list(range(0, n - win + 1, stride))
    if starts[-1] != n - win:
        starts.append(n - win)
    return starts


def edge_weight(h, w, margin):
    """Blend weight: 1 in the middle, ramping to ~0 at the window edge over `margin` px."""
    dy = np.minimum(np.arange(h), np.arange(h)[::-1]) + 1
    dx = np.minimum(np.arange(w), np.arange(w)[::-1]) + 1
    wy = np.minimum(dy / float(margin), 1.0)
    wx = np.minimum(dx / float(margin), 1.0)
    return (wy[:, None] * wx[None, :]).astype(np.float32)


def _to_np(x):
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def soft_from_output(out, h, w):
    """Turn a Sam3Processor output dict into one soft map (h, w) in [0, 1]."""
    masks = _to_np(out["masks"])
    scores = _to_np(out["scores"]).reshape(-1)
    if masks.size == 0 or scores.size == 0:
        return np.zeros((h, w), np.float32)
    masks = masks.reshape(masks.shape[0], -1, masks.shape[-2], masks.shape[-1])[:, 0]
    if masks.min() < 0:  # logits -> probabilities
        masks = 1.0 / (1.0 + np.exp(-masks))
    if masks.shape[-2:] != (h, w):
        masks = np.stack([sk_resize(m, (h, w), order=1, preserve_range=True) for m in masks])
    n = min(len(masks), len(scores))
    soft = (np.clip(masks[:n], 0, 1) * scores[:n, None, None]).max(axis=0)
    return soft.astype(np.float32)


def stretch_params(src):
    """Per-band 2-98 % stretch from a decimated read, so every tile uses the same scale."""
    small = src.read([1, 2, 3], out_shape=(3, max(src.height // 16, 1), max(src.width // 16, 1))).astype(np.float32)
    lo = np.percentile(small.reshape(3, -1), 2, axis=1).reshape(3, 1, 1)
    hi = np.percentile(small.reshape(3, -1), 98, axis=1).reshape(3, 1, 1)
    return lo, np.maximum(hi, lo + 1)


def open_acc(path, H, W, reset):
    mode = "w+" if (reset or not os.path.exists(path)) else "r+"
    return np.memmap(path, dtype=np.float32, mode=mode, shape=(H, W))


def load_model(device, confidence_floor):
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model()
    # Keep every instance (score floor ~0). Confidence is handled by the soft map, not a hard cut.
    return Sam3Processor(model, device=device, confidence_threshold=confidence_floor)


# ----------------------------------------------------------------------------
# stage 1: inference
# ----------------------------------------------------------------------------
def infer(a, proc=None):
    from PIL import Image

    prompts = {"field": a.p_field, "bund": a.p_bund, "tree": a.p_tree, "road": a.p_road}
    os.makedirs(a.out, exist_ok=True)
    progress_path = os.path.join(a.out, "progress.json")
    done = set(json.load(open(progress_path))) if os.path.exists(progress_path) else set()

    with rasterio.open(a.ortho) as src:
        H, W = src.height, src.width
        if src.count < 3:
            raise SystemExit("orthomosaic needs at least 3 bands (RGB)")
        lo, hi = stretch_params(src)
        acc = {c: open_acc(os.path.join(a.out, f"acc_{c}.dat"), H, W, reset=not done) for c in CLASSES}
        wsum = open_acc(os.path.join(a.out, "acc_wsum.dat"), H, W, reset=not done)

        if proc is None:
            proc = load_model(a.device, a.confidence_floor)

        ys, xs = tile_starts(H, a.win, a.stride), tile_starts(W, a.win, a.stride)
        tiles = [(y, x) for y in ys for x in xs]
        print(f"scene {W}x{H} px, {len(tiles)} windows of {a.win} px (stride {a.stride}), {len(done)} already done")
        margin = max(a.overlap // 2, 1)
        t0 = time.time()

        for k, (y, x) in enumerate(tiles):
            if str(k) in done:
                continue
            h, w = min(a.win, H - y), min(a.win, W - x)
            rgb = src.read([1, 2, 3], window=Window(x, y, w, h)).astype(np.float32)
            valid = (rgb.max(axis=0) > 0)
            if valid.mean() >= 0.05:
                rgb = np.clip((rgb - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
                state = proc.set_image(Image.fromarray(np.transpose(rgb, (1, 2, 0))))
                wgt = edge_weight(h, w, margin) * valid.astype(np.float32)
                for c in CLASSES:
                    soft = soft_from_output(proc.set_text_prompt(prompt=prompts[c], state=state), h, w)
                    acc[c][y:y + h, x:x + w] += soft * wgt
                wsum[y:y + h, x:x + w] += wgt
            done.add(str(k))

            if len(done) % 20 == 0 or k == len(tiles) - 1:
                for m in list(acc.values()) + [wsum]:
                    m.flush()
                json.dump(sorted(done, key=int), open(progress_path, "w"))
                rate = (time.time() - t0) / max(len(done), 1)
                print(f"  {len(done)}/{len(tiles)} windows, ~{rate:.1f}s each")

        for m in list(acc.values()) + [wsum]:
            m.flush()
        json.dump(sorted(done, key=int), open(progress_path, "w"))

        # blend to final probabilities, written block-by-block as tiled GeoTIFFs
        profile = dict(driver="GTiff", height=H, width=W, count=1, dtype="float32", crs=src.crs,
                       transform=src.transform, tiled=True, blockxsize=512, blockysize=512,
                       compress="deflate", nodata=None)
        for c in CLASSES:
            with rasterio.open(os.path.join(a.out, f"prob_{c}.tif"), "w", **profile) as dst:
                for r0 in range(0, H, 2048):
                    r1 = min(H, r0 + 2048)
                    blk = acc[c][r0:r1] / np.maximum(wsum[r0:r1], 1e-6)
                    dst.write(blk.astype(np.float32), 1, window=Window(0, r0, W, r1 - r0))
    print("inference done ->", a.out)


# ----------------------------------------------------------------------------
# stage 2: post-processing
# ----------------------------------------------------------------------------
def components(mask, min_px, core):
    """Connected components of mask, keeping those whose centroid is inside core and >= min_px."""
    lab, n = ndi.label(mask)
    return lab, n, keep_by_centroid(lab, n, min_px, core)


def keep_by_centroid(lab, n, min_px, core):
    ch0, cw0, ch, cw = core
    rr, cc = np.nonzero(lab)
    lbl = lab[rr, cc]
    cnt = np.bincount(lbl, minlength=n + 1).astype(np.float64)
    cr = np.bincount(lbl, rr, minlength=n + 1) / np.maximum(cnt, 1)
    ccn = np.bincount(lbl, cc, minlength=n + 1) / np.maximum(cnt, 1)
    incore = (cr >= ch0) & (cr < ch0 + ch) & (ccn >= cw0) & (ccn < cw0 + cw)
    keep = (cnt >= min_px) & incore
    keep[0] = False
    return keep


def polygons(lab, keep, tf, prob_map, simplify_m, min_area_m2):
    """Vectorise kept labels to simplified polygons in CRS units (metres)."""
    lab_k = np.where(keep[lab], lab, 0).astype(np.int32)
    means = None
    if prob_map is not None and lab_k.max() > 0:  # per-label mean prob in one pass
        n = int(lab_k.max())
        cnt = np.bincount(lab_k.ravel(), minlength=n + 1).astype(np.float64)
        sums = np.bincount(lab_k.ravel(), prob_map.ravel(), minlength=n + 1)
        means = sums / np.maximum(cnt, 1)
    out = []
    for geom, val in shapes(lab_k, mask=lab_k > 0, transform=tf):
        poly = shape(geom).buffer(0)
        if poly.is_empty:
            continue
        if simplify_m > 0:
            poly = poly.simplify(simplify_m, preserve_topology=True)
        if poly.is_empty or poly.area < min_area_m2:
            continue
        mean_p = float(means[int(val)]) if means is not None else None
        out.append((poly, mean_p))
    return out


def block_features(P, a, core, tf, px_area):
    """All class polygons for one block. P: dict class -> (h, w) soft map. Returns dict class -> list."""
    res = {}
    min_px_field = a.min_area_m2 / px_area
    veg = (P["tree"] >= a.t_tree) | (P["road"] >= a.t_road)
    field = (P["field"] >= a.t_field) & ~veg
    bund = P["bund"] >= a.t_bund

    # field cores: remove bunds so touching fields separate, then grow labels back
    core_mask = ndi.binary_opening(field & ~bund, structure=np.ones((3, 3)))
    seeds, n = ndi.label(core_mask)
    lab = watershed(-P["field"], markers=seeds, mask=field) if n else np.zeros(field.shape, np.int32)
    keep = keep_by_centroid(lab, int(lab.max()), min_px_field, core) if lab.max() else np.zeros(1, bool)
    res["fields"] = polygons(lab, keep, tf, P["field"], a.simplify_m, a.min_area_m2)

    specs = {
        "bunds": (P["bund"] >= a.t_bund, a.min_bund_m2),
        "trees": (P["tree"] >= a.t_tree, a.min_tree_m2),
        "roads": (P["road"] >= a.t_road, a.min_road_m2),
    }
    for name, (mask, min_m2) in specs.items():
        lab_c, n_c, keep_c = components(mask, max(min_m2 / px_area, 1), core)
        res[name] = polygons(lab_c, keep_c, tf, None, a.simplify_m, min_m2)
    return res


def to_feature(poly, props):
    return {"type": "Feature", "properties": props, "geometry": mapping(poly)}


def postprocess(a):
    probs = {c: rasterio.open(os.path.join(a.out, f"prob_{c}.tif")) for c in CLASSES}
    ref = probs["field"]
    H, W, tf, crs = ref.height, ref.width, ref.transform, ref.crs
    px_area = abs(tf.a * tf.e)
    B, Hh = a.block, a.halo
    collected = {k: [] for k in ("fields", "bunds", "trees", "roads")}

    n_blocks = 0
    for cy in range(0, H, B):
        for cx in range(0, W, B):
            ch, cw = min(B, H - cy), min(B, W - cx)
            y0, x0 = max(0, cy - Hh), max(0, cx - Hh)
            y1, x1 = min(H, cy + ch + Hh), min(W, cx + cw + Hh)
            win = Window(x0, y0, x1 - x0, y1 - y0)
            P = {c: probs[c].read(1, window=win).astype(np.float32) for c in CLASSES}
            feats = block_features(P, a, (cy - y0, cx - x0, ch, cw), window_transform(win, tf), px_area)
            for k in collected:
                collected[k].extend(feats[k])
            n_blocks += 1
            print(f"  block ({cy},{cx}) -> fields so far {len(collected['fields'])}")

    summary = {"blocks": n_blocks, "crs": str(crs), "pixel_size_m": [abs(tf.a), abs(tf.e)]}
    for k, items in collected.items():
        feats = []
        for poly, mean_p in items:
            geom_wgs = transform_geom(crs.to_string(), "EPSG:4326", mapping(poly))
            props = {"area_m2": round(poly.area, 2), "perimeter_m": round(poly.length, 2)}
            if mean_p is not None:
                props["mean_prob"] = round(mean_p, 3)
            feats.append({"type": "Feature", "properties": props, "geometry": geom_wgs})
        path = os.path.join(a.out, f"{k}.geojson")
        with open(path, "w") as f:
            json.dump({"type": "FeatureCollection", "name": k, "features": feats}, f)
        summary[k] = len(feats)
        print(f"{k}: {len(feats)} features -> {path}")
    with open(os.path.join(a.out, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ----------------------------------------------------------------------------
# stage 3: evaluation (one-to-one IoU match, UTM metres)
# ----------------------------------------------------------------------------
def load_polys(path, crs_utm):
    gj = json.load(open(path))
    out = []
    for ft in gj["features"]:
        g = ft["geometry"]
        if g is None:
            continue
        if crs_utm:
            g = transform_geom("EPSG:4326", crs_utm, g)
        p = shape(g).buffer(0)
        if not p.is_empty:
            out.append(p)
    return out


def evaluate(a):
    crs_utm = a.crs
    pred, ref = load_polys(a.pred, crs_utm), load_polys(a.ref, crs_utm)
    tree = STRtree(pred)
    pairs = []
    for i, r in enumerate(ref):
        for j in tree.query(r):
            p = pred[j]
            inter = r.intersection(p).area
            if inter <= 0:
                continue
            iou = inter / r.union(p).area
            if iou >= a.iou:
                pairs.append((iou, i, int(j)))
    pairs.sort(reverse=True)
    used_r, used_p, tp = set(), set(), 0
    matched_area_ratio = []
    for iou, i, j in pairs:
        if i in used_r or j in used_p:
            continue
        used_r.add(i)
        used_p.add(j)
        tp += 1
        matched_area_ratio.append(pred[j].area / max(ref[i].area, 1e-6))
    precision = tp / max(len(pred), 1)
    recall = tp / max(len(ref), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    res = {"n_pred": len(pred), "n_ref": len(ref), "tp": tp, "precision": round(precision, 3),
           "recall": round(recall, 3), "f1": round(f1, 3),
           "median_area_ratio_pred_over_ref": round(float(np.median(matched_area_ratio)), 3) if matched_area_ratio else None,
           "iou_threshold": a.iou}
    print(json.dumps(res, indent=2))
    return res


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("infer", help="SAM 3 sliding-window soft maps")
    p.add_argument("--ortho", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--win", type=int, default=1000, help="window px (SAM 3 resizes to 1008, ~native at 0.5 m)")
    p.add_argument("--overlap", type=int, default=200, help="px shared by neighbouring windows")
    p.add_argument("--device", default="cuda")
    p.add_argument("--confidence-floor", type=float, default=0.0)
    p.add_argument("--p-field", default=DEFAULT_PROMPTS["field"])
    p.add_argument("--p-bund", default=DEFAULT_PROMPTS["bund"])
    p.add_argument("--p-tree", default=DEFAULT_PROMPTS["tree"])
    p.add_argument("--p-road", default=DEFAULT_PROMPTS["road"])
    p.set_defaults(fn=infer)

    q = sub.add_parser("postprocess", help="block-wise polygonisation to GeoJSON")
    q.add_argument("--out", required=True)
    q.add_argument("--block", type=int, default=4000, help="block px, memory bound")
    q.add_argument("--halo", type=int, default=200, help="px of context around each block")
    q.add_argument("--t-field", type=float, default=0.5)
    q.add_argument("--t-bund", type=float, default=0.4)
    q.add_argument("--t-tree", type=float, default=0.5)
    q.add_argument("--t-road", type=float, default=0.5)
    q.add_argument("--min-area-m2", type=float, default=150.0, help="smallest field (Indian smallholdings ~400 m2+)")
    q.add_argument("--min-bund-m2", type=float, default=2.0)
    q.add_argument("--min-tree-m2", type=float, default=4.0)
    q.add_argument("--min-road-m2", type=float, default=20.0)
    q.add_argument("--simplify-m", type=float, default=0.6, help="edge simplification tolerance (m), topology kept")
    q.set_defaults(fn=postprocess)

    e = sub.add_parser("eval", help="IoU match predicted fields against reference polygons")
    e.add_argument("--pred", required=True)
    e.add_argument("--ref", required=True)
    e.add_argument("--crs", default="EPSG:32643", help="equal-area metric CRS (UTM 43N for Nashik/Loni)")
    e.add_argument("--iou", type=float, default=0.5)
    e.set_defaults(fn=evaluate)

    args = ap.parse_args(argv)
    if args.cmd == "infer":
        args.stride = args.win - args.overlap
    return args.fn(args)


if __name__ == "__main__":
    main()
