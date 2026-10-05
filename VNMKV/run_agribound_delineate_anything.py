"""Run agribound's real Delineate-Anything engine on the georeferenced screenshot.
Needs internet to Hugging Face (weights: MykolaL/DelineateAnything) + ideally a GPU (Colab is fine):
    pip install "agribound[delineate-anything]"
Citation: Majumdar, S. et al. (2026) Agribound v1.0.1, Zenodo, doi:10.5281/zenodo.19229665
          Lavreniuk, M. et al. (2025) Delineate Anything, ECAI 2025, doi:10.48550/arXiv.2504.02534
"""
import agribound, geopandas as gpd, rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

SRC, UTM = "image1_georef_EPSG3857.tif", "image1_georef_UTM43N.tif"
with rasterio.open(SRC) as s:                       # Web Mercator -> UTM 43N (metric, true scale)
    t, w, h = calculate_default_transform(s.crs, "EPSG:32643", s.width, s.height, *s.bounds)
    prof = s.profile | dict(crs="EPSG:32643", transform=t, width=w, height=h)
    with rasterio.open(UTM, "w", **prof) as d:
        for b in range(1, s.count + 1):
            reproject(rasterio.band(s, b), rasterio.band(d, b), resampling=Resampling.bilinear)

gdf = agribound.delineate(
    source="local", local_tif_path=UTM, bands={"R": 1, "G": 2, "B": 3},
    engine="delineate-anything",
    engine_params={"sam_refine": True},             # optional SAM 2 box refinement (agribound[samgeo])
    lulc_filter=False,                              # LULC filter needs Earth Engine
    output_path="fields_DA_parbhani.gpkg",
)
ref = gpd.read_file("2_fields.geojson")
ref = ref[ref["__gm_id"] != "feature-5"].to_crs(32643)
print(agribound.evaluate(gdf.to_crs(32643), ref, iou_threshold=0.5))
