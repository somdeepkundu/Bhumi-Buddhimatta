# Loni field viewer v3

The Loni Devkar field-by-field crop and water viewer, rebuilt on the October 2026 drone plots.
Somdeep Kundu, RuDRA Lab, C-TARA, IIT Bombay

| File | What it is |
|---|---|
| `plots_v3_raw.geojson` | 3,770 plots from the drone run (`aps2/sam3_field_boundaries/samgeo_fields.py`, full 0.5 m COG, release `aps2-drone-0.5m`) |
| `plots_v3.geojson` | 3,581 plots after `resolve_overlaps_v3.py`: near-duplicates dropped, no two plots overlap |
| `units_v3.gpkg` | 5,716 units (4,965 plot pieces + 751 rest of gut), from `prepare_units_v1.py` and `aps2/withSurveyNo_fieldsOnly_wgs84.geojson`; EPSG:32643 |
| `prepare_units_v1.py` | the same unit builder as v2 (plots clipped to guts, plus each gut's remainder) |
| `resolve_overlaps_v3.py` | makes the plot layer non-overlapping |
| `notes_v3.md` | lines for the viewer's "Where the data is imprecise" panel |
| `SERVER_STEPS.md` | NDVI, crop cycles and build on the RuDRA server with its current scripts |
| `loni_field_viewer_drone_v3.html` | the viewer, once built on the server |

Rebuild the units:

```bash
python resolve_overlaps_v3.py --plots plots_v3_raw.geojson --out plots_v3.geojson
python prepare_units_v1.py --guts ../aps2/withSurveyNo_fieldsOnly_wgs84.geojson --sam plots_v3.geojson --out units_v3.gpkg
```
