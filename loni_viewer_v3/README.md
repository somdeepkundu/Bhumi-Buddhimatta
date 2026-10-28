# Loni field viewer v3

The Loni Devkar field-by-field crop and water viewer, rebuilt on the new-model plots.
Somdeep Kundu, RuDRA Lab, C-TARA, IIT Bombay

| File | What it is |
|---|---|
| `plots_v3_raw.geojson` | 3,770 plots from the new model (`aps2/sam3_field_boundaries/samgeo_fields.py`, full 0.5 m COG, release `aps2-drone-0.5m`) |
| `plots_v3.geojson` | 3,581 plots after `resolve_overlaps_v3.py`: near-duplicates dropped, no two plots overlap |
| `units_v3.gpkg` | 5,716 units (4,965 plot pieces + 751 rest of gut), from `prepare_units_v1.py` and `aps2/withSurveyNo_fieldsOnly_wgs84.geojson`; EPSG:32643 |
| `plots_v3b.geojson` | 3,759 plots: v3 plus 178 fallow plots (`source` = `drone` or `fallow`), from `add_fallow_v3.py` |
| `units_v3b.gpkg` | 5,970 units (5,196 plot pieces + 774 rest of gut) from `plots_v3b.geojson` |
| `gats_outside_drone.csv` | the 16 guts (91 ha) less than half covered by the drone image |
| `add_fallow_v3.py` | picks fallow and bare plots from the run's set-aside segments (`maybe_not_field.geojson`) |
| `prepare_units_v1.py` | the same unit builder as v2 (plots clipped to guts, plus each gut's remainder) |
| `resolve_overlaps_v3.py` | makes the plot layer non-overlapping |
| `notes_v3.md` | lines for the viewer's "Where the data is imprecise" panel |
| `SERVER_STEPS.md` | NDVI, crop cycles and build on the RuDRA server with its current scripts |
| `loni_field_viewer_drone_v3b.html` | the viewer, once built on the server |
| `loni_plots_per_gut.html` | interactive map + chart of every gut over the kharif 2023 drone image: old segmentation model vs new model (plot count, median size, gut covered, plots under 200 m²; old / new / change). Search or pick a gut to zoom to it and see its plots and numbers. Works offline and on phones, with a 5-step guide |
| `village_map/` | from the Loni Devkar village map (DWG → `LONI-DEOKAR_dwg.gpkg`): roads, streams, nala, cart tracks, wells, mango trees, temple, 550 boundary stones (`village_map_features.geojson`); and 233 ground-truth crop points from 2022-23 and 2024 field work (`loni_ground_truth.geojson`, flagged in/out of each model's plots) |
| `plots_per_gut.py` | per-gut plot areas: each plot goes to the gut holding its centre point |
| `plots_per_gut_map.py`, `plots_per_gut_template.html` | build the page (Leaflet and the drone image inlined) |
| `loni_plot_boundaries.html` | old SAM fields and new drone plots side by side on synced maps over the drone image (works offline) |

Rebuild the units:

```bash
python resolve_overlaps_v3.py --plots plots_v3_raw.geojson --out plots_v3.geojson
python prepare_units_v1.py --guts ../aps2/withSurveyNo_fieldsOnly_wgs84.geojson --sam plots_v3.geojson --out units_v3.gpkg
# v3b: add fallow plots (maybe_not_field.geojson and the COG are on Drive / the release), then the same two steps
python add_fallow_v3.py --plots plots_v3.geojson --maybe maybe_not_field.geojson --ortho loni_0.5m_cog.tif --out plots_v3b_raw.geojson
python resolve_overlaps_v3.py --plots plots_v3b_raw.geojson --out plots_v3b.geojson
python prepare_units_v1.py --guts ../aps2/withSurveyNo_fieldsOnly_wgs84.geojson --sam plots_v3b.geojson --out units_v3b.gpkg
```

Note: `resolve_overlaps_v3.py` keeps only geometry and an id; the `source` column in
`plots_v3b.geojson` was put back afterwards by matching each plot to the fallow candidates.
