# Case Study 2 — ETL Pipeline

## Overview

This pipeline extends Case Study 1 by integrating wildfire hotspot data with the
existing cleaned PM2.5 dataset, producing a model-ready feature table for the
**Lower Fraser Valley** and **Southern Interior** BC air zones.

The output supports three-model comparison (CS1 baseline, feature-augmented baseline,
transformer-style model) focused on improving PM2.5 forecasting and warning reliability
during wildfire smoke episodes.

---

## Files

| File | Description |
|---|---|
| `ETL_pipeline_cs2.py` | Self-contained ETL script (run directly) |
| `Dataset/Wildfire/[year]_hotspots.csv` | Raw CWFIS Fire M3 hotspot data (input) |
| `Dataset/Outputs/PM25_zone_wide_imputed_2022_2025.csv` | CS1 cleaned PM2.5 output (input) |
| `Dataset/Outputs/CS2_model_input.csv` | Final feature table (output) |

---

## How to Run

```bash
uv run python "ETL & EDA/ETL_pipeline_cs2.py"
```

No arguments needed. All paths are resolved relative to the project root.
Runtime: approximately 3–5 minutes (dominated by loading 9.6M global hotspot rows).

---

## Data Sources

### PM2.5 (from Case Study 1)
- **File**: `Dataset/Outputs/PM25_zone_wide_imputed_2022_2025.csv`
- **Content**: Hourly zone-median PM2.5 (µg/m³) across 7 BC air zones, 2022–2025,
  already cleaned and imputed by the CS1 pipeline.
- **Time reference**: BC AQMS local PST (UTC−8, fixed, no DST adjustment).

### Wildfire Hotspots (CWFIS Fire M3)
- **Source**: Canadian Wildland Fire Information System (CWFIS) datamart
- **URL**: `http://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/`
- **Files**: `2022_hotspots.zip` through `2025_hotspots.zip` (extracted to `Dataset/Wildfire/`)
- **Detection**: Satellite-based (MODIS + VIIRS-I sensors, NASA)
- **Time reference**: UTC (satellite overpass times)
- **Coverage**: Global; pipeline filters to BC geographic bounding box
  (lat 48.3–60.0°N, lon −139.0–−114.0°E)

#### Why Fire M3 Hotspots over other CWFIS datasets

| Dataset | Reason rejected |
|---|---|
| National Burned Area Composite (NBAC) | Annual granularity only; cannot align to hourly PM2.5 |
| National Fire Database (NFDB) | Aggregated annual fire reports; no sub-daily timestamps |
| Large Fire Database (LFDB) | Only covers 1959–1999; no longer maintained |
| Alberta Smoke Plume Observations | Wrong province; only 2010–2015 |
| FWI System mapped grid | Weather-based danger index only; no fire locations or download CSV |

Fire M3 is the only CWFIS product that provides: (1) precise lat/lon per detection,
(2) a datetime timestamp enabling hourly aggregation, and (3) quantitative fire
intensity columns (FRP, HFI, FWI) covering the full 2022–2025 range.

---

## Key Design Decisions

### Alert Threshold: 25 µg/m³
BC's ambient air quality objective for PM2.5 (24-hour average) is 25 µg/m³ — the
threshold used by the BC Ministry of Health to issue public advisories. Applied here
to hourly zone-median values. Because the zone median smooths readings across many
stations, an hourly zone-median of 25 µg/m³ represents a genuine, widespread smoke
event rather than noise from a single station. This makes the binary alert label
directly actionable for the stakeholder (Environmental Health Director) and defensible
in the final presentation.

### Regional Signal: Distance-weighted IDW within 300 km
BC wildfire smoke does not respect administrative zone boundaries. Major fires in the
Interior (e.g., Lytton, Kamloops corridor) routinely produce plumes that reach the
Lower Fraser Valley within 6–24 hours via the Fraser Valley wind corridor. A strict
zone-boundary filter would miss this mechanism entirely.

The 300 km radius was chosen because:
- 200 km captures immediately adjacent zones
- 300 km covers the broader BC Interior wildfire belt that influences both target zones
- Beyond ~400 km, smoke transport lag exceeds the hourly forecast horizon

IDW weight formula: `w = 1 / dist²` — the same inverse-distance-squared scheme used
in CS1's spatial imputation, ensuring methodological consistency.

Two complementary signals are produced per zone:
- **Local** (`fire_count_local`, `frp_local_sum`): fires strictly inside the zone polygon
  (spatial join with `bcairzones.geojson`)
- **Regional** (`hfi_weighted`, `frp_regional_sum`, `fire_count_regional`): all fires
  within 300 km, distance-weighted

### Timezone: UTC throughout
Wildfire `rep_date` is in UTC (satellite overpass). BC AQMS PM2.5 is in PST (UTC−8,
fixed). All timestamps are standardised to UTC internally; local time is only used
for display in figures. This avoids DST ambiguity (BC changes offset in March and
November, within the data range).

### Why FRP + HFI + FWI as signals

| Signal | Type | Why included |
|---|---|---|
| `frp` (Fire Radiative Power, MW) | Satellite-measured | Directly observed; no modelling assumptions; best proxy for actual fire energy output |
| `hfi` (Head Fire Intensity, kW/m) | Physics-modelled | Incorporates fuel type and weather; more physically meaningful than raw FRP for smoke production |
| `fwi` (Fire Weather Index) | Weather-driven composite | Available even on non-fire days; signals fire-prone atmospheric conditions before ignition |

---

## Pipeline Steps

| Step | Function | What it does |
|---|---|---|
| 1 | `load_wildfire_data()` | Load 4 yearly CSVs, filter to BC bbox, parse UTC datetime |
| 2 | `get_zone_centroids()` | Compute WGS-84 zone centroids from `bcairzones.geojson` |
| 3 | `tag_proximity()` | Haversine distance + IDW weight per hotspot per zone |
| 4 | `tag_local_fires()` | Spatial join (EPSG:3005) to flag fires inside zone polygons |
| 5 | `aggregate_to_hourly()` | Collapse detections to hourly local + regional signals per zone |
| 6 | `fill_complete_time_index()` | Build gapless hourly UTC index; zero-fill non-fire hours |
| 7 | `load_pm25_zone()` | Load CS1 output, select two zones, melt to long format, PST→UTC |
| 8 | `merge_pm25_wildfire()` | Left-join PM2.5 onto wildfire index on (Datetime_UTC, Zone) |
| 9 | `build_features()` | Lag/rolling/calendar features + binary alert label |
| 10 | `save_output()` | Write `CS2_model_input.csv` to `Dataset/Outputs/` |

---

## Output Schema

**File**: `Dataset/Outputs/CS2_model_input.csv`
**Shape**: 70,128 rows × 24 columns
**Format**: Long (one row per UTC hour per zone)

| Column | Description |
|---|---|
| `Datetime_UTC` | Hourly timestamp in UTC |
| `Zone` | `Lower Fraser Valley` or `Southern Interior` |
| `PM25` | Zone-median PM2.5 (µg/m³), imputed by CS1 |
| `fire_count_local` | Satellite detections inside zone polygon |
| `frp_local_sum` | Total FRP (MW) inside zone polygon |
| `fire_count_regional` | Detections within 300 km of zone centroid |
| `frp_regional_sum` | Total FRP (MW) within 300 km |
| `hfi_weighted` | Sum of HFI × (1/dist²) — IDW intensity signal |
| `fwi_mean` | Mean Fire Weather Index within 300 km |
| `PM25_lag_1h` | PM2.5 one hour ago |
| `PM25_lag_6h` | PM2.5 six hours ago |
| `PM25_lag_24h` | PM2.5 24 hours ago |
| `PM25_lag_48h` | PM2.5 48 hours ago |
| `PM25_roll_24h_mean` | 24-hour rolling mean PM2.5 |
| `PM25_roll_24h_max` | 24-hour rolling max PM2.5 |
| `PM25_roll_7d_mean` | 7-day rolling mean PM2.5 |
| `frp_roll_24h_sum` | 24-hour cumulative regional FRP |
| `frp_roll_72h_sum` | 72-hour cumulative regional FRP |
| `hour` | Hour of day (0–23, UTC) |
| `day_of_week` | Day of week (0=Monday, 6=Sunday) |
| `month` | Month (1–12) |
| `season` | Winter / Spring / Summer / Autumn |
| `is_wildfire_season` | 1 if May–October, else 0 |
| `alert` | 1 if PM25 > 25 µg/m³, else 0 (classification target) |

### Alert rate summary (2022–2025)
| Zone | Hours above 25 µg/m³ | Rate |
|---|---|---|
| Lower Fraser Valley | 185 | 0.53% |
| Southern Interior | 587 | 1.67% |

The higher rate in Southern Interior reflects its proximity to BC's most active
wildfire corridors (Okanagan, Kamloops).

---

## Notes on Station-Level PM2.5

This pipeline uses zone-aggregated PM2.5 (median across all stations per zone) from
the CS1 output. If station-level PM2.5 is needed for the transformer model's spatial
input, run the CS1 ETL pipeline (`ETL & EDA/ETL_EDA_pipeline.py`) and save the
intermediate `full_imputed_data_with_air_zones` DataFrame before the final aggregation
step. Filter to `Zone in ["Lower Fraser Valley", "Southern Interior"]` before saving.

---

## Dependencies

Managed by `uv` / `pyproject.toml`. Key packages: `pandas`, `geopandas`, `numpy`,
`shapely`. No internet access required at runtime — all data is local.
