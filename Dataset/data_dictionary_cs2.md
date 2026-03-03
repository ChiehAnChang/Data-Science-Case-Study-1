# Data Dictionary – Case Study 2

**File:** `Dataset/Outputs/CS2_model_input.csv`
**Rows:** 70,128 (one row per UTC hour per zone, 2022–2025)
**Columns:** 24
**Zones covered:** Lower Fraser Valley, Southern Interior

---

## Identifiers

| Features | Definition |
| :--- | :--- |
| `Datetime_UTC` | Timestamp of the observation in UTC. Hourly resolution. PM2.5 source data was originally in BC local PST (UTC−8, fixed, no daylight saving adjustment) and has been shifted +8 hours to align with the wildfire data. |
| `Zone` | The BC air quality zone this row belongs to. One of: `Lower Fraser Valley` or `Southern Interior`. |

---

## PM2.5 (from Case Study 1)

| Features | Definition |
| :--- | :--- |
| `PM25` | Hourly zone-median PM2.5 concentration in micrograms per cubic metre (µg/m³). Computed as the median across all monitoring stations within the zone. Already cleaned and imputed by the CS1 pipeline (type conversion, deduplication, 20% missing threshold removal, common station filtering, and 3-stage imputation). Source: BC Air Quality Monitoring System (AQMS), 2022–2025. |

---

## Wildfire Signals – Regional (CWFIS Fire M3, within 300 km of zone centroid)

These four columns aggregate all satellite-detected hotspots within 300 km of the zone centroid for each UTC hour. The 300 km radius is chosen to capture cross-zone smoke transport (e.g., Interior fires affecting Lower Fraser Valley via the Fraser Valley wind corridor). Hours with no fire detections are set to 0, not missing.

| Features | Definition |
| :--- | :--- |
| `fire_count_regional` | Number of satellite hotspot detections within 300 km of the zone centroid in that hour. Each detection represents one satellite pixel flagged as an active fire by MODIS or VIIRS-I sensors. |
| `frp_regional_sum` | Total Fire Radiative Power (FRP) in megawatts (MW) from all hotspots within 300 km. FRP is directly measured by satellite and represents the rate of radiant energy released by fire. Higher values indicate more intense or widespread burning. |
| `hfi_weighted` | Distance-weighted Head Fire Intensity (HFI) signal. Computed as the sum of (HFI × 1/distance²) for all hotspots within 300 km, where distance is the Haversine great-circle distance in km from the hotspot to the zone centroid. HFI (kW/m) is a physics-based model of fire front energy output. Fires closer to the zone centroid receive higher weight. |
| `fwi_mean` | Mean Fire Weather Index (FWI) across all hotspots within 300 km. FWI is a composite index combining temperature, relative humidity, wind speed, and precipitation to measure fire weather danger. Available even on days with no active fire, making it a useful atmospheric condition feature. |

---

## Wildfire Signals – Local (hotspots inside zone polygon)

These two columns count only fires physically located inside the zone boundary, determined by a spatial join with the BC air zone GeoJSON polygons (EPSG:3005 projection). Hours with no local fire detections are set to 0.

| Features | Definition |
| :--- | :--- |
| `fire_count_local` | Number of satellite hotspot detections inside the zone polygon in that hour. |
| `frp_local_sum` | Total Fire Radiative Power (FRP) in megawatts (MW) from all hotspots inside the zone polygon. |

---

## PM2.5 Lag Features (Feature Engineering)

Autoregressive features computed per zone. Represent the PM2.5 value at a fixed number of hours prior to the current row. The first few rows of each zone's series will contain NaN (normal boundary effect at the start of 2022).

| Features | Definition |
| :--- | :--- |
| `PM25_lag_1h` | PM2.5 value from 1 hour before the current timestamp, within the same zone. |
| `PM25_lag_6h` | PM2.5 value from 6 hours before the current timestamp, within the same zone. |
| `PM25_lag_24h` | PM2.5 value from 24 hours before the current timestamp, within the same zone. Captures same-hour-yesterday baseline. |
| `PM25_lag_48h` | PM2.5 value from 48 hours before the current timestamp, within the same zone. |

---

## PM2.5 Rolling Features (Feature Engineering)

Rolling window statistics computed per zone over the hours preceding (and including) the current timestamp. `min_periods=1` is used so no additional NaN is introduced beyond the lag boundary.

| Features | Definition |
| :--- | :--- |
| `PM25_roll_24h_mean` | Mean PM2.5 over the trailing 24-hour window, within the same zone. Smooths short-term noise and captures recent air quality trend. |
| `PM25_roll_24h_max` | Maximum PM2.5 over the trailing 24-hour window, within the same zone. Captures the worst reading in the recent period, useful for spike detection. |
| `PM25_roll_7d_mean` | Mean PM2.5 over the trailing 7-day (168-hour) window, within the same zone. Represents the medium-term background level. |

---

## Wildfire Rolling Features (Feature Engineering)

Rolling cumulative fire activity, computed per zone to capture the build-up of smoke-producing fire over time.

| Features | Definition |
| :--- | :--- |
| `frp_roll_24h_sum` | Total regional FRP (MW) accumulated over the trailing 24-hour window. Captures the fire load in the past day. |
| `frp_roll_72h_sum` | Total regional FRP (MW) accumulated over the trailing 72-hour window. Captures the fire load over the past 3 days, accounting for the lag between ignition and PM2.5 impact. |

---

## Calendar / Season Features (Feature Engineering)

| Features | Definition |
| :--- | :--- |
| `hour` | Hour of day in UTC (0–23). Captures diurnal patterns in PM2.5 (e.g., morning traffic, afternoon mixing height). |
| `day_of_week` | Day of the week as an integer (0 = Monday, 6 = Sunday). Captures weekly behavioural patterns. |
| `month` | Month of the year as an integer (1–12). |
| `season` | Season label derived from month: `Winter` (Dec–Feb), `Spring` (Mar–May), `Summer` (Jun–Aug), `Autumn` (Sep–Nov). |
| `is_wildfire_season` | Binary indicator equal to 1 if the month is May through October (wildfire season in BC), and 0 otherwise. |

---

## Target Variable

| Features | Definition |
| :--- | :--- |
| `alert` | Binary classification target. Equal to 1 if PM25 exceeds 25 µg/m³, and 0 otherwise. The 25 µg/m³ threshold is the BC ambient air quality objective for PM2.5 and the standard used by the BC Ministry of Health for public advisories. Alert rate: 0.53% for Lower Fraser Valley (185 hours), 1.67% for Southern Interior (587 hours). |
