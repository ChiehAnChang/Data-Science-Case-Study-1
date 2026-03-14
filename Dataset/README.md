# Dataset Folder

This folder contains all raw and processed data for both Case Study 1 (CS1) and Case Study 2 (CS2).

---

## Folder Structure

```text
Dataset/
├── raw_datasets/                          # CS1 — Raw input data
│   ├── PM25_with_geo_2021.csv             # Hourly station-level PM2.5 with lat/lon (2021)
│   ├── PM25_with_geo_2022.csv             # (2022)
│   ├── PM25_with_geo_2023.csv             # (2023)
│   ├── PM25_with_geo_2024.csv             # (2024)
│   ├── PM25_with_geo_2025.csv             # (2025)
│   └── bcairzones.geojson                 # BC air zone polygons (EPSG:4326)
│
├── Wildfire/                              # CS2 — Raw CWFIS Fire M3 hotspot CSVs
│   └── (populated by running ETL_pipeline_cs2.py — see note below)
│
├── Outputs/                               # CS1 & CS2 — Processed outputs
│   ├── PM25_zone_wide_imputed_2022_2025.csv   # CS1: Cleaned & imputed zone-median PM2.5
│   └── CS2_model_input.csv                    # CS2: Final feature table (70,128 rows × 24 cols)
│
├── README.md                              # This file
└── data_dictionary_cs2.md                 # CS2: Column-level definitions for CS2_model_input.csv
```

---

## Note on the Wildfire/ Folder

The raw CWFIS Fire M3 hotspot files (`2022_hotspots.csv` through `2025_hotspots.csv`) are **not committed** to the repository because the full global dataset is ~9.6 million rows and too large for version control.

To populate the folder, download the yearly archives from the CWFIS datamart and extract them into `Dataset/Wildfire/`, then run the CS2 ETL pipeline **locally**:

```bash
python "ETL & EDA/ETL_pipeline_cs2.py"
```

This will read the hotspot CSVs, join them with the PM2.5 data, and write `Dataset/Outputs/CS2_model_input.csv`. See `ETL & EDA/README_CS2.md` for required dependencies and full details.

---

## Data Dictionary

Column-level definitions for the CS2 feature table are in [`data_dictionary_cs2.md`](data_dictionary_cs2.md).
