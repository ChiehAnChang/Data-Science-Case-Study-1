"""Verify that all expected figures exist in Assets/Outputs/.

Called after the ETL & EDA pipeline step. If the pipeline succeeded,
these are fresh figures; if it failed, these are pre-committed fallbacks.
Exits non-zero only when figures are truly missing.
"""

import sys
from pathlib import Path

EDA_FIGURES_DIR = Path(__file__).resolve().parents[1] / "Assets" / "Outputs" / "EDA" / "image"

EDA_EXPECTED_FIGURES = [
    "eda_micro_analysis_daily_weekly_patterns.png",
    "eda_act2_monthly_yearly_patterns.png",
    "eda_act2_long_term_timeline_seasons.png",
    "eda_station_corr_heatmap_before_grouping.png",
    "bc_air_zones_station_network_corrected.png",
    "eda_zone_act1_1_daily_rhythm.png",
    "eda_zone_act1_2_weekday_weekend_facets.png",
    "eda_zone_act1_3_weekly_accumulation.png",
    "eda_zone_act2_4_monthly_seasonality.png",
    "eda_zone_act2_5_yoy_volatility_facets.png",
    "eda_station_corr_heatmap_grouped_by_air_zone.png",
    "bc_air_quality_network_assigned_zones.png",
]

missing = [f for f in EDA_EXPECTED_FIGURES if not (EDA_FIGURES_DIR / f).exists()]

if missing:
    print(f"ERROR: {len(missing)} figure(s) missing from {EDA_FIGURES_DIR}:")
    for f in missing:
        print(f"  - {f}")
    sys.exit(1)

print(f"All {len(EDA_EXPECTED_FIGURES)} figures present in {EDA_FIGURES_DIR}")
