"""
ETL Pipeline – Case Study 2
===========================
Wildfire hotspot cleaning and integration with PM2.5 for
Lower Fraser Valley and Southern Interior BC air zones.

Data Sources
------------
PM2.5 (zone-level, already cleaned by CS1 pipeline):
    Dataset/Outputs/PM25_zone_wide_imputed_2022_2025.csv

Wildfire hotspots (CWFIS Fire M3, satellite-detected, 2022-2025):
    Dataset/Wildfire/2022_hotspots.csv
    Dataset/Wildfire/2023_hotspots.csv
    Dataset/Wildfire/2024_hotspots.csv
    Dataset/Wildfire/2025_hotspots.csv

Output
------
Dataset/Outputs/CS2_model_input.csv
    Hourly long-format table (Datetime_UTC × Zone) ready for modelling, with:
      - PM2.5 (zone-median, imputed by CS1)
      - Local wildfire signals  (fires inside zone polygon)
      - Regional wildfire signals (IDW within 300 km of zone polygon boundary)
      - PM2.5 lag / rolling features
      - Fire rolling features
      - Calendar / season features
      - Binary alert label (PM2.5 > 25 µg/m³)

Design Decisions
----------------
Alert threshold  : 25 µg/m³  — BC regulatory advisory standard
IDW radius       : 300 km    — captures cross-zone smoke transport (measured from zone polygon boundary, not centroid)
Timezone         : UTC throughout; PM2.5 shifted +8 h from PST (fixed, no DST)
Wildfire signals : FRP (satellite-measured), HFI (modelled intensity), FWI (weather index)
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).resolve().parents[1]
RAW_DIR      = PROJECT_DIR / "Dataset" / "raw_datasets"
WILDFIRE_DIR = PROJECT_DIR / "Dataset" / "Wildfire"
PM25_CLEAN   = PROJECT_DIR / "Dataset" / "Outputs" / "PM25_zone_wide_imputed_2022_2025.csv"
GEOJSON_PATH = RAW_DIR / "bcairzones.geojson"
DATA_OUT_DIR = PROJECT_DIR / "Dataset" / "Outputs"

DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
TARGET_ZONES    = ["Lower Fraser Valley", "Southern Interior"]
ALERT_THRESHOLD = 25.0   # µg/m³  — BC ambient air quality advisory threshold
MAX_DIST_KM     = 300.0  # radius for regional IDW wildfire signal
PST_OFFSET_H    = 8      # BC AQMS data is PST (UTC−8), fixed, no DST adjustment
WILDFIRE_YEARS  = [2022, 2023, 2024, 2025]

# BC geographic bounding box — fast pre-filter before spatial ops
BC_LAT = (48.3, 60.0)
BC_LON = (-139.0, -114.0)


# ── Step 1: Load wildfire hotspot data ─────────────────────────────────────
def load_wildfire_data() -> pd.DataFrame:
    """
    Load CWFIS Fire M3 hotspot CSVs for 2022–2025.

    - Reads only the columns needed for cleaning and feature construction.
    - Filters to BC geographic bounding box (fast pre-filter).
    - Parses rep_date as UTC-aware datetime (satellite timestamps are UTC).
    - Drops rows with missing lat, lon, or rep_date.

    Returns
    -------
    pd.DataFrame with columns:
        rep_date (UTC), lat, lon, frp, hfi, fwi, source_year
    """
    COLS = ["rep_date", "lat", "lon", "frp", "hfi", "fwi"]

    frames = []
    for year in WILDFIRE_YEARS:
        path = WILDFIRE_DIR / f"{year}_hotspots.csv"
        df = pd.read_csv(path, usecols=COLS, low_memory=False)
        df["source_year"] = year
        frames.append(df)
        print(f"  {year}: {len(df):,} rows loaded")

    df_all = pd.concat(frames, ignore_index=True)
    print(f"\n  Global total: {len(df_all):,} rows")

    # Parse rep_date — satellite timestamps are already UTC; localize as UTC
    df_all["rep_date"] = pd.to_datetime(df_all["rep_date"], errors="coerce", utc=True)

    # Drop rows missing spatial or temporal identifiers
    before = len(df_all)
    df_all = df_all.dropna(subset=["rep_date", "lat", "lon"])
    dropped = before - len(df_all)
    if dropped:
        print(f"  Dropped {dropped:,} rows with missing rep_date / lat / lon")

    # Filter to BC bounding box
    df_bc = df_all[
        df_all["lat"].between(*BC_LAT) &
        df_all["lon"].between(*BC_LON)
    ].copy()
    print(f"  BC bounding-box rows: {len(df_bc):,}")

    return df_bc.reset_index(drop=True)


# ── Step 2: Zone polygons from GeoJSON ─────────────────────────────────────
def get_zone_geometries(geojson_path: Path) -> dict:
    """
    Load and dissolve the BC air-zone polygon for each target zone, projected
    to EPSG:3005 (BC Albers) for accurate planar distance calculations.

    Returns
    -------
    dict : { zone_name : shapely geometry (EPSG:3005) }
    """
    gdf = gpd.read_file(geojson_path).to_crs(epsg=3005)

    # Identify zone-name column (mirrors CS1 logic)
    candidates = [c for c in gdf.columns if "zone" in c.lower() or "name" in c.lower()]
    if not candidates:
        raise ValueError(f"Cannot find zone/name column in: {list(gdf.columns)}")
    gdf["ZoneName"] = gdf[candidates[0]]

    geometries = {}
    for zone in TARGET_ZONES:
        rows = gdf[gdf["ZoneName"] == zone]
        if rows.empty:
            raise ValueError(
                f"Zone '{zone}' not found in GeoJSON. "
                f"Available: {sorted(gdf['ZoneName'].unique())}"
            )
        try:
            union = rows.geometry.union_all()
        except AttributeError:
            # geopandas < 0.14 fallback
            union = rows.geometry.unary_union
        geometries[zone] = union
        print(f"  {zone}: polygon loaded ({len(rows)} feature(s))")

    return geometries


# ── Step 3: Tag each hotspot with zone proximity ───────────────────────────
def tag_proximity(df_fire: pd.DataFrame, zone_geometries: dict) -> pd.DataFrame:
    """
    For each target zone, compute the shortest distance (km) from every BC
    hotspot to the zone polygon boundary and derive an IDW weight (1 / dist²).

    Distance is measured in EPSG:3005 (BC Albers, metres) and converted to km.
    Fires inside the zone have distance = 0; a minimum of 0.1 km is applied
    before computing IDW weights to prevent division-by-zero.

    Columns added per zone:
        dist_<zone>   — shortest distance to zone polygon in km
        weight_<zone> — IDW weight = 1 / max(dist, 0.1)²

    Rows farther than MAX_DIST_KM from BOTH zone polygons are dropped —
    they cannot contribute to either zone's signal.
    """
    # Project hotspots to EPSG:3005 for planar distance measurement in metres
    gdf_fire = gpd.GeoDataFrame(
        df_fire.copy(),
        geometry=gpd.points_from_xy(df_fire["lon"], df_fire["lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3005)

    for zone, zone_geom in zone_geometries.items():
        dist_km = gdf_fire.geometry.distance(zone_geom) / 1000.0
        gdf_fire[f"dist_{zone}"]   = dist_km.values
        gdf_fire[f"weight_{zone}"] = 1.0 / (np.maximum(dist_km.values, 0.1) ** 2)

    df = pd.DataFrame(gdf_fire.drop(columns="geometry"))

    # Keep hotspots within MAX_DIST_KM of at least one target zone
    within = np.zeros(len(df), dtype=bool)
    for zone in zone_geometries:
        within |= df[f"dist_{zone}"] <= MAX_DIST_KM

    before = len(df)
    df = df[within].copy().reset_index(drop=True)
    print(
        f"  Retained {len(df):,} / {before:,} hotspots "
        f"within {MAX_DIST_KM:.0f} km of target zone polygons"
    )
    return df


# ── Step 4: Spatial join — tag fires inside zone polygons ──────────────────
def tag_local_fires(df_fire: pd.DataFrame, geojson_path: Path) -> pd.DataFrame:
    """
    Spatially join each hotspot against the BC air-zone polygons (EPSG:3005,
    same projection as CS1) to identify fires physically inside a target zone.

    Adds column:
        local_zone — zone name if fire is inside a target zone, else NaN
    """
    gdf_zones = gpd.read_file(geojson_path).to_crs(epsg=3005)
    candidates = [c for c in gdf_zones.columns if "zone" in c.lower() or "name" in c.lower()]
    gdf_zones["ZoneName"] = gdf_zones[candidates[0]]

    gdf_fire = gpd.GeoDataFrame(
        df_fire,
        geometry=gpd.points_from_xy(df_fire["lon"], df_fire["lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3005)

    joined = gpd.sjoin(
        gdf_fire,
        gdf_zones[["ZoneName", "geometry"]],
        how="left",
        predicate="within",
    )

    # Only mark as local if zone is one of our two target zones
    joined["local_zone"] = joined["ZoneName"].where(
        joined["ZoneName"].isin(TARGET_ZONES), other=np.nan
    )

    result = pd.DataFrame(
        joined.drop(columns=["geometry", "index_right", "ZoneName"], errors="ignore")
    ).reset_index(drop=True)

    local_count = result["local_zone"].notna().sum()
    print(f"  Hotspots inside target-zone polygons: {local_count:,}")

    return result


# ── Step 5: Aggregate to hourly signals per zone ───────────────────────────
def aggregate_to_hourly(df_fire: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse satellite detections to one row per (UTC-hour × zone).

    For each zone the following signals are computed:

    Regional (all hotspots within MAX_DIST_KM of zone polygon boundary):
        fire_count_regional  — number of satellite detections
        frp_regional_sum     — total Fire Radiative Power (MW)
        hfi_weighted         — Σ(hfi × 1/dist²)  IDW intensity signal
        fwi_mean             — mean Fire Weather Index

    Local (hotspots spatially inside the zone polygon):
        fire_count_local     — number of detections inside the zone
        frp_local_sum        — total FRP inside the zone (MW)

    Returns long-format DataFrame: (Datetime_UTC, Zone, …signals…)
    """
    df = df_fire.copy()
    df["hour_utc"] = df["rep_date"].dt.floor("h")

    # Coerce intensity columns to numeric; NaN → 0 (no active flame modelled)
    for col in ["hfi", "fwi", "frp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    records = []

    for zone in TARGET_ZONES:
        dist_col   = f"dist_{zone}"
        weight_col = f"weight_{zone}"

        # ── Regional: within 300 km ──────────────────────────────────────
        df_reg = df[df[dist_col] <= MAX_DIST_KM].copy()
        # Pre-compute per-row weighted HFI to avoid closure issues in agg
        df_reg["hfi_w"] = df_reg["hfi"] * df_reg[weight_col]

        reg_agg = (
            df_reg.groupby("hour_utc")
            .agg(
                fire_count_regional=("hfi_w",  "count"),
                frp_regional_sum   =("frp",    "sum"),
                hfi_weighted       =("hfi_w",  "sum"),
                fwi_mean           =("fwi",    "mean"),
            )
            .reset_index()
        )
        reg_agg["Zone"] = zone

        # ── Local: inside zone polygon ───────────────────────────────────
        df_loc = df[df["local_zone"] == zone]
        if not df_loc.empty:
            loc_agg = (
                df_loc.groupby("hour_utc")
                .agg(
                    fire_count_local=("frp", "count"),
                    frp_local_sum   =("frp", "sum"),
                )
                .reset_index()
            )
        else:
            loc_agg = pd.DataFrame(
                columns=["hour_utc", "fire_count_local", "frp_local_sum"]
            )

        merged = reg_agg.merge(loc_agg, on="hour_utc", how="left")
        merged["fire_count_local"] = merged["fire_count_local"].fillna(0).astype(int)
        merged["frp_local_sum"]    = merged["frp_local_sum"].fillna(0.0)

        records.append(merged)

    out = pd.concat(records, ignore_index=True).rename(
        columns={"hour_utc": "Datetime_UTC"}
    )
    print(f"  Hourly wildfire records: {len(out):,} rows across {len(TARGET_ZONES)} zones")
    return out


# ── Step 6: Complete hourly UTC index + zero-fill ──────────────────────────
def fill_complete_time_index(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Build a gapless hourly UTC DatetimeIndex spanning the full 2022–2025
    period, cross-joined with TARGET_ZONES.

    Merges aggregated wildfire data into this skeleton and zero-fills all
    fire-signal columns — hours with no satellite detections are confirmed
    zero-fire (not missing data).
    """
    start = pd.Timestamp("2022-01-01 00:00:00", tz="UTC")
    end   = pd.Timestamp("2025-12-31 23:00:00", tz="UTC")
    full_idx = pd.date_range(start=start, end=end, freq="h", tz="UTC")

    skeleton = pd.MultiIndex.from_product(
        [full_idx, TARGET_ZONES], names=["Datetime_UTC", "Zone"]
    ).to_frame(index=False)

    merged = skeleton.merge(df_hourly, on=["Datetime_UTC", "Zone"], how="left")

    fire_cols = [
        "fire_count_regional", "frp_regional_sum",
        "hfi_weighted", "fwi_mean",
        "fire_count_local", "frp_local_sum",
    ]
    merged[fire_cols] = merged[fire_cols].fillna(0.0)

    coverage_pct = len(df_hourly) / len(merged) * 100
    print(f"  Complete index: {len(merged):,} rows  |  fire-detection coverage: {coverage_pct:.1f}%")
    return merged


# ── Step 7: Load and prep CS1 PM2.5 zone output ───────────────────────────
def load_pm25_zone(pm25_path: Path) -> pd.DataFrame:
    """
    Read the CS1 zone-wide PM2.5 output, select target zones, convert
    Datetime from PST to UTC, and return a long-format DataFrame.

    Timezone note
    -------------
    BC's AQMS reports in PST (UTC−8) year-round with no DST adjustment.
    A fixed +8 h offset is added to align with the wildfire rep_date (UTC).
    The first timestamp in the file is 01:00:00 PST = 09:00:00 UTC, which
    confirms the PST convention.
    """
    df = pd.read_csv(pm25_path, parse_dates=["Datetime"])

    missing_zones = set(TARGET_ZONES) - set(df.columns)
    if missing_zones:
        raise ValueError(f"Zones missing from PM2.5 output: {missing_zones}")

    df = df[["Datetime"] + TARGET_ZONES].copy()

    # Melt wide → long
    df_long = df.melt(id_vars="Datetime", var_name="Zone", value_name="PM25")

    # PST → UTC: add fixed 8-hour offset
    df_long["Datetime_UTC"] = (
        df_long["Datetime"] + pd.Timedelta(hours=PST_OFFSET_H)
    ).dt.tz_localize("UTC")
    df_long = df_long.drop(columns=["Datetime"])

    missing_pm25 = df_long["PM25"].isna().sum()
    print(f"  PM2.5 rows: {len(df_long):,}  |  missing PM2.5: {missing_pm25:,}")
    return df_long


# ── Step 8: Merge PM2.5 + wildfire ─────────────────────────────────────────
def merge_pm25_wildfire(
    df_pm25: pd.DataFrame,
    df_wildfire: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join PM2.5 onto the complete wildfire index on (Datetime_UTC, Zone).

    PM2.5 is the anchor series — all its timestamps are preserved. Any
    wildfire-signal NaNs introduced by the merge (e.g., timestamps outside
    the wildfire data range) are zero-filled and reported.
    """
    merged = df_pm25.merge(df_wildfire, on=["Datetime_UTC", "Zone"], how="left")

    fire_cols = [
        "fire_count_regional", "frp_regional_sum",
        "hfi_weighted", "fwi_mean",
        "fire_count_local", "frp_local_sum",
    ]
    for col in fire_cols:
        n_nan = merged[col].isna().sum()
        if n_nan > 0:
            print(f"  Post-merge NaN in '{col}': {n_nan:,} — zero-filling")
            merged[col] = merged[col].fillna(0.0)

    print(f"  Merged shape: {merged.shape}")
    return merged


# ── Step 9: Feature engineering ────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct all model-input features on the merged PM2.5 + wildfire table.

    PM2.5 autoregressive lags (per zone):
        PM25_lag_1h, PM25_lag_6h, PM25_lag_24h, PM25_lag_48h

    PM2.5 rolling windows (per zone, min_periods=1):
        PM25_roll_24h_mean, PM25_roll_24h_max, PM25_roll_7d_mean

    Wildfire rolling (per zone, regional FRP cumulation):
        frp_roll_24h_sum, frp_roll_72h_sum

    Calendar features:
        hour, day_of_week (0=Mon), month,
        season (Winter/Spring/Summer/Autumn),
        is_wildfire_season (1 for May–Oct, 0 otherwise)

    Alert label:
        alert — binary (1 if PM25 > ALERT_THRESHOLD, else 0)
    """
    df = df.sort_values(["Zone", "Datetime_UTC"]).copy()

    # ── PM2.5 lags ─────────────────────────────────────────────────────────
    for h in [1, 6, 24, 48]:
        df[f"PM25_lag_{h}h"] = df.groupby("Zone")["PM25"].shift(h)

    # ── PM2.5 rolling ──────────────────────────────────────────────────────
    roll_specs = {
        "PM25_roll_24h_mean": (24,     "mean"),
        "PM25_roll_24h_max":  (24,     "max"),
        "PM25_roll_7d_mean":  (24 * 7, "mean"),
    }
    for col_name, (window, func) in roll_specs.items():
        df[col_name] = df.groupby("Zone")["PM25"].transform(
            lambda x, w=window, f=func: getattr(x.rolling(w, min_periods=1), f)()
        )

    # ── Wildfire rolling (regional FRP) ────────────────────────────────────
    for window, label in [(24, "24h"), (72, "72h")]:
        df[f"frp_roll_{label}_sum"] = df.groupby("Zone")["frp_regional_sum"].transform(
            lambda x, w=window: x.rolling(w, min_periods=1).sum()
        )

    # ── Calendar features ───────────────────────────────────────────────────
    dt = df["Datetime_UTC"].dt
    df["hour"]               = dt.hour
    df["day_of_week"]        = dt.dayofweek          # 0 = Monday … 6 = Sunday
    df["month"]              = dt.month
    df["is_wildfire_season"] = dt.month.isin([5, 6, 7, 8, 9, 10]).astype(int)

    season_map = {
        12: "Winter", 1: "Winter",  2: "Winter",
        3:  "Spring", 4: "Spring",  5: "Spring",
        6:  "Summer", 7: "Summer",  8: "Summer",
        9:  "Autumn", 10: "Autumn", 11: "Autumn",
    }
    df["season"] = df["month"].map(season_map)

    # ── Alert label ─────────────────────────────────────────────────────────
    df["alert"] = (df["PM25"] > ALERT_THRESHOLD).astype(int)

    print("  Alert rate by zone:")
    for zone, rate in df.groupby("Zone")["alert"].mean().items():
        n_alerts = (df[df["Zone"] == zone]["alert"] == 1).sum()
        print(f"    {zone}: {rate:.2%}  ({n_alerts:,} hours above {ALERT_THRESHOLD} ug/m3)")

    return df


# ── Step 10: Save output ───────────────────────────────────────────────────
def save_output(df: pd.DataFrame, filename: str = "CS2_model_input.csv") -> Path:
    """Save the final feature table to Dataset/Outputs/."""
    out_path = DATA_OUT_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"\n  [Saved] {out_path}")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    return out_path


# ── Main pipeline ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("CS2 ETL Pipeline - Wildfire Integration")
    print("=" * 60)

    # 1. Wildfire data
    print("\n[1/8] Loading wildfire hotspot data ...")
    df_fire = load_wildfire_data()

    # 2. Zone polygons
    print("\n[2/8] Loading zone polygons from GeoJSON ...")
    zone_geometries = get_zone_geometries(GEOJSON_PATH)

    # 3. Proximity tagging (polygon boundary distances + IDW weights)
    print("\n[3/8] Tagging hotspot proximity to target zones ...")
    df_fire = tag_proximity(df_fire, zone_geometries)

    # 4. Local fire tagging (spatial join with zone polygons)
    print("\n[4/8] Spatial join: identifying fires inside zone polygons ...")
    df_fire = tag_local_fires(df_fire, GEOJSON_PATH)

    # 5. Aggregate to hourly zone-level signals
    print("\n[5/8] Aggregating hotspots to hourly zone signals ...")
    df_hourly_fire = aggregate_to_hourly(df_fire)

    # 6. Complete hourly UTC index + zero-fill non-fire hours
    print("\n[6/8] Building complete hourly index and zero-filling ...")
    df_wildfire_complete = fill_complete_time_index(df_hourly_fire)

    # 7. Load CS1 PM2.5 output and convert to UTC long format
    print("\n[7/8] Loading CS1 PM2.5 zone data ...")
    df_pm25 = load_pm25_zone(PM25_CLEAN)

    # 8. Merge + feature engineering
    print("\n[8/8] Merging datasets and building features ...")
    df_merged = merge_pm25_wildfire(df_pm25, df_wildfire_complete)
    df_final  = build_features(df_merged)

    # Save
    save_output(df_final)

    print("\nDone. CS2 ETL pipeline complete.")
