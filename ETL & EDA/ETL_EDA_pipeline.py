# package import
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
import seaborn as sns

# -------------------------
# Paths
# -------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]   # script in "ETL & EDA/"
RAW_DIR = PROJECT_DIR / "Dataset" / "raw_datasets"

DATA_OUT_DIR = PROJECT_DIR / "Dataset" / "Outputs"     # cleaned datasets
PLOT_OUT_DIR = PROJECT_DIR / "Assets" / "Outputs" / "EDA" / "image"     # EDA plots

DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Functions for steps
# -------------------------
def load_data():
    """Load raw data files."""
    df_2022 = pd.read_csv(RAW_DIR / "PM25_with_geo_2022.csv")
    df_2023 = pd.read_csv(RAW_DIR / "PM25_with_geo_2023.csv")
    df_2024 = pd.read_csv(RAW_DIR / "PM25_with_geo_2024.csv")
    df_2025 = pd.read_csv(RAW_DIR / "PM25_with_geo_2025.csv")
    return [df_2022, df_2023, df_2024, df_2025]

def inspect_raw_data(all_raw_dataframes, start_year=2022):
    """Print shape + df.info() for each yearly dataframe."""
    for year, df in enumerate(all_raw_dataframes, start=start_year):
        print(f"\n=== Year {year} ===")
        print(f"Shape: {df.shape}")
        print(df.info())


def convert_types(all_raw_dataframes):
    """
    Convert Datetime to pandas datetime and Station to string.
    Returns the same list (mutated in place) for convenience.
    """
    for df in all_raw_dataframes:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        df["Station"] = df["Station"].astype(str)

        # quick check (safe even if some NaT exist)
        year_guess = df["Datetime"].dropna().dt.year.iloc[0] if df["Datetime"].notna().any() else "Unknown"
        print(f"Type conversion done (year ~ {year_guess}). Key dtypes:")
        print(df[["Datetime", "Station"]].dtypes)
        print()
    return all_raw_dataframes


def deduplicate_each_year(all_raw_dataframes, start_year=2022):
    """
    Drop duplicates based on ['Datetime', 'Station'] within each year's dataframe.
    Keeps the LAST occurrence (same as your code).
    Returns the same list (mutated in place) for convenience.
    """
    for year, df in enumerate(all_raw_dataframes, start=start_year):
        before = df.shape[0]
        after = df.drop_duplicates(subset=["Datetime", "Station"], keep="last").shape[0]
        num_duplicates = before - after

        print(f"Year {year}: duplicates by (Datetime, Station) = {num_duplicates}")

        # remove duplicates in place
        df.drop_duplicates(subset=["Datetime", "Station"], keep="last", inplace=True)

        print(f"Year {year}: rows after dedup = {df.shape[0]}\n")

    return all_raw_dataframes

def summarize_yearly_coverage(all_raw_dataframes, start_year=2022):
    """Print basic yearly stats: rows, time range, overall missing percentage, unique stations."""
    for year, df in enumerate(all_raw_dataframes, start=start_year):
        print(f"\nYEAR: {year}")
        print(f"Records: {len(df)}")

        if df.empty:
            print("Time Range: N/A (empty dataframe)")
            print("Missing Values Percentage: N/A (empty dataframe)")
            print("Total Unique Stations: 0")
            continue

        time_min = df["Datetime"].min()
        time_max = df["Datetime"].max()

        missing_pct = (df.isna().sum().sum() / df.size * 100) if df.size > 0 else 0.0
        n_stations = df["Station"].nunique(dropna=True)

        print(f"Time Range: {time_min} to {time_max}")
        print(f"Missing Values Percentage (all columns): {missing_pct:.2f}%")
        print(f"Total Unique Stations: {n_stations}")


def report_missing_pm25_by_station(all_raw_dataframes, start_year=2022, pm_col="PM25"):
    """Print stations with >0% missing PM2.5, sorted descending, for each year."""
    print("--- Missing PM2.5 Percentage by Station (> 0%) ---")

    for year, df in enumerate(all_raw_dataframes, start=start_year):
        print(f"\nYEAR: {year}\n")

        if df.empty:
            print("Empty dataframe — skip.")
            print("\n" + "-" * 50)
            continue

        if pm_col not in df.columns:
            print(f"Column '{pm_col}' not found — skip.")
            print("\n" + "-" * 50)
            continue

        missing_stats = df.groupby("Station")[pm_col].apply(lambda x: x.isna().mean() * 100)
        missing_stats = missing_stats[missing_stats > 0].sort_values(ascending=False)

        if not missing_stats.empty:
            print(missing_stats.apply(lambda x: f"{x:.2f}%"))
        else:
            print("No stations have missing PM2.5 values (NaNs) in the recorded rows.")

        print("\n" + "-" * 50)


def remove_stations_by_missing_threshold(
    all_raw_dataframes,
    start_year=2022,
    pm_col="PM25",
    threshold=0.20,
):
    """
    Remove stations whose PM2.5 missing rate > threshold in ANY year.
    Returns:
      - updated_dfs: list[pd.DataFrame]
      - stations_to_remove: set[str]
    """
    print(f"--- Removing Stations with > {threshold*100:.0f}% Missing Data (PM2.5) ---")

    stations_to_remove = set()

    # 1) Identify stations to remove (union over years)
    for year, df in enumerate(all_raw_dataframes, start=start_year):
        if df.empty or pm_col not in df.columns:
            continue

        missing_rate = df.groupby("Station")[pm_col].apply(lambda x: x.isna().mean())
        bad_stations = missing_rate[missing_rate > threshold].index.tolist()

        if bad_stations:
            print(f"Year {year}: Found {len(bad_stations)} stations with > {threshold*100:.0f}% missing.")
            stations_to_remove.update(bad_stations)

    # 2) Print summary
    print(f"\nTotal Unique Stations to Remove: {len(stations_to_remove)}")
    if stations_to_remove:
        print(f"Stations: {sorted(stations_to_remove)}")

    # 3) Remove them from all years
    updated_dfs = []
    if stations_to_remove:
        for year, df in enumerate(all_raw_dataframes, start=start_year):
            before = len(df)
            df_new = df[~df["Station"].isin(stations_to_remove)].copy()
            after = len(df_new)
            print(f"Year {year}: Dropped {before - after} rows (removed {len(stations_to_remove)} stations).")
            updated_dfs.append(df_new)
    else:
        updated_dfs = all_raw_dataframes

    # 4) Final check
    if updated_dfs:
        remaining = updated_dfs[0]["Station"].nunique(dropna=True) if not updated_dfs[0].empty else 0
        print(f"\nFinal Common Station Count (based on first year df): {remaining}")

    return updated_dfs, stations_to_remove

def haversine_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Great-circle distance (Haversine) for multiple points.

    Args:
        coords: array of shape (N, 2) with [[lat, lon], ...] in decimal degrees.

    Returns:
        (N, N) distance matrix in kilometers.
    """
    R = 6371.0  # Earth radius in km

    rads = np.radians(coords)
    lat = rads[:, 0][:, np.newaxis]
    lon = rads[:, 1][:, np.newaxis]

    dlat = lat - lat.T
    dlon = lon - lon.T

    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def rigorous_imputation_pipeline(df: pd.DataFrame, year_label: str, pm_col: str = "PM25") -> pd.DataFrame:
    """
    3-stage imputation pipeline:
      1) Temporal forward fill (limit=1 hour) for micro-gaps
      2) Spatial IDW (Haversine distance) for macro-gaps
      3) Linear interpolation for final cleanup

    Requirements in df:
      - 'Datetime', 'Station', pm_col
      - 'Latitude', 'Longitude' for spatial step

    Returns:
      A new dataframe with pm_col imputed.
    """
    df = df.copy()

    required_cols = {"Datetime", "Station", pm_col}
    missing_req = required_cols - set(df.columns)
    if missing_req:
        raise ValueError(f"[{year_label}] Missing required columns: {sorted(missing_req)}")

    # Ensure Datetime is parsed and sort for grouped operations
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["Station"] = df["Station"].astype(str)
    df = df.sort_values(["Station", "Datetime"])

    initial_missing = df[pm_col].isna().sum()
    if initial_missing == 0:
        print(f"[{year_label}] Data is already complete.")
        return df

    # --- Step 1: Temporal forward fill (limit 1 hour) ---
    df[pm_col] = df.groupby("Station")[pm_col].ffill(limit=1)
    missing_s1 = df[pm_col].isna().sum()

    # --- Step 2: Spatial IDW (Haversine Distance) ---
    can_do_spatial = missing_s1 > 0 and {"Latitude", "Longitude"}.issubset(df.columns)

    if can_do_spatial:
        pivot = df.pivot_table(index="Datetime", columns="Station", values=pm_col)

        # Station coordinates aligned to pivot columns
        stations_info = (
            df[["Station", "Latitude", "Longitude"]]
            .drop_duplicates("Station")
            .set_index("Station")
            .loc[pivot.columns]
        )
        coords = stations_info[["Latitude", "Longitude"]].to_numpy()

        # Distances + weights
        dists = haversine_matrix(coords)
        np.fill_diagonal(dists, np.inf)  # ignore self-distance
        weights = 1.0 / (dists + 1e-6) ** 2  # inverse distance squared

        values = pivot.to_numpy()
        mask_present = (~np.isnan(values)).astype(float)
        values_zeroed = np.nan_to_num(values)

        with np.errstate(divide="ignore", invalid="ignore"):
            numerator = values_zeroed @ weights.T
            denominator = mask_present @ weights.T
            imputed = numerator / denominator

        pivot_filled = pivot.copy()
        mask_missing = pivot.isna()
        imputed_df = pd.DataFrame(imputed, index=pivot.index, columns=pivot.columns)
        pivot_filled[mask_missing] = imputed_df[mask_missing]

        # Back to long and merge
        df_filled = (
            pivot_filled.reset_index()
            .melt(id_vars="Datetime", var_name="Station", value_name=f"{pm_col}_Filled")
        )
        df = df.merge(df_filled, on=["Datetime", "Station"], how="left")
        df[pm_col] = df[pm_col].fillna(df[f"{pm_col}_Filled"])
        df = df.drop(columns=[f"{pm_col}_Filled"])

    missing_s2 = df[pm_col].isna().sum()

    # --- Step 3: Linear interpolation (final cleanup) ---
    if missing_s2 > 0:
        df[pm_col] = df.groupby("Station")[pm_col].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )

    missing_final = df[pm_col].isna().sum()

    # Summary
    print(f"[{year_label}] Imputation Stats:")
    print(f"  - Initial Missing: {initial_missing}")
    print(f"  - Recovered by S1 (Time):  {initial_missing - missing_s1}")
    print(f"  - Recovered by S2 (Space): {missing_s1 - missing_s2}")
    print(f"  - Recovered by S3 (Clean): {missing_s2 - missing_final}")
    print(f"  - Final Missing: {missing_final}")

    if missing_s1 > 0 and not {"Latitude", "Longitude"}.issubset(df.columns):
        print(f"[{year_label}] Note: Spatial IDW skipped (Latitude/Longitude not available).")

    return df


def impute_all_years(all_raw_dataframes, start_year=2022, pm_col="PM25"):
    """
    Apply the imputation pipeline to each yearly dataframe in the list.
    Returns a new list of imputed dataframes (same order).
    """
    out = []
    for year, df in enumerate(all_raw_dataframes, start=start_year):
        out.append(rigorous_imputation_pipeline(df, year_label=str(year), pm_col=pm_col))
    return out

def filter_common_stations(all_raw_dataframes, start_year=2022):
    """
    Find stations present in ALL years (intersection) and filter each dataframe.
    Returns:
      - filtered_dfs: list[pd.DataFrame]
      - common_stations: set[str]
    """
    print("--- Filtering for Common Stations ---")

    if not all_raw_dataframes:
        return [], set()

    common_stations = set(all_raw_dataframes[0]["Station"].unique())
    for df in all_raw_dataframes[1:]:
        common_stations &= set(df["Station"].unique())

    print(f"Identified {len(common_stations)} common stations across {len(all_raw_dataframes)} years.")

    filtered_dfs = []
    for year, df in enumerate(all_raw_dataframes, start=start_year):
        df_target = df[df["Station"].isin(common_stations)].copy()
        filtered_dfs.append(df_target)

    return filtered_dfs, common_stations


def run_imputation_on_years(all_raw_dataframes, start_year=2022, pm_col="PM25"):
    """
    Runs:
      1) common-station filtering
      2) rigorous imputation for each year
    Returns:
      - cleaned_dfs: list[pd.DataFrame]
      - common_stations: set[str]
    """
    filtered_dfs, common_stations = filter_common_stations(all_raw_dataframes, start_year=start_year)

    print("\n--- Running Rigorous Imputation ---")
    cleaned_dfs = []
    for year, df in enumerate(filtered_dfs, start=start_year):
        df_clean = rigorous_imputation_pipeline(df, year_label=str(year), pm_col=pm_col)
        cleaned_dfs.append(df_clean)

    print("\nProcessing complete.")
    return cleaned_dfs, common_stations

def add_time_features(df: pd.DataFrame, time_col: str = "Datetime") -> pd.DataFrame:
    """Add time-based features used for EDA plots."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(by=time_col)

    df["hour"] = df[time_col].dt.hour
    df["day_of_week"] = df[time_col].dt.day_name()
    df["month"] = df[time_col].dt.month
    df["year"] = df[time_col].dt.year
    df["day_type"] = df[time_col].dt.dayofweek.apply(lambda x: "Weekend" if x >= 5 else "Weekday")

    return df


def plot_micro_analysis(
    full_imputed_data: pd.DataFrame,
    plot_out_dir,
    time_col: str = "Datetime",
    val_col: str = "PM25",
    filename: str = "eda_micro_analysis_daily_weekly_patterns.png",
    show: bool = True,
):
    """
    ACT 1: Micro-Analysis (Daily & Weekly Patterns)
    Saves the plot to Assets/Outputs/ (plot_out_dir).
    """
    plot_out_dir = Path(plot_out_dir)
    plot_out_dir.mkdir(parents=True, exist_ok=True)

    df = add_time_features(full_imputed_data, time_col=time_col)

    # Custom weekday order so plots are Mon->Sun
    week_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    fig1, axes1 = plt.subplots(3, 1, figsize=(14, 18), constrained_layout=True)

    # --- Chart 1: Daily Cycle with Rush Hours ---
    hourly_avg = df.groupby("hour")[val_col].mean().reset_index()

    sns.lineplot(
        data=hourly_avg, x="hour", y=val_col, ax=axes1[0],
        marker="o", linewidth=3
    )
    axes1[0].axvspan(7, 9, alpha=0.15, label="Rush Hour")
    axes1[0].axvspan(16, 19, alpha=0.15)
    axes1[0].set_title("1. Daily Rhythm: Highlighting Rush Hours", fontsize=16, fontweight="bold", loc="left")
    axes1[0].set_xticks(range(0, 24))
    axes1[0].legend(["Avg Value", "Rush Hours"], loc="upper left")

    # --- Chart 2: Weekday vs. Weekend Comparison ---
    split_avg = df.groupby(["hour", "day_type"])[val_col].mean().reset_index()

    sns.lineplot(
        data=split_avg, x="hour", y=val_col, hue="day_type", ax=axes1[1],
        marker="o", linewidth=3
    )
    axes1[1].set_title("2. Lifestyle Impact: Weekday vs. Weekend", fontsize=16, fontweight="bold", loc="left")
    axes1[1].set_xticks(range(0, 24))

    # --- Chart 3: Weekly Accumulation ---
    weekly_avg = df.groupby("day_of_week")[val_col].mean().reindex(week_order).reset_index()

    sns.lineplot(
        data=weekly_avg, x="day_of_week", y=val_col, ax=axes1[2],
        marker="s", linewidth=3
    )
    axes1[2].set_title("3. Weekly Accumulation Trend", fontsize=16, fontweight="bold", loc="left")
    axes1[2].grid(True, axis="x")

    outpath = plot_out_dir / filename
    plt.savefig(outpath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig1)

    print(f"✅ Saved EDA plot to: {outpath}")

def plot_monthly_yearly_and_seasons(
    full_imputed_data: pd.DataFrame,
    plot_out_dir,
    time_col: str = "Datetime",
    val_col: str = "PM25",
    filename_monthly: str = "eda_act2_monthly_yearly_patterns.png",
    filename_seasons: str = "eda_act2_long_term_timeline_seasons.png",
    show: bool = True,
):
    """
    ACT 2:
      Part 1) Monthly & Yearly patterns (2-panel)
      Part 2) Long-term timeline with seasonal background shading

    Saves plots to Assets/Outputs/ (plot_out_dir).
    """
    plot_out_dir = Path(plot_out_dir)
    plot_out_dir.mkdir(parents=True, exist_ok=True)

    # Use the same feature engineering step
    df = add_time_features(full_imputed_data, time_col=time_col)

    # ---------------------------------------------------------
    # ACT 2 Part 1: Monthly & Yearly Patterns
    # ---------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 12), constrained_layout=True)

    # Chart 4: Overall Monthly Trend (Aggregated)
    monthly_avg = df.groupby("month")[val_col].mean().reset_index()
    sns.lineplot(data=monthly_avg, x="month", y=val_col, ax=axes2[0], marker="D", linewidth=3)
    axes2[0].set_title("4. Overall Monthly Trend (Aggregated)", fontsize=16, fontweight="bold", loc="left")
    axes2[0].set_xticks(range(1, 13))

    # Chart 5: Year-over-Year Volatility
    monthly_yearly_avg = df.groupby(["year", "month"])[val_col].mean().reset_index()
    sns.lineplot(data=monthly_yearly_avg, x="month", y=val_col, hue="year", ax=axes2[1], marker="o", linewidth=2.5)
    axes2[1].set_title("5. Year-over-Year Volatility", fontsize=16, fontweight="bold", loc="left")
    axes2[1].set_xticks(range(1, 13))
    axes2[1].legend(title="Year", loc="upper right")

    outpath_monthly = plot_out_dir / filename_monthly
    fig2.savefig(outpath_monthly, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig2)

    print(f"✅ Saved EDA plot to: {outpath_monthly}")

    # ---------------------------------------------------------
    # ACT 2 Part 2: Long-term Timeline with Seasons
    # ---------------------------------------------------------
    # Month-end frequency: use 'M' for month end (widely supported).
    # Your original 'ME' may fail depending on pandas version.
    try:
        grouper = pd.Grouper(key=time_col, freq="ME")  # newer pandas
    except ValueError:
        grouper = pd.Grouper(key=time_col, freq="M")  # older pandas

    long_term_trend = df.groupby(grouper)[val_col].mean().reset_index()

    fig3, ax = plt.subplots(figsize=(14, 6))

    sns.lineplot(
        data=long_term_trend, x=time_col, y=val_col,
        ax=ax, marker="o", linewidth=2.5, zorder=10
    )

    ax.set_title(f"6. Long-term Timeline with Seasons ({val_col})", fontsize=16, fontweight="bold", loc="left")
    ax.set_xlabel("Date (Season Starts: Mar/Jun/Sep/Dec)")
    ax.set_ylabel(f"Avg {val_col}")

    # Seasonal background shading
    season_colors = {
        "Spring": "#55efc4",
        "Summer": "#ff7675",
        "Autumn": "#fdcb6e",
        "Winter": "#74b9ff",
    }

    start_year = int(df[time_col].dt.year.min())
    end_year = int(df[time_col].dt.year.max())

    for year in range(start_year, end_year + 1):
        # Spring (Mar-May)
        ax.axvspan(pd.Timestamp(f"{year}-03-01"), pd.Timestamp(f"{year}-06-01"),
                   color=season_colors["Spring"], alpha=0.2, lw=0)
        # Summer (Jun-Aug)
        ax.axvspan(pd.Timestamp(f"{year}-06-01"), pd.Timestamp(f"{year}-09-01"),
                   color=season_colors["Summer"], alpha=0.2, lw=0)
        # Autumn (Sep-Nov)
        ax.axvspan(pd.Timestamp(f"{year}-09-01"), pd.Timestamp(f"{year}-12-01"),
                   color=season_colors["Autumn"], alpha=0.2, lw=0)
        # Winter (Dec-Feb) split around year boundary
        ax.axvspan(pd.Timestamp(f"{year}-12-01"), pd.Timestamp(f"{year+1}-01-01"),
                   color=season_colors["Winter"], alpha=0.2, lw=0)
        ax.axvspan(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-03-01"),
                   color=season_colors["Winter"], alpha=0.2, lw=0)

    legend_seasons = [
        Patch(facecolor=season_colors["Spring"], alpha=0.2, label="Spring (Starts Mar)"),
        Patch(facecolor=season_colors["Summer"], alpha=0.2, label="Summer (Starts Jun)"),
        Patch(facecolor=season_colors["Autumn"], alpha=0.2, label="Autumn (Starts Sep)"),
        Patch(facecolor=season_colors["Winter"], alpha=0.2, label="Winter (Starts Dec)"),
    ]
    ax.legend(handles=legend_seasons, loc="upper left", ncol=4, frameon=True)

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)

    plt.tight_layout()

    outpath_seasons = plot_out_dir / filename_seasons
    fig3.savefig(outpath_seasons, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig3)

    print(f"✅ Saved EDA plot to: {outpath_seasons}")

def plot_station_corr_heatmap_before_grouping(
    full_imputed_data: pd.DataFrame,
    plot_out_dir,
    time_col: str = "Datetime",
    station_col: str = "Station",
    val_col: str = "PM25",
    filename: str = "eda_station_corr_heatmap_before_grouping.png",
    show: bool = True,
):
    """
    Create a station-by-station correlation heatmap (before any grouping).
    Saves to Assets/Outputs/ (plot_out_dir).
    """
    plot_out_dir = Path(plot_out_dir)
    plot_out_dir.mkdir(parents=True, exist_ok=True)

    df = full_imputed_data.copy()

    # Basic type safety
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, station_col])

    # Pivot to wide: index=Datetime, columns=Station, values=PM25
    pivot_df = df.pivot_table(index=time_col, columns=station_col, values=val_col)

    # Correlation across stations
    corr_before = pivot_df.corr()

    # Mask upper triangle (keep lower triangle)
    mask_upper = np.triu(np.ones_like(corr_before, dtype=bool))

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr_before,
        mask=mask_upper,
        cmap="RdBu_r",
        vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Station Correlation Heatmap (Before Grouping)", fontsize=16, fontweight="bold")
    plt.tight_layout()

    outpath = plot_out_dir / filename
    plt.savefig(outpath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    print(f"✅ Saved correlation heatmap to: {outpath}")

def map_stations_to_air_zones_and_plot(
    full_imputed_data: pd.DataFrame,
    raw_dir: Path,
    plot_out_dir: Path,
    geojson_name: str = "bcairzones.geojson",
    filename: str = "bc_air_zones_station_network_corrected.png",
    show: bool = True,
):
    """
    Map each station to an air zone using a local GeoJSON (no downloads),
    apply manual corrections, merge zone labels back to full_imputed_data,
    and save a network map plot to Assets/Outputs.

    Returns:
      - full_imputed_data_with_air_zones (DataFrame)
      - mapped_stations (DataFrame with Station + Official_Air_Zone)
    """
    if full_imputed_data is None or full_imputed_data.empty:
        raise ValueError("full_imputed_data is empty. Please generate it first.")

    raw_dir = Path(raw_dir)
    plot_out_dir = Path(plot_out_dir)
    plot_out_dir.mkdir(parents=True, exist_ok=True)

    station_col = "Station"
    lat_col = "Latitude"
    lon_col = "Longitude"
    time_col = "Datetime"
    zone_output_col = "Official_Air_Zone"

    # --------------------------
    # 1) Station centroids
    # --------------------------
    unique_stations = (
        full_imputed_data.groupby(station_col, as_index=False)[[lat_col, lon_col]]
        .mean()
    )

    gdf_stations = gpd.GeoDataFrame(
        unique_stations,
        geometry=gpd.points_from_xy(unique_stations[lon_col], unique_stations[lat_col]),
        crs="EPSG:4326",
    ).to_crs(epsg=3005)

    # --------------------------
    # 2) Load local air zone geojson
    # --------------------------
    geojson_path = raw_dir / geojson_name
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")

    gdf_zones = gpd.read_file(geojson_path).to_crs(epsg=3005)

    zone_col_candidates = [
        c for c in gdf_zones.columns
        if ("zone" in c.lower()) or ("name" in c.lower())
    ]
    if not zone_col_candidates:
        raise ValueError(f"Cannot find zone/name column in: {list(gdf_zones.columns)}")

    zone_col = zone_col_candidates[0]
    gdf_zones["ZoneName"] = gdf_zones[zone_col]

    # --------------------------
    # 3) Spatial join (within)
    # --------------------------
    join_exact = gpd.sjoin(gdf_stations, gdf_zones, how="left", predicate="within")
    mapped_stations = join_exact[[station_col, "ZoneName"]].rename(
        columns={"ZoneName": zone_output_col}
    )

    # --------------------------
    # 4) Manual corrections
    # --------------------------
    orphan_correction_map = {
        # Georgia Strait (East Vancouver Island / Sunshine Coast)
        "Crofton Substation": "Georgia Strait",
        "Elk Falls Dogwood": "Georgia Strait",
        "Harmac Cedar Woobank": "Georgia Strait",
        "Langdale Elementary": "Georgia Strait",
        "Gibsons Municipal Hall": "Georgia Strait",
        "Nanaimo Labieux Road": "Georgia Strait",
        "Nanaimo Departure Bay": "Georgia Strait",
        "Duncan Cairnsmore": "Georgia Strait",

        # Coastal
        "Kitimat Haisla Village": "Coastal",
        "Kitimat Haul Road": "Coastal",
        "Kitimat Riverlodge": "Coastal",
        "Kitimat Whitesail": "Coastal",
        "Prince Rupert Fairview": "Coastal",
        "Port Alberni Elementary": "Coastal",

        # Keep exactly as your code (even though your comment said Coastal)
        "Squamish Elementary": "Georgia Strait",
        "Whistler Meadow Park": "Georgia Strait",
    }

    def apply_fixes(row):
        st = row[station_col]
        if st in orphan_correction_map:
            return orphan_correction_map[st]
        if pd.notna(row[zone_output_col]):
            return row[zone_output_col]
        return "Unknown"

    mapped_stations[zone_output_col] = mapped_stations.apply(apply_fixes, axis=1)

    full_imputed_data_with_air_zones = pd.merge(
        full_imputed_data,
        mapped_stations.drop_duplicates(subset=[station_col]),
        on=station_col,
        how="left",
    )

    # --------------------------
    # 5) Plot & export
    # --------------------------
    plot_data = (
        full_imputed_data_with_air_zones.groupby(station_col, as_index=False)
        .agg({lat_col: "mean", lon_col: "mean", zone_output_col: "first"})
    )

    gdf_plot = gpd.GeoDataFrame(
        plot_data,
        geometry=gpd.points_from_xy(plot_data[lon_col], plot_data[lat_col]),
        crs="EPSG:4326",
    ).to_crs(epsg=3005)

    bg_palette = {
        "Northeast": "#FFF3E0",
        "Central Interior": "#FFFFF0",
        "Southern Interior": "#F1F8E9",
        "Lower Fraser Valley": "#FDE0DC",
        "Georgia Strait": "#E1F5FE",
        "Coastal": "#ECEFF1",
        "Northwest": "#F3E5F5",
    }
    point_palette = {
        "Northeast": "#FF8C00",
        "Central Interior": "#FFD700",
        "Southern Interior": "#33A02C",
        "Lower Fraser Valley": "#E31A1C",
        "Georgia Strait": "#1F78B4",
        "Coastal": "#084594",
        "Northwest": "#6A3D9A",
    }

    fig, ax = plt.subplots(figsize=(20, 18))

    # A) Zones background
    for zone_name in gdf_zones["ZoneName"].unique():
        zone_geom = gdf_zones[gdf_zones["ZoneName"] == zone_name]
        zone_geom.plot(
            ax=ax,
            color=bg_palette.get(zone_name, "#eeeeee"),
            edgecolor="white",
            linewidth=1.5,
        )

    # B) Zone labels
    for _, row in gdf_zones.iterrows():
        pt = row.geometry.representative_point()
        ax.annotate(
            text=str(row["ZoneName"]).replace(" ", "\n"),
            xy=(pt.x, pt.y),
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#555555",
            path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.7)],
            zorder=5,
        )

    # C) Points
    gdf_plot["color"] = gdf_plot[zone_output_col].map(point_palette)
    gdf_plot.plot(
        ax=ax,
        color=gdf_plot["color"],
        marker="o",
        markersize=80,
        edgecolor="black",
        linewidth=1.2,
        zorder=10,
    )

    ax.set_title(
        "British Columbia Air Quality Monitoring Network (Corrected)",
        fontsize=24,
        fontweight="bold",
        pad=20,
        color="#333333",
    )
    ax.set_axis_off()

    # Legend (using Patch from your imports)
    legend_labels = sorted(plot_data[zone_output_col].dropna().unique())
    legend_handles = [Patch(color=point_palette[z], label=z) for z in legend_labels if z in point_palette]
    ax.legend(
        handles=legend_handles,
        title="Air Zone",
        title_fontsize=14,
        fontsize=12,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.05),
        frameon=True,
    )

    outpath = plot_out_dir / filename
    plt.savefig(outpath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"✅ Saved BC air zone network plot to: {outpath}")

    # Quick verification
    check_list = [
        "Squamish Elementary",
        "Whistler Meadow Park",
        "Crofton Substation",
        "Langdale Elementary",
        "Kitimat Haul Road",
    ]
    print("=== Corrected station-to-zone assignments (sample) ===")
    print(mapped_stations[mapped_stations[station_col].isin(check_list)].drop_duplicates())

    return full_imputed_data_with_air_zones, mapped_stations

def plot_zone_micro_analysis(
    full_imputed_data_with_air_zones: pd.DataFrame,
    plot_out_dir: Path,
    time_col: str = "Datetime",
    val_col: str = "PM25",
    zone_col: str = "Official_Air_Zone",
    show: bool = True,
    prefix: str = "eda_zone_act1",
):
    """
    ACT 1 (by Air Zone):
      1) Daily rhythm comparison across zones
      2) Weekday vs weekend faceted by zone
      3) Weekly accumulation comparison across zones

    Saves plots into Assets/Outputs/ (plot_out_dir).
    """
    plot_out_dir = Path(plot_out_dir)
    plot_out_dir.mkdir(parents=True, exist_ok=True)

    df = full_imputed_data_with_air_zones.copy()

    # Ensure datetime and sort
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, zone_col]).sort_values(by=time_col)

    # Time features
    df["hour"] = df[time_col].dt.hour
    df["day_of_week"] = df[time_col].dt.day_name()
    df["month"] = df[time_col].dt.month
    df["year"] = df[time_col].dt.year
    df["day_type"] = df[time_col].dt.dayofweek.apply(lambda x: "Weekend" if x >= 5 else "Weekday")

    week_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Color palette for zones (consistent across plots)
    zones = sorted(df[zone_col].dropna().unique())
    distinct_palette = dict(zip(zones, sns.color_palette("bright", n_colors=len(zones))))

    # ---------------------------------------------------------
    # Chart 1: Daily Rhythm Comparison
    # ---------------------------------------------------------
    daily_zone = df.groupby(["hour", zone_col])[val_col].mean().reset_index()

    fig1 = plt.figure(figsize=(14, 6))
    sns.lineplot(
        data=daily_zone, x="hour", y=val_col, hue=zone_col,
        marker="o", linewidth=2.5, palette=distinct_palette
    )
    plt.title("1. Daily Rhythm: Regional Comparison", fontsize=16, fontweight="bold", loc="left")
    plt.xlabel("Hour of Day (0-23)")
    plt.ylabel(f"Avg {val_col}")
    plt.xticks(range(0, 24))
    plt.legend(title="Air Zone", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out1 = plot_out_dir / f"{prefix}_1_daily_rhythm.png"
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig1)

    print(f"✅ Saved plot: {out1}")

    # ---------------------------------------------------------
    # Chart 2: Weekday vs Weekend (Faceted)
    # ---------------------------------------------------------
    split_zone = df.groupby(["hour", "day_type", zone_col])[val_col].mean().reset_index()

    g = sns.relplot(
        data=split_zone,
        x="hour", y=val_col,
        hue="day_type", col=zone_col,
        kind="line", marker="o", linewidth=2.5,
        palette={"Weekday": "#6c5ce7", "Weekend": "#ff7675"},
        col_wrap=3, height=4, aspect=1.2
    )
    g.fig.suptitle("2. Weekday vs Weekend Comparison by Air Zone", fontsize=16, fontweight="bold", y=1.05)

    out2 = plot_out_dir / f"{prefix}_2_weekday_weekend_facets.png"
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(g.fig)

    print(f"✅ Saved plot: {out2}")

    # ---------------------------------------------------------
    # Chart 3: Weekly Accumulation Comparison
    # ---------------------------------------------------------
    weekly_zone = df.groupby(["day_of_week", zone_col])[val_col].mean().reset_index()
    weekly_zone["day_of_week"] = pd.Categorical(
        weekly_zone["day_of_week"], categories=week_order, ordered=True
    )
    weekly_zone = weekly_zone.sort_values("day_of_week")

    fig3 = plt.figure(figsize=(14, 6))
    sns.lineplot(
        data=weekly_zone, x="day_of_week", y=val_col, hue=zone_col,
        marker="s", linewidth=2.5, palette=distinct_palette
    )
    plt.title("3. Weekly Accumulation: Regional Comparison", fontsize=16, fontweight="bold", loc="left")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out3 = plot_out_dir / f"{prefix}_3_weekly_accumulation.png"
    plt.savefig(out3, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig3)

    print(f"✅ Saved plot: {out3}")

def plot_zone_macro_analysis(
    full_imputed_data_with_air_zones: pd.DataFrame,
    plot_out_dir: Path,
    time_col: str = "Datetime",
    val_col: str = "PM25",
    zone_col: str = "Official_Air_Zone",
    show: bool = True,
    prefix: str = "eda_zone_act2",
):
    """
    ACT 2 (by Air Zone):
      4) Monthly seasonality comparison across zones
      5) Year-over-year volatility faceted by zone (hue=year)

    Saves plots into Assets/Outputs/ (plot_out_dir).
    """
    plot_out_dir = Path(plot_out_dir)
    plot_out_dir.mkdir(parents=True, exist_ok=True)

    df = full_imputed_data_with_air_zones.copy()

    # Ensure datetime and sort
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, zone_col]).sort_values(by=time_col)

    # Time features (needed for grouping)
    df["month"] = df[time_col].dt.month
    df["year"] = df[time_col].dt.year

    # Palette for zones (consistent)
    zones = sorted(df[zone_col].dropna().unique())
    distinct_palette = dict(zip(zones, sns.color_palette("bright", n_colors=len(zones))))

    # ---------------------------------------------------------
    # Chart 4: Monthly Seasonality (by zone)
    # ---------------------------------------------------------
    monthly_zone = df.groupby(["month", zone_col])[val_col].mean().reset_index()

    fig4 = plt.figure(figsize=(14, 6))
    sns.lineplot(
        data=monthly_zone, x="month", y=val_col, hue=zone_col,
        marker="D", linewidth=2.5, palette=distinct_palette
    )

    plt.title("4. Monthly Seasonality: Regional Comparison", fontsize=16, fontweight="bold", loc="left")
    plt.xticks(range(1, 13))
    plt.xlabel("Month (1-12)")
    plt.ylabel(f"Avg {val_col}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out4 = plot_out_dir / f"{prefix}_4_monthly_seasonality.png"
    plt.savefig(out4, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig4)

    print(f"✅ Saved plot: {out4}")

    # ---------------------------------------------------------
    # Chart 5: Year-over-Year Volatility (faceted by zone)
    # ---------------------------------------------------------
    monthly_yearly_zone = df.groupby(["year", "month", zone_col])[val_col].mean().reset_index()

    g = sns.relplot(
        data=monthly_yearly_zone,
        x="month", y=val_col,
        hue="year", col=zone_col,
        kind="line", marker="o", linewidth=2,
        palette="tab10",
        col_wrap=3, height=4, aspect=1.2
    )
    g.fig.suptitle("5. Annual Volatility Trends per Air Zone", fontsize=16, fontweight="bold", y=1.05)
    g.set(xticks=range(1, 13))

    out5 = plot_out_dir / f"{prefix}_5_yoy_volatility_facets.png"
    plt.savefig(out5, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(g.fig)

    print(f"✅ Saved plot: {out5}")

def plot_station_corr_heatmap_grouped_by_air_zone(
    full_imputed_data_with_air_zones: pd.DataFrame,
    plot_out_dir: Path,
    time_col: str = "Datetime",
    station_col: str = "Station",
    zone_col: str = "Official_Air_Zone",
    val_col: str = "PM25",
    filename: str = "eda_station_corr_heatmap_grouped_by_air_zone.png",
    show: bool = True,
):
    """
    Create a station correlation heatmap, with stations sorted by Official_Air_Zone,
    including zone separators and colored strips on the axes.

    Saves the plot to Assets/Outputs/ (plot_out_dir).
    """
    plot_out_dir = Path(plot_out_dir)
    plot_out_dir.mkdir(parents=True, exist_ok=True)

    df = full_imputed_data_with_air_zones.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, station_col])

    # Wide format: columns = stations
    pivot_df = df.pivot_table(index=time_col, columns=station_col, values=val_col)

    # Correlation across stations
    corr = pivot_df.corr()

    # Build station -> zone mapping and sort: first by zone, then by station name
    zone_info = (
        df[[station_col, zone_col]]
        .drop_duplicates()
        .sort_values(by=[zone_col, station_col])
    )

    # Stations in desired display order (only keep those present in corr)
    sorted_stations = [s for s in zone_info[station_col].tolist() if s in corr.index]

    # Reorder correlation matrix
    corr_after = corr.loc[sorted_stations, sorted_stations]

    # Upper-triangle mask
    mask_after = np.triu(np.ones_like(corr_after, dtype=bool))

    # Zone colors (distinct)
    zones = zone_info[zone_col].dropna().unique()
    zone_colors = dict(zip(zones, sns.color_palette("tab10", len(zones))))

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        corr_after,
        mask=mask_after,
        cmap="RdBu_r",
        vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
        ax=ax,
    )

    # Find change points where air zone changes in the sorted list
    zone_values = zone_info.set_index(station_col).loc[sorted_stations][zone_col].values
    change_points = np.where(zone_values[:-1] != zone_values[1:])[0] + 1

    # Draw separator lines
    for cp in change_points:
        ax.hlines(cp, *ax.get_xlim(), color="black", linewidth=2, linestyle="--")
        ax.vlines(cp, *ax.get_ylim(), color="black", linewidth=2, linestyle="--")

    # Add colored strips to indicate zone membership for each station
    # (left strip for y-axis, bottom strip for x-axis)
    for i, st in enumerate(sorted_stations):
        z = zone_info.loc[zone_info[station_col] == st, zone_col].iloc[0]
        c = zone_colors.get(z, (0.8, 0.8, 0.8))

        rect_y = plt.Rectangle(
            (-1.5, i), 1.5, 1,
            facecolor=c, clip_on=False, transform=ax.transData
        )
        ax.add_patch(rect_y)

        rect_x = plt.Rectangle(
            (i, len(sorted_stations)), 1, 1.5,
            facecolor=c, clip_on=False, transform=ax.transData
        )
        ax.add_patch(rect_x)

    # Legend (use Patch from your imports)
    legend_elements = [Patch(facecolor=c, label=z) for z, c in zone_colors.items()]
    ax.legend(
        handles=legend_elements,
        title="Official Air Zones",
        bbox_to_anchor=(1.02, -0.2),
        loc="lower left",
        frameon=True,
    )

    plt.title(
        "Station Correlation Heatmap: Grouped by Official Air Zone",
        fontsize=18, fontweight="bold", pad=20
    )

    plt.tight_layout()

    outpath = plot_out_dir / filename
    plt.savefig(outpath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"✅ Saved grouped correlation heatmap to: {outpath}")

def plot_bc_network_map_assigned_zones(
    full_imputed_data_with_air_zones: pd.DataFrame,
    raw_dir: Path,
    plot_out_dir: Path,
    geojson_name: str = "bcairzones.geojson",
    filename: str = "bc_air_quality_network_assigned_zones.png",
    show: bool = True,
):
    """
    Plot BC air zones + zone labels + station points colored by Official_Air_Zone
    (uses existing assignments already in full_imputed_data_with_air_zones).

    Saves to Assets/Outputs/ (plot_out_dir).
    """
    if full_imputed_data_with_air_zones is None or full_imputed_data_with_air_zones.empty:
        raise ValueError("full_imputed_data_with_air_zones is empty. Run the mapping step first.")

    raw_dir = Path(raw_dir)
    plot_out_dir = Path(plot_out_dir)
    plot_out_dir.mkdir(parents=True, exist_ok=True)

    station_col = "Station"
    lat_col = "Latitude"
    lon_col = "Longitude"
    zone_col = "Official_Air_Zone"

    # --------------------------
    # 1) Prepare station centroids for plotting
    # --------------------------
    plot_data = full_imputed_data_with_air_zones.groupby(station_col, as_index=False).agg(
        {lat_col: "mean", lon_col: "mean", zone_col: "first"}
    )

    gdf_stations = gpd.GeoDataFrame(
        plot_data,
        geometry=gpd.points_from_xy(plot_data[lon_col], plot_data[lat_col]),
        crs="EPSG:4326",
    ).to_crs(epsg=3005)

    # --------------------------
    # 2) Load local zone polygons
    # --------------------------
    geojson_path = raw_dir / geojson_name
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")

    gdf_zones = gpd.read_file(geojson_path).to_crs(epsg=3005)
    zone_col_candidates = [c for c in gdf_zones.columns if ("zone" in c.lower()) or ("name" in c.lower())]
    if not zone_col_candidates:
        raise ValueError(f"Cannot find zone/name column in: {list(gdf_zones.columns)}")

    zone_name_col = zone_col_candidates[0]
    gdf_zones["ZoneName"] = gdf_zones[zone_name_col]

    # --------------------------
    # 3) Palettes
    # --------------------------
    bg_palette = {
        "Northeast": "#FFF3E0",
        "Central Interior": "#FFFFF0",
        "Southern Interior": "#F1F8E9",
        "Lower Fraser Valley": "#FDE0DC",
        "Georgia Strait": "#E1F5FE",
        "Coastal": "#ECEFF1",
        "Northwest": "#F3E5F5",
    }
    point_palette = {
        "Northeast": "#FF8C00",
        "Central Interior": "#FFD700",
        "Southern Interior": "#33A02C",
        "Lower Fraser Valley": "#E31A1C",
        "Georgia Strait": "#1F78B4",
        "Coastal": "#084594",
        "Northwest": "#6A3D9A",
    }

    # --------------------------
    # 4) Plot
    # --------------------------
    fig, ax = plt.subplots(figsize=(20, 18))

    # Layer 1: Polygons
    for zone_name in gdf_zones["ZoneName"].unique():
        zone_geom = gdf_zones[gdf_zones["ZoneName"] == zone_name]
        zone_geom.plot(
            ax=ax,
            color=bg_palette.get(zone_name, "#F5F5F5"),
            edgecolor="white",
            linewidth=1.5,
            alpha=1.0,
        )

    # Layer 2: Zone labels (use representative_point for safer placement)
    for _, row in gdf_zones.iterrows():
        pt = row.geometry.representative_point()
        label_text = str(row["ZoneName"]).replace(" ", "\n")
        ax.annotate(
            text=label_text,
            xy=(pt.x, pt.y),
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#555555",
            path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.7)],
            zorder=5,
        )

    # Layer 3: Station points
    gdf_stations["color"] = gdf_stations[zone_col].map(point_palette).fillna("#777777")
    gdf_stations.plot(
        ax=ax,
        color=gdf_stations["color"],
        marker="o",
        markersize=80,
        edgecolor="black",
        linewidth=1.2,
        zorder=10,
    )

    ax.set_title(
        "British Columbia Air Quality Monitoring Network",
        fontsize=24,
        fontweight="bold",
        pad=20,
        color="#333333",
    )
    ax.set_axis_off()

    # Legend (use Patch from your imports)
    existing_zones = sorted(plot_data[zone_col].dropna().unique())
    legend_handles = [Patch(facecolor=point_palette[z], label=z) for z in existing_zones if z in point_palette]
    ax.legend(
        handles=legend_handles,
        title="Assigned Air Zone",
        title_fontsize=14,
        fontsize=12,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.05),
        frameon=True,
        shadow=True,
    )

    plt.tight_layout()

    outpath = plot_out_dir / filename
    plt.savefig(outpath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"✅ Saved BC network map to: {outpath}")

def build_df_imputed_zone_wide(
    full_imputed_data_with_air_zones: pd.DataFrame,
    data_out_dir: Path,
    time_col: str = "Datetime",
    zone_col: str = "Official_Air_Zone",
    val_col: str = "PM25",
    agg: str = "median",
    filename: str = "PM25_zone_wide_imputed_2022_2025.csv",
) -> pd.DataFrame:
    """
    Create df_imputed (wide format) from long data:
      index = Datetime
      columns = Official_Air_Zone
      values = aggregated PM2.5 (median by default)

    Also saves to Dataset/Outputs/ (data_out_dir).
    """
    data_out_dir = Path(data_out_dir)
    data_out_dir.mkdir(parents=True, exist_ok=True)

    df = full_imputed_data_with_air_zones.copy()

    # Ensure datetime
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, zone_col])

    # Aggregate and pivot to wide
    if agg == "median":
        df_wide = df.groupby([time_col, zone_col])[val_col].median().unstack()
    elif agg == "mean":
        df_wide = df.groupby([time_col, zone_col])[val_col].mean().unstack()
    else:
        raise ValueError("agg must be 'median' or 'mean'")

    df_imputed = df_wide.sort_index()

    # Save (Datetime index -> column in CSV)
    outpath = data_out_dir / filename
    df_imputed.to_csv(outpath, index=True)

    print(f"✅ Built df_imputed (wide) with shape: {df_imputed.shape}")
    print(f"✅ Saved df_imputed to: {outpath}")

    return df_imputed



# Process control and run
if __name__ == "__main__":
    # =========================================================
    # Pipeline settings (no pop-up windows)
    # =========================================================
    SHOW_PLOTS = False
    plt.switch_backend("Agg")  # prevent GUI popups in automated runs

    # =========================
    # 1) Load + basic cleaning
    # =========================
    all_raw_dataframes = load_data()

    inspect_raw_data(all_raw_dataframes, start_year=2022)

    all_raw_dataframes = convert_types(all_raw_dataframes)
    all_raw_dataframes = deduplicate_each_year(all_raw_dataframes, start_year=2022)

    summarize_yearly_coverage(all_raw_dataframes, start_year=2022)

    all_raw_dataframes, stations_removed = remove_stations_by_missing_threshold(
        all_raw_dataframes,
        start_year=2022,
        pm_col="PM25",
        threshold=0.20,
    )

    # =========================================
    # 2) Common stations + rigorous imputation
    # =========================================
    cleaned_dfs, common_stations = run_imputation_on_years(
        all_raw_dataframes,
        start_year=2022,
        pm_col="PM25",
    )

    full_imputed_data = pd.concat(cleaned_dfs, ignore_index=True)

    # =========================
    # 3) EDA plots (overall)
    # =========================
    plot_micro_analysis(
        full_imputed_data,
        plot_out_dir=PLOT_OUT_DIR,
        time_col="Datetime",
        val_col="PM25",
        show=SHOW_PLOTS,
    )

    plot_monthly_yearly_and_seasons(
        full_imputed_data,
        plot_out_dir=PLOT_OUT_DIR,
        time_col="Datetime",
        val_col="PM25",
        show=SHOW_PLOTS,
    )

    plot_station_corr_heatmap_before_grouping(
        full_imputed_data,
        plot_out_dir=PLOT_OUT_DIR,
        time_col="Datetime",
        station_col="Station",
        val_col="PM25",
        show=SHOW_PLOTS,
    )

    # =========================================
    # 4) Air zone mapping + maps + zone EDA
    # =========================================
    full_imputed_data_with_air_zones, mapped_stations = map_stations_to_air_zones_and_plot(
        full_imputed_data=full_imputed_data,
        raw_dir=RAW_DIR,
        plot_out_dir=PLOT_OUT_DIR,
        geojson_name="bcairzones.geojson",
        show=SHOW_PLOTS,
    )

    plot_zone_micro_analysis(
        full_imputed_data_with_air_zones,
        plot_out_dir=PLOT_OUT_DIR,
        time_col="Datetime",
        val_col="PM25",
        zone_col="Official_Air_Zone",
        show=SHOW_PLOTS,
    )

    plot_zone_macro_analysis(
        full_imputed_data_with_air_zones,
        plot_out_dir=PLOT_OUT_DIR,
        time_col="Datetime",
        val_col="PM25",
        zone_col="Official_Air_Zone",
        show=SHOW_PLOTS,
    )

    plot_station_corr_heatmap_grouped_by_air_zone(
        full_imputed_data_with_air_zones,
        plot_out_dir=PLOT_OUT_DIR,
        time_col="Datetime",
        station_col="Station",
        zone_col="Official_Air_Zone",
        val_col="PM25",
        show=SHOW_PLOTS,
    )

    plot_bc_network_map_assigned_zones(
        full_imputed_data_with_air_zones,
        raw_dir=RAW_DIR,
        plot_out_dir=PLOT_OUT_DIR,
        geojson_name="bcairzones.geojson",
        show=SHOW_PLOTS,
    )

    # =========================================
    # 5) Build final curated df_imputed (wide)
    #    and save to Dataset/Outputs
    # =========================================
    df_imputed = build_df_imputed_zone_wide(
        full_imputed_data_with_air_zones,
        data_out_dir=DATA_OUT_DIR,  # Dataset/Outputs
        time_col="Datetime",
        zone_col="Official_Air_Zone",
        val_col="PM25",
        agg="median",
        filename="PM25_zone_wide_imputed_2022_2025.csv",
    )

    # Quick checkpoint print (head 5)
    print("\n=== df_imputed (head 5) ===")
    print(df_imputed.head(5))

    print("\n✅ Full pipeline completed successfully (no pop-up plots).")

    

