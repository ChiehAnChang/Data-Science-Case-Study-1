from pathlib import Path
import re
import numpy as np
import pandas as pd

YEAR_STATION_RE = re.compile(r"^(?P<year>\d{4})_(?P<station>.+)_dataset\.csv$")

DATE_RE = re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{4}\s*$")

def drop_non_date_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop summary/metadata rows where Date is not like M/D/YYYY.
    """
    if "Date" not in df.columns:
        return df
    mask = df["Date"].astype(str).str.match(DATE_RE)
    return df.loc[mask].copy()

def read_hourly_csv_with_weird_header(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, header=1)

    # Drop unit row + summary rows by keeping only real dates
    df = drop_non_date_rows(df)
    if "Date" in df.columns:
        df["Date"] = df["Date"].astype(str).str.strip()
    if "Time" in df.columns:
        df["Time"] = df["Time"].astype(str).str.strip()
    return df

def _extract_pm25_method(colname: str) -> str:
    """
    Returns a method label from a PM2.5 column name.
    Examples:
      PM25_BAM -> BAM
      PM25_SHARP -> SHARP
      PM25 -> PM25
      PM2_5 -> PM25
    """
    s = str(colname).strip()
    s_norm = s.lower().replace(".", "").replace(" ", "").replace("-", "_")

    # base columns (no explicit method)
    if s_norm in ("pm25", "pm2_5"):
        return "PM25"

    # method suffix
    if "_" in s:
        return s.split("_", 1)[1].strip().upper() or "PM25"

    return "PM25"

def unify_pm25_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 1) find PM2.5 columns
    pm_cols = []
    for c in df.columns:
        c_norm = str(c).strip().lower()
        c_norm = c_norm.replace(".", "").replace(" ", "").replace("-", "_")
        if c_norm == "pm25" or c_norm.startswith("pm25_") or c_norm == "pm2_5" or c_norm.startswith("pm2_5_"):
            pm_cols.append(c)

    if not pm_cols:
        return df

    # 2) numeric conversion
    df_pm = df[pm_cols].apply(pd.to_numeric, errors="coerce")

    # 3) priority order: BAM first, then PM25, then others
    def col_rank(c: str) -> tuple:
        c_norm = str(c).strip().lower().replace(".", "").replace(" ", "").replace("-", "_")

        # BAM variants first (PM25_BAM, PM25_BAM_xxx, etc.)
        if "_bam" in c_norm:
            return (0, c_norm)

        # base PM25 next
        if c_norm == "pm25" or c_norm == "pm2_5":
            return (1, c_norm)

        # everything else
        return (2, c_norm)

    priority_cols = sorted(pm_cols, key=col_rank)

    # 4) row-wise: choose first non-NA column in priority order
    chosen_val = pd.Series(np.nan, index=df.index, dtype="float64")
    chosen_col = pd.Series(pd.NA, index=df.index, dtype="object")

    for c in priority_cols:
        mask = chosen_val.isna() & df_pm[c].notna()
        if mask.any():
            chosen_val.loc[mask] = df_pm.loc[mask, c]
            chosen_col.loc[mask] = c

    # 5) method label based on chosen column (NA if all NA)
    method_map = {c: _extract_pm25_method(c) for c in pm_cols}
    pm25_method = chosen_col.map(method_map)

    # 6) keep count of available sources (no imputation)
    n_sources = df_pm.notna().sum(axis=1)

    out = df.copy()
    out["PM25"] = chosen_val
    out["PM25_method"] = pm25_method
    out["PM25_n_sources"] = n_sources

    keep = [c for c in ["Date", "Time"] if c in out.columns] + ["PM25", "PM25_method", "PM25_n_sources"]
    return out[keep].copy()


def build_bc(bc_dir: Path, year_min = 2022, year_max = 2025, stations_limit = None, output = None):
    # only subfolders will be collected to station_dirs
    station_dirs = sorted([p for p in bc_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    if stations_limit is not None: # pick first stations_limit stations as test subset
        station_dirs = station_dirs[:stations_limit]

    dfs = []
    loaded = 0

    for station_dir in station_dirs:
        station_folder = station_dir.name
        for fp in station_dir.glob("*_dataset.csv"):
            m = YEAR_STATION_RE.match(fp.name)
            if not m:
                continue

            year = int(m.group("year"))
            if not (year_min <= year <= year_max):
                continue

            df = read_hourly_csv_with_weird_header(fp)
            df = unify_pm25_columns(df)

            df["Station"] = station_folder

            dfs.append(df)
            loaded += 1

    if not dfs:
        raise ValueError("No files loaded. Check bc_dir path and year range.")
    out_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded files: {loaded}")
    print(f"Rows: {len(out_df)}")
    print(f"Stations: {out_df['Station'].nunique()}")

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output, index=False)
        print(f"Saved to: {output.resolve()}")

    return out_df

if __name__ == "__main__":
    # script is: Dataset/Air Pollutant Data/_script/build_bc_2022_2025.py
    # so parent[1] is: Dataset/Air Pollutant Data
    AIR_DIR = Path(__file__).resolve().parents[1]
    BC_DIR = AIR_DIR / "British Columbia"
    OUT_DIR = AIR_DIR / "_processed"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
     # -----------------------
    # Quick check (small run)
    # -----------------------
    build_bc(
        bc_dir=BC_DIR,
        year_min=2022,
        year_max=2025,
        stations_limit=3,  # change to 1/2/5 as needed
        output=OUT_DIR / "BC_2022_2025_quick.csv",
    )

    # --------------------------------
    # Full collection (all stations)
    # --------------------------------
    build_bc(
        bc_dir=BC_DIR,
        year_min=2022,
        year_max=2025,
        stations_limit=None,  # None = all station folders
        output=OUT_DIR / "BC_2022_2025_combined.csv",
    )


