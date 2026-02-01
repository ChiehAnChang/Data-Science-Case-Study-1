from pathlib import Path
import pandas as pd
import runpy

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "build"
BUILD.mkdir(exist_ok=True)

# Will run our actual EDA script once they are ready
# runpy.run_path(str(REPO / "EDA" / "air_pollutant_collect.py"), run_name="__main__")

# Minimal deterministic output for CI + report
pd.DataFrame({"status": ["ok"], "stage": ["smoke"]}).to_csv(BUILD / "summary.csv", index=False)
print(f"Wrote {BUILD / 'summary.csv'}")
