# EDA Pipeline Guide

## Scope: Case Study 1 Only

This pipeline covers **Case Study 1 (CS1)** work only. Case Study 2 (CS2) is an extension of CS1 — it consumes the CS1 cleaned PM2.5 output (`Dataset/Outputs/PM25_zone_wide_imputed_2022_2025.csv`) and adds wildfire signals on top. Because CS2 depends on raw CWFIS hotspot data that is too large to commit (~9.6M rows), it cannot be reproduced in CI and is intended to be run **locally only**. The CS1 pipeline is therefore kept intact and unchanged as the upstream dependency that CS2 builds on.

For CS2 pipeline documentation, see `ETL & EDA/README_CS2.md`.

---


## How the CI Pipeline Works

```
push to main / feature/pipeline
        |
        v
ETL_EDA_pipeline.py          # runs the full ETL + EDA, saves PNGs
  (continue-on-error)         # if it fails, pre-committed PNGs are used
        |
        v
ci.py                         # verifies all 12 expected PNGs exist
        |
        v
quarto render executive_summary.qmd   # reads PNGs, builds HTML report
        |
        v
GitHub Pages + artifact upload
```

## How to Add a New Figure

Three steps: (1) save the figure in the pipeline, (2) register it in ci.py, (3) reference it in the QMD.

### Step 1 - Save the figure in `ETL & EDA/ETL_EDA_pipeline.py`

In your plotting function, use `savefig` to save to `Assets/Outputs/`:

```python
# inside your function
outpath = plot_out_dir / "my_new_plot.png"
plt.savefig(outpath, dpi=200, bbox_inches="tight")
```

Then commit the generated PNG to `Assets/Outputs/` so it serves as a fallback if CI can't regenerate it.

### Step 2 - Register it in `EDA/ci.py`

Add the filename to the `EXPECTED_FIGURES` list:

```python
EXPECTED_FIGURES = [
    # ... existing figures ...
    "my_new_plot.png",       # <-- add this
]
```

### Step 3 - Reference it in `EDA/executive_summary.qmd`

Add a markdown image reference (path is relative to `EDA/`):

```markdown
![Description of the plot](../Assets/Outputs/my_new_plot.png)
```

### Commit checklist

- [ ] PNG saved to `Assets/Outputs/` (committed as fallback)
- [ ] Filename added to `EXPECTED_FIGURES` in `EDA/ci.py`
- [ ] Image referenced in `EDA/executive_summary.qmd`

## Key File Locations

| File | Purpose |
|------|---------|
| `ETL & EDA/ETL_EDA_pipeline.py` | Main pipeline: ETL + EDA + figure generation |
| `EDA/ci.py` | CI guard: verifies all expected figures exist |
| `EDA/executive_summary.qmd` | Quarto report template (reads PNGs) |
| `Assets/Outputs/*.png` | Generated figures (also fallback copies) |
| `.github/workflows/ci_pipeline.yml` | GitHub Actions workflow definition |
| `pyproject.toml` | Python dependencies (managed by `uv`) |

## Local Testing

```bash
# Run the pipeline locally
uv run python "ETL & EDA/ETL_EDA_pipeline.py"

# Verify figures
uv run python EDA/ci.py

# Render the report (requires Quarto installed)
quarto render EDA/executive_summary.qmd
```
