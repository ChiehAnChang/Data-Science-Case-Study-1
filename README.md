# Data Science Case Study 1

This repository contains the full workflow for STAT 946 Case Study 1, from raw PM2.5 data processing to exploratory analysis, modeling, and presentation outputs.

**Authors:** Chieh-An (Andy) Chang, Wenjia (Erick) Gu, Haoran Pan, Yujie Wen, Henry Zhang (in alphabetical order)

## Acknowledgements
This project was completed under the instruction and guidance of Professor Lysy and Teaching Assistant Diane Zhang.  
We also used generative AI tools to refine grammar and debug code. However, all project ideas, decisions, and interpretations were made by the authors.

## Project Goal

Forecast zone-level hourly PM2.5 and evaluate a warning system for exceedance events during wildfire smoke episodes in British Columbia.

This case study integrates:

- Cleaned/imputed PM2.5 time series (Case Study 1 pipeline output)
- Wildfire hotspot signals (CWFIS Fire M3) engineered into local and regional features

## Key Deliverables

- Presentation deck source: `stat946-CS2-PPT.qmd`
- ETL pipeline: `ETL & EDA/ETL_pipeline_cs2.py` (produces the model-ready CS2 table)
- EDA writeup: `EDA/executive_summary.qmd`
- Modeling notebooks: `Model/CS2 Modeling.ipynb` and `Model/CS2 Modeling Generalization Interface.ipynb`
- Final model input table: `Dataset/Outputs/CS2_model_input.csv`

## Quickstart (Repro)

### 1) Environment

- Python: 3.11+ (see `pyproject.toml`)
- Optional (recommended): `uv` for dependency management
- Optional: Quarto (for rendering `.qmd` outputs)

Install dependencies (choose one):

```bash
# Option A: uv (recommended if installed)
uv run python -c "import sys; print(sys.version)"
```

```bash
# Option B: pip
python -m pip install -U pip
python -m pip install -e .
```

### 2) Data prep (ETL)

If you have not downloaded the wildfire hotspot CSVs yet, follow `Dataset/README.md` to populate `Dataset/Wildfire/`.

Then run the CS2 ETL pipeline:

```bash
uv run python "ETL & EDA/ETL_pipeline_cs2.py"
```

This produces the model-ready table:

- `Dataset/Outputs/CS2_model_input.csv`

### 3) EDA

- Open and run `EDA/executive_summary.qmd` (or follow notes in `EDA/README.md`).

### 4) Modeling

- Run the modeling notebooks in `Model/`:
	- `Model/CS2 Modeling.ipynb`
	- `Model/CS2 Modeling Generalization Interface.ipynb`

### 5) Render slides

Render the presentation from the project root:

```bash
quarto render stat946-CS2-PPT.qmd
```

Outputs are saved under `Assets/Outputs/CS2-PPT-output/`.

## Repository Structure

### Top-Level Files

- `developer_guideline.md`: team/project conventions and development notes.
- `pyproject.toml`: Python project configuration and dependency metadata.
- `stat946-CS2-PPT.qmd`: Quarto presentation source file for the final deck.
- `LICENSE`: project license.

### Top-Level Folders

- `Assets/`: generated figures and export artifacts used in reports and slides.
	- `Assets/Outputs/CS2-PPT-output/`: rendered presentation outputs.
	- `Assets/Outputs/EDA/image/`: EDA charts and plots.
	- `Assets/Outputs/Model/image/`: model-related figures and comparison plots.

- `Dataset/`: all datasets and data documentation.
	- `Dataset/raw_datasets/`: original source files (CSV/GeoJSON).
	- `Dataset/Outputs/`: cleaned/imputed/model-ready tables.
	- `Dataset/data_dictionary_cs2.md`: column definitions and data meaning.
	- `Dataset/README.md`: dataset-specific instructions.

- `EDA/`: analysis scripts and narrative writeups for exploratory analysis.
	- `EDA/ci.py`: EDA utility/statistical helper script.
	- `EDA/executive_summary.qmd`: Quarto executive summary for EDA findings.
	- `EDA/README.md`: EDA module usage notes.

- `ETL & EDA/`: end-to-end preprocessing and feature preparation pipelines.
	- `ETL & EDA/ETL_EDA_pipeline.py`: combined ETL + EDA pipeline script.
	- `ETL & EDA/ETL_pipeline_cs2.py`: core ETL pipeline.
	- `ETL & EDA/README_CS2.md`: ETL/EDA workflow documentation.

- `Model/`: modeling notebooks, training pipeline, and model implementations.
	- `Model/CS2 Modeling.ipynb`: main modeling notebook.
	- `Model/CS2 Modeling Generalization Interface.ipynb`: generalization-focused modeling notebook.
	- `Model/model_pipeline.py`: reusable model training/evaluation pipeline.
	- `Model/Models/`: model class implementations.
		- `Model/Models/base_model.py`: base model interfaces/utilities.
		- `Model/Models/lightboost.py`: gradient boosting style model implementation.
		- `Model/Models/LSTM_AE_model.py`: LSTM autoencoder model implementation.
	- `Model/Modelling_README.md`: modeling-specific reproduction notes.

## Typical Workflow

1. Start with `Dataset/` and run ETL scripts in `ETL & EDA/`.
2. Perform exploration and reporting in `EDA/`.
3. Train and evaluate models from `Model/`.
4. Export figures to `Assets/Outputs/` and render the presentation/report artifacts.

## Notes

- Keep raw files in `Dataset/raw_datasets/` unchanged.
- Write all generated outputs to `Dataset/Outputs/` or `Assets/Outputs/`.
- Put module-specific details in each folder-level README, and keep this root README focused on navigation.

## Data Notes

- `Dataset/Outputs/PM25_zone_wide_imputed_2022_2025.csv` is the cleaned/imputed PM2.5 table produced by the CS1 pipeline.
- Wildfire hotspot CSVs for CWFIS Fire M3 are not committed (large global files). See `Dataset/README.md` and `ETL & EDA/README_CS2.md` for download + setup instructions.

## Outputs

- Processed tables: `Dataset/Outputs/`
- Figures and rendered artifacts: `Assets/Outputs/`

## References / Data Sources

- Canadian Wildland Fire Information System (CWFIS) hotspot archive: http://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/

## Development Notes

- Team conventions: `developer_guideline.md`
