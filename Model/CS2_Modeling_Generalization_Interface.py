import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Path Configurations & Environmental Setups
# ---------------------------------------------------------
# Automatically resolve directories dynamically utilizing pathlib 
# to ensure reproducible path execution irrespective of the OS ecosystem.
current_dir = Path.cwd()
models_dir = current_dir / 'Models'

# Append the custom Models module directory to the system path
if str(models_dir) not in sys.path:
    sys.path.append(str(models_dir))

# Import the Abstract Base setup configuration constraints
from Models.base_model import TARGET_COL, EXCEEDANCE_THRESHOLD, WILDFIRE_COLUMNS
# Import implementations of specific predictive models constructed in the folder
from Models.lightboost import LightGBMModel         
from Models.LSTM_AE_model import AutoencoderLSTMModel 

if __name__ == "__main__":
    # Define absolute input sources based on repository project hierarchy abstractions.
    PROJECT_DIRECTORY = Path.cwd().parents[0]
    dataset_file_path = PROJECT_DIRECTORY / "Dataset" / "Outputs" / "CS2_model_input.csv"
    
    # ---------------------------------------------------------
    # Central Data Ingestion & Preprocessing Core 
    # ---------------------------------------------------------
    raw_input_dataframe = pd.read_csv(dataset_file_path)
    
    # Uniformly align datetime types to prepare for strict chronological sorting operations
    raw_input_dataframe['Datetime_UTC'] = pd.to_datetime(raw_input_dataframe['Datetime_UTC'])
    available_regions_list = raw_input_dataframe['Zone'].unique()
    all_results = []

    print("--- Running Unified Object-Oriented Pipeline (Horizon=3) ---")

    # Iterate independently over every isolated unique geographic region mappings
    for region_name in available_regions_list:
        print(f"\nProcessing Region: {region_name} ...")
        
        # Partition dataset specific to a singular geographic Zone constraint mapping
        regional_df = raw_input_dataframe[raw_input_dataframe['Zone'] == region_name].copy()
        regional_df = regional_df.set_index('Datetime_UTC').sort_index()
        
        # Normalize timezone states mapping redundant timezone metadata back to standard naive UTC constraints
        if regional_df.index.tz is not None:
            regional_df.index = regional_df.index.tz_localize(None)
            
        # ---------------------------------------------------------
        # Temporal Train/Validation/Test Target Splitting Structure
        # ---------------------------------------------------------
        # Maintain strict time precedence boundaries eliminating Data Leakage biases.
        # Train securely on historical data structures originating from before/on 2023. 
        train_data = regional_df[regional_df.index.year <= 2023].copy()
        # Tuning mapping an isolated threshold strictly evaluated entirely onto an independent 2024 timeframe window.
        val_data = regional_df[regional_df.index.year == 2024].copy()
        # Secure ground-testing validations strictly bounded exclusively onto unseen future 2025 forecasting frames.
        test_data = regional_df[regional_df.index.year == 2025].copy()
        
        # Halt model assessments lacking statistically viable row capacities mapping parameters.
        if len(train_data) < 200: continue

        # ---------------------------------------------------------
        # Model Interface Definition Configurations
        # ---------------------------------------------------------
        # Utilize Object Polymorphism declaring diverse architecture instances bridging parent methodology bounds.
        models_to_run = {
            "3_LightGBM_Wildfire": LightGBMModel(candidate_features=WILDFIRE_COLUMNS),
            "9_LSTM_AE_Wildfire": AutoencoderLSTMModel(candidate_features=WILDFIRE_COLUMNS),
        }

        # Iteratively optimize each instantiated sub-class architecture parameter sequentially per region. 
        for model_name, model in models_to_run.items():
            print(f"  -> Training {model_name}...")
            
            # Step 1: Initialize parameter fittings pre-processing subsets dynamically mapped towards structural designs. 
            model.fit(train_data)
            
            # Step 2: Extract mapping optimization bounds via non-intrusive F2 Metric evaluations avoiding test-set biases. 
            model.tune_threshold(val_data)
            
            # Step 3: Conclude final performance metrics mapping computed test projections securely on 2025 bounded frames.
            test_preds = model.predict(test_data)
            processed_test = model.preprocess(test_data) # Aligning input matrices onto processed actualizations bounds
            actuals = (processed_test[TARGET_COL] > EXCEEDANCE_THRESHOLD).astype(int)
            
            # Output Dictionary structuring historical metric performances parsed during evaluation matrix validations.
            all_results.append({
                "Region_Name": region_name,
                "Model_Type": model_name,
                "Probability_Threshold": round(model.alert_probability_threshold, 2),
                "Recall": float(recall_score(actuals, test_preds, zero_division=0)),
                "Precision": float(precision_score(actuals, test_preds, zero_division=0)),
                "Accuracy": float(accuracy_score(actuals, test_preds)),
                "F1": float(f1_score(actuals, test_preds, zero_division=0))
            })

            # ---------------------------------------------------------
            # Matplotlib Output Evaluation Visualizations Mapping Data 
            # ---------------------------------------------------------
            plt.figure(figsize=(15, 5))
            test_dates = processed_test.index
            actual_pm25 = processed_test[TARGET_COL].values
            
            plt.plot(test_dates, actual_pm25, label='Actual PM2.5', color='steelblue', alpha=0.7, linewidth=1.5)
            # Display target exceedances evaluations relative towards bounded threshold thresholds caps mappings.
            plt.axhline(EXCEEDANCE_THRESHOLD, color='gray', linestyle='--', linewidth=2, label=f'Threshold ({EXCEEDANCE_THRESHOLD})')
            
            predicted_alert_indices = np.where(test_preds == 1)[0]
            plt.scatter(
                test_dates[predicted_alert_indices], actual_pm25[predicted_alert_indices], 
                color='red', s=30, zorder=5, label=f'Model Warning (Predicted > {EXCEEDANCE_THRESHOLD} in 3 Hours)'
            )
            
            plt.title(f'[{region_name}] {model_name} - Prediction Results (Test 2025)', fontsize=14, fontweight='bold')
            plt.xlabel('Date (UTC)', fontsize=12)
            plt.ylabel('PM2.5 Concentration', fontsize=12)
            plt.legend(loc='upper right')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            plt.show()

    print("\n=== Final Classification Report ===")
    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df = results_df.set_index(["Region_Name", "Model_Type"]).sort_index()
        try:
            display(results_df) 
        except NameError:
            print(results_df) 
    else:
        print("No results generated.")
