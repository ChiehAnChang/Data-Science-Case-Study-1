# Model Training and Generalization Interface

This directory contains the codebase for training, evaluating, and comparing different machine learning models designed to predict PM2.5 exceedance events based on historical air quality and wildfire data. The architecture emphasizes modularity, readability, and reproducibility.

## Directory Structure

- **`CS2 Modeling Generalization Interface.py` / `.ipynb`**: The main entry point for the modeling pipeline. It handles data loading, region-based iteration, temporal train/validation/test splitting, model instantiation, training, threshold tuning, and final evaluation visualizing the results.
- **`Models/`**: A package containing the object-oriented implementation of the predictive models. 
  - **`base_model.py`**: Defines the abstract base class `BasePM25Model` that all specific model implementations inherit from. It sets the standard interface (`preprocess`, `fit`, `predict_proba`) and implements common utilities like dynamic threshold tuning based on the F2-score to optimize recall for exceedance events.
  - **`lightboost.py`**: Contains the `LightGBMModel` class, which implements a tree-based gradient boosting classifier using LightGBM. It includes feature engineering logic for creating temporal lag features to deduce trends accurately.
  - **`LSTM_AE_model.py`**: Contains the `AutoencoderLSTMModel` class, implementing a PyTorch-based Long Short-Term Memory Autoencoder. It captures complex temporal dependencies and models data sequences using deep learning, optimized natively with Metal Performance Shaders (MPS) for Mac Apple Silicon GPUs or CUDA.

## Reproducibility and Usage
1. Instantiate desired models from the `Models` package in the notebook or python script interface.
2. The core system uniformly processes the designated training, validation, and testing periods, effectively eliminating temporal data leakage.
3. Call `.fit()` on training data, `.tune_threshold()` on validation data to identify the optimal F2-score cutoff, and finally `.predict()` on the withheld test data.
