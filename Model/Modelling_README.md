# PM2.5 Forecasting: Modeling Pipeline and Experiments

This directory contains the codebase for training, evaluating, and comparing different machine learning models designed to predict PM2.5 exceedance events. We incorporate both historical air quality signals and wildfire-enhanced data to effectively forecast severe pollution events. 

The models explored range from statistical baselines to deep learning sequence models, culminating in a Transformer-based architecture chosen as the main model for final comparisons, alongside additional models that guided our modeling decisions and sequential data representation process.

---

## Directory Structure & File Overview

- **`CS2 Additional Modeling.ipynb`**: 
  The core generalization interface notebook for the modeling pipeline. It handles data loading, region-based iteration, chronological temporal splitting (preventing data leakage), model instantiation, training, threshold tuning, and final visualizations.
  
- **`model_pipeline.py`**:
  The previous modeling pipeline established during Phase 1 of the case study. It handles foundational sequence construction, train/validation boundaries, and earlier exploratory executions, serving as historical modeling context for earlier iterations.

- **`CS2 Main Experimental Modeling.ipynb`**:
  The main experimental pipelines supporting the Transformer architectures. They oversee sequence construction, train/validation boundaries, and the execution of the LLaMA-style and GPT-style models.

- **`Models/`**: A package containing the object-oriented implementations of the additional predictive architectures:
  - **`base_model.py`**: Defines the abstract base class `BasePM25Model`. It enforces a standard API (`preprocess`, `fit`, `predict_proba`) and implements common utilities such as dynamic threshold tuning optimized for the F2-score (to prioritize recall for critical exceedance events).
  - **`lightboost.py`**: Contains `LightGBMModel`, a tree-based gradient boosting classifier. It natively includes feature engineering logic for temporal lag features, serving as an exploratory baseline.
  - **`LSTM_AE_model.py`**: Contains `AutoencoderLSTMModel`, implementing a PyTorch-based Long Short-Term Memory Autoencoder. This model captures complex temporal dependencies and maps sequences using deep learning, representing an intermediate developmental step towards the final Transformer models.

---

## 1. Feature Engineering & Sequence Construction

### Feature Groups
Two primary feature sets are defined across the pipeline:
- **`BASE_COLS`**: Historical PM2.5 lag variables and cyclical time features.
- **`WILDFIRE_COLS`**: Wildfire-enhanced variables (e.g., regional fire counts, local fire counts, and Fire Radiative Power).
This split supports direct comparisons isolating the exact modeling impact of incorporating external wildfire features.

### Time Encoding & Sequences
- Calendar variables (day, week, month) are transformed into cyclical sine/cosine representations, providing a continuous structure ideal for neural network interpretation.
- 1D chronological datasets are converted into 3D tensors `(batch, seq_len, features)` using a sliding window. `seq_len` determines the historical lookback, while `horizon` dictates the forward forecasting target. Standard scaling is applied uniformly.

---

## 2. Models Overview

Our exploration follows a progressive complexity curve:

### A. Phase 1 & Exploratory Baselines
- **GAM (LinearGAM)**: Establishes a foundational statistical baseline for standard air-quality vs. wildfire-enhanced predictions.
- **LightGBM (`lightboost.py`)**: Provides a highly-efficient, tree-based model that utilizes engineered lag features to deduce trends accurately without deep sequence logic.

### B. Recurrent Deep Learning (Intermediate Phase)
- **LSTM Autoencoder (`LSTM_AE_model.py`)**: An architecture combining an encoder-decoder reconstruction branch with a classifier. Designed to extract robust latent representations of sequential time-series data, it natively optimizes for both structure and binary exceedance probabilities using Apple Silicon (MPS) or CUDA. 

### C. Transformer Architectures (The Final Models)
The final pipeline incorporates state-of-the-art PyTorch-based sequence regression architectures to execute predictions:
- **LLaMA-Style Regressor**: Adapts LLaMA design principles (e.g., `LlamaRMSNorm`, gated SwiGLU-style `LlamaMLP`, learnable positional embeddings) to timeseries regression, offering superior numerical stability and nonlinear modeling capacity over long historical windows.
- **GPT-2 Style Transformer Regressor**: A conventional Transformer leveraging standard building blocks (`TransformerEncoderLayer`, fixed sine-based absolute positional encoding). It serves as an interpretable baseline comparison against the customized LLaMA-style design.

---

## 3. Training & Evaluation Methodology

### Chronological Splitting
Strict time precedence boundaries eliminate data leakage:
- **Train**: Historical data (2022-2023)
- **Tune/Validation**: 2024 timeframe
- **Test**: Strictly unseen future 2025 forecasting frames.

### Threshold Tuning
Once trained, the models undergo a threshold tuning phase on the validation set. By default, it spans prediction probabilities to optimize the **F2-score**, ensuring the system favors High Recall to catch dangerous PM2.5 > 25.0 exceedance spikes without crippling precision.

### The Time Window Experiment
A dedicated experimental block loops through multiple input sequence lengths (12h, 24h, 48h, 72h, 96h, 168h) for the deep learning models. This answers a core practical question: *How much historical context is actually useful for forecasting future PM2.5?* Plotting these RMSE/MAE metrics against sequence lengths empirically validates at what point expanding the temporal boundary introduces noise rather than signal.

---

## 4. Reproducibility & Usage Notes

To successfully reproduce the modeling environment:
1. Ensure identical feature definitions for the base and wildfire subgroups.
2. The core system must uniformly process the designated training, validation, and testing periods chronologically. 
3. Call `.fit()` on training data, use `.tune_threshold()` on validating matrices to isolate the optimal alert cutoff, and evaluate final `.predict()` behavior against 2025 values across target geographic zones (filtered specifically for Wildfire seasons when evaluating spike sensitivity).
4. For baseline/additional models, simply instantiate from the `Models` package utilizing the shared OOP interfaces.
5. Use consistent random seeds, learning rate schedulers (`CosineAnnealingLR`), and optimizers (`AdamW` with grouped weight decay) when replicating the Transformer training procedures.
