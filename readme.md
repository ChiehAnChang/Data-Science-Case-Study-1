# CS2_Modeling Model Reproduction README


### Feature Groups

Two feature sets are defined:

- `BASE_COLS`: historical PM2.5 lag variables and cyclical time features
- `WILDFIRE_COLS`: wildfire-enhanced variables such as regional fire counts, local fire counts, and Fire Radiative Power

This split supports direct comparison between models trained with standard air-quality signals and models trained with additional wildfire context.

### Time Encoding

Calendar variables such as day, week, and month are converted into sine and cosine representations. This makes time information continuous and cyclical, which is more appropriate for neural networks than raw integer encodings.

Examples include:

- `sin(day)` and `cos(day)`
- `sin(week)` and `cos(week)`
- `sin(month)` and `cos(month)`

This encoding helps the model capture repeating seasonal or weekly patterns without introducing artificial discontinuities.

### Sequence Construction

The `make_sequences` function converts 1D chronological data into 3D tensors using a sliding window:

- Input shape: `(batch, seq_len, features)`
- Output target: future PM2.5 value at the specified forecast horizon

Key logic:

- `seq_len` defines how many historical time steps are used as input
- `horizon` defines how far ahead the model predicts
- a rolling window is applied across the full timeline
- `StandardScaler` is used to standardize features before model training

This step is essential because transformer models expect fixed-length sequential inputs with consistent feature dimensions.

## 2. Baseline Models: GAM

The notebook establishes statistical baselines using `LinearGAM` from the `pygam` library.

### Train/Test Split

The time-based split is:

- Training set: 2022-2024
- Test set: 2025

This preserves chronological order and avoids information leakage from future data into training.

### Baseline Comparison Design

For each zone, two GAM models are trained:

1. Base-only model using `BASE_COLS`
2. Wildfire-enhanced model using `BASE_COLS + WILDFIRE_COLS`

This setup isolates the effect of wildfire features on predictive accuracy.

### Evaluation Metrics

The GAM models are evaluated from two perspectives.

Regression metrics:

- RMSE
- MAE
- R2

Event-detection metrics using `PM2.5 > 25.0` as the pollution warning threshold:

- Precision
- Recall
- F1 score

The final evaluation is filtered to wildfire season, which makes the results more relevant for high-risk periods when smoke-driven PM2.5 spikes are most important.

## 3. Deep Learning Architecture

The notebook implements two PyTorch-based sequence regression architectures.

### A. LLaMA-Style Regressor (`LlamaRegressor`)

This model adapts several design ideas from LLaMA-style transformers for time-series regression.

Core components:

- `LlamaRMSNorm` replaces standard `LayerNorm`
- `LlamaMLP` uses a gated feed-forward design similar to SwiGLU
- learnable positional embeddings are added to input sequences

Why it matters:

- RMSNorm can improve numerical stability and training efficiency
- gated MLP blocks offer stronger nonlinear modeling capacity
- learnable positional embeddings allow the model to adapt position information directly from the dataset

### B. GPT-2 Style Transformer (`TransformerRegressor`)

This version uses standard PyTorch transformer building blocks.

Core components:

- `TransformerEncoderLayer`
- fixed sine-based absolute positional encoding via `PositionalEncoding`
- GELU activation

Why it matters:

- it provides a strong and interpretable transformer baseline
- fixed positional encodings are simple and widely used
- it allows direct comparison between a conventional transformer and the more customized LLaMA-style design

## 4. Training Procedure

The `train_one_zone` function manages model training for a single geographic zone.

### Optimizer and Scheduler

Training uses:

- `AdamW` optimizer
- grouped weight decay
- `CosineAnnealingLR` learning rate scheduler

This combination helps stabilize optimization and can improve generalization over long training runs.

### AMP and Hardware Acceleration

The code supports Automatic Mixed Precision (AMP) with `torch.cuda.amp.GradScaler`.

Benefits:

- faster training on GPU
- lower memory usage
- reduced compute cost without major loss of numerical stability

This is especially useful when training transformers on long historical windows.

## 5. Time Window Experiment

The final modeling block runs a time window experiment for the `Southern Interior` zone.

### Experiment Design

The system loops through multiple input sequence lengths:

- 12 hours
- 24 hours
- 48 hours
- 72 hours
- 96 hours
- 168 hours

For each `seq_len`, the model is trained from scratch and evaluated on the test set. The resulting metrics are recorded for comparison.

### Purpose

This experiment answers a practical modeling question:

How much historical context is actually useful for forecasting future PM2.5?

Short windows may miss important temporal patterns, while very long windows may introduce noise, redundancy, or harder optimization.

## 6. Meaning of the Generated Images

The plots generated by the time window experiment visualize how prediction quality changes as the historical input length increases.

Typical axes:

- x-axis: input sequence length (`seq_len`)
- y-axis: model performance metric such as RMSE, MAE, or R2

### How to Interpret the Plots

- If RMSE or MAE decreases as `seq_len` increases, longer historical context is helping the model
- If R2 increases, the model is explaining more variance in PM2.5
- If performance plateaus, the model has likely captured most useful temporal information
- If performance worsens at very large windows, the model may be suffering from redundant information or overfitting

### Main Insight

The figures are meant to identify the most effective amount of lookback history for air-quality forecasting. In other words, they help determine the optimal sequence length that balances useful context with model complexity.

## 7. Reproduction Notes

To reproduce the modeling section successfully, make sure the following match the original notebook setup:

- identical feature definitions for `BASE_COLS` and `WILDFIRE_COLS`
- the same chronological train/test split
- the same `seq_len` and `horizon` settings
- standardized inputs using `StandardScaler`
- consistent zone filtering, especially for wildfire-season evaluation
- the same random seed and hardware configuration when possible

Differences in preprocessing, feature engineering, or split logic can change both the reported metrics and the shapes of the final plots.

## 8. Summary

The modeling workflow in `CS2 Modeling.ipynb` is structured around three goals:

1. prepare time-series data in a sequence-friendly format
2. compare baseline GAM models with transformer-based deep learning models
3. determine how much historical context is most useful through the time window experiment

Together, these components provide both a predictive benchmark and an interpretable experimental framework for understanding PM2.5 forecasting performance under wildfire-related conditions.
