import pandas as pd
import numpy as np
import lightgbm as lgb
from base_model import BasePM25Model, TARGET_COL, EXCEEDANCE_THRESHOLD

class LightGBMModel(BasePM25Model):
    """
    LightGBM classification model for predicting PM2.5 exceedances.
    Inherits from BasePM25Model and implements tree-based learning with automatic feature engineering.
    """
    def __init__(self, candidate_features=None, target_col=TARGET_COL, threshold=EXCEEDANCE_THRESHOLD, horizon=3):
        super().__init__(candidate_features, target_col, threshold, horizon)
        # Initialize the LightGBM classifier with standard hyperparameters
        self.model = lgb.LGBMClassifier(random_state=2026, n_estimators=100, verbose=-1)
        self.all_train_cols = [] # Stores the exact sequence of columns used during training

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates historical lag features and extracts datetime components to assist the tree model
        in understanding temporal dynamics without requiring explicit sequential RNN modeling.
        """
        processed_df = df.copy()
        
        # Create historical lag features based on 3-hour, 24-hour, and 48-hour prior observations
        for lag in [3, 24, 48]:
            processed_df[f'lag_{lag}h'] = processed_df[self.target_col].shift(lag)
            
        # Extract cyclical and seasonal features from the datetime index
        processed_df['hour'] = processed_df.index.hour
        processed_df['dayofweek'] = processed_df.index.dayofweek
        processed_df['month'] = processed_df.index.month
        
        # Ensure all candidate features exist in the dataframe, defaulting to 0 if missing
        for col in self.candidate_features:
            processed_df[col] = processed_df.get(col, 0).fillna(0)
            
        # Define the final column structure expected by the model during both training and inference
        self.all_train_cols = ['lag_3h', 'lag_24h', 'lag_48h', 'hour', 'dayofweek', 'month'] + self.candidate_features
        
        # Drop rows with missing values induced by the shift() operations
        return processed_df.dropna(subset=[self.target_col, 'lag_3h', 'lag_24h', 'lag_48h'])

    def fit(self, train_df: pd.DataFrame):
        """
        Trains the LightGBM classifier. Shifts the target forward by 'horizon' steps to construct
        the supervised future target variable maps (y).
        """
        processed_train = self.preprocess(train_df)
        
        # Shift the continuous target forward by the specified horizon to create the future labels
        y_train_raw = processed_train[self.target_col].shift(-self.horizon)
        
        # Identify valid indices where we have both engineered features and a valid future target label
        valid_idx = y_train_raw.dropna().index
        
        # Subset the predictors and binarize the future target labels based on the exceedance threshold
        X_train = processed_train.loc[valid_idx, self.all_train_cols]
        y_train = (y_train_raw.loc[valid_idx] > self.threshold).astype(int)
        
        self.selected_features = self.candidate_features 
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predicts the probability of the positive class happening in the future (at time t + horizon).
        Returns a probability array geographically aligned to the original preprocessed index.
        """
        if not self.is_fitted: raise ValueError("Model must be fitted before calling predict_proba.")
        
        processed_df = self.preprocess(df)
        # Extract probability of class 1 (exceedance)
        probs = self.model.predict_proba(processed_df[self.all_train_cols])[:, 1]
        
        # Align probabilities by forward-shifting the predictions by 'horizon' 
        # so that the prediction mapping aligns strictly with the observed outcome sequences in time.
        aligned_probs = np.zeros(len(probs))
        aligned_probs[self.horizon:] = probs[:-self.horizon]
        
        return aligned_probs