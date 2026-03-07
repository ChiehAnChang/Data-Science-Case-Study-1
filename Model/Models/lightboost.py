import pandas as pd
import numpy as np
import lightgbm as lgb
from base_model import BasePM25Model, TARGET_COL, EXCEEDANCE_THRESHOLD

class LightGBMModel(BasePM25Model):
    def __init__(self, candidate_features=None, target_col=TARGET_COL, threshold=EXCEEDANCE_THRESHOLD, horizon=3):
        super().__init__(candidate_features, target_col, threshold, horizon)
        self.model = lgb.LGBMClassifier(random_state=2026, n_estimators=100, verbose=-1, device="gpu")
        self.all_train_cols = []

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_df = df.copy()
        for lag in [3, 24, 48]:
            processed_df[f'lag_{lag}h'] = processed_df[self.target_col].shift(lag)
        processed_df['hour'] = processed_df.index.hour
        processed_df['dayofweek'] = processed_df.index.dayofweek
        processed_df['month'] = processed_df.index.month
        for col in self.candidate_features:
            processed_df[col] = processed_df.get(col, 0).fillna(0)
        self.all_train_cols = ['lag_3h', 'lag_24h', 'lag_48h', 'hour', 'dayofweek', 'month'] + self.candidate_features
        return processed_df.dropna(subset=[self.target_col, 'lag_3h', 'lag_24h', 'lag_48h'])

    def fit(self, train_df: pd.DataFrame):
        processed_train = self.preprocess(train_df)
        y_train_raw = processed_train[self.target_col].shift(-self.horizon)
        valid_idx = y_train_raw.dropna().index
        
        X_train = processed_train.loc[valid_idx, self.all_train_cols]
        y_train = (y_train_raw.loc[valid_idx] > self.threshold).astype(int)
        
        self.selected_features = self.candidate_features 
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted: raise ValueError("Model must be fitted.")
        processed_df = self.preprocess(df)
        probs = self.model.predict_proba(processed_df[self.all_train_cols])[:, 1]
        
        aligned_probs = np.zeros(len(probs))
        aligned_probs[self.horizon:] = probs[:-self.horizon]
        return aligned_probs