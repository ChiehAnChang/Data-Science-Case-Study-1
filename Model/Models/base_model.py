import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import fbeta_score

TARGET_COL = 'PM25'
EXCEEDANCE_THRESHOLD = 15
WILDFIRE_COLUMNS = [
    'fire_count_regional', 'frp_regional_sum', 'hfi_weighted', 
    'fwi_mean', 'fire_count_local', 'frp_local_sum'
]

class BasePM25Model(ABC):
    def __init__(self, candidate_features=None, target_col=TARGET_COL, threshold=EXCEEDANCE_THRESHOLD, horizon=3):
        self.target_col = target_col
        self.threshold = threshold
        self.candidate_features = candidate_features if candidate_features else []
        self.horizon = horizon  # 預設預測未來 3 小時
        self.selected_features = [] 
        self.alert_probability_threshold = 0.5 
        self.is_fitted = False

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def fit(self, train_df: pd.DataFrame):
        pass

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        pass

    def tune_threshold(self, val_df: pd.DataFrame, optimization_metric='f2'):
        print(f"      [Tuning] Optimizing Threshold for High Recall (F2-Score)...")
        val_probs = self.predict_proba(val_df)
        processed_val = self.preprocess(val_df)
        actuals = (processed_val[self.target_col].values > self.threshold).astype(int)
        
        best_thresh = 0.5
        best_score = -1
        
        for t in np.arange(0.05, 0.95, 0.05):
            preds = (val_probs >= t).astype(int)
            score = fbeta_score(actuals, preds, beta=2, zero_division=0)
            if score > best_score:
                best_score = score
                best_thresh = t
                
        self.alert_probability_threshold = best_thresh
        print(f"      [Tuning] Selected Threshold: {best_thresh:.2f} (F2 Score: {best_score:.3f})")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(df)
        return (probs >= self.alert_probability_threshold).astype(int)