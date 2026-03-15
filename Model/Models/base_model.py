import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import fbeta_score

# Define global constants for the modeling pipeline
TARGET_COL = 'PM25' # The target variable to predict
EXCEEDANCE_THRESHOLD = 25 # The PM2.5 concentration threshold defining an "exceedance" or positive event
WILDFIRE_COLUMNS = [
    'fire_count_regional', 'frp_regional_sum', 'hfi_weighted', 
    'fwi_mean', 'fire_count_local', 'frp_local_sum'
] # Canonical set of wildfire-related features used across models

class BasePM25Model(ABC):
    """
    Abstract Base Class for all PM2.5 prediction models.
    Enforces a standard interface to ensure generalization and modularity across different modeling algorithms.
    """
    def __init__(self, candidate_features=None, target_col=TARGET_COL, threshold=EXCEEDANCE_THRESHOLD, horizon=3):
        """
        Initializes the base model configuration.
        
        Args:
            candidate_features (list): List of feature column names to use.
            target_col (str): The name of the target column in the dataset.
            threshold (float): The threshold for binary classification.
            horizon (int): The forecasting horizon (e.g., predict the value 3 hours into the future).
        """
        self.target_col = target_col
        self.threshold = threshold
        self.candidate_features = candidate_features if candidate_features else []
        self.horizon = horizon  # Default forecast horizon is 3 hours
        self.selected_features = [] # To store features actually used by the model
        self.alert_probability_threshold = 0.5 # Default probability cutoff for predicting class 1 (Alert)
        self.is_fitted = False

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to preprocess raw data into model-ready features.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def fit(self, train_df: pd.DataFrame):
        """
        Abstract method to train the model.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Abstract method to output probability arrays for the positive class.
        Must be implemented by subclasses.
        """
        pass

    def tune_threshold(self, val_df: pd.DataFrame, optimization_metric='f2'):
        """
        Dynamically tunes the probability threshold for classification by evaluating
        performance on a validation dataset. Optimizes for the F2-score by default to 
        prioritize recall (catching exceedance events) while maintaining acceptable precision.
        
        Args:
            val_df (pd.DataFrame): Validation dataset.
            optimization_metric (str): The metric to optimize (currently hardcoded for F2).
        """
        print(f"      [Tuning] Optimizing Threshold for High Recall (F2-Score)...")
        # Generate probability predictions for the validation set
        val_probs = self.predict_proba(val_df)
        
        # Preprocess validation set to align with predictions and construct the ground truth binary labels
        processed_val = self.preprocess(val_df)
        actuals = (processed_val[self.target_col].values > self.threshold).astype(int)
        
        best_thresh = 0.5
        best_score = -1
        
        # Grid search over probability thresholds from 0.05 to 0.90
        for t in np.arange(0.05, 0.95, 0.05):
            preds = (val_probs >= t).astype(int)
            # F2 score weights recall twice as much as precision
            score = fbeta_score(actuals, preds, beta=2, zero_division=0)
            if score > best_score:
                best_score = score
                best_thresh = t
                
        # Save the optimal threshold as a class attribute to be used during prediction
        self.alert_probability_threshold = best_thresh
        print(f"      [Tuning] Selected Threshold: {best_thresh:.2f} (F2 Score: {best_score:.3f})")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generates final binary predictions based on the tuned probability threshold.
        
        Args:
            df (pd.DataFrame): Input dataset.
            
        Returns:
            np.ndarray: Array of binary predictions (0 or 1).
        """
        probs = self.predict_proba(df)
        return (probs >= self.alert_probability_threshold).astype(int)