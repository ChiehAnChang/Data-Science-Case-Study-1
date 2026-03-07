import pandas as pd
import numpy as np
from scipy.stats import norm
from base_model import BasePM25Model, TARGET_COL, EXCEEDANCE_THRESHOLD
from darts import TimeSeries
from darts.models import PatchTSTModel as DartsPatchTST
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import QuantileRegression

class PatchTSTModel(BasePM25Model):
    def __init__(self, candidate_features=None, target_col=TARGET_COL, threshold=EXCEEDANCE_THRESHOLD, input_chunk_length=48, horizon=3):
        super().__init__(candidate_features, target_col, threshold, horizon)
        self.input_chunk_length = input_chunk_length
        self.model = None
        self.scaler = Scaler()
        self.cov_scaler = Scaler()
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy().iloc[self.input_chunk_length:]

    def _to_timeseries(self, df: pd.DataFrame):
        df_filled = df.ffill().fillna(0)
        target_ts = TimeSeries.from_dataframe(df_filled, value_cols=[self.target_col])
        df_cov = df_filled.copy()
        df_cov['hour'] = df_cov.index.hour
        df_cov['dayofweek'] = df_cov.index.dayofweek
        df_cov['month'] = df_cov.index.month
        cov_cols = ['hour', 'dayofweek', 'month'] + (self.candidate_features or [])
        return target_ts, TimeSeries.from_dataframe(df_cov, value_cols=cov_cols)

    def fit(self, train_df: pd.DataFrame):
        target_ts, cov_ts = self._to_timeseries(train_df)
        self.model = DartsPatchTST(
            input_chunk_length=self.input_chunk_length, output_chunk_length=self.horizon, patch_length=12,
            n_epochs=10, batch_size=64, dropout=0.1, likelihood=QuantileRegression(quantiles=[0.25, 0.5, 0.75]),
            random_state=2026, pl_trainer_kwargs={"accelerator": "gpu", "devices": 1}
        )
        self.model.fit(series=self.scaler.fit_transform(target_ts), past_covariates=self.cov_scaler.fit_transform(cov_ts), verbose=False)
        self.is_fitted = True

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted: raise ValueError("Model must be fitted.")
        processed_df = self.preprocess(df)
        target_ts, cov_ts = self._to_timeseries(df)
        
        pred_ts = self.model.historical_forecasts(
            series=self.scaler.transform(target_ts), past_covariates=self.cov_scaler.transform(cov_ts),
            start=self.input_chunk_length, forecast_horizon=self.horizon, stride=1, retrain=False, verbose=False
        )
        preds_df = self.scaler.inverse_transform(pred_ts).to_dataframe()
        q25, mu_pred, q75 = preds_df.iloc[:, 0].values, preds_df.iloc[:, 1].values, preds_df.iloc[:, 2].values
        probs = 1 - norm.cdf((self.threshold - mu_pred) / ((q75 - q25) / 1.349 + 1e-12))
        
        aligned_probs = pd.Series(0.0, index=processed_df.index)
        aligned_probs.loc[preds_df.index] = probs
        return aligned_probs.values