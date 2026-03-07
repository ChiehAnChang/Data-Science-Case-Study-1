import pandas as pd
import numpy as np
from scipy.stats import norm
from itertools import combinations
from pygam import LinearGAM, s, te
from arch import arch_model
from base_model import BasePM25Model, TARGET_COL, EXCEEDANCE_THRESHOLD, WILDFIRE_COLUMNS

class GAMGARCHModel(BasePM25Model):
    def __init__(self, candidate_features=None, lam=0.2454, target_col=TARGET_COL, threshold=EXCEEDANCE_THRESHOLD, horizon=3):
        super().__init__(candidate_features, target_col, threshold, horizon)
        self.lam = lam
        self.gam_model = None
        self.garch_params = {}
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_df = df.copy()
        for lag in [3, 24, 48]:
            processed_df[f'lag_{lag}h'] = processed_df[self.target_col].shift(lag)
        processed_df['hour_of_day'] = processed_df.index.hour
        processed_df['day_of_week'] = processed_df.index.dayofweek
        processed_df['month_of_year'] = processed_df.index.month
        for col in WILDFIRE_COLUMNS:
            processed_df[col] = processed_df.get(col, 0.0).fillna(0).astype(float)
        cols_to_keep = [self.target_col, 'lag_3h', 'lag_24h', 'lag_48h', 'hour_of_day', 'day_of_week', 'month_of_year'] + WILDFIRE_COLUMNS
        return processed_df[cols_to_keep].dropna()

    def _build_gam_formula(self, processed_df, features_list):
        idx = {col: processed_df.columns.get_loc(col) for col in processed_df.columns}
        gam_formula = (s(idx['lag_3h']) + s(idx['lag_24h']) + s(idx['lag_48h']) +
                       te(idx['hour_of_day'], idx['day_of_week'], n_splines=[12, 7]) +
                       te(idx['hour_of_day'], idx['month_of_year'], basis=['cp', 'ps'], n_splines=[12, 6]))
        for col in features_list:
            gam_formula += s(idx[col], n_splines=10)
        return gam_formula

    def fit(self, train_df: pd.DataFrame):
        processed_train = self.preprocess(train_df)
        y_raw = processed_train[self.target_col].shift(-self.horizon)
        valid_idx = y_raw.dropna().index
        
        X_train = processed_train.loc[valid_idx]
        y_train = y_raw.loc[valid_idx]
        
        self.selected_features = self.candidate_features
        final_formula = self._build_gam_formula(X_train, self.selected_features)
        self.gam_model = LinearGAM(final_formula, lam=self.lam).fit(X_train, y_train)
        self.is_fitted = True

    def tune_threshold(self, val_df: pd.DataFrame, optimization_metric='f2'):
        processed_val = self.preprocess(val_df)
        y_val_raw = processed_val[self.target_col].shift(-self.horizon)
        valid_idx = y_val_raw.dropna().index
        
        mu_val = self.gam_model.predict(processed_val.loc[valid_idx])
        res_val = y_val_raw.loc[valid_idx].values - mu_val
        
        garch_res = arch_model(res_val, p=1, q=1, vol="Garch", dist="normal", rescale=False).fit(disp="off")
        self.garch_params = {
            "omega": float(garch_res.params["omega"]), "alpha": float(garch_res.params["alpha[1]"]),
            "beta": float(garch_res.params["beta[1]"]), "last_resid": float(np.asarray(garch_res.resid)[-1]),
            "last_var": float(np.asarray(garch_res.conditional_volatility)[-1] ** 2)
        }
        super().tune_threshold(val_df, optimization_metric)

    def _get_garch_sigma(self, residuals_array):
        T = len(residuals_array)
        conditional_vars = np.zeros(T)
        curr_resid, curr_var = self.garch_params["last_resid"], self.garch_params["last_var"]
        for t in range(T):
            next_var = self.garch_params["omega"] + self.garch_params["alpha"] * (curr_resid**2) + self.garch_params["beta"] * curr_var
            conditional_vars[t] = next_var
            curr_var, curr_resid = next_var, residuals_array[t]
        return np.sqrt(np.maximum(conditional_vars, 1e-12))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or not self.garch_params: raise ValueError("Model must be fitted.")
        processed_df = self.preprocess(df)
        mu_pred = self.gam_model.predict(processed_df)
        sigma_pred = self._get_garch_sigma(np.zeros(len(mu_pred))) 
        probs = 1 - norm.cdf((self.threshold - mu_pred) / (sigma_pred + 1e-12))
        
        aligned_probs = np.zeros(len(probs))
        aligned_probs[self.horizon:] = probs[:-self.horizon]
        return aligned_probs