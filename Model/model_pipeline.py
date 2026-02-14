import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pygam import LinearGAM, s, te
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from arch import arch_model
from scipy.stats import norm
from sklearn.metrics import (
    recall_score, precision_score, accuracy_score, f1_score,
    balanced_accuracy_score, confusion_matrix
)

# -------------------------
# Paths
# -------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]   # script in "Model/"
RAW_DIR = PROJECT_DIR / "Dataset" / "raw_datasets"

PLOT_OUT_DIR = PROJECT_DIR / "Assets" / "Outputs" / "Model" / "image"     # Model plots

PLOT_OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# helper functions
# -------------------------
def get_garch_sigma_manual(residuals_array, omega, alpha, beta, last_resid_train, last_var_train):
    T = len(residuals_array)
    conditional_vars = np.zeros(T)
    curr_resid = float(last_resid_train)
    curr_var = float(last_var_train)

    for t in range(T):
        next_var = omega + alpha * (curr_resid**2) + beta * curr_var
        conditional_vars[t] = next_var
        curr_var = next_var
        curr_resid = residuals_array[t]

    return np.sqrt(np.maximum(conditional_vars, 1e-12))

def _naive_max3_recall(series, threshold=15, eval_year=2024):
    s = series.astype(float)
    d = pd.DataFrame(index=s.index)
    d["lag0"] = s
    d["lag1"] = s.shift(1)
    d["lag2"] = s.shift(2)
    d["y_tplus3"] = s.shift(-3)
    d["target_time"] = d.index + pd.Timedelta(hours=3)
    d["target_year"] = d["target_time"].dt.year
    d = d.dropna()
    d = d[d["target_year"] == eval_year]
    if len(d) == 0:
        return np.nan

    y_true = (d["y_tplus3"] > threshold).astype(int).values
    y_pred = ((d["lag0"] > threshold) | (d["lag1"] > threshold) | (d["lag2"] > threshold)).astype(int).values
    return float(recall_score(y_true, y_pred, zero_division=0))

def _pick_threshold(probs, y_true_bin, mode="fixed", fixed_tau=0.3, target_recall=None):
    if mode == "fixed":
        return float(fixed_tau), None

    grid = np.linspace(0.01, 0.99, 99)
    rows = []
    for tau in grid:
        y_pred = (probs >= tau).astype(int)
        rec = recall_score(y_true_bin, y_pred, zero_division=0)
        prec = precision_score(y_true_bin, y_pred, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred, zero_division=0)
        rows.append((tau, rec, prec, f1))

    tab = pd.DataFrame(rows, columns=["tau", "recall", "precision", "f1"])

    if mode == "target_recall_max3":
        if target_recall is not None and np.isfinite(target_recall):
            feasible = tab[tab["recall"] >= target_recall]
            if len(feasible) > 0:
                best = feasible.sort_values(["precision", "f1", "tau"], ascending=[False, False, False]).iloc[0]
                return float(best["tau"]), tab
        best = tab.sort_values(["f1", "precision"], ascending=[False, False]).iloc[0]
        return float(best["tau"]), tab

    raise ValueError("threshold_mode must be 'fixed' or 'target_recall_max3'")


# -------------------------
# Main pipeline Functions
# -------------------------
def train_eval_regional_gams_no_save(
    df,
    train_end_year=2023,        
    eval_years=(2024, 2025),
    threshold=15,
    plot_hours=300,
    plot_regions=None
):
    """
    Train GAMs using the same method (features/terms/lambda are consistent), but do not save the models.
    """

    # ---- 0) Index processing ----
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    all_metrics = []
    pred_store = {}     # key: (region, year) -> dataframe
    model_store = {}    # key: region -> fitted gam

    for region_name in df.columns:
        print(f"\n>>> Processing Region: {region_name}")

        # ---- A) Feature engineering (consistent with yours) ----
        data = pd.DataFrame(index=df.index)
        data['y'] = df[region_name]
        data['lag3'] = data['y'].shift(3)
        data['lag24'] = data['y'].shift(24)
        data['lag48'] = data['y'].shift(48)
        data['hour'] = data.index.hour
        data['weekday'] = data.index.dayofweek
        data['month'] = data.index.month
        data = data.dropna()

        feature_cols = ['lag3', 'lag24', 'lag48', 'hour', 'weekday', 'month']

        # ---- B) Training set ----
        train_data = data[data.index.year < train_end_year]
        if len(train_data) < 200:
            print(f"   [Skip] Too few train samples: {len(train_data)}")
            continue

        X_train = train_data[feature_cols]
        y_train = train_data['y']

        print(f"   Train samples: {len(X_train)}")

        # ---- C) GAM (consistent with your code) ----
        gam = LinearGAM(
            s(0) + s(1) + s(2) +
            te(3, 4, n_splines=[12, 7]) +
            te(3, 5, basis=['cp', 'ps'], n_splines=[12, 6]),
            lam=0.2454
        ).fit(X_train, y_train)

        model_store[region_name] = gam
        print(f"   ✅ Fitted. Pseudo R2: {gam.statistics_['pseudo_r2']['explained_deviance']:.4f}")

        # ---- D) Yearly testing ----
        for yy in eval_years:
            test_data = data[data.index.year == yy].copy()
            if len(test_data) == 0:
                continue

            X_test = test_data[feature_cols]
            y_test = test_data['y'].values
            y_pred = gam.predict(X_test)

            resid = y_test - y_pred

            # Regression metrics
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            # Event metrics (based on point prediction threshold)
            y_true_bin = (y_test > threshold).astype(int)
            y_pred_bin = (y_pred > threshold).astype(int)

            precision = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
            recall = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
            f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))

            # Residual time series tests
            acf1 = float(np.corrcoef(resid[1:], resid[:-1])[0, 1]) if len(resid) > 1 else np.nan
            lb = acorr_ljungbox(resid, lags=[24], return_df=True)
            lb_p24 = float(lb['lb_pvalue'].iloc[0])

            all_metrics.append({
                "Region": region_name,
                "Year": yy,
                "N": len(test_data),
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "Precision@15": precision,
                "Recall@15": recall,
                "F1@15": f1,
                "ActualEvents": int(y_true_bin.sum()),
                "PredEvents": int(y_pred_bin.sum()),
                "ACF1_resid": acf1,
                "LB_p24": lb_p24
            })

            pred_store[(region_name, yy)] = pd.DataFrame({
                "Datetime": test_data.index,
                "y_true": y_test,
                "y_pred": y_pred,
                "resid": resid,
                "y_true_bin": y_true_bin,
                "y_pred_bin": y_pred_bin
            }).set_index("Datetime")

    report_df = pd.DataFrame(all_metrics).sort_values(["Year", "R2"], ascending=[True, False]).reset_index(drop=True)

    #print("\n================== METRICS ==================")
    #display(report_df)

    # ---- E) Visualization evidence ----
    if plot_regions is None:
        plot_regions = sorted(set([k[0] for k in pred_store.keys()]))

    for region in plot_regions:
        for yy in eval_years:
            key = (region, yy)
            if key not in pred_store:
                continue

            d = pred_store[key].copy()
            d_short = d.iloc[:plot_hours] if plot_hours is not None else d
            resid = d["resid"].values
            acf_vals = sm.tsa.stattools.acf(resid, nlags=48, fft=True) if len(resid) > 50 else np.array([0])

            fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
            fig.suptitle(f"{region} | Test {yy}", fontsize=14)

            # 1) Time series fit (first plot_hours)
            axes[0, 0].plot(d_short.index, d_short["y_true"], label="Actual", lw=1.3)
            axes[0, 0].plot(d_short.index, d_short["y_pred"], label="Pred", lw=1.2)
            axes[0, 0].set_title(f"Actual vs Pred (first {len(d_short)}h)")
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)

            # 2) Scatter plot fit
            axes[0, 1].scatter(d["y_true"], d["y_pred"], s=8, alpha=0.3)
            lo = min(d["y_true"].min(), d["y_pred"].min())
            hi = max(d["y_true"].max(), d["y_pred"].max())
            axes[0, 1].plot([lo, hi], [lo, hi], "r--", lw=1)
            axes[0, 1].set_title("Pred vs Actual")
            axes[0, 1].set_xlabel("Actual")
            axes[0, 1].set_ylabel("Pred")
            axes[0, 1].grid(alpha=0.3)

            # 3) Residual time series
            axes[0, 2].plot(d_short.index, d_short["resid"], lw=0.9, color="gray")
            axes[0, 2].axhline(0, color="black", lw=1)
            axes[0, 2].set_title("Residual (first window)")
            axes[0, 2].grid(alpha=0.3)

            # 4) Residual ACF
            if len(acf_vals) > 1:
                lags = np.arange(1, min(49, len(acf_vals)))
                axes[1, 0].bar(lags, acf_vals[1:len(lags)+1], alpha=0.8)
            axes[1, 0].set_title("Residual ACF")
            axes[1, 0].set_xlabel("Lag")
            axes[1, 0].grid(alpha=0.3)

            # 5) Residual distribution
            sns.histplot(d["resid"], bins=40, kde=True, ax=axes[1, 1], color="steelblue")
            axes[1, 1].set_title("Residual Distribution")
            axes[1, 1].grid(alpha=0.3)

            # 6) Diurnal profile (observed vs predicted)
            tmp = d.copy()
            tmp["hour"] = tmp.index.hour
            hh = tmp.groupby("hour")[["y_true", "y_pred"]].mean().reset_index()
            axes[1, 2].plot(hh["hour"], hh["y_true"], marker="o", label="Actual")
            axes[1, 2].plot(hh["hour"], hh["y_pred"], marker="o", label="Pred")
            axes[1, 2].set_title("Diurnal Profile")
            axes[1, 2].set_xlabel("Hour")
            axes[1, 2].legend()
            axes[1, 2].grid(alpha=0.3)

            #plt.show()
            plt.savefig(PLOT_OUT_DIR / f"GAM_NoSave_{region}_Test{yy}.png")
            plt.close()

    return report_df, pred_store, model_store

def run_pipeline_with_inmemory_models(
    df,
    gam_models,                         # dict: {region_name: fitted GAM}
    threshold=15,
    risk_threshold=0.35,
    ci_level=0.90,
    garch_train_year=2024,
    test_year=2025,
    threshold_mode="fixed"              # 'fixed' | 'target_recall_max3'
):
    """
    Full pipeline:
    1) region-wise feature engineering
    2) use in-memory GAM mean model
    3) fit GARCH(1,1) on calibration-year residuals
    4) compute test-year exceedance probability
    5) evaluate classification + CI coverage
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    z_ci = norm.ppf(0.5 + ci_level / 2)

    results = []
    detail_dict = {}
    threshold_table = {}

    for region in df.columns:
        if region not in gam_models:
            print(f"[Skip] in-memory GAM not found: {region}")
            continue

        gam = gam_models[region]

        data = pd.DataFrame(index=df.index)
        data["y"] = df[region].astype(float)
        data["lag3"] = data["y"].shift(3)
        data["lag24"] = data["y"].shift(24)
        data["lag48"] = data["y"].shift(48)
        data["hour"] = data.index.hour
        data["weekday"] = data.index.dayofweek
        data["month"] = data.index.month
        data = data.dropna()

        feature_cols = ["lag3", "lag24", "lag48", "hour", "weekday", "month"]

        mask_cal = (data.index.year == garch_train_year)
        mask_test = (data.index.year == test_year)
        if mask_cal.sum() == 0 or mask_test.sum() == 0:
            print(f"[Skip] Missing {garch_train_year}/{test_year}: {region}")
            continue

        # ---- Calibration year (fit GARCH on GAM residuals) ----
        X_cal = data.loc[mask_cal, feature_cols]
        y_cal = data.loc[mask_cal, "y"].values
        mu_cal = gam.predict(X_cal)
        res_cal = y_cal - mu_cal

        am = arch_model(res_cal, p=1, q=1, vol="Garch", dist="normal", rescale=False)
        garch_res = am.fit(disp="off")

        params = garch_res.params
        omega = float(params["omega"])
        alpha = float(params["alpha[1]"])
        beta = float(params["beta[1]"])

        resid_arr = np.asarray(garch_res.resid)
        vol_arr = np.asarray(garch_res.conditional_volatility)
        last_resid_cal = float(resid_arr[-1])
        last_var_cal = float(vol_arr[-1] ** 2)

        # sigma/prob on calibration year (for threshold tuning)
        sigma_cal = get_garch_sigma_manual(res_cal, omega, alpha, beta, last_resid_cal, last_var_cal)
        p_cal = 1 - norm.cdf((threshold - mu_cal) / (sigma_cal + 1e-12))
        y_cal_bin = (y_cal > threshold).astype(int)

        # threshold selection
        target_rec = None
        if threshold_mode == "target_recall_max3":
            target_rec = _naive_max3_recall(df[region], threshold=threshold, eval_year=garch_train_year)

        tau_region, tau_tab = _pick_threshold(
            probs=p_cal,
            y_true_bin=y_cal_bin,
            mode=threshold_mode,
            fixed_tau=risk_threshold,
            target_recall=target_rec
        )
        threshold_table[region] = tau_tab

        # ---- Test year ----
        X_test = data.loc[mask_test, feature_cols]
        y_test = data.loc[mask_test, "y"].values
        idx_test = data.loc[mask_test].index

        mu_test = gam.predict(X_test)
        res_test = y_test - mu_test

        sigma_test = get_garch_sigma_manual(res_test, omega, alpha, beta, last_resid_cal, last_var_cal)
        probs_test = 1 - norm.cdf((threshold - mu_test) / (sigma_test + 1e-12))

        y_true_bin = (y_test > threshold).astype(int)
        y_pred_bin = (probs_test >= tau_region).astype(int)

        rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        acc = accuracy_score(y_true_bin, y_pred_bin)
        bacc = balanced_accuracy_score(y_true_bin, y_pred_bin)

        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        ci_lo = mu_test - z_ci * sigma_test
        ci_hi = mu_test + z_ci * sigma_test
        coverage = float(np.mean((y_test >= ci_lo) & (y_test <= ci_hi)))

        results.append({
            "Region": region,
            "Tau_Region": float(tau_region),
            "TargetRecall_Max3_CalYear": target_rec,
            "Recall": float(rec),
            "Precision": float(prec),
            "F1": float(f1),
            "Accuracy": float(acc),
            "Balanced_Acc": float(bacc),
            f"CI{int(ci_level*100)}_Coverage": coverage,
            "Actual Events": int(y_true_bin.sum()),
            "Model Alerts": int(y_pred_bin.sum()),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)
        })

        d = pd.DataFrame(index=idx_test)
        d["y_true"] = y_test
        d["mu"] = mu_test
        d["sigma"] = sigma_test
        d["ci_lo"] = ci_lo
        d["ci_hi"] = ci_hi
        d["p_event"] = probs_test
        d["y_true_bin"] = y_true_bin
        d["y_pred_bin"] = y_pred_bin
        d["tp"] = (d["y_true_bin"].eq(1) & d["y_pred_bin"].eq(1))
        d["fp"] = (d["y_true_bin"].eq(0) & d["y_pred_bin"].eq(1))
        d["fn"] = (d["y_true_bin"].eq(1) & d["y_pred_bin"].eq(0))
        detail_dict[region] = d

    report_df = pd.DataFrame(results).set_index("Region").sort_values("Recall", ascending=False)
    return report_df, detail_dict, threshold_table

def plot_region(region, detail_dict, report_df, threshold=15, risk_threshold=0.3, days=60):
    d = detail_dict[region].copy().sort_index()
    if days is not None:
        d = d.loc[d.index < (d.index.min() + pd.Timedelta(days=days))]

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True, constrained_layout=True)
    row = report_df.loc[region]

    # 1) Fit + CI
    axes[0].plot(d.index, d['y_true'], color='black', lw=1.2, label='Actual')
    axes[0].plot(d.index, d['mu'], color='tab:blue', lw=1.2, label='Pred Mean')
    axes[0].fill_between(d.index, d['ci_lo'], d['ci_hi'], color='tab:blue', alpha=0.2, label='90% CI')
    axes[0].axhline(threshold, color='red', ls='--', lw=1.2, label=f'Threshold={threshold}')
    evt_idx = d.index[d['y_true_bin'] == 1]
    axes[0].scatter(evt_idx, d.loc[evt_idx, 'y_true'], s=18, color='red', alpha=0.8, label='Actual Events')
    axes[0].set_title(f"{region} | P={row['Precision']:.3f} R={row['Recall']:.3f} F1={row['F1']:.3f} Acc={row['Accuracy']:.3f}")
    axes[0].legend(loc='upper right')
    axes[0].grid(alpha=0.25)

    # 2) Exceedance probability + TP/FP/FN
    axes[1].plot(d.index, d['p_event'], color='purple', lw=1.1, label='P(y>threshold)')
    axes[1].axhline(risk_threshold, color='orange', ls='--', lw=1.2, label=f'RiskThr={risk_threshold}')
    axes[1].scatter(d.index[d['tp']], d.loc[d['tp'], 'p_event'], s=22, color='green', label='TP')
    axes[1].scatter(d.index[d['fp']], d.loc[d['fp'], 'p_event'], s=22, color='darkorange', label='FP')
    axes[1].scatter(d.index[d['fn']], d.loc[d['fn'], 'p_event'], s=22, color='red', label='FN')
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(loc='upper right', ncol=4)
    axes[1].grid(alpha=0.25)

    # 3) Residual
    resid = d['y_true'] - d['mu']
    axes[2].plot(d.index, resid, color='gray', lw=0.8, label='Residual')
    axes[2].plot(d.index, resid.rolling(24, min_periods=1).mean(), color='tab:blue', lw=1.2, label='24h rolling mean')
    axes[2].axhline(0, color='black', lw=1)
    axes[2].legend(loc='upper right')
    axes[2].grid(alpha=0.25)
    #plt.show()
    plt.savefig(PLOT_OUT_DIR / f"Pipeline_{region}_DetailPlot.png")
    plt.close()

    # confusion matrix
    cm = confusion_matrix(d['y_true_bin'], d['y_pred_bin'])
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{region} Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    #plt.show()
    plt.savefig(PLOT_OUT_DIR / f"Pipeline_{region}_ConfusionMatrix.png")
    plt.close()


if __name__ == "__main__":
    # =========================
    # import imputed data
    # =========================

    df_imputed = pd.read_csv(PROJECT_DIR / "Dataset" / "Outputs" / "PM25_zone_wide_imputed_2022_2025.csv", index_col=0)

    #print(df_imputed.head())
    #print(df_imputed.index.min(), df_imputed.index.max())
    
    # =========================
    # Train GAM model
    # =========================
    report_df, pred_store, model_store = train_eval_regional_gams_no_save(
        df_imputed,
        train_end_year=2023,      # Keep your original logic (<2023)
        eval_years=(2024, 2025),
        threshold=15,
        plot_hours=300,
        plot_regions=None         # or ['Northeast']
    )
    
    # =========================
    # Inference
    # =========================
    m_report_df, detail_dict, threshold_table = run_pipeline_with_inmemory_models(
        df_imputed,
        gam_models=model_store,              # or regional_models
        threshold=15,
        risk_threshold=0.35,
        garch_train_year=2024,
        test_year=2025,
        threshold_mode="fixed" # or "target_recall_max3"
    )
    
    # =========================
    # Plotting per region + confusion matrix
    # =========================
    for r in m_report_df.index:
        plot_region(r, detail_dict, m_report_df, threshold=15, risk_threshold=0.35, days=60)
    