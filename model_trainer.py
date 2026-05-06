# ╔══════════════════════════════════════════════════════════════════╗
# ║  model_trainer.py — NetSmart ML Model Training & Prediction     ║
# ║  Houses feature encoding, all four model definitions,           ║
# ║  the unified prediction helper, and the best-time recommender.  ║
# ║                                                                  ║
# ║  Split: 80% Training / 10% Validation / 10% Testing             ║
# ║  Trained on COMBINED data (Online + Google Forms)                ║
# ╚══════════════════════════════════════════════════════════════════╝

import numpy     as np
import pandas    as pd
import streamlit as st
import scipy.stats as stats

from sklearn.linear_model    import LinearRegression, BayesianRidge
from sklearn.ensemble        import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_squared_error, r2_score

from config import (
    TIME_MAP, PURPOSE_MAP, DEVICE_MAP,
    FEATURE_COLS, SLOW_THRESHOLD, TIME_SLOTS,
)


# ── Feature Encoding ───────────────────────────────────────────────

def encodeFeatures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical text columns to numeric encodings required
    by the sklearn models.  Original columns are preserved alongside
    the new encoded columns.

    Parameters
    ----------
    df : pd.DataFrame
        Unified hostel dataset with TimeOfDay, Purpose, Device columns.

    Returns
    -------
    pd.DataFrame
        Copy of df with added columns: TimeEncoded, PurposeEncoded,
        DeviceEncoded.
    """
    enc = df.copy()
    enc["TimeEncoded"]    = enc["TimeOfDay"].map(TIME_MAP).fillna(1).astype(int)
    enc["PurposeEncoded"] = enc["Purpose"].map(PURPOSE_MAP).fillna(0).astype(int)
    enc["DeviceEncoded"]  = enc["Device"].map(DEVICE_MAP).fillna(0).astype(int)
    return enc


# ── Model Training ─────────────────────────────────────────────────

@st.cache_resource
def trainAllModels(df: pd.DataFrame):
    """
    Train all four NetSmart prediction models on the provided dataset
    using an 80/10/10 train/validation/test split.

    Models trained
    --------------
    1. Linear Regression     — simple & interpretable baseline
    2. Bayesian Ridge        — probabilistic, handles uncertainty
    3. Gradient Boosting     — most accurate ensemble method
    4. Parametric Statistical Model — group-wise Gaussian baseline

    Parameters
    ----------
    df : pd.DataFrame
        Combined (Online + Forms) unified dataset.

    Returns
    -------
    tuple
        (lr, br, gb, gbUL, metrics, parametric_model, split_info)
    """
    enc  = encodeFeatures(df)
    X    = enc[FEATURE_COLS]
    yDL  = enc["SpeedDownload"]
    yUL  = enc["SpeedUpload"]

    # ── 80 / 10 / 10 split ────────────────────────────────────────
    # First split: 80% train, 20% temp
    Xtr, Xtemp, yDLtr, yDLtemp = train_test_split(
        X, yDL, test_size=0.2, random_state=42
    )
    _,   Xtemp2, yULtr, yULtemp = train_test_split(
        X, yUL, test_size=0.2, random_state=42
    )
    # Second split: 50/50 of the 20% → 10% val + 10% test
    Xval, Xte, yDLval, yDLte = train_test_split(
        Xtemp, yDLtemp, test_size=0.5, random_state=42
    )
    _, _, yULval, yULte = train_test_split(
        Xtemp2, yULtemp, test_size=0.5, random_state=42
    )

    split_info = {
        "total":      len(df),
        "train":      len(Xtr),
        "validation": len(Xval),
        "test":       len(Xte),
    }

    # ── 1. Linear Regression (download) ──────────────────────────
    lr = LinearRegression()
    lr.fit(Xtr, yDLtr)
    lrPred = lr.predict(Xte)

    # ── 2. Bayesian Ridge (download) ─────────────────────────────
    br = BayesianRidge()
    br.fit(Xtr, yDLtr)
    brPred = br.predict(Xte)

    # ── 3. Gradient Boosting (download) ──────────────────────────
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(Xtr, yDLtr)
    gbPred = gb.predict(Xte)

    # ── 3b. Gradient Boosting (upload) ───────────────────────────
    gbUL = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, random_state=42)
    gbUL.fit(Xtr, yULtr)

    # ── 4. Parametric Statistical Model ──────────────────────────
    parametric_model = (
        df.groupby(["TimeOfDay", "Purpose", "Device"])
          .agg({"SpeedDownload": ["mean", "std"], "SpeedUpload": ["mean", "std"]})
          .fillna(df[["SpeedDownload", "SpeedUpload"]].mean())
    )
    
    global_mean = float(yDLtr.mean())
    def _get_p_pred(row):
        try:
            return float(parametric_model.loc[(row["TimeOfDay"], row["Purpose"], row["Device"]), ("SpeedDownload", "mean")])
        except Exception:
            return global_mean
            
    pPred = df.loc[Xte.index].apply(_get_p_pred, axis=1)
    pValPred = df.loc[Xval.index].apply(_get_p_pred, axis=1)

    # ── Validation metrics (for model selection / tuning) ─────────
    lrValPred = lr.predict(Xval)
    brValPred = br.predict(Xval)
    gbValPred = gb.predict(Xval)

    def _rmse(y_true, y_pred):
        return round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3)

    metrics = {
        "Linear Regression": {
            "RMSE":       _rmse(yDLte, lrPred),
            "R²":         round(r2_score(yDLte, lrPred), 3),
            "Val RMSE":   _rmse(yDLval, lrValPred),
        },
        "Bayesian Model": {
            "RMSE":       _rmse(yDLte, brPred),
            "R²":         round(r2_score(yDLte, brPred), 3),
            "Val RMSE":   _rmse(yDLval, brValPred),
        },
        "Gradient Boosting": {
            "RMSE":       _rmse(yDLte, gbPred),
            "R²":         round(r2_score(yDLte, gbPred), 3),
            "Val RMSE":   _rmse(yDLval, gbValPred),
        },
        "Parametric Stats": {
            "RMSE":       _rmse(yDLte, pPred),
            "R²":         round(r2_score(yDLte, pPred), 3),
            "Val RMSE":   _rmse(yDLval, pValPred),
        },
    }

    return lr, br, gb, gbUL, metrics, parametric_model, split_info


# ── Prediction Helper ──────────────────────────────────────────────

def predictSpeed(
    timeOfDay, purpose, device, numDevices, peakHour,
    lr, br, gb, gbUL, pModel,
):
    """
    Predict download and upload speeds for a given set of conditions
    using all four NetSmart models.

    Returns
    -------
    tuple
        (lrSpeed, brSpeed, gbSpeed, ulSpeed, pSpeed, isSlow, slowProb)
    """
    feats = pd.DataFrame([[
        TIME_MAP.get(timeOfDay, 0),
        PURPOSE_MAP.get(purpose, 0),
        DEVICE_MAP.get(device, 0),
        numDevices,
        peakHour,
    ]], columns=FEATURE_COLS)

    lrSpeed = max(0.0, round(float(lr.predict(feats)[0]),   2))
    brSpeed = max(0.0, round(float(br.predict(feats)[0]),   2))
    gbSpeed = max(0.0, round(float(gb.predict(feats)[0]),   2))
    ulSpeed = max(0.0, round(float(gbUL.predict(feats)[0]), 2))

    # ── Parametric lookup + Gaussian slow-probability ─────────────
    try:
        group_stats = pModel.loc[(timeOfDay, purpose, device)]
        pSpeed = float(group_stats[("SpeedDownload", "mean")])
        pStd   = float(group_stats[("SpeedDownload", "std")])
        if pStd == 0 or np.isnan(pStd):
            pStd = 5.0
        slowProb = float(stats.norm.cdf(SLOW_THRESHOLD, loc=pSpeed, scale=pStd))
    except Exception:
        pSpeed   = gbSpeed
        slowProb = 0.5 if gbSpeed < 12 else 0.1

    isSlow = 1 if slowProb > 0.5 else 0

    return lrSpeed, brSpeed, gbSpeed, ulSpeed, round(pSpeed, 2), isSlow, slowProb


# ── Best-Time Recommender ──────────────────────────────────────────

def recommendBestTime(purpose, device, numDevices, peakHour,
                      lr, br, gb, gbUL, pModel):
    """
    Run predictSpeed for all four time slots and return the slot with
    the highest predicted Gradient Boosting download speed.
    """
    results = {}
    for slot in TIME_SLOTS:
        _, _, speed, _, _, _, _ = predictSpeed(
            slot, purpose, device, numDevices, peakHour,
            lr, br, gb, gbUL, pModel,
        )
        results[slot] = speed

    bestSlot = max(results, key=results.get)
    return bestSlot, results


# ── Dataset Comparison Helper ──────────────────────────────────────

def compareDatasets(df: pd.DataFrame, df2: pd.DataFrame,
                    target_col: str = "SpeedDownload"):
    """
    Compute Z-scores for both datasets and run Welch's T-Test.
    """
    import scipy.stats as _stats
    s1 = df[target_col]
    s2 = df2[target_col]
    z1 = _stats.zscore(s1)
    z2 = _stats.zscore(s2)
    tStat, pVal = _stats.ttest_ind(s1, s2, equal_var=False)
    return z1, z2, round(float(tStat), 4), round(float(pVal), 5)
