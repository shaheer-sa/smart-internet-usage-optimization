# ╔══════════════════════════════════════════════════════════════════╗
# ║  views/page_model_report.py — NetSmart AI Model Report Page     ║
# ╚══════════════════════════════════════════════════════════════════╝

import numpy  as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import scipy.stats as stats

from config           import SLOW_THRESHOLD, PLOT_BG, FONT_DICT
from prob_stats_tools import (
    gaussianPdfCurve, gaussianIntervalMask, gaussianSlowProbability,
    poissonPmfCurve, poissonTailProbabilities,
    binomialPmfCurve, binomialExpected,
)


def render(df: pd.DataFrame, modelMetrics: dict, splitInfo: dict):
    """Render the AI Model Report page."""

    st.markdown("## AI Model Report - How Accurate Are Our Predictions?")
    st.markdown(f"""
    <div class="info-card">
    Models trained on <b>{splitInfo['total']} total records</b> (Online + Google Forms combined).<br>
    Split: <b>{splitInfo['train']} train</b> / <b>{splitInfo['validation']} validation</b> / <b>{splitInfo['test']} test</b> (80/10/10).
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Section 1: Descriptive Stats ──────────────────────────────
    st.markdown("### 1. Basic Speed Statistics")
    speed = df["SpeedDownload"]
    dc1, dc2, dc3, dc4, dc5 = st.columns(5)
    with dc1: st.metric("Mean Speed",    f"{speed.mean():.2f} Mbps")
    with dc2: st.metric("Std Deviation", f"{speed.std():.2f} Mbps")
    with dc3: st.metric("Variance",      f"{speed.var():.2f}")
    with dc4: st.metric("Median Speed",  f"{speed.median():.2f} Mbps")
    with dc5: st.metric("Speed Range",   f"{speed.max()-speed.min():.1f} Mbps")

    with st.expander("View Full Statistics Table"):
        st.dataframe(
            df[["NumberOfDevices", "SpeedDownload", "SpeedUpload"]]
              .describe().round(3)
              .style.background_gradient(cmap="Blues"),
            use_container_width=True,
        )

    # ── Section 2: Model Metrics ──────────────────────────────────
    st.markdown("---")
    st.markdown("### 2. How Accurate Is Each AI Model?")
    st.markdown("""
    <div class="info-card">
    <b>RMSE</b> = average error in Mbps (lower is better). <b>R-squared</b> = fraction of variance explained (closer to 1.0 is better).
    <b>Val RMSE</b> = error on the validation set (used for tuning, not final evaluation).
    </div>""", unsafe_allow_html=True)

    model_info = {
        "Linear Regression": {"icon": "LR", "plain": "Simple & Fast"},
        "Bayesian Model":    {"icon": "BR", "plain": "Probabilistic"},
        "Gradient Boosting": {"icon": "GB", "plain": "Most Accurate"},
        "Parametric Stats":  {"icon": "S",  "plain": "Pure Math"},
    }
    cols = st.columns(len(modelMetrics))
    for col, (modelName, mets) in zip(cols, modelMetrics.items()):
        info = model_info.get(modelName, {"icon": "?", "plain": ""})
        with col:
            st.markdown(f"""
            <div class="kpi-box">
                <div style="font-size:2rem; margin-bottom:6px;">{info['icon']}</div>
                <div style="font-size:0.95rem; font-weight:700; color:#3D2B1F;">{modelName}</div>
                <div style="font-size:0.75rem; color:#8B5E3C; font-weight:600; margin-bottom:14px;">{info['plain']}</div>
            </div>""", unsafe_allow_html=True)
            for k, v in mets.items():
                st.metric(k, v)

    # RMSE and R-squared comparison charts
    dl_models = ["Linear Regression", "Bayesian Model", "Gradient Boosting"]
    rmse_vals = [modelMetrics[m]["RMSE"] for m in dl_models]
    r2_vals   = [modelMetrics[m]["R²"]   for m in dl_models]
    m_names   = ["Linear\nRegression", "Bayesian\nModel", "Gradient\nBoosting"]

    compCol1, compCol2 = st.columns(2)
    with compCol1:
        figR = go.Figure(go.Bar(x=m_names, y=rmse_vals,
                                marker_color=["#B48464","#D4A373","#2D6A4F"],
                                text=[f"{v:.3f}" for v in rmse_vals], textposition="outside"))
        figR.update_layout(title="RMSE (Lower = Better)", yaxis_title="Error (Mbps)",
                           plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, font=FONT_DICT,
                           yaxis=dict(range=[0, max(rmse_vals)*1.35]))
        st.plotly_chart(figR, use_container_width=True, theme=None)

    with compCol2:
        figR2 = go.Figure(go.Bar(x=m_names, y=r2_vals,
                                 marker_color=["#B48464","#D4A373","#2D6A4F"],
                                 text=[f"{v:.3f}" for v in r2_vals], textposition="outside"))
        figR2.update_layout(title="R-squared (Closer to 1.0 = Better)", yaxis_title="R-squared",
                            plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, font=FONT_DICT,
                            yaxis=dict(range=[0, 1.15]))
        st.plotly_chart(figR2, use_container_width=True, theme=None)

    best_model = min(dl_models, key=lambda m: modelMetrics[m]["RMSE"])
    st.markdown(f"""
    <div class="meaning-box">
    <div class="mb-title">Winner: {best_model}</div>
    Lowest RMSE of {modelMetrics[best_model]['RMSE']} Mbps and R-squared of {modelMetrics[best_model]['R\u00b2']}.
    </div>""", unsafe_allow_html=True)

    # ── Section 3: Probability Distributions ──────────────────────
    st.markdown("---")
    st.markdown("### 3. Probability Distributions")

    dTab1, dTab2, dTab3 = st.tabs(["Normal Distribution", "Poisson Distribution", "Binomial Distribution"])

    with dTab1:
        st.markdown("#### Normal (Gaussian) Distribution - Speed Uncertainty")
        mu, sigma = float(speed.mean()), float(speed.std())
        xN, yN = gaussianPdfCurve(mu, sigma)
        mask   = gaussianIntervalMask(xN, mu, sigma)
        figN = go.Figure()
        figN.add_trace(go.Scatter(x=xN, y=yN, mode="lines", fill="tozeroy",
                                   fillcolor="rgba(139,94,60,0.08)",
                                   line=dict(color="#8B5E3C", width=3), name="Bell Curve"))
        figN.add_trace(go.Scatter(
            x=np.concatenate([xN[mask], xN[mask][::-1]]),
            y=np.concatenate([yN[mask], np.zeros(mask.sum())]),
            fill="toself", fillcolor="rgba(255,183,3,0.22)", line=dict(width=0), name="68% region"))
        figN.add_vline(x=mu, line_dash="dash", line_color="#000000",
                       annotation_text=f"Mean = {mu:.1f}", annotation_font_color="#000000")
        figN.add_vline(x=SLOW_THRESHOLD, line_dash="dash", line_color="#9B2226",
                       annotation_text="Slow Threshold", annotation_font_color="#9B2226")
        figN.update_layout(title=f"Bell Curve: Mean={mu:.1f}, Std={sigma:.1f}",
                           plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, font=FONT_DICT,
                           legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(figN, use_container_width=True, theme=None)
        slow_pct = gaussianSlowProbability(mu, sigma) * 100
        st.markdown(f"""
        <div class="meaning-box">
        <b>{slow_pct:.1f}%</b> of sessions expected below {SLOW_THRESHOLD} Mbps threshold.
        68% of sessions fall between <b>{mu-sigma:.1f}</b> and <b>{mu+sigma:.1f}</b> Mbps.
        </div>""", unsafe_allow_html=True)

    with dTab2:
        st.markdown("#### Poisson Distribution - Device Load Modelling")
        lambdaVal = float(df["NumberOfDevices"].mean())
        kVals, pPmf = poissonPmfCurve(lambdaVal, k_max=int(min(600, lambdaVal*3)))
        poiStats = poissonTailProbabilities(lambdaVal, k_max=int(min(600, lambdaVal*3)))
        figPo = go.Figure(go.Bar(x=kVals, y=pPmf, marker_color="#D4A373", opacity=0.85))
        figPo.add_vline(x=lambdaVal, line_dash="dash", line_color="#000000",
                        annotation_text=f"Avg = {lambdaVal:.0f}", annotation_font_color="#000000")
        figPo.update_layout(title=f"Poisson: Probability of k Devices (lambda = {lambdaVal:.0f})",
                            xaxis_title="Number of Devices", yaxis_title="Probability",
                            plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, font=FONT_DICT)
        st.plotly_chart(figPo, use_container_width=True, theme=None)
        st.markdown(f"""
        <div class="meaning-box">
        Peak at <b>{poiStats['peak_k']}</b> devices. Average lambda = {lambdaVal:.0f}.
        </div>""", unsafe_allow_html=True)

    with dTab3:
        st.markdown("#### Binomial Distribution - Weekly Slow Session Count")
        pSlow = float(df["IsSlow"].mean())
        nTrials = 50
        kBin, binPmf = binomialPmfCurve(nTrials, pSlow)
        expected = binomialExpected(nTrials, pSlow)
        figBi = go.Figure(go.Bar(
            x=kBin, y=binPmf,
            marker_color=["#2D6A4F" if abs(k - expected) <= 3 else "#D4A373" for k in kBin],
            opacity=0.85))
        figBi.add_vline(x=expected, line_dash="dash", line_color="#000000",
                        annotation_text=f"Expected: {expected:.1f}", annotation_font_color="#000000")
        figBi.update_layout(title=f"Binomial: Slow Sessions in 50 Trials (p={pSlow:.2f})",
                            xaxis_title="Slow Sessions Out of 50", yaxis_title="Probability",
                            plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, font=FONT_DICT)
        st.plotly_chart(figBi, use_container_width=True, theme=None)
        st.markdown(f"""
        <div class="meaning-box">
        With {pSlow*100:.0f}% slow rate, expect <b>{expected:.0f} slow sessions</b> out of every 50.
        </div>""", unsafe_allow_html=True)
