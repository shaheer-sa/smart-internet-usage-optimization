# ╔══════════════════════════════════════════════════════════════════╗
# ║  views/page_comparison.py — NetSmart Data Comparison Page       ║
# ║  Compares Online WiFi dataset vs Google Forms survey data       ║
# ╚══════════════════════════════════════════════════════════════════╝

import pandas    as pd
import streamlit as st
import plotly.express       as px
import plotly.graph_objects as go

from config        import SLOW_THRESHOLD, PLOT_BG, FONT_DICT
from model_trainer import compareDatasets


def render(df_online: pd.DataFrame, df_forms: pd.DataFrame):
    """Render the Data Comparison page: Online Dataset vs Google Forms."""

    st.markdown("## Data Comparison - Online Dataset vs Google Forms")

    var_to_compare = st.radio("Select metric to compare:",
                              ["SpeedDownload", "SpeedUpload"], horizontal=True)

    st.markdown(f"""
    <div class="info-card">
    Comparing <b>{var_to_compare}</b> across two real data sources:<br>
    <b>Dataset 1 (Online):</b> {len(df_online)} curated WiFi performance records.<br>
    <b>Dataset 2 (Google Forms):</b> {len(df_forms)} real hostel student survey responses (grows live as users submit).
    </div>""", unsafe_allow_html=True)

    z1, z2, tStat, pVal = compareDatasets(df_online, df_forms, target_col=var_to_compare)

    # ── Summary Metrics ───────────────────────────────────────────
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("#### Dataset 1 - Online WiFi Data")
        d1a, d1b, d1c = st.columns(3)
        with d1a: st.metric(f"Avg {var_to_compare}", f"{df_online[var_to_compare].mean():.2f} Mbps")
        with d1b: st.metric("Std Deviation",          f"{df_online[var_to_compare].std():.2f} Mbps")
        with d1c: st.metric("Records",                len(df_online))
    with cc2:
        st.markdown("#### Dataset 2 - Google Forms Surveys")
        d2a, d2b, d2c = st.columns(3)
        with d2a: st.metric(f"Avg {var_to_compare}", f"{df_forms[var_to_compare].mean():.2f} Mbps")
        with d2b: st.metric("Std Deviation",          f"{df_forms[var_to_compare].std():.2f} Mbps")
        with d2c: st.metric("Responses",              len(df_forms))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Z-Score Histogram ─────────────────────────────────────────
    st.markdown("### Step 1 - Z-Score Analysis")
    st.markdown("""
    <div class="info-card">
    <b>Z-Score</b> = how far a value is from the dataset average, in units of standard deviation.
    Z = 0 means average. Z = +2 means unusually fast. Z = -2 means unusually slow.
    </div>""", unsafe_allow_html=True)

    figZ = go.Figure()
    figZ.add_trace(go.Histogram(x=z1, name="Online Dataset", marker_color="#8B5E3C",
                                 opacity=0.72, nbinsx=30))
    figZ.add_trace(go.Histogram(x=z2, name="Google Forms", marker_color="#D4A373",
                                 opacity=0.72, nbinsx=20))
    figZ.add_vline(x=0, line_dash="dash", line_color="#000000",
                   annotation_text="Z = 0 (Average)", annotation_font_color="#000000")
    figZ.update_layout(barmode="overlay", title="Z-Score Distribution Overlay",
                       plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, font=FONT_DICT,
                       legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(figZ, use_container_width=True, theme=None)

    # ── Box Plot ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Step 2 - Box Plot Comparison")

    combined = pd.DataFrame({
        "Speed (Mbps)": list(df_online[var_to_compare]) + list(df_forms[var_to_compare]),
        "Dataset": (["Online WiFi Data"] * len(df_online) +
                    ["Google Forms Surveys"] * len(df_forms)),
    })
    figBox = px.box(combined, x="Dataset", y="Speed (Mbps)", color="Dataset",
                    color_discrete_map={"Online WiFi Data": "#8B5E3C",
                                        "Google Forms Surveys": "#D4A373"},
                    title=f"{var_to_compare} Comparison", points="outliers")
    figBox.add_hline(y=SLOW_THRESHOLD, line_dash="dash", line_color="#9B2226",
                     annotation_text="Slow threshold")
    figBox.update_layout(plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, font=FONT_DICT,
                         showlegend=False)
    st.plotly_chart(figBox, use_container_width=True, theme=None)

    # ── T-Test Results ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Step 3 - Welch's T-Test")
    st.markdown("""
    <div class="info-card">
    <b>T-Test</b> answers: "Is the difference between these two datasets real, or just random noise?"<br>
    If <b>P-Value &lt; 0.05</b>, the difference is statistically significant (real).
    </div>""", unsafe_allow_html=True)

    tc1, tc2 = st.columns(2)
    with tc1: st.metric("T-Statistic", tStat)
    with tc2: st.metric("P-Value", pVal)

    if pVal < 0.05:
        st.markdown(f"""
        <div class="insight-card" style="border-color:#9B2226; border-width:2px;">
        <div class="insight-title">Result: Statistically Significant Difference (p = {pVal})</div>
        <p style="font-size:0.9rem;">
        The online dataset and Google Forms survey show <b>meaningfully different speed patterns</b>.
        This is expected since the online data comes from a different measurement environment than
        real hostel student perceptions. Both datasets are used together for training to capture
        the widest range of conditions.
        </p></div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-card" style="border-color:#2D6A4F; border-width:2px;">
        <div class="insight-title">Result: No Significant Difference (p = {pVal})</div>
        <p style="font-size:0.9rem;">
        Both datasets show <b>statistically similar speed distributions</b>. This validates
        that the online dataset is a fair representation of real hostel conditions.
        </p></div>""", unsafe_allow_html=True)
