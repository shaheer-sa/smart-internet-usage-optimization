# ╔══════════════════════════════════════════════════════════════════╗
# ║  views/page_insights.py — NetSmart Usage Insights Page          ║
# ║  Five interactive Plotly tabs + correlation matrix               ║
# ╚══════════════════════════════════════════════════════════════════╝

import numpy  as np
import pandas as pd
import streamlit as st
import plotly.express       as px
import plotly.graph_objects as go
import scipy.stats as stats

from config        import SLOW_THRESHOLD, PLOT_BG, FONT_DICT, THEME_COLORS
from model_trainer import encodeFeatures


def render(df: pd.DataFrame):
    """Render the Usage Insights page (trained on combined dataset)."""

    st.markdown("## Usage Insights - Visual Data Explorer")
    st.markdown(f"""
    <div class="info-card">
    Analysing <b>{len(df)} total records</b> from the combined Online + Google Forms datasets.
    Every chart is interactive. Hover, click, and zoom to explore!
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Speed Distribution", "Usage Breakdown", "Speed vs Time",
        "Devices vs Speed", "Activity Heatmap",
    ])

    # ── Tab 1: Speed Histograms ───────────────────────────────────
    with tab1:
        cHist1, cHist2 = st.columns(2)
        with cHist1:
            st.markdown("#### Download Speed Distribution")
            muDL, sigmaDL = df["SpeedDownload"].mean(), df["SpeedDownload"].std()
            xFit = np.linspace(df["SpeedDownload"].min(), df["SpeedDownload"].max(), 300)
            yFit = stats.norm.pdf(xFit, muDL, sigmaDL)
            binW = (df["SpeedDownload"].max() - df["SpeedDownload"].min()) / 30
            figH = go.Figure()
            figH.add_trace(go.Histogram(x=df["SpeedDownload"], nbinsx=30,
                                         marker_color="#3D2B1F", opacity=0.8, name="Sessions"))
            figH.add_trace(go.Scatter(x=xFit, y=yFit * len(df) * binW, mode="lines",
                                       line=dict(color="#D4A373", width=3), name="Normal Fit"))
            figH.update_layout(title="Download Frequency", plot_bgcolor=PLOT_BG,
                               paper_bgcolor=PLOT_BG, font=FONT_DICT, showlegend=False, height=350)
            st.plotly_chart(figH, use_container_width=True, theme=None)

        with cHist2:
            st.markdown("#### Upload Speed Distribution")
            muUL, sigmaUL = df["SpeedUpload"].mean(), df["SpeedUpload"].std()
            xFitU = np.linspace(df["SpeedUpload"].min(), df["SpeedUpload"].max(), 300)
            yFitU = stats.norm.pdf(xFitU, muUL, sigmaUL)
            binWU = (df["SpeedUpload"].max() - df["SpeedUpload"].min()) / 30
            figHU = go.Figure()
            figHU.add_trace(go.Histogram(x=df["SpeedUpload"], nbinsx=30,
                                          marker_color="#8B5E3C", opacity=0.8, name="Sessions"))
            figHU.add_trace(go.Scatter(x=xFitU, y=yFitU * len(df) * binWU, mode="lines",
                                        line=dict(color="#D4A373", width=3), name="Normal Fit"))
            figHU.update_layout(title="Upload Frequency", plot_bgcolor=PLOT_BG,
                                paper_bgcolor=PLOT_BG, font=FONT_DICT, showlegend=False, height=350)
            st.plotly_chart(figHU, use_container_width=True, theme=None)

        st.markdown(f"""
        <div class="meaning-box">
        <div class="mb-title">What does this mean?</div>
        Average Download: <b>{muDL:.1f} Mbps</b> | Average Upload: <b>{muUL:.1f} Mbps</b>
        </div>""", unsafe_allow_html=True)

    # ── Tab 2: Pie Charts ─────────────────────────────────────────
    with tab2:
        st.markdown("### What Are Students Actually Using the Internet For?")
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            pcP = df["Purpose"].value_counts()
            figP = px.pie(values=pcP.values, names=pcP.index, title="Activity Breakdown",
                          color_discrete_sequence=THEME_COLORS, hole=0.35)
            figP.update_traces(textinfo="label+percent", textposition="outside")
            figP.update_layout(paper_bgcolor=PLOT_BG, font=FONT_DICT, showlegend=False, height=380)
            st.plotly_chart(figP, use_container_width=True, theme=None)

        with r1c2:
            pcS = df["IsSlow"].value_counts().sort_index()
            names_m = []
            if 0 in pcS.index: names_m.append(f"Fast (>={SLOW_THRESHOLD} Mbps)")
            if 1 in pcS.index: names_m.append(f"Slow (<{SLOW_THRESHOLD} Mbps)")
            figS = px.pie(values=pcS.values, names=names_m, title="Fast vs Slow Sessions",
                          color_discrete_sequence=["#2D6A4F", "#9B2226"], hole=0.35)
            figS.update_traces(textinfo="label+percent", textposition="outside")
            figS.update_layout(paper_bgcolor=PLOT_BG, font=FONT_DICT, showlegend=False, height=380)
            st.plotly_chart(figS, use_container_width=True, theme=None)

        st.markdown("<br>", unsafe_allow_html=True)
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            pcT = df["TimeOfDay"].value_counts().reindex(["Morning","Afternoon","Evening","Night"])
            figT = px.pie(values=pcT.values, names=pcT.index, title="When Are Students Online?",
                          color_discrete_sequence=["#8B5E3C","#D4A373","#9B2226","#3D2B1F"], hole=0.35)
            figT.update_traces(textinfo="label+percent", textposition="outside")
            figT.update_layout(paper_bgcolor=PLOT_BG, font=FONT_DICT, showlegend=False, height=380)
            st.plotly_chart(figT, use_container_width=True, theme=None)

        with r2c2:
            pcD = df["Device"].value_counts()
            figD = px.pie(values=pcD.values, names=pcD.index, title="Which Devices Are Used?",
                          color_discrete_sequence=["#D4A373","#3D2B1F"], hole=0.35)
            figD.update_traces(textinfo="label+percent", textposition="outside")
            figD.update_layout(paper_bgcolor=PLOT_BG, font=FONT_DICT, showlegend=False, height=380)
            st.plotly_chart(figD, use_container_width=True, theme=None)

    # ── Tab 3: Speed vs Time ──────────────────────────────────────
    with tab3:
        st.markdown("### How Does Speed Change Throughout the Day?")
        ORDER = ["Morning", "Afternoon", "Evening", "Night"]
        avgDL = df.groupby("TimeOfDay")["SpeedDownload"].mean().reindex(ORDER)
        avgUL = df.groupby("TimeOfDay")["SpeedUpload"].mean().reindex(ORDER)
        figL = go.Figure()
        figL.add_trace(go.Scatter(x=ORDER, y=avgDL, mode="lines+markers", name="Download Avg",
                                   line=dict(color="#3D2B1F", width=4), marker=dict(size=10)))
        figL.add_trace(go.Scatter(x=ORDER, y=avgUL, mode="lines+markers", name="Upload Avg",
                                   line=dict(color="#8B5E3C", width=3, dash="dot"), marker=dict(size=8)))
        figL.add_hline(y=SLOW_THRESHOLD, line_dash="dash", line_color="#9B2226",
                       annotation_text="Slow threshold")
        figL.update_layout(title="Average Speed by Time of Day",
                           plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, font=FONT_DICT,
                           legend=dict(orientation="h", y=1.1), margin=dict(t=80, b=40))
        st.plotly_chart(figL, use_container_width=True, theme=None)

        best_t, worst_t = avgDL.idxmax(), avgDL.idxmin()
        st.markdown(f"""
        <div class="meaning-box">
        <div class="mb-title">Key Finding</div>
        Best time: <b>{best_t}</b> ({avgDL.max():.1f} Mbps) | Worst: <b>{worst_t}</b> ({avgDL.min():.1f} Mbps)
        </div>""", unsafe_allow_html=True)

    # ── Tab 4: Scatter Plot ───────────────────────────────────────
    with tab4:
        st.markdown("### Does More Devices = Slower Internet?")
        figSc = px.scatter(
            df, x="NumberOfDevices", y="SpeedDownload", color="TimeOfDay",
            hover_data={"Purpose": True, "Device": True, "SpeedDownload": ":.2f"},
            color_discrete_map={"Morning":"#8B5E3C","Afternoon":"#D4A373",
                                "Evening":"#9B2226","Night":"#3D2B1F"},
            labels={"SpeedDownload":"Speed (Mbps)", "NumberOfDevices":"Total Devices"},
            opacity=0.75,
        )
        figSc.add_hline(y=SLOW_THRESHOLD, line_dash="dot", line_color="#9B2226",
                        annotation_text="Slow threshold")
        figSc.update_layout(plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, font=FONT_DICT,
                            legend=dict(orientation="h", y=-0.25), margin=dict(t=50, b=80))
        st.plotly_chart(figSc, use_container_width=True, theme=None)

        corr = df[["NumberOfDevices", "SpeedDownload"]].corr().iloc[0, 1]
        st.markdown(f"""
        <div class="meaning-box">
        <div class="mb-title">Correlation: {corr:.2f}</div>
        {'Negative correlation confirms: more devices = slower speeds.' if corr < 0 else 'Weak or positive correlation suggests device count alone is not the main bottleneck in this dataset.'}
        </div>""", unsafe_allow_html=True)

    # ── Tab 5: Heatmap ────────────────────────────────────────────
    with tab5:
        st.markdown("### Which Activity + Time Combination Is Worst?")
        pivot = (df.pivot_table(values="SpeedDownload", index="Purpose",
                                columns="TimeOfDay", aggfunc="mean")
                   .round(1).reindex(columns=["Morning","Afternoon","Evening","Night"]))
        figHM = px.imshow(pivot, color_continuous_scale="RdYlGn", text_auto=True,
                          title="Average Speed (Mbps) by Activity x Time", aspect="auto")
        figHM.update_traces(textfont_size=14)
        figHM.update_layout(paper_bgcolor=PLOT_BG, font=FONT_DICT, height=400)
        st.plotly_chart(figHM, use_container_width=True, theme=None)

    # ── Correlation Matrix ────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    with st.expander("Advanced: Correlation Matrix"):
        enc = encodeFeatures(df)
        corr_cols = ["TimeEncoded", "NumberOfDevices", "PeakHour",
                     "DeviceEncoded", "SpeedDownload"]
        corr_m = enc[corr_cols].corr().round(2)
        display_names = {"TimeEncoded":"Time of Day", "NumberOfDevices":"Total Devices",
                         "PeakHour":"Peak Hour", "DeviceEncoded":"Device Type",
                         "SpeedDownload":"Download Speed"}
        corr_m = corr_m.rename(index=display_names, columns=display_names)
        figC = px.imshow(corr_m, text_auto=True, color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, title="Variable Correlation Matrix", aspect="auto")
        figC.update_layout(paper_bgcolor=PLOT_BG, font=FONT_DICT, height=380)
        st.plotly_chart(figC, use_container_width=True, theme=None)
