# ╔══════════════════════════════════════════════════════════════════╗
# ║  views/page_speed_check.py — NetSmart Speed Check Page          ║
# ║  Uses a bidirectional Streamlit component so the JS speed test  ║
# ║  automatically sends measured values back to Python.            ║
# ╚══════════════════════════════════════════════════════════════════╝

import pathlib
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import pandas as pd

from config           import SLOW_THRESHOLD, PLOT_BG, FONT_DICT, PURPOSES, DEVICES
from model_trainer    import predictSpeed, recommendBestTime
from data_loader      import appendSubmission
from prob_stats_tools import (
    gaussianSlowProbability,
    poissonInstantProbability,
    binomialWeeklyReliability,
)

# ── Declare the bidirectional speed-test component ────────────────
_COMP_DIR = pathlib.Path(__file__).parent / "speed_test_component"
_speed_test_component = components.declare_component("netsmart_speedtest", path=str(_COMP_DIR))


def render(df, lr, br, gb, gbUL, pModel, modelMetrics):
    """Render the Check My Speed page."""

    st.markdown("## Check My Internet Speed")
    st.markdown("""
    <div class="info-card">
    <b>How this works:</b> Run the speed test below — your download and upload speeds are
    <b>captured automatically</b>. Then fill in your situation and click Generate Report.
    Your result is saved to the dataset!
    </div>
    """, unsafe_allow_html=True)

    # ── Equal-height two-column layout ────────────────────────────
    colSpeed, colForm = st.columns(2, gap="large")

    with colSpeed:
        st.markdown("### Step 1 — Run Speed Test")
        # Bidirectional component — returns {dl: X, ul: Y} after test
        speed_result = _speed_test_component(key="speed_test_result", default=None)

        # Store results in session state
        if speed_result and isinstance(speed_result, dict):
            st.session_state.measured_dl = speed_result.get("dl", 0.0)
            st.session_state.measured_ul = speed_result.get("ul", 0.0)

        # Show captured values
        actualDL = st.session_state.get("measured_dl", 0.0)
        actualUL = st.session_state.get("measured_ul", 0.0)

        if actualDL > 0:
            st.success(f"Captured: **{actualDL:.1f} Mbps** download / **{actualUL:.1f} Mbps** upload")
        else:
            st.info("Click START FULL TEST above. Results will appear here automatically.")

    with colForm:
        st.markdown("### Step 2 — Describe Your Situation")
        timeOfDay = st.selectbox(
            "What time is it right now?",
            ["Morning", "Afternoon", "Evening", "Night"],
            help="Morning = 6am-12pm | Afternoon = 12-5pm | Evening = 5-9pm | Night = 9pm-6am",
        )
        purpose = st.selectbox("What are you doing online?", PURPOSES)
        device  = st.selectbox("What device are you using?", DEVICES)
        numDevices = st.slider(
            "Estimated total devices on the network",
            50, 550, 200, step=50,
        )
        peakHour = 1 if timeOfDay in ["Evening", "Night"] else 0
        st.markdown(f"<div class='tag'>Peak Hour: {'Yes' if peakHour else 'No'}</div>",
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        runPred = st.button("Generate My Internet Report", use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    if runPred:
        lrSpeed, brSpeed, gbSpeed, gbULSpeed, pSpeed, isSlow, slowProb = predictSpeed(
            timeOfDay, purpose, device, numDevices, peakHour,
            lr, br, gb, gbUL, pModel,
        )

        # ── Save submission to live CSV ───────────────────────────
        newRow = {
            "TimeOfDay":      timeOfDay,
            "Purpose":        purpose,
            "Device":         device,
            "NumberOfDevices": numDevices,
            "SpeedDownload":  actualDL if actualDL > 0 else gbSpeed,
            "SpeedUpload":    actualUL if actualUL > 0 else gbULSpeed,
            "PeakHour":       peakHour,
            "IsSlow":         1 if (actualDL < SLOW_THRESHOLD and actualDL > 0) else (1 if gbSpeed < SLOW_THRESHOLD else 0),
        }
        appendSubmission(newRow)
        st.toast("Your session has been saved to the Google Forms dataset!")

        # ── Connection quality badge ──────────────────────────────
        st.markdown("## Your Internet Report")
        if isSlow == 0:
            quality_html = '<span class="badge-good">Connection: GOOD</span>'
            quality_msg  = "Your connection quality is acceptable for most activities."
        elif slowProb > 0.75:
            quality_html = '<span class="badge-bad">Connection: SLOW</span>'
            quality_msg  = "Your connection is likely congested. Try again later."
        else:
            quality_html = '<span class="badge-warn">Connection: MARGINAL</span>'
            quality_msg  = "Your connection is borderline. Heavy tasks may struggle."

        st.markdown(quality_html + f" &nbsp; {quality_msg}", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Model predictions ─────────────────────────────────────
        res1, res2, res3, res4 = st.columns(4)
        with res1: st.metric("Linear Regression", f"{lrSpeed:.1f} Mbps")
        with res2: st.metric("Bayesian Model",     f"{brSpeed:.1f} Mbps")
        with res3: st.metric("Gradient Boosting",  f"{gbSpeed:.1f} Mbps")
        with res4: st.metric("Parametric Stats",   f"{pSpeed:.1f} Mbps")

        st.markdown(f"""
        <div class="meaning-box">
        <div class="mb-title">4-Model Comparison Report</div>
        Predicted speed using all 4 models. <b>Gradient Boosting</b> is typically the most accurate.
        Chance of experiencing lag: <b>{slowProb*100:.0f}%</b>.
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Predicted vs Actual bar chart ─────────────────────────
        devCol1, devCol2 = st.columns(2)
        with devCol1:
            figDev = go.Figure()
            figDev.add_trace(go.Bar(name="AI Predicted", x=["Download", "Upload"],
                                     y=[gbSpeed, gbULSpeed], marker_color="#8B5E3C"))
            figDev.add_trace(go.Bar(name="Your Measured", x=["Download", "Upload"],
                                     y=[actualDL, actualUL], marker_color="#3D2B1F"))
            figDev.update_layout(barmode="group", title="Predicted vs. Measured (Mbps)",
                                 height=340, margin=dict(t=50, b=40, l=30, r=30),
                                 plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, font=FONT_DICT)
            st.plotly_chart(figDev, use_container_width=True, theme=None)

        with devCol2:
            if actualDL > 0:
                diffDL = actualDL - gbSpeed
                st.markdown("#### Performance Gap")
                st.write(f"**Download:** {diffDL:+.1f} Mbps vs AI Prediction")
                if actualDL > gbSpeed * 1.1:
                    st.success("Your connection is performing **better than expected**!")
                elif abs(diffDL) <= 5:
                    st.info("Your connection is **right on target**.")
                else:
                    st.warning("Your speed is **lower than expected**.")
            else:
                st.info("Run the speed test first to see the full comparison!")

        # ── Slow probability gauge ────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Probability: Will Your Internet Feel Slow?")

        gaugeCol, textCol = st.columns(2)
        with gaugeCol:
            figGauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(slowProb * 100, 1),
                number={"suffix": "%", "font": {"size": 36, "color": "#3D2B1F"}},
                title={"text": "Chance of Slow Internet", "font": {"size": 15, "color": "#6B4F3B"}},
                gauge={
                    "axis":  {"range": [0, 100]},
                    "bar":   {"color": "#D4A373" if slowProb < 0.5 else "#9B2226"},
                    "steps": [
                        {"range": [0,   33],  "color": "#D8E2DC"},
                        {"range": [33,  66],  "color": "#FEF3C7"},
                        {"range": [66, 100],  "color": "#FECACA"},
                    ],
                    "threshold": {"line": {"color": "#3D2B1F", "width": 3}, "value": 50},
                },
            ))
            figGauge.update_layout(height=260, margin=dict(t=30, b=10, l=10, r=10),
                                   paper_bgcolor=PLOT_BG, font=FONT_DICT)
            st.plotly_chart(figGauge, use_container_width=True, theme=None)

        with textCol:
            rmse_val = modelMetrics["Gradient Boosting"]["RMSE"]
            prob_slow_gaussian = gaussianSlowProbability(gbSpeed, rmse_val)
            avg_devices = df["NumberOfDevices"].mean()
            prob_devices_poisson = poissonInstantProbability(numDevices, avg_devices)
            prob_3_slow_week = binomialWeeklyReliability(slowProb, n_days=7, min_slow_days=3)

            st.markdown(f"""
            <div class="insight-card" style="margin-top:20px; border-left: 5px solid #8B5E3C;">
            <div class="insight-title" style="font-size:1.1rem;">Mathematical Breakdown</div>
            <div style="font-size:0.88rem; line-height:1.6;">
            <b>1. Gaussian:</b> Mean={gbSpeed:.1f}, sigma={rmse_val:.1f} ->
            <b>{prob_slow_gaussian*100:.1f}%</b> below {SLOW_THRESHOLD} Mbps.<br><br>
            <b>2. Poisson:</b> Avg devices={avg_devices:.0f}. Your {numDevices} devices:
            <b>{prob_devices_poisson*100:.2f}%</b> probability.<br><br>
            <b>3. Binomial:</b> Over 7 days, <b>{prob_3_slow_week*100:.1f}%</b> chance of 3+ slow sessions.
            </div></div>""", unsafe_allow_html=True)

        # ── Best time recommendation ──────────────────────────────
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("### Best Time to Go Online Today")

        bestSlot, allSlots = recommendBestTime(
            purpose, device, numDevices, peakHour, lr, br, gb, gbUL, pModel)

        rec1, rec2 = st.columns(2)
        with rec1:
            figRec = go.Figure(go.Bar(
                x=list(allSlots.keys()), y=list(allSlots.values()),
                marker_color=["#2D6A4F" if s == bestSlot else "#D4A373" for s in allSlots.keys()],
                text=[f"{v:.1f}" for v in allSlots.values()], textposition="outside"))
            figRec.add_hline(y=SLOW_THRESHOLD, line_dash="dash", line_color="#9B2226",
                             annotation_text="Min acceptable")
            figRec.update_layout(title="Predicted Speed by Time of Day", yaxis_title="Speed (Mbps)",
                                 height=320, plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, font=FONT_DICT,
                                 margin=dict(t=50, b=30, l=30, r=30),
                                 yaxis=dict(range=[0, max(allSlots.values()) * 1.3]))
            st.plotly_chart(figRec, use_container_width=True, theme=None)

        with rec2:
            st.markdown(f"""
            <div class="insight-card" style="margin-top:20px;">
            <div class="insight-title" style="font-size:1.15rem;">Best Time: {bestSlot}</div>
            <p style="font-size:0.9rem;">
            For <b>{purpose}</b> on <b>{device}</b>, the <b>{bestSlot}</b> slot gives the fastest
            predicted speed of <b>{allSlots[bestSlot]:.1f} Mbps</b>.
            </p></div>""", unsafe_allow_html=True)
