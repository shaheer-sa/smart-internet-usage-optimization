# ╔══════════════════════════════════════════════════════════════════╗
# ║  views/page_home.py — NetSmart Home Page                        ║
# ╚══════════════════════════════════════════════════════════════════╝

import streamlit as st
import pandas    as pd


def render(df_combined, splitInfo, goto):
    """Render the Home page with abstracted total dataset stats."""

    # ── Hero Banner ───────────────────────────────────────────────
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-content">
            <h1 class="hero-title" style="font-size: 2.8rem !important; line-height: 1.2;">Smart Internet Usage Optimization</h1>
            <p class="hero-sub" style="max-width: 700px; margin: 10px auto 0;">
                Ever wondered <b>why your internet is slow at night</b>, or <b>what time is best to download
                that lecture video</b>? NetSmart uses <b>real hostel data</b> and smart AI to answer exactly that.
                No technical knowledge needed.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row (total / abstracted) ──────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Data Points",  f"{len(df_combined)}",                                "Total Records"),
        ("Avg Speed",    f"{df_combined['SpeedDownload'].mean():.1f} Mbps",    "Mean"),
        ("Slow %",       f"{df_combined['IsSlow'].mean()*100:.0f}%",           "Ratio"),
        ("AI Models",    "4",                                                  "Active"),
        ("Train Split",  f"{splitInfo['train']}",                              "80% Training"),
    ]
    for col, (icon, val, lbl) in zip([c1, c2, c3, c4, c5], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value">{val}</div>
                <div style="color: var(--text-muted); font-size: 0.8rem; margin-top: 6px; font-weight: 600;">{icon}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Data Source Info ──────────────────────────────────────────
    st.markdown(f"""
    <div class="info-card">
    <b>About the Data:</b> NetSmart is trained on <b>{len(df_combined)} real records</b> combining
    an online WiFi performance dataset with real hostel student survey responses.
    Models use an <b>80/10/10</b> split
    ({splitInfo['train']} train / {splitInfo['validation']} validation / {splitInfo['test']} test).
    Every speed check you run is automatically added to the dataset!
    </div>""", unsafe_allow_html=True)

    # ── Section Header ────────────────────────────────────────────
    st.markdown("## Click a Section to Get Started")

    # ── Navigation Cards — Row 1 ──────────────────────────────────
    r1c1, r1c2, r1c3 = st.columns(3)

    with r1c1:
        st.markdown("""
        <div class="nav-feature-card">
            <div class="card-icon">Speed</div>
            <div class="card-title">Check My Speed</div>
            <div class="card-desc">
                Run a live speed test right in the browser. Tell us your situation
                and our AI will predict what speed you should be getting. Your result
                is automatically saved to the dataset!
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Go to Speed Check", key="nav_speed"):
            goto("Check My Speed")

    with r1c2:
        st.markdown("""
        <div class="nav-feature-card">
            <div class="card-icon">Chart</div>
            <div class="card-title">Usage Insights</div>
            <div class="card-desc">
                See colourful charts answering everyday questions: When is the internet fastest?
                Which activity uses the most bandwidth? All charts are interactive!
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Go to Usage Insights", key="nav_insights"):
            goto("Usage Insights")

    with r1c3:
        st.markdown("""
        <div class="nav-feature-card">
            <div class="card-icon">AI</div>
            <div class="card-title">AI Model Report</div>
            <div class="card-desc">
                See how accurate each of our 4 AI engines is, with RMSE, R-squared, and
                validation scores. Plus probability distribution charts explained simply.
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Go to AI Models", key="nav_models"):
            goto("AI Model Report")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Navigation Cards — Row 2 ──────────────────────────────────
    r2c1, r2c2, r2c3 = st.columns(3)

    with r2c1:
        st.markdown("""
        <div class="nav-feature-card">
            <div class="card-icon">Data</div>
            <div class="card-title">Data Comparison</div>
            <div class="card-desc">
                Compare the online WiFi dataset against real Google Forms survey
                responses using Z-Score analysis, box plots, and Welch's T-Test.
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Go to Data Comparison", key="nav_compare"):
            goto("Data Comparison")

    with r2c2:
        st.markdown("""
        <div class="nav-feature-card">
            <div class="card-icon">Team</div>
            <div class="card-title">About</div>
            <div class="card-desc">
                Meet the team behind NetSmart. Learn about our roles, the technologies
                we used, and the academic context of this project.
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Go to About", key="nav_about"):
            goto("About")

    with r2c3:
        st.markdown("""
        <div class="nav-feature-card" style="cursor:default;">
            <div class="card-icon">INFO</div>
            <div class="card-title">Quick Tips</div>
            <div class="card-desc">
                - <b>Best time to download:</b> Early morning or late night<br><br>
                - <b>Fewer devices</b> connected = more speed per device<br><br>
                - <b>Avoid peak hours:</b> 5pm to 9pm sees the most congestion
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Sample Dataset Preview (combined, single table) ───────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### Sample Data Preview")
    st.markdown(f"Showing first 10 rows from the combined dataset ({len(df_combined)} total records).")

    styled = (
        df_combined.head(10).style
          .background_gradient(subset=["SpeedDownload"], cmap="Blues")
          .background_gradient(subset=["SpeedUpload"],   cmap="Purples")
          .format({"SpeedDownload": "{:.2f} Mbps", "SpeedUpload": "{:.2f} Mbps"})
    )
    st.dataframe(styled, use_container_width=True)
