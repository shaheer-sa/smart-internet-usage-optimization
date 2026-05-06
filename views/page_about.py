# ╔══════════════════════════════════════════════════════════════════╗
# ║  views/page_about.py — NetSmart About / Team Page               ║
# ╚══════════════════════════════════════════════════════════════════╝

import streamlit as st


def render():
    """Render the About page with team member information."""

    st.markdown("""
    <div class="hero-wrap" style="padding:40px 30px;">
        <div class="hero-title" style="font-size:2.2rem;">About NetSmart</div>
        <p class="hero-sub" style="max-width:720px;">
            NetSmart is a university project that combines <b>Artificial Intelligence</b>,
            <b>Probability Theory</b>, and <b>Statistical Analysis</b> to understand and predict
            hostel internet performance. Built with real data collected from students.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## The Team Behind NetSmart")
    st.markdown("""
    <div class="info-card" style="margin-bottom:24px;">
    This project was developed as part of our <b>4th Semester Probability & Statistics</b> course.
    Each member contributed a specialized skill set to build a complete, professional-grade
    data science application from the ground up.
    </div>
    """, unsafe_allow_html=True)

    # ── Team Members ──────────────────────────────────────────────

    # Row 1
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("""
        <div class="nav-feature-card" style="cursor:default; border-left:5px solid #3D2B1F;">
            <div style="font-size:0.7rem; color:#8B5E3C; font-weight:700; letter-spacing:2px;
                        text-transform:uppercase; margin-bottom:6px;">Project Leader</div>
            <div class="card-title" style="margin-bottom:10px;">Shaheer Ahmad</div>
            <div class="card-desc" style="line-height:1.8;">
                <b>Role:</b> AI Models Implementation & Training<br><br>
                Led the development and implementation of all four machine learning models
                (Linear Regression, Bayesian Ridge, Gradient Boosting, and Parametric Statistical Model).
                Responsible for data analysis, speed prediction algorithms, model evaluation metrics
                (RMSE, R-Squared), and the 80/10/10 train/validation/test pipeline.
                Managed the overall project architecture and code integration.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r1c2:
        st.markdown("""
        <div class="nav-feature-card" style="cursor:default; border-left:5px solid #8B5E3C;">
            <div style="font-size:0.7rem; color:#8B5E3C; font-weight:700; letter-spacing:2px;
                        text-transform:uppercase; margin-bottom:6px;">Statistical Analysis</div>
            <div class="card-title" style="margin-bottom:10px;">Shahan Haider</div>
            <div class="card-desc" style="line-height:1.8;">
                <b>Role:</b> Statistical Tools & Data Comparison<br><br>
                Applied and implemented the three core probability distributions used throughout the app:
                Gaussian (Normal) Distribution for speed uncertainty, Poisson Distribution for
                congestion modelling, and Binomial Distribution for weekly reliability analysis.
                Designed the Data Comparison module including Z-Score analysis, Box Plots,
                and Welch's T-Test for comparing the online dataset against Google Forms survey data.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("""
        <div class="nav-feature-card" style="cursor:default; border-left:5px solid #D4A373;">
            <div style="font-size:0.7rem; color:#8B5E3C; font-weight:700; letter-spacing:2px;
                        text-transform:uppercase; margin-bottom:6px;">Data Management</div>
            <div class="card-title" style="margin-bottom:10px;">Mohammad Rohail</div>
            <div class="card-desc" style="line-height:1.8;">
                <b>Role:</b> Data Analyst<br><br>
                Handled all aspects of data acquisition and preparation. Researched and sourced
                the online WiFi performance dataset, designed and deployed the Google Forms
                survey for collecting real student responses, identified the key variables
                for prediction (time of day, device count, purpose, etc.), performed data
                cleaning and normalization to create the unified dataset schema used by all models.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r2c2:
        st.markdown("""
        <div class="nav-feature-card" style="cursor:default; border-left:5px solid #2D6A4F;">
            <div style="font-size:0.7rem; color:#8B5E3C; font-weight:700; letter-spacing:2px;
                        text-transform:uppercase; margin-bottom:6px;">UI & Frontend</div>
            <div class="card-title" style="margin-bottom:10px;">Areeba Fatima & Hafsa Akram</div>
            <div class="card-desc" style="line-height:1.8;">
                <b>Role:</b> UI Design, Report Generation & Frontend<br><br>
                Designed and implemented the complete user interface including the beige-and-brown
                academic theme, responsive CSS styling, interactive navigation system,
                and all visual layouts across every page. Created the report generation views
                for AI model results, speed check reports, and probability chart presentations.
                Ensured a professional, accessible, and visually polished user experience.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Project Details ───────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("## Project Details")

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.markdown("""
        <div class="kpi-box">
            <div style="font-size:0.7rem; color:#8B5E3C; font-weight:700;
                        letter-spacing:1.5px; text-transform:uppercase;">Course</div>
            <div class="kpi-val" style="font-size:1.1rem;">Probability & Statistics</div>
            <div class="kpi-lbl">4th Semester</div>
        </div>""", unsafe_allow_html=True)

    with dc2:
        st.markdown("""
        <div class="kpi-box">
            <div style="font-size:0.7rem; color:#8B5E3C; font-weight:700;
                        letter-spacing:1.5px; text-transform:uppercase;">Tech Stack</div>
            <div class="kpi-val" style="font-size:1.1rem;">Python + Streamlit</div>
            <div class="kpi-lbl">scikit-learn, scipy, plotly</div>
        </div>""", unsafe_allow_html=True)

    with dc3:
        st.markdown("""
        <div class="kpi-box">
            <div style="font-size:0.7rem; color:#8B5E3C; font-weight:700;
                        letter-spacing:1.5px; text-transform:uppercase;">Architecture</div>
            <div class="kpi-val" style="font-size:1.1rem;">Modular MVC</div>
            <div class="kpi-lbl">12 Files, 4 AI Models</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="meaning-box">
    <div class="mb-title">Technologies & Methods Used</div>
    <table style="width:100%; font-size:0.88rem; border-collapse:collapse;">
    <tr style="background:#E8E0D5;">
      <th style="padding:8px; text-align:left; color:#3D2B1F;">Category</th>
      <th style="padding:8px; text-align:left; color:#3D2B1F;">Details</th>
    </tr>
    <tr style="border-bottom:1px solid #E8E0D5;">
      <td style="padding:8px; color:#3D2B1F;"><b>Machine Learning</b></td>
      <td style="padding:8px; color:#3D2B1F;">Linear Regression, Bayesian Ridge, Gradient Boosting Regressor</td>
    </tr>
    <tr style="border-bottom:1px solid #E8E0D5;">
      <td style="padding:8px; color:#3D2B1F;"><b>Probability Distributions</b></td>
      <td style="padding:8px; color:#3D2B1F;">Gaussian (Normal), Poisson, Binomial</td>
    </tr>
    <tr style="border-bottom:1px solid #E8E0D5;">
      <td style="padding:8px; color:#3D2B1F;"><b>Statistical Tests</b></td>
      <td style="padding:8px; color:#3D2B1F;">Z-Score Analysis, Welch's T-Test, Correlation Matrix</td>
    </tr>
    <tr style="border-bottom:1px solid #E8E0D5;">
      <td style="padding:8px; color:#3D2B1F;"><b>Data Sources</b></td>
      <td style="padding:8px; color:#3D2B1F;">Curated online WiFi dataset + Real Google Forms student surveys</td>
    </tr>
    <tr>
      <td style="padding:8px; color:#3D2B1F;"><b>Model Evaluation</b></td>
      <td style="padding:8px; color:#3D2B1F;">80/10/10 Train/Validation/Test split, RMSE, R-Squared</td>
    </tr>
    </table>
    </div>
    """, unsafe_allow_html=True)
