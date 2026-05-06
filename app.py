# ╔══════════════════════════════════════════════════════════════════╗
# ║  app.py — NetSmart Main Entry Point                             ║
# ║  HOW TO RUN:  streamlit run app.py                              ║
# ╚══════════════════════════════════════════════════════════════════╝

import warnings
warnings.filterwarnings("ignore")

import re, pathlib
import streamlit as st

from config       import PAGE_LIST
from data_loader  import loadOnlineDataset, loadFormsDataset, loadCombinedDataset
from model_trainer import trainAllModels

from views.page_home         import render as renderHome
from views.page_speed_check  import render as renderSpeedCheck
from views.page_insights     import render as renderInsights
from views.page_model_report import render as renderModelReport
from views.page_comparison   import render as renderComparison
from views.page_about        import render as renderAbout


# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="NetSmart - Internet Optimization",
    page_icon=":signal_strength:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────
def _loadCSS() -> str:
    css_path = pathlib.Path(__file__).parent / "styles" / "main.css"
    raw = css_path.read_text(encoding="utf-8")
    return re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)

st.markdown(f"<style>{_loadCSS()}</style>", unsafe_allow_html=True)

# ── Data & Models ─────────────────────────────────────────────────
df_online   = loadOnlineDataset()
df_forms    = loadFormsDataset()
df_combined = loadCombinedDataset()
lr, br, gb, gbUL, modelMetrics, pModel, splitInfo = trainAllModels(df_combined)

# ── Session State ─────────────────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

def goto(target_page: str):
    st.session_state.current_page = target_page
    st.rerun()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:14px 0 8px;">
        <div style="font-family:'Playfair Display',serif; font-size:2.2rem;
                    font-weight:800; color:#7D1F2E; margin:4px 0;">NetSmart</div>
        <div style="font-size:0.68rem; color:#8B7355; letter-spacing:2px;
                    font-weight:700; text-transform:uppercase;">
            Analytics Platform
        </div>
    </div>
    <hr style="border-color:rgba(139,94,60,0.20); margin:10px 0 18px;"/>
    """, unsafe_allow_html=True)

    # Custom Navigation List
    for p_name in PAGE_LIST:
        is_active = st.session_state.current_page == p_name
        # Use a container to style the active button if possible, or just buttons
        if st.sidebar.button(
            p_name,
            key=f"nav_{p_name}",
            use_container_width=True,
            type="secondary" if not is_active else "primary"
        ):
            st.session_state.current_page = p_name
            st.rerun()

    page = st.session_state.current_page

    st.markdown("<hr style='border-color:rgba(139,94,60,0.20); margin:18px 0;'/>",
                unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:0.70rem; color:#7D1F2E; font-weight:700; "
        "letter-spacing:1.5px; text-transform:uppercase; margin-bottom:10px;'>"
        "LIVE STATS</div>", unsafe_allow_html=True)
    st.metric("Total Records",   f"{len(df_combined)}")
    st.metric("Avg Speed",       f"{df_combined['SpeedDownload'].mean():.1f} Mbps")
    st.metric("Slow Sessions",   f"{df_combined['IsSlow'].mean()*100:.0f}%")
    st.metric("AI Models",       "4")
    st.markdown(
        f"<div style='font-size:0.70rem; color:#8B7355; margin-top:12px;'>"
        f"Split: {splitInfo['train']} train / {splitInfo['validation']} val / {splitInfo['test']} test"
        f"</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(139,94,60,0.20); margin:18px 0 10px;'/>",
                unsafe_allow_html=True)
    st.caption("University Project\nSmart Internet Usage\nOptimization & Prediction")

# ── Page Routing ──────────────────────────────────────────────────
if   page == "Home":
    renderHome(df_combined, splitInfo, goto)
elif page == "Check My Speed":
    renderSpeedCheck(df_combined, lr, br, gb, gbUL, pModel, modelMetrics)
elif page == "Usage Insights":
    renderInsights(df_combined)
elif page == "AI Model Report":
    renderModelReport(df_combined, modelMetrics, splitInfo)
elif page == "Data Comparison":
    renderComparison(df_online, df_forms)
elif page == "About":
    renderAbout()
