# ╔══════════════════════════════════════════════════════════════════╗
# ║  config.py — NetSmart Shared Constants & Feature Maps           ║
# ║  All encoding maps, thresholds, and plot-style tokens live here.║
# ║  Import this module in every other module to stay in sync.      ║
# ╚══════════════════════════════════════════════════════════════════╝

import pathlib

# ── Project Root ───────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent

# ── Data File Paths ────────────────────────────────────────────────
ONLINE_DATASET_PATH = PROJECT_ROOT / "Data" / "hostel_wifi_dataset_.csv"
FORMS_DATASET_PATH  = PROJECT_ROOT / "Data" / "google_form_respnoses.xlsx"
FORMS_LIVE_CSV_PATH = PROJECT_ROOT / "Data" / "google_forms_live.csv"

# ── Unified Column Schema ──────────────────────────────────────────
# Both datasets are cleaned/mapped to these column names.
UNIFIED_COLS = [
    "TimeOfDay",        # Morning / Afternoon / Evening / Night
    "Purpose",          # Gaming / Streaming / Browsing / Research / Social Media / Study
    "Device",           # Mobile / Laptop
    "NumberOfDevices",   # integer — total devices connected
    "SpeedDownload",    # float Mbps
    "SpeedUpload",      # float Mbps
    "PeakHour",         # 1 = yes / 0 = no
    "IsSlow",           # 1 if SpeedDownload < SLOW_THRESHOLD else 0
]

# ── Feature Encoding Maps ──────────────────────────────────────────
TIME_MAP = {
    "Morning":   0,
    "Afternoon": 1,
    "Evening":   2,
    "Night":     3,
}

PURPOSE_MAP = {
    "Browsing":     0,
    "Research":     1,
    "Study":        2,
    "Social Media": 3,
    "Streaming":    4,
    "Gaming":       5,
    "Meeting":      6,
    "Assignment":   7,
}

DEVICE_MAP = {
    "Mobile":  0,
    "Laptop":  1,
}

# ── Feature Column Names (for ML model input) ─────────────────────
FEATURE_COLS = [
    "TimeEncoded",
    "PurposeEncoded",
    "DeviceEncoded",
    "NumberOfDevices",
    "PeakHour",
]

# ── Domain Constants ───────────────────────────────────────────────
SLOW_THRESHOLD = 10.0          # Mbps — below this is considered "slow"

TIME_SLOTS  = ["Morning", "Afternoon", "Evening", "Night"]
PURPOSES    = list(PURPOSE_MAP.keys())
DEVICES     = list(DEVICE_MAP.keys())

# ── Navigation Pages ───────────────────────────────────────────────
PAGE_LIST = [
    "Home",
    "Check My Speed",
    "Usage Insights",
    "AI Model Report",
    "Data Comparison",
    "About",
]

# ── Plot / Chart Style Tokens ──────────────────────────────────────
PLOT_BG      = "rgba(0,0,0,0)"    # Transparent to let CSS background show
FONT_DICT    = dict(family="DM Sans, sans-serif", color="#3D2B1A", size=13)
THEME_COLORS = [
    "#7D1F2E", "#5C6B2E", "#A83248", "#7A8E3C",
    "#8B5E3C", "#B89E82", "#5C3D22", "#C5A882",
]
