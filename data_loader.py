# ╔══════════════════════════════════════════════════════════════════╗
# ║  data_loader.py — NetSmart Real Dataset Loader & Cleaner        ║
# ║  Replaces the old data_generator.py (synthetic data).           ║
# ║  Loads, cleans, and normalises both real datasets into a        ║
# ║  unified column schema for model training and comparison.       ║
# ╚══════════════════════════════════════════════════════════════════╝

import numpy   as np
import pandas  as pd
import pathlib
import streamlit as st

from config import (
    ONLINE_DATASET_PATH, FORMS_DATASET_PATH, FORMS_LIVE_CSV_PATH,
    SLOW_THRESHOLD, PURPOSE_MAP, DEVICE_MAP, UNIFIED_COLS,
)


# ════════════════════════════════════════════════════════════════════
#   HELPER: Normalise free-text categorical values
# ════════════════════════════════════════════════════════════════════

def _normalisePurpose(raw: str) -> str:
    """Map messy purpose strings to the canonical set."""
    if not isinstance(raw, str):
        return "Browsing"
    r = raw.strip().lower()
    if any(k in r for k in ["game", "gaming"]):
        return "Gaming"
    if any(k in r for k in ["stream", "youtube", "movie", "video"]):
        return "Streaming"
    if any(k in r for k in ["social", "instagram", "whatsapp", "facebook", "tiktok"]):
        return "Social Media"
    if any(k in r for k in ["research", "learn"]):
        return "Research"
    if any(k in r for k in ["study", "studies", "assignment", "submission", "github", "lecture", "uni work"]):
        return "Study"
    if any(k in r for k in ["meeting", "call", "zoom"]):
        return "Meeting"
    if any(k in r for k in ["brows"]):
        return "Browsing"
    return "Browsing"      # Fallback


def _normaliseDevice(raw: str) -> str:
    """Map messy device strings to Mobile / Laptop."""
    if not isinstance(raw, str):
        return "Mobile"
    r = raw.strip().lower()
    if any(k in r for k in ["mobile", "phone", "iphone", "android"]):
        return "Mobile"
    if any(k in r for k in ["pc", "desktop", "windows", "macbook", "mac", "laptop"]):
        return "Laptop"
    if "both" in r:
        return "Laptop"
    return "Mobile"


def _speedRangeMidpoint(raw: str) -> float:
    """Convert a speed range string like '10 - 20' to its midpoint."""
    if not isinstance(raw, str):
        return 5.0         # Default fallback
    r = raw.strip()
    if r == "0 - 10":
        return 5.0
    if r == "10 - 20":
        return 15.0
    if r == "20 - 30":
        return 25.0
    if r == "> 30":
        return 35.0
    # Try to parse a numeric value
    try:
        return float(r)
    except ValueError:
        return 5.0


def _timeFromHour(time_str: str) -> str:
    """Map an HH:MM time string to Morning / Afternoon / Evening / Night."""
    try:
        hour = int(time_str.split(":")[0])
    except (ValueError, IndexError, AttributeError):
        return "Afternoon"
    if   6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


def _usersToDevices(raw: str) -> int:
    """Map the Google Forms student-count ranges to a representative device count."""
    if not isinstance(raw, str):
        return 200
    r = raw.strip().lower()
    if "< 10" in r or "<10" in r:
        return 50
    if "10" in r and "25" in r:
        return 150
    if "26" in r or "50" in r:
        return 350
    if "> 50" in r or ">50" in r:
        return 450
    return 200      # "Not Sure" fallback


def _cleanDevicesCount(raw) -> int:
    """Parse the 'devices connected at once' field from the form."""
    if isinstance(raw, (int, float)):
        return max(1, int(raw))
    if isinstance(raw, str):
        # "Sometimes 2 but most of the time only 1" → 1
        digits = [c for c in raw if c.isdigit()]
        if digits:
            return max(1, int(digits[0]))
    return 1


# ════════════════════════════════════════════════════════════════════
#   LOAD: Online WiFi Dataset (150 rows)
# ════════════════════════════════════════════════════════════════════

@st.cache_data
def loadOnlineDataset() -> pd.DataFrame:
    """
    Load and clean the online hostel WiFi dataset (CSV).

    Mapping from raw columns → unified schema:
        Time                   → TimeOfDay  (via hour extraction)
        Purpose                → Purpose    (direct, already clean)
        Device                 → Device     (normalised)
        NumberOfDevices        → NumberOfDevices
        SpeedDownload          → SpeedDownload
        SpeedUpload            → SpeedUpload
        PeakHour               → PeakHour   (Yes/No → 1/0)

    Returns
    -------
    pd.DataFrame with UNIFIED_COLS columns
    """
    raw = pd.read_csv(ONLINE_DATASET_PATH)

    df = pd.DataFrame()
    df["TimeOfDay"]       = raw["Time"].apply(_timeFromHour)
    df["Purpose"]         = raw["Purpose"].apply(_normalisePurpose)
    df["Device"]          = raw["Device"].apply(_normaliseDevice)
    df["NumberOfDevices"] = raw["NumberOfDevices"].astype(int)
    df["SpeedDownload"]   = raw["SpeedDownload"].astype(float).round(2)
    df["SpeedUpload"]     = raw["SpeedUpload"].astype(float).round(2)
    df["PeakHour"]        = raw["PeakHour"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    df["IsSlow"]          = (df["SpeedDownload"] < SLOW_THRESHOLD).astype(int)

    return df


# ════════════════════════════════════════════════════════════════════
#   LOAD: Google Forms Dataset (105+ rows, grows live)
# ════════════════════════════════════════════════════════════════════

def _loadFormsFromExcel() -> pd.DataFrame:
    """
    Load and clean the Google Forms responses (XLSX).

    Mapping from raw columns → unified schema:
        "What time of day..."                 → TimeOfDay
        "What is your primary purpose..."     → Purpose  (normalised)
        "Which device were you using?"        → Device   (normalised)
        "Approximately how many students..."  → NumberOfDevices (range → int)
        "Approximate internet download..."    → SpeedDownload  (range → midpoint)
        "Approximate internet upload..."      → SpeedUpload    (range → midpoint)
        (derived from TimeOfDay)              → PeakHour

    Returns
    -------
    pd.DataFrame with UNIFIED_COLS columns
    """
    raw = pd.read_excel(FORMS_DATASET_PATH)

    df = pd.DataFrame()
    df["TimeOfDay"]       = raw.iloc[:, 1].astype(str).str.strip()
    df["Purpose"]         = raw.iloc[:, 3].apply(_normalisePurpose)
    df["Device"]          = raw.iloc[:, 6].apply(_normaliseDevice)
    df["NumberOfDevices"] = raw.iloc[:, 2].apply(_usersToDevices)
    df["SpeedDownload"]   = raw.iloc[:, 7].apply(_speedRangeMidpoint)
    df["SpeedUpload"]     = raw.iloc[:, 8].apply(_speedRangeMidpoint)
    # Peak hours = Evening or Night
    df["PeakHour"]        = df["TimeOfDay"].isin(["Evening", "Night"]).astype(int)
    df["IsSlow"]          = (df["SpeedDownload"] < SLOW_THRESHOLD).astype(int)

    return df


@st.cache_data
def loadFormsDataset() -> pd.DataFrame:
    """
    Load Google Forms data.  If a live CSV exists (with appended
    user submissions), use that; otherwise fall back to the original
    Excel file.

    Returns
    -------
    pd.DataFrame with UNIFIED_COLS columns
    """
    if FORMS_LIVE_CSV_PATH.exists():
        try:
            live = pd.read_csv(FORMS_LIVE_CSV_PATH)
            # Ensure all required columns exist
            if set(UNIFIED_COLS).issubset(set(live.columns)):
                return live
        except Exception:
            pass

    # First run — load from Excel, create the live CSV for future appends
    df = _loadFormsFromExcel()
    df.to_csv(FORMS_LIVE_CSV_PATH, index=False)
    return df


# ════════════════════════════════════════════════════════════════════
#   APPEND: Save a new user submission to the live Forms CSV
# ════════════════════════════════════════════════════════════════════

def appendSubmission(row_dict: dict) -> None:
    """
    Append a single row to the live Google Forms CSV file.
    If the file doesn't exist, it is initialized from the Excel file first.
    Also clears the Streamlit cache so the next loadFormsDataset()
    picks up the new row.

    Parameters
    ----------
    row_dict : dict
        Must contain all keys from UNIFIED_COLS.
    """
    # Ensure the live CSV exists and is fully populated before appending
    if not FORMS_LIVE_CSV_PATH.exists():
        df = _loadFormsFromExcel()
        df.to_csv(FORMS_LIVE_CSV_PATH, index=False)
        
    row_df = pd.DataFrame([row_dict])
    row_df.to_csv(FORMS_LIVE_CSV_PATH, mode="a", header=False, index=False)

    # Clear cached data so fresh counts are shown
    loadFormsDataset.clear()


# ════════════════════════════════════════════════════════════════════
#   COMBINED: Merge both datasets for model training
# ════════════════════════════════════════════════════════════════════

def loadCombinedDataset() -> pd.DataFrame:
    """
    Return the union of the online dataset and Google Forms dataset.
    This combined frame is used for model training.

    Returns
    -------
    pd.DataFrame with UNIFIED_COLS columns
    """
    df_online = loadOnlineDataset()
    df_forms  = loadFormsDataset()
    combined  = pd.concat([df_online, df_forms], ignore_index=True)
    return combined
# Cache invalidation line
