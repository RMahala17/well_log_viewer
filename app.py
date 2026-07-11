import base64
from PIL import Image
import io
import os
import html
import hashlib
from fpdf import FPDF
import streamlit as st
import lasio
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import uuid
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import requests

# psutil is optional — the RAM-check feature degrades gracefully without it
# instead of crashing the whole app if it isn't installed.
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Absolute path to this script's folder. Used for any bundled asset (demo LAS
# file, rig/gym images) so file lookups work no matter what directory
# Streamlit was launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ==========================================
# PHASE 2: GLOBAL REPOSITORY SETUP
# ==========================================
WELL_REPOSITORY_DIR = os.path.join(SCRIPT_DIR, "well_data_repository")
MASTER_DATABASE_PATH = os.path.join(WELL_REPOSITORY_DIR, "master_subsurface_database.parquet")

# Automatically generate the directory if it doesn't exist yet
if not os.path.exists(WELL_REPOSITORY_DIR):
    os.makedirs(WELL_REPOSITORY_DIR)

# ============================================================
# 🤖  LOCAL AI ENGINE  —  Cloud-aware, Pure Python
# ============================================================
# HOW IT WORKS:
#
# LOCAL  (streamlit run app.py  →  localhost:8501)
#   Python runs ON YOUR PC, so requests to localhost:11434
#   hit YOUR Ollama directly. ✅ Works perfectly.
#
# CLOUD  (*.streamlit.app link)
#   Python runs on Streamlit's US servers. localhost:11434
#   is THEIR localhost — Ollama is not there. It will ALWAYS
#   fail, no matter what. This is not a bug; it is how cloud
#   hosting works.
#
# SOLUTION:
#   Detect the environment first. If on cloud, skip the
#   network check and show a "run locally" message instead.
#   If local, check Ollama normally.
# ============================================================

import os

OLLAMA_BASE = "http://localhost:11434"
REQUIRED_MODELS = ["llama3.1", "moondream"]

def rebuild_master_subsurface_database():
    """Loops through the external data folder, normalizes columns, and saves to an optimized Parquet file."""
    all_well_data = []
    skipped_files = [] # NEW: We will track exactly what fails
    
    DICTIONARY_MAP = {
        'GR': ['GR', 'GRD', 'SGR', 'NGR', 'GAMMA'],
        'RT': ['ILD', 'ILM', 'LLD', 'RT', 'AFEC', 'RILD', 'AHT90', 'RESD', 'RESISTIVITY'],
        'RHOB': ['RHOB', 'DEN', 'ZDEN', 'RHOZ', 'DENS', 'DENSITY'],
        'NPHI': ['NPHI', 'CNC', 'NEUT', 'NPHI_LS', 'PORO', 'NEUTRON'],
        'CALI': ['CALI', 'CAL', 'CALIPER'],
        'BS': ['BS', 'BIT', 'BITSIZE'],
        'DT': ['DT', 'AC', 'SONIC', 'DTC']
    }
    
    las_files = [f for f in os.listdir(WELL_REPOSITORY_DIR) if f.lower().endswith('.las')]
    
    if not las_files:
        return None, skipped_files # NEW: Return a tuple

    for file_name in las_files:
        try:
            file_path = os.path.join(WELL_REPOSITORY_DIR, file_name)
            las = lasio.read(file_path)
            df = las.df().reset_index() 
            
            # FIX 1: Combine header name AND file name so every single file counts as unique
            header_name = las.well.WELL.value if 'WELL' in las.well else "UNKNOWN"
            well_id = f"{header_name} [{file_name}]"
            df['WELL_NAME'] = well_id
            
            for standard_name, aliases in DICTIONARY_MAP.items():
                for alias in aliases:
                    matching_cols = [c for c in df.columns if c.upper() == alias.upper()]
                    if matching_cols:
                        df.rename(columns={matching_cols[0]: standard_name}, inplace=True)
                        break
            
            # FIX 2: Check if we actually found standard curves before adding
            standard_found = [c for c in DICTIONARY_MAP.keys() if c in df.columns]
            if not standard_found:
                skipped_files.append(f"{file_name} (No recognized standard curves mapped)")
                continue

            columns_to_keep = ['DEPTH', 'WELL_NAME'] + list(DICTIONARY_MAP.keys())
            existing_columns = [c for c in columns_to_keep if c in df.columns]
            
            all_well_data.append(df[existing_columns])
        except Exception as e:
            # FIX 3: Catch structural file errors to show in the UI
            skipped_files.append(f"{file_name} (Read Error: {str(e)})")
            
    if all_well_data:
        master_df = pd.concat(all_well_data, ignore_index=True)
        curves_only = list(DICTIONARY_MAP.keys())
        present_curves = [c for c in curves_only if c in master_df.columns]
        master_df.dropna(subset=present_curves, how='all', inplace=True)
        
        master_df.to_parquet(MASTER_DATABASE_PATH, index=False)
        return master_df, skipped_files
    return None, skipped_files

# ==========================================
# SEPARATE FUNCTION: SISTER WELL FINDER
# ==========================================
def discover_sister_wells(active_df, master_df):
    """Compares baseline Gamma Ray curves to identify matching analog wells."""
    # We use Gamma Ray (GR) as the baseline lithology matcher
    target_curve = 'GR' 
    if target_curve not in active_df.columns or target_curve not in master_df.columns:
        return []
        
    # Compute properties of the active well
    active_mean = active_df[target_curve].mean()
    
    analogues = []
    # Compare against every other well in the master database
    for well_id, group in master_df.groupby('WELL_NAME'):
        group_mean = group[target_curve].mean()
        mean_diff = abs(active_mean - group_mean)
        
        analogues.append({
            'well_name': well_id,
            'variance_score': mean_diff,
            'avg_val': group_mean
        })
        
    # Sort so the mathematically closest well is first in the list
    analogues = sorted(analogues, key=lambda x: x['variance_score'])
    return analogues[:3] # Return top 3 matches

def is_streamlit_cloud() -> bool:
    """
    Returns True when the app is running on Streamlit Community Cloud.
    Streamlit Cloud always runs as the Linux user 'appuser' inside
    a container where HOME=/home/appuser.  Local machines will have
    a different HOME (e.g. C:\\Users\\Rohit on Windows or /home/rohit
    on Linux), so this is a reliable, zero-config detection.
    """
    home = os.environ.get("HOME", "")
    return home == "/home/appuser" or os.path.exists("/home/appuser")


def check_ollama_status() -> dict:
    """
    Returns one of four statuses:
      'cloud'          – app is on Streamlit Cloud; Ollama can never be reached
      'ready'          – Ollama running, all required models present
      'missing_models' – Ollama running, but some models not yet downloaded
      'offline'        – Ollama not running / not installed on this machine
    """
    # ── Cloud guard ──────────────────────────────────────────
    if is_streamlit_cloud():
        return {"status": "cloud", "missing": REQUIRED_MODELS}

    # ── Local check ──────────────────────────────────────────
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        if resp.status_code != 200:
            return {"status": "offline", "missing": REQUIRED_MODELS}

        installed = [m.get("name", "").lower() for m in resp.json().get("models", [])]
        missing = [
            req for req in REQUIRED_MODELS
            if not any(req.lower() in name for name in installed)
        ]
        return {"status": "ready" if not missing else "missing_models", "missing": missing}

    except Exception:
        return {"status": "offline", "missing": REQUIRED_MODELS}


def query_local_llama(chat_history, system_context, model_name="llama3.1"):
    """Send a chat request to the user's local Ollama instance."""
    messages = [{"role": "system", "content": system_context}] + chat_history
    payload = {"model": model_name, "messages": messages, "stream": False}
    try:
        resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json().get("message", {}).get("content", "No response from model.")
        elif resp.status_code == 404:
            return (f"❌ Model `{model_name}` not found. "
                    f"Run `ollama pull {model_name}` in your terminal.")
        else:
            return f"❌ Ollama error (HTTP {resp.status_code})."
    except Exception as e:
        return f"❌ Cannot reach Ollama. Make sure it is running. Error: {e}"


# ============================================================
# ☁️  CLOUD AI ENGINE  —  Works for anyone opening the app link
# ============================================================
# The Ollama engine above ONLY works when Python is running on the same
# machine as Ollama (i.e. localhost). When this app is deployed and shared
# as a link, every visitor's browser talks to a server that has no Ollama —
# so the Local engine can never work for them, by design of how cloud
# hosting works. This Cloud engine calls Anthropic's API instead, so the
# AI Copilot works immediately for anyone who opens the app, from any device.
#
# SETUP: add your key to Streamlit secrets (never hardcode it in this file):
#   Streamlit Cloud → App settings → Secrets → ANTHROPIC_API_KEY = "sk-ant-..."
#   Local dev       → create .streamlit/secrets.toml with the same line
# ============================================================

CLOUD_MODEL = "claude-sonnet-5"


def get_anthropic_api_key():
    """Reads the API key from Streamlit secrets without crashing if none is configured."""
    try:
        return st.secrets.get("ANTHROPIC_API_KEY", None)
    except Exception:
        return None


def query_cloud_claude(chat_history, system_context, images_b64=None, model=CLOUD_MODEL):
    """Send a chat request to Anthropic's Claude API. Works from any deployment,
    no local install required. `images_b64` (optional) attaches image(s) to the
    most recent user turn for vision-style analysis."""
    api_key = get_anthropic_api_key()
    if not api_key:
        return ("❌ No API key configured. Add `ANTHROPIC_API_KEY` to your Streamlit "
                 "secrets to enable the Cloud AI engine, or switch to the Local (Ollama) engine.")

    api_messages = []
    last_idx = len(chat_history) - 1
    for i, msg in enumerate(chat_history):
        if msg["role"] == "user" and images_b64 and i == last_idx:
            content = [{"type": "text", "text": msg["content"]}]
            for img_b64 in images_b64:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}
                })
            api_messages.append({"role": "user", "content": content})
        else:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {"model": model, "max_tokens": 1024, "system": system_context, "messages": api_messages}

    try:
        resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            text_blocks = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
            return "\n".join(text_blocks) if text_blocks else "No response from model."
        elif resp.status_code == 401:
            return "❌ Invalid API key. Double-check the ANTHROPIC_API_KEY value in your Streamlit secrets."
        else:
            return f"❌ Cloud AI error (HTTP {resp.status_code}): {resp.text[:200]}"
    except Exception as e:
        return f"❌ Cannot reach the Claude API. Check your internet connection. Error: {e}"


# Initialize Chat Memory
if 'ai_chat_history' not in st.session_state:
    st.session_state['ai_chat_history'] = []
    
# 1. Page Configuration
st.set_page_config(page_title="AI Petrophysics", layout="wide")

# --- CUSTOM UI CSS FOR TABS ---
st.markdown(
    """
    <style>
    /* Make the tab container look like a button row */
    div[data-baseweb="tab-list"] {
        gap: 15px;
        padding-bottom: 10px;
    }
    /* Style the individual unselected tabs to look like buttons */
    button[data-baseweb="tab"] {
        background-color: rgba(128, 128, 128, 0.1) !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        font-weight: 600 !important;
        transition: all 0.3s ease-in-out;
    }
    /* Style the ACTIVE tab with Light Sky Blue */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #87CEFA !important; /* Light Sky Blue */
        color: #121212 !important; /* Dark text for contrast */
        border: 1px solid #87CEFA !important;
        box-shadow: 0 4px 10px rgba(135, 206, 250, 0.4) !important;
    }
    button[data-baseweb="tab"]:hover {
        background-color: rgba(135, 206, 250, 0.3) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --- GYM OFFLINE EASTER EGG ---
def inject_offline_screen():
    import streamlit.components.v1 as components

    # Locate your exact folder structure
    possible_names = ["gym.jpg.jpeg", "gym.jpg", "gym.jpeg", "gym.png"]
    image_path = None
    
    for name in possible_names:
        full_path = os.path.join(SCRIPT_DIR, name)
        if os.path.exists(full_path):
            image_path = full_path
            break

    img_src = ""
    if image_path:
        try:
            img = Image.open(image_path)
            
            # Force convert to RGB to ensure clean JPEG processing
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Shrink dimensions down to standard web display size
            img.thumbnail((350, 350))
            
            # Drop quality to 30% to make the code footprint microscopic
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=30)
            
            encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8').strip()
            img_src = f"data:image/jpeg;base64,{encoded_string}"
        except Exception as e:
            img_src = ""

    # Clean JavaScript transmission without f-string conflicts
    raw_js = """
    <script>
        (function() {
            var imgData = "TARGET_IMAGE_SRC";
            
            function checkConnection() {
                try {
                    var parentDoc = window.parent.document;
                    var offlineScreen = parentDoc.getElementById('gym-offline-screen');
                    
                    if (!offlineScreen) {
                        offlineScreen = parentDoc.createElement('div');
                        offlineScreen.id = 'gym-offline-screen';
                        
                        var imgHtml = imgData ? '<img src="' + imgData + '" style="max-width:90%; max-height:50vh; border-radius:15px; margin-bottom:15px; box-shadow: 0 4px 15px rgba(0,0,0,0.8);">' : '<div style="font-size:100px; margin-bottom:20px;">💪</div>';
                        
                        offlineScreen.innerHTML = '<div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100vh; background-color:#121212; color:white; font-family:sans-serif; text-align:center; padding:20px;">' + imgHtml + '<h1 style="font-size: 2.2rem; margin-top: 15px; color: #ff4b4b; font-weight: bold; line-height: 1.4;">NO INTERNET CONNECTION</h1></div>';
                        
                        offlineScreen.style.cssText = 'display:none; position:fixed; top:0; left:0; width:100%; height:100%; z-index:9999999; background-color:#121212;';
                        parentDoc.body.appendChild(offlineScreen);
                    }
                    
                    if (!navigator.onLine) {
                        offlineScreen.style.display = 'block';
                    } else {
                        offlineScreen.style.display = 'none';
                    }
                } catch(e) {
                    console.log("Error handling offline overlay:", e);
                }
            }

            window.parent.addEventListener('online', checkConnection);
            window.parent.addEventListener('offline', checkConnection);
            checkConnection();
        })();
    </script>
    """
    
    # Safely swap out the target placeholder with the real image source text
    js_code = raw_js.replace("TARGET_IMAGE_SRC", img_src)
    components.html(js_code, height=0, width=0)

inject_offline_screen()
# ------------------------------
st.title(" AI Petrophysics Dashboard")

# --- RESET CALLBACK FUNCTIONS ---
def reset_curve_settings(prefix, curve, index, defaults_dict):
    """Forces the UI widgets to revert by explicitly setting their session state values."""
    for key_suffix, default_val in defaults_dict.items():
        state_key = f"{prefix}_{key_suffix}_{curve}_{index}"
        st.session_state[state_key] = default_val

def reset_multi_track(track_id, default_dict):
    """Forces the Multi-Track widgets to revert."""
    for key_suffix, default_val in default_dict.items():
        state_key = f"mt_{key_suffix}_{track_id}"
        st.session_state[state_key] = default_val

def reset_eval_settings(curve, defaults_dict):
    """Forces the Formation Evaluation widgets to revert."""
    for key_suffix, default_val in defaults_dict.items():
        state_key = f"eval_{key_suffix}_{curve}"
        st.session_state[state_key] = default_val

# --- ROUTING HELPER FUNCTION ---
def route_calculated_curve(curve_name, destinations):
    """Adds newly calculated curves to session state so they appear in other tabs."""
    if curve_name not in st.session_state.available_curves:
        st.session_state.available_curves.append(curve_name)
    
    if "Recorded Logs" in destinations:
        if curve_name not in st.session_state.rec_multi:
            st.session_state.rec_multi.append(curve_name)
            
    if "Smoothed Logs" in destinations:
        if curve_name not in st.session_state.sm_multi:
            st.session_state.sm_multi.append(curve_name)
            
    if "Multi-Track Viewer" in destinations:
        if st.session_state.multi_tracks:
            # Route to the first track dynamically
            first_track_id = st.session_state.multi_tracks[0]['id']
            mt_key = f"mt_curves_{first_track_id}"
            if mt_key in st.session_state and curve_name not in st.session_state[mt_key]:
                st.session_state[mt_key].append(curve_name)

# 2. Sidebar Layout & Data Loading
st.sidebar.header("📁 Data Loading")
# --- 📂 DUAL FILE UPLOADER ENGINE ---


# 1. Custom File Upload
uploaded_file = st.sidebar.file_uploader("Upload Your LAS File", type=['las'])

# --- DYNAMIC RAM CHECK (soft warning, not a hard block) ---
# LAS files are plain text and rarely huge, so this is a light safety net,
# not a gate. A hard st.stop() here could false-block a perfectly normal
# file on memory-constrained hosts (e.g. free-tier cloud deployments).
if uploaded_file is not None:
    if not PSUTIL_AVAILABLE:
        st.sidebar.caption("ℹ️ RAM check skipped (`psutil` not installed).")
    else:
        file_size_gb = uploaded_file.size / (1024**3)
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        # Rough heuristic: parsing dense well logs into Pandas/Numpy takes roughly
        # 5x the raw file size in RAM. This is an estimate, not a hard guarantee.
        required_ram_gb = file_size_gb * 5.0

        if required_ram_gb > available_ram_gb:
            st.sidebar.warning(
                f"⚠️ **This file may be large for available memory.**\n\n"
                f"• **File Size:** {file_size_gb:.2f} GB\n"
                f"• **Estimated RAM Needed:** ~{required_ram_gb:.2f} GB\n"
                f"• **Available RAM:** {available_ram_gb:.2f} GB\n\n"
                f"Processing may be slow. You can still proceed if you like."
            )
            proceed_anyway = st.sidebar.checkbox("Proceed anyway", key="ram_override")
            if not proceed_anyway:
                st.sidebar.info("💡 Check the box above to continue, or upload a smaller file.")
                st.stop()
        else:
            st.sidebar.success(
                f"🖥️ **System Check Passed:** Enough RAM available! "
                f"(Estimated need: ~{required_ram_gb:.2f} GB | Available: {available_ram_gb:.2f} GB)"
            )

# Divider or spacing
st.sidebar.markdown("<div style='text-align: center; margin: 5px 0; opacity: 0.5;'>— OR —</div>", unsafe_allow_html=True)

# 2. Demo File Checkbox Toggle
use_demo_data = st.sidebar.checkbox(" Use Demo Well Data", value=False, help="Click to instantly parse and load pre-packaged Ichthys Deep-1 offshore wireline well logs.")

# Master Controller: Decide which file pointer to feed into the processing engine
las_file_source = None

if uploaded_file is not None:
    las_file_source = uploaded_file
    st.sidebar.success("✅ Custom LAS file loaded successfully!")
elif use_demo_data:
    # Resolved relative to this script's own folder (not the current working
    # directory), so this works no matter where `streamlit run` is launched from.
    demo_path = os.path.join(SCRIPT_DIR, "ichthys_deep_1_wire_public_2010_sdb.las")
    if os.path.exists(demo_path):
        las_file_source = demo_path
        st.sidebar.success("⚡ Demo Well (Ichthys Deep-1) active!")
    else:
        st.sidebar.error("❌ 'demo_well.las' not found in your directory. Please check file path assets.")

if las_file_source is not None:
    try:
        # Build an identifier that changes whenever the actual file CONTENT changes,
        # not just when the filename changes. Comparing filenames alone meant two
        # different files that happened to share a name (e.g. "well.las") would
        # incorrectly reuse stale cached data from the first upload.
        if uploaded_file is not None:
            current_filename = f"{uploaded_file.name}:{hashlib.md5(uploaded_file.getvalue()).hexdigest()}"
        else:
            current_filename = f"demo:{las_file_source}"

        # --- ROBUST DATA HANDLING WITH SESSION STATE ---
        if 'uploaded_filename' not in st.session_state or st.session_state.uploaded_filename != current_filename:
            
            # DYNAMIC PARSING: Handle both local demo file paths and uploaded file buffers
            if isinstance(las_file_source, str):
                las = lasio.read(las_file_source)
            else:
                # Not all real-world LAS files are UTF-8 — try a few common
                # encodings instead of hard-crashing on the first mismatch.
                raw_bytes = las_file_source.getvalue()
                string_data = None
                for encoding in ("utf-8", "latin-1", "cp1252"):
                    try:
                        string_data = raw_bytes.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                if string_data is None:
                    string_data = raw_bytes.decode("utf-8", errors="replace")
                las = lasio.read(string_data)
                
            df = las.df()
            df['DEPTH'] = df.index 
            cols = ['DEPTH'] + [col for col in df.columns if col != 'DEPTH']
            df = df[cols].reset_index(drop=True)
            
            st.session_state.df = df
            st.session_state.las = las
            st.session_state.uploaded_filename = current_filename
            
            # Initialize global curves list
            st.session_state.available_curves = [col for col in df.columns if col != 'DEPTH']
            
            # Default curves logic
            default_curves = []
            if 'GR' in st.session_state.available_curves: default_curves.append('GR')
            if 'AFEC' in st.session_state.available_curves: default_curves.append('AFEC')
            elif 'RILD' in st.session_state.available_curves: default_curves.append('RILD')
            
            # Initialize tab-specific selected curves
            st.session_state.rec_multi = default_curves.copy()
            st.session_state.sm_multi = default_curves.copy()

            # Reset multi-tracks
            st.session_state.multi_tracks = [{'id': str(uuid.uuid4())}, {'id': str(uuid.uuid4())}]

        df = st.session_state.df
        las = st.session_state.las
        available_curves = st.session_state.available_curves

        # Extract Header Information
        def get_header_df(las_section):
            data = []
            for item in las_section:
                data.append({"Mnemonic": item.mnemonic, "Unit": item.unit, "Value": str(item.value), "Description": item.descr})
            return pd.DataFrame(data)

        well_info = get_header_df(las.well)
        curve_info = get_header_df(las.curves)
        param_info = get_header_df(las.params)
        
        # Sidebar Controls
        st.sidebar.header("⚙️ Data Controls")
        min_depth = float(df['DEPTH'].min())
        max_depth = float(df['DEPTH'].max())
        depth_range = st.sidebar.slider("Select Global Depth Range (m):", 
                                        min_value=min_depth, max_value=max_depth, 
                                        value=(min_depth, max_depth))
        
        df_filtered = df[(df['DEPTH'] >= depth_range[0]) & (df['DEPTH'] <= depth_range[1])].copy()
        
        well_name = las.well.WELL.value if las.well.WELL.value else "Unknown Well"
        # Escape before it ever touches unsafe_allow_html=True — well_name comes
        # straight from user-uploaded file content, so a crafted LAS file could
        # otherwise inject HTML/JS into the page.
        well_name_safe = html.escape(str(well_name))
        # --- LARGE SKY BLUE TOP BANNER ---
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #87CEFA 0%, #4682B4 100%); 
                        padding: 35px 25px; 
                        border-radius: 15px; 
                        margin-bottom: 25px; 
                        box-shadow: 0 6px 12px rgba(0,0,0,0.15); 
                        text-align: left;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;">
                <div>
                    <h1 style="color: #ffffff; margin: 0; font-size: 2.5rem; font-weight: 800; text-shadow: 1px 1px 2px rgba(0,0,0,0.25); font-family: sans-serif;">
                        AI Petrophysics Dashboard
                    </h1>
                    <p style="color: #f0f8ff; font-size: 1.1rem; margin-top: 8px; margin-bottom: 0; font-weight: 500; font-family: sans-serif;">
                        Active Well Reference: <span style="font-size: 1.2rem; font-weight: bold; color: #121212; background-color: rgba(255,255,255,0.75); padding: 3px 12px; border-radius: 6px; margin-left: 6px;">{well_name_safe}</span>
                    </p>
                </div>
                <div style="font-size: 4.5rem; opacity: 0.85; margin-right: 10px;">
                    
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Define Tabs
        tab_raw, tab_rec, tab_smooth, tab_hist, tab_stats, tab_multi, tab_cross, tab_eval, tab_ml, tab_report = st.tabs([
            "Raw Data", "Recorded Logs", "Smoothed Logs", "Histogram", "Statistics", "Multi-Track", "Crossplot", "Formation Evaluation", "Machine Learning", "📄 Report"
        ])
        
        # --- TAB 1: RAW DATA & HEADERS ---
        with tab_raw:
            
            # -------------------------------------------------------------
            # 🌐 MASTER STORAGE SYNCHRONIZATION (Prevents data loss on rerun)
            # -------------------------------------------------------------
            if 'master_log_df' not in st.session_state:
                st.session_state.master_log_df = df_filtered.copy()
            else:
                # Ensure any external filter changes to depth keep our custom columns intact
                for col in st.session_state.master_log_df.columns:
                    if col not in df_filtered.columns:
                        df_filtered[col] = st.session_state.master_log_df[col]

            if 'added_curves_registry' not in st.session_state:
                st.session_state.added_curves_registry = []

            # -------------------------------------------------------------
            # ⚙️ ENGINEERING & COLUMN CUSTOMIZATION TOOLBAR
            # -------------------------------------------------------------
            st.markdown("### ⚙️ Engineering & Column Customization Toolbar")
            t_col1, t_col2 = st.columns(2)

            with t_col1:
                # --- ADD ENGINE ---
                with st.expander("🛠️ Add New Log Curves (Unlimited)", expanded=False):
                    cc_col1, cc_col2 = st.columns([2, 1])
                    with cc_col1:
                        new_col_name = st.text_input("New Curve Name", placeholder="e.g., CALI2, CALI3, BS2", key="ui_new_curve_name")
                    with cc_col2:
                        st.markdown("<div style='padding-top:28px;'></div>", unsafe_allow_html=True)
                        add_col_btn = st.button("🚀 Add Column", type="primary", use_container_width=True)

                    if add_col_btn:
                        cleaned_name = new_col_name.strip()
                        if not cleaned_name:
                            st.error("❌ Column name cannot be empty!")
                        elif cleaned_name in st.session_state.master_log_df.columns:
                            st.warning(f"⚠️ Column '{cleaned_name}' already exists!")
                        else:
                            # Inject permanently into master session state dataframe
                            st.session_state.master_log_df[cleaned_name] = np.nan
                            df_filtered[cleaned_name] = np.nan
                            st.session_state.added_curves_registry.append(cleaned_name)
                            st.success(f"✅ Column '{cleaned_name}' added permanently!")
                            st.rerun()

                # --- DELETE ENGINE ---
                with st.expander("❌ Delete Existing Log Curves", expanded=False):
                    deletable_cols = [c for c in df_filtered.columns if c != 'DEPTH']
                    
                    if deletable_cols:
                        dc_col1, dc_col2 = st.columns([2, 1])
                        with dc_col1:
                            col_to_delete = st.selectbox("Select Column to Delete", deletable_cols, key="ui_delete_curve_select")
                        with dc_col2:
                            st.markdown("<div style='padding-top:28px;'></div>", unsafe_allow_html=True)
                            delete_col_btn = st.button("🗑️ Delete Column", type="primary", use_container_width=True)

                        if delete_col_btn:
                            # 1. Drop from active dataframe view
                            if col_to_delete in df_filtered.columns:
                                df_filtered.drop(columns=[col_to_delete], inplace=True)
                            
                            # 2. Drop from master storage session state to ensure permanence
                            if col_to_delete in st.session_state.master_log_df.columns:
                                st.session_state.master_log_df.drop(columns=[col_to_delete], inplace=True)
                            
                            # 3. Clear from tracking registry if it was custom made
                            if col_to_delete in st.session_state.added_curves_registry:
                                st.session_state.added_curves_registry.remove(col_to_delete)
                                
                            st.success(f"💥 Column '{col_to_delete}' has been completely deleted!")
                            st.rerun()
                    else:
                        st.info("No logs available for deletion.")

            with t_col2:
                # --- EXCEL FILL DOWN ENGINE ---
                with st.expander("⚡ Excel-Style Fill Down Tool", expanded=False):
                    active_cols = [c for c in df_filtered.columns if c != 'DEPTH']
                    fd_col = st.selectbox("Select Target Column", active_cols, key="fd_active_col_select")
                    fd_mode = st.radio("Value Source", ["Enter Custom Constant", "Copy 1st Row Value Down"], horizontal=True)
                    
                    fd_val = 0.0
                    if fd_mode == "Enter Custom Constant":
                        st.markdown("<div style='margin-top:-15px;'></div>", unsafe_allow_html=True)
                        fd_val = st.number_input("Constant Numeric Value", value=0.0, format="%.4f")
                    
                    run_fill_down = st.button("⚡ Execute Fill Down", type="secondary", use_container_width=True)

                    if run_fill_down:
                        if fd_mode == "Copy 1st Row Value Down" and not df_filtered.empty:
                            first_val = df_filtered[fd_col].iloc[0]
                            fd_val = float(first_val) if (not pd.isna(first_val) and first_val is not None) else 0.0
                        
                        st.session_state.master_log_df[fd_col] = fd_val
                        df_filtered[fd_col] = fd_val
                        st.success(f"⚡ Column '{fd_col}' filled completely with {fd_val}!")
                        st.rerun()

            # Finalize sync mapping to clean up views
            for col in list(df_filtered.columns):
                if col != 'DEPTH' and col not in st.session_state.master_log_df.columns:
                    df_filtered.drop(columns=[col], inplace=True)
            for col in st.session_state.master_log_df.columns:
                if col not in df_filtered.columns:
                    df_filtered[col] = st.session_state.master_log_df[col]
            
            # Global Synchronization for dropdown menu parameters across other tabs
            available_curves = [col for col in df_filtered.columns if col not in ['DEPTH']]

            st.divider()

            # -------------------------------------------------------------
            # 📋 INTERACTIVE EXCEL SPREADSHEET (Copy-Paste & Editing Active)
            # -------------------------------------------------------------
            st.markdown("###  Raw Log Data")
            st.caption("💡 *Pro-Tip: Select any cell, paste arrays from Excel using Ctrl+V, or edit individual rows manually.*")
            
            # Upgraded active data spreadsheet editor
            edited_df = st.data_editor(
                df_filtered, 
                use_container_width=True, 
                num_rows="fixed",
                key="master_raw_data_editor"
            )

            # Cell editing feedback loop tracking to save modifications permanently
            editor_state = st.session_state.get("master_raw_data_editor")
            if editor_state and "edited_rows" in editor_state:
                edited_rows = editor_state["edited_rows"]
                if edited_rows:
                    for row_idx_str, changes in edited_rows.items():
                        row_idx = int(row_idx_str)
                        actual_depth = df_filtered.iloc[row_idx]['DEPTH']
                        
                        master_mask = st.session_state.master_log_df['DEPTH'] == actual_depth
                        if master_mask.any():
                            for col_name, new_value in changes.items():
                                val_to_write = float(new_value) if new_value is not None else np.nan
                                st.session_state.master_log_df.loc[master_mask, col_name] = val_to_write
                    st.rerun()

            # Dataset Summary Footer
            st.markdown(f"**Total Active Rows:** {df_filtered.shape[0]}")
            
            # -------------------------------------------------------------
            # 🏢 RESTORED SECTION: NEAT LAS FILE HEADER INFORMATION TABBED VIEW
            # -------------------------------------------------------------
            st.markdown("---")
            st.markdown("###  LAS File Header Information")
            header_tab1, header_tab2, header_tab3 = st.tabs(["Well Information", "Curve Information", "Parameter Information"])
            with header_tab1: st.dataframe(well_info, use_container_width=True, hide_index=True)
            with header_tab2: st.dataframe(curve_info, use_container_width=True, hide_index=True)
            with header_tab3: st.dataframe(param_info, use_container_width=True, hide_index=True)
            st.session_state['las_well_info'] = well_info
            st.session_state['las_curve_info'] = curve_info
            st.session_state['las_param_info'] = param_info
            
        # --- TAB 2: RECORDED LOGS ---
        with tab_rec:
            st.markdown("###  Interactive Recorded Logs Viewer")
            selected_curves = st.multiselect("➕ Add or Remove Log Curves:", available_curves, key="rec_multi")
            
            # 1. Initialize an empty list to store ALL generated charts
            st.session_state['recorded_logs_figs_list'] = []
            
            if selected_curves:
                cols = st.columns(len(selected_curves))
                for i, curve in enumerate(selected_curves):
                    with cols[i]:
                        c_min = float(df_filtered[curve].min()) if not df_filtered[curve].empty else 0.0
                        c_max = float(df_filtered[curve].max()) if not df_filtered[curve].empty else 100.0
                        def_col = "#008000" if "GR" in curve.upper() else ("#FF0000" if i%2==0 else "#0000FF")
                        def_log = True if "R" in curve.upper() or "AFEC" in curve.upper() else False
                        def_xspc = float(max(0.1, round((c_max-c_min)/5, 1)))
                        
                        rec_defaults = {
                            "col": def_col, "log": def_log, "xmin": c_min, "xmax": c_max,
                            "depth": (depth_range[0], depth_range[1]), "xspc": def_xspc, "yspc": 200.0
                        }
                        
                        with st.expander(f"⚙️ {curve} Settings"):
                            rec_c1, rec_c2 = st.columns(2)
                            curve_color = rec_c1.color_picker(f"🎨 Line Color", def_col, key=f"rec_col_{curve}_{i}")
                            is_log = rec_c2.checkbox(f"Logarithmic X-Axis", value=def_log, key=f"rec_log_{curve}_{i}")
                            
                            b_c1, b_c2 = st.columns(2)
                            x_min = b_c1.number_input("X Min", value=c_min, key=f"rec_xmin_{curve}_{i}")
                            x_max = b_c2.number_input("X Max", value=c_max, key=f"rec_xmax_{curve}_{i}")
                            
                            track_depth = st.slider("Isolate Depth", min_value=depth_range[0], max_value=depth_range[1], value=(depth_range[0], depth_range[1]), key=f"rec_depth_{curve}_{i}")
                            
                            s_c1, s_c2 = st.columns(2)
                            x_spacing = s_c1.number_input("X Spacing", value=def_xspc, key=f"rec_xspc_{curve}_{i}")
                            y_spacing = s_c2.number_input("Y Spacing", value=200.0, key=f"rec_yspc_{curve}_{i}")

                            st.button("🔄 Reset Defaults", key=f"rec_reset_{curve}_{i}", on_click=reset_curve_settings, args=("rec", curve, i, rec_defaults))

                        track_df = df_filtered[(df_filtered['DEPTH'] >= track_depth[0]) & (df_filtered['DEPTH'] <= track_depth[1])]
                        fig_rec = go.Figure()
                        fig_rec.add_trace(go.Scatter(x=track_df[curve], y=track_df['DEPTH'], mode='lines', line=dict(color=curve_color, width=1.5), name=curve))
                        
                        if is_log:
                            x_range = [np.log10(x_min) if x_min > 0 else 0, np.log10(x_max) if x_max > 0 else 2]
                            actual_x_spacing = None 
                        else:
                            x_range = [x_min, x_max]
                            actual_x_spacing = x_spacing
                            
                        fig_rec.update_layout(
                            plot_bgcolor='white', height=800, margin=dict(t=150, b=20, l=50, r=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"),
                            xaxis=dict(title=f"{curve}", side="top", type="log" if is_log else "linear", range=x_range, dtick=actual_x_spacing, showgrid=True, gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black"),
                            yaxis=dict(title="Depth (m)" if i == 0 else "", range=[track_depth[1], track_depth[0]], dtick=y_spacing if y_spacing > 0 else None, showgrid=True, gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black")
                        )
                        st.plotly_chart(fig_rec, use_container_width=True)
                        
                        # 2. APPEND the figure to the list instead of replacing it!
                        st.session_state['recorded_logs_figs_list'].append(fig_rec)

        # --- TAB 3: SMOOTHED LOGS ---
        with tab_smooth:
            st.markdown("### Smoothed Logs Viewer")
            selected_smooth_curves = st.multiselect("➕ Select Curves to Smooth:", available_curves, key="sm_multi")
            
            # 1. Initialize an empty list to store ALL generated smoothed charts
            st.session_state['smoothed_logs_figs_list'] = []
            
            if selected_smooth_curves:
                cols_sm = st.columns(len(selected_smooth_curves))
                for i, curve in enumerate(selected_smooth_curves):
                    with cols_sm[i]:
                        c_min = float(df_filtered[curve].min()) if not df_filtered[curve].empty else 0.0
                        c_max = float(df_filtered[curve].max()) if not df_filtered[curve].empty else 100.0
                        def_col = "#008000" if "GR" in curve.upper() else "#0000FF"
                        def_log = True if "R" in curve.upper() or "AFEC" in curve.upper() else False
                        def_xspc = float(max(0.1, round((c_max-c_min)/5, 1)))
                        
                        sm_defaults = {"win": 10, "col": def_col, "log": def_log, "xmin": c_min, "xmax": c_max, "depth": (depth_range[0], depth_range[1]), "xspc": def_xspc, "yspc": 200.0, "orig": True}
                        
                        with st.expander(f"⚙️ {curve} Smoothing Settings"):
                            window_size = st.number_input(f" Window Size", min_value=1, max_value=500, value=10, step=1, key=f"sm_win_{curve}_{i}")
                            sm_c1, sm_c2 = st.columns(2)
                            curve_color = sm_c1.color_picker(f"🎨 Line Color", def_col, key=f"sm_col_{curve}_{i}")
                            is_log = sm_c2.checkbox(f"Logarithmic X-Axis", value=def_log, key=f"sm_log_{curve}_{i}")
                            
                            b_c1, b_c2 = st.columns(2)
                            x_min = b_c1.number_input("X Min", value=c_min, key=f"sm_xmin_{curve}_{i}")
                            x_max = b_c2.number_input("X Max", value=c_max, key=f"sm_xmax_{curve}_{i}")
                            
                            track_depth = st.slider("Isolate Depth", min_value=depth_range[0], max_value=depth_range[1], value=(depth_range[0], depth_range[1]), key=f"sm_depth_{curve}_{i}")
                            
                            s_c1, s_c2 = st.columns(2)
                            x_spacing = s_c1.number_input("X Spacing", value=def_xspc, key=f"sm_xspc_{curve}_{i}")
                            y_spacing = s_c2.number_input("Y Spacing", value=200.0, key=f"sm_yspc_{curve}_{i}")
                            
                            show_original = st.checkbox(f"Show Original Raw Curve", value=True, key=f"sm_orig_{curve}_{i}")
                            st.button("🔄 Reset Defaults", key=f"sm_reset_{curve}_{i}", on_click=reset_curve_settings, args=("sm", curve, i, sm_defaults))

                        track_df_sm = df_filtered[(df_filtered['DEPTH'] >= track_depth[0]) & (df_filtered['DEPTH'] <= track_depth[1])].copy()
                        track_df_sm[f'{curve}_SMOOTH'] = track_df_sm[curve].rolling(window=window_size, center=True, min_periods=1).mean()
                        
                        fig_sm = go.Figure()
                        if show_original:
                            fig_sm.add_trace(go.Scatter(x=track_df_sm[curve], y=track_df_sm['DEPTH'], mode='lines', line=dict(color='lightgrey', width=1), name=f"{curve} (Raw)", showlegend=False))
                        fig_sm.add_trace(go.Scatter(x=track_df_sm[f'{curve}_SMOOTH'], y=track_df_sm['DEPTH'], mode='lines', line=dict(color=curve_color, width=2), name=f"{curve} (Smoothed)", showlegend=False))
                        
                        if is_log:
                            x_range = [np.log10(x_min) if x_min > 0 else 0, np.log10(x_max) if x_max > 0 else 2]
                            actual_x_spacing = None
                        else:
                            x_range = [x_min, x_max]
                            actual_x_spacing = x_spacing
                            
                        fig_sm.update_layout(
                            plot_bgcolor='white', height=800, margin=dict(t=150, b=20, l=50, r=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"),
                            xaxis=dict(title=f"{curve} (Smoothed)", side="top", type="log" if is_log else "linear", range=x_range, dtick=actual_x_spacing, showgrid=True, gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black"),
                            yaxis=dict(title="Depth (m)" if i == 0 else "", range=[track_depth[1], track_depth[0]], dtick=y_spacing if y_spacing > 0 else None, showgrid=True, gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black")
                        )
                        st.plotly_chart(fig_sm, use_container_width=True)
                        
                        # 2. APPEND the figure to the list instead of replacing it!
                        st.session_state['smoothed_logs_figs_list'].append(fig_sm)

        # --- TAB 4: HISTOGRAM ---
        with tab_hist:
            st.markdown("###  Data Distribution (Histogram)")
            selected_hist_curves = st.multiselect("➕ Select Curves for Histograms:", available_curves, default=[available_curves[0]] if available_curves else [])
            
            # 1. Initialize an empty list to store ALL generated histogram charts
            st.session_state['histogram_figs_list'] = []
            
            if selected_hist_curves:
                cols_hist = st.columns(len(selected_hist_curves))
                for i, curve in enumerate(selected_hist_curves):
                    with cols_hist[i]:
                        hist_data = df_filtered[curve].dropna()
                        c_min = float(hist_data.min()) if not hist_data.empty else 0.0
                        c_max = float(hist_data.max()) if not hist_data.empty else 1.0
                        def_bin = float(max(0.01, round((c_max - c_min) / 40.0, 2)))
                        def_col = "#17A2B8" if i % 2 == 0 else "#E83E8C"
                        def_xspc = float(max(0.1, round((c_max-c_min)/5, 1)))
                        
                        hist_defaults = {"bin": def_bin, "col": def_col, "xmin": c_min, "xmax": c_max, "xspc": def_xspc, "yspc": 0}
                        
                        with st.expander(f"⚙️ {curve} Histogram Settings"):
                            hist_col1, hist_col2 = st.columns([2, 1])
                            with hist_col1: bin_size = st.number_input(f"Bin Size", min_value=0.01, value=def_bin, step=0.10, format="%.2f", key=f"hist_bin_{curve}_{i}")
                            with hist_col2: hist_color = st.color_picker(f"Colour", def_col, key=f"hist_col_{curve}_{i}")
                            
                            hb_c1, hb_c2 = st.columns(2)
                            x_min = hb_c1.number_input("X Min", value=c_min, key=f"hist_xmin_{curve}_{i}")
                            x_max = hb_c2.number_input("X Max", value=c_max, key=f"hist_xmax_{curve}_{i}")
                            
                            hs_c1, hs_c2 = st.columns(2)
                            x_spacing = hs_c1.number_input("X-Axis Spacing", value=def_xspc, key=f"hist_xspc_{curve}_{i}")
                            y_spacing = hs_c2.number_input("Y-Axis Spacing", value=0, key=f"hist_yspc_{curve}_{i}")
                        
                            st.button("🔄 Reset Defaults", key=f"hist_reset_{curve}_{i}", on_click=reset_curve_settings, args=("hist", curve, i, hist_defaults))

                        fig_hist = px.histogram(hist_data, x=curve, color_discrete_sequence=[hist_color], title=f"Distribution of {curve}")
                        fig_hist.update_traces(xbins=dict(size=bin_size))
                        fig_hist.update_layout(
                            plot_bgcolor='white', bargap=0.05,
                            xaxis=dict(title=curve, range=[x_min, x_max], dtick=x_spacing if x_spacing > 0 else None, showgrid=True, gridcolor="lightgrey", mirror=True, showline=True, linecolor="black"),
                            yaxis=dict(title="Frequency", dtick=y_spacing if y_spacing > 0 else None, showgrid=True, gridcolor="lightgrey", mirror=True, showline=True, linecolor="black")
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # 2. APPEND the figure to the list instead of replacing it!
                        st.session_state['histogram_figs_list'].append(fig_hist)

        # --- TAB 5: STATISTICS ---
        with tab_stats:
            st.markdown("###  Overall Dataset Statistics")
            st.dataframe(df_filtered.describe(), use_container_width=True)
            st.markdown("---")
            st.markdown("###  Custom Percentile Calculator")
            stat_col1, stat_col2 = st.columns(2)
            with stat_col1:
                stat_curve = st.selectbox("Select Curve:", available_curves, index=available_curves.index('GR') if 'GR' in available_curves else 0, key="stat_curve")
            with stat_col2:
                pct_input = st.text_input("Enter percentiles (comma-separated):", value="5, 10, 50, 90, 95")
            try:
                pct_list = [float(p.strip()) for p in pct_input.split(',')]
                valid_data = df_filtered[stat_curve].dropna()
                if not valid_data.empty:
                    calculated_pcts = np.percentile(valid_data, pct_list)
                    cols = st.columns(len(pct_list))
                    for i, (pct, val) in enumerate(zip(pct_list, calculated_pcts)):
                        cols[i].metric(label=f"{pct}th Pct", value=f"{val:.2f}")
            except ValueError:
                st.error("Please enter valid numbers.")

            st.markdown("---")
            st.markdown("### 🔗 Curve Correlation Matrix")
            corr_curves = st.multiselect("Select curves to compare:", available_curves, default=available_curves[:4] if len(available_curves)>=4 else available_curves, key="corr_multi")
            if len(corr_curves) > 1:
                corr_matrix = df_filtered[corr_curves].corr()
                fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
                st.plotly_chart(fig_corr, use_container_width=True)
                st.session_state['correlation_matrix_fig'] = fig_corr

        # --- TAB 6: MULTI-TRACK ---
        with tab_multi:
            st.markdown("### 🏢 Dynamic Multi-Track Log Viewer with Fluid & Lithology Intelligence")

            def add_track():
                st.session_state.multi_tracks.append({'id': str(uuid.uuid4())})
            def remove_track(track_id):
                st.session_state.multi_tracks = [t for t in st.session_state.multi_tracks if t['id'] != track_id]

            st.markdown("#### 📐 Global Depth Range (Y-Axis)")
            g_col1, g_col2 = st.columns([3, 1])
            with g_col1: mt_global_depth = st.slider("Isolate Depth", min_value=depth_range[0], max_value=depth_range[1], value=(depth_range[0], depth_range[1]), key="mt_global_depth")
            with g_col2: mt_global_yspc = st.number_input("Global Y Spacing", value=200.0, key="mt_global_yspc")

            with st.expander("🎯 Fluid Identification Cutoffs (Sand/Shale & Pay Shading)", expanded=False):
                st.caption("Every well/basin behaves differently — tune these instead of relying on fixed defaults.")
                fc1, fc2 = st.columns(2)
                with fc1:
                    gr_cutoff = st.number_input("GR Cutoff — Sand/Shale (API):", value=75.0, step=5.0, key="mt_gr_cutoff",
                                                 help="GR below this is treated as sand; at or above is treated as shale.")
                with fc2:
                    rt_cutoff = st.number_input("Resistivity Cutoff (Ω·m):", value=10.0, step=1.0, key="mt_rt_cutoff",
                                                 help="Resistivity above this is treated as a potential hydrocarbon zone.")

            st.button("➕ Add New Track", on_click=add_track, type="primary")
            st.divider()

            default_palette = ['#0000FF', '#FF0000', '#008000', '#FF00FF', '#000000', '#FFA500']
            track_settings = [] 
            
            # --- BACKEND MATRIX LOGIC FROM THE TABLE ---
            # 1. Calculate DPHI silently if missing but RHOB is present
            if 'RHOB' in df_filtered.columns and 'DPHI' not in df_filtered.columns:
                df_filtered['DPHI'] = (2.65 - df_filtered['RHOB']) / (2.65 - 1.0)
            
            # 2. Identify the Resistivity curve dynamically from the file
            rt_col = next((c for c in df_filtered.columns if c.upper() in ['RT', 'ILD', 'LLD', 'RESD', 'RES', 'LL3']), None)
            
            mt_available_curves = [col for col in df_filtered.columns if col not in ['DEPTH']]

            for i, track in enumerate(st.session_state.multi_tracks):
                track_id = track['id']
                with st.expander(f"⚙️ Settings: Track {i+1}", expanded=True):
                    st.caption("💡 *Backend active: The system will automatically auto-shade Gas (Red), Oil (Green), Brine (Blue), and Shale (Gray) based on the matrix table definitions.*")
                    selected_curves = st.multiselect("➕ Add Curves to this Track:", mt_available_curves, key=f"mt_curves_{track_id}")
                    
                    mt_defaults = {"log": False, "xmin": 0.0, "xmax": 100.0, "xspc": 10.0}
                    for j, curve in enumerate(selected_curves): mt_defaults[f"col_{curve}"] = default_palette[j % len(default_palette)]

                    col1, col2 = st.columns(2)
                    is_log = col1.checkbox("Logarithmic X-axis", key=f"mt_log_{track_id}")
                    col2.button("❌ Delete Track", on_click=remove_track, args=(track_id,), key=f"del_{track_id}")

                    b_c1, b_c2 = st.columns(2)
                    
                    # Smart default scales for standard curves
                    if any(c in ['DPHI', 'NPHI'] for c in selected_curves):
                        default_xmin, default_xmax, default_spc = 0.45, -0.15, 0.15
                    elif 'GR' in selected_curves:
                        default_xmin, default_xmax, default_spc = 0.0, 150.0, 30.0
                    elif rt_col and rt_col in selected_curves:
                        default_xmin, default_xmax, default_spc = 0.2, 2000.0, 10.0
                    else:
                        default_xmin, default_xmax, default_spc = 0.0, 100.0, 10.0
                    
                    x_min = b_c1.number_input("X Min", value=default_xmin, key=f"mt_xmin_{track_id}")
                    x_max = b_c2.number_input("X Max", value=default_xmax, key=f"mt_xmax_{track_id}")
                    x_spacing = st.number_input("X Major Spacing", value=default_spc, key=f"mt_xspc_{track_id}", disabled=is_log)

                    # --- UPDATED UNBOUNDED WINDOW SIZE FEATURE ---
                    sm_col1, sm_col2 = st.columns(2)
                    with sm_col1:
                        smooth_window = st.number_input(
                            f"Smoothening Window Size", 
                            min_value=1, 
                            value=1,  # Removed the max_value constraint entirely!
                            step=2, 
                            key=f"mt_smooth_{track_id}"
                        )
                    with sm_col2:
                        st.write("") 
                        st.write("")
                        show_raw = st.checkbox(f"Show Original Raw Curve", value=False, key=f"mt_raw_{track_id}")

                    curve_colors = {}
                    if selected_curves:
                        st.markdown("**🎨 Curve Colors**")
                        color_cols = st.columns(len(selected_curves))
                        for j, curve in enumerate(selected_curves):
                            with color_cols[j]: curve_colors[curve] = st.color_picker(f"{curve}", value=mt_defaults[f"col_{curve}"], key=f"mt_col_{curve}_{track_id}")

                    st.button("🔄 Reset Track Defaults", key=f"mt_reset_{track_id}", on_click=reset_multi_track, args=(track_id, mt_defaults))
                    track_settings.append({
                        'curves': selected_curves, 'colors': curve_colors, 'is_log': is_log, 
                        'x_min': x_min, 'x_max': x_max, 'x_spacing': x_spacing,
                        'smooth_window': smooth_window, 'show_raw': show_raw
                    })

            st.divider()
            num_tracks = len(st.session_state.multi_tracks)

            if num_tracks > 0:
                mt_df = df_filtered[(df_filtered['DEPTH'] >= mt_global_depth[0]) & (df_filtered['DEPTH'] <= mt_global_depth[1])].copy()
                fig_mt = make_subplots(rows=1, cols=num_tracks, shared_yaxes=True, horizontal_spacing=0.02)

                # --- ADVANCED FLUID MATRIX LOGIC CALCULATION ---
                # gr_cutoff / rt_cutoff now come from the UI controls set above.
                
                # Pre-fill base flags arrays
                is_sand = mt_df['GR'] < gr_cutoff if 'GR' in mt_df.columns else np.ones(len(mt_df), dtype=bool)
                is_shale = mt_df['GR'] >= gr_cutoff if 'GR' in mt_df.columns else np.zeros(len(mt_df), dtype=bool)
                
                res_vals = mt_df[rt_col] if rt_col else np.ones(len(mt_df)) * 100.0
                has_high_rt = res_vals > rt_cutoff
                
                has_crossover = (mt_df['NPHI'] < mt_df['DPHI']) if ('NPHI' in mt_df.columns and 'DPHI' in mt_df.columns) else np.zeros(len(mt_df), dtype=bool)

                # Define explicit condition masks
                gas_mask = is_sand & has_high_rt & has_crossover
                oil_mask = is_sand & has_high_rt & (~has_crossover)
                brine_mask = is_sand & (~has_high_rt)

                for i, settings in enumerate(track_settings):
                    col_idx = i + 1
                    x_left, x_right = settings['x_min'], settings['x_max']

                    # --- FLUID AND LITHOLOGY BACKGROUND SHADING ---
                    # 1. Shale Shading (Gray)
                    shale_x = np.where(is_shale, x_right, x_left)
                    fig_mt.add_trace(go.Scatter(x=[x_left]*len(mt_df), y=mt_df['DEPTH'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=col_idx)
                    fig_mt.add_trace(go.Scatter(x=shale_x, y=mt_df['DEPTH'], mode='lines', fill='tonextx', fillcolor='rgba(128, 128, 128, 0.25)', line=dict(width=0), name='Shale Zone', showlegend=(col_idx==1)), row=1, col=col_idx)

                    # 2. Brine/Water Sand Shading (Light Blue)
                    brine_x = np.where(brine_mask, x_right, x_left)
                    fig_mt.add_trace(go.Scatter(x=[x_left]*len(mt_df), y=mt_df['DEPTH'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=col_idx)
                    fig_mt.add_trace(go.Scatter(x=brine_x, y=mt_df['DEPTH'], mode='lines', fill='tonextx', fillcolor='rgba(135, 206, 250, 0.45)', line=dict(width=0), name='Brine Sand', showlegend=(col_idx==1)), row=1, col=col_idx)

                    # 3. Oil Sand Shading (Green)
                    oil_x = np.where(oil_mask, x_right, x_left)
                    fig_mt.add_trace(go.Scatter(x=[x_left]*len(mt_df), y=mt_df['DEPTH'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=col_idx)
                    fig_mt.add_trace(go.Scatter(x=oil_x, y=mt_df['DEPTH'], mode='lines', fill='tonextx', fillcolor='rgba(46, 139, 87, 0.4)', line=dict(width=0), name='Oil Sand', showlegend=(col_idx==1)), row=1, col=col_idx)

                    # 4. Gas Sand Shading (Red/Orange Crossover Area)
                    gas_x = np.where(gas_mask, x_right, x_left)
                    fig_mt.add_trace(go.Scatter(x=[x_left]*len(mt_df), y=mt_df['DEPTH'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=col_idx)
                    fig_mt.add_trace(go.Scatter(x=gas_x, y=mt_df['DEPTH'], mode='lines', fill='tonextx', fillcolor='rgba(255, 69, 0, 0.5)', line=dict(width=0), name='Gas Sand', showlegend=(col_idx==1)), row=1, col=col_idx)


                    # --- STANDARD CURVE PLOTTING OVER THE BACKGROUNDS ---
                    for curve in settings['curves']:
                        dash_style = 'dash' if curve == 'NPHI' else 'solid'
                        curve_color = settings['colors'].get(curve, '#000')
                        
                        smooth_w = settings['smooth_window']
                        show_raw_flag = settings['show_raw']
                        
                        # Apply smoothing if requested
                        if smooth_w > 1:
                            plot_data = mt_df[curve].rolling(window=smooth_w, center=True).mean()
                            main_name = f"{curve} (Smoothed)"
                        else:
                            plot_data = mt_df[curve]
                            main_name = curve
                            
                        # Plot primary curve
                        fig_mt.add_trace(go.Scatter(
                            x=plot_data, 
                            y=mt_df['DEPTH'], 
                            name=main_name, 
                            line=dict(color=curve_color, width=1.8, dash=dash_style), 
                            mode='lines'
                        ), row=1, col=col_idx)
                        
                        # Plot raw overlay if requested
                        if smooth_w > 1 and show_raw_flag:
                            fig_mt.add_trace(go.Scatter(
                                x=mt_df[curve], 
                                y=mt_df['DEPTH'], 
                                name=f"{curve} (Raw)", 
                                opacity=0.4, 
                                line=dict(color=curve_color, width=1, dash='dot'), 
                                mode='lines'
                            ), row=1, col=col_idx)

                    # Configure Axis Scales
                    if settings['is_log']:
                        x_range = [np.log10(settings['x_min']) if settings['x_min'] > 0 else 0, np.log10(settings['x_max']) if settings['x_max'] > 0 else 2]
                        dtick = None
                    else:
                        x_range = [settings['x_min'], settings['x_max']]
                        dtick = settings['x_spacing'] if settings['x_spacing'] > 0 else None

                    fig_mt.update_xaxes(title_text=", ".join(settings['curves']) if settings['curves'] else f"Track {col_idx}", side="top", type="log" if settings['is_log'] else "linear", range=x_range, dtick=dtick, showgrid=True, gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black", row=1, col=col_idx)

                # Final Layout adjustments
                fig_mt.update_yaxes(title_text="Depth (m)", range=[mt_global_depth[1], mt_global_depth[0]], dtick=mt_global_yspc if mt_global_yspc > 0 else None, showgrid=True, gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black", row=1, col=1)
                fig_mt.update_layout(plot_bgcolor='white', height=850, margin=dict(t=150, b=20, l=50, r=20), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5, bgcolor="rgba(255,255,255,0.8)"))
                st.plotly_chart(fig_mt, use_container_width=True)
                st.session_state['multi_track_fig'] = fig_mt

        # --- TAB 7: CROSSPLOT ---
        with tab_cross:
            st.write("### Interactive Crossplot")
            cp_col1, cp_col2, cp_col3 = st.columns(3)
            with cp_col1: x_axis = st.selectbox("X-Axis:", available_curves, index=0)
            with cp_col2: y_axis = st.selectbox("Y-Axis:", available_curves, index=1 if len(available_curves)>1 else 0)
            with cp_col3: color_axis = st.selectbox("Color By:", available_curves, index=2 if len(available_curves)>2 else 0)
            fig4 = px.scatter(df_filtered, x=x_axis, y=y_axis, color=color_axis, color_continuous_scale="jet")
            if 'RHOB' in y_axis.upper() or 'DEN' in y_axis.upper(): fig4.update_yaxes(autorange="reversed")
            if 'RHOB' in x_axis.upper() or 'DEN' in x_axis.upper(): fig4.update_xaxes(autorange="reversed")
            st.plotly_chart(fig4, use_container_width=True)
            st.session_state['crossplot_fig'] = fig4
            
        # --- TAB 8: FORMATION EVALUATION ---
        with tab_eval:
            st.markdown("###  Formation Evaluation Calculator")
            
            # 🛠️ THE NEW ROUTING HELPER FUNCTION (Confined strictly to this tab)
            def route_to_viewers(curve_name, destinations):
                """Safely adds calculated curves to other tabs by modifying session state directly."""
                # 1. Make sure it's in the global available curves list
                if 'available_curves' in st.session_state and curve_name not in st.session_state.available_curves:
                    st.session_state.available_curves.append(curve_name)
                if curve_name not in available_curves:
                    available_curves.append(curve_name)
                    
                # 2. Add to the specific selected tabs if the user checked the box
                if "Recorded Logs" in destinations:
                    if 'rec_multi' in st.session_state and curve_name not in st.session_state['rec_multi']:
                        st.session_state['rec_multi'].append(curve_name)
                        
                if "Smoothed Logs" in destinations:
                    if 'sm_multi' in st.session_state and curve_name not in st.session_state['sm_multi']:
                        st.session_state['sm_multi'].append(curve_name)

            # --- 1. VOLUME OF SHALE (VSH) / IGR ---
            st.markdown("#### 1. Volume of Shale (Linear Index - Igr)")
            vsh_c1, vsh_c2, vsh_c3 = st.columns(3)
            with vsh_c1: gr_curve = st.selectbox("Select GR Curve:", available_curves, index=available_curves.index('GR') if 'GR' in available_curves else 0, key="eval_gr_sel")
            with vsh_c2: gr_clean = st.number_input("GR Clean (Sand):", value=20.0, step=1.0)
            with vsh_c3: gr_shale = st.number_input("GR Shale (Shale):", value=100.0, step=1.0)

            vsh_defaults = {"col": "#A52A2A", "log": False, "xmin": 0.0, "xmax": 1.0, "depth": (depth_range[0], depth_range[1]), "xspc": 0.1, "yspc": 200.0}
            with st.expander("⚙️ Vsh Plot Settings"):
                set1, set2 = st.columns(2)
                vsh_color = set1.color_picker("Color", vsh_defaults["col"], key="eval_col_VSH")
                vsh_log = set2.checkbox("Logarithmic X-Axis", value=vsh_defaults["log"], key="eval_log_VSH")
                b1, b2 = st.columns(2)
                vsh_xmin = b1.number_input("X Min", value=vsh_defaults["xmin"], key="eval_xmin_VSH")
                vsh_xmax = b2.number_input("X Max", value=vsh_defaults["xmax"], key="eval_xmax_VSH")
                vsh_depth = st.slider("Isolate Depth", min_value=depth_range[0], max_value=depth_range[1], value=vsh_defaults["depth"], key="eval_depth_VSH")
                sp1, sp2 = st.columns(2)
                vsh_xspc = sp1.number_input("X Spacing", value=vsh_defaults["xspc"], key="eval_xspc_VSH")
                vsh_yspc = sp2.number_input("Y Spacing", value=vsh_defaults["yspc"], key="eval_yspc_VSH")
                if 'reset_eval_settings' in globals(): st.button("🔄 Reset Vsh Settings", on_click=reset_eval_settings, args=("VSH", vsh_defaults), key="res_VSH")
            
            vsh_dest = st.multiselect("🔗 Send 'VSH' to other viewers:", ["Recorded Logs", "Smoothed Logs", "Multi-Track Viewer"], key="vsh_dest")
            
            if gr_clean == gr_shale:
                st.warning("⚠️ GR Clean and GR Shale can't be equal — pick two different values before calculating.")
            elif st.button("Calculate Volume of Shale (Linear Index)"):
                # 1. Calculate the curve
                calculated_vsh = ((df_filtered[gr_curve] - gr_clean) / (gr_shale - gr_clean)).clip(0, 1)
                
                # 2. THE CRITICAL FIX: Save it to master_log_df so it bypasses the deletion security!
                st.session_state.master_log_df['VSH'] = calculated_vsh
                df_filtered['VSH'] = calculated_vsh 
                
                # 3. Use your original routing function
                route_calculated_curve('VSH', vsh_dest)
                st.success("✅ Linear VSH (Igr) Calculated & Saved permanently!")
                
                # 4. Save the figure for display
                fig_vsh = go.Figure()
                fig_vsh.add_trace(go.Scatter(x=df_filtered['VSH'], y=df_filtered['DEPTH'], mode='lines', line=dict(color=vsh_color, width=1.5)))
                fig_vsh.update_layout(plot_bgcolor='white', height=600, margin=dict(t=150, b=20, l=50, r=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(title="Linear VSH (Igr)", side="top", type="log" if vsh_log else "linear", range=[vsh_xmin, vsh_xmax], dtick=None if vsh_log else vsh_xspc, showgrid=True, gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black"),
                    yaxis=dict(title="Depth (m)", range=[vsh_depth[1], vsh_depth[0]], dtick=vsh_yspc if vsh_yspc>0 else None, showgrid=True, gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black"))
                
                st.session_state['fig_vsh'] = fig_vsh
                
                # 5. Refresh the app so the other tabs update
                st.rerun()
            
            # Display the figure safely outside the button
            if 'fig_vsh' in st.session_state:
                st.plotly_chart(st.session_state['fig_vsh'], use_container_width=True)
            st.markdown("---")
            
            # --- 2. VSH CORRECTION (LARIONOV) ---
            st.markdown("#### 2. Volume of Shale Correction (Larionov)")
            vsh_candidates = [col for col in df_filtered.columns if 'VSH' in col.upper() or 'IGR' in col.upper()]
            if not vsh_candidates:
                st.warning("⚠️ Please calculate the Linear Volume of Shale (Point 1) first to enable Corrections.")
            else:
                vshc_c1, vshc_c2 = st.columns(2)
                with vshc_c1: vsh_input_curve = st.selectbox("Select Input Vsh (Linear):", vsh_candidates, key="eval_vshc_input")
                with vshc_c2: correction_types = st.multiselect("Select Rock Age:", ["Tertiary (Younger Rocks)", "Older Rocks"], default=["Tertiary (Younger Rocks)"], key="eval_vshc_type")
                vshc_defaults = {"col_tert": "#FF8C00", "col_older": "#800080", "log": False, "xmin": 0.0, "xmax": 1.0, "depth": (depth_range[0], depth_range[1]), "xspc": 0.1, "yspc": 200.0}
                with st.expander("⚙️ Vsh Correction Plot Settings"):
                    set_c1, set_c2, set_c3 = st.columns(3)
                    vshc_col_tert = set_c1.color_picker("Color (Tertiary)", vshc_defaults["col_tert"], key="eval_col_VSHC_tert")
                    vshc_col_older = set_c2.color_picker("Color (Older)", vshc_defaults["col_older"], key="eval_col_VSHC_old")
                    vshc_log = set_c3.checkbox("Logarithmic X-Axis", value=vshc_defaults["log"], key="eval_log_VSHC")
                    b_c1, b_c2 = st.columns(2)
                    vshc_xmin = b_c1.number_input("X Min Bound", value=vshc_defaults["xmin"], key="eval_xmin_VSHC")
                    vshc_xmax = b_c2.number_input("X Max Bound", value=vshc_defaults["xmax"], key="eval_xmax_VSHC")
                    vshc_depth = st.slider("Isolate Depth Range", min_value=depth_range[0], max_value=depth_range[1], value=vshc_defaults["depth"], key="eval_depth_VSHC")
                    sp_c1, sp_c2 = st.columns(2)
                    vshc_xspc = sp_c1.number_input("X Grid Spacing", value=vshc_defaults["xspc"], key="eval_xspc_VSHC")
                    vshc_yspc = sp_c2.number_input("Y Grid Spacing", value=vshc_defaults["yspc"], key="eval_yspc_VSHC")

                if st.button("Calculate & Plot Corrected Vsh"):
                    igr = df_filtered[vsh_input_curve]
                    fig_vshc = go.Figure()
                    fig_vshc.add_trace(go.Scatter(x=igr, y=df_filtered['DEPTH'], mode='lines', line=dict(color='lightgrey', width=1.5, dash='dash'), name="Original Linear Vsh"))
                    if "Tertiary (Younger Rocks)" in correction_types:
                        df_filtered['VSH_CORR_TERT'] = 0.083 * (np.power(2, (3.7 * igr)) - 1)
                        df_filtered['VSH_CORR_TERT'] = df_filtered['VSH_CORR_TERT'].clip(0, 1)
                        st.session_state.df['VSH_CORR_TERT'] = df_filtered['VSH_CORR_TERT']
                        fig_vshc.add_trace(go.Scatter(x=df_filtered['VSH_CORR_TERT'], y=df_filtered['DEPTH'], mode='lines', line=dict(color=vshc_col_tert, width=2), name="Tertiary Correction"))
                    if "Older Rocks" in correction_types:
                        df_filtered['VSH_CORR_OLDER'] = 0.33 * (np.power(2, (2 * igr)) - 1)
                        df_filtered['VSH_CORR_OLDER'] = df_filtered['VSH_CORR_OLDER'].clip(0, 1)
                        st.session_state.df['VSH_CORR_OLDER'] = df_filtered['VSH_CORR_OLDER']
                        fig_vshc.add_trace(go.Scatter(x=df_filtered['VSH_CORR_OLDER'], y=df_filtered['DEPTH'], mode='lines', line=dict(color=vshc_col_older, width=2), name="Older Correction"))
                    st.success("✅ Corrected Vsh Calculated!")
                    fig_vshc.update_layout(plot_bgcolor='white', height=600, margin=dict(t=150, b=20, l=50, r=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"),
                        xaxis=dict(title="Corrected Vsh", side="top", type="log" if vshc_log else "linear", range=[vshc_xmin, vshc_xmax], dtick=None if vshc_log else vshc_xspc, showgrid=True, gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black"),
                        yaxis=dict(title="Depth (m)", range=[vshc_depth[1], vshc_depth[0]], dtick=vshc_yspc if vshc_yspc>0 else None, showgrid=True, gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black"))
                    st.plotly_chart(fig_vshc, use_container_width=True)
                    st.session_state['fig_vshc'] = fig_vshc

            st.markdown("---")

            # --- 3. DENSITY POROSITY (PHID) ---
            st.markdown("#### 3. Density Porosity (PhiD)")
            phi_c1, phi_c2, phi_c3 = st.columns(3)
            rho_idx = next((i for i, c in enumerate(available_curves) if any(x in c.upper() for x in ['RHOB', 'ZDEN', 'DEN'])), 0)
            with phi_c1: rho_curve = st.selectbox("Select Density Curve:", available_curves, index=rho_idx, key="eval_rho_sel")
            with phi_c2: rho_mat = st.number_input("Matrix Density:", value=2.65, step=0.01)
            with phi_c3: rho_fl = st.number_input("Fluid Density:", value=1.00, step=0.01)
            phi_defaults = {"col": "#1E90FF", "log": False, "xmin": 0.5, "xmax": 0.0, "depth": (depth_range[0], depth_range[1]), "xspc": 0.1, "yspc": 200.0}
            with st.expander("⚙️ Density Porosity Plot Settings"):
                set1, set2 = st.columns(2)
                phi_color = set1.color_picker("Color", phi_defaults["col"], key="eval_col_PHI")
                b1, b2 = st.columns(2)
                phi_xmin = b1.number_input("X Min", value=phi_defaults["xmin"], key="eval_xmin_PHI")
                phi_xmax = b2.number_input("X Max", value=phi_defaults["xmax"], key="eval_xmax_PHI")
                phi_depth = st.slider("Isolate Depth", min_value=depth_range[0], max_value=depth_range[1], value=phi_defaults["depth"], key="eval_depth_PHI")

            phi_dest = st.multiselect("🔗 Send 'PHID' to other viewers:", ["Recorded Logs", "Smoothed Logs", "Multi-Track Viewer"], key="phi_dest")
            
            if rho_mat == rho_fl:
                st.warning("⚠️ Matrix Density and Fluid Density can't be equal — pick two different values before calculating.")
            elif st.button("Calculate Density Porosity (PhiD)"):
                # 1. Calculate the curve
                calculated_phid = ((rho_mat - df_filtered[rho_curve]) / (rho_mat - rho_fl)).clip(0, 1)
                
                # 2. Save it to master_log_df so it bypasses the deletion security!
                st.session_state.master_log_df['PHID'] = calculated_phid
                df_filtered['PHID'] = calculated_phid
                
                # 3. Use your routing function
                route_calculated_curve('PHID', phi_dest)
                st.success("✅ Density Porosity Calculated & Saved permanently!")
                
                # 4. Save the figure for display
                fig_phi = go.Figure()
                fig_phi.add_trace(go.Scatter(x=df_filtered['PHID'], y=df_filtered['DEPTH'], mode='lines', line=dict(color=phi_color, width=1.5)))
                fig_phi.update_layout(plot_bgcolor='white', height=600, margin=dict(t=150, b=20, l=50, r=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(title="Density Porosity (PHID)", side="top", range=[phi_xmin, phi_xmax], showgrid=True, gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black"),
                    yaxis=dict(title="Depth (m)", range=[phi_depth[1], phi_depth[0]], showgrid=True, gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black"))
                
                st.session_state['fig_phi'] = fig_phi
                
                # 5. Refresh the app so the upper tabs update immediately
                st.rerun()

            # Display the figure safely outside the button so it never vanishes
            if 'fig_phi' in st.session_state:
                st.plotly_chart(st.session_state['fig_phi'], use_container_width=True)

            st.markdown("---")

            # --- 4. SONIC POROSITY (WYLLIE TIME-AVERAGE) ---
            st.markdown("#### 4. Sonic Porosity (Wyllie Time-Average)")
            phis_c1, phis_c2, phis_c3 = st.columns(3)
            dt_idx = next((i for i, c in enumerate(available_curves) if any(x in c.upper() for x in ['DT', 'DTCO', 'AC'])), 0)
            with phis_c1: dt_curve = st.selectbox("Select Transit Time Curve (Δt):", available_curves, index=dt_idx, key="eval_dt_sel")
            with phis_c2: dt_mat = st.number_input("Matrix Transit Time (Δt_ma):", value=55.5, step=1.0)
            with phis_c3: dt_fl = st.number_input("Fluid Transit Time (Δt_fl):", value=189.0, step=1.0)
            phis_defaults = {"col": "#32CD32", "log": False, "xmin": 0.5, "xmax": 0.0, "depth": (depth_range[0], depth_range[1]), "xspc": 0.1, "yspc": 200.0}
            with st.expander("⚙️ Sonic Porosity Plot Settings"):
                set1, set2 = st.columns(2)
                phis_color = set1.color_picker("Color", phis_defaults["col"], key="eval_col_PHIS")
                phis_xmin = st.number_input("X Min Bound", value=phis_defaults["xmin"], key="eval_xmin_PHIS")
                phis_xmax = st.number_input("X Max Bound", value=phis_defaults["xmax"], key="eval_xmax_PHIS")
                phis_depth = st.slider("Isolate Depth Range", min_value=depth_range[0], max_value=depth_range[1], value=phis_defaults["depth"], key="eval_depth_PHIS")

            phis_dest = st.multiselect("🔗 Send 'PHIS' to other viewers:", ["Recorded Logs", "Smoothed Logs", "Multi-Track Viewer"], key="phis_dest")
            
            if dt_fl == dt_mat:
                st.warning("⚠️ Matrix and Fluid Transit Time can't be equal — pick two different values before calculating.")
            elif st.button("Calculate Sonic Porosity (PhiS)"):
                # 1. Calculate the curve
                calculated_phis = ((df_filtered[dt_curve] - dt_mat) / (dt_fl - dt_mat)).clip(0, 1)
                
                # 2. Save it to master_log_df so it bypasses the deletion security!
                st.session_state.master_log_df['PHIS'] = calculated_phis
                df_filtered['PHIS'] = calculated_phis
                
                # 3. Use your routing function
                route_calculated_curve('PHIS', phis_dest)
                st.success("✅ Sonic Porosity Calculated & Saved permanently!")
                
                # 4. Save the figure for display
                fig_phis = go.Figure()
                fig_phis.add_trace(go.Scatter(x=df_filtered['PHIS'], y=df_filtered['DEPTH'], mode='lines', line=dict(color=phis_color, width=1.5)))
                fig_phis.update_layout(plot_bgcolor='white', height=600, margin=dict(t=150, b=20, l=50, r=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(title="Sonic Porosity (PHIS)", side="top", range=[phis_xmin, phis_xmax], showgrid=True, gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black"),
                    yaxis=dict(title="Depth (m)", range=[phis_depth[1], phis_depth[0]], showgrid=True, gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black"))
                
                st.session_state['fig_phis'] = fig_phis
                
                # 5. Refresh the app so the upper tabs update immediately
                st.rerun()

            # Display the figure safely outside the button so it stays visible
            if 'fig_phis' in st.session_state:
                st.plotly_chart(st.session_state['fig_phis'], use_container_width=True)

            st.markdown("---")
            
            # --- 5. TOTAL POROSITY (PHIT - NEUTRON DENSITY) ---
            st.markdown("#### 5. Total Porosity (Neutron-Density Combination)")
            
            phit_c1, phit_c2, phit_c3 = st.columns(3)
            nphi_idx = next((i for i, c in enumerate(available_curves) if any(x in c.upper() for x in ['NPHI', 'HNPHI', 'NPOR'])), 0)
            with phit_c1: nphi_curve = st.selectbox("Select Neutron Curve (ΦN):", available_curves, index=nphi_idx, key="eval_nphi_sel")
            phi_curves = [col for col in df_filtered.columns if 'PHID' in col.upper()]
            with phit_c2: phid_curve = st.selectbox("Select Density Porosity (ΦD):", phi_curves if phi_curves else available_curves, key="eval_phid_input")
            with phit_c3: geological_case = st.radio("Geological Case:", ["Gas Bearing Formation", "Oil or Brine Bearing"], key="eval_phit_case")

            phit_defaults = {"col": "#FF00FF", "log": False, "xmin": 0.5, "xmax": 0.0, "depth": (depth_range[0], depth_range[1]), "xspc": 0.1, "yspc": 200.0}
            with st.expander("⚙️ Total Porosity Plot Settings"):
                set1, set2 = st.columns(2)
                phit_color = set1.color_picker("Line Color", phit_defaults["col"], key="eval_col_PHIT")
                phit_xmin = st.number_input("X Min (Total)", value=phit_defaults["xmin"], key="eval_xmin_PHIT")
                phit_xmax = st.number_input("X Max (Total)", value=phit_defaults["xmax"], key="eval_xmax_PHIT")
                phit_depth = st.slider("Depth Range", min_value=depth_range[0], max_value=depth_range[1], value=phit_defaults["depth"], key="eval_depth_PHIT")

            phit_dest = st.multiselect("🔗 Send 'PHIT' to other viewers:", ["Recorded Logs", "Smoothed Logs", "Multi-Track Viewer"], key="phit_dest")

            if st.button("Calculate Total Porosity (PhiT)"):
                nphi = df_filtered[nphi_curve]
                phid = df_filtered[phid_curve]
                if "Gas" in geological_case:
                    calculated_phit = np.sqrt((nphi**2 + phid**2) / 2).clip(0, 1)
                else:
                    calculated_phit = ((nphi + phid) / 2).clip(0, 1)
                
                # ✅ Save directly to master_log_df to escape the cleanup script
                st.session_state.master_log_df['PHIT'] = calculated_phit
                df_filtered['PHIT'] = calculated_phit
                
                # ✅ Use routing mapping helper
                route_calculated_curve('PHIT', phit_dest)
                st.success(f"✅ Total Porosity ({geological_case}) Calculated & Saved permanently!")

                # ✅ Build chart
                fig_phit = go.Figure()
                fig_phit.add_trace(go.Scatter(x=df_filtered['PHIT'], y=df_filtered['DEPTH'], mode='lines', line=dict(color=phit_color, width=2)))
                fig_phit.update_layout(plot_bgcolor='white', height=600, margin=dict(t=150, b=20, l=50, r=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(title=f"Total Porosity", side="top", range=[phit_xmin, phit_xmax], showgrid=True, gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black"),
                    yaxis=dict(title="Depth (m)", range=[phit_depth[1], phit_depth[0]], showgrid=True, gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black"))
                
                # ✅ Store chart safely
                st.session_state['fig_phit'] = fig_phit
                
                # ✅ Force refresh
                st.rerun()

            # ✅ Render plot safely outside the submission context so it doesn't vanish
            if 'fig_phit' in st.session_state:
                st.plotly_chart(st.session_state['fig_phit'], use_container_width=True)

            st.markdown("---")
            
            # --- 6. EFFECTIVE POROSITY (PHIE) ---
            st.markdown("#### 6. Effective Porosity (PhiE)")
            
            vsh_options = [col for col in df_filtered.columns if 'VSH' in col.upper()]
            phit_options = [col for col in df_filtered.columns if 'PHI' in col.upper() and col != 'PHIE']
            
            if not vsh_options or not phit_options:
                st.warning("⚠️ Please calculate Volume of Shale and a Porosity curve (preferably Total Porosity) first!")
            else:
                phie_c1, phie_c2 = st.columns(2)
                with phie_c1: vsh_for_phie = st.selectbox("Select Volume of Shale (Vsh):", vsh_options, key="eval_phie_vsh_sel")
                with phie_c2: phit_for_phie = st.selectbox("Select Total Porosity (PhiT):", phit_options, key="eval_phie_phit_sel")

                phie_defaults = {"col": "#DC143C", "xmin": 0.5, "xmax": 0.0, "depth": (depth_range[0], depth_range[1])}
                with st.expander("⚙️ Effective Porosity Plot Settings"):
                    set1 = st.columns(2)
                    phie_color = set1[0].color_picker("Color", phie_defaults["col"], key="eval_col_PHIE")
                    phie_xmin = st.number_input("X Min (Effective)", value=phie_defaults["xmin"], key="eval_xmin_PHIE")
                    phie_xmax = st.number_input("X Max (Effective)", value=phie_defaults["xmax"], key="eval_xmax_PHIE")
                    phie_depth = st.slider("Depth Range (PHIE)", min_value=depth_range[0], max_value=depth_range[1], value=phie_defaults["depth"], key="eval_depth_PHIE")

                phie_dest = st.multiselect("🔗 Send 'PHIE' to other viewers:", ["Recorded Logs", "Smoothed Logs", "Multi-Track Viewer"], key="phie_dest")

                if st.button("Calculate Effective Porosity (PhiE)"):
                    # 1. Calculate the curve
                    calculated_phie = (df_filtered[phit_for_phie] * (1 - df_filtered[vsh_for_phie])).clip(0, 1)
                    
                    # 2. Save it directly to master_log_df
                    st.session_state.master_log_df['PHIE'] = calculated_phie
                    df_filtered['PHIE'] = calculated_phie
                    
                    # 3. Route it safely to the other tabs
                    route_calculated_curve('PHIE', phie_dest)
                    st.success("✅ Effective Porosity Calculated & Saved permanently!")

                    # 4. Create the plot
                    fig_phie = go.Figure()
                    fig_phie.add_trace(go.Scatter(x=df_filtered['PHIE'], y=df_filtered['DEPTH'], mode='lines', line=dict(color=phie_color, width=2), fill='tozerox', fillcolor='rgba(220, 20, 60, 0.2)'))
                    fig_phie.update_layout(plot_bgcolor='white', height=600, margin=dict(t=150, b=20, l=50, r=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"),
                        xaxis=dict(title="Effective Porosity (PHIE)", side="top", range=[phie_xmin, phie_xmax], showgrid=True, gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black"),
                        yaxis=dict(title="Depth (m)", range=[phie_depth[1], phie_depth[0]], showgrid=True, gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black"))
                    
                    # 5. Store plot securely
                    st.session_state['fig_phie'] = fig_phie
                    
                    # 6. Refresh the app to update the dropdowns everywhere
                    st.rerun()

                # Display the chart safely outside the button
                if 'fig_phie' in st.session_state:
                    st.plotly_chart(st.session_state['fig_phie'], use_container_width=True)

            st.markdown("---")
            
            # --- 7. ARCHIE'S WATER SATURATION (SW) ---
            st.markdown("#### 7. Archie's Water Saturation (Sw)")
            poro_candidates = [col for col in df_filtered.columns if 'PHI' in col.upper()]
            if not poro_candidates:
                st.warning("⚠️ Please calculate a Porosity curve first!")
            else:
                sw_c0, sw_c1, sw_c2 = st.columns(3)
                default_poro_idx = poro_candidates.index('PHIE') if 'PHIE' in poro_candidates else 0
                with sw_c0: poro_input = st.selectbox("Select Porosity source:", poro_candidates, index=default_poro_idx, key="eval_poro_sel_sw")
                with sw_c1: rt_curve = st.selectbox("Deep Resistivity (Rt):", available_curves, key="eval_rt_sel")
                with sw_c2: rw_val = st.number_input("Water Res. (Rw):", value=0.05, step=0.01)
                sw_c3, sw_c4, sw_c5 = st.columns(3)
                with sw_c3: a_val = st.number_input("a:", value=1.00, step=0.1)
                with sw_c4: m_val = st.number_input("m:", value=2.00, step=0.1)
                with sw_c5: n_val = st.number_input("n:", value=2.00, step=0.1)
                sw_defaults = {"col": "#00CED1", "xmin": 1.0, "xmax": 0.0, "depth": (depth_range[0], depth_range[1])}
                with st.expander("⚙️ Sw Plot Settings"):
                    set1 = st.columns(2)
                    sw_color = set1[0].color_picker("Color", sw_defaults["col"], key="eval_col_SW")
                    sw_xmin = st.number_input("X Min Sw", value=sw_defaults["xmin"], key="eval_xmin_SW")
                    sw_xmax = st.number_input("X Max Sw", value=sw_defaults["xmax"], key="eval_xmax_SW")
                    sw_depth = st.slider("Isolate Depth Sw", min_value=depth_range[0], max_value=depth_range[1], value=sw_defaults["depth"], key="eval_depth_SW")

                sw_dest = st.multiselect("🔗 Send 'SW' to other viewers:", ["Recorded Logs", "Smoothed Logs", "Multi-Track Viewer"], key="sw_dest")
                
                if n_val == 0:
                    st.warning("⚠️ The 'n' exponent can't be zero.")
                elif st.button("Calculate Water Saturation (Sw)"):
                    # 1. Calculate the curve parameters, guarding against zero/negative
                    # porosity and zero resistivity so bad zones produce NaN (skipped)
                    # instead of inf or silently wrong values feeding the Net Pay flag.
                    poro_safe = df_filtered[poro_input].where(df_filtered[poro_input] > 0)
                    rt_safe = df_filtered[rt_curve].where(df_filtered[rt_curve] > 0)

                    f_factor = a_val / (poro_safe ** m_val)
                    sw_base = ((f_factor * rw_val) / rt_safe).clip(lower=0)  # avoid negative base ** fractional power
                    calculated_sw = (sw_base ** (1 / n_val)).clip(0, 1)
                    
                    # 2. Save directly to master_log_df to clear the security filter
                    st.session_state.master_log_df['SW'] = calculated_sw
                    df_filtered['SW'] = calculated_sw
                    
                    # 3. Route it to selected viewers
                    route_calculated_curve('SW', sw_dest)
                    st.success("✅ Water Saturation Calculated & Saved permanently!")
                    
                    # 4. Generate the persistent plot
                    fig_sw = go.Figure()
                    fig_sw.add_trace(go.Scatter(x=df_filtered['SW'], y=df_filtered['DEPTH'], mode='lines', line=dict(color=sw_color, width=1.5), fill='tozerox', fillcolor='rgba(0, 206, 209, 0.3)'))
                    fig_sw.update_layout(plot_bgcolor='white', height=600, margin=dict(t=150, b=20, l=50, r=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"),
                        xaxis=dict(title="Water Saturation (SW)", side="top", range=[sw_xmin, sw_xmax], showgrid=True, gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black"),
                        yaxis=dict(title="Depth (m)", range=[sw_depth[1], sw_depth[0]], showgrid=True, gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black"))
                    
                    # 5. Cache the figure and trigger application rerun
                    st.session_state['fig_sw'] = fig_sw
                    st.rerun()

                # Display the chart safely outside the execution frame
                if 'fig_sw' in st.session_state:
                    st.plotly_chart(st.session_state['fig_sw'], use_container_width=True)
            
            st.markdown("---")

            # --- 8. RESERVOIR IDENTIFICATION (FLAG) ---
            st.markdown("#### 8. Reservoir Identification (Flag)")
            st.info("Identify reservoir zones based on conditional logic (e.g., Vsh <= 0.4 AND Sw <= 0.7).")

            vsh_options = [col for col in df_filtered.columns if 'VSH' in col.upper()]
            sw_options = [col for col in df_filtered.columns if 'SW' in col.upper()]

            if not vsh_options or not sw_options:
                st.warning("⚠️ Please calculate Volume of Shale and Water Saturation (Sw) first to generate a flag!")
            else:
                res_c1, res_c2, res_c3 = st.columns(3)
                with res_c1: res_vsh_curve = st.selectbox("Select Vsh Curve:", vsh_options, key="res_vsh_sel")
                with res_c2: res_vsh_op_str = st.selectbox("Vsh Operator:", ["<", "<=", "==", ">=", ">"], index=1, key="res_vsh_op")
                with res_c3: res_vsh_cutoff = st.number_input("Vsh Cutoff:", value=0.40, step=0.05, key="res_vsh_cut")

                res_c4, res_c5, res_c6 = st.columns(3)
                with res_c4: res_sw_curve = st.selectbox("Select Sw Curve:", sw_options, key="res_sw_sel")
                with res_c5: res_sw_op_str = st.selectbox("Sw Operator:", ["<", "<=", "==", ">=", ">"], index=1, key="res_sw_op")
                with res_c6: res_sw_cutoff = st.number_input("Sw Cutoff:", value=0.70, step=0.05, key="res_sw_cut")

                res_defaults = {"col": "#39FF14", "depth": (depth_range[0], depth_range[1])}
                with st.expander("⚙️ Reservoir Flag Plot Settings"):
                    set1, set2 = st.columns(2)
                    res_color = set1.color_picker("Flag Color", res_defaults["col"], key="eval_col_RES")
                    res_depth = set2.slider("Depth Range (Flag)", min_value=depth_range[0], max_value=depth_range[1], value=res_defaults["depth"], key="eval_depth_RES")

                # 🔗 Limited routing options specifically for the flag!
                res_dest = st.multiselect("🔗 Send 'RES_FLAG' to viewers:", ["Multi-Track Viewer"], default=["Multi-Track Viewer"], key="res_dest")

                if st.button("Generate Reservoir Flag"):
                    # Helper function to map string operators to Python logic
                    def apply_operator(op_str, val_array, cutoff):
                        if op_str == "<": return val_array < cutoff
                        if op_str == "<=": return val_array <= cutoff
                        if op_str == "==": return val_array == cutoff
                        if op_str == ">=": return val_array >= cutoff
                        if op_str == ">": return val_array > cutoff
                        return val_array <= cutoff

                    cond_vsh = apply_operator(res_vsh_op_str, df_filtered[res_vsh_curve], res_vsh_cutoff)
                    cond_sw = apply_operator(res_sw_op_str, df_filtered[res_sw_curve], res_sw_cutoff)

                    # 1. Create Flag: 1 if both conditions met, else 0
                    calculated_flag = np.where(cond_vsh & cond_sw, 1, 0)
                    
                    # 2. Save securely to master_log_df
                    st.session_state.master_log_df['RES_FLAG'] = calculated_flag
                    df_filtered['RES_FLAG'] = calculated_flag
                    
                    # 3. Route to Multi-Track Viewer
                    route_calculated_curve('RES_FLAG', res_dest)
                    st.success(f"✅ Reservoir Flag Generated & Saved permanently!")

                    # 4. Create and update plot
                    fig_res = go.Figure()
                    fig_res.add_trace(go.Scatter(
                        x=df_filtered['RES_FLAG'], 
                        y=df_filtered['DEPTH'], 
                        mode='lines', 
                        line=dict(color=res_color, width=1.5),
                        fill='tozerox',
                        fillcolor=f"rgba({int(res_color[1:3], 16)}, {int(res_color[3:5], 16)}, {int(res_color[5:7], 16)}, 0.6)",
                        name="Reservoir"
                    ))
                    
                    fig_res.update_layout(
                        plot_bgcolor='white', height=600, margin=dict(t=150, b=20, l=50, r=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"),
                        xaxis=dict(title="Reservoir Flag", side="top", range=[0, 1.2], showgrid=False, zeroline=False, dtick=1, linecolor="black", mirror=True),
                        yaxis=dict(title="Depth (m)", range=[res_depth[1], res_depth[0]], showgrid=True, gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black")
                    )
                    
                    # 5. Cache it and refresh
                    st.session_state['fig_res'] = fig_res
                    st.rerun()

                # Display the chart safely
                if 'fig_res' in st.session_state:
                    st.plotly_chart(st.session_state['fig_res'], use_container_width=True)
            
            st.markdown("---")

            # --- 9. ROCK PHYSICS & IMPEDANCE PROFILES ---
            st.markdown("#### 9. Rock Physics & Impedance Profiles")
            st.info("Calculate Acoustic & Shear Impedance and view depth profiles color-coded by Porosity.")
            
            rp_c1, rp_c2, rp_c3 = st.columns(3)
            with rp_c1: vp_curve = st.selectbox("Select Vp (Compressional) Curve:", available_curves, key="rp_vp_sel")
            with rp_c2: vs_curve = st.selectbox("Select Vs (Shear) Curve:", available_curves, key="rp_vs_sel")
            with rp_c3: den_curve = st.selectbox("Select Density Curve:", available_curves, key="rp_den_sel")
            
            rp_c4, rp_c5 = st.columns(2)
            phi_options = [col for col in df_filtered.columns if 'PHI' in col.upper()]
            with rp_c4: 
                color_curve = st.selectbox("Color Code By (Porosity):", phi_options if phi_options else available_curves, key="rp_color_sel")
            with rp_c5:
                rp_dest = st.multiselect("🔗 Send Impedances to viewers:", ["Multi-Track Viewer", "Recorded Logs", "Smoothed Logs"], default=["Multi-Track Viewer", "Recorded Logs", "Smoothed Logs"], key="rp_dest")

            if st.button("Calculate Impedances & Plot Profiles"):
                # 1. Perform the calculations
                calculated_ai = df_filtered[vp_curve] * df_filtered[den_curve]
                calculated_si = df_filtered[vs_curve] * df_filtered[den_curve]
                
                # 2. Save both permanently into master_log_df to escape cleaning cycles
                st.session_state.master_log_df['ACOUSTIC_IMP'] = calculated_ai
                st.session_state.master_log_df['SHEAR_IMP'] = calculated_si
                df_filtered['ACOUSTIC_IMP'] = calculated_ai
                df_filtered['SHEAR_IMP'] = calculated_si
                
                # 3. Route both parameters to the selected logs / viewers
                route_calculated_curve('ACOUSTIC_IMP', rp_dest)
                route_calculated_curve('SHEAR_IMP', rp_dest)
                
                st.success("✅ Acoustic and Shear Impedances Calculated & Saved permanently across selected tabs!")
                
                # 4. Generate the persistent subplot figure
                fig_rp = make_subplots(
                    rows=1, cols=5, shared_yaxes=True, horizontal_spacing=0.02,
                    subplot_titles=("Vp Profile", "Vs Profile", "Acoustic Impedance", "Shear Impedance", "Density Profile")
                )
                
                plot_curves = [vp_curve, vs_curve, 'ACOUSTIC_IMP', 'SHEAR_IMP', den_curve]
                
                for i, curve in enumerate(plot_curves):
                    fig_rp.add_trace(go.Scatter(
                        x=df_filtered[curve], 
                        y=df_filtered['DEPTH'],
                        mode='markers',
                        marker=dict(
                            color=df_filtered[color_curve],
                            colorscale='Jet',
                            showscale=True if i == 4 else False, 
                            colorbar=dict(title=color_curve, x=1.05) if i == 4 else None,
                            size=4
                        ),
                        name=curve
                    ), row=1, col=i+1)
                    
                    fig_rp.update_xaxes(
                        title_text=curve, side="top", showgrid=True, gridcolor="lightgrey", 
                        griddash="dash", mirror=True, showline=True, linecolor="black", row=1, col=i+1
                    )
                
                fig_rp.update_yaxes(
                    title_text="Depth (m)", range=[depth_range[1], depth_range[0]], showgrid=True, 
                    gridcolor="lightgrey", griddash="dash", mirror=True, showline=True, linecolor="black", row=1, col=1
                )
                
                fig_rp.update_layout(
                    plot_bgcolor='white', height=800, 
                    margin=dict(t=150, b=20, l=50, r=20),
                    showlegend=False, 
                    legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)") 
                )
                
                # 5. Cache the layout state and refresh
                st.session_state['fig_rp'] = fig_rp
                st.rerun()

            # Render plot safely outside the execution scope so it doesn't vanish
            if 'fig_rp' in st.session_state:
                st.plotly_chart(st.session_state['fig_rp'], use_container_width=True)

            st.markdown("---")
            
            # --- 10. NET PAY CUTOFFS ---
            st.markdown("#### 10. Net Pay Cutoffs")
            poro_candidates = [col for col in df_filtered.columns if 'PHI' in col.upper()]
            vsh_candidates = [col for col in df_filtered.columns if 'VSH' in col.upper()]
            pay_c0, pay_c1, pay_c2, pay_c3 = st.columns(4)
            with pay_c0: vsh_src = st.selectbox("Vsh source:", vsh_candidates if vsh_candidates else ["N/A"], key="eval_pay_vsh")
            default_pay_poro_idx = poro_candidates.index('PHIE') if 'PHIE' in poro_candidates else 0
            with pay_c1: pay_poro_input = st.selectbox("Porosity source:", poro_candidates if poro_candidates else ["N/A"], index=default_pay_poro_idx if poro_candidates else 0, key="eval_pay_poro")
            with pay_c2: vsh_cutoff_pay = st.number_input("Max Vsh Cutoff:", value=0.40, step=0.05, key="eval_pay_vsh_cut")
            with pay_c3: phi_cutoff_pay = st.number_input("Min Phi Cutoff:", value=0.08, step=0.01, key="eval_pay_phi_cut")
            sw_cutoff_pay = st.number_input("Max Sw Cutoff:", value=0.50, step=0.05, key="eval_pay_sw_cut")

            if st.button("Calculate Net Pay"):
                if not all(col in df_filtered.columns for col in [vsh_src, 'SW']) or not poro_candidates:
                    st.error("⚠️ Ensure VSH, PHI, and SW are calculated!")
                else:
                    is_pay = (df_filtered[vsh_src] <= vsh_cutoff_pay) & (df_filtered[pay_poro_input] >= phi_cutoff_pay) & (df_filtered['SW'] <= sw_cutoff_pay)
                    depth_step = abs(df_filtered['DEPTH'].iloc[1] - df_filtered['DEPTH'].iloc[0])
                    st.success("✅ Net Pay Calculated!")
                    st.metric(" Total Net Pay Thickness", f"{is_pay.sum() * depth_step:.2f} m")

        # --- TAB 9: MACHINE LEARNING ---
        with tab_ml:
            # ==========================================
            # PHASE 1: PETROPHYSICAL KNOWLEDGE BASE
            # ==========================================
            PETROPHYSICAL_GUIDE = {
                "DT": {
                    "recommended": ["NPHI", "RHOB", "GR"],
                    "explanation": "Sonic logs (DT) are primarily driven by rock porosity and compaction. Neutron (NPHI) and Density (RHOB) are excellent physical indicators."
                },
                "RHOB": {
                    "recommended": ["NPHI", "DT", "GR"],
                    "explanation": "Bulk Density (RHOB) depends heavily on matrix lithology and fluid porosity, correlating tightly with Neutron and Sonic indicators."
                },
                "NPHI": {
                    "recommended": ["RHOB", "DT", "GR"],
                    "explanation": "Neutron Porosity (NPHI) responds to hydrogen index, pairing mathematically and physically with Density/Sonic overlays."
                },
                "CALI": {
                    "recommended": ["BS", "GR"],
                    "explanation": "Caliper (CALI) measures borehole physical geometry. It is best predicted using Bit Size (BS) or shale indicators like Gamma Ray (GR) to capture washout zones."
                }
            }
            
            st.write("### AI Log Predictor (Random Forest)")
            ml_col1, ml_col2 = st.columns(2)
            
            with ml_col1:
                target_curve = st.selectbox("Target Curve:", available_curves, index=0)
                # Look up physical validity for the selected target curve
                target_base = next((k for k in PETROPHYSICAL_GUIDE.keys() if k in target_curve.upper()), None)
            
                if target_base:
                    rec_features = PETROPHYSICAL_GUIDE[target_base]["recommended"]
                    explanation = PETROPHYSICAL_GUIDE[target_base]["explanation"]
                    
                    st.info(f"💡 **Petrophysical Guidance for {target_curve}:** {explanation}")
                    st.caption(f"👍 **Highly Recommended Features to use:** {', '.join(rec_features)}")
            
            with ml_col2:
                default_features = [c for c in available_curves if c != target_curve][:3]
                feature_curves = st.multiselect("Feature Curves:", available_curves, default=default_features)
                
                # Validate user feature selections against physical recommendations
                if target_base and feature_curves:
                    selected_upper = [f.upper() for f in feature_curves]
                    has_valid_pair = any(any(rec in sel for sel in selected_upper) for rec in PETROPHYSICAL_GUIDE[target_base]["recommended"])
                    
                    if not has_valid_pair:
                        st.warning(
                            "⚠️ **Geological Disconnect Detected:** The selected feature curves do not share a known "
                            "physical dependency with your target curve. The algorithm will still compute a mathematical match, "
                            "but the resulting synthetic profile may be scientifically invalid."
                        )
                        
            # ==========================================
            # PHASE 3: GLOBAL DATA CENTER UI
            # ==========================================
            st.divider()
            st.subheader("Global Subsurface Learning Engine")
        
            # Button to trigger a background sweep of the database folder
            if st.button("🔄 Sync & Rebuild Global Repository Data"):
                with st.spinner("Parsing repository wells and standardizing logs..."):
                    # Unpack the two returned items
                    master_df, skipped = rebuild_master_subsurface_database()
                    
                    if master_df is not None:
                        st.success(f"Successfully compiled database! Total data points: {len(master_df)} across {master_df['WELL_NAME'].nunique()} compiled files.")
                        
                        # Display transparent warnings for skipped files
                        if skipped:
                            st.warning("⚠️ **Some files were ignored:**")
                            for skip_msg in skipped:
                                st.write(f"- {skip_msg}")
                    else:
                        st.warning("The repository folder is currently empty or no files contained valid standard log curves.")
        
            # ML Source Selector toggle
            training_mode = st.radio(
                "Select Model Training Data Range:",
                ["Current Well Only (Local Mode)", "All Repository Wells Combined (Global Data Center Mode)"]
            )
        
            # Set the active training dataframe based on user selection
            if training_mode == "All Repository Wells Combined (Global Data Center Mode)":
                if os.path.exists(MASTER_DATABASE_PATH):
                    training_df = pd.read_parquet(MASTER_DATABASE_PATH)
                    st.info(f" **Global Mode Active:** Training model on {len(training_df):,} historical data points.")
                    # --- PHASE 4: SISTER WELL DISCOVERY UI ---
                    sister_wells = discover_sister_wells(df_filtered, training_df)
                    if sister_wells and sister_wells[0]['variance_score'] > 0.01: # Avoid matching with itself
                        st.success(
                            f"🏢 **Lithological Analog Discovered:** Based on GR footprint, your active wellbore shows "
                            f"strong structural similarities with **{sister_wells[0]['well_name']}** "
                            f"(Variance: {sister_wells[0]['variance_score']:.2f})."
                        )
                else:
                    st.warning("No Master Database file found. Defaulting back to current active log.")
                    training_df = df_filtered.copy()
            else:
                training_df = df_filtered.copy()
                
            if st.button("Train AI & Predict"):
                if len(feature_curves) < 1:
                    st.warning("Please select at least one Feature Curve.")
                else:
                    with st.spinner(" Training Random Forest Model... Please wait!"):
                        # 1. Clean the training data
                        ml_train_data = training_df.dropna(subset=feature_curves + [target_curve])
                        X_model = ml_train_data[feature_curves]
                        y_model = ml_train_data[target_curve]

                        # 2. Split into Train & Test (FIXED)
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_model, y_model, test_size=0.2, random_state=42
                        )

                        # 3. Train the Model
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train)

                        train_r2 = model.score(X_train, y_train)
                        test_r2 = model.score(X_test, y_test)

                        # 4. Predict on the CURRENT well (df_filtered)
                        pred_name = f'{target_curve}_PREDICTED'
                        predict_df = df_filtered.dropna(subset=feature_curves).copy()
                        
                        # Generate raw AI predictions
                        raw_predictions = model.predict(predict_df[feature_curves])
                        
                        # --- PHASE 4: HARD PHYSICAL CONSTRAINTS ---
                        # Prevent negative values for standard logs (Porosity, Resistivity, Sonic, Caliper cannot be < 0)
                        raw_predictions = np.clip(raw_predictions, 0.1, None)
                        
                        # If predicting CALI and Bit Size (BS) is present, Caliper generally shouldn't be much smaller than the bit
                        if "CALI" in target_curve.upper() and "BS" in predict_df.columns:
                            bs_min = predict_df["BS"].min() - 0.25 # Allow for a tiny mudcake tolerance
                            raw_predictions = np.clip(raw_predictions, bs_min, None)

                        predict_df[pred_name] = raw_predictions
                        
                        st.session_state.df[pred_name] = np.nan
                        st.session_state.df.loc[predict_df.index, pred_name] = predict_df[pred_name]
                        
                        # Add AI curve to global list
                        if pred_name not in st.session_state.available_curves:
                            st.session_state.available_curves.append(pred_name)

                        st.success(f"Test R² Score: {test_r2:.2f}  (Train R²: {train_r2:.2f}). '{pred_name}' added globally.")
                        if test_r2 < 0.3:
                            st.warning("⚠️ Low test R² — these feature curves may not predict this target well. Try a different combination.")
                        elif train_r2 - test_r2 > 0.3:
                            st.info("ℹ️ Train R² is much higher than test R², which suggests the model may be overfitting.")
                        
                        # 5. Plot the Results (FIXED to use df_filtered)
                        fig_ml = go.Figure()
                        
                        # We use df_filtered here to plot the original target curve if it exists
                        original_plot_df = df_filtered.dropna(subset=[target_curve])
                        fig_ml.add_trace(go.Scatter(x=original_plot_df[target_curve], y=original_plot_df['DEPTH'], mode='lines', name=f'Original {target_curve}', line=dict(color='black', width=3)))
                        fig_ml.add_trace(go.Scatter(x=predict_df[pred_name], y=predict_df['DEPTH'], mode='lines', name=f'AI Predicted', line=dict(color='red', width=2, dash='dash')))
                        
                        fig_ml.update_yaxes(autorange="reversed")
                        fig_ml.update_layout(
                            margin=dict(t=150, b=20, l=50, r=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)")
                        )
                        st.plotly_chart(fig_ml, use_container_width=True)
                        st.session_state['ml_fig'] = fig_ml
                        
        # --- TAB 10: REPORT GENERATOR ---
        import datetime
        import tempfile
        import io
        import gc
        import base64
        import html
        import os
        
        with tab_report:  
        
            def pdf_safe_text(value):
                """FPDF's built-in core fonts (Arial etc.) only support Latin-1.
                LAS header/description text can legitimately contain characters
                outside that range (°, µ, –, non-English names...) which would
                otherwise throw and abort the whole report. Replace instead of
                crashing."""
                if value is None:
                    return ""
                return str(value).encode('latin-1', errors='replace').decode('latin-1')
        
            class PremiumPetrophysicsReport(FPDF):
                def header(self):
                    self.set_font('Arial', 'B', 15)
                    self.set_text_color(30, 58, 138)
                    self.cell(0, 10, 'Comprehensive Subsurface Evaluation & Petrophysical Report', border=0, ln=1, align='C')
                    self.set_font('Arial', 'I', 9)
                    self.set_text_color(100, 116, 139)
                    self.cell(0, 5, f'Report Timestamp: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}', border=0, ln=1, align='C')
                    self.ln(6)
        
                def footer(self):
                    self.set_y(-15)
                    self.set_font('Arial', 'I', 8)
                    self.set_text_color(148, 163, 184)
                    self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
                def add_main_heading(self, title):
                    self.ln(4)
                    self.set_font('Arial', 'B', 12)
                    self.set_fill_color(30, 58, 138)
                    self.set_text_color(255, 255, 255)
                    self.cell(0, 9, f'  {title}', border=0, ln=1, align='L', fill=True)
                    self.ln(2)
        
                def add_sub_heading(self, title):
                    self.set_font('Arial', 'B', 10)
                    self.set_fill_color(241, 245, 249)
                    self.set_text_color(15, 23, 42)
                    self.cell(0, 7, f'  {title}', border=0, ln=1, align='L', fill=True)
                    self.ln(2)
        
                def add_metric_row(self, label, value, unit=""):
                    self.set_font('Arial', 'B', 10)
                    self.set_text_color(71, 85, 105)
                    self.cell(75, 7, f" {label}:", border=1)
                    self.set_font('Arial', '', 10)
                    self.set_text_color(15, 23, 42)
                    self.cell(115, 7, f" {value} {unit}", border=1, ln=1)
        
                def add_metadata_table(self, df_meta, col_widths=[30, 25, 55, 80]):
                    self.set_font('Arial', 'B', 9)
                    self.set_fill_color(226, 232, 240)
                    self.set_text_color(15, 23, 42)
                    cols = df_meta.columns.tolist()
                    for i, col in enumerate(cols):
                        self.cell(col_widths[i], 7, pdf_safe_text(col), border=1, fill=True, align='C')
                    self.ln()
                    self.set_font('Arial', '', 8)
                    self.set_text_color(51, 65, 85)
                    for _, row in df_meta.iterrows():
                        if self.get_y() > 260: self.add_page()
                        for i, col in enumerate(cols):
                            val_str = pdf_safe_text(row[col])[:45]
                            self.cell(col_widths[i], 6, f" {val_str}", border=1)
                        self.ln()
                    self.ln(3)
        
                def add_plotly_track(self, fig, width=175):
                    try:
                        # Use in-memory buffer to save RAM instead of writing to disk
                        img_buffer = io.BytesIO()
                        fig.write_image(img_buffer, format="png", width=850, height=450)
                        
                        if self.get_y() > 190: self.add_page()
                        self.image(img_buffer, w=width)
                        self.ln(4)
                        
                        # Close the buffer immediately
                        img_buffer.close()
                        
                        # Force Python to clear the kaleido renderer from memory
                        gc.collect()
                        
                    except Exception as e:
                        self.set_font('Arial', 'I', 9)
                        self.set_text_color(220, 38, 38)
                        self.cell(0, 7, f"  [Log image profile compiled via system cache. Engine status offline: {e}]", ln=1)
                        self.ln(2)
        
            # Execution Interface
            st.markdown("### 📄 Enterprise Report Generation Hub")
            st.markdown("Compile all operations, LAS text headings, custom smoothing logs, and complete 10-track Formation Evaluations into a structured asset dossier.")
        
            report_name_input = st.text_input("Enter Output Asset Document Name:", value="Complete_Field_Petrophysics_Report")
        
            if st.button("Compile Full Report Suite", type="primary"):
                with st.spinner("Analyzing log archives, mining metadata, and rendering charts..."):
                    
                    pdf = PremiumPetrophysicsReport()
                    pdf.add_page()
                    
                    # --- SECTION 1: LAS HEADER METADATA ---
                    pdf.add_main_heading("1. LAS Archive Document Metadata Headers")
                    if 'las_well_info' in st.session_state:
                        pdf.add_sub_heading("1.1 Well Component Attributes Summary")
                        pdf.add_metadata_table(st.session_state['las_well_info'], col_widths=[35, 25, 55, 75])
                        
                    # --- SECTION 2: CORE SIGNAL PROCESSING LOGS ---
                    pdf.add_main_heading("2. Core Signal Processing Logs")
                    
                    # Loop through ALL Recorded Logs charts
                    if 'recorded_logs_figs_list' in st.session_state and len(st.session_state['recorded_logs_figs_list']) > 0:
                        pdf.add_sub_heading("2.1 Baseline Recorded Signal Array")
                        for idx, rec_fig in enumerate(st.session_state['recorded_logs_figs_list']):
                            pdf.add_plotly_track(rec_fig)
        
                    # Loop through ALL Smoothed Logs charts
                    if 'smoothed_logs_figs_list' in st.session_state and len(st.session_state['smoothed_logs_figs_list']) > 0:
                        pdf.add_sub_heading("2.2 De-noised / Smoothed Evaluation Signal Array")
                        for idx, sm_fig in enumerate(st.session_state['smoothed_logs_figs_list']):
                            pdf.add_plotly_track(sm_fig)
        
                    # Loop through ALL Histogram charts
                    if 'histogram_figs_list' in st.session_state and len(st.session_state['histogram_figs_list']) > 0:
                        pdf.add_sub_heading("2.3 Data Distribution Diagnostics (Histograms)")
                        for idx, hist_fig in enumerate(st.session_state['histogram_figs_list']):
                            pdf.add_plotly_track(hist_fig)
        
                    # THIS IS WHAT PULLS THE MULTI-TRACK VIEWER INTO THE REPORT
                    if 'multi_track_fig' in st.session_state:
                        pdf.add_sub_heading("2.4 Unified Composite Multi-Track Viewer Profile")
                        pdf.add_plotly_track(st.session_state['multi_track_fig'])
                        
                    # --- SECTION 3: FORMATION EVALUATION (ALL 10 TRACKS) ---
                    pdf.add_main_heading("3. Advanced Formation Evaluation Suite")
                    
                    pdf.add_sub_heading("3.1 Linear Volume of Shale (Vsh)")
                    if 'fig_vsh' in st.session_state: pdf.add_plotly_track(st.session_state['fig_vsh'])
                    
                    pdf.add_sub_heading("3.2 Non-Linear Shale Correction (Larionov)")
                    if 'fig_vshc' in st.session_state: pdf.add_plotly_track(st.session_state['fig_vshc'])
        
                    pdf.add_sub_heading("3.3 Bulk Density Porosity Profile (PhiD)")
                    if 'fig_phi' in st.session_state: pdf.add_plotly_track(st.session_state['fig_phi'])
        
                    pdf.add_sub_heading("3.4 Acoustic Sonic Porosity Profile (PhiS)")
                    if 'fig_phis' in st.session_state: pdf.add_plotly_track(st.session_state['fig_phis'])
        
                    pdf.add_sub_heading("3.5 Total Combination Porosity Matrix (PhiT)")
                    if 'fig_phit' in st.session_state: pdf.add_plotly_track(st.session_state['fig_phit'])
        
                    pdf.add_sub_heading("3.6 Effective Hydrocarbon Space Porosity Matrix (PhiE)")
                    if 'fig_phie' in st.session_state: pdf.add_plotly_track(st.session_state['fig_phie'])
        
                    pdf.add_sub_heading("3.7 Fluid Saturation Profile (Sw - Archie)")
                    if 'fig_sw' in st.session_state: pdf.add_plotly_track(st.session_state['fig_sw'])
        
                    pdf.add_sub_heading("3.8 Net Pay Matrix Qualifier Flags")
                    if 'fig_res' in st.session_state: pdf.add_plotly_track(st.session_state['fig_res'])
        
                    pdf.add_sub_heading("3.9 Structural Rock Physics & Elastic Impedance Profiles")
                    if 'fig_rp' in st.session_state: pdf.add_plotly_track(st.session_state['fig_rp'])
        
                    # --- SECTION 4: DIAGNOSTICS & ML ---
                    pdf.add_main_heading("4. Crossplots & Automated Intelligence Profiles")
                    if 'crossplot_fig' in st.session_state:
                        pdf.add_sub_heading("4.1 Dual Lithology/Fluid Crossplot Mapping")
                        pdf.add_plotly_track(st.session_state['crossplot_fig'])
                    if 'ml_fig' in st.session_state:
                        pdf.add_sub_heading("4.2 AI Synthetic Log Verification Vector (Actual vs Predicted)")
                        pdf.add_plotly_track(st.session_state['ml_fig'])
        
                    # Output to Streamlit Downloader
                    try:
                        # errors='replace' is a last-resort safety net: any LAS header
                        # text (well descriptions, etc.) containing characters outside
                        # Latin-1 (° µ – etc.) is sanitized earlier via pdf_safe_text(),
                        # but this guards against anything that slipped through.
                        # Modern fpdf2 outputs a bytearray directly, while legacy fpdf outputs a string.
                        # This safely handles both to prevent encode crashes.
                        raw_output = pdf.output()
                        if isinstance(raw_output, str):
                            pdf_bytes = raw_output.encode('latin1', errors='replace')
                        else:
                            pdf_bytes = bytes(raw_output)

                        # --- FIX: don't embed the file as a base64 data-URI <a href> link ---
                        # The old code built a giant base64 string (file size x~1.33) and
                        # injected it into the page via st.markdown(unsafe_allow_html=True).
                        # That round-trips fine on localhost, but over the network on
                        # Streamlit Community Cloud that huge single payload can get
                        # truncated/corrupted in transit or when the browser's data-URI
                        # download handling kicks in - producing a PDF that "compiles"
                        # successfully server-side but fails to open once downloaded
                        # ("We can't open this file / Something went wrong").
                        # st.download_button streams the bytes through Streamlit's own
                        # file-serving endpoint instead, which is the reliable, native
                        # way to offer downloads and sidesteps that failure mode.
                        safe_report_name = "".join(
                            c for c in (report_name_input.strip() or "Petrophysics_Report")
                            if c.isalnum() or c in (" ", "_", "-")
                        ).strip() or "Petrophysics_Report"

                        st.session_state['generated_report_bytes'] = pdf_bytes
                        st.session_state['generated_report_name'] = f"{safe_report_name}.pdf"
                        st.success("🚀 Compilation complete! Header registry and all evaluation layers parsed cleanly.")
                    except Exception as e:
                        st.error(f"Asset compiling fault detected: {e}")

            # Rendered outside the "Compile" button's if-block on purpose:
            # st.download_button triggers its own script rerun the moment it's
            # clicked, which would make st.button(...) above evaluate to False
            # again on that rerun and the button would vanish mid-download.
            # Keeping the PDF bytes in session_state means the download button
            # stays put across reruns, so the download reliably completes.
            if 'generated_report_bytes' in st.session_state:
                st.download_button(
                    label="📥 Download Comprehensive Subsurface Dossier",
                    data=st.session_state['generated_report_bytes'],
                    file_name=st.session_state.get('generated_report_name', 'Petrophysics_Report.pdf'),
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                )
        # --- EXPORT DATA ENGINE ---
        st.sidebar.markdown("---")
        st.sidebar.header("💾 Export Data")
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(label="⬇️ Download Processed Data (CSV)", data=csv, file_name=f"{well_name}_processed.csv", mime='text/csv')
       
        # --- 🤖 MASTER COPILOT TOGGLE CONTROL ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("🤖 AI Petrophysics Copilot")
        copilot_enabled = st.sidebar.toggle(
            "Activate AI Copilot", value=False,
            help="Chat with an AI about your well log data — Cloud works instantly for anyone, Local runs fully offline via Ollama."
        )

        system_copilot_instruction = """
        You are an expert Petrophysics AI Copilot integrated into an advanced well log dashboard. 
        Your job is to help users analyze well log data, interpret crossplots, evaluate fluid saturations, and read dashboard charts. 
        Be concise, accurate, and highly technical. If looking at an image, focus carefully on data lines, curves, grid scales, and highlighted anomalies.
        """

        def render_copilot_chat(text_query_fn, vision_query_fn, engine_label):
            """Shared chat UI used by both the Cloud and Local engines, so the
            interaction logic (history, image attach, clear button) is written once."""
            chat_container = st.sidebar.container(height=350)
            with chat_container:
                for message in st.session_state['ai_chat_history']:
                    if message["role"] == "user":
                        st.markdown(f"**🧑‍💻 You:** {message['content']}")
                        if "images" in message:
                            st.markdown(f"*(📎 {len(message['images'])} image(s) attached)*")
                    else:
                        st.markdown(f"**🤖 AI:** {message['content']}")
                        st.markdown("---")

            uploaded_imgs = st.sidebar.file_uploader(
                "📎 Attach Image(s)", type=["png", "jpg", "jpeg"],
                accept_multiple_files=True, key=f"copilot_img_{engine_label}"
            )
            user_query = st.sidebar.chat_input(
                "Ask about the logs or analyze the screenshot(s)...", key=f"copilot_input_{engine_label}"
            )

            if user_query:
                if uploaded_imgs:
                    combined_ai_response = ""
                    total_images = len(uploaded_imgs)
                    for idx, uploaded_img in enumerate(uploaded_imgs, start=1):
                        with st.sidebar.spinner(f"Analyzing image {idx}/{total_images}..."):
                            img = Image.open(uploaded_img)
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            img.thumbnail((800, 800))
                            buffered = io.BytesIO()
                            img.save(buffered, format="JPEG", quality=85)
                            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                            enhanced_prompt = (
                                f"Please look extremely closely at the text, numbers, and UI "
                                f"elements in this image to answer accurately: {user_query}"
                            )
                            img_response = vision_query_fn(enhanced_prompt, img_b64)
                            combined_ai_response += (
                                f"### 📊 Image {idx} — {uploaded_img.name}:\n{img_response}\n\n---\n\n"
                            )
                    st.session_state['ai_chat_history'].append({
                        "role": "user",
                        "content": user_query,
                        "images": [base64.b64encode(u.getvalue()).decode("utf-8") for u in uploaded_imgs]
                    })
                    st.session_state['ai_chat_history'].append({
                        "role": "assistant", "content": combined_ai_response
                    })
                    st.rerun()

                else:
                    st.session_state['ai_chat_history'].append({"role": "user", "content": user_query})
                    with st.sidebar.spinner(f"Thinking with {engine_label}..."):
                        ai_response = text_query_fn(st.session_state['ai_chat_history'])
                    st.session_state['ai_chat_history'].append({
                        "role": "assistant", "content": ai_response
                    })
                    st.rerun()

            if st.sidebar.button("🗑️ Clear Chat History", key=f"copilot_clear_{engine_label}"):
                st.session_state['ai_chat_history'] = []
                st.rerun()

        if copilot_enabled:
            engine_choice = st.sidebar.radio(
                "AI Engine:",
                ["☁️ Cloud (works instantly)", "💻 Local (Ollama, fully offline)"],
                key="ai_engine_choice",
                help="Cloud calls Anthropic's API and works for anyone who opens this app, deployed or not. "
                     "Local uses Ollama on YOUR machine and never sends data anywhere."
            )

            # ═══════════════════════════════════════════════════════════
            # ☁️ CLOUD ENGINE — the fix for the old Ollama-only limitation.
            # Works the same whether this app is run locally or deployed
            # and opened by someone else, unlike a localhost-only engine.
            # ═══════════════════════════════════════════════════════════
            if engine_choice.startswith("☁️"):
                if not get_anthropic_api_key():
                    st.sidebar.warning("⚠️ **No API key configured**")
                    st.sidebar.markdown(
                        """
**Add an Anthropic API key to enable the Cloud engine:**

**Deployed on Streamlit Cloud:**
App settings → Secrets → add:
```
ANTHROPIC_API_KEY = "sk-ant-..."
```

**Running locally:**
Create `.streamlit/secrets.toml` in your project folder with the same line.

Get a key at [console.anthropic.com](https://console.anthropic.com/).
Or switch to the **Local (Ollama)** engine above to run fully offline instead.
                        """
                    )
                else:
                    st.sidebar.success("✅ **AI Copilot Ready (Cloud)**")
                    render_copilot_chat(
                        text_query_fn=lambda history: query_cloud_claude(history, system_copilot_instruction),
                        vision_query_fn=lambda prompt, img_b64: query_cloud_claude(
                            [{"role": "user", "content": prompt}], system_copilot_instruction, images_b64=[img_b64]
                        ),
                        engine_label="Cloud"
                    )

            # ═══════════════════════════════════════════════════════════
            # 💻 LOCAL ENGINE — Ollama running on the visitor's own machine.
            # Only reachable when Python and Ollama share the same host,
            # i.e. when `streamlit run app.py` is launched locally.
            # ═══════════════════════════════════════════════════════════
            else:
                ollama_info = check_ollama_status()

                if ollama_info["status"] == "cloud":
                    st.sidebar.warning("☁️ **Cloud Link Detected**")
                    st.sidebar.markdown(
                        """
**The Local engine needs Ollama on the SAME machine running this app.**

When this app is opened via a deployed link, Python runs on a remote
server — not your computer — so it can never reach an Ollama instance
on your laptop. This isn't a bug, it's how cloud hosting works.

**✅ Easiest fix:** switch to the **☁️ Cloud** engine above — it works
immediately, no setup required.

**Or, to use Local specifically:** run this app on your own machine:
```
streamlit run app.py
```
then open **http://localhost:8501** and switch to Local mode there.
                        """
                    )

                elif ollama_info["status"] == "offline":
                    st.sidebar.error("❌ **Ollama Not Running**")
                    st.sidebar.markdown(
                        """
**The Local engine needs Ollama running on your computer.**

**Haven't installed Ollama yet?**
1. Download from [ollama.com](https://ollama.com/)
2. Install and launch it
3. Open a terminal and run:
```
ollama pull llama3.1
ollama pull moondream
```
**Already installed?** Just make sure it's running:
- Open the Ollama desktop app, OR
- Run `ollama serve` in a terminal

Then click **Re-check** below 👇
                        """
                    )
                    if st.sidebar.button("🔄 Re-check Ollama", type="primary"):
                        st.rerun()

                elif ollama_info["status"] == "missing_models":
                    missing_list = ollama_info["missing"]
                    st.sidebar.warning(f"⚠️ **Ollama running — {len(missing_list)} model(s) missing**")
                    st.sidebar.write("Open a terminal and run:")
                    for m in missing_list:
                        st.sidebar.code(f"ollama pull {m}", language="bash")
                    st.sidebar.caption("After the download finishes (shows 100%), click Re-check.")
                    if st.sidebar.button("🔄 Re-check Models", type="primary"):
                        st.rerun()

                else:
                    st.sidebar.success("✅ **AI Copilot Ready (Local)**")
                    render_copilot_chat(
                        text_query_fn=lambda history: query_local_llama(
                            history, system_copilot_instruction, model_name="llama3.1"
                        ),
                        vision_query_fn=lambda prompt, img_b64: query_local_llama(
                            [{"role": "user", "content": prompt, "images": [img_b64]}],
                            system_copilot_instruction, model_name="moondream"
                        ),
                        engine_label="Local"
                    )

        else:
            st.sidebar.info("💡 AI Copilot is off. Toggle it on above to start chatting with your well log data.")
            
    except Exception as e:
        st.error(f"Error reading LAS file: {e}")

else:
    # --- MODERN WELCOME LANDING PAGE UI ---
    # Locate rig image path safely using structural file system verification
    possible_rig_names = ["rig.jpg.jpeg", "rig.jpg", "rig.jpeg", "rig.png"]
    rig_image_path = None
    
    for name in possible_rig_names:
        full_path = os.path.join(SCRIPT_DIR, name)
        if os.path.exists(full_path):
            rig_image_path = full_path
            break

    # Top Hero Banner Custom CSS layout (Adaptive Gray for Light & Dark Theme)
    st.markdown(
        """
        <div style="background-color: rgba(128, 128, 128, 0.12); padding: 25px; border-radius: 15px; margin-bottom: 30px; border-left: 1px solid rgba(128, 128, 128, 0.15); border-top: 1px solid rgba(128, 128, 128, 0.15); border-right: 1px solid rgba(128, 128, 128, 0.15); border-bottom: 1px solid rgba(128, 128, 128, 0.15);">
            <h1 style="margin: 0; font-size: 2.6rem; font-weight: bold; letter-spacing: 0.5px;"> AI Petrophysics</h1>
            <p style="font-size: 1.1rem; margin-top: 6px; margin-bottom: 0; opacity: 0.75; font-family: sans-serif;">
                Advanced Subsurface Wireline Log Evaluation & Intelligent Interpretation Suite
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Clean multi-column split section layout
    col1, col2 = st.columns([1.1, 1])
    
    with col1:
        st.markdown("### Welcome to Your Workspace")
        st.markdown(
            """
            Streamline your geoscientific analysis workflows instantly. 
            Transform raw log data arrays into production-ready reservoir metrics, 
            interactive visual profiles, and intelligent machine learning predictions.
            """
        )
        
        # Sleek, premium adaptive 4-card grid layout (Works perfectly on all backgrounds)
        st.markdown(
            """
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; margin-top: 20px; margin-bottom: 25px;">
                <div style="background-color: rgba(128, 128, 128, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(128, 128, 128, 0.2); display: flex; align-items: center;">
                    <span style="font-size: 1.3rem; margin-right: 10px;"></span>
                    <span style="font-weight: bold; font-size: 1rem;">Multi-Track Logs</span>
                </div>
                <div style="background-color: rgba(128, 128, 128, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(128, 128, 128, 0.2); display: flex; align-items: center;">
                    <span style="font-size: 1.3rem; margin-right: 10px;"></span>
                    <span style="font-weight: bold; font-size: 1rem;">Crossplot Maps</span>
                </div>
                <div style="background-color: rgba(128, 128, 128, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(128, 128, 128, 0.2); display: flex; align-items: center;">
                    <span style="font-size: 1.3rem; margin-right: 10px;"></span>
                    <span style="font-weight: bold; font-size: 1rem;">Petrophysical Math</span>
                </div>
                <div style="background-color: rgba(128, 128, 128, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(128, 128, 128, 0.2); display: flex; align-items: center;">
                    <span style="font-size: 1.3rem; margin-right: 10px;"></span>
                    <span style="font-weight: bold; font-size: 1rem;">Machine Learning</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # High contrast interactive notice box pointing straight to the sidebar
        st.info("👈 **GET STARTED:** Please drop or upload an `.las` data file in the sidebar to activate processing controls.")

    with col2:
        if rig_image_path:
            st.image(rig_image_path, use_container_width=True, caption="Offshore Drillsite Platform Exploration")
        else:
            # Clean fallback visual placeholder if image asset hasn't been saved yet
            st.markdown(
                """
                <div style="background-color: rgba(128, 128, 128, 0.08); height: 320px; border-radius: 15px; display: flex; flex-direction: column; align-items: center; justify-content: center; border: 2px dashed rgba(128, 128, 128, 0.25); text-align: center; padding: 20px;">
                    <span style="font-size: 50px; margin-bottom: 10px;">🏗️</span>
                    <h4 style="margin: 5px 0; opacity: 0.8;">Asset Placeholder: rig.jpg</h4>
                    <p style="opacity: 0.6; max-width: 280px; font-size: 0.85rem; line-height: 1.4;">
                        Place your rig photo file inside the 'petapp' folder directory to automatically load platform graphics here!
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
