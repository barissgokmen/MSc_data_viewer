# app.py

"""
to run this app:
streamlit run app.py

"""
import shutil
import sqlite3
import os
from typing import Dict, List, Optional
import numpy as np
import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ISTANBUL_TZ = ZoneInfo("Europe/Istanbul")

# -------- Helpers: DB Introspection --------
def list_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [r[0] for r in cur.fetchall()]

def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f"PRAGMA table_info('{table}')")
    return [r[1] for r in cur.fetchall()]

def detect_keyboard_table(conn: sqlite3.Connection) -> Optional[str]:
    candidates = list_tables(conn)
    # Common AWARE plugin names
    preferred = [
        "plugin_keyboard",
        "keyboard",
        "aware_keyboard",
        "applications_keyboard",
    ]
    for p in preferred:
        if p in candidates:
            return p
    # Fallback: pick any table that has likely keyboard columns
    for t in candidates:
        cols = set(table_columns(conn, t))
        if {"timestamp"}.intersection(cols) and {"key_code","key_character","text"}.intersection(cols):
            return t
    return candidates[0] if candidates else None

def detect_pk_column(conn: sqlite3.Connection, table: str) -> str:
    # Try common primary key names; else use SQLite intrinsic rowid
    cols = set(table_columns(conn, table))
    for c in ["_id", "id", "ID"]:
        if c in cols:
            return c
    return "rowid"

def guess_columns(cols: List[str]) -> Dict[str, Optional[str]]:
    # Attempt to map standard names to found columns
    lower = {c.lower(): c for c in cols}
    def pick(*names):
        for n in names:
            if n in lower:
                return lower[n]
        return None
    mapping = {
        "timestamp": pick("timestamp", "time", "ts"),
        "key_code": pick("key_code", "code", "keycode"),
        "key_character": pick("key_character", "character", "char", "key_char", "unicode"),
        "text": pick("text", "current_text", "message", "content", "field_text"),
        "package": pick("package_name", "package", "app", "application", "app_package"),
        "label": pick("label", "field", "hint", "view_label"),
        "action": pick("key_action", "action", "event"),
    }
    return mapping

# -------- Segmentation Logic --------
ENTER_KEYCODES = {66}  # Android KEYCODE_ENTER
BACKSPACE_KEYCODES = {67}  # Android KEYCODE_DEL

def ms_to_local_str(ms: int, tz=ISTANBUL_TZ) -> str:
    try:
        dt = datetime.fromtimestamp(ms/1000.0, tz=timezone.utc).astimezone(tz)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        # maybe already in seconds
        try:
            dt = datetime.fromtimestamp(int(ms), tz=timezone.utc).astimezone(tz)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(ms)

def segment_keystrokes(df: pd.DataFrame, cfg, *,
                       enter_codes={66}, backspace_codes={67},
                       use_ime_actions=True, ime_submit={2,3,4,6},
                       finalize_on_context_change=True, gap_seconds=10):
    ts_col  = cfg.get("timestamp")
    code_col = cfg.get("key_code") if cfg.get("key_code") in df.columns else None
    char_col = cfg.get("key_character") if cfg.get("key_character") in df.columns else None
    text_col = cfg.get("text") if cfg.get("text") in df.columns else None
    pkg_col  = cfg.get("package") if cfg.get("package") in df.columns else None
    lbl_col  = cfg.get("label") if cfg.get("label") in df.columns else None
    act_col  = cfg.get("action") if cfg.get("action") in df.columns else None
    pk_col   = cfg.get("pk", "rowid")

    if not ts_col or ts_col not in df.columns:
        # fall back to primary key ordering
        ts_col = pk_col

    # Sort stable by (timestamp, pk)
    df = df.sort_values([ts_col, pk_col]).reset_index(drop=True)

    segments = []
    buf_chars = []
    last_text_snapshot = ""
    current_rows = []
    seg_start_ts = None
    last_pkg = None
    last_lbl = None
    last_ts_val = None

    def end_segment(end_ts):
        nonlocal buf_chars, last_text_snapshot, current_rows, seg_start_ts, last_pkg, last_lbl
        if not current_rows:
            buf_chars.clear(); last_text_snapshot = ""; seg_start_ts = None
            return
        # Prefer text snapshot if meaningful
        final_txt = (last_text_snapshot or "").strip()
        if not final_txt:
            final_txt = "".join(buf_chars).strip()
        if final_txt:
            segments.append({
                "text": final_txt,
                "start_ts": seg_start_ts if seg_start_ts is not None else current_rows[0][ts_col],
                "end_ts": end_ts,
                "row_ids": [r[pk_col] for _, r in pd.DataFrame(current_rows).iterrows()],
                "package": last_pkg,
                "label": last_lbl,
                "time_str": ms_to_local_str(int(end_ts)) if isinstance(end_ts, (int, float, np.integer)) else str(end_ts)
            })
        # reset
        buf_chars.clear()
        last_text_snapshot = ""
        current_rows = []
        seg_start_ts = None

    def to_int_or_none(x):
        try:
            return int(x)
        except Exception:
            return None

    for _, row in df.iterrows():
        cur_ts = row[ts_col]
        if seg_start_ts is None:
            seg_start_ts = cur_ts
        current_rows.append(row)

        # Idle gap finalization
        if gap_seconds and last_ts_val is not None:
            try:
                dt_ms = float(cur_ts) - float(last_ts_val)
                if dt_ms > gap_seconds * 1000 and (buf_chars or (last_text_snapshot and last_text_snapshot.strip())):
                    end_segment(last_ts_val)
            except Exception:
                pass
        last_ts_val = cur_ts

        # Track context
        if pkg_col:
            if finalize_on_context_change and last_pkg is not None and row[pkg_col] != last_pkg and (buf_chars or last_text_snapshot.strip()):
                end_segment(cur_ts)
            last_pkg = row[pkg_col]
        if lbl_col:
            if finalize_on_context_change and last_lbl is not None and row[lbl_col] != last_lbl and (buf_chars or last_text_snapshot.strip()):
                end_segment(cur_ts)
            last_lbl = row[lbl_col]

        # Track text snapshot
        if text_col and isinstance(row[text_col], str):
            prev = last_text_snapshot
            last_text_snapshot = row[text_col]
            # Text cleared heuristic
            if prev and prev.strip() and not last_text_snapshot.strip():
                end_segment(cur_ts)

        # Build char buffer & handle backspace
        if char_col and isinstance(row[char_col], str) and len(row[char_col]) == 1:
            ch = row[char_col]
            if ch in ["\n", "\r"]:
                end_segment(cur_ts)
            else:
                buf_chars.append(ch)

        if code_col:
            code = to_int_or_none(row[code_col])
            if code in backspace_codes and buf_chars:
                buf_chars.pop()
            if code in enter_codes:
                end_segment(cur_ts)

        # IME actions (SEND/DONE/GO/SEARCH)
        if use_ime_actions and act_col:
            act = to_int_or_none(row[act_col])
            if act in ime_submit:
                end_segment(cur_ts)

    # finalize trailing buffer if non-empty
    if current_rows and (buf_chars or (last_text_snapshot and last_text_snapshot.strip())):
        end_segment(current_rows[-1][ts_col])

    return segments

# -------- UI --------
st.set_page_config(page_title="AWARE Keyboard Viewer", layout="wide")
st.title("AWARE Keyboard Segments")
st.caption("View, segment, and clean keyboard keystroke logs from AWARE.")

st.sidebar.header("Database")
db_file_uploader = st.sidebar.file_uploader("Upload keyboard .db (SQLite)", type=["db","sqlite","sqlite3"], accept_multiple_files=False)
db_path_text = st.sidebar.text_input("...or path to .db on disk", value="")

if db_file_uploader is not None:
    # Persist uploaded file to a temp path
    os.makedirs("data", exist_ok=True)
    temp_db_path = os.path.join("data", "uploaded_keyboard.db")
    with open(temp_db_path, "wb") as f:
        f.write(db_file_uploader.getbuffer())
    db_path = temp_db_path
elif db_path_text.strip():
    db_path = db_path_text.strip()
else:
    db_path = None

if not db_path or not os.path.exists(db_path):
    st.info("Upload a .db file or provide a valid path to begin.")
    st.stop()

# Optional: backup button
st.sidebar.button("Create backup copy", on_click=lambda: shutil.copyfile(db_path, db_path + ".bak"))

# Connect
conn = sqlite3.connect(db_path)
with conn:
    table = detect_keyboard_table(conn)
if not table:
    st.error("No tables found in the database.")
    st.stop()

st.sidebar.write(f"**Detected table:** `{table}`")

cols = table_columns(conn, table)
mapping = guess_columns(cols)
pk_col = detect_pk_column(conn, table)
mapping["pk"] = pk_col

st.sidebar.subheader("Column Mapping")
for k in ["timestamp","key_code","key_character","text","package","label","action"]:
    mapping[k] = st.sidebar.selectbox(k, ["(none)"] + cols, index=(["(none)"]+cols).index(mapping[k]) if mapping[k] in cols else 0)

# ---- normalize "(none)" and invalid columns to None
for k, v in list(mapping.items()):
    if v == "(none)" or (isinstance(v, str) and v not in cols):
        mapping[k] = None

# --- add in sidebar after column mapping ---
use_ime_actions = st.sidebar.checkbox("Use IME actions as delimiters", value=True)
ime_action_delims = st.sidebar.multiselect(
    "IME action codes to treat as submit",
    options=[2,3,4,6], default=[2,3,4,6], help="GO=2, SEARCH=3, SEND=4, DONE=6"
)
finalize_on_context_change = st.sidebar.checkbox("Finalize on app/field change", value=True)
gap_seconds = st.sidebar.number_input("Finalize if idle for (seconds)", min_value=0, value=10)
enter_keycodes = st.sidebar.text_input("Enter keycodes (comma-separated)", "66")
enter_keycodes = {int(x.strip()) for x in enter_keycodes.split(",") if x.strip().isdigit()}



# Load rows (limit for performance, user-adjustable)
limit = st.sidebar.number_input("Load at most N rows (0 = all)", min_value=0, max_value=2_000_000, value=0, step=1000)
order = "ASC"
sql = f"SELECT rowid as rowid, * FROM '{table}' ORDER BY {mapping['timestamp']} {order}"
if limit and limit > 0:
    sql += f" LIMIT {int(limit)}"
df = pd.read_sql_query(sql, conn)

st.write(f"Loaded **{len(df)}** rows from `{table}`.")

# Segment
segments = segment_keystrokes(
    df, mapping,
    enter_codes=enter_keycodes,
    use_ime_actions=use_ime_actions,
    ime_submit=set(ime_action_delims),
    finalize_on_context_change=finalize_on_context_change,
    gap_seconds=int(gap_seconds)
)

st.subheader("Segments")
st.write(f"Found **{len(segments)}** message segments.")

# Controls
q = st.text_input("Filter by text contains:", "")
segments_view = segments
if q:
    q_low = q.lower()
    segments_view = [s for s in segments if q_low in s["text"].lower()]

# Batch actions
col_a, col_b, col_c = st.columns([1,1,2])
with col_a:
    export_btn = st.button("Export filtered segments (CSV)")
with col_b:
    delete_all_btn = st.button("Delete ALL filtered segments from DB", type="secondary")

if export_btn:
    seg_df = pd.DataFrame([
        {"text": s["text"], "time": s["time_str"], "start_ts": s["start_ts"], "end_ts": s["end_ts"], "package": s["package"], "label": s["label"]}
        for s in segments_view
    ])
    st.download_button("Download CSV", data=seg_df.to_csv(index=False).encode("utf-8"), file_name="segments.csv", mime="text/csv")

if delete_all_btn and st.warning("This will permanently remove the underlying keystrokes for ALL filtered segments. Are you sure?").button("Yes, delete all"):
    with conn:
        pk = mapping["pk"]
        for s in segments_view:
            ids = s["row_ids"]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(f"DELETE FROM '{table}' WHERE {pk} IN ({placeholders})", ids)
    st.success(f"Deleted {len(segments_view)} segments. Please reload.")
    st.stop()

# List UI
for i, s in enumerate(segments_view, start=1):
    with st.container(border=True):
        top_cols = st.columns([6,2,2,2])
        with top_cols[0]:
            st.markdown(f"**{i}.** {s['text'][:200]}{'...' if len(s['text'])>200 else ''}")
        with top_cols[1]:
            st.caption(f"Time: {s['time_str']}")
        with top_cols[2]:
            if s.get("package"):
                st.caption(f"App: {s['package']}")
            if s.get("label"):
                st.caption(f"Field: {s['label']}")
        with top_cols[3]:
            if st.button("Delete", key=f"del_{i}"):
                # Delete rows for this segment
                with conn:
                    ids = s["row_ids"]
                    if ids:
                        pk = mapping["pk"]
                        placeholders = ",".join("?" for _ in ids)
                        conn.execute(f"DELETE FROM '{table}' WHERE {pk} IN ({placeholders})", ids)
                st.success("Segment deleted. Please reload.")
                st.stop()

st.info("Tip: Use the filter box to find sensitive messages and delete them in bulk.")
