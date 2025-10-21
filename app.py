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
import time

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
    # Persist uploaded file to a temp path (read-only, never modified)
    os.makedirs("data", exist_ok=True)
    temp_db_path = os.path.join("data", "uploaded_keyboard.db")
    
    # Always overwrite to ensure we have the original file
    with open(temp_db_path, "wb") as f:
        f.write(db_file_uploader.getbuffer())
    st.sidebar.success(f"✅ Loaded: {db_file_uploader.name}")
    
    db_path = temp_db_path
elif db_path_text.strip():
    db_path = db_path_text.strip()
else:
    db_path = None

# Initialize deleted segments tracking in session state
if 'deleted_segment_ids' not in st.session_state:
    st.session_state['deleted_segment_ids'] = set()

# Initialize expanded segments tracking (only load data for expanded ones)
if 'expanded_segments' not in st.session_state:
    st.session_state['expanded_segments'] = set()

if not db_path or not os.path.exists(db_path):
    st.info("Upload a .db file or provide a valid path to begin.")
    st.stop()

# Optional: backup button
st.sidebar.button("Create backup copy", on_click=lambda: shutil.copyfile(db_path, db_path + ".bak"))

# Connect (close any existing connection first to ensure fresh data)
conn = sqlite3.connect(db_path)
try:
    with conn:
        table = detect_keyboard_table(conn)
    if not table:
        st.error("No tables found in the database.")
        conn.close()
        st.stop()
except Exception as e:
    st.error(f"Error connecting to database: {e}")
    conn.close()
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

# Ensure we always have the primary key column in the result
pk = mapping["pk"]
if pk == "rowid":
    sql = f"SELECT rowid as rowid, * FROM '{table}' ORDER BY {mapping['timestamp']} {order}"
else:
    sql = f"SELECT * FROM '{table}' ORDER BY {mapping['timestamp']} {order}"

if limit and limit > 0:
    sql += f" LIMIT {int(limit)}"
df = pd.read_sql_query(sql, conn)

# Check total row count in the table
total_rows_query = f"SELECT COUNT(*) FROM '{table}'"
total_rows = conn.execute(total_rows_query).fetchone()[0]

st.write(f"Loaded **{len(df)}** rows from `{table}`. Total rows in table: **{total_rows}**. Primary key: `{pk}`")

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

# First, filter out deleted segments from the original list
segments_not_deleted = []
segments_not_deleted_indices = []
for i, s in enumerate(segments):
    if i not in st.session_state['deleted_segment_ids']:
        segments_not_deleted.append(s)
        segments_not_deleted_indices.append(i)

# Then apply text search filter
segments_view = segments_not_deleted
if q:
    q_low = q.lower()
    segments_view = [s for s in segments_not_deleted if q_low in s["text"].lower()]

st.write(f"Showing **{len(segments_view)}** segments (of {len(segments)} total)")
if st.session_state['deleted_segment_ids']:
    st.warning(f"⚠️ {len(st.session_state['deleted_segment_ids'])} segments marked for deletion. Click 'Save Filtered DB' to create cleaned database.")

# Pagination settings
segments_per_page = st.sidebar.number_input("Segments per page", min_value=5, max_value=100, value=20, step=5)
total_pages = max(1, (len(segments_view) + segments_per_page - 1) // segments_per_page)

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 1

# Ensure current page is within bounds
if st.session_state['current_page'] > total_pages:
    st.session_state['current_page'] = total_pages

# Pagination controls
col_prev, col_page, col_next = st.columns([1, 2, 1])
with col_prev:
    if st.button("⬅️ Previous", disabled=(st.session_state['current_page'] <= 1), key="prev_page"):
        st.session_state['current_page'] -= 1
        st.rerun()
with col_page:
    st.write(f"**Page {st.session_state['current_page']} of {total_pages}**")
with col_next:
    if st.button("Next ➡️", disabled=(st.session_state['current_page'] >= total_pages), key="next_page"):
        st.session_state['current_page'] += 1
        st.rerun()

# Calculate which segments to show on this page
start_idx = (st.session_state['current_page'] - 1) * segments_per_page
end_idx = min(start_idx + segments_per_page, len(segments_view))
segments_page = segments_view[start_idx:end_idx]

# Batch actions
col_a, col_b, col_c, col_d = st.columns([1,1,1,2])
with col_a:
    export_btn = st.button("Export filtered segments (CSV)")
with col_b:
    if st.button("Mark ALL for deletion", type="secondary"):
        # Mark all currently visible segments for deletion
        # We need to find the original indices of segments_view in the segments list
        for s in segments_view:
            # Find this segment's index in the original segments list
            for i, orig_s in enumerate(segments):
                if orig_s is s:  # Same object reference
                    st.session_state['deleted_segment_ids'].add(i)
                    break
        st.rerun()
with col_c:
    save_filtered_db_btn = st.button("Save Filtered DB", type="primary")
with col_d:
    if st.session_state['deleted_segment_ids']:
        if st.button("Clear deletion marks", type="secondary", key="clear_marks_top"):
            st.session_state['deleted_segment_ids'] = set()
            st.rerun()

if export_btn:
    seg_df = pd.DataFrame([
        {"text": s["text"], "time": s["time_str"], "start_ts": s["start_ts"], "end_ts": s["end_ts"], "package": s["package"], "label": s["label"]}
        for s in segments_view
    ])
    st.download_button("Download CSV", data=seg_df.to_csv(index=False).encode("utf-8"), file_name="segments.csv", mime="text/csv")

if save_filtered_db_btn:
    # Create a new database with only the rows from segments that are NOT marked for deletion
    os.makedirs("data", exist_ok=True)
    filtered_db_path = os.path.join("data", "keyboard-filtered.db")
    
    # Remove existing filtered database if it exists
    if os.path.exists(filtered_db_path):
        os.remove(filtered_db_path)
    
    # Collect all row IDs from segments that are NOT deleted (keep these)
    keep_row_ids = set()
    for i, s in enumerate(segments):
        if i not in st.session_state['deleted_segment_ids']:
            keep_row_ids.update(s["row_ids"])
    
    if not keep_row_ids:
        st.warning("No segments to save (all segments marked for deletion).")
    else:
        with st.spinner("Creating filtered database..."):
            # Copy the database structure and data
            with sqlite3.connect(filtered_db_path) as new_conn:
                # Copy schema from original database
                with conn:
                    # Get all table schemas (excluding internal SQLite tables)
                    tables_to_copy = [t for t in list_tables(conn) if not t.startswith('sqlite_')]
                    for tbl in tables_to_copy:
                        # Get CREATE TABLE statement
                        schema_query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{tbl}'"
                        schema = conn.execute(schema_query).fetchone()
                        if schema and schema[0]:
                            new_conn.execute(schema[0])
                    
                    # Copy indices
                    indices = conn.execute("SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL").fetchall()
                    for idx in indices:
                        if idx[0]:
                            try:
                                new_conn.execute(idx[0])
                            except sqlite3.OperationalError:
                                pass  # Skip if index already exists or conflicts
                    
                    # For the keyboard table, only copy rows that are in keep_row_ids
                    pk = mapping["pk"]
                    placeholders = ",".join("?" for _ in keep_row_ids)
                    
                    # Get column names for INSERT
                    columns = table_columns(conn, table)
                    col_list = ", ".join(f'"{c}"' for c in columns)
                    
                    # Select only the rows we want to keep
                    select_query = f'SELECT {col_list} FROM "{table}" WHERE {pk} IN ({placeholders})'
                    rows_to_copy = conn.execute(select_query, list(keep_row_ids)).fetchall()
                    
                    # Insert into new database
                    question_marks = ",".join("?" for _ in columns)
                    insert_query = f'INSERT INTO "{table}" ({col_list}) VALUES ({question_marks})'
                    new_conn.executemany(insert_query, rows_to_copy)
                    
                    # Copy other tables completely (if any)
                    for tbl in tables_to_copy:
                        if tbl != table:
                            cols_other = table_columns(conn, tbl)
                            col_list_other = ", ".join(f'"{c}"' for c in cols_other)
                            rows_other = conn.execute(f'SELECT {col_list_other} FROM "{tbl}"').fetchall()
                            if rows_other:
                                qmarks = ",".join("?" for _ in cols_other)
                                new_conn.executemany(f'INSERT INTO "{tbl}" ({col_list_other}) VALUES ({qmarks})', rows_other)
                    
                    # Update sqlite_sequence table to reflect the new row counts
                    # This ensures autoincrement values are correct
                    for tbl in tables_to_copy:
                        row_count = new_conn.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
                        if row_count > 0:
                            max_id_query = f'SELECT MAX(ROWID) FROM "{tbl}"'
                            try:
                                max_id = new_conn.execute(max_id_query).fetchone()[0]
                                if max_id:
                                    # Update or insert into sqlite_sequence
                                    new_conn.execute(
                                        "INSERT OR REPLACE INTO sqlite_sequence (name, seq) VALUES (?, ?)",
                                        (tbl, max_id)
                                    )
                            except:
                                pass  # Table might not have autoincrement
                    
                    new_conn.commit()
        
        kept_segments = len(segments) - len(st.session_state['deleted_segment_ids'])
        st.success(f"✅ Saved filtered database with {kept_segments} segments ({len(keep_row_ids)} rows) to: `{filtered_db_path}`")
        
        # Provide download button
        with open(filtered_db_path, "rb") as f:
            st.download_button(
                label="📥 Download keyboard-filtered.db",
                data=f.read(),
                file_name="keyboard-filtered.db",
                mime="application/x-sqlite3"
            )
        
        # Option to clear deletion marks after saving
        if st.button("Clear deletion marks", key="clear_marks_after_save"):
            st.session_state['deleted_segment_ids'] = set()
            st.rerun()

# List UI - Only show current page
display_num = start_idx  # Start numbering from the page offset
for s in segments_page:
    # Find the original index in the full segments list
    original_idx = None
    for i, orig_s in enumerate(segments):
        if orig_s is s:
            original_idx = i
            break
    
    if original_idx is None:
        continue
    
    display_num += 1
    
    with st.container(border=True):
        top_cols = st.columns([6,2,2,2])
        with top_cols[0]:
            st.markdown(f"**{display_num}.** {s['text'][:200]}{'...' if len(s['text'])>200 else ''}")
        with top_cols[1]:
            st.caption(f"Time: {s['time_str']}")
        with top_cols[2]:
            if s.get("package"):
                st.caption(f"App: {s['package']}")
            if s.get("label"):
                st.caption(f"Field: {s['label']}")
        with top_cols[3]:
            if st.button("Mark for deletion", key=f"del_{original_idx}"):
                st.session_state['deleted_segment_ids'].add(original_idx)
                st.rerun()
        
        # Expander to show keystrokes directly under this segment - no page refresh!
        with st.expander(f"🔍 View {len(s['row_ids'])} keystrokes"):
            try:
                pk = mapping["pk"]
                row_ids = s["row_ids"]
                
                if row_ids:
                    with st.spinner("Loading keystrokes..."):
                        # Fetch all rows for this segment
                        placeholders = ",".join("?" for _ in row_ids)
                        query = f'SELECT * FROM "{table}" WHERE {pk} IN ({placeholders}) ORDER BY {mapping["timestamp"]}'
                        segment_rows_cursor = conn.execute(query, row_ids)
                        
                        # Get column names
                        col_names = [description[0] for description in segment_rows_cursor.description]
                        
                        # Fetch rows
                        segment_rows = segment_rows_cursor.fetchall()
                        
                        # Create DataFrame for display
                        rows_df = pd.DataFrame(segment_rows, columns=col_names)
                    
                    # Show summary stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Rows", len(rows_df))
                    with col2:
                        if mapping["timestamp"] and mapping["timestamp"] in rows_df.columns:
                            try:
                                duration_ms = float(rows_df[mapping["timestamp"]].max() - rows_df[mapping["timestamp"]].min())
                                st.metric("Duration", f"{duration_ms/1000:.1f}s")
                            except:
                                st.metric("Duration", "N/A")
                    with col3:
                        st.metric("Row IDs", f"{min(row_ids)}-{max(row_ids)}")
                    
                    # Select important columns to display
                    display_cols = []
                    if mapping["timestamp"] and mapping["timestamp"] in rows_df.columns:
                        rows_df.insert(0, 'Time', rows_df[mapping["timestamp"]].apply(
                            lambda x: ms_to_local_str(int(x)) if pd.notna(x) else ""
                        ))
                        display_cols.append('Time')
                    
                    # Add other important columns
                    for col in ['key_character', 'key_code', 'text', 'package_name', 'label']:
                        mapped_col = mapping.get(col.replace('_', ''))
                        if mapped_col and mapped_col in rows_df.columns:
                            display_cols.append(mapped_col)
                    
                    # If no important columns found, show all
                    if not display_cols:
                        display_cols = list(rows_df.columns)
                    
                    # Display limited columns for better performance
                    st.dataframe(
                        rows_df[display_cols] if display_cols else rows_df,
                        use_container_width=True,
                        hide_index=True,
                        height=300
                    )
                    
                    # Option to show all columns
                    if st.checkbox("Show all columns", key=f"allcols_{original_idx}"):
                        st.dataframe(
                            rows_df,
                            use_container_width=True,
                            hide_index=True,
                            height=400
                        )
                else:
                    st.info("No keystroke data available for this segment.")
            except Exception as e:
                st.error(f"Error loading keystrokes: {e}")
                import traceback
                st.code(traceback.format_exc())

# Bottom pagination controls
st.markdown("---")
col_prev2, col_page2, col_next2 = st.columns([1, 2, 1])
with col_prev2:
    if st.button("⬅️ Previous", disabled=(st.session_state['current_page'] <= 1), key="prev_page2"):
        st.session_state['current_page'] -= 1
        st.rerun()
with col_page2:
    st.write(f"**Page {st.session_state['current_page']} of {total_pages}**")
with col_next2:
    if st.button("Next ➡️", disabled=(st.session_state['current_page'] >= total_pages), key="next_page2"):
        st.session_state['current_page'] += 1
        st.rerun()

st.info("Tip: Use the filter box to find sensitive messages and delete them in bulk.")
