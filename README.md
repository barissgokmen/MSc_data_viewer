# AWARE Keyboard Viewer

A tiny Streamlit app to read an AWARE keyboard SQLite database, segment keystrokes into messages, and delete segments (and their underlying keystroke rows).

## Features
- Auto-detect table & columns (with manual override in the sidebar)
- Segments keystrokes between Enter/Send presses
- Shows message text + timestamp (Europe/Istanbul by default)
- Delete a single segment or all filtered segments
- Export filtered segments to CSV
- Upload a `.db` or point to a file path on disk

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the shown local URL in your browser.

## Notes
- The app tries to guess columns like `timestamp`, `key_code`, `key_character`, `text`, `package_name`, `_id` etc. If your schema is different, adjust the selections in the sidebar.
- Deletion uses the detected primary key (tries `_id`, `id`, then falls back to `rowid`). It deletes *only* from the detected keyboard table.
- Timestamps are assumed to be milliseconds since epoch (AWARE style). If your DB stores seconds, the app tries to detect and still renders a reasonable time.
