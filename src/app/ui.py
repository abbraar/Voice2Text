import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from src.main import run

st.set_page_config(page_title="Voice2Notes", page_icon="🎙️", layout="centered")

STAGES = [
    ("upload", "Uploading file"),
    ("transcribe", "Transcribing audio"),
    ("summarize", "Summarizing & extracting actions"),
    ("export", "Generating outputs"),
    ("done", "Completed"),
]

def stage_index(stage: str) -> int:
    for i, (k, _) in enumerate(STAGES):
        if k == stage:
            return i
    return 0

steps_placeholder = st.empty()

def render_steps(active_stage: str):
    idx = stage_index(active_stage)
    with steps_placeholder.container():
        cols = st.columns(len(STAGES))
        for i, (k, label) in enumerate(STAGES):
            with cols[i]:
                if i < idx:
                    st.success(label)
                elif i == idx:
                    st.info(label)
                else:
                    st.write(label)

st.markdown("## 🎙️ Voice2Notes")
st.caption("Upload an audio file → get transcript, key points, and downloadable outputs.")

with st.container(border=True):
    uploaded = st.file_uploader(
        "Audio file",
        type=["wav", "mp3", "m4a", "aac", "ogg"],
        key="file_uploader",
    )
    col1, col2 = st.columns([1, 1])

    with col1:
        model_ui = st.selectbox("LLM Model", ["Gemini 2.5 Flash", "Gemini 2.5 Pro"], index=0)

    with col2:
        st.write("")
        st.write("")
        start = st.button("🚀 Process", type="primary", use_container_width=True, disabled=(uploaded is None))

MODEL_MAP = {
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
}

if start and uploaded:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)

    safe_name = Path(uploaded.name).name
    path = uploads_dir / f"{int(time.time())}_{safe_name}"

    st.divider()
    progress_bar = st.progress(0, text="Starting…")
    pct_label = st.empty()
    status_line = st.empty()

    stage_labels = dict(STAGES)

    def progress_cb(stage: str, pct: int, msg: str = ""):
        pct = max(0, min(100, int(pct)))
        render_steps(stage)
        progress_bar.progress(pct, text=msg or stage_labels.get(stage, "Working…"))
        pct_label.markdown(f"**{pct}%**")
        if msg:
            status_line.caption(msg)

    progress_cb("upload", 3, "Uploading file…")
    data = uploaded.getvalue()

    if not data or len(data) < 1024:
        st.error("Uploaded file looks empty/corrupted. Please re-upload.")
        st.stop()

    with open(path, "wb") as f:
        f.write(data)
    del data

    size_mb = os.path.getsize(path) / (1024 * 1024)
    status_line.caption(f"Saved: {path.name} ({size_mb:.2f} MB)")

    try:
        t0 = time.time()
        result = run(
            str(path),
            out_dir="outputs",
            progress_cb=progress_cb,
            llm_model=MODEL_MAP.get(model_ui, "gemini-2.5-flash"),
        )
        elapsed = time.time() - t0
        progress_cb("done", 100, f"Done in {elapsed:.1f}s")
    except Exception as e:
        st.exception(e)
        st.stop()

    st.divider()

    with st.container(border=True):
        st.markdown("### ✅ Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Language", result.get("detected_language", "—"))
        c2.metric("Duration (s)", f"{result.get('duration', 0):.1f}")
        c3.metric("Outputs", "Ready")

        st.markdown("#### Downloads")
        d1, d2, d3, d4 = st.columns(4)

        def dl(col, key, label):
            p = result.get(key)
            if not p or not os.path.exists(p):
                col.button(label, disabled=True, use_container_width=True)
                return
            with open(p, "rb") as f:
                col.download_button(label, f, file_name=os.path.basename(p), use_container_width=True)

        dl(d1, "transcript", "📄 Transcript")
        dl(d2, "md", "📝 Markdown")
        dl(d3, "pdf", "📑 PDF")
        dl(d4, "json", "🧾 JSON")

    # ✅ Clear uploader state
    st.session_state["file_uploader"] = None

st.markdown("---")
st.caption("Made by **Abrar Abdulaziz**")
