import os
from src.asr.transcribe import transcribe, segments_to_text
from src.llm.summarize import summarize
from src.export.render import save_json, save_md, save_pdf


def _fmt_ts(seconds: float | int) -> str:
    s = int(round(float(seconds)))
    mm = s // 60
    ss = s % 60
    return f"{mm:02d}:{ss:02d}"


def segments_to_timed_text(segments) -> str:
    lines = []
    for seg in segments or []:
        start = _fmt_ts(seg.get("start", 0))
        end = _fmt_ts(seg.get("end", 0))
        text = (seg.get("text") or "").strip()
        if text:
            lines.append(f"[{start}-{end}] {text}")
    return "\n".join(lines)


def run(audio_path: str, out_dir="outputs", progress_cb=None, llm_model: str | None = None):
    os.makedirs(out_dir, exist_ok=True)

    def prog(stage: str, pct: int, msg: str = ""):
        if progress_cb:
            progress_cb(stage, int(pct), msg)

    # 1) ASR
    prog("transcribe", 5, "Loading audio…")
    asr = transcribe(audio_path)

    prog("transcribe", 55, "Building transcript…")
    transcript_text = segments_to_text(asr["segments"])              # for saving to txt (raw)
    timed_transcript = segments_to_timed_text(asr["segments"])       # for LLM evidence

    # 2) LLM (pass timed transcript so it can cite evidence)
    prog("summarize", 70, "Summarizing + adding evidence…")
    summary = (
        summarize(timed_transcript, model_name=llm_model)
        if llm_model
        else summarize(timed_transcript)
    )

    # 3) Export
    prog("export", 85, "Saving outputs…")
    base = os.path.splitext(os.path.basename(audio_path))[0]
    transcript_path = os.path.join(out_dir, f"{base}_transcript.txt")
    json_path = os.path.join(out_dir, f"{base}_summary.json")
    md_path = os.path.join(out_dir, f"{base}_notes.md")
    pdf_path = os.path.join(out_dir, f"{base}_notes.pdf")

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    save_json(json_path, {"asr": asr, "summary": summary})
    save_md(md_path, transcript_text, summary)  # transcript still available, even if you don't print it

    prog("export", 95, "Rendering PDF…")
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    save_pdf(pdf_path, md_text)

    prog("done", 100, "Completed ✅")

    return {
        "transcript": transcript_path,
        "json": json_path,
        "md": md_path,
        "pdf": pdf_path,
        "detected_language": asr["detected_language"],
        "duration": asr["duration"],
    }


if __name__ == "__main__":
    import sys
    audio = sys.argv[1]
    print(run(audio))
