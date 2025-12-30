from faster_whisper import WhisperModel
import os


def _pick_device_and_compute(device: str | None, compute_type: str | None):
    """
    Decide device/compute safely:
    - Prefer user choice
    - Auto fallback to CPU if CUDA not available or cuDNN issues likely
    """
    # If user explicitly forces CPU
    if device == "cpu":
        return "cpu", (compute_type or "int8")

    # If user wants auto, try CUDA first (if visible)
    if device in (None, "auto", "cuda"):
        # Respect CUDA_VISIBLE_DEVICES
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "":
            # If user exported CUDA_VISIBLE_DEVICES="" => force CPU
            return "cpu", (compute_type or "int8")

        # Try CUDA by default
        chosen_device = "cuda"
        chosen_compute = compute_type or "float16"
        return chosen_device, chosen_compute

    # Any other device value -> fall back safely
    return "cpu", (compute_type or "int8")


def transcribe(
    audio_path: str,
    model_size: str = "medium",
    device: str | None = "auto",          # auto/cuda/cpu
    compute_type: str | None = None,      # float16/int8/...
):
    chosen_device, chosen_compute = _pick_device_and_compute(device, compute_type)

    # 1) Try preferred device
    try:
        model = WhisperModel(model_size, device=chosen_device, compute_type=chosen_compute)
        segments, info = model.transcribe(
            audio_path,
            language=None,        # auto-detect (Arabic/English/mixed)
            vad_filter=True,
            word_timestamps=False,
            beam_size=5,
        )
    except Exception as e:
        # 2) Fallback to CPU (prevents Streamlit crash when cuDNN missing)
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, info = model.transcribe(
            audio_path,
            language=None,
            vad_filter=True,
            word_timestamps=False,
            beam_size=5,
        )

    transcript = []
    for s in segments:
        transcript.append(
            {"start": float(s.start), "end": float(s.end), "text": (s.text or "").strip()}
        )

    return {
        "detected_language": getattr(info, "language", None),
        "duration": getattr(info, "duration", None),
        "segments": transcript,
    }


def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def segments_to_text(segments):
    lines = []
    for seg in segments:
        ts = f"[{format_timestamp(seg['start'])}–{format_timestamp(seg['end'])}]"
        lines.append(f"{ts} {seg['text']}")
    return "\n".join(lines)
