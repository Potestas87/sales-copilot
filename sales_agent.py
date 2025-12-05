import argparse
import datetime
import json
import os
import queue
import subprocess
import threading
import uuid
from pathlib import Path

import sounddevice as sd
from faster_whisper import WhisperModel
from scipy.io.wavfile import write

# ===== Basic config =====
# Tune these as needed or move to env vars later.
CONFIG = {
    "sample_rate": 16000,              # Hz, good for speech
    "duration_seconds": 5,             # length of each recording
    "audio_file": "test.wav",          # reused each time
    "chunk_seconds": 3,                # streaming chunk length
    "whisper_model": "small",          # try "medium" or "large-v3" on GPU
    "whisper_device": "cpu",           # set to "cuda" for GPU; cpu avoids cuDNN issues
    "whisper_compute_type": "int8",    # int8 on CPU; use float16 on GPU
    "conversation_log": "conversation_log.jsonl",
}

# ===== Conversation memory =====
# Each entry: {"role": "customer" | "agent", "content": "<text>"}
conversation_history: list[dict] = []

# Track numeric buying temperature over time (0-100)
temperature_history: list[int] = []
_whisper_model: WhisperModel | None = None
SESSION_ID: str | None = None
SESSION_START: datetime.datetime | None = None


def map_temp_label_to_score(label: str) -> int:
    """
    Map COLD / WARM / HOT to a numeric score.
    You can tweak these numbers later if you want.
    """
    if not label:
        return 50
    label = label.upper()
    if "HOT" in label:
        return 85
    if "WARM" in label:
        return 60
    if "COLD" in label:
        return 25
    return 50  # fallback / UNKNOWN


def render_temp_bar(score: int, width: int = 20) -> str:
    """
    Render a simple text meter like ######------.
    """
    score = max(0, min(100, score))  # clamp
    filled = int((score / 100) * width)
    return "#" * filled + "-" * (width - filled)


def get_trend(temps: list[int]) -> str:
    """
    Compare last two scores to show warming / cooling / steady.
    """
    if len(temps) < 2:
        return "n/a"  # no trend yet

    diff = temps[-1] - temps[-2]
    if diff > 5:
        return "warming"
    elif diff < -5:
        return "cooling"
    else:
        return "steady"


def append_jsonl_line(data: dict) -> None:
    """Append a dict to the JSONL log file."""
    log_path = Path(CONFIG["conversation_log"])
    line = json.dumps(data, ensure_ascii=False)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:  # logging failures are non-blocking
        print(f"[warn] Failed to write conversation log: {exc}")


def log_event(event: str, **fields) -> None:
    """Attach session info and write an event to the log."""
    payload = {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "event": event,
        "session_id": SESSION_ID,
    }
    payload.update(fields)
    append_jsonl_line(payload)


def ensure_wav(audio_path: str) -> str:
    """
    If given an mp3, attempt to convert to wav via ffmpeg.
    Returns path to a wav (or original path if conversion fails).
    """
    path = Path(audio_path)
    if path.suffix.lower() != ".mp3":
        return audio_path

    target = path.with_suffix(".wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-ar",
        str(CONFIG["sample_rate"]),
        "-ac",
        "1",
        str(target),
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError:
        print("[warn] ffmpeg not found; cannot convert mp3 to wav. Using original path.")
        return audio_path

    if result.returncode != 0:
        print(f"[warn] ffmpeg conversion failed (code {result.returncode}). Using original path.")
        return audio_path

    print(f"[info] Converted mp3 to wav: {target}")
    return str(target)


def record_audio() -> bool:
    """Record audio and save to CONFIG['audio_file']. Returns True on success."""
    audio_file = CONFIG["audio_file"]
    duration = CONFIG["duration_seconds"]
    sample_rate = CONFIG["sample_rate"]

    if os.path.exists(audio_file):
        try:
            os.remove(audio_file)
        except OSError as exc:
            print(f"[warn] Could not remove existing audio file: {exc}")

    print(f"\n[rec] Recording for {duration} seconds... speak now.")
    try:
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()  # Wait until recording is finished
        write(audio_file, sample_rate, recording)
        print(f"[ok] Saved recording to {audio_file}")
        return True
    except Exception as exc:
        print(f"[error] Recording failed: {exc}")
        return False


def record_chunk(duration: float, filename: str) -> bool:
    """Record a short chunk to a specific filename."""
    sample_rate = CONFIG["sample_rate"]
    try:
        if os.path.exists(filename):
            os.remove(filename)
    except OSError as exc:
        print(f"[warn] Could not remove existing audio file: {exc}")
    print(f"[rec] Capturing {duration}s chunk...")
    try:
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        write(filename, sample_rate, recording)
        return True
    except Exception as exc:
        print(f"[error] Chunk recording failed: {exc}")
        return False


def _load_whisper() -> WhisperModel | None:
    """Lazy-load WhisperModel so we do not reload every turn."""
    global _whisper_model
    if _whisper_model:
        return _whisper_model
    try:
        _whisper_model = WhisperModel(
            CONFIG["whisper_model"],
            device=CONFIG["whisper_device"],
            compute_type=CONFIG["whisper_compute_type"],
        )
        return _whisper_model
    except Exception as exc:
        print(f"[error] Failed to load Whisper on {CONFIG['whisper_device']} ({CONFIG['whisper_compute_type']}): {exc}")
        # Auto-fallback: try CPU int8 if CUDA/cuDNN is missing
        if CONFIG["whisper_device"] != "cuda":
            return None
        print("[info] Falling back to CPU/int8 for Whisper due to GPU init error.")
        try:
            _whisper_model = WhisperModel(
                CONFIG["whisper_model"],
                device="cpu",
                compute_type="int8",
            )
            return _whisper_model
        except Exception as exc2:
            print(f"[error] CPU fallback also failed: {exc2}")
            return None


def transcribe_audio() -> str:
    """Transcribe AUDIO_FILE using Whisper and return transcript text."""
    audio_file = ensure_wav(CONFIG["audio_file"])
    return transcribe_file(audio_file)


def transcribe_file(audio_file: str) -> str:
    """Transcribe a specific audio file path using Whisper."""
    if not os.path.exists(audio_file):
        if CONFIG.get("absolute_paths"):
            print(f"[warn] No audio file found to transcribe: {Path(audio_file).resolve()}")
        else:
            print(f"[warn] No audio file found to transcribe: {audio_file}")
        return ""

    print("[asr] Transcribing audio...")
    model = _load_whisper()
    if not model:
        return ""

    try:
        segments, info = model.transcribe(audio_file)
        text = " ".join(segment.text for segment in segments).strip()
        print(f"[asr] Detected language: {info.language} (confidence {info.language_probability:.2f})")
        print(f"[asr] Transcript: {text}")
        return text
    except Exception as exc:
        print(f"[error] Transcription failed: {exc}")
        return ""


def run_stream_mode() -> None:
    """Continuously capture short chunks and analyze until interrupted (non-blocking recording).

    Heuristic end-of-utterance: accumulate speech chunks; when we hit a silent/empty chunk,
    flush the buffered transcript to the LLM. Keeps recording while LLM processes.
    """
    chunk_seconds = CONFIG.get("chunk_seconds", 3)
    chunk_dir = Path("stream_chunks")
    chunk_dir.mkdir(exist_ok=True)
    print(f"[stream] Capturing ~{chunk_seconds}s chunks. Press Ctrl+C to stop.\n")

    q: queue.Queue[tuple[int, Path]] = queue.Queue(maxsize=4)
    stop_event = threading.Event()
    idx_counter = {"value": 0}
    pending_parts: list[str] = []
    last_speech_idx: int | None = None

    def record_loop():
        while not stop_event.is_set():
            idx_counter["value"] += 1
            idx = idx_counter["value"]
            chunk_path = chunk_dir / f"chunk_{idx}.wav"
            ok = record_chunk(chunk_seconds, str(chunk_path))
            if not ok:
                continue
            try:
                q.put((idx, chunk_path), timeout=1)
            except queue.Full:
                print("[stream] Queue full; dropping chunk.")
                try:
                    chunk_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def process_loop():
        while not stop_event.is_set() or not q.empty():
            try:
                idx, chunk_path = q.get(timeout=0.5)
            except queue.Empty:
                continue

            transcript = transcribe_file(str(chunk_path))

            is_silence = not transcript.strip()
            if not is_silence:
                pending_parts.append(transcript)
                last_speech_idx = idx
                print(f"[chunk {idx}] {transcript}")
            else:
                print(f"[chunk {idx}] (silence)")

            should_flush = is_silence and pending_parts
            if should_flush:
                utterance = " ".join(pending_parts).strip()
                pending_parts.clear()
                conversation_history.append({"role": "customer", "content": utterance})
                log_event("customer_turn", text=utterance, chunk_index=idx, mode="stream")

                analysis = analyze_with_llm(utterance)
                label = analysis.get("buying_temperature")
                objection = analysis.get("objection")
                suggested_reply = analysis.get("suggested_reply")

                score = map_temp_label_to_score(label)
                temperature_history.append(score)

                print(f"\n=== Utterance ending at chunk {idx} ===")
                print(f"Transcript: {utterance}")
                print(f"Buying temperature: {label} ({score}/100)")
                print(f"  Meter: {render_temp_bar(score)}")
                print(f"  Trend: {get_trend(temperature_history)}")
                print(f"Objection: {objection}")
                print("Suggested reply:")
                print(suggested_reply)
                print("-----------------------------")

                if suggested_reply:
                    conversation_history.append({"role": "agent", "content": suggested_reply})
                    log_event(
                        "agent_suggestion",
                        text=suggested_reply,
                        buying_temperature=label,
                        temperature_score=score,
                        objection=objection,
                        chunk_index=idx,
                        mode="stream",
                    )
            q.task_done()
            try:
                chunk_path.unlink(missing_ok=True)
            except Exception:
                pass

    recorder = threading.Thread(target=record_loop, daemon=True)
    processor = threading.Thread(target=process_loop, daemon=True)

    try:
        recorder.start()
        processor.start()
        while recorder.is_alive() and processor.is_alive():
            recorder.join(timeout=0.5)
            processor.join(timeout=0.5)
    except KeyboardInterrupt:
        print("\n[stream] Stopping...")
    finally:
        stop_event.set()
        recorder.join(timeout=1.0)
        processor.join(timeout=1.0)


def build_context_block(max_turns: int = 6) -> str:
    """
    Build a short text block representing the recent conversation.
    We'll include up to `max_turns` most recent entries.
    """
    if not conversation_history:
        return "No previous context. This is the first message."

    recent = conversation_history[-max_turns:]
    lines = []
    for entry in recent:
        role = entry["role"]
        content = entry["content"]
        if role == "customer":
            lines.append(f"Customer: {content}")
        else:
            lines.append(f"Agent: {content}")
    return "\n".join(lines)


def analyze_with_llm(transcript_text: str) -> dict:
    """Send transcript + recent context to local LLM and return structured analysis."""
    if not transcript_text:
        return {
            "buying_temperature": "UNKNOWN",
            "objection": "UNKNOWN",
            "suggested_reply": "(No transcript text available.)",
        }

    if CONFIG.get("skip_llm") or os.environ.get("SKIP_LLM") == "1":
        return {
            "buying_temperature": "WARM",
            "objection": "none",
            "suggested_reply": "Got it—how does this align with what you need today?",
        }

    context_block = build_context_block()

    prompt = f"""
You are a sales-call AI coach.

Here is the recent conversation between the customer and the agent:
\"\"\"
{context_block}
\"\"\"

The customer just said (most recent line):
\"\"\"{transcript_text}\"\"\"

1. Based on the entire conversation so far, classify their current buying temperature as one of:
   - "COLD"  (just browsing, low commitment, lots of hesitation)
   - "WARM"  (interested but needs reassurance or clarification)
   - "HOT"   (ready or very close to saying yes)

2. Identify their main objection or concern in a short phrase.
   Examples: "price", "timing", "trust", "needs spouse approval",
   "already has a provider", "needs more information", etc.

3. Suggest a short, friendly reply the agent could say next on a sales call.
   - 1-3 sentences
   - Conversational
   - No jargon
   - Acknowledge their concern and move the conversation forward.

Respond ONLY in valid JSON using this exact structure:

{{
  "buying_temperature": "COLD | WARM | HOT",
  "objection": "string",
  "suggested_reply": "string"
}}
"""

    print("[llm] Asking local LLM (llama3 via Ollama) with conversation context...")
    try:
        result = subprocess.run(
            # --format json nudges Ollama to return strict JSON
            ["ollama", "run", "llama3", "--format", "json"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        print("[error] Ollama is not installed or not on PATH.")
        return {
            "buying_temperature": "UNKNOWN",
            "objection": "UNKNOWN",
            "suggested_reply": "(Ollama not available)",
        }

    raw_output = result.stdout.decode("utf-8", errors="ignore").strip()
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="ignore").strip()
        print(f"[error] Ollama returned non-zero exit code {result.returncode}: {err}")
        return {
            "buying_temperature": "UNKNOWN",
            "objection": "UNKNOWN",
            "suggested_reply": raw_output or err,
        }

    # Try to extract JSON in case the model adds extra text
    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start != -1 and end != -1:
            json_str = raw_output[start : end + 1]
        else:
            json_str = raw_output

        data = json.loads(json_str)
        return data
    except Exception as exc:
        print("[warn] Failed to parse JSON from model output.")
        print("Raw output was:\n", raw_output)
        print("Error:", exc)
        fallback_reply = "I understand—can you share what matters most so I can address it?"
        return {
            "buying_temperature": "UNKNOWN",
            "objection": "UNKNOWN",
            "suggested_reply": raw_output or fallback_reply,
        }


def main():
    parser = argparse.ArgumentParser(description="Sales Copilot (local Whisper + Ollama)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip recording; transcribe an existing audio file instead.",
    )
    parser.add_argument(
        "--audio-file",
        default=CONFIG["audio_file"],
        help="Path to WAV file to transcribe (used for dry-run and recording output).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Override Whisper device (default from CONFIG).",
    )
    parser.add_argument(
        "--compute-type",
        choices=["int8", "int16", "float16", "float32"],
        help="Override Whisper compute type (default from CONFIG).",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM call and return a stubbed suggestion (for fast smoke tests).",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Show absolute paths in warnings for missing audio files.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Continuous chunked capture and analysis until Ctrl+C.",
    )
    args = parser.parse_args()

    # Override audio file if provided
    CONFIG["audio_file"] = args.audio_file
    if args.device:
        CONFIG["whisper_device"] = args.device
    if args.compute_type:
        CONFIG["whisper_compute_type"] = args.compute_type
    if args.skip_llm:
        CONFIG["skip_llm"] = True
    CONFIG["absolute_paths"] = args.absolute_paths

    # Session metadata
    global SESSION_ID, SESSION_START
    SESSION_ID = uuid.uuid4().hex
    SESSION_START = datetime.datetime.utcnow()
    log_event(
        "session_start",
        mode="dry-run" if args.dry_run else "record",
        audio_file=CONFIG["audio_file"],
        whisper_model=CONFIG["whisper_model"],
        whisper_device=CONFIG["whisper_device"],
        whisper_compute_type=CONFIG["whisper_compute_type"],
    )

    print("=== Sales Copilot | Local AI Sales Assistant (with memory) ===")
    print("This script will:")
    if args.stream:
        print("  - Continuously capture short chunks from your mic")
    elif args.dry_run:
        print("  - Transcribe an existing audio file (dry-run mode)")
    else:
        print("  - Record a short audio clip from your mic")
    print("  - Transcribe it locally with Whisper")
    print("  - Use recent conversation history to analyze state")
    print("  - Suggest a reply you can say next")
    if not args.dry_run:
        print("\nPress Enter to start a new recording, or type 'q' and press Enter to quit.\n")

    if args.stream:
        run_stream_mode()
    else:
        while True:
            if args.dry_run:
                choice = ""
            else:
                choice = input("[menu] Press Enter to record, or 'q' to quit: ").strip().lower()

            if choice == "q":
                print("\n[bye] Exiting Sales Copilot. Goodbye.")
                break

            # 1) Record (skip in dry-run)
            if not args.dry_run:
                if not record_audio():
                    continue
            else:
                print(f"[dry-run] Using existing audio file: {CONFIG['audio_file']}")

            # 2) Transcribe
            transcript = transcribe_audio()
            if not transcript:
                print("[warn] No transcript, skipping analysis.\n")
                if args.dry_run:
                    break
                continue

            # Store customer turn in history
            conversation_history.append({"role": "customer", "content": transcript})
            log_event("customer_turn", text=transcript)

            # 3) Analyze with LLM (using history)
            analysis = analyze_with_llm(transcript)

            # 4) Show results with temperature meter
            label = analysis.get("buying_temperature")
            objection = analysis.get("objection")
            suggested_reply = analysis.get("suggested_reply")

            # Convert label to numeric score and update history
            score = map_temp_label_to_score(label)
            temperature_history.append(score)

            print("\n=== Analysis ===")
            print(f"Buying temperature: {label} ({score}/100)")
            print(f"  Meter: {render_temp_bar(score)}")
            print(f"  Trend: {get_trend(temperature_history)}")
            print(f"Objection: {objection}")
            print("\nSuggested reply:\n")
            print(suggested_reply)
            print("\n-----------------------------\n")

            # Store agent turn in history
            suggestion_used: bool | None = None
            outcome = None
            if suggested_reply:
                conversation_history.append({"role": "agent", "content": suggested_reply})
                log_event(
                    "agent_suggestion",
                    text=suggested_reply,
                    buying_temperature=label,
                    temperature_score=score,
                    objection=objection,
                )
                # Capture feedback/outcome
                if args.dry_run:
                    outcome = "dry-run"
                else:
                    feedback = input("[feedback] Did you use this reply? (y/n/skip): ").strip().lower()
                    if feedback in ("y", "yes"):
                        suggestion_used = True
                    elif feedback in ("n", "no"):
                        suggestion_used = False
                    else:
                        suggestion_used = None
                    outcome = input("[feedback] Call outcome (won/lost/other, blank=unknown): ").strip() or "unknown"

                log_event(
                    "suggestion_feedback",
                    suggestion_used=suggestion_used,
                    outcome=outcome,
                    suggested_reply=suggested_reply,
                )

            if args.dry_run:
                # Run once in dry-run mode, then exit
                print("[dry-run] Completed one pass. Exiting.")
                break

    # Session end
    if SESSION_START:
        duration_s = (datetime.datetime.utcnow() - SESSION_START).total_seconds()
    else:
        duration_s = None
    log_event("session_end", duration_seconds=duration_s)


if __name__ == "__main__":
    main()
