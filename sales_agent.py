import os
import subprocess
import json
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

# ===== Recording config =====
SAMPLE_RATE = 16000       # 16 kHz is good for speech
DURATION_SECONDS = 5      # length of each recording
AUDIO_FILE = "test.wav"   # reused each time


def record_audio():
    """Record DURATION_SECONDS of mic audio and save to AUDIO_FILE."""
    # Remove previous file just to be explicit
    if os.path.exists(AUDIO_FILE):
        os.remove(AUDIO_FILE)

    print(f"\n[🎙️] Recording for {DURATION_SECONDS} seconds... speak now.")
    recording = sd.rec(
        int(DURATION_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()  # Wait until recording is finished
    write(AUDIO_FILE, SAMPLE_RATE, recording)
    print(f"[💾] Saved recording to {AUDIO_FILE}")


def transcribe_audio() -> str:
    """Transcribe AUDIO_FILE using Whisper and return transcript text."""
    if not os.path.exists(AUDIO_FILE):
        print("[⚠️] No audio file found to transcribe.")
        return ""

    print("[📝] Transcribing audio...")
    # CPU mode for now; we can swap to CUDA later
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(AUDIO_FILE)

    text = " ".join(segment.text for segment in segments).strip()
    print(f"[🌐] Detected language: {info.language} (confidence {info.language_probability:.2f})")
    print(f"[🗣️] Transcript: {text}")
    return text


def analyze_with_llm(transcript_text: str) -> dict:
    """Send transcript to local LLM (Ollama) and return structured analysis."""
    if not transcript_text:
        return {
            "buying_temperature": "UNKNOWN",
            "objection": "UNKNOWN",
            "suggested_reply": "(No transcript text available.)",
        }

    prompt = f"""
You are a sales-call AI coach.

The customer just said:
\"\"\"{transcript_text}\"\"\"

1. Classify their current buying temperature as one of:
   - "COLD"  (just browsing, low commitment, lots of hesitation)
   - "WARM"  (interested but needs reassurance or clarification)
   - "HOT"   (ready or very close to saying yes)

2. Identify their main objection or concern in a short phrase.
   Examples: "price", "timing", "trust", "needs spouse approval",
   "already has a provider", "needs more information", etc.

3. Suggest a short, friendly reply I could say next on a sales call.
   - 1–3 sentences
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

    print("[🤖] Asking local LLM (llama3 via Ollama)...")
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
    )
    raw_output = result.stdout.decode("utf-8", errors="ignore").strip()

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
    except Exception as e:
        print("[⚠️] Failed to parse JSON from model output.")
        print("Raw output was:\n", raw_output)
        print("Error:", e)
        return {
            "buying_temperature": "UNKNOWN",
            "objection": "UNKNOWN",
            "suggested_reply": raw_output,
        }


def main():
    print("=== Sales Copilot — Local AI Sales Assistant ===")
    print("This script will:")
    print("  - Record a short audio clip from your mic")
    print("  - Transcribe it locally with Whisper")
    print("  - Analyze buying temperature & objection")
    print("  - Suggest a reply you can say next")
    print("\nPress Enter to start a new recording, or type 'q' and press Enter to quit.\n")

    while True:
        choice = input("[⌨️] Press Enter to record, or 'q' to quit: ").strip().lower()
        if choice == "q":
            print("\n[👋] Exiting Sales Copilot. Goodbye.")
            break

        # 1) Record
        record_audio()

        # 2) Transcribe
        transcript = transcribe_audio()

        # 3) Analyze with LLM
        analysis = analyze_with_llm(transcript)

        # 4) Show results
        print("\n=== Analysis ===")
        print(f"Buying temperature: {analysis.get('buying_temperature')}")
        print(f"Objection: {analysis.get('objection')}")
        print("\nSuggested reply:\n")
        print(analysis.get("suggested_reply"))
        print("\n-----------------------------\n")


if __name__ == "__main__":
    main()
