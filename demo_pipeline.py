import subprocess
import json
from faster_whisper import WhisperModel

AUDIO_FILE = "test.wav"


def transcribe_audio():
    # CPU mode for now
    model = WhisperModel("small", device="cpu", compute_type="int8")
    print(f"Transcribing {AUDIO_FILE}...")
    segments, info = model.transcribe(AUDIO_FILE)

    text = " ".join(segment.text for segment in segments)
    text = text.strip()
    print(f"Detected language: {info.language} (confidence {info.language_probability:.2f})")
    print("Transcript:", text)
    return text


def ask_llm(transcript_text: str) -> dict:
    prompt = f"""
You are a sales-call AI coach.

The customer just said:
\"\"\"{transcript_text}\"\"\"

1. Classify their current buying temperature as one of:
   - "COLD"  (just browsing, low commitment, lots of hesitation)
   - "WARM"  (interested but needs reassurance or clarification)
   - "HOT"   (ready or very close to saying yes)

2. Identify their main objection or concern in a short phrase.
   Examples: "price", "timing", "trust", "needs spouse approval", "already has a provider", "needs more information", etc.

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

    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
    )
    raw_output = result.stdout.decode("utf-8", errors="ignore").strip()

    # Try to extract JSON (in case the model adds extra text)
    try:
        # Find first "{" and last "}"
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start != -1 and end != -1:
            json_str = raw_output[start : end + 1]
        else:
            json_str = raw_output

        data = json.loads(json_str)
        return data
    except Exception as e:
        print("Failed to parse JSON from model output.")
        print("Raw output was:\n", raw_output)
        print("Error:", e)
        return {
            "buying_temperature": "UNKNOWN",
            "objection": "UNKNOWN",
            "suggested_reply": raw_output,
        }


def main():
    text = transcribe_audio()
    print("\nAsking LLM based on that transcript...\n")
    analysis = ask_llm(text)

    print("Buying temperature:", analysis.get("buying_temperature"))
    print("Objection:", analysis.get("objection"))
    print("\nSuggested reply:\n")
    print(analysis.get("suggested_reply"))


if __name__ == "__main__":
    main()
