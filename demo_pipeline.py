import subprocess
from faster_whisper import WhisperModel

AUDIO_FILE = "test.wav"

def transcribe_audio():
    # CPU mode for now
    model = WhisperModel("small", device="cpu", compute_type="int8")
    print(f"Transcribing {AUDIO_FILE}...")
    segments, info = model.transcribe(AUDIO_FILE)

    text = " ".join(segment.text for segment in segments)
    print(f"Detected language: {info.language} (confidence {info.language_probability:.2f})")
    print("Transcript:", text.strip())
    return text.strip()

def ask_llm(transcript_text):
    prompt = f"""
You are a helpful sales coach.
The customer just said: "{transcript_text}"

Give me a short, friendly reply I could say next on a sales call.
Make it 1–2 sentences max.
"""

    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8", errors="ignore")

def main():
    text = transcribe_audio()
    print("\nAsking LLM based on that transcript...\n")
    reply = ask_llm(text)
    print("Suggested reply:\n")
    print(reply)

if __name__ == "__main__":
    main()
