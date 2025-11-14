from faster_whisper import WhisperModel

AUDIO_FILE = "test.wav"

def main():
    # You can change "small" to "medium" or "large-v3" later if you want
    model = WhisperModel("small", device="cpu", compute_type="int8")

    print(f"Transcribing {AUDIO_FILE}...")
    segments, info = model.transcribe(AUDIO_FILE, beam_size=5)

    print(f"Detected language: {info.language} (confidence {info.language_probability:.2f})")
    print("\nTranscript:")
    print("-" * 40)
    for segment in segments:
        print(f"[{segment.start:.2f} -> {segment.end:.2f}] {segment.text}")

if __name__ == "__main__":
    main()
