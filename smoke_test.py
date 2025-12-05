"""
Simple smoke test: transcribe a WAV and run LLM analysis once.
Usage:
    python smoke_test.py --audio-file path/to/clip.wav
Defaults to CONFIG["audio_file"] from sales_agent.py.
"""

import argparse

from sales_agent import (
    CONFIG,
    analyze_with_llm,
    conversation_history,
    map_temp_label_to_score,
    temperature_history,
    transcribe_audio,
)


def main():
    parser = argparse.ArgumentParser(description="Smoke test for Sales Copilot")
    parser.add_argument(
        "--audio-file",
        default=CONFIG["audio_file"],
        help="Path to WAV file to transcribe.",
    )
    args = parser.parse_args()

    CONFIG["audio_file"] = args.audio_file
    print(f"[smoke] Using audio file: {CONFIG['audio_file']}")

    transcript = transcribe_audio()
    if not transcript:
        print("[smoke] No transcript produced. Exiting.")
        return

    conversation_history.append({"role": "customer", "content": transcript})
    analysis = analyze_with_llm(transcript)

    label = analysis.get("buying_temperature")
    objection = analysis.get("objection")
    suggested_reply = analysis.get("suggested_reply")

    score = map_temp_label_to_score(label)
    temperature_history.append(score)

    print("\n=== Smoke Test Result ===")
    print(f"Transcript: {transcript}")
    print(f"Buying temperature: {label} ({score}/100)")
    print(f"Objection: {objection}")
    print("Suggested reply:")
    print(suggested_reply)


if __name__ == "__main__":
    main()
