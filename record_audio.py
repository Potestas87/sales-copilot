import sounddevice as sd
from scipy.io.wavfile import write

SAMPLE_RATE = 16000  # 16 kHz is good for speech
DURATION_SECONDS = 5
OUTPUT_FILE = "test.wav"

def main():
    print(f"Recording for {DURATION_SECONDS} seconds... speak now!")
    recording = sd.rec(
        int(DURATION_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()  # Wait until recording is finished
    write(OUTPUT_FILE, SAMPLE_RATE, recording)
    print(f"Saved recording to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
