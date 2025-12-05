# Sales Copilot (local Whisper + Ollama)

Local-first sales-call assistant that records or ingests audio, transcribes with Whisper, keeps short-term context, and asks an LLM (Ollama) for objection classification and a suggested reply. Logs everything to JSONL for later analysis.

## Quickstart
```bash
cd sales-copilot
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # PowerShell on Windows
pip install -r requirements.txt

# Ensure Ollama is running and llama3 is available
ollama pull llama3
```

### Run (interactive record)
```bash
python sales_agent.py
```
Press Enter to record ~5s from the mic, then view transcript and suggested reply. Results are logged to `conversation_log.jsonl`.

### Run (dry-run from existing WAV)
```bash
python sales_agent.py --dry-run --audio-file path\to\clip.wav
```

### Smoke test (no mic needed)
```bash
python smoke_test.py --audio-file path\to\clip.wav
```
Runs one transcribe + LLM pass for quick verification.

### Samples
Place your test WAVs under `samples/`, e.g. `samples/sample.wav`, then run:
```bash
python smoke_test.py --audio-file samples/sample.wav
python sales_agent.py --dry-run --audio-file samples/sample.wav
```

## Config notes
- Whisper settings are in `CONFIG` at the top of `sales_agent.py`.
- Defaults use CPU (`whisper_device="cpu"`, `whisper_compute_type="int8"`) to avoid CUDA/cuDNN setup. Switch to `cuda`/`float16` if GPU is ready.
- Recording defaults: 16 kHz, 5s, output `test.wav`.

## Troubleshooting
- **Ollama not found**: Install/start Ollama and ensure `ollama` is on PATH.
- **cuDNN DLL missing**: Switch to CPU in `CONFIG` or install CUDA + cuDNN and set `whisper_device="cuda"` / `whisper_compute_type="float16"`.
- **No audio captured**: Check mic permissions and default input device in your OS; adjust device selection in `sounddevice` if needed.
