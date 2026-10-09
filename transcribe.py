"""Resumable Whisper transcription, writing the same cache the notebook reads.

Usage:  .venv\\Scripts\\python transcribe.py medium.en
Exit code 0 once every .wav under data/ is in cache/transcripts_<model>.json.
"""
import json
import sys
import time
from pathlib import Path

import torch
from faster_whisper import WhisperModel

MODEL = sys.argv[1] if len(sys.argv) > 1 else "small.en"
CACHE = Path("cache") / f"transcripts_{MODEL}.json"
CACHE.parent.mkdir(exist_ok=True)


def save(transcripts):
    # Write to a temp file then rename, so a kill mid-write never corrupts the cache.
    tmp = CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(transcripts))
    tmp.replace(CACHE)


transcripts = json.loads(CACHE.read_text()) if CACHE.exists() else {}
# Keys match the notebook: relative paths like data\Dataset_Final\train\audio_1.wav
paths = sorted(Path("data").rglob("*.wav"))
todo = [p for p in paths if str(p) not in transcripts]
print(f"{MODEL}: {len(paths) - len(todo)}/{len(paths)} cached, {len(todo)} to go", flush=True)

if todo:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # large-v3 in float16 (~3.1 GB of weights) does not fit a 4 GB GPU alongside activations; int8 weights do.
    compute_type = ("int8_float16" if "large" in MODEL else "float16") if device == "cuda" else "int8"
    asr = WhisperModel(MODEL, device=device, compute_type=compute_type)
    t0 = time.time()
    for i, p in enumerate(todo, 1):
        segments, info = asr.transcribe(
            str(p), language="en", beam_size=5, temperature=0.0,
            condition_on_previous_text=False, vad_filter=True,
        )
        segs = [{"start": s.start, "end": s.end, "text": s.text.strip(), "avg_logprob": s.avg_logprob,
                 "no_speech_prob": s.no_speech_prob} for s in segments]
        transcripts[str(p)] = {"duration": info.duration, "segments": segs}
        if i % 10 == 0 or i == len(todo):
            save(transcripts)
            rate = (time.time() - t0) / i
            print(f"{len(transcripts)}/{len(paths)} done, ~{rate * (len(todo) - i) / 60:.0f} min left", flush=True)

print("complete" if len(transcripts) >= len(paths) else "incomplete", flush=True)
sys.exit(0 if len(transcripts) >= len(paths) else 1)
