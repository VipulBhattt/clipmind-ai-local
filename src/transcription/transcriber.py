"""
transcriber.py
Transcription module for ClipMind AI.

Uses faster-whisper (CPU) to transcribe audio into a timestamped
transcript with both segment-level and word-level timing.
"""

import sys
import json
from pathlib import Path

from faster_whisper import WhisperModel


class TranscriptionError(Exception):
    """Raised when transcription fails."""
    pass


# Loaded once and reused across calls in the same process (model load is slow)
_model_cache = {}


def _get_model(model_size: str = "small") -> WhisperModel:
    if model_size not in _model_cache:
        try:
            _model_cache[model_size] = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",  # faster + lighter on CPU, small accuracy tradeoff
            )
        except Exception as e:
            raise TranscriptionError(f"Failed to load Whisper model '{model_size}': {e}")
    return _model_cache[model_size]


def transcribe_audio(audio_path: str, model_size: str = "small") -> dict:
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    print(f"[1/4] Audio file found: {audio_file} ({audio_file.stat().st_size / 1024:.1f} KB)", flush=True)

    print(f"[2/4] Loading Whisper model '{model_size}' (first run downloads ~500MB, please wait)...", flush=True)
    model = _get_model(model_size)
    print(f"[2/4] Model loaded successfully.", flush=True)

    print(f"[3/4] Starting transcription...", flush=True)
    try:
        segments_iter, info = model.transcribe(
            str(audio_file),
            word_timestamps=True,
            vad_filter=True,
        )
    except Exception as e:
        raise TranscriptionError(f"Whisper transcription failed: {e}")

    print(f"[3/4] Transcription stream started. Detected language: {info.language} "
          f"(confidence: {info.language_probability:.2f})", flush=True)

    segments = []
    try:
        for i, seg in enumerate(segments_iter):
            words = []
            if seg.words:
                for w in seg.words:
                    words.append({
                        "word": w.word.strip(),
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                    })
            segments.append({
                "id": i,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "words": words,
            })
            print(f"    -> Segment {i}: [{seg.start:.1f}s - {seg.end:.1f}s] {seg.text.strip()[:60]}", flush=True)
    except Exception as e:
        raise TranscriptionError(f"Error while processing transcription segments: {e}")

    print(f"[3/4] Processed {len(segments)} segments.", flush=True)

    if not segments:
        raise TranscriptionError(
            "Transcription completed but produced no segments — "
            "the audio may be silent, corrupted, or unsupported."
        )

    transcript_path = audio_file.parent / "transcript.json"
    result = {
        "audio_path": str(audio_file),
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration": round(info.duration, 2),
        "segments": segments,
        "transcript_path": str(transcript_path),
    }

    print(f"[4/4] Saving transcript to {transcript_path}...", flush=True)
    try:
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    except OSError as e:
        raise TranscriptionError(f"Failed to write transcript.json: {e}")

    print(f"[4/4] Done.", flush=True)

    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/transcription/transcriber.py <path_to_audio.wav>")
        sys.exit(1)

    path = sys.argv[1]
    try:
        result = transcribe_audio(path)
        print(f"\nTranscription successful. Language: {result['language']} "
              f"(confidence: {result['language_probability']})")
        print(f"Duration: {result['duration']}s | Segments: {len(result['segments'])}")
        print(f"Saved to: {result['transcript_path']}\n")
        print("First segment preview:")
        print(json.dumps(result['segments'][0], indent=2, ensure_ascii=False))
    except TranscriptionError as e:
        print(f"\nTranscription failed: {e}")
        sys.exit(1)