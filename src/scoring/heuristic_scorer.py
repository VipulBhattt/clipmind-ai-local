"""
heuristic_scorer.py
Layer 1 of the ClipMind AI scoring pipeline.

Scores transcript segments using cheap, fast text-based heuristics —
no API calls, no audio processing. Produces a filtered, ranked candidate
list to hand off to the LLM scoring layer.
"""

import re
import sys
import json
from pathlib import Path


class ScoringError(Exception):
    """Raised when scoring fails due to bad input."""
    pass


HOOK_KEYWORDS = [
    "secret", "never", "always", "crazy", "insane", "shocking", "unbelievable",
    "mistake", "wrong", "truth", "honestly", "actually", "biggest", "worst",
    "best", "hate", "love", "afraid", "scared", "warning", "important",
]

QUESTION_PATTERN = re.compile(r"\?")
NUMBER_PATTERN = re.compile(r"\b\d+\b")
EXCLAMATION_PATTERN = re.compile(r"!")


def _keyword_score(text: str) -> float:
    text_lower = text.lower()
    hits = sum(1 for kw in HOOK_KEYWORDS if kw in text_lower)
    return min(hits / 3, 1.0)


def _question_score(text: str) -> float:
    return 1.0 if QUESTION_PATTERN.search(text) else 0.0


def _number_score(text: str) -> float:
    return 1.0 if NUMBER_PATTERN.search(text) else 0.0


def _exclamation_score(text: str) -> float:
    return 1.0 if EXCLAMATION_PATTERN.search(text) else 0.0


def _length_score(text: str) -> float:
    word_count = len(text.split())
    if word_count < 4:
        return 0.1
    if 8 <= word_count <= 40:
        return 1.0
    return 0.5


def score_segment(text: str) -> dict:
    signals = {
        "keyword_score": round(_keyword_score(text), 2),
        "question_score": round(_question_score(text), 2),
        "number_score": round(_number_score(text), 2),
        "exclamation_score": round(_exclamation_score(text), 2),
        "length_score": round(_length_score(text), 2),
    }

    weights = {
        "keyword_score": 0.35,
        "question_score": 0.20,
        "number_score": 0.10,
        "exclamation_score": 0.10,
        "length_score": 0.25,
    }

    combined = sum(signals[k] * weights[k] for k in signals)

    reasons = []
    if signals["keyword_score"] > 0:
        reasons.append("contains attention-grabbing keywords")
    if signals["question_score"] > 0:
        reasons.append("poses a question")
    if signals["number_score"] > 0:
        reasons.append("mentions a specific number/statistic")
    if signals["exclamation_score"] > 0:
        reasons.append("has emphatic/exclamatory tone")
    if signals["length_score"] == 1.0:
        reasons.append("well-sized for a standalone clip")

    explanation = "; ".join(reasons) if reasons else "no strong heuristic signals detected"

    return {
        "signals": signals,
        "heuristic_score": round(combined, 3),
        "explanation": explanation,
    }


def score_transcript(transcript_path: str, top_n: int = 10) -> dict:
    path = Path(transcript_path)
    if not path.exists():
        raise ScoringError(f"Transcript file not found: {transcript_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ScoringError(f"Failed to read transcript.json: {e}")

    segments = transcript.get("segments", [])
    if not segments:
        raise ScoringError("Transcript contains no segments to score.")

    scored_segments = []
    for seg in segments:
        text = seg.get("text", "")
        if not text.strip():
            continue
        score_result = score_segment(text)
        scored_segments.append({
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "text": text,
            **score_result,
        })

    if not scored_segments:
        raise ScoringError("No scorable text found in any segment.")

    ranked = sorted(scored_segments, key=lambda s: s["heuristic_score"], reverse=True)
    top_candidates = ranked[:top_n]

    output_path = path.parent / "heuristic_scores.json"
    result = {
        "source_transcript": str(path),
        "total_segments": len(scored_segments),
        "top_candidates": top_candidates,
        "all_scored_segments": ranked,
        "output_path": str(output_path),
    }

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    except OSError as e:
        raise ScoringError(f"Failed to write heuristic_scores.json: {e}")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/scoring/heuristic_scorer.py <path_to_transcript.json>")
        sys.exit(1)

    try:
        result = score_transcript(sys.argv[1])
        print(f"\nScored {result['total_segments']} segments.")
        print(f"Saved full results to: {result['output_path']}\n")
        print(f"Top {len(result['top_candidates'])} candidates:\n")
        for c in result["top_candidates"]:
            print(f"[{c['start']:.1f}s - {c['end']:.1f}s] score={c['heuristic_score']} "
                  f"| {c['explanation']}")
            print(f"    \"{c['text'][:80]}\"\n")
    except ScoringError as e:
        print(f"\nScoring failed: {e}")
        sys.exit(1)