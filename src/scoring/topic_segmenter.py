"""
topic_segmenter.py
Splits a full transcript into topically coherent sections using an LLM.

Long transcripts are processed in chunks (to stay within prompt size limits),
then the per-chunk topic lists are stitched together into one continuous map.
"""

import os
import sys
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
CHUNK_DURATION_SECONDS = 480  # ~8 minutes of transcript per LLM call

TOPIC_PROMPT_TEMPLATE = """You are analyzing a transcript excerpt from a longer video, to identify distinct topics being discussed.

Transcript (each line shows [start-end] followed by what was said):
{transcript_lines}

Task: Split this excerpt into topic sections. A new section should start whenever the speaker meaningfully shifts to a new subject, question, or idea — not for every sentence.

Respond with ONLY valid JSON, a list like this, no other text:
[
  {{"start": <float seconds>, "end": <float seconds>, "topic": "<short 3-8 word topic title>"}},
  ...
]
"""


class TopicSegmentationError(Exception):
    pass


def _call_groq(prompt: str) -> str:
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise TopicSegmentationError("GROQ_API_KEY not found in .env")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def _parse_json_list(raw_text: str) -> list:
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start_idx = raw_text.find("[")
        end_idx = raw_text.rfind("]")
        if start_idx != -1 and end_idx != -1:
            return json.loads(raw_text[start_idx:end_idx + 1])
        raise


def _chunk_segments(segments: list, chunk_duration: float) -> list:
    """Groups transcript segments into time-based chunks for LLM processing."""
    chunks = []
    current_chunk = []
    chunk_start_time = segments[0]["start"]

    for seg in segments:
        if seg["start"] - chunk_start_time > chunk_duration and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            chunk_start_time = seg["start"]
        current_chunk.append(seg)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _segment_chunk(chunk_segments: list, retries: int = 2) -> list:
    lines = "\n".join(
        f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in chunk_segments
    )
    prompt = TOPIC_PROMPT_TEMPLATE.format(transcript_lines=lines)

    last_error = None
    for _ in range(retries):
        try:
            raw = _call_groq(prompt)
            topics = _parse_json_list(raw)
            if not isinstance(topics, list) or not topics:
                raise ValueError("Empty or invalid topic list returned.")
            return topics
        except Exception as e:
            last_error = e
            time.sleep(1)

    raise TopicSegmentationError(f"Failed to segment chunk after {retries} retries: {last_error}")


def segment_transcript(transcript_path: str) -> dict:
    """
    Loads transcript.json, splits it into time-based chunks, asks the LLM
    to identify topic boundaries within each chunk, and stitches the results
    into one continuous topic map for the full video.

    Returns dict with keys: source_transcript, topics, output_path
    """
    path = Path(transcript_path)
    if not path.exists():
        raise TopicSegmentationError(f"Transcript file not found: {transcript_path}")

    with open(path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    segments = transcript.get("segments", [])
    if not segments:
        raise TopicSegmentationError("Transcript has no segments to analyze.")

    chunks = _chunk_segments(segments, CHUNK_DURATION_SECONDS)

    all_topics = []
    for i, chunk in enumerate(chunks):
        print(f"Segmenting chunk {i + 1}/{len(chunks)} "
              f"[{chunk[0]['start']:.1f}s - {chunk[-1]['end']:.1f}s]...", flush=True)
        chunk_topics = _segment_chunk(chunk)
        all_topics.extend(chunk_topics)
        print(f"    -> found {len(chunk_topics)} topic(s)", flush=True)

    # Sort by start time, just in case chunks returned out of order
    all_topics.sort(key=lambda t: t["start"])

    output_path = path.parent / "topics.json"
    result = {
        "source_transcript": str(path),
        "topics": all_topics,
        "output_path": str(output_path),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/scoring/topic_segmenter.py <path_to_transcript.json>")
        sys.exit(1)

    try:
        result = segment_transcript(sys.argv[1])
        print(f"\nFound {len(result['topics'])} topics total.")
        print(f"Saved to: {result['output_path']}\n")
        for t in result["topics"]:
            print(f"[{t['start']:.1f}s - {t['end']:.1f}s] {t['topic']}")
    except TopicSegmentationError as e:
        print(f"\nTopic segmentation failed: {e}")
        sys.exit(1)