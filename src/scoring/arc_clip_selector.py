"""
arc_clip_selector.py
Within each identified topic, finds the best short-form clip that follows
a setup -> build -> payoff narrative arc, rather than just picking the
single highest-scoring sentence.
"""

import os
import sys
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
MIN_CLIP_DURATION = 25.0
MAX_CLIP_DURATION = 90.0

ARC_PROMPT_TEMPLATE = """You are selecting the best short-form clip (like a YouTube Short) from ONE topic section of a longer video.

Topic: "{topic_title}"

Transcript of this topic (each line: [start-end] text):
{transcript_lines}

Your job: find the best possible clip of roughly {min_dur}-{max_dur} seconds within this topic that has a strong narrative arc:
- It should OPEN with a hook, question, or teaser that makes someone curious (not a random mid-thought sentence).
- The MIDDLE should build on that curiosity (explanation, buildup, tension).
- It should END right after the payoff/answer/punchline lands — not trail off into unrelated content afterward.

If this topic doesn't contain a clear tease-then-payoff moment, just pick the most self-contained, engaging {min_dur}-{max_dur} second window instead, and note that in your reasoning.

Respond with ONLY valid JSON in this exact format, no other text:
{{
  "start": <float seconds, must match a real timestamp from the transcript above>,
  "end": <float seconds, must match a real timestamp from the transcript above>,
  "has_arc": <true or false>,
  "hook_line": "<the opening line that hooks the viewer>",
  "payoff_line": "<the line where the payoff/answer lands, or empty string if has_arc is false>",
  "score": <int 0-10, how strong this clip is as a standalone short>,
  "reasoning": "<one or two sentences explaining the choice>"
}}
"""


class ArcSelectionError(Exception):
    pass


def _call_groq(prompt: str) -> str:
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ArcSelectionError("GROQ_API_KEY not found in .env")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


def _parse_json_obj(raw_text: str) -> dict:
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            return json.loads(raw_text[start_idx:end_idx + 1])
        raise


def _snap_to_nearest_boundary(time_value: float, segments: list, key: str) -> float:
    """Snaps an LLM-estimated time to the nearest real segment boundary."""
    closest = min(segments, key=lambda s: abs(s[key] - time_value))
    return closest[key]

def _extend_to_sentence_completion(end_time: float, all_segments: list, max_duration: float,
                                    clip_start: float) -> float:
    """
    If the segment ending at `end_time` doesn't end on clear sentence-final
    punctuation, and the following segment appears to continue the same
    sentence, extend the end boundary to include it — as long as this
    doesn't push the clip too far past the max duration.
    """
    sorted_segments = sorted(all_segments, key=lambda s: s["start"])
    current_idx = next((i for i, s in enumerate(sorted_segments) if abs(s["end"] - end_time) < 0.01), None)
    if current_idx is None or current_idx >= len(sorted_segments) - 1:
        return end_time

    current_text = sorted_segments[current_idx]["text"].strip()
    next_text = sorted_segments[current_idx + 1]["text"].strip()

    ends_cleanly = current_text.endswith((".", "?", "!"))
    next_continues = next_text and next_text[0].islower()

    candidate_end = sorted_segments[current_idx + 1]["end"]
    would_still_fit = (candidate_end - clip_start) <= (max_duration + 15)  # allow modest overshoot for a clean payoff

    if not ends_cleanly and next_continues and would_still_fit:
        return candidate_end

    return end_time

def _expand_to_minimum(start: float, end: float, all_segments: list,
                        topic_start: float, topic_end: float) -> dict:
    """
    If a clip is under MIN_CLIP_DURATION, expands it outward using real
    transcript segment boundaries. Prefers staying within the topic's own
    boundaries; only spills into neighboring topics if the topic itself
    is too short to reach the minimum on its own.
    """
    duration = end - start
    if duration >= MIN_CLIP_DURATION:
        return {"start": start, "end": end, "extended_beyond_topic": False}

    sorted_segments = sorted(all_segments, key=lambda s: s["start"])
    left_idx = next((i for i, s in enumerate(sorted_segments) if s["start"] >= start), 0)
    right_idx = next((i for i, s in enumerate(sorted_segments) if s["end"] >= end), len(sorted_segments) - 1)

    extended_beyond_topic = False

    while (sorted_segments[right_idx]["end"] - sorted_segments[left_idx]["start"]) < MIN_CLIP_DURATION:
        can_left = left_idx > 0
        can_right = right_idx < len(sorted_segments) - 1
        if not can_left and not can_right:
            break

        would_leave_topic_left = can_left and sorted_segments[left_idx - 1]["start"] < topic_start
        would_leave_topic_right = can_right and sorted_segments[right_idx + 1]["end"] > topic_end

        # Prefer expanding within the topic first
        if can_left and not would_leave_topic_left:
            left_idx -= 1
        elif can_right and not would_leave_topic_right:
            right_idx += 1
        elif can_left:
            left_idx -= 1
            extended_beyond_topic = True
        elif can_right:
            right_idx += 1
            extended_beyond_topic = True
        else:
            break

    return {
        "start": sorted_segments[left_idx]["start"],
        "end": sorted_segments[right_idx]["end"],
        "extended_beyond_topic": extended_beyond_topic,
    }


def _validate_arc_consistency(result: dict) -> dict:
    """Forces has_arc=False if a payoff_line wasn't actually provided."""
    if result.get("has_arc") and not result.get("payoff_line", "").strip():
        result["has_arc"] = False
        result["reasoning"] = (result.get("reasoning", "") +
                                " [Auto-corrected: marked has_arc=False because no payoff_line was provided.]").strip()
    return result


def _select_arc_for_topic(topic: dict, all_segments: list, retries: int = 2) -> dict:
    topic_segments = [
        s for s in all_segments if s["start"] >= topic["start"] and s["end"] <= topic["end"]
    ]
    if not topic_segments:
        raise ArcSelectionError(f"No transcript segments found within topic '{topic['topic']}'.")

    lines = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in topic_segments)
    prompt = ARC_PROMPT_TEMPLATE.format(
        topic_title=topic["topic"],
        transcript_lines=lines,
        min_dur=int(MIN_CLIP_DURATION),
        max_dur=int(MAX_CLIP_DURATION),
    )

    last_error = None
    for _ in range(retries):
        try:
            raw = _call_groq(prompt)
            result = _parse_json_obj(raw)
            required = ["start", "end", "has_arc", "hook_line", "score", "reasoning"]
            if not all(k in result for k in required):
                raise ValueError(f"Missing required keys: {result}")

            result["start"] = _snap_to_nearest_boundary(float(result["start"]), topic_segments, "start")
            result["end"] = _snap_to_nearest_boundary(float(result["end"]), topic_segments, "end")

            if result["end"] <= result["start"]:
                raise ValueError("Resulting clip has non-positive duration after snapping.")

            # Enforce minimum duration by expanding using real transcript boundaries
            # Enforce minimum duration by expanding using real transcript boundaries
            expanded = _expand_to_minimum(result["start"], result["end"], all_segments,
                                           topic["start"], topic["end"])
            result["start"] = expanded["start"]
            result["end"] = expanded["end"]
            result["extended_beyond_topic"] = expanded["extended_beyond_topic"]

            # NEW: extend end boundary if it's cutting off mid-sentence
            result["end"] = _extend_to_sentence_completion(
                result["end"], all_segments, MAX_CLIP_DURATION, result["start"]
            )

            # Cap at maximum duration if expansion overshot
            if (result["end"] - result["start"]) > MAX_CLIP_DURATION:
                result["end"] = result["start"] + MAX_CLIP_DURATION

            result = _validate_arc_consistency(result)

            result["duration"] = round(result["end"] - result["start"], 2)
            result["topic"] = topic["topic"]
            return result
        except Exception as e:
            last_error = e
            time.sleep(1)

    raise ArcSelectionError(f"Failed to select arc for topic '{topic['topic']}': {last_error}")


def select_clips(transcript_path: str, topics_path: str) -> dict:
    """
    For each topic in topics.json, selects the best setup->payoff clip
    within that topic's time boundaries.

    Returns dict with keys: clips, output_path
    """
    transcript_p = Path(transcript_path)
    topics_p = Path(topics_path)

    if not transcript_p.exists():
        raise ArcSelectionError(f"Transcript file not found: {transcript_path}")
    if not topics_p.exists():
        raise ArcSelectionError(f"Topics file not found: {topics_path}")

    with open(transcript_p, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    all_segments = sorted(transcript.get("segments", []), key=lambda s: s["start"])

    with open(topics_p, "r", encoding="utf-8") as f:
        topics_data = json.load(f)
    topics = topics_data.get("topics", [])

    if not topics:
        raise ArcSelectionError("No topics found in topics.json.")

    clips = []
    for i, topic in enumerate(topics):
        duration = topic["end"] - topic["start"]
        if duration < 10:
            print(f"Skipping topic {i + 1} ('{topic['topic']}') — too short ({duration:.1f}s) to yield a clip.", flush=True)
            continue

        print(f"Selecting clip for topic {i + 1}/{len(topics)}: '{topic['topic']}' "
              f"[{topic['start']:.1f}s - {topic['end']:.1f}s]...", flush=True)
        try:
            clip = _select_arc_for_topic(topic, all_segments)
            clips.append(clip)
            print(f"    -> [{clip['start']:.1f}s-{clip['end']:.1f}s] ({clip['duration']:.1f}s) "
                  f"score={clip['score']} has_arc={clip['has_arc']}", flush=True)
            time.sleep(1.5)  # pace requests to stay under Groq's per-minute token limit
        except ArcSelectionError as e:
            print(f"    Skipped: {e}", flush=True)
            continue

    if not clips:
        raise ArcSelectionError("No clips could be selected from any topic.")

    clips.sort(key=lambda c: c["score"], reverse=True)

    output_path = transcript_p.parent / "final_clips_v2.json"
    result = {
        "source_transcript": str(transcript_p),
        "source_topics": str(topics_p),
        "clips": clips,
        "output_path": str(output_path),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/scoring/arc_clip_selector.py <transcript.json> <topics.json>")
        sys.exit(1)

    try:
        result = select_clips(sys.argv[1], sys.argv[2])
        print(f"\nSelected {len(result['clips'])} clips across topics.")
        print(f"Saved to: {result['output_path']}\n")
        for c in result["clips"]:
            print(f"[{c['start']:.1f}s-{c['end']:.1f}s] ({c['duration']:.1f}s) score={c['score']} "
                  f"topic='{c['topic']}' has_arc={c['has_arc']}")
            print(f"    hook: \"{c['hook_line']}\"")
            if c["has_arc"]:
                print(f"    payoff: \"{c['payoff_line']}\"")
            print(f"    reasoning: {c['reasoning']}\n")
    except ArcSelectionError as e:
        print(f"\nClip selection failed: {e}")
        sys.exit(1)