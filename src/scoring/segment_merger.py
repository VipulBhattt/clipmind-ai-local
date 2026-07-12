"""
segment_merger.py
Step 8 of the ClipMind AI pipeline — Boundary Refinement.

Expands high-scoring "peak" segments into coherent, self-contained
clip-length windows (target: 20-60s) by merging neighboring transcript
segments, respecting natural silence-gap boundaries.
"""

import sys
import json
from pathlib import Path


class MergeError(Exception):
    """Raised when segment merging fails."""
    pass


MIN_CLIP_DURATION = 20.0
MAX_CLIP_DURATION = 60.0
SILENCE_GAP_THRESHOLD = 1.2  # seconds — a gap bigger than this is treated as a natural break


def _load_all_segments(transcript_path: str) -> list:
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    segments = transcript.get("segments", [])
    if not segments:
        raise MergeError("Transcript has no segments to merge.")
    return sorted(segments, key=lambda s: s["start"])


def _expand_window(peak_id: int, all_segments: list) -> dict:
    """
    Expands outward from a peak segment (by id) in both directions,
    stopping at silence gaps only once the minimum duration is reached.
    Below the minimum duration, gaps are ignored — we must keep expanding.
    """
    id_to_index = {seg["id"]: i for i, seg in enumerate(all_segments)}
    if peak_id not in id_to_index:
        raise MergeError(f"Peak segment id {peak_id} not found in transcript segments.")

    peak_idx = id_to_index[peak_id]
    left = right = peak_idx

    def current_duration():
        return all_segments[right]["end"] - all_segments[left]["start"]

    while current_duration() < MAX_CLIP_DURATION:
        can_left = left > 0
        can_right = right < len(all_segments) - 1

        if not can_left and not can_right:
            break  # fully exhausted both directions, nothing more to do

        left_gap = (all_segments[left]["start"] - all_segments[left - 1]["end"]) if can_left else None
        right_gap = (all_segments[right + 1]["start"] - all_segments[right]["end"]) if can_right else None

        if current_duration() < MIN_CLIP_DURATION:
            # Still under minimum — must keep expanding, ignore gap blocking entirely.
            if can_left and can_right:
                if left_gap <= right_gap:
                    left -= 1
                else:
                    right += 1
            elif can_left:
                left -= 1
            elif can_right:
                right += 1
            continue

        # At or above minimum duration — now respect silence gaps as natural stop points.
        left_blocked = (not can_left) or (left_gap > SILENCE_GAP_THRESHOLD)
        right_blocked = (not can_right) or (right_gap > SILENCE_GAP_THRESHOLD)

        if left_blocked and right_blocked:
            break  # clean natural boundary on both sides — good place to stop

        if can_left and not left_blocked and (right_blocked or left_gap <= right_gap):
            left -= 1
        elif can_right and not right_blocked:
            right += 1
        else:
            break

    merged_text = " ".join(all_segments[i]["text"] for i in range(left, right + 1))

    return {
        "start": all_segments[left]["start"],
        "end": all_segments[right]["end"],
        "duration": round(all_segments[right]["end"] - all_segments[left]["start"], 2),
        "text": merged_text.strip(),
        "peak_segment_id": peak_id,
        "included_segment_ids": [all_segments[i]["id"] for i in range(left, right + 1)],
    }

def _windows_overlap(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def merge_candidates(transcript_path: str, final_scores_path: str, max_clips: int = 5) -> dict:
    """
    Expands top-ranked peak segments into full clip windows, removing
    overlapping duplicates, and returns the top `max_clips` non-overlapping
    clips ranked by their original peak's final_score.

    Returns dict with keys: clips, output_path
    """
    transcript_p = Path(transcript_path)
    scores_p = Path(final_scores_path)

    if not transcript_p.exists():
        raise MergeError(f"Transcript file not found: {transcript_path}")
    if not scores_p.exists():
        raise MergeError(f"Final scores file not found: {final_scores_path}")

    all_segments = _load_all_segments(str(transcript_p))

    with open(scores_p, "r", encoding="utf-8") as f:
        scores_data = json.load(f)

    ranked_candidates = scores_data.get("final_ranked_candidates", [])
    if not ranked_candidates:
        raise MergeError("No ranked candidates found in final scores file.")

    accepted_windows = []
    for cand in ranked_candidates:
        window = _expand_window(cand["id"], all_segments)
        window["source_final_score"] = cand["final_score"]
        window["source_reasoning"] = cand.get("reasoning", "")

        # Skip if this window overlaps significantly with an already-accepted one
        if any(_windows_overlap(window, accepted) for accepted in accepted_windows):
            continue

        accepted_windows.append(window)
        if len(accepted_windows) >= max_clips:
            break

    if not accepted_windows:
        raise MergeError("No non-overlapping clip windows could be produced.")

    output_path = transcript_p.parent / "final_clips.json"
    result = {
        "source_transcript": str(transcript_p),
        "source_scores": str(scores_p),
        "clips": accepted_windows,
        "output_path": str(output_path),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/scoring/segment_merger.py <transcript.json> <final_scored_clips.json>")
        sys.exit(1)

    try:
        result = merge_candidates(sys.argv[1], sys.argv[2])
        print(f"\nProduced {len(result['clips'])} clip windows.")
        print(f"Saved to: {result['output_path']}\n")
        for i, clip in enumerate(result["clips"]):
            print(f"Clip {i + 1}: [{clip['start']:.1f}s - {clip['end']:.1f}s] "
                  f"({clip['duration']:.1f}s) score={clip['source_final_score']}")
            print(f"    \"{clip['text'][:100]}...\"")
            print(f"    reasoning: {clip['source_reasoning']}\n")
    except MergeError as e:
        print(f"\nMerging failed: {e}")
        sys.exit(1)
        
        
        
        