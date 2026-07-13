"""
clip_ranker.py
Takes all arc-selected candidate clips from one video and re-ranks them
relative to each other, since scoring each clip in isolation tends to
cluster scores together (e.g., everything scoring 8/10).
"""

import os
import re
import sys
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"

RANKING_PROMPT_TEMPLATE = """You are ranking candidate short-form video clips from the SAME source video, to decide which ones deserve to be published first.

Here are all the candidates:
{candidates_block}

Task: Rank ALL of these candidates from best to worst as standalone short-form clips, judging them RELATIVE TO EACH OTHER — not in isolation. Consider: strength of hook, clarity of payoff, how likely someone is to watch the whole thing and not scroll away, and how different/non-repetitive each clip is compared to the others.

Give each one a rank_score from 1-100 (ties allowed only if truly equal), where the differences in score should reflect real differences in quality — do not cluster everything near the same number.

Respond with ONLY valid JSON, a list like this, no other text:
[
  {{"index": <int, matching the candidate index below>, "rank_score": <int 1-100>, "rank_reasoning": "<short comparison-based reasoning>"}},
  ...
]
"""


class RankingError(Exception):
    pass


def _call_groq(prompt: str) -> str:
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RankingError("GROQ_API_KEY not found in .env")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
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


def _extract_wait_seconds(error_message: str) -> float:
    """Parses Groq's rate-limit error to find the suggested wait time."""
    match = re.search(r"try again in ([\d.]+)s", error_message)
    if match:
        return float(match.group(1)) + 0.5  # small buffer
    return 5.0  # sensible default if we can't parse it


def rank_clips(clips_path: str, retries: int = 4) -> dict:
    """
    Loads final_clips_v2.json and re-ranks all clips relative to each other.
    Returns dict with keys: clips (re-ranked, with rank_score + rank_reasoning), output_path
    """
    path = Path(clips_path)
    if not path.exists():
        raise RankingError(f"Clips file not found: {clips_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clips = data.get("clips", [])
    if not clips:
        raise RankingError("No clips found to rank.")

    candidates_block = ""
    for i, c in enumerate(clips):
        candidates_block += (
            f"\n[Candidate {i}] Topic: {c['topic']} | Duration: {c['duration']}s\n"
            f"  Hook: \"{c['hook_line']}\"\n"
            f"  Payoff: \"{c.get('payoff_line', '') or '(none)'}\"\n"
            f"  Has arc: {c['has_arc']}\n"
        )

    prompt = RANKING_PROMPT_TEMPLATE.format(candidates_block=candidates_block)

    last_error = None
    rankings = None
    for attempt in range(retries):
        try:
            raw = _call_groq(prompt)
            rankings = _parse_json_list(raw)
            if not isinstance(rankings, list) or not rankings:
                raise ValueError("Empty or invalid ranking list returned.")
            break
        except Exception as e:
            last_error = e
            error_str = str(e)
            if "rate_limit_exceeded" in error_str or "429" in error_str:
                wait_time = _extract_wait_seconds(error_str)
                print(f"    Rate limit hit — waiting {wait_time:.1f}s before retry "
                      f"({attempt + 1}/{retries})...", flush=True)
                time.sleep(wait_time)
            else:
                time.sleep(1)

    if rankings is None:
        raise RankingError(f"Failed to get rankings after {retries} retries: {last_error}")

    rank_by_index = {r["index"]: r for r in rankings if "index" in r}

    ranked_clips = []
    for i, clip in enumerate(clips):
        rank_info = rank_by_index.get(i, {})
        merged = {
            **clip,
            "rank_score": rank_info.get("rank_score", 0),
            "rank_reasoning": rank_info.get("rank_reasoning", "No reasoning provided by LLM."),
        }
        ranked_clips.append(merged)

    ranked_clips.sort(key=lambda c: c["rank_score"], reverse=True)

    output_path = path.parent / "final_clips_ranked.json"
    result = {
        "source": str(path),
        "clips": ranked_clips,
        "output_path": str(output_path),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/scoring/clip_ranker.py <path_to_final_clips_v2.json>")
        sys.exit(1)

    try:
        result = rank_clips(sys.argv[1])
        print(f"\nRanked {len(result['clips'])} clips.")
        print(f"Saved to: {result['output_path']}\n")
        for c in result["clips"]:
            print(f"rank_score={c['rank_score']} | [{c['start']:.1f}s-{c['end']:.1f}s] "
                  f"({c['duration']:.1f}s) topic='{c['topic']}'")
            print(f"    {c['rank_reasoning']}\n")
    except RankingError as e:
        print(f"\nRanking failed: {e}")
        sys.exit(1)