"""
llm_scorer.py
Layer 2 of the ClipMind AI scoring pipeline.

Takes heuristic-filtered candidate segments and scores them using an LLM
(Groq primary, Ollama local fallback) for semantic judgment: hook strength,
standalone clarity, emotional payload, and quotability.
"""

import os
import sys
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
OLLAMA_MODEL = "llama3.2:3b"

SCORING_PROMPT_TEMPLATE = """You are evaluating a short segment of transcript from a longer video to judge whether it would work well as a standalone short-form video clip (like a YouTube Short or Instagram Reel).

Transcript segment:
"{text}"

Context: this segment runs from {start}s to {end}s in the source video.

Rate this segment on the following, each from 0 to 10:
- hook_strength: Does it grab attention in the first few seconds?
- standalone_clarity: Does it make sense without any other context from the video?
- emotional_payload: Does it carry humor, surprise, tension, or strong emotion?
- quotability: Is there a strong, memorable, shareable line in it?

Respond with ONLY valid JSON in exactly this format, no other text:
{{
  "hook_strength": <int 0-10>,
  "standalone_clarity": <int 0-10>,
  "emotional_payload": <int 0-10>,
  "quotability": <int 0-10>,
  "reasoning": "<one short sentence explaining the scores>"
}}
"""


class LLMScoringError(Exception):
    """Raised when LLM scoring fails for both providers."""
    pass


def _build_prompt(text: str, start: float, end: float) -> str:
    return SCORING_PROMPT_TEMPLATE.format(text=text, start=start, end=end)


def _parse_llm_json(raw_text: str) -> dict:
    """Extracts and parses JSON from an LLM response, tolerating minor formatting noise."""
    raw_text = raw_text.strip()
    # Strip markdown code fences if the model added them despite instructions
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to salvage by locating the first { and last }
        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            try:
                return json.loads(raw_text[start_idx:end_idx + 1])
            except json.JSONDecodeError:
                pass
        raise


def _score_with_groq(text: str, start: float, end: float) -> dict:
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise LLMScoringError("GROQ_API_KEY not found in .env")

    client = Groq(api_key=api_key)
    prompt = _build_prompt(text, start, end)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # low temperature — we want consistent scoring, not creativity
    )
    raw = response.choices[0].message.content
    return _parse_llm_json(raw)


def _score_with_ollama(text: str, start: float, end: float) -> dict:
    import ollama

    prompt = _build_prompt(text, start, end)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3},
    )
    raw = response["message"]["content"]
    return _parse_llm_json(raw)


def score_segment_llm(text: str, start: float, end: float, provider: str = "groq", retries: int = 2) -> dict:
    """
    Scores a single segment using the LLM, with automatic fallback to Ollama
    if the primary provider (Groq) fails after retries.

    Returns dict with hook_strength, standalone_clarity, emotional_payload,
    quotability, reasoning, and a combined llm_score (0-10 average), plus
    which provider actually produced the result.
    """
    providers_to_try = [provider]
    if provider != "ollama":
        providers_to_try.append("ollama")  # fallback

    last_error = None
    for prov in providers_to_try:
        for attempt in range(retries):
            try:
                if prov == "groq":
                    result = _score_with_groq(text, start, end)
                else:
                    result = _score_with_ollama(text, start, end)

                required_keys = ["hook_strength", "standalone_clarity", "emotional_payload", "quotability"]
                if not all(k in result for k in required_keys):
                    raise ValueError(f"LLM response missing required keys: {result}")

                llm_score = sum(result[k] for k in required_keys) / (len(required_keys) * 10)

                return {
                    "hook_strength": result["hook_strength"],
                    "standalone_clarity": result["standalone_clarity"],
                    "emotional_payload": result["emotional_payload"],
                    "quotability": result["quotability"],
                    "reasoning": result.get("reasoning", ""),
                    "llm_score": round(llm_score, 3),
                    "llm_provider_used": prov,
                }
            except Exception as e:
                last_error = e
                time.sleep(1)  # brief pause before retry
                continue
        # move to next provider if this one exhausted retries

    raise LLMScoringError(
        f"LLM scoring failed on all providers ({providers_to_try}) after {retries} retries each. "
        f"Last error: {last_error}"
    )


def score_candidates(heuristic_scores_path: str, provider: str = "groq") -> dict:
    """
    Loads heuristic_scores.json, sends only the top candidates to the LLM
    for semantic scoring, and combines heuristic + LLM scores into a final
    ranked list.

    Returns dict with keys: source, final_ranked_candidates, output_path
    """
    path = Path(heuristic_scores_path)
    if not path.exists():
        raise LLMScoringError(f"Heuristic scores file not found: {heuristic_scores_path}")

    with open(path, "r", encoding="utf-8") as f:
        heuristic_data = json.load(f)

    candidates = heuristic_data.get("top_candidates", [])
    if not candidates:
        raise LLMScoringError("No top_candidates found in heuristic scores file.")

    final_candidates = []
    for i, cand in enumerate(candidates):
        print(f"Scoring candidate {i + 1}/{len(candidates)} "
              f"[{cand['start']:.1f}s - {cand['end']:.1f}s] via {provider}...", flush=True)
        try:
            llm_result = score_segment_llm(cand["text"], cand["start"], cand["end"], provider=provider)
        except LLMScoringError as e:
            print(f"    Skipped due to error: {e}", flush=True)
            continue

        # Final combined score: weighted blend of heuristic (cheap signal) and LLM (deep judgment)
        final_score = round(0.3 * cand["heuristic_score"] + 0.7 * llm_result["llm_score"], 3)

        final_candidates.append({
            **cand,
            **llm_result,
            "final_score": final_score,
        })
        print(f"    -> llm_score={llm_result['llm_score']} "
              f"final_score={final_score} (via {llm_result['llm_provider_used']})", flush=True)

    if not final_candidates:
        raise LLMScoringError("All candidates failed LLM scoring.")

    final_candidates.sort(key=lambda c: c["final_score"], reverse=True)

    output_path = path.parent / "final_scored_clips.json"
    result = {
        "source": str(path),
        "final_ranked_candidates": final_candidates,
        "output_path": str(output_path),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/scoring/llm_scorer.py <path_to_heuristic_scores.json> [provider]")
        sys.exit(1)

    scores_path = sys.argv[1]
    provider_arg = sys.argv[2] if len(sys.argv) > 2 else "groq"

    try:
        result = score_candidates(scores_path, provider=provider_arg)
        print(f"\nDone. Saved final ranked clips to: {result['output_path']}\n")
        print("Top 3 final candidates:\n")
        for c in result["final_ranked_candidates"][:3]:
            print(f"[{c['start']:.1f}s - {c['end']:.1f}s] final_score={c['final_score']}")
            print(f"    \"{c['text'][:80]}\"")
            print(f"    reasoning: {c['reasoning']}\n")
    except LLMScoringError as e:
        print(f"\nLLM scoring failed: {e}")
        sys.exit(1)