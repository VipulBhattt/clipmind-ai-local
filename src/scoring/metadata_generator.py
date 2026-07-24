"""
metadata_generator.py
Generates a scroll-stopping title, short caption/description, and
relevant hashtags for a rendered clip, based on its transcript text.
"""

import os
import sys
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"

METADATA_PROMPT_TEMPLATE = """You are writing social media metadata for a short-form video clip (YouTube Shorts / Instagram Reels / TikTok) taken from a longer video.

Topic: "{topic}"
Clip transcript: "{text}"

Write:
- A short, scroll-stopping title (under 60 characters, no clickbait lies, must accurately reflect the content)
- A 1-2 sentence caption suitable for the post description
- 5-8 relevant hashtags (mix of broad and specific, no spaces, include the # symbol)

Respond with ONLY valid JSON in this exact format, no other text:
{{
  "title": "<title>",
  "caption": "<caption>",
  "hashtags": ["#tag1", "#tag2", "..."]
}}
"""


class MetadataError(Exception):
    pass


def _call_groq(prompt: str) -> str:
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise MetadataError("GROQ_API_KEY not found in .env")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,  # a bit more creative than scoring calls
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


def generate_metadata_for_clip(topic: str, text: str, retries: int = 3) -> dict:
    """
    Generates title, caption, and hashtags for a single clip.

    Returns dict with keys: title, caption, hashtags
    """
    prompt = METADATA_PROMPT_TEMPLATE.format(topic=topic, text=text)

    last_error = None
    for attempt in range(retries):
        try:
            raw = _call_groq(prompt)
            result = _parse_json_obj(raw)
            required = ["title", "caption", "hashtags"]
            if not all(k in result for k in required):
                raise ValueError(f"Missing required keys: {result}")
            if not isinstance(result["hashtags"], list):
                raise ValueError("hashtags must be a list")
            return result
        except Exception as e:
            last_error = e
            error_str = str(e)
            if "rate_limit_exceeded" in error_str or "429" in error_str:
                time.sleep(5)
            else:
                time.sleep(1)

    raise MetadataError(f"Failed to generate metadata after {retries} retries: {last_error}")


def add_metadata_to_manifest(manifest_path: str, transcript_path: str) -> dict:
    """
    Loads a render_manifest.json (produced by renderer.py), generates
    metadata for each clip using its full transcript text, and saves an
    updated manifest with title, caption, and hashtags added.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise MetadataError(f"Manifest file not found: {manifest_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    all_segments = transcript.get("segments", [])

    clips = data.get("rendered_clips", [])
    if not clips:
        raise MetadataError("No rendered clips found in manifest.")

    for i, clip in enumerate(clips):
        print(f"[{i+1}/{len(clips)}] Generating metadata for '{clip['topic']}'...", flush=True)

        # Pull the FULL transcript text for this clip's actual time range
        clip_text = " ".join(
            seg["text"] for seg in all_segments
            if seg["start"] >= clip["start"] and seg["end"] <= clip["end"]
        )
        if not clip_text.strip():
            clip_text = clip.get("hook_line", clip["topic"])  # fallback only if truly nothing found

        try:
            meta = generate_metadata_for_clip(clip["topic"], clip_text)
            clip["title"] = meta["title"]
            clip["caption"] = meta["caption"]
            clip["hashtags"] = meta["hashtags"]
            print(f"    -> \"{meta['title']}\"", flush=True)
        except MetadataError as e:
            print(f"    Skipped: {e}", flush=True)
            clip["title"] = clip.get("topic", "Untitled Clip")
            clip["caption"] = ""
            clip["hashtags"] = []

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/scoring/metadata_generator.py <render_manifest.json> <transcript.json>")
        sys.exit(1)

    try:
        result = add_metadata_to_manifest(sys.argv[1], sys.argv[2])
        print(f"\nMetadata generation complete. Updated manifest saved.\n")
        for c in result["rendered_clips"]:
            print(f"'{c['title']}'")
            print(f"  {c['caption']}")
            print(f"  {' '.join(c['hashtags'])}\n")
    except MetadataError as e:
        print(f"\nMetadata generation failed: {e}")
        sys.exit(1)