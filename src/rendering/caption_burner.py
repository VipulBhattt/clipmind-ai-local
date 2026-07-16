"""
caption_burner.py
Burns styled, timed captions into rendered vertical clips, using the
word-level timestamps already produced by the transcription module.
Generates an .ass subtitle file (supports styling, unlike plain .srt)
and burns it in via ffmpeg's subtitles filter.
"""

import sys
import json
import subprocess
from pathlib import Path


class CaptionError(Exception):
    pass


WORDS_PER_CAPTION = 5  # how many words to group into one on-screen caption chunk

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,72,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,4,2,2,60,60,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


class CaptionError(Exception):
    pass


def _format_ass_time(seconds: float) -> str:
    """Formats seconds as ASS timestamp: H:MM:SS.CC"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _get_words_in_range(transcript_path: str, clip_start: float, clip_end: float) -> list:
    """Extracts all words (with timestamps) falling within the clip's time range,
    re-based so timestamps are relative to the clip's own start (0.0)."""
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    words = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            if w["start"] >= clip_start and w["end"] <= clip_end:
                words.append({
                    "word": w["word"],
                    "start": round(w["start"] - clip_start, 2),
                    "end": round(w["end"] - clip_start, 2),
                })
    return words


def _group_words(words: list, group_size: int = WORDS_PER_CAPTION) -> list:
    """Groups words into caption chunks of group_size, each with a start/end time
    spanning its first and last word."""
    groups = []
    for i in range(0, len(words), group_size):
        chunk = words[i:i + group_size]
        if not chunk:
            continue
        groups.append({
            "text": " ".join(w["word"].strip() for w in chunk),
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
        })
    return groups


def generate_ass_file(transcript_path: str, clip_start: float, clip_end: float, output_ass_path: str) -> str:
    """
    Builds an .ass subtitle file for one clip, using word-level timestamps
    from the full video's transcript, re-based to the clip's own timeline.
    """
    words = _get_words_in_range(transcript_path, clip_start, clip_end)
    if not words:
        raise CaptionError(f"No words found in transcript for range [{clip_start}-{clip_end}]s.")

    groups = _group_words(words)

    lines = [ASS_HEADER]
    for g in groups:
        start_str = _format_ass_time(g["start"])
        end_str = _format_ass_time(g["end"])
        # Escape any characters that have special meaning in ASS text
        text = g["text"].replace("\n", " ").replace("{", "").replace("}", "")
        lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}")

    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_ass_path


def burn_captions(video_path: str, ass_path: str, output_path: str) -> str:
    """
    Burns the given .ass subtitle file into the video using ffmpeg's
    subtitles filter.
    """
    video_p = Path(video_path).resolve()
    ass_p = Path(ass_path).resolve()

    if not video_p.exists():
        raise CaptionError(f"Video file not found: {video_path}")
    if not ass_p.exists():
        raise CaptionError(f"Subtitle file not found: {ass_path}")

    # ffmpeg's subtitles filter needs forward slashes and escaped colons on Windows paths
    ass_filter_path = str(ass_p).replace("\\", "/").replace(":", "\\:")

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_p),
                "-vf", f"subtitles='{ass_filter_path}'",
                "-c:a", "copy",
                str(output_path),
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        raise CaptionError(f"ffmpeg caption burn-in failed: {e.stderr}")

    return output_path


def add_captions_to_clip(video_path: str, transcript_path: str, clip_start: float,
                          clip_end: float, output_path: str) -> str:
    """
    Full pipeline: generates the .ass file for this clip's time range and
    burns it into the given video, producing the final captioned output.
    """
    ass_path = str(Path(output_path).with_suffix(".ass"))
    generate_ass_file(transcript_path, clip_start, clip_end, ass_path)
    return burn_captions(video_path, ass_path, output_path)


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python src/rendering/caption_burner.py <vertical_video.mp4> <transcript.json> <clip_start> <clip_end> <output.mp4>")
        sys.exit(1)

    video = sys.argv[1]
    transcript = sys.argv[2]
    start = float(sys.argv[3])
    end = float(sys.argv[4])
    output = sys.argv[5]

    try:
        result = add_captions_to_clip(video, transcript, start, end, output)
        print(f"\nCaptioned video saved to: {result}")
    except CaptionError as e:
        print(f"\nCaption burn-in failed: {e}")
        sys.exit(1)