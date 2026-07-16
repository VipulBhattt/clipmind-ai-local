"""
renderer.py
Takes ranked clip windows and the source YouTube video, downloads each
clip's video segment, and reframes it to 9:16 vertical, dynamically
following the active speaker (instant cuts, no smoothing) using the
timeline produced by speaker_detector.py. Falls back to plain center-crop
if speaker detection fails or no timeline is available.
"""

import sys
import json
import subprocess
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # allow importing sibling modules
from ingestion.downloader import download_video_segment, DownloadError, _check_ffmpeg_available
from rendering.speaker_detector import analyze_video, SpeakerDetectionError
from rendering.caption_burner import add_captions_to_clip, CaptionError





class RenderError(Exception):
    pass


def _get_video_dimensions(video_path: str) -> tuple:
    """Uses ffprobe to get width, height of a video file."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            video_path,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RenderError(f"ffprobe failed on {video_path}: {result.stderr}")
    width, height = map(int, result.stdout.strip().split(","))
    return width, height


def _target_crop_dimensions(width: int, height: int, target_ratio: float = 9 / 16) -> tuple:
    """Returns (crop_width, crop_height) for a 9:16 crop of a given source frame."""
    target_width = int(height * target_ratio)
    if target_width <= width:
        return target_width, height
    target_height = int(width / target_ratio)
    return width, target_height


def _center_crop_to_vertical(input_path: str, output_path: str, target_ratio: float = 9 / 16) -> str:
    """Fallback: plain center-crop, used if speaker detection fails entirely."""
    width, height = _get_video_dimensions(input_path)
    crop_width, crop_height = _target_crop_dimensions(width, height, target_ratio)
    x = (width - crop_width) // 2

    crop_filter = f"crop={crop_width}:{crop_height}:{x}:0"

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-vf", crop_filter,
                "-c:a", "copy",
                output_path,
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RenderError(f"ffmpeg center-crop failed: {e.stderr}")

    return output_path


def _center_x_to_crop_x(center_x_norm: float, frame_width: int, crop_width: int) -> int:
    """Converts a normalized (0-1) face center X into a pixel crop X position, clamped to frame bounds."""
    pixel_center = center_x_norm * frame_width
    crop_x = int(pixel_center - crop_width / 2)
    crop_x = max(0, min(crop_x, frame_width - crop_width))
    return crop_x


def _build_dynamic_crop_filter(timeline: list, frame_width: int, frame_height: int,
                                crop_width: int, crop_height: int) -> str:
    """
    Builds an ffmpeg crop filter with a time-varying x position that jumps
    (instant cut, no smoothing) to follow the active speaker according to
    the given timeline.
    """
    default_x = (frame_width - crop_width) // 2

    segments = []
    for entry in timeline:
        if entry.get("no_detection") or entry.get("face_center_x") is None:
            x_pos = default_x
        else:
            x_pos = _center_x_to_crop_x(entry["face_center_x"], frame_width, crop_width)
        segments.append((entry["start"], entry["end"], x_pos))

    if not segments:
        return f"crop={crop_width}:{crop_height}:{default_x}:0"

    expr = str(segments[-1][2])
    for start, end, x_pos in reversed(segments[:-1]):
        expr = f"if(lt(t\\,{end})\\,{x_pos}\\,{expr})"

    return f"crop={crop_width}:{crop_height}:{expr}:0"


def _dynamic_crop_to_vertical(input_path: str, output_path: str, timeline: list,
                               target_ratio: float = 9 / 16) -> str:
    width, height = _get_video_dimensions(input_path)
    crop_width, crop_height = _target_crop_dimensions(width, height, target_ratio)
    crop_filter = _build_dynamic_crop_filter(timeline, width, height, crop_width, crop_height)

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-vf", crop_filter,
                "-c:a", "copy",
                output_path,
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RenderError(f"ffmpeg dynamic crop failed: {e.stderr}")

    return output_path


def render_clips(ranked_clips_path: str, youtube_url: str, transcript_path: str,
                  output_dir: str = None, top_n: int = 5) -> dict:
    """
    Renders the top N ranked clips into vertical MP4 files, with the crop
    dynamically following the active speaker (instant cuts on speaker change).
    Falls back to center-crop if speaker detection fails for a given clip.
    """
    _check_ffmpeg_available()

    path = Path(ranked_clips_path)
    if not path.exists():
        raise RenderError(f"Ranked clips file not found: {ranked_clips_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clips = data.get("clips", [])
    if not clips:
        raise RenderError("No clips found in ranked clips file.")

    top_clips = sorted(clips, key=lambda c: c.get("rank_score", 0), reverse=True)[:top_n]

    out_dir = Path(output_dir) if output_dir else path.parent / "rendered_clips"
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = []
    for i, clip in enumerate(top_clips):
        raw_path = out_dir / f"clip_{i+1}_raw.mp4"
        cropped_path = out_dir / f"clip_{i+1}_vertical.mp4"

        print(f"[{i+1}/{len(top_clips)}] Downloading video segment "
              f"[{clip['start']:.1f}s - {clip['end']:.1f}s] ({clip['topic']})...", flush=True)
        try:
            download_video_segment(youtube_url, clip["start"], clip["end"], str(raw_path))
        except DownloadError as e:
            print(f"    Skipped — download failed: {e}", flush=True)
            continue

        print(f"    Analyzing active speaker position...", flush=True)
        crop_mode = "dynamic"
        timeline = None
        try:
            speaker_result = analyze_video(str(raw_path))
            timeline = speaker_result["timeline"]
        except SpeakerDetectionError as e:
            print(f"    Speaker detection failed ({e}) — falling back to center-crop.", flush=True)
            crop_mode = "center"

        print(f"    Cropping to vertical 9:16 ({crop_mode})...", flush=True)
        try:
            if crop_mode == "dynamic":
                _dynamic_crop_to_vertical(str(raw_path), str(cropped_path), timeline)
            else:
                _center_crop_to_vertical(str(raw_path), str(cropped_path))
        except RenderError as e:
            print(f"    Skipped — crop failed: {e}", flush=True)
            continue

        captioned_path = out_dir / f"clip_{i+1}_final.mp4"
        print(f"    Burning in captions...", flush=True)
        try:
            add_captions_to_clip(str(cropped_path), transcript_path, clip["start"], clip["end"], str(captioned_path))
            final_output = str(captioned_path)
        except CaptionError as e:
            print(f"    Caption burn-in failed ({e}) — using uncaptioned version instead.", flush=True)
            final_output = str(cropped_path)

        rendered.append({
            "rank": i + 1,
            "topic": clip["topic"],
            "start": clip["start"],
            "end": clip["end"],
            "rank_score": clip.get("rank_score"),
            "hook_line": clip.get("hook_line"),
            "crop_mode": crop_mode,
            "output_path": final_output,
        })
        print(f"    Done: {final_output}\n", flush=True)

    if not rendered:
        raise RenderError("No clips were successfully rendered.")

    manifest_path = out_dir / "render_manifest.json"
    result = {"rendered_clips": rendered, "output_dir": str(out_dir), "manifest_path": str(manifest_path)}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python src/rendering/renderer.py <final_clips_ranked.json> <youtube_url> <transcript.json> [top_n]")
        sys.exit(1)

    ranked_path = sys.argv[1]
    url = sys.argv[2]
    transcript_path = sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    try:
        result = render_clips(ranked_path, url, transcript_path, top_n=n)
        print(f"\nRendered {len(result['rendered_clips'])} clips to: {result['output_dir']}")
        for c in result["rendered_clips"]:
            print(f"  rank {c['rank']} | crop_mode={c['crop_mode']} | {c['output_path']}")
    except RenderError as e:
        print(f"\nRendering failed: {e}")
        sys.exit(1)