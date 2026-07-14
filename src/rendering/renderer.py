"""
renderer.py
Takes ranked clip windows and the source YouTube video, downloads each
clip's video segment, and reframes it to 9:16 vertical (simple center-crop
for now — smart face-tracking crop is a planned future upgrade).
"""

import sys
import json
import subprocess
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # allow importing sibling modules
from ingestion.downloader import download_video_segment, DownloadError, _check_ffmpeg_available


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


def _center_crop_to_vertical(input_path: str, output_path: str, target_ratio: float = 9 / 16) -> str:
    """
    Crops a landscape video to a vertical (9:16) aspect ratio, centered
    horizontally. This is a placeholder for smart, face-tracked cropping
    planned for a future session.
    """
    width, height = _get_video_dimensions(input_path)
    target_width = int(height * target_ratio)

    if target_width > width:
        # Video is already narrower than target — crop height instead
        target_height = int(width / target_ratio)
        crop_filter = f"crop={width}:{target_height}:0:(ih-{target_height})/2"
    else:
        crop_filter = f"crop={target_width}:{height}:(iw-{target_width})/2:0"

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
        raise RenderError(f"ffmpeg crop failed: {e.stderr}")

    return output_path


def render_clips(ranked_clips_path: str, youtube_url: str, output_dir: str = None, top_n: int = 5) -> dict:
    """
    Renders the top N ranked clips into vertical, cropped MP4 files.

    Args:
        ranked_clips_path: path to final_clips_ranked.json
        youtube_url: original source video URL (needed to re-fetch video bytes)
        output_dir: where to save rendered clips (defaults to same folder as ranked_clips_path)
        top_n: how many top-ranked clips to render

    Returns:
        dict with keys: rendered_clips (list of output paths + metadata), output_dir
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

        print(f"    Cropping to vertical 9:16...", flush=True)
        try:
            _center_crop_to_vertical(str(raw_path), str(cropped_path))
        except RenderError as e:
            print(f"    Skipped — crop failed: {e}", flush=True)
            continue

        rendered.append({
            "rank": i + 1,
            "topic": clip["topic"],
            "start": clip["start"],
            "end": clip["end"],
            "rank_score": clip.get("rank_score"),
            "hook_line": clip.get("hook_line"),
            "output_path": str(cropped_path),
        })
        print(f"    Done: {cropped_path}\n", flush=True)

    if not rendered:
        raise RenderError("No clips were successfully rendered.")

    manifest_path = out_dir / "render_manifest.json"
    result = {"rendered_clips": rendered, "output_dir": str(out_dir), "manifest_path": str(manifest_path)}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/rendering/renderer.py <final_clips_ranked.json> <youtube_url> [top_n]")
        sys.exit(1)

    ranked_path = sys.argv[1]
    url = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    try:
        result = render_clips(ranked_path, url, top_n=n)
        print(f"\nRendered {len(result['rendered_clips'])} clips to: {result['output_dir']}")
    except RenderError as e:
        print(f"\nRendering failed: {e}")
        sys.exit(1)