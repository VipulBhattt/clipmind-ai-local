"""
downloader.py
Ingestion module for ClipMind AI.

Downloads the audio track of a YouTube video, converts it to a clean
16kHz mono WAV (Whisper-ready format), and saves accompanying metadata.
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

import yt_dlp


class DownloadError(Exception):
    """Raised when a video cannot be fetched or processed."""
    pass


def _check_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise DownloadError(
            "ffmpeg was not found on PATH. Install it and make sure it's "
            "accessible from the terminal before running this script."
        )


def download_audio(youtube_url: str, output_dir: str = "downloads") -> dict:
    """
    Downloads the audio stream of a YouTube video and converts it to a
    16kHz mono WAV file suitable for Whisper transcription.

    Args:
        youtube_url: Full YouTube video URL.
        output_dir: Base directory where per-video folders will be created.

    Returns:
        dict with keys:
            video_id, title, channel, duration, upload_date,
            original_url, audio_path, metadata_path

    Raises:
        DownloadError: on invalid URL, unavailable video, or failed download.
    """
    if not youtube_url or not isinstance(youtube_url, str):
        raise DownloadError("A valid YouTube URL string must be provided.")

    _check_ffmpeg_available()

    # Step 1: Extract metadata first (without downloading) to validate the URL early
    ydl_opts_probe = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts_probe) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise DownloadError(f"Could not access video at '{youtube_url}': {e}")
    except Exception as e:
        raise DownloadError(f"Unexpected error while probing URL: {e}")

    if not info:
        raise DownloadError(f"No video information could be retrieved for '{youtube_url}'.")

    video_id = info.get("id")
    if not video_id:
        raise DownloadError("Could not determine a video ID — the URL may be invalid.")

    # Step 2: Set up per-video output folder
    video_folder = Path(output_dir) / video_id
    video_folder.mkdir(parents=True, exist_ok=True)

    raw_audio_template = str(video_folder / "raw_audio.%(ext)s")
    final_audio_path = video_folder / "audio.wav"
    metadata_path = video_folder / "metadata.json"

    # Step 3: Download best available audio-only stream
    ydl_opts_download = {
        "format": "bestaudio/best",
        "outtmpl": raw_audio_template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts_download) as ydl:
            ydl.download([youtube_url])
    except yt_dlp.utils.DownloadError as e:
        raise DownloadError(f"Failed to download audio for '{youtube_url}': {e}")
    except Exception as e:
        raise DownloadError(f"Unexpected error during download: {e}")

    # Find whatever file yt-dlp actually produced (extension varies: .webm, .m4a, etc.)
    raw_files = list(video_folder.glob("raw_audio.*"))
    if not raw_files:
        raise DownloadError(
            "Download appeared to succeed but no raw audio file was found on disk."
        )
    raw_audio_path = raw_files[0]

    # Step 4: Convert to 16kHz mono WAV using ffmpeg
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(raw_audio_path),
                "-ac", "1",          # mono
                "-ar", "16000",      # 16kHz sample rate
                "-vn",                # no video stream
                str(final_audio_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise DownloadError(f"ffmpeg conversion failed: {e.stderr}")

    # Clean up the raw intermediate file now that WAV conversion succeeded
    try:
        raw_audio_path.unlink()
    except OSError:
        pass  # not critical if cleanup fails

    if not final_audio_path.exists():
        raise DownloadError("Audio conversion finished but WAV file was not created.")

    # Step 5: Build and save metadata
    metadata = {
        "video_id": video_id,
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "duration": info.get("duration"),  # seconds
        "upload_date": info.get("upload_date"),  # YYYYMMDD
        "original_url": youtube_url,
        "audio_path": str(final_audio_path),
        "metadata_path": str(metadata_path),
    }

    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except OSError as e:
        raise DownloadError(f"Failed to write metadata.json: {e}")

    return metadata


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/ingestion/downloader.py <youtube_url>")
        sys.exit(1)

    url = sys.argv[1]
    try:
        result = download_audio(url)
        print("\nDownload successful.\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except DownloadError as e:
        print(f"\nDownload failed: {e}")
        sys.exit(1)