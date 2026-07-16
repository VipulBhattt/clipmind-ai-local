"""
main.py
ClipMind AI — full pipeline orchestrator.

Runs the entire pipeline end-to-end from a single YouTube URL:
ingestion -> transcription -> topic segmentation -> arc-based clip
selection -> relative ranking -> rendering (crop + captions).

Usage:
    python main.py <youtube_url> [--top_n N]
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from ingestion.downloader import download_audio, DownloadError
from transcription.transcriber import transcribe_audio, TranscriptionError
from scoring.topic_segmenter import segment_transcript, TopicSegmentationError
from scoring.arc_clip_selector import select_clips, ArcSelectionError
from scoring.clip_ranker import rank_clips, RankingError
from rendering.renderer import render_clips, RenderError


class PipelineError(Exception):
    pass


def _stage_header(stage_num: int, total: int, name: str):
    print(f"\n{'='*60}")
    print(f"STAGE {stage_num}/{total}: {name}")
    print(f"{'='*60}", flush=True)


def run_pipeline(youtube_url: str, top_n: int = 5) -> dict:
    """
    Runs the full ClipMind AI pipeline on a single YouTube video.

    Returns dict with paths to all key intermediate and final outputs.
    """
    total_stages = 6
    pipeline_start = time.time()
    results = {}

    # Stage 1: Ingestion
    _stage_header(1, total_stages, "Downloading audio + metadata")
    try:
        ingestion_result = download_audio(youtube_url)
    except DownloadError as e:
        raise PipelineError(f"Ingestion failed: {e}")
    audio_path = ingestion_result["audio_path"]
    video_folder = str(Path(audio_path).parent)
    results["metadata"] = ingestion_result
    print(f"Done. Video: '{ingestion_result['title']}' ({ingestion_result['duration']}s)")

    # Stage 2: Transcription
    _stage_header(2, total_stages, "Transcribing audio")
    try:
        transcript_result = transcribe_audio(audio_path)
    except TranscriptionError as e:
        raise PipelineError(f"Transcription failed: {e}")
    transcript_path = transcript_result["transcript_path"]
    results["transcript_path"] = transcript_path
    print(f"Done. Language: {transcript_result['language']}, "
          f"{len(transcript_result['segments'])} segments.")

    # Stage 3: Topic Segmentation
    _stage_header(3, total_stages, "Segmenting into topics")
    try:
        topics_result = segment_transcript(transcript_path)
    except TopicSegmentationError as e:
        raise PipelineError(f"Topic segmentation failed: {e}")
    topics_path = topics_result["output_path"]
    results["topics_path"] = topics_path
    print(f"Done. Found {len(topics_result['topics'])} topics.")

    # Stage 4: Arc-based Clip Selection
    _stage_header(4, total_stages, "Selecting best clip per topic")
    try:
        clips_result = select_clips(transcript_path, topics_path)
    except ArcSelectionError as e:
        raise PipelineError(f"Clip selection failed: {e}")
    clips_v2_path = clips_result["output_path"]
    results["clips_v2_path"] = clips_v2_path
    print(f"Done. Selected {len(clips_result['clips'])} candidate clips.")

    # Stage 5: Relative Ranking
    _stage_header(5, total_stages, "Ranking clips relative to each other")
    try:
        ranked_result = rank_clips(clips_v2_path)
    except RankingError as e:
        raise PipelineError(f"Ranking failed: {e}")
    ranked_path = ranked_result["output_path"]
    results["ranked_path"] = ranked_path
    print(f"Done. Top clip: '{ranked_result['clips'][0]['topic']}' "
          f"(rank_score={ranked_result['clips'][0]['rank_score']})")

    # Stage 6: Rendering (crop + captions)
    _stage_header(6, total_stages, f"Rendering top {top_n} clips (crop + captions)")
    try:
        render_result = render_clips(ranked_path, youtube_url, transcript_path, top_n=top_n)
    except RenderError as e:
        raise PipelineError(f"Rendering failed: {e}")
    results["rendered_clips"] = render_result["rendered_clips"]
    results["output_dir"] = render_result["output_dir"]

    elapsed = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    print(f"\nFinal clips saved to: {results['output_dir']}\n")
    for c in results["rendered_clips"]:
        print(f"  rank {c['rank']} | score={c.get('rank_score')} | "
              f"'{c['topic']}' | {c['output_path']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClipMind AI — full pipeline")
    parser.add_argument("youtube_url", help="YouTube video URL to process")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top clips to render (default: 5)")
    args = parser.parse_args()

    try:
        run_pipeline(args.youtube_url, top_n=args.top_n)
    except PipelineError as e:
        print(f"\nPIPELINE FAILED: {e}")
        sys.exit(1)