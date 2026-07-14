"""
speaker_detector.py
Detects which face is actively speaking at each point in a video clip,
using MediaPipe Face Mesh to track lip movement over time. Outputs a
timeline of active-speaker screen positions, to later drive dynamic
vertical cropping (instead of naive center-crop).

Includes gap-filling: short stretches with no confident face detection
(e.g., a speaker briefly looking down) carry forward the last known
position instead of leaving a blank gap, since a real editor wouldn't
yank the crop away for a brief head-tilt.
"""

import sys
import json
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp


class SpeakerDetectionError(Exception):
    pass


SAMPLE_FPS = 5
UPPER_LIP_IDX = 13
LOWER_LIP_IDX = 14
FACE_MATCH_MAX_DIST = 0.15
WINDOW_SECONDS = 1.0
MAX_GAP_TO_FILL_SECONDS = 3.0  # gaps shorter than this get filled with last known position


def _face_bbox_and_center(landmarks, frame_w, frame_h):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return {
        "bbox": (x_min, y_min, x_max, y_max),
        "center": (center_x, center_y),
    }


def _mouth_openness(landmarks):
    upper = landmarks[UPPER_LIP_IDX]
    lower = landmarks[LOWER_LIP_IDX]
    return abs(upper.y - lower.y)


def _match_face_to_tracks(center, tracks, frame_idx):
    best_id, best_dist = None, FACE_MATCH_MAX_DIST
    for track_id, track in tracks.items():
        last_center = track["last_center"]
        dist = ((center[0] - last_center[0]) ** 2 + (center[1] - last_center[1]) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_id = track_id
    return best_id


def _fill_gaps(raw_windows: list, duration: float) -> list:
    """
    raw_windows: list of dicts (one per WINDOW_SECONDS slice, in order),
    each either a real detection dict or None if nothing was detected.

    Returns a cleaned list where short None-gaps are replaced by carrying
    forward the last known detection; longer gaps remain as explicit
    'no_detection' entries.
    """
    filled = []
    last_known = None
    gap_start_idx = None

    for i, window in enumerate(raw_windows):
        if window is not None:
            filled.append(dict(window))
            last_known = window
            gap_start_idx = None
        else:
            if gap_start_idx is None:
                gap_start_idx = i
            gap_length = (i - gap_start_idx + 1) * WINDOW_SECONDS

            if last_known is not None and gap_length <= MAX_GAP_TO_FILL_SECONDS:
                carried = dict(last_known)
                carried["start"] = round(i * WINDOW_SECONDS, 2)
                carried["end"] = round(min((i + 1) * WINDOW_SECONDS, duration), 2)
                carried["carried_forward"] = True
                filled.append(carried)
            else:
                filled.append({
                    "start": round(i * WINDOW_SECONDS, 2),
                    "end": round(min((i + 1) * WINDOW_SECONDS, duration), 2),
                    "active_face_id": None,
                    "face_center_x": None,
                    "face_center_y": None,
                    "movement_score": 0.0,
                    "no_detection": True,
                })

    return filled


def analyze_video(video_path: str, debug: bool = False) -> dict:
    """
    Analyzes a video clip and produces a timeline of which detected face
    (by screen position) is the active speaker at each time window.

    Returns dict with keys: fps, duration, num_faces_detected, timeline, output_path
    """
    path = Path(video_path)
    if not path.exists():
        raise SpeakerDetectionError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SpeakerDetectionError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / native_fps if native_fps else 0

    frame_interval = max(int(round(native_fps / SAMPLE_FPS)), 1)

    mp_face_mesh = mp.solutions.face_mesh
    tracks = {}
    next_track_id = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=4,
        refine_landmarks=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as face_mesh:

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / native_fps
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks = face_landmarks.landmark
                        info = _face_bbox_and_center(landmarks, frame_w, frame_h)
                        mouth_open = _mouth_openness(landmarks)

                        matched_id = _match_face_to_tracks(info["center"], tracks, frame_idx)
                        if matched_id is None:
                            matched_id = next_track_id
                            tracks[matched_id] = {"last_center": info["center"], "history": []}
                            next_track_id += 1

                        tracks[matched_id]["last_center"] = info["center"]
                        tracks[matched_id]["history"].append({
                            "time": round(timestamp, 2),
                            "mouth_open": mouth_open,
                            "center": info["center"],
                        })

            frame_idx += 1

    cap.release()

    if not tracks:
        raise SpeakerDetectionError("No faces detected in this video clip.")

    if debug:
        print("\n[DEBUG] Track summary:", flush=True)
        for track_id, track in tracks.items():
            history = track["history"]
            if history:
                print(f"  face_id={track_id}: {len(history)} detections, "
                      f"time range {history[0]['time']:.1f}s - {history[-1]['time']:.1f}s", flush=True)
            else:
                print(f"  face_id={track_id}: 0 detections", flush=True)
        print("", flush=True)

    # Build raw per-window results (None where nothing was detected)
    raw_windows = []
    num_windows = int(np.ceil(duration / WINDOW_SECONDS))

    for w in range(num_windows):
        window_start = w * WINDOW_SECONDS
        window_end = window_start + WINDOW_SECONDS

        window_scores = []
        active_id, active_score, active_center = None, -1, None

        for track_id, track in tracks.items():
            window_points = [h for h in track["history"] if window_start <= h["time"] < window_end]
            if len(window_points) < 2:
                continue
            mouth_values = [h["mouth_open"] for h in window_points]
            movement_score = float(np.std(mouth_values))
            window_scores.append((track_id, movement_score))

            if movement_score > active_score:
                active_score = movement_score
                active_id = track_id
                avg_x = sum(h["center"][0] for h in window_points) / len(window_points)
                avg_y = sum(h["center"][1] for h in window_points) / len(window_points)
                active_center = (round(avg_x, 3), round(avg_y, 3))

        if debug and window_scores:
            scores_str = ", ".join(f"face_{tid}={score:.4f}" for tid, score in window_scores)
            print(f"[DEBUG] window {window_start:.1f}-{window_end:.1f}s -> {scores_str}", flush=True)

        if active_id is not None:
            raw_windows.append({
                "start": round(window_start, 2),
                "end": round(min(window_end, duration), 2),
                "active_face_id": active_id,
                "face_center_x": active_center[0],
                "face_center_y": active_center[1],
                "movement_score": round(active_score, 4),
                "carried_forward": False,
            })
        else:
            raw_windows.append(None)

    filled_windows = _fill_gaps(raw_windows, duration)

    # Merge consecutive windows with the same active speaker into single segments
    merged_timeline = []
    for entry in filled_windows:
        same_speaker = (
            merged_timeline
            and merged_timeline[-1].get("active_face_id") == entry.get("active_face_id")
            and merged_timeline[-1].get("no_detection", False) == entry.get("no_detection", False)
        )
        if same_speaker:
            merged_timeline[-1]["end"] = entry["end"]
        else:
            merged_timeline.append(dict(entry))

    output_path = path.parent / f"{path.stem}_speaker_timeline.json"
    result = {
        "video_path": str(path),
        "fps": native_fps,
        "duration": round(duration, 2),
        "num_faces_detected": len(tracks),
        "timeline": merged_timeline,
        "output_path": str(output_path),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/rendering/speaker_detector.py <path_to_video_clip.mp4> [--debug]")
        sys.exit(1)

    debug_mode = "--debug" in sys.argv

    try:
        result = analyze_video(sys.argv[1], debug=debug_mode)
        print(f"\nDetected {result['num_faces_detected']} distinct face(s) across the clip.")
        print(f"Video duration: {result['duration']}s")
        print(f"Saved timeline to: {result['output_path']}\n")
        print("Active speaker timeline:\n")
        for seg in result["timeline"]:
            if seg.get("no_detection"):
                print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] NO DETECTION (gap too long to fill)")
            else:
                carried_note = " (carried forward)" if seg.get("carried_forward") else ""
                print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] face_id={seg['active_face_id']} "
                      f"position=({seg['face_center_x']:.2f}, {seg['face_center_y']:.2f}) "
                      f"movement_score={seg['movement_score']}{carried_note}")
    except SpeakerDetectionError as e:
        print(f"\nSpeaker detection failed: {e}")
        sys.exit(1)