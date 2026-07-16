"""
extract_pose_json.py — Run MediaPipe Pose on a video, save landmarks as JSON.

Output JSON structure (matches Unnoticed Dance's PoseFrame internal format):
{
  "fps": 30.0,
  "num_frames": 300,
  "duration_ms": 10000.0,
  "frames": [
    { "t": 0.0,     "landmarks": [ {"x":0.5,"y":0.5,"z":0.0,"visibility":0.99}, ... 33 items ] },
    { "t": 33.33,   "landmarks": [ ... ] },
    ...
  ]
}

Usage:
    conda activate momask
    python extract_pose_json.py \\
      --video ~/Downloads/momask-codes-main/input_videos/webcam_20260708_202958.mp4 \\
      --out   pose_data.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out",   required=True)
    ap.add_argument("--model-complexity", type=int, default=1, help="0/1/2 — higher = more accurate, slower")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video).expanduser()
    out_path   = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise SystemExit(f"[fatal] video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"[fatal] cv2 could not open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[pose] video: {video_path}  fps={fps:.2f}  frames={total}")

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)

        landmarks = None
        if results.pose_landmarks:
            landmarks = [
                {"x": float(lm.x), "y": float(lm.y),
                 "z": float(lm.z), "visibility": float(lm.visibility)}
                for lm in results.pose_landmarks.landmark
            ]

        frames.append({
            "t": idx * 1000.0 / fps,   # ms
            "landmarks": landmarks,
        })
        idx += 1
        if idx % 30 == 0:
            print(f"[pose]   processed {idx}/{total} frames", end="\r")

    cap.release()
    pose.close()

    detected = sum(1 for f in frames if f["landmarks"] is not None)
    duration_ms = frames[-1]["t"] if frames else 0.0
    print(f"\n[pose] done. detected pose in {detected}/{len(frames)} frames")

    out_path.write_text(json.dumps({
        "fps": float(fps),
        "num_frames": len(frames),
        "duration_ms": float(duration_ms),
        "frames": frames,
    }))
    print(f"[pose] wrote {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
