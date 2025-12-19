# scripts/preprocess.py

import os
import cv2
from pathlib import Path
import argparse

def extract_frames(video_path, output_dir, resize=(640, 480), skip_frames=1):
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % skip_frames == 0:
            if resize:
                frame = cv2.resize(frame, resize)
            frame_path = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1

        count += 1

    cap.release()
    print(f"Extracted {saved} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess video data for YOLO training")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="data/annotated/frames", help="Directory to save frames")
    parser.add_argument("--resize", type=int, nargs=2, default=(640, 480), help="Resize dimensions (width height)")
    parser.add_argument("--skip", type=int, default=1, help="Skip every N frames")
    args = parser.parse_args()

    extract_frames(args.video, args.output, tuple(args.resize), args.skip)
