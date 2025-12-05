import os
import sys
import json
import argparse
from loguru import logger
from tqdm import tqdm

# Core Data Science Libraries
import numpy as np
import cv2
import albumentations as A

# Parallel Computing
import ray
import ray.data
from ray.exceptions import RaySystemError

# --- Configuration ---
# (Directories are now handled via arguments, but we keep defaults just in case)

# --- Utility Functions ---


def setup_logger():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )


def initialize_ray():
    """Initializes the Ray cluster for distributed computing."""
    try:
        if ray.is_initialized():
            ray.shutdown()

        info = ray.init(
            log_to_driver=True, include_dashboard=False
        )  # Dashboard off for batch jobs usually
        logger.info(
            f"Ray initialized successfully. CPUs: {ray.available_resources().get('CPU', 0)}"
        )

    except RaySystemError as e:
        logger.error(f"Failed to initialize Ray: {e}")
        sys.exit(1)


def video_to_frames(video_path: str) -> list[dict]:
    if not os.path.exists(video_path):
        logger.warning(f"Input video not found at: {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file {video_path}")
        return []

    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Note: TQDM inside the worker might clutter logs if not careful, keeping it minimal
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append({"frame_id": i, "image": frame, "fps": fps})
    cap.release()
    return frames


def write_frames_to_video(frames_data: list, output_path: str):
    if not frames_data:
        return

    sorted_frames = sorted(frames_data, key=lambda x: x["frame_id"])
    frame_example = sorted_frames[0]["augmented_image"]
    height, width, _ = frame_example.shape
    fps = sorted_frames[0].get("fps", 30.0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_data in sorted_frames:
        out.write(frame_data["augmented_image"])

    out.release()
    logger.success(f"Saved: {output_path}")


# --- Pipeline Components ---


def apply_augmentation(frame_data: dict) -> dict:
    # Define augmentation pipeline
    augmenter = A.Compose(
        [
            A.CoarseDropout(
                max_holes=1,
                max_height=64,
                max_width=64,
                min_holes=1,
                min_height=16,
                min_width=16,
                fill_value=0,
                p=0.8,
            ),
            A.Rotate(
                limit=10,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.7,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.Resize(height=480, width=480, interpolation=cv2.INTER_LINEAR, p=1.0),
        ],
        p=1.0,
    )

    original_image = frame_data["image"]
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    augmented_image_rgb = augmenter(image=image_rgb)["image"]
    augmented_image_bgr = cv2.cvtColor(augmented_image_rgb, cv2.COLOR_RGB2BGR)

    return {
        "frame_id": frame_data["frame_id"],
        "augmented_image": augmented_image_bgr,
        "fps": frame_data["fps"],
    }


def process_pipeline(input_json: str, output_dir: str):
    setup_logger()

    # 1. Load List of Videos
    if not os.path.exists(input_json):
        logger.error(f"Input JSON list not found: {input_json}")
        sys.exit(1)

    with open(input_json, "r") as f:
        video_paths = json.load(f)

    if not video_paths:
        logger.warning("Video list is empty. Nothing to process.")
        return

    # 2. Initialize Ray
    initialize_ray()
    os.makedirs(output_dir, exist_ok=True)

    # 3. Process
    logger.info(f"Starting distributed augmentation for {len(video_paths)} videos...")

    for video_path in video_paths:
        logger.info(f"Processing: {os.path.basename(video_path)}")

        # Ingestion
        frames_list = video_to_frames(video_path)
        if not frames_list:
            continue

        # Ray Processing
        # We process one video's frames in parallel, then move to the next video
        # (This prevents memory overflow if we tried to load ALL videos into Ray at once)
        ds = ray.data.from_items(frames_list)
        augmented_ds = ds.map(apply_augmentation)
        results = augmented_ds.take_all()

        # Write Output
        basename = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{basename}_aug.mp4")
        write_frames_to_video(results, output_path)

    ray.shutdown()
    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ray Augmentation pipeline.")
    parser.add_argument(
        "--input_json", required=True, help="Path to JSON list of videos to process"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save augmented videos"
    )

    args = parser.parse_args()

    process_pipeline(args.input_json, args.output_dir)
