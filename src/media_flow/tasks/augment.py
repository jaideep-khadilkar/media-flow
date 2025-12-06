import os
import sys
import json
import argparse
from loguru import logger
import numpy as np
import cv2
import albumentations as A
import ray


def setup_logger():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )


def initialize_ray():
    if ray.is_initialized():
        ray.shutdown()
    # Limit object store memory to prevent Ray from eating all RAM
    ray.init(
        log_to_driver=True,
        include_dashboard=False,
        object_store_memory=2 * 1024 * 1024 * 1024,
    )


# --- The Augmented Worker Function ---
# We make this a remote function so Ray runs it in parallel
@ray.remote
def augment_batch(batch_frames):
    """
    Receives a list of (frame_id, image) tuples.
    Returns a list of (frame_id, augmented_image) tuples.
    """
    # Define pipeline inside the worker to ensure serialization
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

    results = []
    for frame_data in batch_frames:
        original = frame_data["image"]
        # Albumentations expects RGB
        image_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        augmented = augmenter(image=image_rgb)["image"]
        # Convert back to BGR for OpenCV video writing
        image_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        results.append(
            {"frame_id": frame_data["frame_id"], "augmented_image": image_bgr}
        )
    return results


def process_video_streaming(video_path: str, output_path: str, batch_size=60):
    """
    Reads, augments, and writes video in chunks to save memory.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open {video_path}")
        return

    # Get Video Properties
    width = 480  # Matches your resize target
    height = 480  # Matches your resize target
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    logger.info(
        f"Processing {os.path.basename(video_path)} ({total_frames} frames) in batches of {batch_size}..."
    )

    current_batch = []
    frame_id = 0
    futures = []

    while True:
        ret, frame = cap.read()

        if ret:
            # Add frame to current batch
            current_batch.append({"frame_id": frame_id, "image": frame})
            frame_id += 1

        # If batch is full or video ended
        if len(current_batch) == batch_size or (not ret and current_batch):
            # Send batch to Ray worker
            future = augment_batch.remote(current_batch)
            futures.append(future)
            current_batch = []  # clear memory

            # Limit the number of active futures to prevent OOM
            # This ensures we don't queue up the whole video in Ray's object store
            if len(futures) > 4:
                ready_ids, futures = ray.wait(futures, num_returns=1)
                result_batch = ray.get(ready_ids[0])

                # Write immediately to disk
                # Sort just in case, though Ray usually preserves order if handled sequentially like this
                result_batch.sort(key=lambda x: x["frame_id"])
                for res in result_batch:
                    out.write(res["augmented_image"])

        if not ret:
            break

    # Process remaining futures
    for future in futures:
        result_batch = ray.get(future)
        result_batch.sort(key=lambda x: x["frame_id"])
        for res in result_batch:
            out.write(res["augmented_image"])

    cap.release()
    out.release()
    logger.success(f"Finished: {output_path}")


def process_pipeline(input_json: str, output_dir: str):
    setup_logger()

    if not os.path.exists(input_json):
        logger.error(f"Input JSON not found: {input_json}")
        sys.exit(1)

    with open(input_json, "r") as f:
        video_paths = json.load(f)

    initialize_ray()
    os.makedirs(output_dir, exist_ok=True)

    for video_path in video_paths:
        basename = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{basename}_aug.mp4")

        # Use the streaming function
        process_video_streaming(video_path, output_path)

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    process_pipeline(args.input_json, args.output_dir)
