import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional
import psycopg2

import albumentations as A

# pylint: disable=no-member
import cv2
import ray
from loguru import logger

# Database Connection Details (pulled from environment)
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")


def setup_logger():
    # ... (log setup remains the same)
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


# --- The Augmented Worker Function (unchanged core logic) ---
@ray.remote
def augment_batch(batch_frames, augmentation_params):  # Accepts parameters now
    # Define pipeline inside the worker to ensure serialization
    params = json.loads(augmentation_params)

    augmenter = A.Compose(
        [
            A.CoarseDropout(**params.get("CoarseDropout", {"p": 0.8})),
            A.Rotate(**params.get("Rotate", {"limit": 10, "p": 0.7})),
            A.RandomBrightnessContrast(
                **params.get("RandomBrightnessContrast", {"p": 0.5})
            ),
            A.GaussNoise(**params.get("GaussNoise", {"p": 0.5})),
            A.Resize(height=480, width=480, interpolation=cv2.INTER_LINEAR, p=1.0),
        ],
        p=1.0,
    )

    results = []
    for frame_data in batch_frames:
        original = frame_data["image"]
        image_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        augmented = augmenter(image=image_rgb)["image"]
        image_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        results.append(
            {"frame_id": frame_data["frame_id"], "augmented_image": image_bgr}
        )
    return results


def process_video_streaming(
    video_id: int,
    video_path: str,
    output_path: str,
    augmentation_params: str,
    batch_size=60,
) -> Optional[Dict]:
    """
    Reads, augments, and writes video in chunks. Returns augmentation metadata.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open {video_path}")
        return None

    # Get Video Properties
    width = 480  # Matches your resize target
    height = 480  # Matches your resize target
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Total frames is often unreliable, use a counter instead
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    try:
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    except Exception as e:
        logger.error(f"Failed to initialize video writer for {output_path}: {e}")
        cap.release()
        return {
            "video_id": video_id,
            "status": "ERROR",
            "error_message": f"Writer failed: {e}",
        }

    logger.info(
        f"Processing {os.path.basename(video_path)} in batches of {batch_size}..."
    )

    current_batch = []
    frame_id = 0
    futures = []

    while True:
        ret, frame = cap.read()

        if ret:
            current_batch.append({"frame_id": frame_id, "image": frame})
            frame_id += 1

        if len(current_batch) == batch_size or (not ret and current_batch):
            # Send batch to Ray worker, passing augmentation parameters
            future = augment_batch.remote(current_batch, augmentation_params)
            futures.append(future)
            current_batch = []

            if len(futures) > 4:  # Limit active futures
                ready_ids, futures = ray.wait(futures, num_returns=1)
                result_batch = ray.get(ready_ids[0])

                # Write immediately to disk
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
    logger.success(f"Finished augmentation: {output_path}")

    # Return success record to be written to DB
    return {
        "video_id": video_id,
        "augmented_path": output_path,
        "augmentation_type": "standard_dropout",
        "parameters_used": augmentation_params,
        "status": "READY",
    }


def insert_augmentation_record(record: Dict):
    """Inserts a single augmentation record into the augmented_videos table."""
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cursor = conn.cursor()

        insert_statement = """
            INSERT INTO augmented_videos (video_id, augmented_path, augmentation_type, parameters_used, status, timestamp_processed)
            VALUES (%s, %s, %s, %s::jsonb, %s, CURRENT_TIMESTAMP);
        """

        cursor.execute(
            insert_statement,
            (
                record["video_id"],
                record["augmented_path"],
                record["augmentation_type"],
                record["parameters_used"],
                record["status"],
            ),
        )
        conn.commit()
        logger.info(f"Augmentation record inserted for video ID {record['video_id']}.")
    except psycopg2.Error as e:
        logger.error(f"DB Error inserting augmentation record: {e}")
        if conn:
            conn.rollback()
        # Do not raise here, as one video failing shouldn't crash the entire task
    finally:
        if conn:
            conn.close()


def augment_pipeline(output_dir: str):
    setup_logger()

    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        logger.error("Missing required DB environment variables.")
        sys.exit(1)

    # 1. Query DB for videos that passed filtering and haven't been augmented
    conn = None
    videos_to_augment = []

    # Define augmentation parameters for this run (passed as JSON string)
    augmentation_settings = json.dumps(
        {
            "CoarseDropout": {
                "max_holes": 1,
                "max_height": 64,
                "max_width": 64,
                "p": 0.8,
            },
            "Rotate": {"limit": 10, "p": 0.7},
            "RandomBrightnessContrast": {"p": 0.5},
            "GaussNoise": {"p": 0.5},
        }
    )

    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cursor = conn.cursor()

        # Select videos marked as good quality and not yet augmented
        query = """
            SELECT vm.video_id, vm.original_path, vm.filename
            FROM video_metadata vm
            WHERE vm.is_quality_video = TRUE
              AND NOT EXISTS (
                  SELECT 1 FROM augmented_videos av 
                  WHERE av.video_id = vm.video_id AND av.augmentation_type = 'standard_dropout'
              );
        """
        cursor.execute(query)
        videos_to_augment = cursor.fetchall()
        conn.close()

    except psycopg2.Error as e:
        logger.error(f"DB Error fetching videos to augment: {e}")
        if conn:
            conn.close()
        sys.exit(1)

    if not videos_to_augment:
        logger.info("No new videos require augmentation. Exiting.")
        return

    initialize_ray()
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Starting augmentation on {len(videos_to_augment)} videos.")
    for video_id, video_path, filename in videos_to_augment:
        basename = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{basename}_aug.mp4")

        # Process video and get result metadata
        result_record = process_video_streaming(
            video_id=video_id,
            video_path=video_path,
            output_path=output_path,
            augmentation_params=augmentation_settings,
        )

        # Record the result in the database
        if result_record:
            insert_augmentation_record(result_record)

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    augment_pipeline(args.output_dir)
