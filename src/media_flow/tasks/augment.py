import json
import os
import sys
from typing import Dict, Optional

import albumentations as A

# pylint: disable=no-member
import cv2
import hydra
import psycopg2
import ray
from loguru import logger
from omegaconf import DictConfig

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
    ray.init(
        address="ray://ray-head:10001",
        ignore_reinit_error=True,
    )
    print("Connected to existing Ray cluster.")


# --- Refactored: Entire video processing moved to Ray Worker ---
# MODIFIED: Combined IO and Processing into one remote task to eliminate data transfer overhead
@ray.remote
def process_video_task(
    video_id: int,
    video_path: str,
    output_path: str,
    augmentation_params: str,
) -> Optional[Dict]:
    """
    Reads, augments, and writes video entirely on the worker node.
    This avoids expensive serialization of video frames between driver and worker.
    """
    # Define pipeline inside the worker to ensure serialization
    params = json.loads(augmentation_params)

    # Initialize Augmentation Pipeline once per video (more efficient)
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

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Using print/logging inside remote function logs to driver stderr by default
        print(f"ERROR: Cannot open {video_path}")
        return None

    # Get Video Properties
    width = 480  # Matches your resize target
    height = 480  # Matches your resize target
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None
    try:
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    except Exception as e:
        print(f"ERROR: Failed to initialize video writer for {output_path}: {e}")
        cap.release()
        return {
            "video_id": video_id,
            "status": "ERROR",
            "error_message": f"Writer failed: {e}",
        }

    print(f"Processing {os.path.basename(video_path)} on worker...")

    # Loop through frames directly on the worker
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB, Augment, Convert back to BGR
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        augmented = augmenter(image=image_rgb)["image"]
        image_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        out.write(image_bgr)

    cap.release()
    out.release()
    print(f"Finished augmentation: {output_path}")

    # Return success record to be written to DB by the driver
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


def augment_pipeline(cfg: DictConfig):

    output_dir = cfg.augment.output_dir
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

    # MODIFIED: Batch submission of tasks to Ray
    futures = []
    for video_id, video_path, filename in videos_to_augment:
        basename = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{basename}.mp4")

        # Submit task to Ray (Non-blocking)
        # This allows multiple videos to be processed in parallel on different workers
        future = process_video_task.remote(
            video_id=video_id,
            video_path=video_path,
            output_path=output_path,
            augmentation_params=augmentation_settings,
        )
        futures.append(future)

    # Wait for all tasks to complete and retrieve results
    logger.info("Waiting for Ray tasks to complete...")
    results = ray.get(futures)

    # Record the result in the database
    for result_record in results:
        if result_record:
            insert_augmentation_record(result_record)

    ray.shutdown()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    augment_pipeline(cfg)


if __name__ == "__main__":
    main()
