# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A

# pylint: disable=no-member
import cv2
import hydra
import numpy as np
import psycopg2
import ray
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from media_flow.utils.fault_tolerance import RAY_TASK_CONFIG, process_ray_results

# Database Connection Details (pulled from environment)
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")


def setup_logger():
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


def set_seed(seed: int):
    """Sets the global seeds for Python and Numpy."""
    random.seed(seed)
    np.random.seed(seed)


def build_augmenter(params: Dict[str, Any], seed: int = 42) -> A.Compose:
    """Builds a deterministic augmentation pipeline using a fixed seed."""
    return A.Compose(
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
        seed=seed,
    )


def create_video_writer(
    output_path: Path, fps: float, width: int, height: int
) -> Tuple[Optional[cv2.VideoWriter], Optional[Exception]]:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    try:
        return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height)), None
    except Exception as e:
        return None, e


def augment_frame(augmenter: A.Compose, frame_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    augmented = augmenter(image=image_rgb)["image"]
    return cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)


def build_success_record(
    video_id: int, output_path: Path, augmentation_params: str
) -> Dict[str, Any]:
    return {
        "video_id": video_id,
        "augmented_path": str(output_path),
        "augmentation_type": "standard_dropout",
        "parameters_used": augmentation_params,
        "status": "READY",
    }


def ensure_db_credentials() -> None:
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        logger.error("Missing required DB environment variables.")
        sys.exit(1)


def fetch_videos_to_augment() -> List[Tuple[int, str, str]]:
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cursor = conn.cursor()
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
        rows = cursor.fetchall()
        conn.close()
        return rows
    except psycopg2.Error as e:
        logger.error(f"DB Error fetching videos to augment: {e}")
        if conn:
            conn.close()
        sys.exit(1)


def prepare_output_path(output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = Path(filename).stem
    return output_dir / f"{basename}.mp4"


def submit_augment_tasks(
    videos: List[Tuple[int, str, str]],
    output_dir: Path,
    augmentation_settings: str,
    seed: int,
) -> Tuple[List[Any], Dict[Any, int]]:
    futures: List[Any] = []
    future_to_id: Dict[Any, int] = {}
    for video_id, video_path, filename in videos:
        output_path = prepare_output_path(output_dir, filename)
        future = process_video_task.remote(
            video_id=video_id,
            video_path=video_path,
            output_path=str(output_path),
            augmentation_params=augmentation_settings,
            seed=seed,
        )
        futures.append(future)
        future_to_id[future] = video_id
    return futures, future_to_id


def handle_results(futures: List[Any], future_to_id: Dict[Any, int]) -> None:
    for result_record in process_ray_results(futures, future_to_id, "augment"):
        insert_augmentation_record(result_record)


@ray.remote(**RAY_TASK_CONFIG)
def process_video_task(
    video_id: int,
    video_path: str,
    output_path: str,
    augmentation_params: str,
    seed: int,
) -> Optional[Dict]:
    """Reads, augments, and writes video deterministically on the worker node."""
    # Ensure global seeds are set for this specific worker process
    set_seed(seed)

    v_path = Path(video_path)
    o_path = Path(output_path)

    params = json.loads(augmentation_params)
    augmenter = build_augmenter(params, seed=seed)

    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open {v_path}")
        return None

    width, height = 480, 480
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer, err = create_video_writer(o_path, fps, width, height)
    if err or writer is None:
        print(f"ERROR: Failed to initialize video writer for {o_path}: {err}")
        cap.release()
        return {
            "video_id": video_id,
            "status": "ERROR",
            "error_message": f"Writer failed: {err}",
        }

    print(f"Processing {v_path.name} (Seed: {seed})...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_bgr = augment_frame(augmenter, frame)
        writer.write(image_bgr)

    cap.release()
    writer.release()
    print(f"Finished augmentation: {o_path}")

    return build_success_record(video_id, o_path, augmentation_params)


def insert_augmentation_record(record: Dict):
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
    except psycopg2.Error as e:
        logger.error(f"DB Error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


def augment_pipeline(cfg: DictConfig):
    setup_logger()
    ensure_db_credentials()

    videos_to_augment = fetch_videos_to_augment()
    if not videos_to_augment:
        logger.info("No new videos require augmentation. Exiting.")
        return

    initialize_ray()

    output_dir = Path(cfg.augment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a specific seed from config for reproducibility
    seed = cfg.augment.get("seed", 42)

    augmentation_settings = json.dumps(
        OmegaConf.to_container(cfg.augment.settings, resolve=True)
    )

    logger.info(
        f"Starting deterministic pipeline (Seed: {seed}) for {len(videos_to_augment)} videos."
    )

    futures, future_to_id = submit_augment_tasks(
        videos_to_augment, output_dir, augmentation_settings, seed
    )
    handle_results(futures, future_to_id)

    ray.shutdown()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    augment_pipeline(cfg)


if __name__ == "__main__":
    main()
