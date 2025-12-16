# pylint: disable=missing-module-docstring, missing-function-docstring
import glob
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

import hydra
import psycopg2
import ray
from loguru import logger
from omegaconf import DictConfig

# --- Configuration & Constants ---
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
BATCH_SIZE = 30


def _setup_logger():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )


def _get_db_connection():
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        logger.error("Missing required DB environment variables.")
        sys.exit(1)
    return psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )


def _extract_single(video_path: str) -> Optional[Dict[str, Any]]:
    """Runs ffprobe on a single file. Executed inside the Ray worker."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration,r_frame_rate,width,height,codec_name,pix_fmt",
        "-of",
        "json",
        video_path,
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=10
        )
        data = json.loads(result.stdout)
        if not data.get("streams"):
            return None

        stream = data["streams"][0]
        fps_str = stream.get("r_frame_rate", "0/1")
        num, den = map(int, fps_str.split("/")) if "/" in fps_str else (0, 1)

        return {
            "original_path": video_path,
            "duration_sec": round(float(stream.get("duration", 0)), 3),
            "frame_rate": round(num / den, 3) if den != 0 else 0,
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "codec_name": stream.get("codec_name"),
            "color_space": stream.get("pix_fmt"),
        }
    except Exception as e:
        logger.warning(f"FFprobe failed for {video_path}: {e}")
        return None


@ray.remote
def _extract_metadata_batch(video_paths: List[str]) -> List[Optional[Dict[str, Any]]]:
    """Ray remote task to process a batch of videos."""
    return [_extract_single(p) for p in video_paths]


def _discover_videos(
    input_dir: str, extensions: List[str], max_videos: Optional[int] = None
) -> List[str]:
    """Scans the input directory for video files based on extensions."""
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))

    video_files.sort()

    # Apply Limit
    if max_videos and 0 < max_videos < len(video_files):
        logger.warning(f"Limiting scan to first {max_videos} videos.")
        video_files = video_files[:max_videos]

    logger.info(f"Discovered {len(video_files)} video files on disk.")
    return video_files


def _register_videos(video_files: List[str]):
    """Inserts found videos into the DB. Skips duplicates (Idempotent)."""
    if not video_files:
        return

    conn = _get_db_connection()
    try:
        cursor = conn.cursor()
        data = [(os.path.abspath(p), os.path.basename(p)) for p in video_files]

        insert_stmt = """
            INSERT INTO video_metadata (original_path, filename) 
            VALUES (%s, %s)
            ON CONFLICT (original_path) DO NOTHING;
        """
        cursor.executemany(insert_stmt, data)
        conn.commit()
        logger.info("Registration phase complete.")
    except Exception as e:
        logger.error(f"Error registering videos: {e}")
        raise
    finally:
        conn.close()


def _get_pending_videos() -> Dict[str, int]:
    """
    Identifies videos that need processing.
    Logic: Exists in DB but 'duration_sec' is NULL.
    Returns: Dict {filepath: video_id}
    """
    conn = _get_db_connection()
    try:
        cursor = conn.cursor()
        query = "SELECT video_id, original_path FROM video_metadata WHERE duration_sec IS NULL;"
        cursor.execute(query)
        # Create map: path -> id
        return {path: vid_id for vid_id, path in cursor.fetchall()}
    finally:
        conn.close()


def _run_distributed_extraction(paths: List[str], batch_size: int) -> List[Dict]:
    """Initializes Ray and runs extraction on the provided paths."""
    if not paths:
        return []

    if not ray.is_initialized():
        ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)
        print("Connected to Ray cluster.")

    batches = [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]
    logger.info(f"Submitting {len(batches)} batches to Ray for {len(paths)} videos...")

    futures = [_extract_metadata_batch.remote(batch) for batch in batches]
    results_nested = ray.get(futures)

    ray.shutdown()

    # Flatten list and remove Nones
    return [item for sublist in results_nested for item in sublist if item]


def _populate_db(metadata_list: List[Dict], id_map: Dict[str, int]):
    """Updates the database with the extracted metadata."""
    if not metadata_list:
        logger.warning("No metadata extracted to update.")
        return

    conn = _get_db_connection()
    try:
        cursor = conn.cursor()
        updates = []

        for meta in metadata_list:
            path = meta["original_path"]
            vid_id = id_map.get(path)
            if vid_id:
                updates.append(
                    (
                        meta["duration_sec"],
                        meta["frame_rate"],
                        meta["width"],
                        meta["height"],
                        meta["codec_name"],
                        meta["color_space"],
                        vid_id,
                    )
                )

        update_stmt = """
            UPDATE video_metadata SET
                duration_sec = %s, frame_rate = %s, width = %s, height = %s,
                codec_name = %s, color_space = %s, scan_date = CURRENT_TIMESTAMP
            WHERE video_id = %s;
        """
        cursor.executemany(update_stmt, updates)
        conn.commit()
        logger.success(f"Successfully updated metadata for {len(updates)} videos.")
    except Exception as e:
        logger.error(f"Error populating DB: {e}")
    finally:
        conn.close()


def ingest_pipeline(cfg: DictConfig):
    # 1. Discover files on disk
    input_dir = cfg.ingest.input_dir
    extensions = cfg.ingest.video_extensions
    max_videos = cfg.ingest.get("max_videos")
    files = _discover_videos(
        input_dir=input_dir, extensions=extensions, max_videos=max_videos
    )

    if not files:
        logger.warning("No files found. Exiting.")
        return

    # 2. Register them in DB (if new)
    _register_videos(files)

    # 3. Find which ones are actually missing metadata (Pending)
    # This automatically filters out videos that were already processed in previous runs.
    pending_map = _get_pending_videos()

    if not pending_map:
        logger.info("All discovered videos already have metadata. Nothing to do.")
        return

    logger.info(f"Found {len(pending_map)} new/pending videos requiring extraction.")

    # 4. Extract Metadata (Distributed)
    metadata = _run_distributed_extraction(
        list(pending_map.keys()), cfg.ingest.metadata_extraction_batch_size
    )

    # 5. Populate DB
    _populate_db(metadata, pending_map)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    _setup_logger()
    ingest_pipeline(cfg)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
