import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional
import ray
import psycopg2
from loguru import logger

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
    # Only initializes Ray locally within the container if not connected
    try:
        ray.init(address="auto", ignore_reinit_error=True)
    except Exception:
        ray.init(local_mode=False)


@ray.remote
def run_ffprobe_extraction(video_path: str) -> Optional[Dict[str, Any]]:
    # ... (run_ffprobe_extraction logic remains the same, assumes correct FFprobe path)
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
        stream_info = data.get("streams", [{}])[0]

        # Calculate FPS
        fps_str = stream_info.get("r_frame_rate", "0/1")
        num, den = map(int, fps_str.split("/")) if "/" in fps_str else (0, 1)
        frame_rate = round(num / den, 3) if den != 0 else 0

        metadata = {
            "original_path": video_path,
            "duration_sec": round(float(stream_info.get("duration", 0)), 3),
            "frame_rate": frame_rate,
            "width": int(stream_info.get("width", 0)),
            "height": int(stream_info.get("height", 0)),
            "codec_name": stream_info.get("codec_name"),
            "color_space": stream_info.get("pix_fmt"),
        }
        return metadata
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(
            f"FFprobe failed for {video_path}. Error: {e.stderr if hasattr(e, 'stderr') else e}"
        )
        return None
    except Exception as e:
        logger.error(f"Unexpected error during extraction for {video_path}: {e}")
        return None


def process_and_update_metadata():
    """Queries DB for pending videos, runs FFprobe, and updates metadata."""

    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        logger.error("Missing required DB environment variables.")
        sys.exit(1)

    conn = None
    id_path_map = {}

    # 1. Fetch file paths for processing (videos missing metadata)
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cursor = conn.cursor()

        # Select video_id and original_path for videos that are missing duration (metadata)
        query = "SELECT video_id, original_path FROM video_metadata WHERE duration_sec IS NULL;"
        cursor.execute(query)

        id_path_map = {path: video_id for video_id, path in cursor.fetchall()}
        paths_to_process = list(id_path_map.keys())

        logger.info(
            f"Found {len(paths_to_process)} videos pending FFprobe metadata extraction."
        )
        conn.close()

    except psycopg2.Error as e:
        logger.error(f"DB Error fetching pending paths: {e}")
        if conn:
            conn.close()
        raise

    if not paths_to_process:
        logger.warning("No videos require metadata extraction. Exiting successfully.")
        return

    # 2. Initialize Ray and Run Distributed Extraction
    initialize_ray()

    futures = [run_ffprobe_extraction.remote(path) for path in paths_to_process]
    all_metadata_list = ray.get(futures)
    ray.shutdown()

    # 3. Assemble and Execute Bulk DB Update
    valid_updates = []
    for metadata in all_metadata_list:
        if metadata is not None:
            path = metadata["original_path"]
            video_id = id_path_map.get(path)
            if video_id is not None:
                # Update tuple order: duration, frame_rate, width, height, codec_name, color_space, video_id
                update_tuple = (
                    metadata["duration_sec"],
                    metadata["frame_rate"],
                    metadata["width"],
                    metadata["height"],
                    metadata["codec_name"],
                    metadata["color_space"],
                    video_id,
                )
                valid_updates.append(update_tuple)

    if not valid_updates:
        logger.warning("No valid metadata collected for database update.")
        return

    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cursor = conn.cursor()

        # Update video_metadata table
        update_statement = """
            UPDATE video_metadata SET
                duration_sec = %s,
                frame_rate = %s,
                width = %s,
                height = %s,
                codec_name = %s,
                color_space = %s,
                scan_date = CURRENT_TIMESTAMP
            WHERE video_id = %s;
        """

        cursor.executemany(update_statement, valid_updates)
        conn.commit()
        logger.success(
            f"Successfully updated metadata for {len(valid_updates)} videos."
        )

    except psycopg2.Error as e:
        logger.error(f"DB Error during metadata update: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    setup_logger()
    # Note: No custom arguments needed, the script reads all pending work from the DB.
    process_and_update_metadata()
