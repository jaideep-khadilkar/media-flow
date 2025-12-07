import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional

# --- Database Dependencies ---
import psycopg2
from loguru import logger

# --- Ray Dependencies ---
import ray

# NOTE: The database credentials will be passed via environment variables
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


# --- Ray Remote Function for Extraction ---


@ray.remote
def run_ffprobe_extraction(video_path: str) -> Optional[Dict[str, Any]]:
    """Executes ffprobe on a single video path and returns structured metadata."""
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
        )  # 10s timeout
        data = json.loads(result.stdout)

        stream_info = data.get("streams", [{}])[0]

        # Calculate FPS
        fps_str = stream_info.get("r_frame_rate", "0/1")
        num, den = map(int, fps_str.split("/")) if "/" in fps_str else (0, 1)
        frame_rate = round(num / den, 3) if den != 0 else 0

        metadata = {
            "original_path": video_path,
            "filename": os.path.basename(video_path),
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


# --- Bulk Insertion Function ---


def insert_metadata_to_db(metadata_list: List[Dict[str, Any]]):
    """Inserts a list of extracted metadata records into the PostgreSQL database."""

    valid_records = [m for m in metadata_list if m]
    if not valid_records:
        logger.info("No valid metadata records to insert.")
        return

    conn = None
    try:
        logger.info(f"Attempting to connect to DB: {DB_HOST}/{DB_NAME}")
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cursor = conn.cursor()

        columns = (
            "original_path",
            "filename",
            "duration_sec",
            "frame_rate",
            "width",
            "height",
            "codec_name",
            "color_space",
        )

        values = [
            (
                record["original_path"],
                record["filename"],
                record["duration_sec"],
                record["frame_rate"],
                record["width"],
                record["height"],
                record["codec_name"],
                record["color_space"],
            )
            for record in valid_records
        ]

        # Use UPSERT to handle re-scans of the same video path
        insert_statement = f"""
            INSERT INTO video_metadata ({', '.join(columns)}) 
            VALUES ({', '.join(['%s'] * len(columns))})
            ON CONFLICT (original_path) DO UPDATE SET
                duration_sec = EXCLUDED.duration_sec,
                frame_rate = EXCLUDED.frame_rate,
                width = EXCLUDED.width,
                height = EXCLUDED.height,
                codec_name = EXCLUDED.codec_name,
                color_space = EXCLUDED.color_space,
                scan_date = CURRENT_TIMESTAMP
        """

        cursor.executemany(insert_statement, values)
        conn.commit()
        logger.success(
            f"Successfully inserted/updated {len(valid_records)} metadata records."
        )

    except psycopg2.Error as e:
        logger.error(f"Database Error during bulk insert: {e}")
        if conn:
            conn.rollback()
        raise  # Re-raise the error to fail the Airflow task
    finally:
        if conn:
            conn.close()


# --- Main Orchestration ---

if __name__ == "__main__":
    setup_logger()

    parser = argparse.ArgumentParser(
        description="Extract FFprobe metadata using Ray and insert into DB."
    )
    parser.add_argument(
        "--input_json",
        required=True,
        help="Path to the JSON file containing the list of video paths",
    )
    args = parser.parse_args()

    # 0. Check Database Environment Variables
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        logger.error(
            "Missing required DB environment variables (DB_HOST, DB_NAME, DB_USER, DB_PASSWORD)."
        )
        sys.exit(1)

    # 1. Load Video Paths
    if not os.path.exists(args.input_json):
        logger.error(f"Input JSON file not found: {args.input_json}")
        sys.exit(1)

    with open(args.input_json, "r") as f:
        video_paths: List[str] = json.load(f)

    if not video_paths:
        logger.warning("Input video list is empty. Skipping extraction.")
        sys.exit(0)

    # 2. Initialize Ray
    try:
        # Connect to a running Ray head or start local Ray (depending on your worker image setup)
        ray.init(address="auto", ignore_reinit_error=True)
        logger.info(f"Ray initialized. Total paths to process: {len(video_paths)}")
    except Exception as e:
        ray.init(local_mode=False)
        logger.warning("Ray auto-connect failed, started a local Ray instance.")
        # logger.error(f"Failed to initialize Ray: {e}")
        # sys.exit(1)

    # 3. Distributed Extraction
    futures = [run_ffprobe_extraction.remote(path) for path in video_paths]
    all_metadata_list = ray.get(futures)

    ray.shutdown()

    # 4. Bulk DB Insertion (runs on the orchestrating container)
    insert_metadata_to_db(all_metadata_list)
