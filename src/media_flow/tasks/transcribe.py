import os
import sys
import psutil
from typing import Dict, Optional

import hydra
import psycopg2
import ray
import whisper
from loguru import logger
from omegaconf import DictConfig

# Database Connection Details
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
        # Optional: Whisper is RAM hungry. Adjust object_store_memory if needed.
    )
    print("Connected to existing Ray cluster.")


# --- Ray Remote Task: Transcribe Video ---
@ray.remote
def transcribe_video_task(
    video_id: int, video_path: str, model_name: str = "base"
) -> Optional[Dict]:
    """
    Loads Whisper model on the worker and transcribes the video file.
    """
    try:
        # Check if file exists
        if not os.path.exists(video_path):
            return {
                "video_id": video_id,
                "status": "ERROR",
                "error_message": f"File not found: {video_path}",
            }

        # Load Model (this happens on the worker node)
        # Note: Models are cached at ~/.cache/whisper.
        # Since we use a persistent volume for /app/data, you might want to map cache there too if re-downloading is an issue.
        print(f"Worker {os.getpid()} loading Whisper model: {model_name}...")
        model = whisper.load_model(model_name)

        print(f"Transcribing {os.path.basename(video_path)}...")

        # Transcribe
        # fp16=False is safer for CPU inference if GPUs are not available
        result = model.transcribe(video_path, fp16=False)
        transcript_text = result["text"].strip()

        return {
            "video_id": video_id,
            "transcription": transcript_text,
            "status": "SUCCESS",
        }

    except Exception as e:
        error_msg = str(e)
        print(f"Error transcribing video {video_id}: {error_msg}")
        return {"video_id": video_id, "status": "ERROR", "error_message": error_msg}


def update_transcription_record(record: Dict):
    """Updates the video_metadata table with the transcription."""
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cursor = conn.cursor()

        if record["status"] == "SUCCESS":
            update_stmt = """
                UPDATE video_metadata 
                SET transcription = %s 
                WHERE video_id = %s;
            """
            cursor.execute(update_stmt, (record["transcription"], record["video_id"]))
            conn.commit()
            logger.info(f"Transcription saved for video ID {record['video_id']}")
        else:
            logger.warning(
                f"Skipping DB update for video ID {record['video_id']}: {record.get('error_message')}"
            )

    except psycopg2.Error as e:
        logger.error(f"DB Error updating transcription: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


def transcribe_pipeline(cfg: DictConfig):
    setup_logger()

    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        logger.error("Missing required DB environment variables.")
        sys.exit(1)

    # 1. Fetch videos that need transcription
    conn = None
    videos_to_process = []

    # Configurable model size (base, small, medium, etc.)
    # You can add this to your Hydra config: transcribe: { model: "base" }
    whisper_model = cfg.get("transcribe", {}).get("model", "base")

    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cursor = conn.cursor()

        # Select videos that exist (have path) but have NO transcription yet
        query = """
            SELECT video_id, original_path 
            FROM video_metadata 
            WHERE original_path IS NOT NULL
              AND video_metadata.is_quality_video = TRUE
              AND transcription IS NULL;
        """
        cursor.execute(query)
        videos_to_process = cursor.fetchall()
        conn.close()

    except psycopg2.Error as e:
        logger.error(f"DB Error fetching videos: {e}")
        if conn:
            conn.close()
        sys.exit(1)

    if not videos_to_process:
        logger.info("No videos require transcription. Exiting.")
        return

    initialize_ray()

    logger.info(
        f"Starting transcription on {len(videos_to_process)} videos using model '{whisper_model}'."
    )

    # 2. Submit tasks to Ray
    futures = []
    for video_id, video_path in videos_to_process:
        # Submit remote task
        future = transcribe_video_task.remote(
            video_id=video_id, video_path=video_path, model_name=whisper_model
        )
        futures.append(future)

    # 3. Collect results
    logger.info("Waiting for Ray transcription tasks...")
    results = ray.get(futures)

    # 4. Update Database
    for res in results:
        if res:
            update_transcription_record(res)

    ray.shutdown()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    transcribe_pipeline(cfg)


if __name__ == "__main__":
    main()
