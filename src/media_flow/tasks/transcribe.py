import json
import os
import sys
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
    )
    print("Connected to existing Ray cluster.")


# --- Ray Remote Task: Transcribe Video ---
@ray.remote(max_retries=2, retry_exceptions=True)
def transcribe_video_task(
    video_id: int, video_path: str, output_json_path: str, model_name: str = "base"
) -> Optional[Dict]:
    """
    Loads Whisper model on the worker and transcribes the video file to JSON.
    """
    try:
        if not os.path.exists(video_path):
            return {
                "video_id": video_id,
                "status": "ERROR",
                "error_message": f"File not found: {video_path}",
            }

        print(f"Worker {os.getpid()} loading Whisper model: {model_name}...")
        model = whisper.load_model(model_name)

        print(f"Transcribing {os.path.basename(video_path)}...")

        # Transcribe (fp16=False for CPU compatibility)
        result = model.transcribe(video_path, fp16=False)
        transcript_text = result["text"].strip()

        # Prepare Output Data
        data = {
            "video_id": video_id,
            "original_path": video_path,
            "model_used": model_name,
            "transcription": transcript_text,
        }

        # Write to JSON file
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        return {
            "video_id": video_id,
            "output_path": output_json_path,
            "status": "SUCCESS",
        }

    except Exception as e:
        error_msg = str(e)
        print(f"Error transcribing video {video_id}: {error_msg}")
        return {"video_id": video_id, "status": "ERROR", "error_message": error_msg}


def transcribe_pipeline(cfg: DictConfig):
    setup_logger()

    # Define Output Directory
    # Defaults to /app/data/transcriptions if not set in config
    base_output_dir = cfg.get("transcribe", {}).get(
        "output_dir", "/app/data/transcriptions"
    )
    os.makedirs(base_output_dir, exist_ok=True)

    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        logger.error("Missing required DB environment variables.")
        sys.exit(1)

    conn = None
    videos_to_process = []
    whisper_model = cfg.get("transcribe", {}).get("model", "base")

    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cursor = conn.cursor()

        # 1. Fetch ALL quality videos
        query = """
            SELECT video_id, original_path, filename
            FROM video_metadata 
            WHERE original_path IS NOT NULL 
              AND is_quality_video = TRUE;
        """
        cursor.execute(query)
        all_videos = cursor.fetchall()
        conn.close()

        # 2. Filter: Only process if the JSON file DOES NOT exist
        for video_id, video_path, filename in all_videos:
            basename = os.path.splitext(filename)[0]
            json_filename = f"{basename}.json"
            output_path = os.path.join(base_output_dir, json_filename)

            if not os.path.exists(output_path):
                # Add to processing list if file doesn't exist
                videos_to_process.append((video_id, video_path, output_path))

    except psycopg2.Error as e:
        logger.error(f"DB Error fetching videos: {e}")
        if conn:
            conn.close()
        sys.exit(1)

    if not videos_to_process:
        logger.info("No new videos require transcription (all JSON files exist).")
        return

    initialize_ray()

    logger.info(f"Starting transcription on {len(videos_to_process)} videos...")

    # 3. Submit tasks to Ray
    futures = []
    for video_id, video_path, output_path in videos_to_process:
        future = transcribe_video_task.remote(
            video_id=video_id,
            video_path=video_path,
            output_json_path=output_path,
            model_name=whisper_model,
        )
        futures.append(future)

    # 4. Wait for completion
    # We use a loop here to log individual successes/failures
    completed_futures, _ = ray.wait(futures, num_returns=len(futures), timeout=None)

    success_count = 0
    for future in completed_futures:
        try:
            res = ray.get(future)
            if res and res["status"] == "SUCCESS":
                success_count += 1
                logger.info(f"Created transcription: {res['output_path']}")
            else:
                logger.error(f"Task failed: {res.get('error_message')}")
        except Exception as e:
            logger.error(f"Ray task exception: {e}")

    logger.success(
        f"Transcription complete. {success_count}/{len(videos_to_process)} processed successfully."
    )
    ray.shutdown()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    transcribe_pipeline(cfg)


if __name__ == "__main__":
    main()
