import json
import os
import sys
from typing import Dict, Optional, List, Tuple, Any

import hydra
import psycopg2
import ray
import whisper
from loguru import logger
from omegaconf import DictConfig

from media_flow.utils.fault_tolerance import RAY_TASK_CONFIG, process_ray_results

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


def load_whisper_model(model_name: str):
    print(f"Worker {os.getpid()} loading Whisper model: {model_name}...")
    return whisper.load_model(model_name)


def transcribe_media(model: Any, video_path: str) -> Dict[str, Any]:
    print(f"Transcribing {os.path.basename(video_path)}...")
    # fp16=False for CPU compatibility (unchanged)
    return model.transcribe(video_path, fp16=False)


def build_transcription_data(
    video_id: int, video_path: str, model_name: str, transcript_text: str
) -> Dict[str, Any]:
    return {
        "video_id": video_id,
        "original_path": video_path,
        "model_used": model_name,
        "transcription": transcript_text.strip(),
    }


def write_json_file(output_json_path: str, data: Dict[str, Any]) -> None:
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def ensure_db_credentials() -> None:
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        logger.error("Missing required DB environment variables.")
        sys.exit(1)


def get_base_output_dir(cfg: DictConfig) -> str:
    base_output_dir = cfg.get("transcribe", {}).get(
        "output_dir", "/app/data/transcriptions"
    )
    os.makedirs(base_output_dir, exist_ok=True)
    return base_output_dir


def get_whisper_model_name(cfg: DictConfig) -> str:
    return cfg.get("transcribe", {}).get("model", "base")


def fetch_all_quality_videos_from_db() -> List[Tuple[int, str, str]]:
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cursor = conn.cursor()
        query = """
            SELECT video_id, original_path, filename
            FROM video_metadata 
            WHERE original_path IS NOT NULL 
              AND is_quality_video = TRUE;
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return rows
    except psycopg2.Error as e:
        logger.error(f"DB Error fetching videos: {e}")
        if conn:
            conn.close()
        sys.exit(1)


def build_output_path(base_output_dir: str, filename: str) -> str:
    basename = os.path.splitext(filename)[0]
    json_filename = f"{basename}.json"
    return os.path.join(base_output_dir, json_filename)


def filter_unprocessed_videos(
    all_videos: List[Tuple[int, str, str]], base_output_dir: str
) -> List[Tuple[int, str, str]]:
    videos_to_process: List[Tuple[int, str, str]] = []
    for video_id, video_path, filename in all_videos:
        output_path = build_output_path(base_output_dir, filename)
        if not os.path.exists(output_path):
            videos_to_process.append((video_id, video_path, output_path))
    return videos_to_process


def submit_transcribe_tasks(
    videos_to_process: List[Tuple[int, str, str]], whisper_model: str
) -> Tuple[List[Any], Dict[Any, int]]:
    futures: List[Any] = []
    future_to_id: Dict[Any, int] = {}
    for video_id, video_path, output_path in videos_to_process:
        future = transcribe_video_task.remote(
            video_id=video_id,
            video_path=video_path,
            output_json_path=output_path,
            model_name=whisper_model,
        )
        futures.append(future)
        future_to_id[future] = video_id
    return futures, future_to_id


def handle_transcribe_results(futures: List[Any], future_to_id: Dict[Any, int]) -> None:
    for res in process_ray_results(futures, future_to_id, "transcribe"):
        logger.info(f"Created transcription: {res['output_path']}")


# --- Ray Remote Task: Transcribe Video ---
@ray.remote(**RAY_TASK_CONFIG)
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

        model = load_whisper_model(model_name)
        result = transcribe_media(model, video_path)
        transcript_text = result["text"]

        data = build_transcription_data(
            video_id, video_path, model_name, transcript_text
        )
        write_json_file(output_json_path, data)

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
    base_output_dir = get_base_output_dir(cfg)
    ensure_db_credentials()

    whisper_model = get_whisper_model_name(cfg)
    all_videos = fetch_all_quality_videos_from_db()

    videos_to_process = filter_unprocessed_videos(all_videos, base_output_dir)
    if not videos_to_process:
        logger.info("No new videos require transcription (all JSON files exist).")
        return

    initialize_ray()
    logger.info(f"Starting transcription on {len(videos_to_process)} videos...")

    futures, future_to_id = submit_transcribe_tasks(videos_to_process, whisper_model)
    handle_transcribe_results(futures, future_to_id)

    ray.shutdown()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    transcribe_pipeline(cfg)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
