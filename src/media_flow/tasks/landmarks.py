import json
import os
import sys
from typing import Dict, Optional, List

import cv2
import hydra
import mediapipe as mp
import numpy as np
import psycopg2
import ray
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


def record_failure(video_id, task_name, error_msg):
    """Logs a failure to the processing_failures table."""
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cursor = conn.cursor()

        # Upsert: Insert new failure or increment count if exists
        upsert_stmt = """
            INSERT INTO processing_failures (video_id, task_name, error_message, attempt_count)
            VALUES (%s, %s, %s, 1)
            ON CONFLICT (video_id, task_name) 
            DO UPDATE SET 
                attempt_count = processing_failures.attempt_count + 1,
                last_attempt = CURRENT_TIMESTAMP,
                error_message = EXCLUDED.error_message;
        """
        cursor.execute(upsert_stmt, (video_id, task_name, str(error_msg)))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to log failure: {e}")
    finally:
        if conn:
            conn.close()


# --- Ray Task with Fault Tolerance ---
# max_retries=2 means Ray will automatically retry up to 2 times if the worker crashes
# retry_exceptions=True means it will also retry if Python raises an Exception
@ray.remote(max_retries=2, retry_exceptions=True)
def detect_landmarks_task(
    video_id: int,
    video_path: str,
    output_video_path: str,
    output_json_path: str,
) -> Optional[Dict]:
    """
    Ray Task: Detects facial landmarks, creates an overlay video, and exports animation JSON.
    """
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    try:
        # Initialize Face Mesh
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {video_path}")

            # Video Properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if width == 0 or height == 0:
                raise RuntimeError(f"Invalid video dimensions for {video_path}")

            # Initialize Video Writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if not out.isOpened():
                raise RuntimeError(
                    f"Failed to open output video writer at {output_video_path}"
                )

            animation_data = {
                "video_id": video_id,
                "fps": fps,
                "total_frames": total_frames,
                "frames": [],
            }

            print(
                f"Processing landmarks for {os.path.basename(video_path)} ({total_frames} frames)..."
            )

            frame_idx = 0
            while True:
                success, image = cap.read()
                if not success:
                    break

                # MediaPipe requires RGB
                image.flags.writeable = False
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)

                # Draw the annotations on the image
                image.flags.writeable = True
                image_bgr = image

                frame_landmarks_data = []

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # 1. Draw Overlay
                        mp_drawing.draw_landmarks(
                            image=image_bgr,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                        )
                        mp_drawing.draw_landmarks(
                            image=image_bgr,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                        )

                        # 2. Extract Data for JSON
                        face_data = []
                        for lm in face_landmarks.landmark:
                            face_data.append(
                                {
                                    "x": round(lm.x, 5),
                                    "y": round(lm.y, 5),
                                    "z": round(lm.z, 5),
                                }
                            )
                        frame_landmarks_data.append(face_data)

                animation_data["frames"].append(
                    {"frame_idx": frame_idx, "faces": frame_landmarks_data}
                )

                out.write(image_bgr)
                frame_idx += 1

            cap.release()
            out.release()

            # Save JSON Data
            with open(output_json_path, "w") as f:
                json.dump(animation_data, f)

            print(f"Finished landmarks: {output_video_path}")

            return {
                "video_id": video_id,
                "overlay_path": output_video_path,
                "json_path": output_json_path,
                "status": "SUCCESS",
            }

    except Exception as e:
        # Re-raise so Ray knows it failed (and can retry if configured)
        raise RuntimeError(f"Task failed for video {video_id}: {str(e)}")


def save_landmark_record(record: Dict):
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cursor = conn.cursor()

        insert_stmt = """
            INSERT INTO video_landmarks (video_id, overlay_path, landmarks_json_path)
            VALUES (%s, %s, %s)
            ON CONFLICT (video_id) DO NOTHING;
        """
        cursor.execute(
            insert_stmt,
            (record["video_id"], record["overlay_path"], record["json_path"]),
        )
        conn.commit()
        logger.info(f"Landmark record saved for video ID {record['video_id']}")

    except psycopg2.Error as e:
        logger.error(f"DB Error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


def landmarks_pipeline(cfg: DictConfig):
    setup_logger()

    output_dir = os.path.join(cfg.augment.output_dir, "landmarks")
    os.makedirs(output_dir, exist_ok=True)

    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        logger.error("Missing DB credentials.")
        sys.exit(1)

    # 1. Fetch videos
    conn = None
    videos_to_process = []
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cursor = conn.cursor()

        # Select videos that:
        # 1. Are quality videos
        # 2. Are NOT already in video_landmarks
        # 3. Have NOT failed this specific task ('landmarks') more than 3 times (Poison Pill)
        query = """
            SELECT vm.video_id, vm.original_path, vm.filename
            FROM video_metadata vm
            WHERE vm.is_quality_video = TRUE
              AND NOT EXISTS (
                  SELECT 1 FROM video_landmarks vl WHERE vl.video_id = vm.video_id
              )
              AND (
                  SELECT COALESCE(MAX(attempt_count), 0) 
                  FROM processing_failures pf 
                  WHERE pf.video_id = vm.video_id AND pf.task_name = 'landmarks'
              ) < 3;
        """
        cursor.execute(query)
        videos_to_process = cursor.fetchall()
        conn.close()
    except Exception as e:
        logger.error(f"DB Fetch Error: {e}")
        sys.exit(1)

    if not videos_to_process:
        logger.info("No videos pending landmark detection.")
        return

    initialize_ray()
    logger.info(f"Detecting landmarks for {len(videos_to_process)} videos...")

    # 2. Submit Futures
    future_to_video_id = {}
    futures = []

    for video_id, video_path, filename in videos_to_process:
        basename = os.path.splitext(filename)[0]
        vid_out = os.path.join(output_dir, f"{basename}_overlay.mp4")
        json_out = os.path.join(output_dir, f"{basename}_landmarks.json")

        future = detect_landmarks_task.remote(
            video_id=video_id,
            video_path=video_path,
            output_video_path=vid_out,
            output_json_path=json_out,
        )
        futures.append(future)
        future_to_video_id[future] = video_id

    # 3. Safe Result Retrieval (The "One Bad Apple" Fix)
    # Use ray.wait loop or iterate carefully to catch individual exceptions
    completed_futures, _ = ray.wait(futures, num_returns=len(futures), timeout=None)

    for future in completed_futures:
        vid_id = future_to_video_id[future]
        try:
            # ray.get() here will raise the exception if the task failed
            res = ray.get(future)

            if res and res["status"] == "SUCCESS":
                save_landmark_record(res)

        except Exception as e:
            # Task failed (after Ray retries). Log the Poison Pill.
            error_message = str(e)
            logger.error(f"Task failed permanently for video {vid_id}: {error_message}")
            record_failure(vid_id, "landmarks", error_message)

    ray.shutdown()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    landmarks_pipeline(cfg)


if __name__ == "__main__":
    main()
