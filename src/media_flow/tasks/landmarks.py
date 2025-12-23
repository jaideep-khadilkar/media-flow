# pylint: disable=missing-function-docstring,too-many-locals,too-many-arguments
import json
import os
import sys
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# pylint: disable=no-member
import cv2
import hydra
import mediapipe as mp
import numpy as np
import psycopg2
import ray
from loguru import logger
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
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
    ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)
    print("Connected to existing Ray cluster.")


def _norm_to_px(lm_x: float, lm_y: float, width: int, height: int) -> Tuple[int, int]:
    x = int(round(lm_x * width))
    y = int(round(lm_y * height))
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def draw_facemesh_overlay(
    frame_bgr: np.ndarray, landmarks: Any, connections: List[Any]
) -> np.ndarray:
    """
    Draws landmarks and connections on the image using OpenCV.
    Handles MediaPipe Tasks API Connection objects.
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    # Pre-calculate pixel coordinates for all landmarks
    pts: List[Tuple[int, int]] = [_norm_to_px(lm.x, lm.y, w, h) for lm in landmarks]

    # Draw Connections (Green)
    if connections:
        for connection in connections:
            start_idx = connection.start
            end_idx = connection.end

            if start_idx < len(pts) and end_idx < len(pts):
                cv2.line(out, pts[start_idx], pts[end_idx], (0, 255, 0), 1, cv2.LINE_AA)

    # Draw Points (Red)
    for x, y in pts:
        cv2.circle(out, (x, y), 1, (0, 0, 255), -1, cv2.LINE_AA)

    return out


def ensure_model_exists(model_path: str, url: str) -> None:
    if not os.path.exists(model_path):
        print(f"Downloading model to {model_path}...")
        try:
            urllib.request.urlretrieve(url, model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}") from e


def build_landmarker_options(model_path: str) -> mp_vision.FaceLandmarkerOptions:
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    return mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )


def get_face_connections():
    # Standard connections from MediaPipe Tasks API
    return mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS


def open_video(video_path: str) -> Tuple[cv2.VideoCapture, int, int, float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, width, height, fps, total_frames


def create_video_writer(
    path: str, fps: float, width: int, height: int
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def init_animation_data(video_id: int, fps: float, total_frames: int) -> Dict[str, Any]:
    return {
        "video_id": video_id,
        "fps": fps,
        "total_frames": total_frames,
        "frames": [],
    }


def compute_timestamp_ms(frame_idx: int, fps: float) -> int:
    return int((frame_idx / fps) * 1000)


def process_frame(
    landmarker: mp_vision.FaceLandmarker,
    image_bgr: np.ndarray,
    frame_idx: int,
    fps: float,
    connections: Any,
) -> Tuple[np.ndarray, Optional[List[List[Dict[str, float]]]]]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    timestamp_ms = compute_timestamp_ms(frame_idx, fps)
    detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if detection_result.face_landmarks:
        face_lms = detection_result.face_landmarks[0]
        annotated = draw_facemesh_overlay(image_bgr, face_lms, connections)
        frame_data: List[List[Dict[str, float]]] = []
        for face in detection_result.face_landmarks:
            face_points = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in face]
            frame_data.append(face_points)
        return annotated, frame_data

    return image_bgr, None


@ray.remote(**RAY_TASK_CONFIG)
def detect_landmarks_task(
    video_id: int,
    video_path: str,
    output_video_path: str,
    output_json_path: str,
) -> Optional[Dict]:
    """
    Ray Task: Detects facial landmarks using MediaPipe Tasks API (0.10+).
    """
    model_path = "/app/data/face_landmarker.task"
    model_url = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker"
        "/face_landmarker/float16/1/face_landmarker.task"
    )
    try:
        ensure_model_exists(model_path, model_url)
        options = build_landmarker_options(model_path)
        connections = get_face_connections()

        with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
            cap, width, height, fps, total_frames = open_video(video_path)
            out = create_video_writer(output_video_path, fps, width, height)
            animation_data = init_animation_data(video_id, fps, total_frames)

            print(f"Processing {os.path.basename(video_path)}...")
            frame_idx = 0
            while True:
                success, image = cap.read()
                if not success:
                    break

                annotated_frame, faces_data = process_frame(
                    landmarker, image, frame_idx, fps, connections
                )

                if faces_data is not None:
                    out.write(annotated_frame)
                    animation_data["frames"].append(
                        {"frame_idx": frame_idx, "faces": faces_data}
                    )
                else:
                    out.write(image)

                frame_idx += 1

            cap.release()
            out.release()

            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(animation_data, f)

            return {
                "video_id": video_id,
                "overlay_path": output_video_path,
                "json_path": output_json_path,
                "status": "SUCCESS",
            }
    except Exception as e:
        raise RuntimeError(f"Task failed: {e}") from e


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


def ensure_db_credentials() -> None:
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        logger.error("Missing DB credentials.")
        sys.exit(1)


def fetch_videos_to_process() -> List[Tuple[int, str, str]]:
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
                  SELECT 1 FROM video_landmarks vl WHERE vl.video_id = vm.video_id
              )
              AND (
                  SELECT COALESCE(MAX(attempt_count), 0) 
                  FROM processing_failures pf 
                  WHERE pf.video_id = vm.video_id AND pf.task_name = 'landmarks'
              ) < 3;
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"DB Fetch Error: {e}")
        sys.exit(1)


def prepare_output_paths(
    videos_output_dir: str, json_output_dir: str, filename: str
) -> Tuple[str, str]:
    basename = os.path.splitext(filename)[0]
    vid_out = os.path.join(videos_output_dir, f"{basename}.mp4")
    json_out = os.path.join(json_output_dir, f"{basename}.json")
    return vid_out, json_out


def submit_landmark_tasks(
    videos_to_process: List[Tuple[int, str, str]],
    videos_output_dir: str,
    json_output_dir: str,
) -> Tuple[List[Any], Dict[Any, int]]:
    futures: List[Any] = []
    future_to_id: Dict[Any, int] = {}
    for video_id, video_path, filename in videos_to_process:
        vid_out, json_out = prepare_output_paths(
            videos_output_dir, json_output_dir, filename
        )
        future = detect_landmarks_task.remote(
            video_id=video_id,
            video_path=video_path,
            output_video_path=vid_out,
            output_json_path=json_out,
        )
        futures.append(future)
        future_to_id[future] = video_id
    return futures, future_to_id


def handle_results(futures: List[Any], future_to_id: Dict[Any, int]) -> None:
    for res in process_ray_results(futures, future_to_id, "landmarks"):
        save_landmark_record(res)


def landmarks_pipeline(cfg: DictConfig):
    setup_logger()
    videos_output_dir = cfg.landmarks.video_output_dir
    json_output_dir = cfg.landmarks.json_output_dir
    os.makedirs(videos_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)

    ensure_db_credentials()
    videos_to_process = fetch_videos_to_process()

    if not videos_to_process:
        logger.info("No videos pending landmark detection.")
        return

    initialize_ray()
    logger.info(f"Detecting landmarks for {len(videos_to_process)} videos...")

    futures, future_to_id = submit_landmark_tasks(
        videos_to_process, videos_output_dir, json_output_dir
    )
    handle_results(futures, future_to_id)

    ray.shutdown()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    landmarks_pipeline(cfg)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
