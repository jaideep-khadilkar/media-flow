import argparse
import json
import os
import subprocess as sp
import sys
from typing import Dict, List, Optional

import albumentations as A
import decord
import ffmpeg
import numpy as np
import psycopg2
import ray
import torch
from loguru import logger

# Decord needs to be configured to use CPU for some ops if not explicitly on GPU
decord.bridge.set_bridge("torch")
# decord.set_num_threads(1)

# Database Connection Details (pulled from environment)
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")


class FFMPEGPipeWriter:
    """
    Manages an asynchronous FFmpeg subprocess pipe for writing video frames.
    Accepts raw RGB NumPy arrays and pipes them to FFmpeg for encoding.
    """

    def __init__(
        self,
        filename,
        width,
        height,
        fps,
        vcodec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
    ):
        self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps
        self.vcodec = vcodec
        self.pix_fmt_in = pix_fmt_in
        self.pix_fmt_out = pix_fmt_out
        self.pipe: Optional[sp.Popen] = None
        self._start_pipe()

    def _start_pipe(self):
        # Build the FFmpeg command stream
        stream = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt=self.pix_fmt_in,
                s=f"{self.width}x{self.height}",
                r=self.fps,
            )
            .output(
                self.filename,
                vcodec=self.vcodec,
                pix_fmt=self.pix_fmt_out,
                r=self.fps,
                # Use a standard mp4 settings for good compatibility and speed
                preset="fast",
                tune="zerolatency",  # Good for streaming pipeline
                movflags="faststart",
            )
            .overwrite_output()
        )

        # Start the asynchronous process
        # pipe_stdin=True opens the pipe for writing frames to FFmpeg
        self.pipe = stream.run_async(pipe_stdin=True, pipe_stderr=True)
        logger.info(f"FFmpeg process started for {self.filename}")

    def write_frame(self, frame_np: np.ndarray):
        """
        Writes a single NumPy frame (HWC, RGB, uint8) to the pipe.
        """
        if self.pipe and self.pipe.stdin:
            self.pipe.stdin.write(
                frame_np.astype(np.uint8).tobytes()  # Ensure it's uint8
            )
        else:
            raise IOError("FFmpeg pipe is not open or process failed to start.")

    def close(self):
        """
        Closes the input pipe and waits for FFmpeg to finish writing.
        """
        if self.pipe:
            if self.pipe.stdin:
                self.pipe.stdin.close()

            # Read stderr and stdout to prevent pipe filling up
            _, stderr = self.pipe.communicate()

            if self.pipe.returncode != 0:
                logger.error(
                    f"FFmpeg process exited with error code {self.pipe.returncode}."
                )
                if stderr:
                    logger.error(f"FFmpeg stderr: {stderr.decode('utf8', 'ignore')}")

            self.pipe.wait()
            self.pipe = None
            logger.info(f"FFmpeg process closed for {self.filename}")

    # Use context manager for reliable close
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def setup_logger():
    # ... (log setup remains the same)
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )


def initialize_ray():
    if ray.is_initialized():
        ray.shutdown()
    # Limit object store memory to prevent Ray from eating all RAM
    # IMPORTANT: Ensure the worker nodes have GPUs available for this to work
    ray.init(
        log_to_driver=True,
        include_dashboard=False,
        # object_store_memory is less critical if data stays on GPU
        object_store_memory=2 * 1024 * 1024 * 1024,
    )


# --- The Augmented Worker Function (optimized for GPU/PyTorch) ---
@ray.remote(num_gpus=1)  # Allocate a GPU for this worker
def augment_batch(batch_frames: List[Dict], augmentation_params: str):
    """
    Augments a batch of frames (PyTorch tensors on GPU) and returns
    augmented frames (PyTorch tensors on CPU).
    """
    # Define pipeline inside the worker to ensure serialization
    params = json.loads(augmentation_params)

    # Albumentations expects HWC (Height, Width, Channel) format, RGB, numpy array
    augmenter = A.Compose(
        [
            A.CoarseDropout(**params.get("CoarseDropout", {"p": 0.8})),
            A.Rotate(**params.get("Rotate", {"limit": 10, "p": 0.7})),
            A.RandomBrightnessContrast(
                **params.get("RandomBrightnessContrast", {"p": 0.5})
            ),
            A.GaussNoise(**params.get("GaussNoise", {"p": 0.5})),
            # Removed cv2.INTER_LINEAR, relying on default linear interpolation
            A.Resize(height=480, width=480, p=1.0),
        ],
        p=1.0,
    )

    results = []
    for frame_data in batch_frames:
        original_tensor = frame_data["image"]

        # Convert PyTorch Tensor (GPU) -> NumPy Array (CPU) for Albumentations
        # Assuming decord reads HWC, RGB, uint8/float, need to handle dtype/scale if float
        # Decord output is often float when use_py_array=True and bridge='torch'
        original_np = original_tensor.cpu().numpy()

        # If tensor is float (0-1), scale to uint8 (0-255) for albumentations
        if original_np.dtype == np.float32 or original_np.dtype == np.float64:
            original_np = (original_np * 255).astype(np.uint8)

        # Apply Augmentation on the NumPy array
        # This augmented_np is HWC, RGB, uint8
        augmented_np = augmenter(image=original_np)["image"]

        # Convert NumPy Array (CPU) -> PyTorch Tensor (CPU) for return
        augmented_tensor_cpu = torch.from_numpy(augmented_np)

        results.append(
            {
                "frame_id": frame_data["frame_id"],
                "augmented_image": augmented_tensor_cpu,
            }
        )

    return results


def process_video_streaming(
    video_id: int,
    video_path: str,
    output_path: str,
    augmentation_params: str,
    batch_size=60,
) -> Optional[Dict]:
    """
    Reads video using Decord (on GPU), augments frames (on GPU via Ray),
    and writes augmented video using FFMPEGPipeWriter (piping to FFmpeg).
    """
    try:
        # Initialize Decord VideoReader to output PyTorch tensors and use GPU
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    except Exception as e:
        logger.error(f"Cannot open or initialize Decord for {video_path}: {e}")
        return {
            "video_id": video_id,
            "status": "ERROR",
            "error_message": f"Decord reader failed: {e}",
        }

    # Get Video Properties
    width = 480  # Matches your resize target
    height = 480  # Matches your resize target
    fps = vr.get_avg_fps()

    # Initialize FFMPEGPipeWriter
    try:
        # Use context manager for reliable resource cleanup
        with FFMPEGPipeWriter(output_path, width, height, fps) as out:
            logger.info(
                f"Processing {os.path.basename(video_path)} in batches of {batch_size}..."
            )

            current_batch = []
            frame_id = 0
            futures = []

            # Stream frames using Decord indices
            for frame_idx, frame_tensor in enumerate(vr):
                # Read frame as PyTorch tensor on GPU
                # frame_tensor = vr[frame_idx]

                current_batch.append({"frame_id": frame_idx, "image": frame_tensor})
                frame_id += 1

                if len(current_batch) == batch_size:
                    # Send batch to Ray worker, passing augmentation parameters
                    future = augment_batch.remote(current_batch, augmentation_params)
                    futures.append(future)
                    current_batch = []

                    if len(futures) > 4:  # Limit active futures
                        ready_ids, futures = ray.wait(futures, num_returns=1)
                        result_batch = ray.get(ready_ids[0])

                        # Write immediately to disk
                        result_batch.sort(key=lambda x: x["frame_id"])
                        for res in result_batch:
                            # Convert PyTorch Tensor (CPU, RGB, uint8) -> NumPy Array (CPU, RGB, uint8)
                            augmented_np_rgb = res["augmented_image"].numpy()
                            # Write RGB frame to FFmpeg pipe
                            out.write_frame(augmented_np_rgb)

            # Process final batch if it's not empty
            if current_batch:
                future = augment_batch.remote(current_batch, augmentation_params)
                futures.append(future)
                current_batch = []

            # Process remaining futures
            for future in futures:
                result_batch = ray.get(future)
                result_batch.sort(key=lambda x: x["frame_id"])
                for res in result_batch:
                    # Convert PyTorch Tensor (CPU, RGB, uint8) -> NumPy Array (CPU, RGB, uint8)
                    augmented_np_rgb = res["augmented_image"].numpy()
                    out.write_frame(augmented_np_rgb)

    except Exception as e:
        logger.error(f"Video writing failed for {output_path}: {e}")
        return {
            "video_id": video_id,
            "status": "ERROR",
            "error_message": f"Writer failed: {e}",
        }

    logger.success(f"Finished augmentation: {output_path}")

    # Return success record to be written to DB
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


def augment_pipeline(output_dir: str):
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
    for video_id, video_path, filename in videos_to_augment:
        basename = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{basename}_aug.mp4")

        # Process video and get result metadata
        result_record = process_video_streaming(
            video_id=video_id,
            video_path=video_path,
            output_path=output_path,
            augmentation_params=augmentation_settings,
        )

        # Record the result in the database
        if result_record:
            insert_augmentation_record(result_record)

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    augment_pipeline(args.output_dir)
