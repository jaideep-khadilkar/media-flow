import os
import sys
import glob
from loguru import logger
from tqdm import tqdm

# Core Data Science Libraries
import numpy as np
import pandas as pd
import cv2 # OpenCV for video I/O
import albumentations as A # High-performance data augmentation

# Parallel Computing
import ray
import ray.data
from ray.exceptions import RaySystemError

# --- Configuration ---
# Use directory-based I/O
RAW_INPUT_DIR = "./data/raw/tmp"
PROCESSED_OUTPUT_DIR = "./data/processed"

# --- Utility Functions ---

def setup_logger():
    """Configures Loguru for structured logging."""
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}", level="INFO")
    logger.add("pipeline.log", rotation="10 MB", compression="zip", level="DEBUG")

def initialize_ray():
    """Initializes the Ray cluster for distributed computing."""
    try:
        if ray.is_initialized():
            ray.shutdown()
        
        # Initialize Ray and capture the returned information dictionary
        info = ray.init(log_to_driver=True, include_dashboard=True)
        
        # Use the 'webui_url' key from the initialization info dictionary
        webui_url = info.get("webui_url")
        
        logger.info(f"Ray initialized successfully. Dashboard URL: {webui_url}")
        logger.info(f"Available CPUs for Ray: {ray.available_resources().get('CPU', 0)}")
    
    except RaySystemError as e:
        logger.error(f"Failed to initialize Ray: {e}")
        sys.exit(1)

def video_to_frames(video_path: str) -> list[dict]:
    """
    Reads a video file and yields a list of dictionaries, one per frame.
    """
    if not os.path.exists(video_path):
        logger.warning(f"Input video not found at: {video_path}. Skipping frame extraction.")
        return []

    logger.info(f"Extracting frames from {video_path}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file {video_path}")
        return []

    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    for i in tqdm(range(frame_count), desc="Reading Frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Store frame as a NumPy array (standard format for CV/ML)
        frames.append({
            "frame_id": i,
            "image": frame,
            "fps": fps
        })
        
    cap.release()
    logger.info(f"Extraction complete. Total frames: {len(frames)}. FPS: {fps:.2f}")
    return frames

def write_frames_to_video(frames_data: list, output_path: str):
    """
    Takes a list of augmented frames (dictionaries) and writes them to a video file.
    The frames must be sorted by frame_id.
    """
    if not frames_data:
        logger.warning("No frames to write to video. Skipping video output.")
        return

    # 1. Sort frames by ID to ensure correct video order (crucial for Ray data collection)
    sorted_frames = sorted(frames_data, key=lambda x: x['frame_id'])
    
    # 2. Get video parameters from the first frame
    frame_example = sorted_frames[0]['augmented_image']
    height, width, _ = frame_example.shape
    
    # Attempt to get the original FPS. Default to 30 if not found.
    fps = sorted_frames[0].get('fps', 30.0)

    # 3. Define the video writer
    # Use MP4 format (MPEG-4) and 'mp4v' codec (widely compatible)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        logger.error(f"Failed to initialize VideoWriter at {output_path}. Check permissions/codec.")
        return

    logger.info(f"Writing {len(sorted_frames)} frames to {output_path} at {fps:.2f} FPS...")

    # 4. Write all frames
    for frame_data in tqdm(sorted_frames, desc="Writing Video"):
        frame = frame_data['augmented_image']
        out.write(frame)

    # 5. Release the writer object
    out.release()
    logger.success(f"Video output complete. Saved to: {output_path}")


# --- Pipeline Components ---

def apply_augmentation(frame_data: dict) -> dict:
    """
    A Ray-mappable function to apply data augmentation to a single frame.
    """
    # 1. Define the augmentation pipeline (Albumentations)
    augmenter = A.Compose([
        # Randomly apply a black box (CoarseDropout) for masking/occlusion augmentation
        A.CoarseDropout(
            max_holes=1, max_height=64, max_width=64, 
            min_holes=1, min_height=16, min_width=16, 
            fill_value=0, p=0.8),
            
        # Random rotation
        A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.7),
        
        # Adjust color and contrast 
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        
        # Random shift, scale, and rotate 
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.7),
        
        # Resize to a consistent square size 
        A.Resize(height=480, width=480, interpolation=cv2.INTER_LINEAR, p=1.0)
    ], p=1.0)

    # 2. Apply the transform
    original_image = frame_data["image"]
    
    # Albumentations expects RGB, OpenCV reads BGR. Convert before augmentation.
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    augmented_image_rgb = augmenter(image=image_rgb)['image']
    
    # Convert back to BGR for OpenCV compatibility when writing the video
    augmented_image_bgr = cv2.cvtColor(augmented_image_rgb, cv2.COLOR_RGB2BGR)

    # 3. Return the result
    return {
        "frame_id": frame_data["frame_id"],
        "augmented_image": augmented_image_bgr,
        "fps": frame_data["fps"]
    }


def main():
    setup_logger()
    logger.info("Starting MediaFlow Data Augmentation Pipeline...")

    # 1. Initialization
    initialize_ray()

    # Ensure directories exist
    os.makedirs(RAW_INPUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_OUTPUT_DIR, exist_ok=True)

    # Discover input videos
    input_videos = sorted(glob.glob(os.path.join(RAW_INPUT_DIR, "*.mp4")))
    if not input_videos:
        logger.warning(f"No videos found in {RAW_INPUT_DIR}. Creating a dummy video for testing.")
        try:
            dummy_path = os.path.join(RAW_INPUT_DIR, "dummy.mp4")
            dummy_frame = np.zeros((480, 480, 3), dtype=np.uint8)
            dummy_writer = cv2.VideoWriter(
                dummy_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                30,
                (480, 480)
            )
            for _ in range(10):
                dummy_writer.write(dummy_frame)
            dummy_writer.release()
            input_videos = [dummy_path]
            logger.info(f"Created dummy video at {dummy_path}.")
        except Exception as e:
            logger.error(f"Failed to create dummy video. Place an MP4 into {RAW_INPUT_DIR}. Error: {e}")
            ray.shutdown()
            return

    # Process each video
    for video_path in input_videos:
        logger.info(f"Processing video: {video_path}")

        # 2. Ingestion
        frames_list = video_to_frames(video_path)
        if not frames_list:
            logger.error(f"No frames extracted from {video_path}. Skipping.")
            continue

        # 3. Parallel Processing with Ray Data
        logger.info("Distributing data and applying augmentation pipeline with Ray...")
        ds = ray.data.from_items(frames_list)
        augmented_ds = ds.map(apply_augmentation)

        # 4. Collection
        logger.info("Augmentation complete. Collecting results...")
        results = augmented_ds.take_all()

        # 5. Output and Write Video
        basename = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(PROCESSED_OUTPUT_DIR, f"{basename}.mp4")
        logger.success("Ray Data processing complete. Ready to write augmented video.")
        write_frames_to_video(results, output_path)

    ray.shutdown()
    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    # Create the necessary placeholder folders
    os.makedirs("./data/processed", exist_ok=True)
    os.makedirs("./data/raw/tmp", exist_ok=True)
    
    main()