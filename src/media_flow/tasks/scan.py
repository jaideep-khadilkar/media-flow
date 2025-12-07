import argparse
import glob
import json
import os
import sys

from loguru import logger


def setup_logger():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )


def scan_videos(input_dir: str, output_json: str):
    logger.info(f"Scanning directory: {input_dir}")

    # Support common video extensions
    extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    video_files = []

    for ext in extensions:
        # Recursive search can be enabled with recursive=True and **/*.mp4
        found = glob.glob(os.path.join(input_dir, ext))
        video_files.extend(found)

    video_files.sort()
    count = len(video_files)

    logger.info(f"Found {count} video files.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w") as f:
        json.dump(video_files, f, indent=2)

    logger.success(f"Video list saved to: {output_json}")


if __name__ == "__main__":
    setup_logger()

    parser = argparse.ArgumentParser(description="Scan folder for video files.")
    parser.add_argument(
        "--input_dir", required=True, help="Path to the folder containing raw videos"
    )
    parser.add_argument(
        "--output_json", required=True, help="Path to save the resulting JSON list"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    scan_videos(args.input_dir, args.output_json)
