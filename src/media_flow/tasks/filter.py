import argparse
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


def filter_videos(input_json: str, output_json: str, keyword: str):
    logger.info(f"Reading video list from: {input_json}")

    if not os.path.exists(input_json):
        logger.error(f"Input JSON not found: {input_json}")
        sys.exit(1)

    with open(input_json, "r") as f:
        video_list = json.load(f)

    logger.info(f"Filtering {len(video_list)} videos with keyword: '{keyword}'")

    # Filter logic: Check if keyword is in the filename (not the full path)
    filtered_list = [
        vid for vid in video_list if keyword.lower() in os.path.basename(vid).lower()
    ]

    logger.info(f"Retained {len(filtered_list)} videos after filtering.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w") as f:
        json.dump(filtered_list, f, indent=2)

    logger.success(f"Filtered list saved to: {output_json}")


if __name__ == "__main__":
    setup_logger()

    parser = argparse.ArgumentParser(description="Filter video list by keyword.")
    parser.add_argument(
        "--input_json", required=True, help="Path to input JSON video list"
    )
    parser.add_argument(
        "--output_json", required=True, help="Path to save filtered JSON list"
    )
    parser.add_argument(
        "--keyword",
        default="",
        help="Keyword to filter by (default: empty = no filter)",
    )

    args = parser.parse_args()

    filter_videos(args.input_json, args.output_json, args.keyword)
