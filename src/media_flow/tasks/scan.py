import glob
import os
import sys

import hydra
import psycopg2
from loguru import logger
from omegaconf import DictConfig

# Database Connection Details (pulled from environment)
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


def scan_and_insert_videos(cfg: DictConfig):
    """Scans the directory and inserts paths of NEW videos into DB."""

    input_dir = cfg.scan.input_dir
    max_videos = cfg.scan.max_videos
    extensions = cfg.scan.video_extensions

    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        logger.error("Missing required DB environment variables.")
        sys.exit(1)

    # 1. Scan Local Directory
    video_files = []
    for ext in extensions:
        # Use os.path.abspath to ensure unique path integrity in the DB
        found = glob.glob(os.path.join(input_dir, f"*{ext}"))
        video_files.extend(found)

    video_files.sort()

    if max_videos is not None and max_videos > 0 and max_videos < len(video_files):
        logger.warning(f"Limiting scan to the first {max_videos} videos.")
        video_files = video_files[:max_videos]

    if not video_files:
        logger.warning("No video files found to process.")
        return

    # 2. Bulk Insert New Videos (Ignore existing via ON CONFLICT)
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cursor = conn.cursor()

        data_to_insert = [
            (os.path.abspath(path), os.path.basename(path)) for path in video_files
        ]

        insert_statement = """
            INSERT INTO video_metadata (original_path, filename) 
            VALUES (%s, %s)
            ON CONFLICT (original_path) DO NOTHING;
        """

        cursor.executemany(insert_statement, data_to_insert)
        conn.commit()
        logger.success(
            f"Scan complete. Processed {len(video_files)} files. New records inserted/skipped."
        )

    except psycopg2.Error as e:
        logger.error(f"DB Error during scan insertion: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    setup_logger()

    if not os.path.exists(cfg.scan.input_dir):
        logger.error(f"Input directory does not exist: {cfg.scan.input_dir}")
        sys.exit(1)

    scan_and_insert_videos(cfg)


if __name__ == "__main__":
    main()
