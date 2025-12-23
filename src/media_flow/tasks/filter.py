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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def filter_and_mark_videos(cfg: DictConfig):
    """Queries DB for videos meeting filtering criteria and updates a status column."""

    min_width = cfg.filter.get("min_width", 360)
    max_duration = cfg.filter.get("max_duration_sec", 60)
    status_column = cfg.filter.get("status_column", "is_quality_video")

    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        logger.error("Missing required DB environment variables.")
        sys.exit(1)

    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cursor = conn.cursor()

        # NOTE: The status_column (is_quality_video) is assumed to be created by the setup DAG.

        # Step 1: Reset the status of all videos to FALSE before running new filter
        # This prevents old flags from sticking if criteria change.
        reset_query = f"UPDATE video_metadata SET {status_column} = FALSE;"
        cursor.execute(reset_query)
        conn.commit()

        # Step 2: Identify and mark videos that satisfy the criteria
        # The criteria uses populated metadata (width, duration_sec).
        update_query = f"""
            UPDATE video_metadata
            SET {status_column} = TRUE
            WHERE width >= %s
              AND duration_sec <= %s
              AND duration_sec IS NOT NULL;
        """

        cursor.execute(update_query, (min_width, max_duration))
        count = cursor.rowcount
        conn.commit()

        logger.success(
            f"Filtering complete. Marked {count} videos as {status_column}=TRUE."
        )

    except psycopg2.Error as e:
        logger.error(f"DB Error during filtering and marking: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    setup_logger()
    # pylint: disable=no-value-for-parameter
    filter_and_mark_videos()
