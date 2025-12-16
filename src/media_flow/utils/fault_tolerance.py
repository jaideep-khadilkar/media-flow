# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring
import os
from typing import Any, Dict, Generator, List

import psycopg2
import ray
from loguru import logger

# --- Configuration ---
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")

# Standard configuration for all Ray tasks
# retry_exceptions=True ensures that even if Python code crashes (e.g. cv2 error), it retries.
RAY_TASK_CONFIG = {"max_retries": 3, "retry_exceptions": True}


def _get_db_connection():
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        logger.error("Missing DB credentials.")
        return None
    try:
        return psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
    except Exception as e:
        logger.error(f"DB Connection failed: {e}")
        return None


def register_failure(video_id: int, task_name: str, error_message: str):
    """Logs a permanent failure to the database (The Poison Pill)."""
    conn = _get_db_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor()
        upsert_stmt = """
            INSERT INTO processing_failures (video_id, task_name, error_message, attempt_count)
            VALUES (%s, %s, %s, 1)
            ON CONFLICT (video_id, task_name) 
            DO UPDATE SET 
                attempt_count = processing_failures.attempt_count + 1,
                last_attempt = CURRENT_TIMESTAMP,
                error_message = EXCLUDED.error_message;
        """
        cursor.execute(upsert_stmt, (video_id, task_name, str(error_message)))
        conn.commit()
        logger.warning(f"Recorded failure for Video {video_id} in task '{task_name}'")
    except Exception as e:
        logger.error(f"Failed to log failure for video {video_id}: {e}")
    finally:
        conn.close()


def process_ray_results(
    futures: List[ray.ObjectRef], future_to_id: Dict[ray.ObjectRef, int], task_name: str
) -> Generator[Any, None, None]:
    """
    Orchestrates the Ray execution with Fault Tolerance.

    1. Waits for tasks to complete.
    2. Catches worker crashes or exceptions.
    3. Logs failures to DB automatically.
    4. Yields ONLY successful results to the caller.
    """
    if not futures:
        return

    logger.info(f"Waiting for {len(futures)} tasks to complete...")

    # We loop until all futures are processed
    while futures:
        # Wait for the next task to finish (returns lists of done and pending)
        done_futures, futures = ray.wait(futures, num_returns=1)

        for future in done_futures:
            video_id = future_to_id.get(future)
            try:
                # This raises an exception if the task failed (after 3 retries)
                result = ray.get(future)

                # Check for explicit application-level errors (dictionaries with "status": "ERROR")
                if isinstance(result, dict) and result.get("status") == "ERROR":
                    raise RuntimeError(
                        result.get("error_message", "Unknown Application Error")
                    )

                # If we got here, it's a success! Yield it.
                if result:
                    yield result

            except Exception as e:
                # Catch-all for RayTaskError, WorkerCrashedError, or Application Error
                error_msg = str(e)
                logger.error(
                    f"Task '{task_name}' failed permanently for Video {video_id}: {error_msg}"
                )

                # Apply the Poison Pill
                if video_id:
                    register_failure(video_id, task_name, error_msg)
