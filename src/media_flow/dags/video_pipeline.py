from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# Constants
REPO_PATH_ON_HOST = "C:/Users/iamja/Documents/GitHub/media-flow"
REPO_PATH_IN_CONTAINER = "/app"
DATA_PATH = "/app/data"
WORKER_IMAGE = "media-flow:1.6"
POSTGRES_CONN_ID = "postgres_default"

default_args = {
    "owner": "data_engineer",
}

# Common configuration for DockerOperator environment variables
DB_ENV_VARS = {
    "DB_HOST": "{{ conn." + POSTGRES_CONN_ID + ".host }}",
    "DB_NAME": "{{ conn." + POSTGRES_CONN_ID + ".schema }}",
    "DB_USER": "{{ conn." + POSTGRES_CONN_ID + ".login }}",
    "DB_PASSWORD": "{{ conn." + POSTGRES_CONN_ID + ".password }}",
    "RAY_LOG_TO_STDERR": "1",
}

# Shared Mount Configuration
SHARED_MOUNTS = [
    Mount(source=REPO_PATH_ON_HOST, target=REPO_PATH_IN_CONTAINER, type="bind")
]

with DAG(
    dag_id="video_pipeline",
    start_date=datetime(2025, 11, 1),
    schedule=None,  # Run manually or via API only
    catchup=False,
    default_args=default_args,
) as dag:

    # --- Ingest Task (Inserts paths into video_metadata, uses max_videos for dev limit) ---
    ingest_videos = DockerOperator(
        task_id="ingest_videos",
        image=WORKER_IMAGE,
        command="pixi run python src/media_flow/tasks/ingest.py",
        mounts=SHARED_MOUNTS,
        mount_tmp_dir=False,
        environment=DB_ENV_VARS,
        working_dir=REPO_PATH_IN_CONTAINER,
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="media-flow_default",
    )

    # --- Filter Task (Reads metadata, Marks is_quality_video in DB) ---
    filter_videos = DockerOperator(
        task_id="filter_videos",
        image=WORKER_IMAGE,
        # Filtering criteria are arguments
        command="pixi run python src/media_flow/tasks/filter.py",
        mounts=SHARED_MOUNTS,
        mount_tmp_dir=False,
        environment=DB_ENV_VARS,
        working_dir=REPO_PATH_IN_CONTAINER,
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="media-flow_default",
    )

    # --- Augment Task (Reads filtered videos, Writes file to disk and record to augmented_videos DB table) ---
    augment_videos = DockerOperator(
        task_id="augment_videos",
        image=WORKER_IMAGE,
        # Output directory is needed to save the augmented files
        command="pixi run python src/media_flow/tasks/augment.py",
        mounts=SHARED_MOUNTS,
        mount_tmp_dir=False,
        environment=DB_ENV_VARS,
        working_dir=REPO_PATH_IN_CONTAINER,
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="media-flow_default",
    )

    # --- Transcribe Task (New) ---
    transcribe_videos = DockerOperator(
        task_id="transcribe",
        image=WORKER_IMAGE,
        command="pixi run python src/media_flow/tasks/transcribe.py",
        mounts=SHARED_MOUNTS,
        mount_tmp_dir=False,
        environment=DB_ENV_VARS,
        working_dir=REPO_PATH_IN_CONTAINER,
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="media-flow_default",
    )

    # --- Landmarks Task (New) ---
    detect_landmarks = DockerOperator(
        task_id="detect_landmarks",
        image=WORKER_IMAGE,
        command="pixi run python src/media_flow/tasks/landmarks.py",
        mounts=SHARED_MOUNTS,
        mount_tmp_dir=False,
        environment=DB_ENV_VARS,
        working_dir=REPO_PATH_IN_CONTAINER,
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="media-flow_default",
    )

    # --- Final Flow ---
    # pylint: disable=pointless-statement
    ingest_videos >> filter_videos
    filter_videos >> augment_videos
    filter_videos >> transcribe_videos
    filter_videos >> detect_landmarks
