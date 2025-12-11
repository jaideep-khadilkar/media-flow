from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import DeviceRequest
from docker.types import Mount

# Constants
REPO_PATH_ON_HOST = "C:/Users/iamja/Documents/GitHub/media-flow"
REPO_PATH_IN_CONTAINER = "/app"
DATA_PATH = "/app/data"
WORKER_IMAGE = "media-flow:1.4"
POSTGRES_CONN_ID = "postgres_default"

default_args = {
    "owner": "data_engineer",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
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
    dag_id="video_db_pipeline",
    start_date=datetime(2025, 11, 1),
    schedule_interval=timedelta(days=1),
    catchup=False,
    default_args=default_args,
) as dag:

    # --- 1. Scan Task (Inserts paths into video_metadata, uses max_videos for dev limit) ---
    scan_videos = DockerOperator(
        task_id="scan_videos",
        image=WORKER_IMAGE,
        # Max videos set to 50 for development testing
        command=f"pixi run python src/media_flow/tasks/scan.py --input_dir {DATA_PATH}/raw --max_videos 20",
        mounts=SHARED_MOUNTS,
        mount_tmp_dir=False,
        environment=DB_ENV_VARS,
        working_dir=REPO_PATH_IN_CONTAINER,
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="media-flow_default",
    )

    # --- 2. Extract Metadata Task (Reads pending, Updates video_metadata) ---
    extract_video_metadata = DockerOperator(
        task_id="extract_video_metadata",
        image=WORKER_IMAGE,
        # No arguments needed; reads all pending work from the DB.
        command="pixi run python src/media_flow/tasks/extract_video_metadata.py",
        mounts=SHARED_MOUNTS,
        mount_tmp_dir=False,
        environment=DB_ENV_VARS,
        working_dir=REPO_PATH_IN_CONTAINER,
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="media-flow_default",
    )

    # --- 3. Filter Task (Reads metadata, Marks is_quality_video in DB) ---
    filter_videos = DockerOperator(
        task_id="filter_videos",
        image=WORKER_IMAGE,
        # Filtering criteria are arguments
        command=f"pixi run python src/media_flow/tasks/filter.py --min_width 360 --max_duration 60",
        mounts=SHARED_MOUNTS,
        mount_tmp_dir=False,
        environment=DB_ENV_VARS,
        working_dir=REPO_PATH_IN_CONTAINER,
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="media-flow_default",
    )

    # --- 4. Augment Task (Reads filtered videos, Writes file to disk and record to augmented_videos DB table) ---
    augment_videos = DockerOperator(
        task_id="augment_videos",
        image=WORKER_IMAGE,
        # Output directory is needed to save the augmented files
        command=f"pixi run python src/media_flow/tasks/augment.py --output_dir {DATA_PATH}/augmented",
        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
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
    scan_videos >> extract_video_metadata >> filter_videos >> augment_videos
