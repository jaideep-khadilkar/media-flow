from docker.types import Mount
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Constants
REPO_PATH_ON_HOST = "C:/Users/iamja/Documents/GitHub/media-flow"  # Update if needed
REPO_PATH_IN_CONTAINER = "/app"  # Where code lives in the worker container
DATA_PATH = "/app/data"  # Shared data location

# The image that contains your dependencies (Pixi, Ray, OpenCV)
WORKER_IMAGE = "media-flow:1.0"

default_args = {
    "owner": "data_engineer",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="video_pipeline",
    start_date=datetime(2025, 11, 1),
    schedule_interval=timedelta(days=1),
    catchup=False,
    default_args=default_args,
) as dag:

    # # 0. Setup shared directories
    # create_dirs = BashOperator(
    #     task_id="create_directories",
    #     bash_command=f"mkdir -p {REPO_PATH_IN_CONTAINER}/data/metadata",
    # )

    # 1. Scan Task
    scan_videos = DockerOperator(
        task_id="scan_videos",
        image=WORKER_IMAGE,
        # We run the specific module script using python -m or direct path
        command=f"pixi run python src/media_flow/tasks/scan.py --input_dir {DATA_PATH}/raw --output_json {DATA_PATH}/metadata/all_videos.json",
        mounts=[
            Mount(source=REPO_PATH_ON_HOST, target=REPO_PATH_IN_CONTAINER, type="bind")
        ],
        working_dir=REPO_PATH_IN_CONTAINER,
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        mount_tmp_dir=False,
    )

    # 2. Filter Task
    filter_videos = DockerOperator(
        task_id="filter_videos",
        image=WORKER_IMAGE,
        # Reads the output of the previous step
        command=f"pixi run python src/media_flow/tasks/filter.py --input_json {DATA_PATH}/metadata/all_videos.json --output_json {DATA_PATH}/metadata/filtered_videos.json --keyword Obama",
        mounts=[
            Mount(source=REPO_PATH_ON_HOST, target=REPO_PATH_IN_CONTAINER, type="bind")
        ],
        working_dir=REPO_PATH_IN_CONTAINER,
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        mount_tmp_dir=False,
    )

    # 3. Augment Task (The heavy lifter)
    augment_videos = DockerOperator(
        task_id="augment_videos",
        image=WORKER_IMAGE,
        # Reads the filtered list
        command=f"pixi run python src/media_flow/tasks/augment.py --input_json {DATA_PATH}/metadata/filtered_videos.json --output_dir {DATA_PATH}/processed",
        mounts=[
            Mount(source=REPO_PATH_ON_HOST, target=REPO_PATH_IN_CONTAINER, type="bind")
        ],
        environment={"RAY_LOG_TO_STDERR": "1"},  # Pass env vars needed for Ray
        working_dir=REPO_PATH_IN_CONTAINER,
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        mount_tmp_dir=False,
    )

    scan_videos >> filter_videos >> augment_videos
