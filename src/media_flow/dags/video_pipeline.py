from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# Constants
REPO_PATH_ON_HOST = "C:/Users/iamja/Documents/GitHub/media-flow"
REPO_PATH_IN_CONTAINER = "/app"
DATA_PATH = "/app/data"  # Shared data location for scan output (still needed for now)

# The image that contains your dependencies (Pixi, Ray, OpenCV, FFprobe, psycopg2)
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

    # NOTE: Assume a separate setup DAG has created the 'video_metadata' table.

    # --- 1. Scan Task (Same as before, generates all_videos.json) ---
    scan_videos = DockerOperator(
        task_id="scan_videos",
        image=WORKER_IMAGE,
        command=f"pixi run python src/media_flow/tasks/scan.py --input_dir {DATA_PATH}/raw --output_json {DATA_PATH}/metadata/all_videos.json",
        mounts=[
            Mount(source=REPO_PATH_ON_HOST, target=REPO_PATH_IN_CONTAINER, type="bind")
        ],
        working_dir=REPO_PATH_IN_CONTAINER,
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",  # Needed if the scanner task writes to a network path
    )

    # --- 2. Extract Metadata Task (New: FFprobe & DB Insert) ---
    extract_video_metadata = DockerOperator(
        task_id="extract_video_metadata",
        image=WORKER_IMAGE,
        # Reads the JSON output of 'scan_videos' and uses the paths for Ray jobs
        command=f"pixi run python src/media_flow/tasks/extract_video_metadata.py --input_json {DATA_PATH}/metadata/all_videos.json",
        mounts=[
            Mount(source=REPO_PATH_ON_HOST, target=REPO_PATH_IN_CONTAINER, type="bind")
        ],
        mount_tmp_dir=False,
        working_dir=REPO_PATH_IN_CONTAINER,
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        # CRITICAL: Use the Docker bridge network to resolve the 'postgres' service
        network_mode="media-flow_default",
        environment={
            # Pass DB connection details from the Airflow Connection 'postgres_default'
            "RAY_LOG_TO_STDERR": "1",
            "DB_HOST": "{{ conn.postgres_default.host }}",
            "DB_NAME": "{{ conn.postgres_default.schema }}",
            "DB_USER": "{{ conn.postgres_default.login }}",
            "DB_PASSWORD": "{{ conn.postgres_default.password }}",
        },
    )

    # --- 3. Filter Task (Modified to read from DB) ---
    # NOTE: The filter task logic (filter.py) must be updated to query the
    # 'video_metadata' table based on resolution, FPS, or other columns,
    # instead of processing the 'all_videos.json' file directly.
    filter_videos = DockerOperator(
        task_id="filter_videos",
        image=WORKER_IMAGE,
        # Hypothetical command now querying the DB and outputting a filtered list
        command=f"pixi run python src/media_flow/tasks/filter.py --min_width 720 --output_json {DATA_PATH}/metadata/filtered_videos.json",
        mounts=[
            Mount(source=REPO_PATH_ON_HOST, target=REPO_PATH_IN_CONTAINER, type="bind")
        ],
        working_dir=REPO_PATH_IN_CONTAINER,
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",  # Needed for DB connection if filter.py uses DB to query
    )

    # --- 4. Augment Task (No change) ---
    augment_videos = DockerOperator(
        task_id="augment_videos",
        image=WORKER_IMAGE,
        command=f"pixi run python src/media_flow/tasks/augment.py --input_json {DATA_PATH}/metadata/filtered_videos.json --output_dir {DATA_PATH}/processed",
        mounts=[
            Mount(source=REPO_PATH_ON_HOST, target=REPO_PATH_IN_CONTAINER, type="bind")
        ],
        environment={"RAY_LOG_TO_STDERR": "1"},
        working_dir=REPO_PATH_IN_CONTAINER,
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
    )

    # --- Updated Pipeline Flow ---
    # pylint: disable=pointless-statement
    scan_videos >> extract_video_metadata >> filter_videos >> augment_videos
