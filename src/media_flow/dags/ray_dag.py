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
    dag_id="ray_dag",
    start_date=datetime(2025, 11, 1),
    schedule=None,  # Run manually or via API only
    catchup=False,
    default_args=default_args,
) as dag:

    ray_task = DockerOperator(
        task_id="ray_task",
        image=WORKER_IMAGE,
        # Output directory is needed to save the augmented files
        command="pixi run python src/media_flow/tasks/ray_task.py",
        mounts=SHARED_MOUNTS,
        mount_tmp_dir=False,
        environment=DB_ENV_VARS,
        working_dir=REPO_PATH_IN_CONTAINER,
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="media-flow_default",
    )
