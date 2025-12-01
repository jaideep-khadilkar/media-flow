from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Define the local folder where the DAG will be placed
AIRFLOW_HOME = "/opt/airflow"
REPO_PATH_IN_AIRFLOW = "/usr/local/airflow_repo"
IMAGE_NAME = "media-flow"  # The name of the Docker image you built

with DAG(
    dag_id="media_flow_ray_pipeline",
    start_date=datetime(2025, 11, 30),
    schedule_interval=timedelta(minutes=10),
    catchup=False,
    default_args={
        "owner": "data_engineer",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["media", "ray", "data-pipeline"],
) as dag:

    # 1. Initialization Step
    # Ensures the necessary data/videos folder exists on the shared volume
    # This task runs inside the Airflow worker environment (which has the volume mount)
    create_dirs = BashOperator(
        task_id="create_data_directories",
        bash_command=f"mkdir -p {REPO_PATH_IN_AIRFLOW}/data/videos",
    )

    # 2. Main Processing Task
    # This uses the DockerOperator to run your specialized 'media-flow' image.
    # The command is set to run your pixi 'start' task.
    run_ray_pipeline = DockerOperator(
        task_id="run_parallel_augmentation",
        image=IMAGE_NAME,
        command="pixi run start",
        # Set environment variables for the Ray Worker (your image)
        environment={
            "RAY_ADDRESS": "auto",
            "RAY_LOG_TO_STDERR": "1",
            # Add other necessary env vars for your Ray/OpenCV tasks here
        },
        # Mount the local repo root so the Ray container can access main.py and the data folder
        volumes=[f"{REPO_PATH_IN_AIRFLOW}:{REPO_PATH_IN_AIRFLOW}:rw"],
        # Set the working directory to the repo path inside the container
        working_dir=REPO_PATH_IN_AIRFLOW,
        # Remove the container after the task finishes
        auto_remove=True,
        # The Docker connection ID (requires Docker provider installed in Airflow)
        docker_conn_id="docker_default",
        # Specify network to allow communication with other Docker services if needed
        network_mode="media-flow-network",
        mount_tmp_dir=False,
    )

    # Define the simple dependency flow
    create_dirs >> run_ray_pipeline
