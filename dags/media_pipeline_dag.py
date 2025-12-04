from docker.types import Mount
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

AIRFLOW_HOME = "/opt/airflow"
REPO_PATH_ON_HOST = "C:/Users/iamja/Documents/GitHub/media-flow"
REPO_PATH_IN_AIRFLOW = "/usr/local/airflow_repo"
IMAGE_NAME = "media-flow:1.0"

with DAG(
    dag_id="media_flow_ray_pipeline",
    start_date=datetime(2025, 11, 1),
    schedule_interval=timedelta(days=1),
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

    create_dirs = BashOperator(
        task_id="create_data_directories",
        bash_command=f"mkdir -p {REPO_PATH_IN_AIRFLOW}/data/videos",
    )

    run_ray_pipeline = DockerOperator(
        task_id="run_parallel_augmentation",
        image=IMAGE_NAME,
        command="pixi run python main.py",
        mounts=[
            Mount(
                source=REPO_PATH_ON_HOST,  # host path (as seen from the Airflow container)
                target=REPO_PATH_IN_AIRFLOW,  # path inside the task container
                type="bind",
                read_only=False,
            )
        ],
        environment={
            # "RAY_ADDRESS": "auto",
            "RAY_LOG_TO_STDERR": "1",
        },
        working_dir=REPO_PATH_IN_AIRFLOW,
        auto_remove=True,
        docker_conn_id=None,
        docker_url="unix://var/run/docker.sock",  # optional, explicit
        # network_mode="media-flow-net",
        mount_tmp_dir=False,
    )

    create_dirs >> run_ray_pipeline
