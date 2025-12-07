from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator

# NOTE: Your docker-compose uses a connection named 'postgres' internally.
# Airflow will use this connection ID to connect to the DB.
POSTGRES_CONN_ID = "postgres_default"

with DAG(
    dag_id="setup_db_schema",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule=None,  # Run manually or via API only
    catchup=False,
    tags=["setup"],
) as dag:

    create_video_metadata_table = PostgresOperator(
        task_id="create_video_metadata_table",
        postgres_conn_id=POSTGRES_CONN_ID,
        sql="./sql/create_video_metadata_table.sql",  # Path is relative to the DAGs folder
    )

    # You could add other necessary setup tasks here, like creating other tables
    # create_video_metadata_table >> create_augmentation_table
