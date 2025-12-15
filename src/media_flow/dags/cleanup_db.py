from __future__ import annotations

import pendulum
from airflow.models.dag import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator

POSTGRES_CONN_ID = "postgres_default"

with DAG(
    dag_id="cleanup_db",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule=None,  # Run manually or via API only
    catchup=False,
    tags=["setup"],
) as dag:

    delete_video_metadata_table = PostgresOperator(
        task_id="delete_video_metadata_table",
        postgres_conn_id=POSTGRES_CONN_ID,
        sql="DROP TABLE IF EXISTS video_metadata CASCADE;",
    )

    delete_augmented_videos_table = PostgresOperator(
        task_id="delete_augmented_videos_table",
        postgres_conn_id=POSTGRES_CONN_ID,
        sql="DROP TABLE IF EXISTS augmented_videos CASCADE;",
    )

    delete_video_landmarks_table = PostgresOperator(
        task_id="delete_video_landmarks_table",
        postgres_conn_id=POSTGRES_CONN_ID,
        sql="DROP TABLE IF EXISTS video_landmarks CASCADE;",
    )

    delete_processing_failures_table = PostgresOperator(
        task_id="delete_processing_failures_table",
        postgres_conn_id=POSTGRES_CONN_ID,
        sql="DROP TABLE IF EXISTS processing_failures CASCADE;",
    )
