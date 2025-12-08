from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator

POSTGRES_CONN_ID = "postgres_default"

SQL_CREATE_VIDEO_METADATA = """
CREATE TABLE IF NOT EXISTS video_metadata (
    video_id BIGSERIAL PRIMARY KEY,
    original_path VARCHAR(512) NOT NULL UNIQUE,
    filename VARCHAR(256) NOT NULL,
    duration_sec NUMERIC(8, 3),
    frame_rate NUMERIC(6, 3),
    width INTEGER,
    height INTEGER,
    codec_name VARCHAR(64),
    color_space VARCHAR(64),
    is_quality_video BOOLEAN DEFAULT FALSE, -- Used by filter.py
    scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_original_path ON video_metadata (original_path);
"""

SQL_CREATE_AUGMENTED_VIDEOS = """
CREATE TABLE IF NOT EXISTS augmented_videos (
    augmentation_id BIGSERIAL PRIMARY KEY,
    video_id BIGINT NOT NULL,
    augmented_path VARCHAR(512) NOT NULL,
    augmentation_type VARCHAR(128) NOT NULL,
    parameters_used JSONB,
    status VARCHAR(50) DEFAULT 'CREATED',
    timestamp_processed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Link to the original video
    CONSTRAINT fk_video
        FOREIGN KEY(video_id) 
        REFERENCES video_metadata(video_id)
        ON DELETE CASCADE,
        
    -- Ensure we don't augment the same video with the same path/type twice
    UNIQUE (video_id, augmented_path)
);
"""

with DAG(
    dag_id="db_schema_setup_dag",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule=None,  # Run manually or via API only
    catchup=False,
    tags=["setup"],
) as dag:

    create_video_metadata_table = PostgresOperator(
        task_id="create_video_metadata_table",
        postgres_conn_id=POSTGRES_CONN_ID,
        sql=SQL_CREATE_VIDEO_METADATA,
    )

    create_augmented_videos_table = PostgresOperator(
        task_id="create_augmented_videos_table",
        postgres_conn_id=POSTGRES_CONN_ID,
        sql=SQL_CREATE_AUGMENTED_VIDEOS,
    )

    create_video_metadata_table >> create_augmented_videos_table
