CREATE TABLE IF NOT EXISTS video_metadata (
    video_id BIGSERIAL PRIMARY KEY,
    original_path VARCHAR(512) NOT NULL,
    filename VARCHAR(256) NOT NULL,
    duration_sec NUMERIC(8, 3),
    frame_rate NUMERIC(6, 3),
    width INTEGER,
    height INTEGER,
    codec_name VARCHAR(64),
    color_space VARCHAR(64),
    scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Ensure fast lookups and prevent duplicate entries for the same path
CREATE UNIQUE INDEX IF NOT EXISTS idx_original_path ON video_metadata (original_path);