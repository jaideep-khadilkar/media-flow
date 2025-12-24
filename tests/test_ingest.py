# pylint: disable=missing-module-docstring, missing-function-docstring, redefined-outer-name
import json
import os
import sys
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from omegaconf import DictConfig

# Mock ray before importing the module to prevent @ray.remote decorator issues
sys.modules["ray"] = MagicMock()

from media_flow.tasks.ingest import (
    _discover_videos,
    _extract_single,
    _get_pending_videos,
    _populate_db,
    _register_videos,
    _run_distributed_extraction,
    ingest_pipeline,
)


@pytest.fixture
def mock_db_env(monkeypatch):
    """Set up database environment variables."""
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_NAME", "testdb")
    monkeypatch.setenv("DB_USER", "testuser")
    monkeypatch.setenv("DB_PASSWORD", "testpass")


@pytest.fixture
def sample_ffprobe_output():
    """Sample ffprobe JSON output."""
    return {
        "streams": [
            {
                "duration": "120.5",
                "r_frame_rate": "30000/1001",
                "width": 1920,
                "height": 1080,
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
            }
        ]
    }


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return DictConfig(
        {
            "ingest": {
                "input_dir": "/test/videos",
                "video_extensions": [".mp4", ".mov"],
                "max_videos": None,
                "metadata_extraction_batch_size": 30,
            }
        }
    )


class TestDiscoverVideos:
    def test_discover_videos_success(self, tmp_path):
        """Test successful video discovery."""
        # Create test video files
        (tmp_path / "video1.mp4").touch()
        (tmp_path / "video2.mp4").touch()
        (tmp_path / "video3.mov").touch()
        (tmp_path / "other.txt").touch()

        result = _discover_videos(str(tmp_path), [".mp4", ".mov"])

        assert len(result) == 3
        assert all(f.endswith((".mp4", ".mov")) for f in result)

    def test_discover_videos_with_limit(self, tmp_path):
        """Test video discovery with max_videos limit."""
        for i in range(5):
            (tmp_path / f"video{i}.mp4").touch()

        result = _discover_videos(str(tmp_path), [".mp4"], max_videos=3)

        assert len(result) == 3

    def test_discover_videos_nonexistent_dir(self):
        """Test discovery with non-existent directory."""
        with pytest.raises(SystemExit):
            _discover_videos("/nonexistent/path", [".mp4"])

    def test_discover_videos_no_matches(self, tmp_path):
        """Test discovery when no videos match extensions."""
        (tmp_path / "document.txt").touch()
        (tmp_path / "image.jpg").touch()

        result = _discover_videos(str(tmp_path), [".mp4"])

        assert len(result) == 0


class TestExtractSingle:
    @patch("media_flow.tasks.ingest.subprocess.run")
    def test_extract_single_success(self, mock_run, sample_ffprobe_output):
        """Test successful metadata extraction from a single video."""
        mock_run.return_value = Mock(
            stdout=json.dumps(sample_ffprobe_output), returncode=0
        )

        result = _extract_single("/path/to/video.mp4")

        assert result is not None
        assert result["original_path"] == "/path/to/video.mp4"
        assert result["duration_sec"] == 120.5
        assert result["width"] == 1920
        assert result["height"] == 1080
        assert result["codec_name"] == "h264"
        assert result["color_space"] == "yuv420p"
        assert result["frame_rate"] == pytest.approx(29.97, rel=0.01)

    @patch("media_flow.tasks.ingest.subprocess.run")
    def test_extract_single_no_streams(self, mock_run):
        """Test extraction when ffprobe returns no streams."""
        mock_run.return_value = Mock(stdout=json.dumps({"streams": []}), returncode=0)

        result = _extract_single("/path/to/video.mp4")

        assert result is None

    @patch("media_flow.tasks.ingest.subprocess.run")
    def test_extract_single_subprocess_error(self, mock_run):
        """Test extraction when subprocess fails."""
        mock_run.side_effect = Exception("FFprobe error")

        result = _extract_single("/path/to/video.mp4")

        assert result is None

    @patch("media_flow.tasks.ingest.subprocess.run")
    def test_extract_single_invalid_fps(self, mock_run):
        """Test extraction with invalid frame rate."""
        output = {"streams": [{"duration": "10", "r_frame_rate": "invalid"}]}
        mock_run.return_value = Mock(stdout=json.dumps(output), returncode=0)

        result = _extract_single("/path/to/video.mp4")

        assert result is not None
        assert result["frame_rate"] == 0


class TestRegisterVideos:
    @patch("media_flow.tasks.ingest._get_db_connection")
    def test_register_videos_success(self, mock_get_conn, mock_db_env):
        """Test successful video registration."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        video_files = ["/path/video1.mp4", "/path/video2.mp4"]
        _register_videos(video_files)

        mock_cursor.executemany.assert_called_once()
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("media_flow.tasks.ingest._get_db_connection")
    def test_register_videos_empty_list(self, mock_get_conn, mock_db_env):
        """Test registration with empty video list."""
        _register_videos([])

        mock_get_conn.assert_not_called()

    @patch("media_flow.tasks.ingest._get_db_connection")
    def test_register_videos_db_error(self, mock_get_conn, mock_db_env):
        """Test registration with database error."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.executemany.side_effect = Exception("DB Error")
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        with pytest.raises(Exception):
            _register_videos(["/path/video.mp4"])

        mock_conn.close.assert_called_once()


class TestGetPendingVideos:
    @patch("media_flow.tasks.ingest._get_db_connection")
    def test_get_pending_videos_success(self, mock_get_conn, mock_db_env):
        """Test retrieving pending videos."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, "/path/video1.mp4"),
            (2, "/path/video2.mp4"),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        result = _get_pending_videos()

        assert len(result) == 2
        assert result["/path/video1.mp4"] == 1
        assert result["/path/video2.mp4"] == 2
        mock_conn.close.assert_called_once()

    @patch("media_flow.tasks.ingest._get_db_connection")
    def test_get_pending_videos_empty(self, mock_get_conn, mock_db_env):
        """Test when no pending videos exist."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        result = _get_pending_videos()

        assert len(result) == 0


class TestRunDistributedExtraction:
    @patch("media_flow.tasks.ingest.ray")
    @patch("media_flow.tasks.ingest._extract_metadata_batch")
    def test_run_distributed_extraction_success(self, mock_batch, mock_ray):
        """Test distributed extraction with Ray."""
        mock_ray.is_initialized.return_value = False
        mock_ray.get.return_value = [
            [{"original_path": "/path/video1.mp4", "duration_sec": 10}],
            [{"original_path": "/path/video2.mp4", "duration_sec": 20}],
        ]
        mock_batch.remote.return_value = "future"

        paths = [f"/path/video{i}.mp4" for i in range(50)]
        result = _run_distributed_extraction(paths, batch_size=30)

        mock_ray.init.assert_called_once()
        assert len(result) == 2
        mock_ray.shutdown.assert_called_once()

    @patch("media_flow.tasks.ingest.ray")
    def test_run_distributed_extraction_empty(self, mock_ray):
        """Test extraction with empty paths list."""
        result = _run_distributed_extraction([], batch_size=30)

        assert len(result) == 0
        mock_ray.init.assert_not_called()

    @patch("media_flow.tasks.ingest.ray")
    @patch("media_flow.tasks.ingest._extract_metadata_batch")
    def test_run_distributed_extraction_filters_none(self, mock_batch, mock_ray):
        """Test that None results are filtered out."""
        mock_ray.is_initialized.return_value = False
        mock_ray.get.return_value = [
            [{"original_path": "/path/video1.mp4"}, None, None]
        ]
        mock_batch.remote.return_value = "future"

        result = _run_distributed_extraction(["/path/video1.mp4"], batch_size=30)

        assert len(result) == 1
        assert result[0]["original_path"] == "/path/video1.mp4"


class TestPopulateDb:
    @patch("media_flow.tasks.ingest._get_db_connection")
    def test_populate_db_success(self, mock_get_conn, mock_db_env):
        """Test successful database population."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        metadata = [
            {
                "original_path": "/path/video1.mp4",
                "duration_sec": 120.5,
                "frame_rate": 29.97,
                "width": 1920,
                "height": 1080,
                "codec_name": "h264",
                "color_space": "yuv420p",
            }
        ]
        id_map = {"/path/video1.mp4": 1}

        _populate_db(metadata, id_map)

        mock_cursor.executemany.assert_called_once()
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("media_flow.tasks.ingest._get_db_connection")
    def test_populate_db_empty_metadata(self, mock_get_conn, mock_db_env):
        """Test population with empty metadata list."""
        _populate_db([], {})

        mock_get_conn.assert_not_called()

    @patch("media_flow.tasks.ingest._get_db_connection")
    def test_populate_db_missing_id(self, mock_get_conn, mock_db_env):
        """Test population when video ID is not in map."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        metadata = [{"original_path": "/path/video1.mp4", "duration_sec": 10}]
        id_map = {}  # Empty map

        _populate_db(metadata, id_map)

        # Should not call executemany since no valid IDs
        call_args = mock_cursor.executemany.call_args
        assert len(call_args[0][1]) == 0


class TestIngestPipeline:
    @patch("media_flow.tasks.ingest._populate_db")
    @patch("media_flow.tasks.ingest._run_distributed_extraction")
    @patch("media_flow.tasks.ingest._get_pending_videos")
    @patch("media_flow.tasks.ingest._register_videos")
    @patch("media_flow.tasks.ingest._discover_videos")
    def test_ingest_pipeline_full_flow(
        self,
        mock_discover,
        mock_register,
        mock_pending,
        mock_extract,
        mock_populate,
        sample_config,
        mock_db_env,
    ):
        """Test complete ingest pipeline."""
        mock_discover.return_value = ["/path/video1.mp4", "/path/video2.mp4"]
        mock_pending.return_value = {"/path/video1.mp4": 1}
        mock_extract.return_value = [
            {"original_path": "/path/video1.mp4", "duration_sec": 10}
        ]

        ingest_pipeline(sample_config)

        mock_discover.assert_called_once()
        mock_register.assert_called_once()
        mock_pending.assert_called_once()
        mock_extract.assert_called_once()
        mock_populate.assert_called_once()

    @patch("media_flow.tasks.ingest._discover_videos")
    def test_ingest_pipeline_no_files(self, mock_discover, sample_config, mock_db_env):
        """Test pipeline when no files are discovered."""
        mock_discover.return_value = []

        ingest_pipeline(sample_config)

        mock_discover.assert_called_once()

    @patch("media_flow.tasks.ingest._get_pending_videos")
    @patch("media_flow.tasks.ingest._register_videos")
    @patch("media_flow.tasks.ingest._discover_videos")
    def test_ingest_pipeline_no_pending(
        self, mock_discover, mock_register, mock_pending, sample_config, mock_db_env
    ):
        """Test pipeline when all videos already processed."""
        mock_discover.return_value = ["/path/video1.mp4"]
        mock_pending.return_value = {}

        ingest_pipeline(sample_config)

        mock_discover.assert_called_once()
        mock_register.assert_called_once()
        mock_pending.assert_called_once()
