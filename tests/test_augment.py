import json
import sys
import types

import numpy as np
from omegaconf import OmegaConf

# Ensure module import works and stub out ray before importing augment
# sys.path.insert(0, "/app/src")

fake_ray = types.SimpleNamespace()


# pylint: disable=unused-argument
def _fake_remote(**kwargs):
    def _decorator(fn):
        # Return function directly so we can call it in tests without Ray
        return fn

    return _decorator


fake_ray.remote = _fake_remote
fake_ray.init = lambda *a, **k: None
fake_ray.shutdown = lambda *a, **k: None

# Force the stubbed ray to be used by augment.py at import time
sys.modules["ray"] = fake_ray

from media_flow.tasks import augment


def test_set_seed_determinism():
    augment.set_seed(123)
    a = np.random.rand(5)
    augment.set_seed(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_prepare_output_path(tmp_path):
    out = augment.prepare_output_path(tmp_path, "sample_video.mp4")
    assert out.parent == tmp_path
    assert out.name == "sample_video.mp4"


def test_build_success_record():
    rec = augment.build_success_record(7, PathLike("/tmp/out.mp4"), '{"k":1}')
    assert rec["video_id"] == 7
    assert rec["augmented_path"].endswith("/tmp/out.mp4")
    assert rec["augmentation_type"] == "standard_dropout"
    assert rec["parameters_used"] == '{"k":1}'
    assert rec["status"] == "READY"


def test_create_video_writer_success(monkeypatch, tmp_path):
    class FakeWriter:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(augment.cv2, "VideoWriter", lambda *a, **k: FakeWriter())
    writer, err = augment.create_video_writer(tmp_path / "o.mp4", 30.0, 100, 100)
    assert writer is not None
    assert err is None


def test_create_video_writer_failure(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise RuntimeError("no codec")

    monkeypatch.setattr(augment.cv2, "VideoWriter", boom)
    writer, err = augment.create_video_writer(tmp_path / "o.mp4", 30.0, 100, 100)
    assert writer is None
    assert isinstance(err, Exception)


def test_augment_frame_with_dummy_augmenter():
    class DummyAug:
        def __call__(self, image):
            # Pass through
            return {"image": image}

    # Make a small BGR image
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[0, 0] = [10, 20, 30]
    out = augment.augment_frame(DummyAug(), bgr)
    assert out.shape == bgr.shape


def test_process_video_task_happy_path(monkeypatch, tmp_path):
    # Fake frames
    frame1 = np.zeros((4, 4, 3), dtype=np.uint8)
    frame2 = np.ones((4, 4, 3), dtype=np.uint8) * 255

    class FakeCap:
        # pylint: disable=unused-argument
        def __init__(self, *a, **k):
            self.frames = [frame1, frame2]
            self.idx = 0

        # pylint: disable=invalid-name
        def isOpened(self):
            return True

        def read(self):
            if self.idx < len(self.frames):
                f = self.frames[self.idx]
                self.idx += 1
                return True, f
            return False, None

        def release(self):
            pass

        def get(self, prop):
            if prop == augment.cv2.CAP_PROP_FPS:  # pylint: disable=no-member
                return 24.0
            return 0.0

    class FakeWriter:
        # pylint: disable=unused-argument
        def __init__(self, *a, **k):
            self.writes = []

        def write(self, frame):
            self.writes.append(frame)

        def release(self):
            pass

    # Patch OpenCV
    monkeypatch.setattr(augment.cv2, "VideoCapture", lambda *a, **k: FakeCap())
    monkeypatch.setattr(augment.cv2, "VideoWriter", lambda *a, **k: FakeWriter())

    # Make build_augmenter return a pass-through augmenter
    class DummyAug:
        def __call__(self, image):
            return {"image": image}

    monkeypatch.setattr(augment, "build_augmenter", lambda params, seed: DummyAug())

    out_path = tmp_path / "out.mp4"
    record = augment.process_video_task(
        video_id=42,
        video_path=str(tmp_path / "in.mp4"),
        output_path=str(out_path),
        augmentation_params=json.dumps({}),
        seed=123,
    )
    assert record is not None
    assert record["video_id"] == 42
    assert record["augmented_path"] == str(out_path)
    assert record["status"] == "READY"


def test_process_video_task_writer_failure(monkeypatch, tmp_path):
    class FakeCap:
        def __init__(self, *a, **k):
            pass

        # pylint: disable=invalid-name
        def isOpened(self):
            return True

        def read(self):
            return False, None

        def release(self):
            pass

        # pylint: disable=unused-argument
        def get(self, prop):
            return 30.0

    def boom_writer(*a, **k):
        raise RuntimeError("no writer")

    monkeypatch.setattr(augment.cv2, "VideoCapture", lambda *a, **k: FakeCap())
    monkeypatch.setattr(augment.cv2, "VideoWriter", boom_writer)
    # No need to patch augmenter since we fail before use
    res = augment.process_video_task(
        video_id=9,
        video_path=str(tmp_path / "in.mp4"),
        output_path=str(tmp_path / "out.mp4"),
        augmentation_params="{}",
        seed=1,
    )
    assert res is not None
    assert res["video_id"] == 9
    assert res["status"] == "ERROR"
    assert "Writer failed" in res["error_message"]


def test_fetch_videos_to_augment(monkeypatch):
    rows = [(1, "/p1.mp4", "f1.mp4"), (2, "/p2.mp4", "f2.mp4")]

    class FakeCursor:
        def __init__(self):
            self._rows = rows

        def execute(self, *a, **k):
            pass

        def fetchall(self):
            return self._rows

    class FakeConn:
        # pylint: disable=unused-argument
        def __init__(self, *a, **k):
            self.closed = False
            self._cursor = FakeCursor()

        def cursor(self):
            return self._cursor

        def close(self):
            self.closed = True

    monkeypatch.setenv("DB_HOST", "h")
    monkeypatch.setenv("DB_NAME", "n")
    monkeypatch.setenv("DB_USER", "u")
    monkeypatch.setenv("DB_PASSWORD", "p")
    monkeypatch.setattr(augment.psycopg2, "connect", lambda *a, **k: FakeConn())
    out = augment.fetch_videos_to_augment()
    assert out == rows


def test_insert_augmentation_record(monkeypatch):
    captured = {}

    class FakeCursor:
        def __init__(self):
            self.statements = []
            self.args = []

        def execute(self, stmt, params=None):
            self.statements.append(stmt)
            self.args.append(params)

    class FakeConn:
        # pylint: disable=unused-argument
        def __init__(self, *a, **k):
            self._cursor = FakeCursor()
            self._committed = False
            self._rolled = False
            self._closed = False

        def cursor(self):
            return self._cursor

        def commit(self):
            self._committed = True

        def rollback(self):
            self._rolled = True

        def close(self):
            self._closed = True

    # pylint: disable=unused-argument
    def fake_connect(*a, **k):
        c = FakeConn()
        captured["conn"] = c
        return c

    monkeypatch.setenv("DB_HOST", "h")
    monkeypatch.setenv("DB_NAME", "n")
    monkeypatch.setenv("DB_USER", "u")
    monkeypatch.setenv("DB_PASSWORD", "p")
    monkeypatch.setattr(augment.psycopg2, "connect", fake_connect)

    rec = {
        "video_id": 11,
        "augmented_path": "/tmp/o.mp4",
        "augmentation_type": "standard_dropout",
        "parameters_used": "{}",
        "status": "READY",
    }
    augment.insert_augmentation_record(rec)

    conn = captured["conn"]
    # pylint: disable=protected-access
    assert conn._committed is True
    stmt = conn._cursor.statements[0]
    args = conn._cursor.args[0]
    assert "INSERT INTO augmented_videos" in stmt
    assert args[0] == 11
    assert args[1] == "/tmp/o.mp4"
    assert args[2] == "standard_dropout"
    assert args[3] == "{}"
    assert args[4] == "READY"


def test_handle_results_calls_insert(monkeypatch):
    results = [
        {
            "video_id": 1,
            "augmented_path": "a",
            "augmentation_type": "t",
            "parameters_used": "{}",
            "status": "READY",
        },
        {
            "video_id": 2,
            "augmented_path": "b",
            "augmentation_type": "t",
            "parameters_used": "{}",
            "status": "READY",
        },
    ]

    monkeypatch.setattr(augment, "process_ray_results", lambda f, m, _: iter(results))
    called = []

    def fake_insert(rec):
        called.append(rec["video_id"])

    monkeypatch.setattr(augment, "insert_augmentation_record", fake_insert)
    augment.handle_results(futures=[], future_to_id={})
    assert called == [1, 2]


def test_submit_augment_tasks_builds_mapping(monkeypatch, tmp_path):
    # Patch process_video_task with a stub that has a .remote method
    class StubRemote:
        @staticmethod
        def remote(**kwargs):
            return f"fut_{kwargs['video_id']}"

    monkeypatch.setattr(augment, "process_video_task", StubRemote)
    videos = [(10, "/p10.mp4", "f10.mp4"), (20, "/p20.mp4", "f20.mp4")]
    futures, mapping = augment.submit_augment_tasks(
        videos, tmp_path, augmentation_settings="{}", seed=42
    )
    assert set(futures) == {"fut_10", "fut_20"}
    assert mapping["fut_10"] == 10
    assert mapping["fut_20"] == 20


def test_augment_pipeline_no_videos_short_circuits(monkeypatch, tmp_path):
    # Avoid env requirement and Ray init
    monkeypatch.setattr(augment, "ensure_db_credentials", lambda: None)
    monkeypatch.setattr(augment, "fetch_videos_to_augment", lambda: [])

    cfg = OmegaConf.create(
        {"augment": {"output_dir": str(tmp_path), "settings": {}, "seed": 7}}
    )

    # Should not raise and not initialize Ray
    called = {"init": 0}

    def fake_init():
        called["init"] += 1

    monkeypatch.setattr(augment, "initialize_ray", fake_init)
    augment.augment_pipeline(cfg)
    assert called["init"] == 0


# Helper to provide a lightweight Path-like object without importing pathlib here
class PathLike(str):
    pass
