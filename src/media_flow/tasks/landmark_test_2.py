"""Overlay MediaPipe Face Mesh landmarks on a video using the *Tasks API* (MediaPipe 0.10.30).

Requirements
- pip install mediapipe==0.10.30 opencv-python
- Download a Tasks face-landmarker model file (e.g. `face_landmarker.task`) and set --model.

Usage
  python mediapipe_tasks_facemesh_overlay.py \
    --input input.mp4 --output output.mp4 \
    --model face_landmarker.task

Notes
- MediaPipe 0.10.30 Python package layout differs from older "solutions" versions.
  This script uses `mediapipe.tasks` (Tasks API) only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# A light-weight set of common Face Mesh-style connections (not the full tesselation).
# These index pairs follow the standard 468-landmark topology used by MediaPipe.
# If you want the full mesh triangles, you'd need the full FACEMESH_TESSELATION list.
FACE_OVAL: Sequence[Tuple[int, int]] = (
    (10, 338),
    (338, 297),
    (297, 332),
    (332, 284),
    (284, 251),
    (251, 389),
    (389, 356),
    (356, 454),
    (454, 323),
    (323, 361),
    (361, 288),
    (288, 397),
    (397, 365),
    (365, 379),
    (379, 378),
    (378, 400),
    (400, 377),
    (377, 152),
    (152, 148),
    (148, 176),
    (176, 149),
    (149, 150),
    (150, 136),
    (136, 172),
    (172, 58),
    (58, 132),
    (132, 93),
    (93, 234),
    (234, 127),
    (127, 162),
    (162, 21),
    (21, 54),
    (54, 103),
    (103, 67),
    (67, 109),
    (109, 10),
)

LIPS: Sequence[Tuple[int, int]] = (
    (61, 146),
    (146, 91),
    (91, 181),
    (181, 84),
    (84, 17),
    (17, 314),
    (314, 405),
    (405, 321),
    (321, 375),
    (375, 291),
    (291, 308),
    (308, 61),
    (78, 95),
    (95, 88),
    (88, 178),
    (178, 87),
    (87, 14),
    (14, 317),
    (317, 402),
    (402, 318),
    (318, 324),
    (324, 308),
    (78, 61),
)

LEFT_EYE: Sequence[Tuple[int, int]] = (
    (33, 7),
    (7, 163),
    (163, 144),
    (144, 145),
    (145, 153),
    (153, 154),
    (154, 155),
    (155, 133),
    (133, 173),
    (173, 157),
    (157, 158),
    (158, 159),
    (159, 160),
    (160, 161),
    (161, 246),
    (246, 33),
)

RIGHT_EYE: Sequence[Tuple[int, int]] = (
    (263, 249),
    (249, 390),
    (390, 373),
    (373, 374),
    (374, 380),
    (380, 381),
    (381, 382),
    (382, 362),
    (362, 398),
    (398, 384),
    (384, 385),
    (385, 386),
    (386, 387),
    (387, 388),
    (388, 466),
    (466, 263),
)

LEFT_EYEBROW: Sequence[Tuple[int, int]] = (
    (70, 63),
    (63, 105),
    (105, 66),
    (66, 107),
    (107, 55),
    (55, 65),
    (65, 52),
    (52, 53),
    (53, 46),
)

RIGHT_EYEBROW: Sequence[Tuple[int, int]] = (
    (336, 296),
    (296, 334),
    (334, 293),
    (293, 300),
    (300, 276),
    (276, 283),
    (283, 282),
    (282, 295),
    (295, 285),
)

CONNECTION_GROUPS: Sequence[Sequence[Tuple[int, int]]] = (
    FACE_OVAL,
    LIPS,
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_EYEBROW,
    RIGHT_EYEBROW,
)


@dataclass
class DrawConfig:
    draw_points: bool = True
    draw_connections: bool = True
    point_radius: int = 1
    point_thickness: int = -1
    line_thickness: int = 1


def _norm_to_px(lm_x: float, lm_y: float, width: int, height: int) -> Tuple[int, int]:
    # Landmarks from FaceLandmarker are normalized [0,1]. Clamp to image bounds.
    x = int(round(lm_x * width))
    y = int(round(lm_y * height))
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def draw_facemesh_overlay(
    frame_bgr: np.ndarray,
    landmarks: Sequence,  # list of NormalizedLandmark
    cfg: DrawConfig,
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    # Precompute pixel coords for speed.
    pts: List[Tuple[int, int]] = [_norm_to_px(lm.x, lm.y, w, h) for lm in landmarks]

    if cfg.draw_connections:
        # Green lines
        for group in CONNECTION_GROUPS:
            for a, b in group:
                if a < len(pts) and b < len(pts):
                    cv2.line(
                        out,
                        pts[a],
                        pts[b],
                        (0, 255, 0),
                        cfg.line_thickness,
                        cv2.LINE_AA,
                    )

    if cfg.draw_points:
        # Red points
        for x, y in pts:
            cv2.circle(
                out,
                (x, y),
                cfg.point_radius,
                (0, 0, 255),
                cfg.point_thickness,
                cv2.LINE_AA,
            )

    return out


def build_face_landmarker(
    model_path: str, num_faces: int = 1
) -> mp_vision.FaceLandmarker:
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=num_faces,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def process_video(
    input_path: str,
    output_path: str,
    model_path: str,
    cfg: DrawConfig,
    num_faces: int = 1,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use mp4v for broad compatibility; change if you prefer h264.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {output_path}")

    landmarker = build_face_landmarker(model_path=model_path, num_faces=num_faces)

    # For VIDEO mode, MediaPipe expects timestamps in milliseconds, strictly increasing.
    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int(round((frame_idx / fps) * 1000.0))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            # Overlay the first detected face (or loop over all faces if you want).
            # result.face_landmarks is a list[ list[NormalizedLandmark] ]
            for face_lms in result.face_landmarks:
                frame_bgr = draw_facemesh_overlay(frame_bgr, face_lms, cfg)

        writer.write(frame_bgr)
        frame_idx += 1

    landmarker.close()
    writer.release()
    cap.release()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to input video (e.g. mp4)")
    p.add_argument("--output", required=True, help="Path to output video (e.g. mp4)")
    p.add_argument("--model", required=True, help="Path to face_landmarker.task")
    p.add_argument("--num-faces", type=int, default=1, help="Max faces to detect")
    p.add_argument(
        "--no-points", action="store_true", help="Disable drawing landmark points"
    )
    p.add_argument(
        "--no-lines", action="store_true", help="Disable drawing connections"
    )
    p.add_argument("--point-radius", type=int, default=1)
    p.add_argument("--line-thickness", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DrawConfig(
        draw_points=not args.no_points,
        draw_connections=not args.no_lines,
        point_radius=args.point_radius,
        line_thickness=args.line_thickness,
    )
    process_video(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        cfg=cfg,
        num_faces=args.num_faces,
    )


if __name__ == "__main__":
    main()
