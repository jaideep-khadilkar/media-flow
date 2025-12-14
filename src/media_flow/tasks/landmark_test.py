import cv2
import mediapipe as mp
import json
import os
import sys

# Change this to point to a valid video file in your repo
TEST_VIDEO_PATH = "data/raw/RD_Radio1_000.mp4"
OUTPUT_DIR = "test_output"


def run_landmark_test(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(OUTPUT_DIR, f"{basename}_overlay.mp4")
    output_json_path = os.path.join(OUTPUT_DIR, f"{basename}_landmarks.json")

    print(f"Processing: {video_path}")
    print(f"Output Video: {output_video_path}")
    print(f"Output JSON: {output_json_path}")

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Initialize Face Mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video stream or file")
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize Video Writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        animation_data = {
            "video_path": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "frames": [],
        }

        frame_idx = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # MediaPipe works with RGB
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            # Draw the annotations on the image
            image.flags.writeable = True
            image_bgr = image  # OpenCV uses BGR

            frame_landmarks_data = []

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 1. Draw Mesh Overlay
                    mp_drawing.draw_landmarks(
                        image=image_bgr,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )

                    # 2. Draw Contours (Lips, Eyes, Eyebrows)
                    mp_drawing.draw_landmarks(
                        image=image_bgr,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                    )

                    # 3. Extract JSON Data (Normalized 0.0 - 1.0)
                    face_data = []
                    for lm in face_landmarks.landmark:
                        face_data.append(
                            {
                                "x": round(lm.x, 5),
                                "y": round(lm.y, 5),
                                "z": round(lm.z, 5),
                            }
                        )
                    frame_landmarks_data.append(face_data)

            # Store frame data
            animation_data["frames"].append(
                {"frame_idx": frame_idx, "faces": frame_landmarks_data}
            )

            # Write frame
            out.write(image_bgr)

            # Progress indicator
            if frame_idx % 50 == 0:
                print(f"Processed frame {frame_idx}/{total_frames}")

            frame_idx += 1

        cap.release()
        out.release()

        # Save JSON
        with open(output_json_path, "w") as f:
            json.dump(animation_data, f)

        print("\nTest Complete!")
        print(f"Verify the overlay video at: {output_video_path}")


if __name__ == "__main__":
    # You can pass a path argument, or default to the variable at the top
    target_video = sys.argv[1] if len(sys.argv) > 1 else TEST_VIDEO_PATH
    run_landmark_test(target_video)
