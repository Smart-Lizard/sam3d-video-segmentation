import argparse
import os

import cv2
import numpy as np

from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together


def main():
    parser = argparse.ArgumentParser(description="Run SAM-3D-Body visualization on every frame of a video.")
    parser.add_argument("--input", required=True, help="Path to input video (e.g. videos/in.mp4)")
    parser.add_argument("--output", required=True, help="Path to output video (e.g. outputs/out.mp4)")
    parser.add_argument("--hf_repo_id", default="facebook/sam-3d-body-dinov3", help="HF model repo id")
    args = parser.parse_args()

    input_video_path = args.input
    output_video_path = args.output

    # Create output directory if needed
    out_dir = os.path.dirname(output_video_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Input video:  {input_video_path}")
    print(f"Output video: {output_video_path}")

    # ---- Load model ----
    print("Loading SAM-3D Body model...")
    estimator = setup_sam_3d_body(hf_repo_id=args.hf_repo_id)

    # ---- Open input video ----
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    ret, frame_bgr = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("No frames found in input video")

    frame_idx = 1

    # Process first frame
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    outputs = estimator.process_one_image(frame_rgb)
    rend_img = visualize_sample_together(frame_bgr, outputs, estimator.faces).astype(np.uint8)

    rend_h, rend_w = rend_img.shape[:2]
    print(f"Rendered frame size: {rend_w}x{rend_h} @ {fps} FPS")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (rend_w, rend_h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter for output: {output_video_path}")

    out.write(rend_img)

    # Process remaining frames
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        outputs = estimator.process_one_image(frame_rgb)
        rend_img = visualize_sample_together(frame_bgr, outputs, estimator.faces).astype(np.uint8)

        # Keep video dimensions consistent
        if rend_img.shape[0] != rend_h or rend_img.shape[1] != rend_w:
            rend_img = cv2.resize(rend_img, (rend_w, rend_h))

        out.write(rend_img)

        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    out.release()
    print(f"Done. Saved segmented video to: {output_video_path}")


if __name__ == "__main__":
    main()
