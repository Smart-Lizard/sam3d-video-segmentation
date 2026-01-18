import cv2
import numpy as np

from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together


def main():
    # ---- 1. Paths ----
    input_video_path = "videos/test_input.mp4"      # change if your file name is different
    output_video_path = "videos/test_output_sam3d.mp4"

    # ---- 2. Load the SAM-3D Body estimator ----
    print("Loading SAM-3D Body (dinov3) model...")
    estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")

    # ---- 3. Open input video ----
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # fallback

    print(f"Video FPS: {fps}")

    # ---- 4. Read *first* frame and process it ----
    ret, frame_bgr = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("No frames found in input video")

    frame_idx = 1

    # Model expects RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    outputs = estimator.process_one_image(frame_rgb)

    # This returns the combined visualization (original + mesh etc.)
    rend_img = visualize_sample_together(frame_bgr, outputs, estimator.faces)
    rend_img = rend_img.astype(np.uint8)

    # Use the rendered frame size for the video writer
    rend_h, rend_w = rend_img.shape[:2]
    print(f"Rendered frame size: {rend_w}x{rend_h}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (rend_w, rend_h))

    if not out.isOpened():
        cap.release()
        raise RuntimeError("Failed to open VideoWriter for output")

    # Write the first processed frame
    out.write(rend_img)

    # ---- 5. Process remaining frames ----
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break  # end of video

        frame_idx += 1

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        outputs = estimator.process_one_image(frame_rgb)
        rend_img = visualize_sample_together(frame_bgr, outputs, estimator.faces)
        rend_img = rend_img.astype(np.uint8)

        # Safety check: if any tiny size drift happens, force resize
        if rend_img.shape[0] != rend_h or rend_img.shape[1] != rend_w:
            rend_img = cv2.resize(rend_img, (rend_w, rend_h))

        out.write(rend_img)

        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames...")

    # ---- 6. Clean up ----
    cap.release()
    out.release()
    print(f"Done. Saved segmented video to: {output_video_path}")


if __name__ == "__main__":
    main()