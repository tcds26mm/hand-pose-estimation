import os
import time

import cv2
import torch
from ultralytics import YOLO

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def record_inference_run(weights_path: str, output_filename: str) -> None:
    """Runs live camera tracking and automatically records an 8-second sample video."""
    if os.path.exists(weights_path):
        print(f"Loading custom optimized model weights from: {weights_path}")
        model = YOLO(weights_path)
    else:
        print(f"Custom weights absent at '{weights_path}'. Loading generic base pretrained model...")
        model = YOLO("yolo26n-pose.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("CRITICAL ERROR: Cannot interface with webcam hardware index 0.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filename, fourcc, 30, (frame_width, frame_height))

    print(f"VIDEO RECORDER STARTED -> Target Path: {output_filename}")
    print(f"Active device: {device.upper()}")
    print("Recording will automatically close after 8 seconds. Press 'q' to abort early...")

    start_time = time.time()
    while (time.time() - start_time) < 8.0:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb_frame, imgsz=640, device=device, verbose=False, conf=0.25)

        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        cv2.imshow("YOLOv26 Live Stream Video Recorder", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"RECORDING COMPLETE: File saved successfully to {output_filename}\n")


if __name__ == "__main__":
    record_inference_run(
        weights_path="./runs/pose/train/weights/best.pt",
        output_filename="./runs/pose/predict/failed_attempt.mp4",
    )
