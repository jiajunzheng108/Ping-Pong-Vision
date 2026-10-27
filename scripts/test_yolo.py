import argparse
import time
from collections import deque

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO-only ping pong ball test")
    parser.add_argument("--source", default="Data/test2.mp4", help="input video")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights")
    parser.add_argument("--output", default="Data/yolo_test.mp4", help="output video")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--classes", default="32", help="comma separated class ids, empty = all")
    parser.add_argument("--display", action="store_true", help="show preview window")
    return parser.parse_args()


def main():
    args = parse_args()
    classes = [int(c) for c in args.classes.split(",") if c.strip()] or None

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    trajectory = deque(maxlen=30)
    frame_count = 0
    detected = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf, classes=classes, verbose=False)[0]

        if len(results.boxes) > 0:
            detected += 1
            box = results.boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            trajectory.append((cx, cy))

            cls_id = int(box.cls[0].cpu().item())
            label = f"{results.names.get(cls_id, cls_id)} {float(box.conf[0]):.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        points = list(trajectory)
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], (255, 255, 0), 2)

        frame_count += 1
        writer.write(frame)

        if args.display:
            cv2.imshow("YOLO test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed = max(time.time() - start, 1e-6)
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"frames: {frame_count}, detected: {detected}")
    print(f"avg fps: {frame_count / elapsed:.2f}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
