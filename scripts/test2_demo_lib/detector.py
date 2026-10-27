from pathlib import Path


def parse_classes(value):
    if not value:
        return None
    classes = [int(x) for x in value.split(',') if x.strip()]
    return classes or None


def load_model(path):
    from ultralytics import YOLO

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError('model not found: {}'.format(path))
    return YOLO(str(path))


def detect_ball(model, frame, conf=0.25, classes=None, imgsz=960):
    """Return (x1, y1, x2, y2, confidence) for the best ball box, or None."""
    results = model(frame, conf=conf, classes=classes, imgsz=imgsz, verbose=False)[0]
    if len(results.boxes) == 0:
        return None

    boxes = results.boxes
    best = int(boxes.conf.argmax())
    x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy().tolist()
    best_conf = float(boxes.conf[best].cpu().item())
    return (x1, y1, x2, y2, best_conf)
