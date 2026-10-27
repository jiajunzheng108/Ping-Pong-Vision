import argparse
import json
import sys
import time
import zipfile
from collections import deque
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'prepare_dataset'))

from prepare_dataset.read_zip import load_ball_markup, load_events  # noqa: E402

MAP_W = 540
MAP_H = 360
TABLE_LEFT = 30
TABLE_TOP = 70
TABLE_RIGHT = 510
TABLE_BOTTOM = 337
MAX_TRAIL_GAP = 12


def order_corners(points):
    """Order 4 points as TL, TR, BL, BR for a roughly horizontal table."""
    points = sorted(points.tolist(), key=lambda p: p[1])
    top = sorted(points[:2], key=lambda p: p[0])
    bottom = sorted(points[2:], key=lambda p: p[0])
    return np.array([top[0], top[1], bottom[0], bottom[1]], dtype=np.float32)


def extract_table_corners(zip_path, video_width, video_height):
    """Derive the 4 table corners from the red table mask in the zip."""
    with zipfile.ZipFile(zip_path) as zf:
        mask_names = sorted(n for n in zf.namelist() if n.endswith('.png'))
        for name in mask_names:
            mask = cv2.imdecode(
                np.frombuffer(zf.read(name), np.uint8), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.ndim != 3:
                continue

            red = ((mask[:, :, 2] > 200) &
                   (mask[:, :, 1] < 50) &
                   (mask[:, :, 0] < 50)).astype(np.uint8) * 255
            if red.sum() < 100:
                continue

            contours, _ = cv2.findContours(
                red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)

            approx = None
            for epsilon in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2):
                candidate = cv2.approxPolyDP(hull, epsilon * perimeter, True)
                if len(candidate) == 4:
                    approx = candidate
                    break
            if approx is None:
                rect = cv2.minAreaRect(hull)
                approx = cv2.boxPoints(rect).reshape(-1, 1, 2)

            points = approx.reshape(-1, 2).astype(np.float32)
            if len(points) < 4:
                continue
            if len(points) > 4:
                points = points[:4]

            points[:, 0] *= video_width / mask.shape[1]
            points[:, 1] *= video_height / mask.shape[0]
            return order_corners(points)

    raise RuntimeError('could not extract table corners from segmentation masks')


def build_homography(corners):
    dst = np.float32([
        [TABLE_LEFT, TABLE_TOP],
        [TABLE_RIGHT, TABLE_TOP],
        [TABLE_LEFT, TABLE_BOTTOM],
        [TABLE_RIGHT, TABLE_BOTTOM],
    ])
    return cv2.getPerspectiveTransform(corners, dst)


def to_topdown(matrix, point):
    if point is None:
        return None
    src = np.array([[point]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, matrix)[0][0]
    x = int(min(max(dst[0], 0), MAP_W - 1))
    y = int(min(max(dst[1], 0), MAP_H - 1))
    return (x, y)


def create_base_map():
    canvas = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)
    canvas[:] = (40, 40, 40)

    left, top = TABLE_LEFT, TABLE_TOP
    right, bottom = TABLE_RIGHT, TABLE_BOTTOM
    cv2.rectangle(canvas, (left, top), (right, bottom), (0, 168, 255), -1)
    cv2.rectangle(canvas, (left, top), (right, bottom), (255, 255, 255), 2)

    center_y = (top + bottom) // 2
    center_x = (left + right) // 2
    cv2.line(canvas, (left, center_y), (right, center_y), (255, 255, 255), 1)
    cv2.line(canvas, (center_x, top), (center_x, bottom), (255, 255, 255), 4)

    cv2.putText(canvas, 'Top-down (z=0 approx)', (12, MAP_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    return canvas


def draw_trail(canvas, trail, color):
    points = list(trail)
    for i in range(1, len(points)):
        frame0, p0 = points[i - 1]
        frame1, p1 = points[i]
        if frame1 - frame0 > MAX_TRAIL_GAP:
            continue
        alpha = 0.25 + 0.75 * (i / len(points))
        c = tuple(int(ch * alpha) for ch in color)
        cv2.line(canvas, p0, p1, c, 2, cv2.LINE_AA)


def draw_minimap(frame, base_map, pred_trail, gt_trail, last_pred,
                 last_gt, bounce_marks, frame_idx, fps):
    canvas = base_map.copy()

    draw_trail(canvas, gt_trail, (0, 255, 0))
    draw_trail(canvas, pred_trail, (0, 200, 255))

    max_age = max(int(fps * 2), 1)
    for mark in bounce_marks:
        age = frame_idx - mark['frame']
        if age < 0 or age > max_age:
            continue
        alpha = 1.0 - age / max_age
        color = (0, 255, 255) if mark['kind'] == 'bounce' else (255, 0, 255)
        color = tuple(int(ch * alpha) for ch in color)
        cv2.circle(canvas, mark['pos'], 6, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, mark['kind'][:1].upper(),
                    (mark['pos'][0] + 8, mark['pos'][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    if last_gt is not None:
        cv2.circle(canvas, last_gt, 5, (0, 255, 0), -1, cv2.LINE_AA)
    if last_pred is not None:
        cv2.circle(canvas, last_pred, 5, (0, 0, 255), -1, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (MAP_W - 1, MAP_H - 1), (200, 200, 200), 1)

    frame_h, frame_w = frame.shape[:2]
    x0 = max(frame_w - MAP_W - 20, 0)
    y0 = 20
    if x0 + MAP_W <= frame_w and y0 + MAP_H <= frame_h:
        frame[y0:y0 + MAP_H, x0:x0 + MAP_W] = canvas
    return frame


def draw_overlay(frame, frame_idx, total_frames, gt, pred, distance, event,
                 recall, matched, gt_frames, fps):
    panel = frame.copy()
    cv2.rectangle(panel, (10, 10), (470, 150), (0, 0, 0), -1)
    frame = cv2.addWeighted(panel, 0.45, frame, 0.55, 0)

    lines = [
        'frame {}/{}  t={:.2f}s'.format(
            frame_idx, total_frames, frame_idx / max(fps, 1)),
        'pred conf: {}'.format('--' if pred is None else '{:.2f}'.format(pred[4])),
        'gt-pred: {}'.format('--' if distance is None else '{:.1f} px'.format(distance)),
        'recall: {}/{} ({:.1f}%)'.format(
            matched, gt_frames, 100.0 * recall if gt_frames else 0.0),
    ]
    y = 36
    for line in lines:
        cv2.putText(frame, line, (22, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        y += 28

    if gt is not None:
        cv2.circle(frame, (int(gt[0]), int(gt[1])), 7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, 'GT', (int(gt[0]) + 10, int(gt[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if pred is not None:
        x1, y1, x2, y2, conf = pred
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 0, 255), 2)
        cv2.circle(frame, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                   4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, 'ball {:.2f}'.format(conf),
                    (int(x1), max(int(y1) - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if gt is not None and pred is not None:
        pred_center = (int((pred[0] + pred[2]) / 2), int((pred[1] + pred[3]) / 2))
        cv2.line(frame, (int(gt[0]), int(gt[1])), pred_center,
                 (0, 255, 255), 2, cv2.LINE_AA)

    if event:
        text = event.upper()
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        x = (frame.shape[1] - tw) // 2
        cv2.putText(frame, text, (x, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 3)
    return frame


def main():
    parser = argparse.ArgumentParser(description='test_2 single-case demo')
    parser.add_argument('--video', default=str(PROJECT_ROOT / 'Data' / 'test_2.mp4'))
    parser.add_argument('--zip', default=str(PROJECT_ROOT / 'Data' / 'test_2.zip'))
    parser.add_argument('--model', default=str(
        PROJECT_ROOT / 'runs' / 'detect' / 'runs' / 'test2_finetune' / 'weights' / 'best.pt'))
    parser.add_argument('--output', default=str(PROJECT_ROOT / 'Data' / 'test2_demo.mp4'))
    parser.add_argument('--stats', default=str(PROJECT_ROOT / 'Data' / 'test2_demo_stats.json'))
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--classes', default='0', help='comma separated class ids')
    parser.add_argument('--match-threshold', type=float, default=50.0)
    parser.add_argument('--output-fps', type=float, default=30.0)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--show-corners', action='store_true')
    parser.add_argument('--display', action='store_true')
    args = parser.parse_args()

    from ultralytics import YOLO

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError('model not found: {}'.format(model_path))

    classes = [int(x) for x in args.classes.split(',') if x.strip()] or None

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError('cannot open video: {}'.format(args.video))

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    markup = load_ball_markup(args.zip)
    events = load_events(args.zip)

    corners = extract_table_corners(args.zip, width, height)
    homography = build_homography(corners)
    print('table corners (TL, TR, BL, BR):')
    print(corners)

    event_display = {}
    for frame, kind in events.items():
        if kind not in ('bounce', 'net'):
            continue
        for delta in range(-8, 17):
            event_display.setdefault(frame + delta, []).append(kind)

    base_map = create_base_map()
    pred_trail = deque(maxlen=120)
    gt_trail = deque(maxlen=120)
    bounce_marks = deque(maxlen=200)

    output_fps = args.output_fps if args.output_fps > 0 else fps
    write_step = max(int(round(fps / output_fps)), 1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*'mp4v'),
        output_fps, (width, height))

    model = YOLO(str(model_path))

    gt_frames = 0
    pred_frames = 0
    matched = 0
    fp_negative = 0
    errors = []
    matched_errors = []
    last_pred = None
    last_gt = None
    last_table_pred = None
    last_table_gt = None

    frame_idx = 0
    processed = 0
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames > 0 and processed >= args.max_frames:
            break

        ball = markup.get(frame_idx)
        gt = None
        is_negative = False
        if ball is not None:
            if ball['x'] >= 0 and ball['y'] >= 0:
                gt = (float(ball['x']), float(ball['y']))
            else:
                is_negative = True

        results = model(frame, conf=args.conf, classes=classes,
                        imgsz=args.imgsz, verbose=False)[0]
        pred = None
        if len(results.boxes) > 0:
            boxes = results.boxes
            best = int(boxes.conf.argmax())
            x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy().tolist()
            conf = float(boxes.conf[best].cpu().item())
            pred = (x1, y1, x2, y2, conf)

        distance = None
        if gt is not None:
            gt_frames += 1
            if pred is not None:
                center = ((pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0)
                distance = float(np.hypot(center[0] - gt[0], center[1] - gt[1]))
                if distance <= args.match_threshold:
                    matched += 1
                    matched_errors.append(distance)
                errors.append(distance)
        elif is_negative and pred is not None:
            fp_negative += 1

        if pred is not None:
            pred_frames += 1
            last_pred = ((pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0)
            last_table_pred = to_topdown(homography, last_pred)
            pred_trail.append((frame_idx, last_table_pred))

        if gt is not None:
            last_gt = gt
            last_table_gt = to_topdown(homography, gt)
            gt_trail.append((frame_idx, last_table_gt))

        if frame_idx in events and events[frame_idx] in ('bounce', 'net'):
            pos = None
            if last_table_pred is not None:
                pos = last_table_pred
            elif last_table_gt is not None:
                pos = last_table_gt
            if pos is not None:
                bounce_marks.append({
                    'frame': frame_idx,
                    'pos': pos,
                    'kind': events[frame_idx],
                })

        active_event = None
        if frame_idx in event_display:
            active_event = event_display[frame_idx][-1]

        recall = matched / gt_frames if gt_frames else 0.0
        frame = draw_overlay(frame, frame_idx, total_frames, gt, pred,
                             distance, active_event, recall, matched,
                             gt_frames, fps)
        frame = draw_minimap(frame, base_map, pred_trail, gt_trail,
                             last_table_pred, last_table_gt, bounce_marks,
                             frame_idx, fps)

        if args.show_corners:
            cv2.polylines(frame, [corners.astype(np.int32)], True,
                          (0, 255, 0), 2)

        if frame_idx % write_step == 0:
            writer.write(frame)

        if args.display:
            cv2.imshow('test_2 demo', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        processed += 1
        frame_idx += 1
        if processed % 300 == 0:
            elapsed = max(time.time() - start, 1e-6)
            print('processed {} frames, {:.1f} FPS'.format(
                processed, processed / elapsed))

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    elapsed = max(time.time() - start, 1e-6)
    if errors:
        errors_arr = np.array(errors)
        mean_error = float(errors_arr.mean())
        median_error = float(np.median(errors_arr))
        p90_error = float(np.percentile(errors_arr, 90))
    else:
        mean_error = median_error = p90_error = 0.0

    if matched_errors:
        matched_arr = np.array(matched_errors)
        mean_matched_error = float(matched_arr.mean())
        median_matched_error = float(np.median(matched_arr))
    else:
        mean_matched_error = median_matched_error = 0.0

    stats = {
        'video': str(Path(args.video).resolve()),
        'zip': str(Path(args.zip).resolve()),
        'model': str(model_path.resolve()),
        'output': str(output_path.resolve()),
        'table_corners': corners.tolist(),
        'conf': args.conf,
        'imgsz': args.imgsz,
        'match_threshold_px': args.match_threshold,
        'processed_frames': processed,
        'gt_frames': gt_frames,
        'pred_frames': pred_frames,
        'matched': matched,
        'missed': gt_frames - matched,
        'recall': recall,
        'fp_on_negative_frames': fp_negative,
        'mean_center_error_px': mean_error,
        'median_center_error_px': median_error,
        'p90_center_error_px': p90_error,
        'mean_matched_error_px': mean_matched_error,
        'median_matched_error_px': median_matched_error,
        'elapsed_sec': elapsed,
        'processed_fps': processed / elapsed,
    }

    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print('\nprocessed {} frames in {:.1f}s ({:.1f} FPS)'.format(
        processed, elapsed, processed / elapsed))
    print('gt frames: {}, pred frames: {}'.format(gt_frames, pred_frames))
    print('matched: {}, missed: {}, recall: {:.4f}'.format(
        matched, gt_frames - matched, recall))
    print('center error px: mean={:.2f}, median={:.2f}, p90={:.2f}'.format(
        mean_error, median_error, p90_error))
    print('matched center error px: mean={:.2f}, median={:.2f}'.format(
        mean_matched_error, median_matched_error))
    print('false positives on explicit negative frames: {}'.format(fp_negative))
    print('output: {}'.format(output_path.resolve()))
    print('stats: {}'.format(stats_path.resolve()))


if __name__ == '__main__':
    main()
