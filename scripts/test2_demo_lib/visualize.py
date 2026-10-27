import cv2
import numpy as np

from .config import (
    COLOR_BOUNCE,
    COLOR_GT,
    COLOR_LINK,
    COLOR_NET,
    COLOR_PRED,
    COLOR_TRAIL_GT,
    COLOR_TRAIL_PRED,
    EVENT_DISPLAY_AFTER,
    EVENT_DISPLAY_BEFORE,
    MAP_H,
    MAP_W,
    MAX_TRAIL_GAP,
    TABLE_BOTTOM,
    TABLE_LEFT,
    TABLE_RIGHT,
    TABLE_TOP,
)


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


def build_event_display(events, before=EVENT_DISPLAY_BEFORE,
                        after=EVENT_DISPLAY_AFTER):
    display = {}
    for frame, kind in events.items():
        if kind not in ('bounce', 'net'):
            continue
        for delta in range(-before, after + 1):
            display.setdefault(frame + delta, []).append(kind)
    return display


def add_bounce_mark(bounce_marks, frame_idx, pos, kind):
    bounce_marks.append({'frame': frame_idx, 'pos': pos, 'kind': kind})


def draw_minimap(frame, base_map, pred_trail, gt_trail, last_pred,
                 last_gt, bounce_marks, frame_idx, fps):
    canvas = base_map.copy()

    draw_trail(canvas, gt_trail, COLOR_TRAIL_GT)
    draw_trail(canvas, pred_trail, COLOR_TRAIL_PRED)

    max_age = max(int(fps * 2), 1)
    for mark in bounce_marks:
        age = frame_idx - mark['frame']
        if age < 0 or age > max_age:
            continue
        alpha = 1.0 - age / max_age
        base_color = COLOR_BOUNCE if mark['kind'] == 'bounce' else COLOR_NET
        color = tuple(int(ch * alpha) for ch in base_color)
        cv2.circle(canvas, mark['pos'], 6, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, mark['kind'][:1].upper(),
                    (mark['pos'][0] + 8, mark['pos'][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    if last_gt is not None:
        cv2.circle(canvas, last_gt, 5, COLOR_GT, -1, cv2.LINE_AA)
    if last_pred is not None:
        cv2.circle(canvas, last_pred, 5, COLOR_PRED, -1, cv2.LINE_AA)

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
    cv2.rectangle(panel, (10, 10), (520, 200), (0, 0, 0), -1)
    frame = cv2.addWeighted(panel, 0.45, frame, 0.55, 0)

    pred_center = None
    if pred is not None:
        pred_center = ((pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0)

    gt_text = '--' if gt is None else '({:.0f}, {:.0f})'.format(gt[0], gt[1])
    pred_text = '--' if pred_center is None else '({:.1f}, {:.1f})'.format(
        pred_center[0], pred_center[1])

    lines = [
        'frame {}/{}  t={:.2f}s'.format(
            frame_idx, total_frames, frame_idx / max(fps, 1)),
        'GT   px: {}'.format(gt_text),
        'pred px: {}'.format(pred_text),
        'conf: {}   err: {}'.format(
            '--' if pred is None else '{:.2f}'.format(pred[4]),
            '--' if distance is None else '{:.1f} px'.format(distance)),
        'recall: {}/{} ({:.1f}%)'.format(
            matched, gt_frames, 100.0 * recall if gt_frames else 0.0),
    ]
    y = 36
    for line in lines:
        cv2.putText(frame, line, (22, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        y += 28

    if gt is not None:
        cv2.circle(frame, (int(gt[0]), int(gt[1])), 7, COLOR_GT, 2, cv2.LINE_AA)
        cv2.putText(frame, 'GT', (int(gt[0]) + 10, int(gt[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GT, 2)

    if pred is not None:
        x1, y1, x2, y2, conf = pred
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                      COLOR_PRED, 2)
        cv2.circle(frame, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                   4, COLOR_PRED, -1, cv2.LINE_AA)
        cv2.putText(frame, 'ball {:.2f}'.format(conf),
                    (int(x1), max(int(y1) - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRED, 2)

    if gt is not None and pred is not None:
        pred_center = (int((pred[0] + pred[2]) / 2), int((pred[1] + pred[3]) / 2))
        cv2.line(frame, (int(gt[0]), int(gt[1])), pred_center,
                 COLOR_LINK, 2, cv2.LINE_AA)

    if event:
        text = event.upper()
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        x = (frame.shape[1] - tw) // 2
        cv2.putText(frame, text, (x, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, COLOR_LINK, 3)
    return frame
