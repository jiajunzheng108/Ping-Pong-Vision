import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'prepare_dataset'))

from prepare_dataset.read_zip import load_ball_markup, load_events  # noqa: E402
from test2_demo_lib.config import BOUNCE_MARK_LENGTH, TRAIL_LENGTH  # noqa: E402
from test2_demo_lib.detector import detect_ball, load_model, parse_classes  # noqa: E402
from test2_demo_lib.metrics import DemoMetrics  # noqa: E402
from test2_demo_lib.table import (  # noqa: E402
    build_homography,
    extract_table_corners,
    to_topdown,
)
from test2_demo_lib.visualize import (  # noqa: E402
    add_bounce_mark,
    build_event_display,
    create_base_map,
    draw_minimap,
    draw_overlay,
)


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

    model_path = Path(args.model)
    model = load_model(args.model)
    classes = parse_classes(args.classes)

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

    event_display = build_event_display(events)
    base_map = create_base_map()
    pred_trail = deque(maxlen=TRAIL_LENGTH)
    gt_trail = deque(maxlen=TRAIL_LENGTH)
    bounce_marks = deque(maxlen=BOUNCE_MARK_LENGTH)

    output_fps = args.output_fps if args.output_fps > 0 else fps
    write_step = max(int(round(fps / output_fps)), 1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*'mp4v'),
        output_fps, (width, height))

    metrics = DemoMetrics(args.match_threshold)
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

        pred = detect_ball(model, frame, conf=args.conf,
                           classes=classes, imgsz=args.imgsz)
        distance = metrics.update(gt, pred, is_negative)

        if pred is not None:
            center = ((pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0)
            last_table_pred = to_topdown(homography, center)
            pred_trail.append((frame_idx, last_table_pred))

        if gt is not None:
            last_table_gt = to_topdown(homography, gt)
            gt_trail.append((frame_idx, last_table_gt))

        if frame_idx in events and events[frame_idx] in ('bounce', 'net'):
            pos = last_table_pred if last_table_pred is not None else last_table_gt
            if pos is not None:
                add_bounce_mark(bounce_marks, frame_idx, pos, events[frame_idx])

        active_event = None
        if frame_idx in event_display:
            active_event = event_display[frame_idx][-1]

        frame = draw_overlay(frame, frame_idx, total_frames, gt, pred,
                             distance, active_event, metrics.recall,
                             metrics.matched, metrics.gt_frames, fps)
        frame = draw_minimap(frame, base_map, pred_trail, gt_trail,
                             last_table_pred, last_table_gt, bounce_marks,
                             frame_idx, fps)

        if args.show_corners:
            cv2.polylines(frame, [corners.astype('int32')], True,
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
    summary = metrics.summary()

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
        **summary,
        'elapsed_sec': elapsed,
        'processed_fps': processed / elapsed,
    }

    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print('\nprocessed {} frames in {:.1f}s ({:.1f} FPS)'.format(
        processed, elapsed, processed / elapsed))
    print('gt frames: {}, pred frames: {}'.format(
        summary['gt_frames'], summary['pred_frames']))
    print('matched: {}, missed: {}, recall: {:.4f}'.format(
        summary['matched'], summary['missed'], summary['recall']))
    print('center error px: mean={:.2f}, median={:.2f}, p90={:.2f}'.format(
        summary['mean_center_error_px'],
        summary['median_center_error_px'],
        summary['p90_center_error_px']))
    print('matched center error px: mean={:.2f}, median={:.2f}'.format(
        summary['mean_matched_error_px'],
        summary['median_matched_error_px']))
    print('false positives on explicit negative frames: {}'.format(
        summary['fp_on_negative_frames']))
    print('output: {}'.format(output_path.resolve()))
    print('stats: {}'.format(stats_path.resolve()))


if __name__ == '__main__':
    main()
