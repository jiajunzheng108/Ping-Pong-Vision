import argparse
import json
import random
import shutil
from bisect import bisect_left
from pathlib import Path

import cv2

from read_zip import load_ball_markup, load_events, summarize
from split_segments import build_segments, segments_to_frames, split_segments

NEGATIVE_GAP = 30


def create_label(ball, img_width, img_height, box_size):
    """Return one YOLO label line, or None when this frame has no ball."""
    x = ball['x']
    y = ball['y']
    if x < 0 or y < 0:
        return None

    x_center = min(max(x / img_width, 0.0), 1.0)
    y_center = min(max(y / img_height, 0.0), 1.0)
    width = box_size / img_width
    height = box_size / img_height
    return '0 {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
        x_center, y_center, width, height)


def sample_extra_negatives(markup, max_frame, count, rng, gap=NEGATIVE_GAP):
    """Sample unannotated frames that are far from every annotated frame."""
    if count <= 0:
        return []

    annotated = sorted(markup)
    candidates = []
    for frame in range(max_frame + 1):
        if frame in markup:
            continue
        index = bisect_left(annotated, frame)
        near = False
        for pos in (index - 1, index):
            if 0 <= pos < len(annotated) and abs(frame - annotated[pos]) <= gap:
                near = True
                break
        if not near:
            candidates.append(frame)

    if not candidates:
        return []
    return sorted(rng.sample(candidates, min(count, len(candidates))))


def subsample_segments(segments, markup, step):
    """Take every step-th frame in each segment.

    The segment start is always kept, explicit negatives are always kept,
    and every segment keeps at least one frame.
    """
    if step <= 1:
        return segments_to_frames(segments)

    frames = set()
    for start, end in segments:
        for frame in range(start, end + 1):
            ball = markup.get(frame)
            is_negative = ball is not None and (ball['x'] < 0 or ball['y'] < 0)
            if frame == start or (frame - start) % step == 0 or is_negative:
                frames.add(frame)
    return sorted(frames)


def _prepare_output(output_dir, overwrite, dry_run):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                'output dir is not empty: {} (use --overwrite)'.format(output_dir))
        if not dry_run:
            for name in ('images', 'labels'):
                shutil.rmtree(output_dir / name, ignore_errors=True)
            for name in ('dataset.yaml', 'split.json'):
                path = output_dir / name
                if path.exists():
                    path.unlink()

    if not dry_run:
        for subset in ('train', 'val'):
            (output_dir / 'images' / subset).mkdir(parents=True, exist_ok=True)
            (output_dir / 'labels' / subset).mkdir(parents=True, exist_ok=True)
    return output_dir


def write_dataset_yaml(output_dir):
    content = (
        'path: {}\n'
        'train: images/train\n'
        'val: images/val\n'
        '\n'
        'names:\n'
        '  0: ball\n'
    ).format(Path(output_dir).resolve().as_posix())
    with open(Path(output_dir) / 'dataset.yaml', 'w', encoding='utf-8') as f:
        f.write(content)


def build_dataset(video_path, zip_path, output_dir, val_ratio=0.2,
                  box_size=30, min_gap=0, split_mode='random', seed=42,
                  frame_step=1, val_frame_step=1,
                  extra_negatives=0, prefix=None, overwrite=False,
                  dry_run=False):
    if frame_step < 1 or val_frame_step < 1:
        raise ValueError('frame_step and val_frame_step must be >= 1')

    video_path = Path(video_path)
    zip_path = Path(zip_path)
    prefix = prefix or video_path.stem

    markup = load_ball_markup(zip_path)
    events = load_events(zip_path)
    ball_info = summarize(markup)

    segments = build_segments(markup.keys())
    train_segments, val_segments, split_info = split_segments(
        segments,
        val_ratio=val_ratio,
        min_gap=min_gap,
        mode=split_mode,
        seed=seed,
    )

    train_frames = set(subsample_segments(train_segments, markup, frame_step))
    val_frames = set(subsample_segments(val_segments, markup, val_frame_step))
    overlap = train_frames & val_frames
    if overlap:
        raise RuntimeError('train/val frame overlap: {}'.format(sorted(overlap)[:10]))

    rng = random.Random(seed)
    extra_negative_frames = sample_extra_negatives(
        markup, ball_info['max_frame'], extra_negatives, rng)

    wanted = {}
    for frame in train_frames:
        wanted[frame] = 'train'
    for frame in val_frames:
        wanted[frame] = 'val'
    for frame in extra_negative_frames:
        wanted[frame] = 'train'

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError('cannot open video: {}'.format(video_path))

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_of_range = [frame for frame in wanted if frame >= total_video_frames]
    for frame in out_of_range:
        wanted.pop(frame)
    for frame in out_of_range:
        train_frames.discard(frame)
        val_frames.discard(frame)

    stats = {
        'train': {'images': 0, 'empty_labels': 0},
        'val': {'images': 0, 'empty_labels': 0},
        'missing_frames': 0,
    }

    print('video: {} ({} frames, {} fps, {}x{})'.format(
        video_path, total_video_frames, video_fps, video_width, video_height))
    print('read: {} annotated ({} positive, {} negative), {} segments'.format(
        ball_info['total'], ball_info['positive'], ball_info['negative'],
        split_info['segment_count']))
    print('split: {} seed={}, train {} seg / {} frames, val {} seg / {} frames'.format(
        split_mode, seed,
        len(train_segments), split_info['train_frames'],
        len(val_segments), split_info['val_frames']))
    print('subsample: train step={} -> {} frames, val step={} -> {} frames'.format(
        frame_step, len(train_frames), val_frame_step, len(val_frames)))
    if extra_negative_frames:
        print('extra negatives: {} frames (train)'.format(len(extra_negative_frames)))
    if out_of_range:
        print('warning: skipped {} frames beyond video length'.format(len(out_of_range)))

    if dry_run:
        print('dry-run: no files written')
        cap.release()
        return stats

    output_dir = _prepare_output(output_dir, overwrite, dry_run)

    written_frames = set()
    max_wanted = max(wanted) if wanted else -1
    frame_idx = 0

    while frame_idx <= max_wanted:
        if frame_idx in wanted:
            if not cap.grab():
                break
            ok, frame = cap.retrieve()
            if not ok:
                frame_idx += 1
                continue

            subset = wanted[frame_idx]
            height, width = frame.shape[:2]
            image_name = '{}_img_{:06d}'.format(prefix, frame_idx)
            image_path = output_dir / 'images' / subset / (image_name + '.jpg')
            label_path = output_dir / 'labels' / subset / (image_name + '.txt')

            cv2.imwrite(str(image_path), frame)

            label = None
            if frame_idx in markup:
                label = create_label(markup[frame_idx], width, height, box_size)

            with open(label_path, 'w', encoding='utf-8') as f:
                if label is not None:
                    f.write(label)

            stats[subset]['images'] += 1
            if label is None:
                stats[subset]['empty_labels'] += 1
            written_frames.add(frame_idx)
        else:
            if not cap.grab():
                break
        frame_idx += 1

    cap.release()

    stats['missing_frames'] = len(wanted) - len(written_frames)
    if stats['missing_frames'] > 0:
        print('warning: {} requested frames could not be read'.format(
            stats['missing_frames']))

    write_dataset_yaml(output_dir)

    split_record = {
        'video': str(video_path.resolve()),
        'zip': str(zip_path.resolve()),
        'output_dir': str(Path(output_dir).resolve()),
        'prefix': prefix,
        'box_size': box_size,
        'val_ratio': val_ratio,
        'min_gap': min_gap,
        'split_mode': split_mode,
        'seed': seed,
        'frame_step': frame_step,
        'val_frame_step': val_frame_step,
        'ball_markup': ball_info,
        'event_count': len(events),
        'selected': {
            'train_frames': len(train_frames),
            'val_frames': len(val_frames),
            'extra_negative_frames': len(extra_negative_frames),
        },
        'extra_negative_frames': extra_negative_frames,
        'split': split_info,
        'written': stats,
    }
    with open(Path(output_dir) / 'split.json', 'w', encoding='utf-8') as f:
        json.dump(split_record, f, ensure_ascii=False, indent=2)

    print('written: train {} images / {} empty labels, val {} images / {} empty labels'.format(
        stats['train']['images'], stats['train']['empty_labels'],
        stats['val']['images'], stats['val']['empty_labels']))
    print('output: {}'.format(Path(output_dir).resolve()))
    return stats


def main():
    parser = argparse.ArgumentParser(description='Build a YOLO dataset from OpenTTGames files')
    parser.add_argument('--video', default='../Data/test_2.mp4')
    parser.add_argument('--zip', default='../Data/test_2.zip')
    parser.add_argument('--output-dir', default='../dataset/test2_yolo')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--box-size', type=int, default=30)
    parser.add_argument('--min-gap', type=int, default=0)
    parser.add_argument('--split-mode', choices=['random', 'chronological'], default='random')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--frame-step', type=int, default=1,
                        help='keep every Nth train frame per segment')
    parser.add_argument('--val-frame-step', type=int, default=1,
                        help='keep every Nth val frame per segment')
    parser.add_argument('--extra-negatives', type=int, default=0)
    parser.add_argument('--prefix', default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    build_dataset(
        video_path=args.video,
        zip_path=args.zip,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        box_size=args.box_size,
        min_gap=args.min_gap,
        split_mode=args.split_mode,
        seed=args.seed,
        frame_step=args.frame_step,
        val_frame_step=args.val_frame_step,
        extra_negatives=args.extra_negatives,
        prefix=args.prefix,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
