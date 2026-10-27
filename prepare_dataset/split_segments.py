import argparse
import json
import random


def build_segments(frame_indices):
    """Group sorted frame indices into maximal consecutive runs."""
    frames = sorted(set(int(f) for f in frame_indices))
    if not frames:
        return []

    segments = []
    start = prev = frames[0]
    for frame in frames[1:]:
        if frame == prev + 1:
            prev = frame
        else:
            segments.append((start, prev))
            start = prev = frame
    segments.append((start, prev))
    return segments


def count_frames(segments):
    return sum(end - start + 1 for start, end in segments)


def segments_to_frames(segments):
    frames = []
    for start, end in segments:
        frames.extend(range(start, end + 1))
    return frames


def merge_close_segments(segments, min_gap=0):
    """Merge segments whose gap is <= min_gap into blocks.

    A block is a list of original (start, end) segments, so unannotated
    frames between segments are never treated as annotated.
    """
    if not segments:
        return []

    blocks = [[tuple(segments[0])]]
    for start, end in segments[1:]:
        _, prev_end = blocks[-1][-1]
        gap = start - prev_end - 1
        if gap <= min_gap:
            blocks[-1].append((start, end))
        else:
            blocks.append([(start, end)])
    return blocks


def split_segments(segments, val_ratio=0.2, min_gap=0, mode='random', seed=42):
    """Split rally segments into train/val.

    Args:
        segments: list of (start, end) consecutive frame runs.
        val_ratio: target fraction of annotated frames for validation.
        min_gap: segments closer than this are kept in the same split.
        mode: 'random' (shuffle blocks) or 'chronological' (split by time).
        seed: random seed used by 'random' mode.

    Returns:
        (train_segments, val_segments, info)
    """
    if not segments:
        raise ValueError('segments is empty')
    if not 0 < val_ratio < 1:
        raise ValueError('val_ratio must be between 0 and 1')
    if mode not in ('random', 'chronological'):
        raise ValueError("mode must be 'random' or 'chronological'")

    blocks = merge_close_segments(segments, min_gap)
    if len(blocks) < 2:
        raise ValueError('need at least two blocks to split train/val')

    total = count_frames(segments)
    target = total * val_ratio

    if mode == 'chronological':
        best_k = None
        best_score = None
        for k in range(1, len(blocks)):
            val_frames = sum(count_frames(block) for block in blocks[k:])
            score = abs(val_frames - target)
            if best_score is None or score < best_score:
                best_k, best_score = k, score
        train_blocks = blocks[:best_k]
        val_blocks = blocks[best_k:]
    else:
        rng = random.Random(seed)
        order = list(range(len(blocks)))
        rng.shuffle(order)

        selected = []
        val_frames = 0
        for index in order:
            if val_frames >= target:
                break
            selected.append(index)
            val_frames += count_frames(blocks[index])

        if not selected:
            selected = [order[0]]
        if len(selected) == len(blocks):
            selected = selected[:-1]

        chosen = set(selected)
        train_blocks = [block for i, block in enumerate(blocks) if i not in chosen]
        val_blocks = [block for i, block in enumerate(blocks) if i in chosen]

    train_segments = [segment for block in train_blocks for segment in block]
    val_segments = [segment for block in val_blocks for segment in block]

    train_frames = count_frames(train_segments)
    val_frames = count_frames(val_segments)

    info = {
        'mode': mode,
        'seed': seed if mode == 'random' else None,
        'val_ratio': val_ratio,
        'min_gap': min_gap,
        'total_frames': total,
        'train_frames': train_frames,
        'val_frames': val_frames,
        'actual_val_ratio': round(val_frames / total, 4) if total else 0.0,
        'segment_count': len(segments),
        'block_count': len(blocks),
        'segments': [[start, end] for start, end in segments],
        'blocks': [[list(segment) for segment in block] for block in blocks],
        'train_segments': [[start, end] for start, end in train_segments],
        'val_segments': [[start, end] for start, end in val_segments],
    }
    return train_segments, val_segments, info


def main():
    parser = argparse.ArgumentParser(description='Split rally segments into train/val')
    parser.add_argument('--zip', help='read frame indices from an OpenTTGames zip')
    parser.add_argument('--frames', help='comma separated frame indices')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--min-gap', type=int, default=0)
    parser.add_argument('--mode', choices=['random', 'chronological'], default='random')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--json', action='store_true', help='print full split info as JSON')
    args = parser.parse_args()

    if args.zip:
        from read_zip import load_ball_markup
        frames = load_ball_markup(args.zip).keys()
    elif args.frames:
        frames = [int(x) for x in args.frames.split(',') if x.strip()]
    else:
        parser.error('provide --zip or --frames')

    segments = build_segments(frames)
    train, val, info = split_segments(
        segments,
        val_ratio=args.val_ratio,
        min_gap=args.min_gap,
        mode=args.mode,
        seed=args.seed,
    )

    if args.json:
        print(json.dumps(info, indent=2))
        return

    print('segments: {} (blocks: {})'.format(info['segment_count'], info['block_count']))
    print('total frames: {}'.format(info['total_frames']))
    print('train: {} segments / {} frames'.format(len(train), info['train_frames']))
    print('val:   {} segments / {} frames'.format(len(val), info['val_frames']))
    print('actual val ratio: {}'.format(info['actual_val_ratio']))
    print('train range: {} ~ {}'.format(train[0][0], train[-1][1]) if train else 'train range: -')
    print('val range:   {} ~ {}'.format(val[0][0], val[-1][1]) if val else 'val range: -')


if __name__ == '__main__':
    main()
