import argparse
import json
import zipfile


def _find_member(zf, filename):
    for name in zf.namelist():
        if name == filename or name.endswith('/' + filename):
            return name
    return None


def _read_json(zf, filename):
    member = _find_member(zf, filename)
    if member is None:
        return None
    with zf.open(member) as f:
        return json.load(f)


def load_ball_markup(zip_path):
    """Read ball_markup.json from an OpenTTGames zip.

    Returns:
        dict[int, dict]: frame index -> {'x': float, 'y': float}
    """
    with zipfile.ZipFile(zip_path) as zf:
        raw = _read_json(zf, 'ball_markup.json')

    if raw is None:
        raise FileNotFoundError('ball_markup.json not found in {}'.format(zip_path))

    markup = {}
    for key, value in raw.items():
        try:
            frame = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, dict) or 'x' not in value or 'y' not in value:
            continue
        markup[frame] = {'x': float(value['x']), 'y': float(value['y'])}
    return markup


def load_events(zip_path):
    """Read events_markup.json (frame -> 'bounce'/'net')."""
    with zipfile.ZipFile(zip_path) as zf:
        raw = _read_json(zf, 'events_markup.json') or {}

    events = {}
    for key, value in raw.items():
        try:
            events[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return events


def list_masks(zip_path):
    """List segmentation mask entries inside the zip."""
    with zipfile.ZipFile(zip_path) as zf:
        return sorted(
            name for name in zf.namelist()
            if 'segmentation_masks' in name and name.lower().endswith('.png')
        )


def summarize(ball_markup):
    """Summarize ball markup for sanity checks."""
    frames = sorted(ball_markup)
    positive = []
    negative = []
    single_coord_negative = []

    for frame in frames:
        x = ball_markup[frame]['x']
        y = ball_markup[frame]['y']
        if x >= 0 and y >= 0:
            positive.append(frame)
        else:
            negative.append(frame)
            if (x < 0) != (y < 0):
                single_coord_negative.append(frame)

    return {
        'total': len(frames),
        'positive': len(positive),
        'negative': len(negative),
        'single_coord_negative': len(single_coord_negative),
        'min_frame': frames[0] if frames else None,
        'max_frame': frames[-1] if frames else None,
    }


def main():
    parser = argparse.ArgumentParser(description='Read OpenTTGames annotation zip')
    parser.add_argument('--zip', required=True, help='path to test_2.zip / game_1.zip')
    parser.add_argument('--events', action='store_true', help='print event counts')
    parser.add_argument('--masks', action='store_true', help='print mask count')
    parser.add_argument('--json', action='store_true', help='print summary as JSON')
    args = parser.parse_args()

    markup = load_ball_markup(args.zip)
    info = summarize(markup)

    if args.events:
        events = load_events(args.zip)
        counts = {}
        for value in events.values():
            counts[value] = counts.get(value, 0) + 1
        info['events'] = counts

    if args.masks:
        info['masks'] = len(list_masks(args.zip))

    if args.json:
        print(json.dumps(info, indent=2))
        return

    print('total annotated frames: {}'.format(info['total']))
    print('  positive: {}'.format(info['positive']))
    print('  negative: {} (single-coord -1: {})'.format(
        info['negative'], info['single_coord_negative']))
    print('  frame range: {} ~ {}'.format(info['min_frame'], info['max_frame']))
    if args.events:
        print('events: {}'.format(info['events']))
    if args.masks:
        print('masks: {}'.format(info['masks']))


if __name__ == '__main__':
    main()
