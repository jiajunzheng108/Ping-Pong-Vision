import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2

BOX_SIZE = 25
SEED = 42


def create_yolo_label(ball_data, img_width, img_height, box_size=BOX_SIZE):
    """Convert ball coordinates to YOLO format."""
    if ball_data['x'] < 0 or ball_data['y'] < 0:
        return None

    x_center = ball_data['x'] / img_width
    y_center = ball_data['y'] / img_height
    width = box_size / img_width
    height = box_size / img_height

    return "0 {:.6f} {:.6f} {:.6f} {:.6f}".format(
        x_center, y_center, width, height
    )


def collect_frames(dataset_dir, game_id):
    """Return (positives, negatives) for one game."""
    game_name = 'game_{}'.format(game_id)
    images_dir = os.path.join(dataset_dir, 'images', game_name)
    annotation_file = os.path.join(dataset_dir, 'annotations', game_name, 'ball_markup.json')

    if not os.path.isfile(annotation_file):
        raise FileNotFoundError('Missing annotation: {}'.format(annotation_file))
    if not os.path.isdir(images_dir):
        raise FileNotFoundError('Missing images dir: {}'.format(images_dir))

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    positives = []
    negatives = []
    for frame_num, ball_data in annotations.items():
        img_path = os.path.join(images_dir, 'img_{}.jpg'.format(frame_num.zfill(6)))
        if not os.path.exists(img_path):
            continue
        if ball_data['x'] < 0 or ball_data['y'] < 0:
            negatives.append(img_path)
        else:
            positives.append((img_path, ball_data))

    return positives, negatives


def sample_negatives(negatives, count, rng):
    if count <= 0 or not negatives:
        return []
    if count >= len(negatives):
        return list(negatives)
    return rng.sample(negatives, count)


def copy_sample(img_path, subset, output_dir, game_id, ball_data=None):
    """Copy one image and write its YOLO label (empty label for negatives)."""
    img = cv2.imread(img_path)
    if img is None:
        print('Warning: could not read {}'.format(img_path))
        return False

    height, width = img.shape[:2]
    name = 'game_{}_{}'.format(game_id, os.path.basename(img_path))
    new_img_path = os.path.join(output_dir, subset, 'images', name)
    label_path = os.path.join(output_dir, subset, 'labels', os.path.splitext(name)[0] + '.txt')

    shutil.copy2(img_path, new_img_path)
    with open(label_path, 'w') as f:
        if ball_data is not None:
            yolo_label = create_yolo_label(ball_data, width, height)
            if yolo_label is not None:
                f.write(yolo_label)

    return True


def prepare_yolo_dataset(dataset_dir, output_dir, train_games, val_games,
                         negative_ratio=0.15):
    overlap = set(train_games) & set(val_games)
    if overlap:
        raise ValueError('train/val games overlap: {}'.format(sorted(overlap)))

    for subset in ('train', 'val'):
        Path(output_dir, subset, 'images').mkdir(parents=True, exist_ok=True)
        Path(output_dir, subset, 'labels').mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)

    def add_game(game_id, subset):
        positives, negatives = collect_frames(dataset_dir, game_id)

        if subset == 'train':
            neg_samples = sample_negatives(
                negatives, int(round(len(positives) * negative_ratio)), rng)
        else:
            neg_samples = negatives

        n_pos = sum(copy_sample(img_path, subset, output_dir, game_id, ball_data)
                    for img_path, ball_data in positives)
        n_neg = sum(copy_sample(img_path, subset, output_dir, game_id)
                    for img_path in neg_samples)

        print('game_{} [{}]: {} positive, {} negative'.format(
            game_id, subset, n_pos, n_neg))
        return n_pos, n_neg

    print('Preparing YOLO dataset...')
    stats = {'train': [0, 0], 'val': [0, 0]}
    for game_id in train_games:
        pos, neg = add_game(game_id, 'train')
        stats['train'][0] += pos
        stats['train'][1] += neg
    for game_id in val_games:
        pos, neg = add_game(game_id, 'val')
        stats['val'][0] += pos
        stats['val'][1] += neg

    yaml_content = """
path: {path}
train: train/images
val: val/images

names:
  0: ball
    """.format(path=os.path.abspath(output_dir))

    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

    print('\nDataset prepared: {}'.format(os.path.abspath(output_dir)))
    for subset in ('train', 'val'):
        print('  {}: {} positive, {} negative'.format(subset, stats[subset][0], stats[subset][1]))


def parse_ids(value):
    return [int(x) for x in value.split(',') if x.strip()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare OpenTTGames YOLO dataset')
    parser.add_argument('--dataset-dir', default='../dataset/training')
    parser.add_argument('--output-dir', default='../dataset/yolo_dataset')
    parser.add_argument('--train-games', default='1,2', help='comma separated game ids')
    parser.add_argument('--val-games', default='3', help='comma separated game ids')
    parser.add_argument('--negative-ratio', type=float, default=0.15,
                        help='negative frames as a ratio of positive frames')
    args = parser.parse_args()

    prepare_yolo_dataset(
        args.dataset_dir,
        args.output_dir,
        parse_ids(args.train_games),
        parse_ids(args.val_games),
        negative_ratio=args.negative_ratio,
    )
