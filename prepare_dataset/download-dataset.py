import argparse
import os

import wget


def make_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)


def parse_ids(value):
    if not value:
        return []
    return [int(x) for x in value.split(',') if x.strip()]


def download_file(url, out_path):
    if os.path.isfile(out_path):
        print('Already exists, skip: {}'.format(out_path))
        return

    part_path = out_path + '.part'
    if os.path.isfile(part_path):
        os.remove(part_path)

    print('Downloading...{}'.format(url))
    wget.download(url, part_path)
    os.replace(part_path, out_path)
    print('\nSaved: {}'.format(out_path))


def main():
    parser = argparse.ArgumentParser(description='Download OpenTTGames dataset')
    parser.add_argument('--train-games', default='1,2,3,4,5',
                        help='comma separated training game ids')
    parser.add_argument('--test-games', default='1,2,3,4,5,6,7',
                        help='comma separated test game ids')
    args = parser.parse_args()

    dataset_dir = '../dataset'
    common_url = 'https://lab.osai.ai/datasets/openttgames/data/'

    splits = [
        ('training', 'game', parse_ids(args.train_games)),
        ('test', 'test', parse_ids(args.test_games)),
    ]

    for dataset_type, prefix, game_ids in splits:
        video_dir = os.path.join(dataset_dir, dataset_type, 'videos')
        annotation_dir = os.path.join(dataset_dir, dataset_type, 'annotations')
        make_folder(video_dir)
        make_folder(annotation_dir)

        for game_id in game_ids:
            for ext, target_dir in (('mp4', video_dir), ('zip', annotation_dir)):
                filename = '{}_{}.{}'.format(prefix, game_id, ext)
                download_file(common_url + filename, os.path.join(target_dir, filename))


if __name__ == '__main__':
    main()
