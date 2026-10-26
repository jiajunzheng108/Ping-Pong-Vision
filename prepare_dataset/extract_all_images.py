import argparse
import os
from glob import glob

import cv2
from tqdm import tqdm


def make_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)


def extract_images_from_videos(video_path, out_images_dir, frame_step=1,
                               max_frames=0, overwrite=False):
    video_fn = os.path.basename(video_path)[:-4]
    sub_images_dir = os.path.join(out_images_dir, video_fn)
    make_folder(sub_images_dir)

    video_cap = cv2.VideoCapture(video_path)
    n_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_width = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    f_height = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print('video_fn: {}.mp4, number of frames: {}, f_width: {}, f_height: {}'.format(
        video_fn, n_frames, f_width, f_height))

    expected = n_frames // frame_step if max_frames <= 0 else max_frames
    pbar = tqdm(total=max(expected, 1))

    frame_cnt = -1
    processed = 0
    saved = 0

    while True:
        ret, img = video_cap.read()
        if not ret:
            break

        frame_cnt += 1
        if frame_cnt % frame_step != 0:
            continue

        processed += 1
        pbar.update(1)

        image_path = os.path.join(sub_images_dir, 'img_{:06d}.jpg'.format(frame_cnt))
        if overwrite or not os.path.isfile(image_path):
            cv2.imwrite(image_path, img)
            saved += 1

        if max_frames > 0 and processed >= max_frames:
            break

    video_cap.release()
    pbar.close()
    print('done extraction: {} (processed {}, saved {})'.format(video_path, processed, saved))


def main():
    parser = argparse.ArgumentParser(description='Extract frames from OpenTTGames videos')
    parser.add_argument('--split', choices=['training', 'test', 'both'], default='both')
    parser.add_argument('--games', default='', help='comma separated video ids, empty = all')
    parser.add_argument('--frame-step', type=int, default=1, help='save every Nth frame')
    parser.add_argument('--max-frames', type=int, default=0, help='max frames per video, 0 = all')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing images')
    args = parser.parse_args()

    splits = ['training', 'test'] if args.split == 'both' else [args.split]
    wanted = {int(x) for x in args.games.split(',') if x.strip()} if args.games else None

    dataset_dir = '../dataset'
    for dataset_type in splits:
        video_dir = os.path.join(dataset_dir, dataset_type, 'videos')
        out_images_dir = os.path.join(dataset_dir, dataset_type, 'images')

        video_paths = sorted(glob(os.path.join(video_dir, '*.mp4')))
        for video_path in video_paths:
            stem = os.path.splitext(os.path.basename(video_path))[0]
            try:
                video_id = int(stem.split('_')[-1])
            except ValueError:
                video_id = None
            if wanted is not None and video_id not in wanted:
                continue

            extract_images_from_videos(
                video_path,
                out_images_dir,
                frame_step=args.frame_step,
                max_frames=args.max_frames,
                overwrite=args.overwrite,
            )


if __name__ == '__main__':
    main()
