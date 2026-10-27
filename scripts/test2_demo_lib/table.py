import zipfile

import cv2
import numpy as np

from .config import MAP_H, MAP_W, TABLE_BOTTOM, TABLE_LEFT, TABLE_RIGHT, TABLE_TOP


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
