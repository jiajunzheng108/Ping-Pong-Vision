import numpy as np


class DemoMetrics:
    def __init__(self, match_threshold=50.0):
        self.match_threshold = match_threshold
        self.gt_frames = 0
        self.pred_frames = 0
        self.matched = 0
        self.fp_negative = 0
        self.errors = []
        self.matched_errors = []

    @property
    def recall(self):
        return self.matched / self.gt_frames if self.gt_frames else 0.0

    def update(self, gt, pred, is_negative=False):
        """Update counters and return the current GT-pred distance (or None)."""
        distance = None
        if gt is not None:
            self.gt_frames += 1
            if pred is not None:
                center = ((pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0)
                distance = float(np.hypot(center[0] - gt[0], center[1] - gt[1]))
                if distance <= self.match_threshold:
                    self.matched += 1
                    self.matched_errors.append(distance)
                self.errors.append(distance)
        elif is_negative and pred is not None:
            self.fp_negative += 1

        if pred is not None:
            self.pred_frames += 1

        return distance

    def summary(self):
        if self.errors:
            errors = np.array(self.errors)
            mean_error = float(errors.mean())
            median_error = float(np.median(errors))
            p90_error = float(np.percentile(errors, 90))
        else:
            mean_error = median_error = p90_error = 0.0

        if self.matched_errors:
            matched = np.array(self.matched_errors)
            mean_matched = float(matched.mean())
            median_matched = float(np.median(matched))
        else:
            mean_matched = median_matched = 0.0

        return {
            'gt_frames': self.gt_frames,
            'pred_frames': self.pred_frames,
            'matched': self.matched,
            'missed': self.gt_frames - self.matched,
            'recall': self.recall,
            'fp_on_negative_frames': self.fp_negative,
            'mean_center_error_px': mean_error,
            'median_center_error_px': median_error,
            'p90_center_error_px': p90_error,
            'mean_matched_error_px': mean_matched,
            'median_matched_error_px': median_matched,
        }
