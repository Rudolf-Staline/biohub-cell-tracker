import numpy as np

from biohub_tracker.config import DetectorConfig
from biohub_tracker.detection import BlobDetector3D


def test_blob_detector_finds_bright_peaks():
    frame = np.zeros((9, 25, 25), dtype=np.float32)
    frame[4, 8, 8] = 1.0
    frame[4, 17, 17] = 0.9
    detector = BlobDetector3D(
        DetectorConfig(
            sigma_vox=(0.8, 0.8, 0.8),
            threshold_quantile=0.99,
            absolute_threshold=0.01,
            min_distance_um=2.0,
            max_detections_per_frame=10,
        ),
        scale_um=(1.0, 1.0, 1.0),
    )
    detections = detector.detect(frame, dataset="demo", t=0, uid_start=0)
    assert len(detections) >= 2
