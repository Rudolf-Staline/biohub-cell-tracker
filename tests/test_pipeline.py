import numpy as np

from biohub_tracker.config import DetectorConfig, PipelineConfig, TrackerConfig
from biohub_tracker.pipeline import BaselinePipeline
from biohub_tracker.submission import audit_submission, graph_to_submission


class FakeMovie:
    dataset_name = "synthetic"
    n_frames = 3

    def read_frame(self, t: int) -> np.ndarray:
        frame = np.zeros((9, 31, 31), dtype=np.float32)
        frame[4, 10 + t, 10 + t] = 1.0
        frame[4, 22 - t, 22 - t] = 0.9
        return frame


def test_process_movie_produces_valid_graph():
    config = PipelineConfig(
        scale_um=(1.0, 1.0, 1.0),
        detector=DetectorConfig(
            sigma_vox=(0.8, 0.8, 0.8),
            threshold_quantile=0.99,
            absolute_threshold=0.01,
            min_distance_um=2.0,
            patch_depth_z=5,
            patch_overlap_z=1,
        ),
        tracker=TrackerConfig(max_link_distance_um=5.0),
    )
    graph, stats, _ = BaselinePipeline(config).process_movie(FakeMovie())
    report = audit_submission(graph_to_submission([graph]))
    assert report.valid, report.errors
    assert stats.frames == 3
    assert stats.nodes >= 6
