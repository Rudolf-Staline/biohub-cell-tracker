import numpy as np

from biohub_tracker.config import PipelineConfig, TrackerConfig
from biohub_tracker.oracle import OracleTrackingPipeline
from biohub_tracker.types import Detection, Edge, LineageGraph


def d(uid: int, t: int, x: float) -> Detection:
    point = np.asarray([0.0, 0.0, x])
    return Detection(uid, "demo", t, point, point, 1.0)


def test_oracle_pipeline_recovers_simple_tracks():
    truth = LineageGraph("demo")
    truth.add_nodes([d(1, 0, 0), d(2, 0, 10), d(3, 1, 1), d(4, 1, 9)])
    truth.add_edge(Edge(1, 3))
    truth.add_edge(Edge(2, 4))

    pipeline = OracleTrackingPipeline(
        PipelineConfig(
            scale_um=(1.0, 1.0, 1.0),
            tracker=TrackerConfig(max_link_distance_um=5.0),
        )
    )
    result = pipeline.evaluate_truth_graph(truth)

    assert result.metrics.edge_jaccard == 1.0
    assert result.metrics.proxy_score == 1.1
