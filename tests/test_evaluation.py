import numpy as np

from biohub_tracker.evaluation import evaluate_tracking_graph
from biohub_tracker.types import Detection, Edge, LineageGraph


def node(uid: int, t: int, x: float) -> Detection:
    point = np.asarray([0.0, 0.0, x])
    return Detection(uid, "demo", t, point, point, 1.0)


def graph(edges: list[tuple[int, int]]) -> LineageGraph:
    result = LineageGraph("demo")
    result.add_nodes(
        [node(1, 0, 0), node(2, 1, -1), node(3, 1, 1), node(4, 2, -2)]
    )
    for source, target in edges:
        result.add_edge(Edge(source, target))
    return result


def test_tracking_metrics_separate_edges_and_divisions():
    truth = graph([(1, 2), (1, 3), (2, 4)])
    predicted = graph([(1, 2), (1, 3)])

    metrics = evaluate_tracking_graph(predicted, truth)

    assert metrics.edge_precision == 1.0
    assert metrics.edge_recall == 2 / 3
    assert metrics.edge_jaccard == 2 / 3
    assert metrics.division_jaccard == 1.0
    assert metrics.proxy_score == 2 / 3 + 0.1
