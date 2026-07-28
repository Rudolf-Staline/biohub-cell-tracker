import numpy as np

from biohub_tracker.submission import audit_submission, graph_to_submission
from biohub_tracker.types import Detection, Edge, LineageGraph


def _node(uid: int, t: int, x: float) -> Detection:
    coord = np.asarray([0.0, 0.0, x])
    return Detection(uid, "demo", t, coord, coord, 1.0)


def test_valid_submission_passes_audit():
    graph = LineageGraph("demo")
    graph.add_nodes([_node(1, 0, 0), _node(2, 1, 1), _node(3, 1, -1)])
    graph.add_edge(Edge(1, 2, "division"))
    graph.add_edge(Edge(1, 3, "division"))
    frame = graph_to_submission([graph])
    report = audit_submission(frame)
    assert report.valid, report.errors


def test_multiple_parents_is_rejected():
    graph = LineageGraph("demo")
    graph.add_nodes([_node(1, 0, 0), _node(2, 0, 2), _node(3, 1, 1)])
    graph.add_edge(Edge(1, 3))
    graph.add_edge(Edge(2, 3))
    report = audit_submission(graph_to_submission([graph]))
    assert not report.valid
    assert report.multiple_parents == 1
