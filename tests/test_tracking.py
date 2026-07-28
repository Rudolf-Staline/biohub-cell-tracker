import numpy as np

from biohub_tracker.config import TrackerConfig
from biohub_tracker.tracking import BaselineTracker
from biohub_tracker.types import Detection


def d(uid, t, x, score=1.0):
    vox = np.asarray([0.0, 0.0, float(x)])
    return Detection(uid, "demo", t, vox, vox, score)


def test_continuity_links_nearest_cells():
    tracker = BaselineTracker(TrackerConfig(max_link_distance_um=5.0))
    edges = tracker.link_pair([d(1, 0, 0), d(2, 0, 10)], [d(3, 1, 1), d(4, 1, 9)])
    assert {(e.source_uid, e.target_uid) for e in edges} == {(1, 3), (2, 4)}


def test_division_replaces_single_match_with_two_children():
    config = TrackerConfig(
        max_link_distance_um=3.0,
        division_parent_radius_um=5.0,
        division_sister_distance_um=5.0,
    )
    tracker = BaselineTracker(config)
    edges = tracker.link_pair([d(1, 0, 0)], [d(2, 1, -1), d(3, 1, 1)])
    assert {(e.source_uid, e.target_uid) for e in edges} == {(1, 2), (1, 3)}
    assert all(e.kind == "division" for e in edges)


def test_frames_preserve_empty_timepoints():
    from biohub_tracker.types import LineageGraph

    graph = LineageGraph("demo")
    graph.add_nodes([d(1, 0, 0), d(2, 2, 2)])

    frames = graph.frames()

    assert len(frames) == 3
    assert frames[1] == []
