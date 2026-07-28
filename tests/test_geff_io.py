from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from biohub_tracker.geff_io import lineage_graph_from_networkx


def test_geff_scalar_coordinates_and_reverse_edges_are_normalized():
    graph = nx.DiGraph()
    graph.add_node(10, t=0, z=1, y=2, x=3)
    graph.add_node(11, t=1, z=1, y=2, x=4)
    graph.add_edge(11, 10)

    lineage = lineage_graph_from_networkx(
        graph,
        dataset="demo",
        fallback_scale_um=(2.0, 1.0, 0.5),
    )

    assert lineage.edge_pairs() == {(10, 11)}
    np.testing.assert_allclose(lineage.nodes[10].centroid_um, [2.0, 2.0, 1.5])


def test_geff_position_vector_uses_metadata_axis_names_and_scales():
    graph = nx.DiGraph()
    graph.add_node("a", position=[0, 2, 4, 6])
    graph.add_node("b", position=[1, 3, 5, 7])
    graph.add_edge("a", "b")
    metadata = SimpleNamespace(
        axes=[
            SimpleNamespace(name="t", scale=1.0),
            SimpleNamespace(name="z", scale=2.0),
            SimpleNamespace(name="y", scale=0.5),
            SimpleNamespace(name="x", scale=0.25),
        ]
    )

    lineage = lineage_graph_from_networkx(graph, dataset="demo", metadata=metadata)

    assert sorted(lineage.nodes) == [0, 1]
    np.testing.assert_allclose(lineage.nodes[0].centroid_um, [4.0, 2.0, 1.5])


def test_geff_rejects_non_adjacent_edges():
    graph = nx.DiGraph()
    graph.add_node(1, t=0, z=0, y=0, x=0)
    graph.add_node(2, t=2, z=0, y=0, x=1)
    graph.add_edge(1, 2)

    with pytest.raises(ValueError, match="adjacent frames"):
        lineage_graph_from_networkx(graph, dataset="demo")
