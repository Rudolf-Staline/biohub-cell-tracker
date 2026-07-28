from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .types import LineageGraph


def _safe_ratio(numerator: int, denominator: int, *, empty_value: float = 1.0) -> float:
    if denominator == 0:
        return empty_value
    return float(numerator / denominator)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


@dataclass(frozen=True, slots=True)
class TrackingMetrics:
    dataset: str
    truth_nodes: int
    predicted_nodes: int
    truth_edges: int
    predicted_edges: int
    true_positive_edges: int
    edge_precision: float
    edge_recall: float
    edge_f1: float
    edge_jaccard: float
    truth_divisions: int
    predicted_divisions: int
    true_positive_divisions: int
    division_precision: float
    division_recall: float
    division_f1: float
    division_jaccard: float
    proxy_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_tracking_graph(
    predicted: LineageGraph,
    truth: LineageGraph,
) -> TrackingMetrics:
    if predicted.dataset != truth.dataset:
        raise ValueError(
            f"Dataset mismatch: predicted={predicted.dataset!r}, truth={truth.dataset!r}"
        )

    predicted_edges = predicted.edge_pairs()
    truth_edges = truth.edge_pairs()
    true_edges = predicted_edges & truth_edges

    edge_precision = _safe_ratio(len(true_edges), len(predicted_edges))
    edge_recall = _safe_ratio(len(true_edges), len(truth_edges))
    edge_union = predicted_edges | truth_edges
    edge_jaccard = _safe_ratio(len(true_edges), len(edge_union))

    predicted_divisions = predicted.division_events()
    truth_divisions = truth.division_events()
    true_divisions = predicted_divisions & truth_divisions
    division_precision = _safe_ratio(len(true_divisions), len(predicted_divisions))
    division_recall = _safe_ratio(len(true_divisions), len(truth_divisions))
    division_union = predicted_divisions | truth_divisions
    division_jaccard = _safe_ratio(len(true_divisions), len(division_union))

    return TrackingMetrics(
        dataset=truth.dataset,
        truth_nodes=len(truth.nodes),
        predicted_nodes=len(predicted.nodes),
        truth_edges=len(truth_edges),
        predicted_edges=len(predicted_edges),
        true_positive_edges=len(true_edges),
        edge_precision=edge_precision,
        edge_recall=edge_recall,
        edge_f1=_f1(edge_precision, edge_recall),
        edge_jaccard=edge_jaccard,
        truth_divisions=len(truth_divisions),
        predicted_divisions=len(predicted_divisions),
        true_positive_divisions=len(true_divisions),
        division_precision=division_precision,
        division_recall=division_recall,
        division_f1=_f1(division_precision, division_recall),
        division_jaccard=division_jaccard,
        proxy_score=edge_jaccard + 0.1 * division_jaccard,
    )
