from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import numpy as np


@dataclass(slots=True)
class Detection:
    uid: int
    dataset: str
    t: int
    centroid_vox: np.ndarray
    centroid_um: np.ndarray
    score: float
    features: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    def __post_init__(self) -> None:
        self.centroid_vox = np.asarray(self.centroid_vox, dtype=np.float64)
        self.centroid_um = np.asarray(self.centroid_um, dtype=np.float64)
        self.features = np.asarray(self.features, dtype=np.float32)
        if self.centroid_vox.shape != (3,) or self.centroid_um.shape != (3,):
            raise ValueError("centroid_vox and centroid_um must have shape (3,)")
        if not np.isfinite(self.centroid_vox).all() or not np.isfinite(self.centroid_um).all():
            raise ValueError("centroid coordinates must be finite")


@dataclass(frozen=True, slots=True)
class Edge:
    source_uid: int
    target_uid: int
    kind: str = "continuity"
    score: float = 0.0


@dataclass(slots=True)
class LineageGraph:
    dataset: str
    nodes: dict[int, Detection] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_nodes(self, detections: Iterable[Detection]) -> None:
        for detection in detections:
            if detection.uid in self.nodes:
                raise ValueError(f"duplicate node id {detection.uid}")
            self.nodes[detection.uid] = detection

    def add_edge(self, edge: Edge) -> None:
        if edge.source_uid not in self.nodes or edge.target_uid not in self.nodes:
            raise ValueError("edge endpoints must exist before adding an edge")
        self.edges.append(edge)
