from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .types import LineageGraph

CSV_COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]


@dataclass(slots=True)
class AuditReport:
    valid: bool
    rows: int
    node_rows: int
    edge_rows: int
    duplicate_nodes: int
    duplicate_edges: int
    missing_sources: int
    missing_targets: int
    self_loops: int
    non_next_frame_edges: int
    multiple_parents: int
    more_than_two_children: int
    non_finite_coordinates: int
    unknown_row_types: int
    errors: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def graph_to_submission(graphs: list[LineageGraph]) -> pd.DataFrame:
    rows: list[dict] = []
    for graph in graphs:
        for node_id, node in sorted(graph.nodes.items(), key=lambda item: (item[1].t, item[0])):
            z, y, x = node.centroid_vox.tolist()
            rows.append(
                {
                    "dataset": graph.dataset,
                    "row_type": "node",
                    "node_id": int(node_id),
                    "t": int(node.t),
                    "z": float(z),
                    "y": float(y),
                    "x": float(x),
                    "source_id": -1,
                    "target_id": -1,
                }
            )
        for edge in graph.edges:
            rows.append(
                {
                    "dataset": graph.dataset,
                    "row_type": "edge",
                    "node_id": -1,
                    "t": -1,
                    "z": -1.0,
                    "y": -1.0,
                    "x": -1.0,
                    "source_id": int(edge.source_uid),
                    "target_id": int(edge.target_uid),
                }
            )
    frame = pd.DataFrame(rows)
    frame.insert(0, "id", np.arange(len(frame), dtype=np.int64))
    return frame[CSV_COLUMNS]


def audit_submission(df: pd.DataFrame, *, enforce_next_frame: bool = True) -> AuditReport:
    errors: list[str] = []
    missing_columns = [column for column in CSV_COLUMNS if column not in df.columns]
    if missing_columns:
        return AuditReport(
            valid=False,
            rows=len(df),
            node_rows=0,
            edge_rows=0,
            duplicate_nodes=0,
            duplicate_edges=0,
            missing_sources=0,
            missing_targets=0,
            self_loops=0,
            non_next_frame_edges=0,
            multiple_parents=0,
            more_than_two_children=0,
            non_finite_coordinates=0,
            unknown_row_types=0,
            errors=[f"missing columns: {missing_columns}"],
        )

    work = df.copy()
    nodes = work[work["row_type"] == "node"]
    edges = work[work["row_type"] == "edge"]
    unknown = int((~work["row_type"].isin(["node", "edge"])).sum())
    duplicate_nodes = int(nodes.duplicated(["dataset", "node_id"]).sum())
    duplicate_edges = int(edges.duplicated(["dataset", "source_id", "target_id"]).sum())
    self_loops = int((edges["source_id"] == edges["target_id"]).sum())
    finite = np.isfinite(nodes[["z", "y", "x"]].to_numpy(dtype=float)).all(axis=1)
    non_finite = int((~finite).sum())

    node_keys = set(zip(nodes["dataset"].astype(str), nodes["node_id"].astype(int)))
    missing_sources = sum(
        (str(dataset), int(source)) not in node_keys
        for dataset, source in zip(edges["dataset"], edges["source_id"])
    )
    missing_targets = sum(
        (str(dataset), int(target)) not in node_keys
        for dataset, target in zip(edges["dataset"], edges["target_id"])
    )
    time_by_node = {
        (str(dataset), int(node_id)): int(t)
        for dataset, node_id, t in zip(nodes["dataset"], nodes["node_id"], nodes["t"])
    }
    non_next = 0
    for dataset, source, target in zip(edges["dataset"], edges["source_id"], edges["target_id"]):
        source_t = time_by_node.get((str(dataset), int(source)))
        target_t = time_by_node.get((str(dataset), int(target)))
        if source_t is None or target_t is None:
            continue
        if target_t <= source_t or (enforce_next_frame and target_t != source_t + 1):
            non_next += 1

    parent_counts = edges.groupby(["dataset", "target_id"]).size() if len(edges) else pd.Series(dtype=int)
    child_counts = edges.groupby(["dataset", "source_id"]).size() if len(edges) else pd.Series(dtype=int)
    multiple_parents = int((parent_counts > 1).sum())
    more_than_two_children = int((child_counts > 2).sum())

    checks = {
        "unknown row types": unknown,
        "duplicate nodes": duplicate_nodes,
        "duplicate edges": duplicate_edges,
        "missing sources": int(missing_sources),
        "missing targets": int(missing_targets),
        "self loops": self_loops,
        "invalid temporal edges": non_next,
        "targets with multiple parents": multiple_parents,
        "sources with >2 children": more_than_two_children,
        "non-finite coordinates": non_finite,
    }
    errors.extend(f"{name}: {value}" for name, value in checks.items() if value)
    return AuditReport(
        valid=not errors,
        rows=len(work),
        node_rows=len(nodes),
        edge_rows=len(edges),
        duplicate_nodes=duplicate_nodes,
        duplicate_edges=duplicate_edges,
        missing_sources=int(missing_sources),
        missing_targets=int(missing_targets),
        self_loops=self_loops,
        non_next_frame_edges=non_next,
        multiple_parents=multiple_parents,
        more_than_two_children=more_than_two_children,
        non_finite_coordinates=non_finite,
        unknown_row_types=unknown,
        errors=errors,
    )


def write_submission(df: pd.DataFrame, path: str | Path) -> AuditReport:
    report = audit_submission(df)
    if not report.valid:
        raise ValueError(f"Refusing to write invalid submission: {report.errors}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return report
