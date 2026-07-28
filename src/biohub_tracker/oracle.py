from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import pandas as pd

from .config import PipelineConfig
from .evaluation import TrackingMetrics, evaluate_tracking_graph
from .geff_io import discover_geff_stores, read_geff_lineage
from .submission import graph_to_submission, write_submission
from .tracking import BaselineTracker
from .types import LineageGraph


@dataclass(frozen=True, slots=True)
class OracleRunResult:
    truth: LineageGraph
    predicted: LineageGraph
    metrics: TrackingMetrics
    tracking_seconds: float


class OracleTrackingPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.tracker = BaselineTracker(self.config.tracker)

    def evaluate_truth_graph(self, truth: LineageGraph) -> OracleRunResult:
        frames = truth.frames()
        started = time.perf_counter()
        predicted = self.tracker.build_graph(frames, dataset=truth.dataset)
        elapsed = time.perf_counter() - started
        metrics = evaluate_tracking_graph(predicted, truth)
        return OracleRunResult(truth, predicted, metrics, elapsed)

    def evaluate_store(self, path: str | Path) -> OracleRunResult:
        truth = read_geff_lineage(path, fallback_scale_um=self.config.scale_um)
        return self.evaluate_truth_graph(truth)

    def run(
        self,
        train_dir: str | Path,
        *,
        output_path: str | Path | None = None,
        predictions_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        stores = discover_geff_stores(train_dir)
        if not stores:
            raise RuntimeError(f"No .geff stores found below {train_dir}")

        rows: list[dict[str, object]] = []
        predictions_root = Path(predictions_dir) if predictions_dir is not None else None
        if predictions_root is not None:
            predictions_root.mkdir(parents=True, exist_ok=True)

        for store in stores:
            result = self.evaluate_store(store)
            row = result.metrics.to_dict()
            row["tracking_seconds"] = result.tracking_seconds
            rows.append(row)
            if predictions_root is not None:
                submission = graph_to_submission([result.predicted])
                write_submission(
                    submission,
                    predictions_root / f"{result.predicted.dataset}.csv",
                )

        frame = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
        if output_path is not None:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output, index=False)
        return frame


def aggregate_oracle_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {"datasets": 0}
    numeric = [
        "edge_precision",
        "edge_recall",
        "edge_f1",
        "edge_jaccard",
        "division_precision",
        "division_recall",
        "division_f1",
        "division_jaccard",
        "proxy_score",
        "tracking_seconds",
    ]
    result: dict[str, float | int] = {"datasets": int(len(frame))}
    for column in numeric:
        if column in frame:
            result[f"mean_{column}"] = float(frame[column].mean())
    result["total_truth_nodes"] = int(frame["truth_nodes"].sum())
    result["total_truth_edges"] = int(frame["truth_edges"].sum())
    result["total_truth_divisions"] = int(frame["truth_divisions"].sum())
    return result
