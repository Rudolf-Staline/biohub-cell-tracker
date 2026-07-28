from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
import pandas as pd

from .config import PipelineConfig
from .detection import BlobDetector3D
from .submission import graph_to_submission, write_submission
from .tracking import BaselineTracker
from .types import LineageGraph
from .zarr_io import ZarrMovie, discover_zarr_stores


@dataclass(slots=True)
class DatasetRunStats:
    dataset: str
    frames: int
    nodes: int
    edges: int
    detection_seconds: float
    tracking_seconds: float


class BaselinePipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.detector = BlobDetector3D(self.config.detector, self.config.scale_um)
        self.tracker = BaselineTracker(self.config.tracker)

    def process_movie(self, movie: ZarrMovie, *, uid_start: int = 0) -> tuple[LineageGraph, DatasetRunStats, int]:
        frames = []
        next_uid = uid_start
        detection_started = time.perf_counter()
        for t in range(movie.n_frames):
            detections = self.detector.detect(
                movie.read_frame(t),
                dataset=movie.dataset_name,
                t=t,
                uid_start=next_uid,
            )
            frames.append(detections)
            next_uid += len(detections)
        detection_seconds = time.perf_counter() - detection_started

        tracking_started = time.perf_counter()
        graph = self.tracker.build_graph(frames, dataset=movie.dataset_name)
        tracking_seconds = time.perf_counter() - tracking_started
        stats = DatasetRunStats(
            dataset=movie.dataset_name,
            frames=movie.n_frames,
            nodes=len(graph.nodes),
            edges=len(graph.edges),
            detection_seconds=detection_seconds,
            tracking_seconds=tracking_seconds,
        )
        return graph, stats, next_uid

    def run(self, test_dir: str | Path, output_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        stores = discover_zarr_stores(test_dir)
        if not stores:
            raise RuntimeError(f"No .zarr test stores found below {test_dir}")
        graphs: list[LineageGraph] = []
        stats_rows: list[dict] = []
        next_uid = 0
        for store in stores:
            graph, stats, next_uid = self.process_movie(ZarrMovie.open(store), uid_start=next_uid)
            graphs.append(graph)
            stats_rows.append(asdict(stats))
        submission = graph_to_submission(graphs)
        write_submission(submission, output_path)
        return submission, pd.DataFrame(stats_rows)
