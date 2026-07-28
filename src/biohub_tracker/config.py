from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    sigma_vox: tuple[float, float, float] = (1.0, 1.4, 1.4)
    threshold_quantile: float = 0.995
    absolute_threshold: float = 0.02
    min_distance_um: float = 3.0
    max_detections_per_frame: int = 5000
    patch_depth_z: int = 64
    patch_overlap_z: int = 8


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    max_link_distance_um: float = 10.0
    score_weight: float = 0.75
    division_parent_radius_um: float = 12.0
    division_sister_distance_um: float = 16.0
    division_max_children: int = 2


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    scale_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        payload: dict[str, Any] = json.loads(Path(path).read_text())
        return cls(
            scale_um=tuple(float(v) for v in payload.get("scale_um", cls().scale_um)),
            detector=DetectorConfig(**payload.get("detector", {})),
            tracker=TrackerConfig(**payload.get("tracker", {})),
        )
