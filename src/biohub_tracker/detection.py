from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_laplace, maximum_filter
from scipy.spatial import cKDTree

from .config import DetectorConfig
from .types import Detection


@dataclass(slots=True)
class BlobDetector3D:
    config: DetectorConfig
    scale_um: tuple[float, float, float]

    @staticmethod
    def _normalize(frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        finite = np.isfinite(frame)
        if not finite.any():
            return np.zeros_like(frame, dtype=np.float32)
        values = frame[finite]
        lo, hi = np.quantile(values, [0.01, 0.999])
        if hi <= lo:
            lo = float(np.min(values))
            hi = float(np.max(values))
        if hi <= lo:
            return np.zeros_like(frame, dtype=np.float32)
        out = np.clip((frame - lo) / (hi - lo), 0.0, 1.0)
        out[~finite] = 0.0
        return out.astype(np.float32, copy=False)

    def _physical_nms(self, coords: np.ndarray, scores: np.ndarray) -> np.ndarray:
        if len(coords) == 0:
            return np.empty(0, dtype=np.int64)
        order = np.argsort(scores)[::-1]
        coords_um = coords * np.asarray(self.scale_um, dtype=np.float64)
        tree = cKDTree(coords_um)
        suppressed = np.zeros(len(coords), dtype=bool)
        keep: list[int] = []
        for idx in order:
            if suppressed[idx]:
                continue
            keep.append(int(idx))
            neighbours = tree.query_ball_point(coords_um[idx], r=self.config.min_distance_um)
            suppressed[np.asarray(neighbours, dtype=np.int64)] = True
        return np.asarray(keep, dtype=np.int64)

    def _detect_chunk(
        self, image: np.ndarray, *, z_offset: int, keep_from: int, keep_to: int
    ) -> tuple[np.ndarray, np.ndarray]:
        response = -gaussian_laplace(image, sigma=self.config.sigma_vox, mode="nearest")
        response = np.maximum(response, 0.0)
        threshold = max(
            float(np.quantile(response, self.config.threshold_quantile)),
            float(self.config.absolute_threshold),
        )
        local_max = response == maximum_filter(response, size=(3, 3, 3), mode="nearest")
        coords = np.argwhere(local_max & (response >= threshold))
        if len(coords) == 0:
            return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float32)
        central = (coords[:, 0] >= keep_from) & (coords[:, 0] < keep_to)
        coords = coords[central]
        if len(coords) == 0:
            return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float32)
        scores = response[tuple(coords.T)].astype(np.float32, copy=False)
        coords[:, 0] += int(z_offset)
        return coords, scores

    def _candidate_peaks(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        depth = int(image.shape[0])
        patch_depth = max(int(self.config.patch_depth_z), 1)
        overlap = max(int(self.config.patch_overlap_z), 0)
        if depth <= patch_depth:
            return self._detect_chunk(image, z_offset=0, keep_from=0, keep_to=depth)
        if patch_depth <= 2 * overlap:
            raise ValueError("patch_depth_z must be greater than 2 * patch_overlap_z")

        coords_parts: list[np.ndarray] = []
        score_parts: list[np.ndarray] = []
        step = patch_depth - 2 * overlap
        core_start = 0
        while core_start < depth:
            core_end = min(core_start + step, depth)
            read_start = max(0, core_start - overlap)
            read_end = min(depth, core_end + overlap)
            keep_from = core_start - read_start
            keep_to = keep_from + (core_end - core_start)
            coords, scores = self._detect_chunk(
                image[read_start:read_end],
                z_offset=read_start,
                keep_from=keep_from,
                keep_to=keep_to,
            )
            if len(coords):
                coords_parts.append(coords)
                score_parts.append(scores)
            core_start = core_end
        if not coords_parts:
            return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float32)
        return np.concatenate(coords_parts), np.concatenate(score_parts)

    def detect(self, frame: np.ndarray, *, dataset: str, t: int, uid_start: int) -> list[Detection]:
        image = self._normalize(frame)
        coords, scores = self._candidate_peaks(image)
        if len(coords) == 0:
            return []
        keep = self._physical_nms(coords, scores)
        if len(keep) > self.config.max_detections_per_frame:
            ranked = np.argsort(scores[keep])[::-1][: self.config.max_detections_per_frame]
            keep = keep[ranked]

        result: list[Detection] = []
        scale = np.asarray(self.scale_um, dtype=np.float64)
        for offset, idx in enumerate(keep):
            coord = coords[idx].astype(np.float64)
            result.append(
                Detection(
                    uid=uid_start + offset,
                    dataset=dataset,
                    t=t,
                    centroid_vox=coord,
                    centroid_um=coord * scale,
                    score=float(scores[idx]),
                    features=np.asarray([float(image[tuple(coords[idx])])], dtype=np.float32),
                )
            )
        return result
