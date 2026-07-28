from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import TrackerConfig
from .types import Detection, Edge, LineageGraph


@dataclass(slots=True)
class BaselineTracker:
    config: TrackerConfig

    def _cost_matrix(self, source: list[Detection], target: list[Detection]) -> np.ndarray:
        if not source or not target:
            return np.empty((len(source), len(target)), dtype=np.float64)
        a = np.stack([d.centroid_um for d in source])
        b = np.stack([d.centroid_um for d in target])
        distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        source_scores = np.asarray([d.score for d in source])[:, None]
        target_scores = np.asarray([d.score for d in target])[None, :]
        score_delta = np.abs(source_scores - target_scores)
        return distances + self.config.score_weight * score_delta

    def _normal_links(
        self, source: list[Detection], target: list[Detection]
    ) -> tuple[list[Edge], set[int], set[int]]:
        if not source or not target:
            return [], set(), set()
        cost = self._cost_matrix(source, target)
        gated = cost.copy()
        physical = np.linalg.norm(
            np.stack([d.centroid_um for d in source])[:, None, :]
            - np.stack([d.centroid_um for d in target])[None, :, :],
            axis=2,
        )
        gated[physical > self.config.max_link_distance_um] = 1e9
        rows, cols = linear_sum_assignment(gated)
        edges: list[Edge] = []
        used_s: set[int] = set()
        used_t: set[int] = set()
        for r, c in zip(rows, cols):
            if gated[r, c] >= 1e8:
                continue
            edges.append(
                Edge(
                    source_uid=source[r].uid,
                    target_uid=target[c].uid,
                    kind="continuity",
                    score=float(-gated[r, c]),
                )
            )
            used_s.add(r)
            used_t.add(c)
        return edges, used_s, used_t

    def _division_candidates(
        self,
        source: list[Detection],
        target: list[Detection],
        normal_edges: list[Edge],
        used_s: set[int],
        used_t: set[int],
    ) -> list[tuple[float, int, int, int, int | None]]:
        source_uid_to_index = {d.uid: i for i, d in enumerate(source)}
        target_uid_to_index = {d.uid: i for i, d in enumerate(target)}
        matched_target_by_source: dict[int, int] = {}
        edge_index_by_source: dict[int, int] = {}
        for edge_idx, edge in enumerate(normal_edges):
            si = source_uid_to_index[edge.source_uid]
            ti = target_uid_to_index[edge.target_uid]
            matched_target_by_source[si] = ti
            edge_index_by_source[si] = edge_idx

        candidates: list[tuple[float, int, int, int, int | None]] = []
        unmatched_targets = [i for i in range(len(target)) if i not in used_t]
        for si, parent in enumerate(source):
            child_pool = list(unmatched_targets)
            existing = matched_target_by_source.get(si)
            if existing is not None:
                child_pool.append(existing)
            if len(child_pool) < 2:
                continue
            for a, b in combinations(sorted(set(child_pool)), 2):
                ca, cb = target[a], target[b]
                da = float(np.linalg.norm(ca.centroid_um - parent.centroid_um))
                db = float(np.linalg.norm(cb.centroid_um - parent.centroid_um))
                ds = float(np.linalg.norm(ca.centroid_um - cb.centroid_um))
                if max(da, db) > self.config.division_parent_radius_um:
                    continue
                if ds > self.config.division_sister_distance_um:
                    continue
                bary = (ca.centroid_um + cb.centroid_um) / 2.0
                bary_error = float(np.linalg.norm(bary - parent.centroid_um))
                score = da + db + 0.5 * ds + bary_error
                replaced_edge = edge_index_by_source.get(si) if existing in (a, b) else None
                candidates.append((score, si, a, b, replaced_edge))
        return sorted(candidates, key=lambda item: item[0])

    def link_pair(self, source: list[Detection], target: list[Detection]) -> list[Edge]:
        normal_edges, used_s, used_t = self._normal_links(source, target)
        candidates = self._division_candidates(source, target, normal_edges, used_s, used_t)

        consumed_sources: set[int] = set()
        consumed_targets: set[int] = set(used_t)
        removed_edges: set[int] = set()
        division_edges: list[Edge] = []

        source_uid_to_index = {d.uid: i for i, d in enumerate(source)}
        target_uid_to_index = {d.uid: i for i, d in enumerate(target)}
        matched_target_by_source = {
            source_uid_to_index[e.source_uid]: target_uid_to_index[e.target_uid]
            for e in normal_edges
        }

        for score, si, a, b, replaced_edge in candidates:
            if si in consumed_sources:
                continue
            existing = matched_target_by_source.get(si)
            allowed_existing = {existing} if existing is not None else set()
            if (a in consumed_targets and a not in allowed_existing) or (
                b in consumed_targets and b not in allowed_existing
            ):
                continue
            if replaced_edge is not None:
                removed_edges.add(replaced_edge)
            consumed_sources.add(si)
            consumed_targets.update((a, b))
            division_edges.extend(
                [
                    Edge(source[si].uid, target[a].uid, "division", float(-score)),
                    Edge(source[si].uid, target[b].uid, "division", float(-score)),
                ]
            )

        kept_normal = [e for i, e in enumerate(normal_edges) if i not in removed_edges]
        return kept_normal + division_edges

    def build_graph(self, frames: list[list[Detection]], *, dataset: str) -> LineageGraph:
        graph = LineageGraph(dataset=dataset)
        for detections in frames:
            graph.add_nodes(detections)
        for current, nxt in zip(frames, frames[1:]):
            for edge in self.link_pair(current, nxt):
                graph.add_edge(edge)
        return graph
