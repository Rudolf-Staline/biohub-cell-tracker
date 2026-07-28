from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .types import Detection, Edge, LineageGraph

_COORD_ALIASES = {
    "t": ("t", "time", "frame", "frame_id"),
    "z": ("z", "position_z", "pos_z"),
    "y": ("y", "position_y", "pos_y"),
    "x": ("x", "position_x", "pos_x"),
}


def _import_geff():
    try:
        import geff  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "GEFF support is optional. Install it with `pip install -e '.[oracle]'`."
        ) from exc
    return geff


def discover_geff_stores(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(path for path in root.rglob("*.geff") if path.is_dir())


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        payload = dump()
        return payload if isinstance(payload, Mapping) else None
    return None


def _axis_records(metadata: Any) -> list[Any]:
    if metadata is None:
        return []
    payload = _as_mapping(metadata)
    if payload is not None:
        axes = payload.get("axes", [])
    else:
        axes = getattr(metadata, "axes", [])
    return list(axes or [])


def _axis_value(axis: Any, key: str) -> Any:
    payload = _as_mapping(axis)
    if payload is not None:
        return payload.get(key)
    return getattr(axis, key, None)


def _axis_names(metadata: Any) -> list[str]:
    return [str(_axis_value(axis, "name") or "").lower() for axis in _axis_records(metadata)]


def _metadata_scale_um(
    metadata: Any,
    fallback: tuple[float, float, float],
) -> tuple[float, float, float]:
    scales: dict[str, float] = {}
    for axis in _axis_records(metadata):
        name = str(_axis_value(axis, "name") or "").lower()
        if name not in {"z", "y", "x"}:
            continue
        scale = _axis_value(axis, "scale")
        if scale is None:
            scale = _axis_value(axis, "axis_scale")
        if scale is not None:
            scales[name] = float(scale)
    if set(scales) == {"z", "y", "x"}:
        return scales["z"], scales["y"], scales["x"]
    return fallback


def _first_attr(attrs: Mapping[str, Any], names: Sequence[str]) -> Any | None:
    for name in names:
        if name in attrs:
            return attrs[name]
    return None


def _position_components(attrs: Mapping[str, Any], metadata: Any) -> dict[str, float]:
    position = _first_attr(attrs, ("position", "pos", "coordinates", "coord"))
    if position is None:
        return {}
    values = np.asarray(position, dtype=np.float64).reshape(-1)
    names = _axis_names(metadata)
    if names and len(names) == len(values):
        return {name: float(value) for name, value in zip(names, values)}
    if len(values) == 4:
        return dict(zip(("t", "z", "y", "x"), values.astype(float)))
    if len(values) == 3:
        return dict(zip(("z", "y", "x"), values.astype(float)))
    raise ValueError(f"Unsupported GEFF position vector with shape {values.shape}")


def _node_coordinates(attrs: Mapping[str, Any], metadata: Any) -> tuple[int, np.ndarray]:
    position = _position_components(attrs, metadata)
    components: dict[str, float] = {}
    for axis, aliases in _COORD_ALIASES.items():
        value = _first_attr(attrs, aliases)
        if value is None:
            value = position.get(axis)
        if value is None:
            raise ValueError(
                f"GEFF node misses coordinate {axis!r}; available attributes: {sorted(attrs)}"
            )
        components[axis] = float(value)
    t_float = components["t"]
    t = int(round(t_float))
    if not np.isclose(t_float, t):
        raise ValueError(f"GEFF time coordinate must be integral, got {t_float}")
    centroid = np.asarray([components["z"], components["y"], components["x"]])
    if not np.isfinite(centroid).all():
        raise ValueError("GEFF node coordinates must be finite")
    return t, centroid


def _stable_node_mapping(node_ids: Sequence[Any]) -> dict[Any, int]:
    integer_ids: dict[Any, int] = {}
    used: set[int] = set()
    preserve = True
    for node_id in node_ids:
        try:
            uid = int(node_id)
        except (TypeError, ValueError):
            preserve = False
            break
        if uid in used:
            preserve = False
            break
        integer_ids[node_id] = uid
        used.add(uid)
    if preserve:
        return integer_ids
    return {node_id: uid for uid, node_id in enumerate(node_ids)}


def lineage_graph_from_networkx(
    graph: Any,
    *,
    dataset: str,
    metadata: Any = None,
    fallback_scale_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
) -> LineageGraph:
    node_items = list(graph.nodes(data=True))
    node_mapping = _stable_node_mapping([node_id for node_id, _ in node_items])
    scale_um = np.asarray(_metadata_scale_um(metadata, fallback_scale_um), dtype=np.float64)

    result = LineageGraph(dataset=dataset)
    for node_id, raw_attrs in node_items:
        attrs = dict(raw_attrs)
        t, centroid_vox = _node_coordinates(attrs, metadata)
        result.add_nodes(
            [
                Detection(
                    uid=node_mapping[node_id],
                    dataset=dataset,
                    t=t,
                    centroid_vox=centroid_vox,
                    centroid_um=centroid_vox * scale_um,
                    score=float(attrs.get("score", 1.0)),
                )
            ]
        )

    oriented_edges: list[tuple[int, int]] = []
    for raw_source, raw_target in graph.edges():
        source = node_mapping[raw_source]
        target = node_mapping[raw_target]
        source_t = result.nodes[source].t
        target_t = result.nodes[target].t
        if source_t == target_t:
            raise ValueError(f"GEFF edge {raw_source!r}->{raw_target!r} stays in frame {source_t}")
        if source_t > target_t:
            source, target = target, source
            source_t, target_t = target_t, source_t
        if target_t != source_t + 1:
            raise ValueError(
                f"GEFF edge {raw_source!r}->{raw_target!r} spans {source_t}->{target_t}; "
                "oracle evaluation currently requires adjacent frames"
            )
        oriented_edges.append((source, target))

    child_counts: dict[int, int] = {}
    for source, _ in oriented_edges:
        child_counts[source] = child_counts.get(source, 0) + 1
    for source, target in oriented_edges:
        kind = "division" if child_counts[source] == 2 else "continuity"
        result.add_edge(Edge(source, target, kind=kind, score=1.0))
    return result


def read_geff_lineage(
    path: str | Path,
    *,
    fallback_scale_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    structure_validation: bool = True,
) -> LineageGraph:
    path = Path(path)
    geff = _import_geff()
    graph, metadata = geff.read(
        str(path),
        structure_validation=structure_validation,
        backend="networkx",
    )
    dataset = path.name[:-5] if path.name.endswith(".geff") else path.name
    return lineage_graph_from_networkx(
        graph,
        dataset=dataset,
        metadata=metadata,
        fallback_scale_um=fallback_scale_um,
    )


@dataclass(frozen=True, slots=True)
class GeffSummary:
    dataset: str
    frames: int
    nodes: int
    edges: int
    divisions: int


def summarize_geff_graph(graph: LineageGraph) -> GeffSummary:
    frames = {node.t for node in graph.nodes.values()}
    divisions = len(graph.division_events())
    return GeffSummary(
        dataset=graph.dataset,
        frames=len(frames),
        nodes=len(graph.nodes),
        edges=len(graph.edges),
        divisions=divisions,
    )
