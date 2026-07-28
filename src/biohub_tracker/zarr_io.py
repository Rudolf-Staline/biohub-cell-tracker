from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
import numpy as np


def _import_zarr():
    try:
        import zarr  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "zarr is required to read competition volumes. Install the project dependencies."
        ) from exc
    return zarr


def discover_zarr_stores(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    stores = sorted(path for path in root.rglob("*.zarr") if path.is_dir())
    return stores


def _walk_arrays(group: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    try:
        for name, array in group.arrays():
            path = f"{prefix}/{name}".strip("/")
            yield path, array
    except Exception:
        pass
    try:
        for name, child in group.groups():
            child_prefix = f"{prefix}/{name}".strip("/")
            yield from _walk_arrays(child, child_prefix)
    except Exception:
        pass


def _axes_from_attrs(attrs: dict[str, Any], ndim: int) -> list[str] | None:
    raw = attrs.get("_ARRAY_DIMENSIONS") or attrs.get("axes")
    if raw is None:
        return None
    axes: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            axes.append(str(item.get("name", "")).lower())
        else:
            axes.append(str(item).lower())
    return axes if len(axes) == ndim else None


@dataclass(slots=True)
class ZarrMovie:
    path: Path
    array: Any
    axes: list[str]

    @property
    def dataset_name(self) -> str:
        name = self.path.name
        return name[:-5] if name.endswith(".zarr") else name

    @property
    def n_frames(self) -> int:
        return int(self.array.shape[self.axes.index("t")])

    @classmethod
    def open(cls, path: str | Path) -> "ZarrMovie":
        zarr = _import_zarr()
        path = Path(path)
        root = zarr.open_group(str(path), mode="r")

        attrs = dict(getattr(root, "attrs", {}))
        multiscales = attrs.get("multiscales")
        if multiscales:
            entry = multiscales[0]
            dataset_path = entry["datasets"][0]["path"]
            array = root[dataset_path]
            axes = _axes_from_attrs({"axes": entry.get("axes")}, array.ndim)
            if axes and {"t", "z", "y", "x"}.issubset(axes):
                return cls(path=path, array=array, axes=axes)

        candidates = list(_walk_arrays(root))
        candidates = [(name, arr) for name, arr in candidates if getattr(arr, "ndim", 0) >= 4]
        if not candidates:
            raise RuntimeError(f"No 4D/5D image array found in {path}")
        _, array = max(candidates, key=lambda pair: int(np.prod(pair[1].shape)))
        axes = _axes_from_attrs(dict(getattr(array, "attrs", {})), array.ndim)
        if axes is None:
            if array.ndim == 4:
                axes = ["t", "z", "y", "x"]
            elif array.ndim == 5 and array.shape[1] <= 4:
                axes = ["t", "c", "z", "y", "x"]
            elif array.ndim == 5 and array.shape[-1] <= 4:
                axes = ["t", "z", "y", "x", "c"]
            else:
                raise RuntimeError(f"Cannot infer axes for array shape {array.shape}")
        if not {"t", "z", "y", "x"}.issubset(axes):
            raise RuntimeError(f"Required axes t,z,y,x missing from {axes}")
        return cls(path=path, array=array, axes=axes)

    def read_frame(self, t: int, channel: int = 0) -> np.ndarray:
        index: list[Any] = [slice(None)] * self.array.ndim
        index[self.axes.index("t")] = int(t)
        if "c" in self.axes:
            index[self.axes.index("c")] = int(channel)
        frame = np.asarray(self.array[tuple(index)])
        remaining_axes = [axis for axis, idx in zip(self.axes, index) if isinstance(idx, slice)]
        order = [remaining_axes.index(axis) for axis in ("z", "y", "x")]
        frame = np.transpose(frame, axes=order)
        if frame.ndim != 3:
            raise RuntimeError(f"Expected a 3D frame, got {frame.shape}")
        return frame
