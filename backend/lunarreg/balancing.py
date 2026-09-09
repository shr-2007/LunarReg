from __future__ import annotations

import numpy as np

from .types import MatchSet


def _cell_indices(points: np.ndarray, shape: tuple[int, int], rows: int, cols: int) -> np.ndarray:
    height, width = shape
    x = np.clip((points[:, 0] / max(width, 1) * cols).astype(int), 0, cols - 1)
    y = np.clip((points[:, 1] / max(height, 1) * rows).astype(int), 0, rows - 1)
    return y * cols + x


def grid_coverage(points: np.ndarray, shape: tuple[int, int], rows: int, cols: int) -> float:
    if len(points) == 0:
        return 0.0
    occupied = np.unique(_cell_indices(points, shape, rows, cols)).size
    return float(occupied / (rows * cols))


def spatial_balance(
    matches: MatchSet,
    source_shape: tuple[int, int],
    reference_shape: tuple[int, int],
    rows: int,
    cols: int,
    max_per_cell: int,
) -> tuple[MatchSet, np.ndarray]:
    if len(matches) == 0:
        return matches, np.empty((0,), dtype=int)
    source_cells = _cell_indices(matches.source_points, source_shape, rows, cols)
    reference_cells = _cell_indices(matches.reference_points, reference_shape, rows, cols)
    order = np.argsort(-matches.scores)
    source_counts: dict[int, int] = {}
    reference_counts: dict[int, int] = {}
    selected: list[int] = []

    for index in order:
        src_cell = int(source_cells[index])
        ref_cell = int(reference_cells[index])
        if source_counts.get(src_cell, 0) >= max_per_cell:
            continue
        if reference_counts.get(ref_cell, 0) >= max_per_cell:
            continue
        selected.append(int(index))
        source_counts[src_cell] = source_counts.get(src_cell, 0) + 1
        reference_counts[ref_cell] = reference_counts.get(ref_cell, 0) + 1

    indices = np.asarray(selected, dtype=int)
    return matches.subset(indices), indices
