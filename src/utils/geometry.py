from typing import Tuple, Optional
import numpy as np


def compute_voxel_centroid(mask_binary: np.ndarray) -> Optional[np.ndarray]:
    tumor_indices = np.argwhere(mask_binary > 0)
    if tumor_indices.size == 0:
        return None
    return tumor_indices.mean(axis=0)


def compute_bounding_box(indices: np.ndarray, voxel_dimensions: Tuple[float, float, float]) -> Tuple[float, float, float]:
    min_coords = indices.min(axis=0)
    max_coords = indices.max(axis=0)
    bbox_voxels = max_coords - min_coords + 1
    bbox_x_mm = bbox_voxels[0] * voxel_dimensions[0]
    bbox_y_mm = bbox_voxels[1] * voxel_dimensions[1]
    bbox_z_mm = bbox_voxels[2] * voxel_dimensions[2]
    return float(bbox_x_mm), float(bbox_y_mm), float(bbox_z_mm)


def polygon_inside(rows: np.ndarray, cols: np.ndarray, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rmin = max(0, int(np.floor(rows.min())))
    rmax = min(shape[0] - 1, int(np.ceil(rows.max())))

    cmin = max(0, int(np.floor(cols.min())))
    cmax = min(shape[1] - 1, int(np.ceil(cols.max())))

    rr, cc = np.meshgrid(
        np.arange(rmin, rmax + 1),
        np.arange(cmin, cmax + 1),
        indexing="ij"
    )

    inside = np.zeros(rr.shape, dtype=bool)
    j = len(rows) - 1

    for i in range(len(rows)):
        condition = (
            ((rows[i] > rr) != (rows[j] > rr))
            & (
                cc < (cols[j] - cols[i]) * (rr - rows[i]) / (rows[j] - rows[i] + 1e-12) + cols[i]
            )
        )
        inside ^= condition
        j = i

    return rr, cc, inside


def shift_mask(mask: np.ndarray, dx_mm: float, dy_mm: float, dz_mm: float, sx: float = 1.0, sy: float = 1.0) -> np.ndarray:
    dx = int(round(dx_mm / sx))
    dy = int(round(dy_mm / sy))
    dz = int(round(dz_mm))

    shifted = np.zeros_like(mask)

    z_src_start = max(0, -dz)
    z_src_end = min(mask.shape[0], mask.shape[0] - dz)

    y_src_start = max(0, -dy)
    y_src_end = min(mask.shape[1], mask.shape[1] - dy)

    x_src_start = max(0, -dx)
    x_src_end = min(mask.shape[2], mask.shape[2] - dx)

    z_dst_start = max(0, dz)
    z_dst_end = min(mask.shape[0], mask.shape[0] + dz)

    y_dst_start = max(0, dy)
    y_dst_end = min(mask.shape[1], mask.shape[1] + dy)

    x_dst_start = max(0, dx)
    x_dst_end = min(mask.shape[2], mask.shape[2] + dx)

    if (
        z_src_end > z_src_start
        and y_src_end > y_src_start
        and x_src_end > x_src_start
    ):
        shifted[
            z_dst_start:z_dst_end,
            y_dst_start:y_dst_end,
            x_dst_start:x_dst_end,
        ] = mask[
            z_src_start:z_src_end,
            y_src_start:y_src_end,
            x_src_start:x_src_end,
        ]

    return shifted
