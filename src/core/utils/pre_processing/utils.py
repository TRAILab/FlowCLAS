"""Random convex polygon masks for synthetic anomaly injection."""

import numpy as np
import torch
from scipy.spatial import ConvexHull
from skimage.draw import polygon as draw_polygon


def generate_polygon(
    height: int,
    width: int,
    num_points: int,
    box_ratios: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Sample a random convex polygon inside a box sized as a fraction of the image.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        num_points: Number of random vertices before taking the convex hull.
        box_ratios: ``(min_ratio, max_ratio)`` for box side lengths relative to
            ``height`` and ``width``.

    Returns:
        Row indices, column indices (local to the box), box width, and box height.
        Callers offset ``rr``/``cc`` to place the patch on a full image.
    """
    min_ratio, max_ratio = box_ratios
    h_min, h_max = int(min_ratio * height), int(max_ratio * height)
    w_min, w_max = int(min_ratio * width), int(max_ratio * width)

    box_h = torch.randint(h_min, h_max, (1,)).item()
    box_w = torch.randint(w_min, w_max, (1,)).item()

    x = (torch.rand(num_points, 1) * box_w).numpy()
    y = (torch.rand(num_points, 1) * box_h).numpy()
    points = np.concatenate([x, y], axis=1)

    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    rr, cc = draw_polygon(vertices[:, 0], vertices[:, 1], shape=(box_w, box_h))

    return rr, cc, box_w, box_h
