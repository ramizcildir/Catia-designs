"""
postprocessing.py

Helpers to map element-wise stresses to nodal values, compute coloring, and scale deformations for visualization.
"""

import numpy as np


def element_centroids(nodes, elems):
    centroids = nodes[elems].mean(axis=1)
    return centroids


def map_element_to_nodes_scalar(elem_values, elems, n_nodes):
    """
    Simple averaging mapping from element scalar values (e.g., von Mises per element)
    to nodal scalar values by averaging values of adjacent elements.
    """
    nodal = np.zeros(n_nodes, dtype=float)
    counts = np.zeros(n_nodes, dtype=int)
    for i, tri in enumerate(elems):
        val = elem_values[i]
        for n in tri:
            nodal[n] += val
            counts[n] += 1
    # avoid division by zero
    nonzero = counts > 0
    nodal[nonzero] = nodal[nonzero] / counts[nonzero]
    return nodal


def normalize_colors(values, cmap_min=None, cmap_max=None):
    """
    Normalize scalar values to [0,1] for color mapping.
    If cmap_min or cmap_max provided, use them; otherwise use min/max of values.
    """
    if cmap_min is None:
        cmap_min = float(np.nanmin(values))
    if cmap_max is None:
        cmap_max = float(np.nanmax(values))
    rng = cmap_max - cmap_min
    if rng <= 0:
        return np.zeros_like(values)
    norm = (values - cmap_min) / rng
    norm = np.clip(norm, 0.0, 1.0)
    return norm


def scalar_to_rgb(norm_values, cmap="hot"):
    """
    Map normalized values [0,1] to RGB tuple (0-255).
    We'll implement a simple 'hot' colormap (black->red->yellow->white).
    """
    rgb = []
    for v in norm_values:
        # hot map: 0->black, 0.33->red, 0.66->yellow, 1.0->white
        if v <= 0.33:
            # black to red
            t = v / 0.33
            r = int(255 * t)
            g = 0
            b = 0
        elif v <= 0.66:
            # red to yellow
            t = (v - 0.33) / (0.33)
            r = 255
            g = int(255 * t)
            b = 0
        else:
            # yellow to white
            t = (v - 0.66) / (0.34)
            r = 255
            g = 255
            b = int(255 * t)
        rgb.append((r, g, b))
    return np.array(rgb, dtype=np.uint8)


def scale_displacements_for_visual(u, scale_factor=None, nodes=None):
    """
    Determine scaling factor if not provided based on nodal positions to make deformation visible.
    Returns scaled displacement array (n_nodes,2) and the scale used.
    """
    n = u.size // 2
    disp = u.reshape((n, 2))
    if scale_factor is None:
        # use bounding box size to pick a reasonable scale
        if nodes is None:
            scale_factor = 1.0
        else:
            bbox = nodes.max(axis=0) - nodes.min(axis=0)
            maxdim = max(bbox[0], bbox[1])
            # aim for max displacement to be ~5% of maxdim
            maxdisp = np.max(np.linalg.norm(disp, axis=1)) if np.any(disp)
