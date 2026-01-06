import numpy as np
import pyvista as pv

def polyline_to_segments(poly: pv.PolyData, close=False) -> pv.PolyData:
    """
    Convert any polyline(s) in `poly` into individual 2-point line cells.
    Keeps the same points array; rewrites `lines` connectivity.
    Adds cell_data:
      - 'parent_polyline': which original polyline cell it came from
      - 'seg_index': segment index along that polyline
    """
    if poly.lines.size == 0:
        raise ValueError("Input PolyData has no line cells (poly.lines is empty).")

    # Parse VTK 'lines' connectivity: [n, id0, id1, ..., n, ...]
    arr = np.asarray(poly.lines).astype(np.int64)
    i = 0
    segments = []
    parent = []
    seg_index = []

    polyline_id = 0
    while i < len(arr):
        n = int(arr[i])
        ids = arr[i+1:i+1+n]
        if n < 2:
            i += n + 1
            polyline_id += 1
            continue

        # consecutive pairs
        for j in range(n - 1):
            segments.append((int(ids[j]), int(ids[j+1])))
            parent.append(polyline_id)
            seg_index.append(j)

        # optional closing segment
        if close and n > 2:
            segments.append((int(ids[-1]), int(ids[0])))
            parent.append(polyline_id)
            seg_index.append(n - 1)

        i += n + 1
        polyline_id += 1

    segments = np.asarray(segments, dtype=np.int64)
    lines = np.empty((segments.shape[0], 3), dtype=np.int64)
    lines[:, 0] = 2
    lines[:, 1:] = segments
    lines = lines.ravel()

    seg_poly = pv.PolyData(poly.points.copy())
    seg_poly.lines = lines
    seg_poly.cell_data["parent_polyline"] = np.asarray(parent, dtype=np.int64)
    seg_poly.cell_data["seg_index"] = np.asarray(seg_index, dtype=np.int64)
    return seg_poly

# Example:
# pline = pv.read("boundary_polyline.vtp")
# segs = polyline_to_segments(pline, close=False)
# segs.save("boundary_segments.vtp")

def _cell_has_edge(vtk_grid, cell_id: int, p0: int, p1: int) -> bool:
    """Check if vtk cell has an edge consisting of point ids (p0,p1)."""
    cell = vtk_grid.GetCell(int(cell_id))
    # Some cell types may not implement edges; handle carefully.
    ne = cell.GetNumberOfEdges()
    if ne > 0:
        for e in range(ne):
            edge = cell.GetEdge(e)
            pid0 = edge.GetPointId(0)
            pid1 = edge.GetPointId(1)
            if (pid0 == p0 and pid1 == p1) or (pid0 == p1 and pid1 == p0):
                return True
        return False

    # Fallback: for simple linear cells, approximate by checking consecutive connectivity
    # (works for triangles/quads/lines in many cases).
    pt_ids = [cell.GetPointId(k) for k in range(cell.GetNumberOfPoints())]
    if p0 not in pt_ids or p1 not in pt_ids:
        return False
    # If no explicit edges, we conservatively accept "both points are in cell"
    # You can tighten this for your known cell types if needed.
    return True


def map_segments_to_vtu_cells(
    seg_poly: pv.PolyData,
    grid: pv.UnstructuredGrid,
    tol: float = 1e-6
):
    """
    For each 2-point line cell in seg_poly, find the VTU cell(s) that contain that edge.

    Returns:
      - cell0: (nseg,) first matching cell id or -1
      - cell1: (nseg,) second matching cell id or -1 (useful for interior edges)
      - snapped_pids: (nseg,2) point ids in VTU used for matching
    Also adds cell_data arrays to seg_poly.
    """
    if seg_poly.n_cells == 0:
        raise ValueError("seg_poly has no line cells.")
    if grid.n_points == 0 or grid.n_cells == 0:
        raise ValueError("grid is empty.")

    vtk_grid = grid.GetDataSet()  # underlying vtkUnstructuredGrid
    # Ensure point->cell links exist for fast point_cell_ids queries
    vtk_grid.BuildLinks()

    cell0 = -np.ones(seg_poly.n_cells, dtype=np.int64)
    cell1 = -np.ones(seg_poly.n_cells, dtype=np.int64)
    snapped = np.full((seg_poly.n_cells, 2), -1, dtype=np.int64)

    # Each line cell has exactly 2 points (by construction above)
    for li in range(seg_poly.n_cells):
        ids = seg_poly.get_cell(li).point_ids
        a = seg_poly.points[ids[0]]
        b = seg_poly.points[ids[1]]

        # Snap to closest VTU points
        pid_a = int(grid.find_closest_point(a))
        pid_b = int(grid.find_closest_point(b))

        # Check snapping tolerance
        if np.linalg.norm(grid.points[pid_a] - a) > tol or np.linalg.norm(grid.points[pid_b] - b) > tol:
            # If this happens often, increase tol or ensure your polyline points are taken from the mesh.
            snapped[li] = [-1, -1]
            continue

        snapped[li] = [pid_a, pid_b]

        # Candidate cells are intersection of cells incident to each endpoint
        ca = set(map(int, grid.point_cell_ids(pid_a)))
        cb = set(map(int, grid.point_cell_ids(pid_b)))
        candidates = list(ca.intersection(cb))

        # Filter candidates to those that truly have the edge
        matches = []
        for cid in candidates:
            if _cell_has_edge(vtk_grid, cid, pid_a, pid_b):
                matches.append(cid)

        if len(matches) >= 1:
            cell0[li] = matches[0]
        if len(matches) >= 2:
            cell1[li] = matches[1]
        # If you expect >2 in 3D, keep `matches` in a dict instead (see note below).

    seg_poly.cell_data["vtu_cell0"] = cell0
    seg_poly.cell_data["vtu_cell1"] = cell1
    seg_poly.cell_data["vtu_pid0"] = snapped[:, 0]
    seg_poly.cell_data["vtu_pid1"] = snapped[:, 1]
    return cell0, cell1, snapped

# Example:
# grid = pv.read("mesh.vtu")
# segs = pv.read("boundary_segments.vtp")
# map_segments_to_vtu_cells(segs, grid, tol=1e-6)
# segs.save("boundary_segments_mapped.vtp")

