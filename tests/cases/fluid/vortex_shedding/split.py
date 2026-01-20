import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
from pyvista import _vtk

Edge = Tuple[int, int]  # (min_pid, max_pid)


@dataclass
class BoundarySegment:
    """One extracted boundary segment (as line cells) with parent-triangle mapping."""
    name: str
    poly: pv.PolyData  # line-only PolyData
    filename: Optional[str] = None


def extract_boundary_segments_by_crease_angle_with_parent_ids(
    mesh: pv.UnstructuredGrid,
    *,
    crease_angle_deg: float = 45.0,
    save_dir: Optional[str] = None,
    prefix: str = "boundary",
    clean_tol: float = 1e-12,
) -> List[BoundarySegment]:
    """
    Given a planar 2D triangulated mesh (VTU / UnstructuredGrid with triangle cells),
    extract boundary edges, split each connected boundary loop by turning ("crease")
    angle > crease_angle_deg, and return line-only PolyData segments.

    For every LINE cell in each output PolyData:
      - cell_data['GlobalElementID'] is set to the parent TRIANGLE cell ID (0-based)
        from the input mesh that owns that boundary edge.

    Guarantees:
      - Outputs contain ONLY VTK_LINE cells (2-point segments): no polylines, no verts.
      - GlobalElementID length == number of line cells in the output.

    Notes:
      - "Crease angle" here means 2D turning angle along the boundary polyline,
        not a 3D dihedral angle (planar meshes have no dihedral creases).
      - Assumes a *manifold* boundary where vertices have degree 2 on closed loops
        (or degree 1 on open chains).
    """

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    if mesh.n_cells == 0 or mesh.n_points == 0:
        return []

    pts = np.asarray(mesh.points)
    angle_thr = math.radians(float(crease_angle_deg))

    # ---------------------------
    # 1) Build edge -> owner triangle id, and edge -> count
    # ---------------------------
    edge_count: Dict[Edge, int] = {}
    edge_owner: Dict[Edge, int] = {}  # only meaningful when count==1

    # Iterate only triangle cells
    # (Robust fallback: check cell type per cell)
    VTK_TRIANGLE = 5
    for cid in range(mesh.n_cells):
        if int(mesh.celltypes[cid]) != VTK_TRIANGLE:
            continue
        cell = mesh.get_cell(cid)
        ids = list(cell.point_ids)
        if len(ids) != 3:
            continue

        e01 = (ids[0], ids[1])
        e12 = (ids[1], ids[2])
        e20 = (ids[2], ids[0])

        for a, b in (e01, e12, e20):
            e = (a, b) if a < b else (b, a)
            edge_count[e] = edge_count.get(e, 0) + 1
            # keep the first owner; if it becomes interior later, we'll ignore by count != 1
            if e not in edge_owner:
                edge_owner[e] = cid

    # Boundary edges are those belonging to exactly one triangle
    boundary_edges: List[Edge] = [e for e, c in edge_count.items() if c == 1]
    if not boundary_edges:
        return []

    # ---------------------------
    # 2) Build adjacency for boundary edges and connected components (by vertices)
    # ---------------------------
    adj: Dict[int, List[int]] = {}
    for a, b in boundary_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    # remove duplicates in adjacency lists
    for k in list(adj.keys()):
        adj[k] = list(dict.fromkeys(adj[k]))

    # Connected components (vertex sets)
    seen = set()
    comps: List[List[int]] = []
    for v in adj.keys():
        if v in seen:
            continue
        stack = [v]
        seen.add(v)
        comp = []
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj.get(x, []):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(comp)

    # Helper: order edges around a component (walk the boundary)
    def order_component_vertices_and_edges(comp_vertices: List[int]) -> Tuple[List[int], List[Edge], bool]:
        comp_set = set(comp_vertices)

        # degrees restricted to component
        deg = {v: sum((n in comp_set) for n in adj.get(v, [])) for v in comp_vertices}
        endpoints = [v for v, d in deg.items() if d == 1]
        is_closed = (len(endpoints) == 0)

        # pick start
        start = endpoints[0] if endpoints else min(comp_vertices)

        ordered_vertices: List[int] = [start]
        ordered_edges: List[Edge] = []

        prev = -1
        cur = start

        # safety limit
        max_steps = len(comp_vertices) + 10

        for _ in range(max_steps):
            nbrs = [n for n in adj.get(cur, []) if n in comp_set]
            if not nbrs:
                break

            # choose next neighbor not equal to prev; if first step in closed loop, take any
            nxt = None
            for n in nbrs:
                if n != prev:
                    nxt = n
                    break

            if nxt is None:
                break

            # record edge
            e = (cur, nxt) if cur < nxt else (nxt, cur)
            ordered_edges.append(e)

            if is_closed and nxt == start:
                # closed loop completed; do not append duplicate start to vertices list
                break

            ordered_vertices.append(nxt)

            prev, cur = cur, nxt

            # stop at the other endpoint for open chain
            if (not is_closed) and (deg.get(cur, 0) == 1) and (cur != start):
                break

        # For a closed loop, ordered_vertices should have length == number of edges,
        # because we didn't append the final "start" again.
        return ordered_vertices, ordered_edges, is_closed

    # Helper: compute turning angles at vertices
    def turning_angles(vertices: List[int], is_closed: bool) -> np.ndarray:
        V = len(vertices)
        if V < 3:
            return np.zeros((V,), dtype=float)

        angles = np.zeros((V,), dtype=float)

        def angle_at(i_prev: int, i: int, i_next: int) -> float:
            p_prev = pts[vertices[i_prev]]
            p = pts[vertices[i]]
            p_next = pts[vertices[i_next]]
            v0 = p - p_prev
            v1 = p_next - p
            n0 = np.linalg.norm(v0)
            n1 = np.linalg.norm(v1)
            if n0 == 0.0 or n1 == 0.0:
                return 0.0
            c = float(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))
            return float(np.arccos(c))

        if is_closed:
            for i in range(V):
                angles[i] = angle_at((i - 1) % V, i, (i + 1) % V)
        else:
            for i in range(1, V - 1):
                angles[i] = angle_at(i - 1, i, i + 1)

        return angles

    # Helper: split edge indices by crease vertices
    def split_edges_by_crease(vertices: List[int], edges: List[Edge], is_closed: bool) -> List[List[int]]:
        m = len(edges)
        if m == 0:
            return []

        ang = turning_angles(vertices, is_closed)

        if is_closed:
            # vertices length should be m for a simple closed walk
            cut_vertices = [i for i in range(len(vertices)) if ang[i] > angle_thr]
            cut_vertices.sort()
            if not cut_vertices:
                return [list(range(m))]

            segs: List[List[int]] = []
            for k in range(len(cut_vertices)):
                a = cut_vertices[k]
                b = cut_vertices[(k + 1) % len(cut_vertices)]
                if a == b:
                    continue
                # edges start at a, end at b-1 (wrapping)
                idxs = []
                i = a
                while True:
                    idxs.append(i % m)
                    i = (i + 1) % m
                    if i % m == b % m:
                        break
                # avoid empty/degenerate segments
                if len(idxs) >= 1:
                    segs.append(idxs)
            return segs
        else:
            # vertices length is m+1 for open chain (usually)
            cut_vertices = [i for i in range(1, len(vertices) - 1) if ang[i] > angle_thr]
            cut_vertices.sort()
            if not cut_vertices:
                return [list(range(m))]

            segs: List[List[int]] = []
            start_edge = 0
            for cv in cut_vertices:
                end_edge = cv - 1
                if end_edge >= start_edge:
                    segs.append(list(range(start_edge, end_edge + 1)))
                start_edge = cv
            if start_edge <= m - 1:
                segs.append(list(range(start_edge, m)))
            return [s for s in segs if len(s) >= 1]

    # Helper: build line-only PolyData for a set of edges (by edge indices into `edges`)
    def build_segment_polydata(edges, edge_indices, *, pts, edge_owner, clean_tol=1e-12):
        """
        Build a line-only PolyData (VTK_LINE cells only; no verts/polys/strips),
        and attach GlobalElementID per line cell mapping to parent triangle id.
        """
        used_pids = set()
        seg_edges = []
        seg_parent = []

        for ei in edge_indices:
            e = edges[ei]
            seg_edges.append(e)
            used_pids.add(e[0])
            used_pids.add(e[1])
            seg_parent.append(int(edge_owner[e]))  # parent triangle id in original mesh

        used_pids = sorted(used_pids)
        pid_map = {old: new for new, old in enumerate(used_pids)}
        seg_points = pts[np.array(used_pids, dtype=int)]

        nlines = len(seg_edges)
        if nlines == 0:
            return pv.PolyData()

        # VTK points
        vtk_pts = _vtk.vtkPoints()
        vtk_pts.SetNumberOfPoints(len(seg_points))
        for i, p in enumerate(seg_points):
            vtk_pts.SetPoint(i, np.float64(p[0]), np.float64(p[1]), np.float64(p[2]))

        # VTK lines (vtkCellArray)
        vtk_lines = _vtk.vtkCellArray()
        vtk_lines.Allocate(nlines)
        for (a, b) in seg_edges:
            ida = pid_map[a]
            idb = pid_map[b]
            vtk_lines.InsertNextCell(2)
            vtk_lines.InsertCellPoint(ida)
            vtk_lines.InsertCellPoint(idb)

        # Build vtkPolyData with ONLY lines
        vtk_poly = _vtk.vtkPolyData()
        vtk_poly.SetPoints(vtk_pts)
        vtk_poly.SetLines(vtk_lines)

        poly = pv.wrap(vtk_poly)

        # Sanity: should contain only line cells
        if poly.n_lines != nlines or poly.n_cells != nlines:
            raise RuntimeError(
                f"Expected only lines: nlines={nlines}, got n_lines={poly.n_lines}, n_cells={poly.n_cells}"
            )

        poly.cell_data["GlobalElementID"] = np.asarray(seg_parent, dtype=np.int32) + 1
        poly.points = seg_points
        poly.point_data["GlobalNodeID"]   = np.array(used_pids, dtype=np.int32) + 1
        return poly
    # ---------------------------
    # 3) Process each connected boundary component, split by turning angle, output segments
    # ---------------------------
    out: List[BoundarySegment] = []
    seg_counter = 0

    for comp in comps:
        vertices_order, edges_order, is_closed = order_component_vertices_and_edges(comp)

        # The walker might not capture all edges if the component is messy; fall back:
        if len(edges_order) == 0:
            continue

        # Split by 2D turning "crease" angle
        seg_edge_lists = split_edges_by_crease(vertices_order, edges_order, is_closed)

        for local_idx, edge_idx_list in enumerate(seg_edge_lists):
            #poly = build_segment_polydata(edges_order, edge_idx_list)
            poly = build_segment_polydata(
                                          edges_order,
                                          edge_idx_list,
                                          pts=pts,
                                          edge_owner=edge_owner,
                                          clean_tol=clean_tol,
                                          )
            # Enforce "no polylines": ensure every line cell is 2 points
            # (If something upstream produced polylines, convert them to segments)
            # Here we created lines ourselves, so this should already hold:
            assert poly.lines.size % 3 == 0, "Unexpected line connectivity size."

            name = f"{prefix}_{seg_counter:03d}"
            filename = None
            if save_dir is not None:
                filename = os.path.join(save_dir, f"{name}.vtp")
                poly.save(filename)

            out.append(BoundarySegment(name=name, poly=poly, filename=filename))
            seg_counter += 1

    return out

