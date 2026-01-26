#!/usr/bin/env python3
"""
make_kovasznay_mesh_2d.py

Generate a 2D quad mesh for the Kovasznay-flow verification domain and save as VTU.

Default domain: [0, 1] x [0, 1] (z=0 plane), quad elements.
Usage:
  python make_kovasznay_mesh_2d.py
  python make_kovasznay_mesh_2d.py --xmin -0.5 --xmax 1.0 --ymin -0.5 --ymax 0.5 --Nx 240 --Ny 120 --out kovasznay.vtu
"""

import argparse
import numpy as np
import pyvista as pv


def make_rect_quads(xmin, xmax, ymin, ymax, Nx, Ny) -> pv.UnstructuredGrid:
    """
    Build a structured quad mesh as a VTK UnstructuredGrid with VTK_QUAD cells.

    Nx, Ny are number of cells in x and y.
    Points are laid out with point id = i + j*(Nx+1), where i in [0..Nx], j in [0..Ny].
    """
    if Nx < 1 or Ny < 1:
        raise ValueError("Nx and Ny must be >= 1 (number of cells in x/y).")
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Invalid bounds.")

    npx, npy = Nx + 1, Ny + 1
    xs = np.linspace(xmin, xmax, npx)
    ys = np.linspace(ymin, ymax, npy)

    # Meshgrid with indexing="ij" gives arrays shaped (npx, npy)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    # VTK points are always 3D; keep z=0
    # Use ravel(order="F") so x-index changes fastest in memory (matches our point-id mapping)
    points = np.column_stack([
        X.ravel(order="F"),
        Y.ravel(order="F"),
        np.zeros(X.size, dtype=float)
    ])

    # Build quad connectivity. VTK expects: [4, p0, p1, p2, p3] for each quad.
    quads = np.empty((Nx * Ny, 5), dtype=np.int64)
    c = 0
    for j in range(Ny):
        for i in range(Nx):
            p0 = i + j * npx
            p1 = (i + 1) + j * npx
            p2 = (i + 1) + (j + 1) * npx
            p3 = i + (j + 1) * npx
            quads[c, :] = (4, p0, p1, p2, p3)
            c += 1

    cells = quads.ravel()
    celltypes = np.full(Nx * Ny, pv.CellType.QUAD, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells, celltypes, points)

    # Helpful metadata
    grid.field_data["bounds"] = np.array([xmin, xmax, ymin, ymax], dtype=float)
    grid["cell_id"] = np.arange(grid.n_cells, dtype=np.int32)
    grid["pt_id"] = np.arange(grid.n_points, dtype=np.int32)

    # Optional: tag boundary points (IDs) for convenience.
    # (These are point-data masks; many FEM codes instead want boundary edge/cell sets.)
    tol = 1e-12
    x = points[:, 0]
    y = points[:, 1]
    grid.point_data["on_xmin"] = (np.abs(x - xmin) < tol).astype(np.uint8)
    grid.point_data["on_xmax"] = (np.abs(x - xmax) < tol).astype(np.uint8)
    grid.point_data["on_ymin"] = (np.abs(y - ymin) < tol).astype(np.uint8)
    grid.point_data["on_ymax"] = (np.abs(y - ymax) < tol).astype(np.uint8)

    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xmin", type=float, default=0.0)
    ap.add_argument("--xmax", type=float, default=1.0)
    ap.add_argument("--ymin", type=float, default=0.0)
    ap.add_argument("--ymax", type=float, default=1.0)
    ap.add_argument("--Nx", type=int, default=160, help="Number of quad cells in x")
    ap.add_argument("--Ny", type=int, default=160, help="Number of quad cells in y")
    ap.add_argument("--out", type=str, default="kovasznay_2d_quads.vtu")
    ap.add_argument("--plot", action="store_true", help="Visualize the mesh (edges)")
    args = ap.parse_args()

    grid = make_rect_quads(args.xmin, args.xmax, args.ymin, args.ymax, args.Nx, args.Ny)
    grid.save(args.out)
    print(f"Wrote: {args.out}")
    print(f"Cells: {grid.n_cells}, Points: {grid.n_points}")

    if args.plot:
        grid.plot(show_edges=True)


if __name__ == "__main__":
    main()
