import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
import numpy as np
import pyvista as pv

def _as_3d_vectors(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v)
    if v.ndim != 2:
        raise ValueError(f"Expected 2D array for vector data, got shape {v.shape}")
    if v.shape[1] == 3:
        return v
    if v.shape[1] == 2:
        out = np.zeros((v.shape[0], 3), dtype=v.dtype)
        out[:, :2] = v
        return out
    raise ValueError(f"Expected 2 or 3 vector components, got {v.shape[1]}")

def _load_pvd_series(pvd_path: Path):
    """
    Parse a VTK .pvd (Collection) file and return a sorted list of (time, dataset_path).

    Dataset paths are resolved relative to the .pvd file location.
    """
    pvd_path = Path(pvd_path)
    root = ET.parse(pvd_path).getroot()
    if root.tag != "VTKFile" or root.get("type") != "Collection":
        raise ValueError(f"Not a VTK Collection (.pvd): {pvd_path}")

    collection = root.find("Collection")
    if collection is None:
        raise ValueError(f"Invalid .pvd (missing Collection): {pvd_path}")

    series = []
    for ds in collection.findall("DataSet"):
        file_attr = ds.get("file")
        if not file_attr:
            continue
        time_attr = ds.get("timestep")
        if time_attr is None:
            raise ValueError(f"Invalid .pvd DataSet (missing timestep): {pvd_path}")
        try:
            t = float(time_attr)
        except Exception as e:
            raise ValueError(f"Invalid timestep '{time_attr}' in {pvd_path}") from e

        f = Path(file_attr)
        if not f.is_absolute():
            f = (pvd_path.parent / f)
        series.append((t, f))

    if not series:
        raise ValueError(f"No DataSet entries found in: {pvd_path}")

    series.sort(key=lambda x: x[0])
    return series

def _extract_step_from_result_name(path: Path) -> Optional[int]:
    """
    Extract the integer time step from filenames like ``result_002.vtu`` or ``result_002.pvtu``.
    Returns None when the pattern is not matched.
    """
    match = re.search(r"result_(\d+)\.(?:vtu|vtp|pvtu|pvtp)$", Path(path).name)
    if not match:
        return None
    return int(match.group(1))

def _estimate_dt_from_steps(t: np.ndarray, steps: np.ndarray) -> Optional[float]:
    """
    Estimate the simulation time step dt from (TimeValue, step_index) pairs.
    """
    if len(t) < 2 or len(steps) != len(t):
        return None
    dt_steps = np.diff(steps)
    dt_time = np.diff(t)
    valid = dt_steps > 0
    if not np.any(valid):
        return None
    ratios = dt_time[valid] / dt_steps[valid].astype(float)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
    if ratios.size == 0:
        return None
    return float(np.median(ratios))

def _local_maxima_indices(y: np.ndarray) -> np.ndarray:
    """
    Return indices of simple local maxima (no SciPy).
    """
    y = np.asarray(y).reshape(-1)
    if y.size < 3:
        return np.array([], dtype=int)
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1

def _periods_from_peaks(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    peaks = _local_maxima_indices(y)
    if peaks.size < 2:
        return np.array([], dtype=float)
    T = np.diff(np.asarray(t)[peaks])
    T = T[np.isfinite(T) & (T > 0.0)]
    return T

def _dominant_frequency_fft(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Return (f_dom, amp_dom) from a 1D real FFT (mean removed, Hann window).

    Assumes approximately uniform sampling.
    """
    t = np.asarray(t).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if y.size < 8:
        return float("nan"), float("nan")

    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        return float("nan"), float("nan")

    y0 = y - float(np.mean(y))
    w = np.hanning(y0.size)
    Y = np.fft.rfft(y0 * w)
    freq = np.fft.rfftfreq(y0.size, d=dt)
    amp = np.abs(Y)

    if freq.size <= 1:
        return float("nan"), float("nan")

    k = int(np.argmax(amp[1:]) + 1)  # ignore DC
    return float(freq[k]), float(amp[k])

def _amplitude_at_frequency_fft(t: np.ndarray, y: np.ndarray, f_target: float) -> float:
    t = np.asarray(t).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if y.size < 8 or not np.isfinite(f_target) or f_target <= 0.0:
        return float("nan")
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        return float("nan")
    y0 = y - float(np.mean(y))
    w = np.hanning(y0.size)
    Y = np.fft.rfft(y0 * w)
    freq = np.fft.rfftfreq(y0.size, d=dt)
    amp = np.abs(Y)
    k = int(np.argmin(np.abs(freq - f_target)))
    return float(amp[k])

def _time_from_mesh_field_data(mesh: pv.DataSet, *, src: Path) -> Optional[float]:
    """
    Extract time from common FieldData keys if present.

    Notes:
    - Serial VTU time series created by helper scripts may include FieldData/TimeValue.
    - Parallel PVTU datasets commonly do *not* include FieldData, so callers should
      provide a fallback (e.g. the timestep attribute from a .pvd Collection file).
    """
    for key in ("TimeValue", "time"):
        if key not in mesh.field_data:
            continue
        try:
            return float(np.asarray(mesh.field_data[key]).ravel()[0])
        except Exception:
            raise ValueError(f"Invalid FieldData '{key}' in {src}")
    return None

def _require_solution_fields(mesh: pv.DataSet, *, src: Path, vel_name: str, p_name: str) -> None:
    missing = []
    if vel_name not in mesh.point_data and vel_name not in mesh.cell_data:
        missing.append(vel_name)
    if p_name not in mesh.point_data and p_name not in mesh.cell_data:
        missing.append(p_name)
    if missing:
        raise ValueError(f"Missing required fields {missing} in {src}")

def _ensure_point_arrays(mesh: pv.DataSet, *names: str) -> pv.DataSet:
    """
    Ensure the provided array names exist in point_data.

    If any of them exist only in cell_data, convert cell data to point data.
    """
    if all(name in mesh.point_data for name in names):
        return mesh
    missing = [name for name in names if name not in mesh.point_data and name not in mesh.cell_data]
    if missing:
        raise ValueError(f"Missing required fields {missing} in dataset.")
    return mesh.cell_data_to_point_data(pass_cell_data=False)

def _divergence_max_and_l2(
    mesh: pv.DataSet,
    *,
    div_name: str = "Divergence",
    vel_name: str = "Velocity",
    cell_area: Optional[np.ndarray] = None,
    cell_area_total: Optional[float] = None,
) -> tuple[float, float]:
    """
    Return (max_abs_div, l2_div) for a timestep.

    - max_abs_div is computed as max(|div|) from point data if present, else from cell data.
    - l2_div is computed as an area-weighted RMS over cells:
        sqrt( ∫ div^2 dΩ / ∫ dΩ )
      using cell 'Area' from VTK's cell size filter.
    """
    # Prefer point divergence if present (higher resolution for max norm).
    if div_name in mesh.point_data:
        div_pt = np.asarray(mesh.point_data[div_name]).reshape(-1)
    elif div_name in mesh.cell_data:
        div_pt = None
    else:
        # Fallback: compute divergence from velocity gradient.
        if vel_name in mesh.point_data:
            m = mesh
        elif vel_name in mesh.cell_data:
            m = mesh.cell_data_to_point_data(pass_cell_data=False)
        else:
            raise ValueError(f"Missing '{div_name}' and '{vel_name}' in dataset; cannot compute divergence.")

        m = _grad_u_at_points(m, vel_name)
        GradU = np.asarray(m.point_data["GradU"]).reshape(-1, 3, 3)
        div_pt = (GradU[:, 0, 0] + GradU[:, 1, 1] + GradU[:, 2, 2]).reshape(-1)

    if div_pt is None:
        div_max = float(np.max(np.abs(np.asarray(mesh.cell_data[div_name]).reshape(-1))))
        cell_mesh = mesh
    else:
        div_max = float(np.max(np.abs(div_pt)))
        # Need cell divergence for area-weighted L2.
        if div_name in mesh.cell_data:
            cell_mesh = mesh
        else:
            m = mesh
            if div_name not in m.point_data:
                # Attach computed divergence to a point-data view for conversion.
                # (This is only hit when divergence was absent in the file.)
                if vel_name in m.point_data:
                    m = m.copy(deep=True)
                    m.point_data[div_name] = div_pt
                else:
                    m = m.cell_data_to_point_data(pass_cell_data=False)
                    m.point_data[div_name] = div_pt
            cell_mesh = m.point_data_to_cell_data(pass_point_data=False)

    area = None
    if cell_area is not None:
        area = np.asarray(cell_area).reshape(-1)
        if area.size != cell_mesh.n_cells:
            area = None
    if area is None:
        cell_mesh = cell_mesh.compute_cell_sizes(length=False, area=True, volume=False)
        area = np.asarray(cell_mesh.cell_data["Area"]).reshape(-1)

    div_cell = np.asarray(cell_mesh.cell_data[div_name]).reshape(-1)
    if cell_area_total is not None and np.isfinite(cell_area_total) and cell_area is not None and area is not None and area.size == np.asarray(cell_area).size:
        denom = float(cell_area_total)
    else:
        denom = float(np.sum(area))
    div_l2 = float(np.sqrt(np.sum((div_cell * div_cell) * area) / denom)) if denom > 0.0 else float("nan")
    return div_max, div_l2

def _ensure_point_data(mesh: pv.DataSet, vel_name: str, p_name: str) -> pv.DataSet:
    """If vel/p are cell data, convert to point data so sampling works cleanly."""
    return _ensure_point_arrays(mesh, vel_name, p_name)

def _grad_u_at_points(mesh: pv.DataSet, vel_name: str) -> pv.DataSet:
    """
    Adds point array 'GradU' of shape (npts, 9) (row-major 3x3) via VTK derivative.
    """
    # compute_derivative works for point-data vectors
    m = mesh.compute_derivative(scalars=vel_name, gradient=True)
    grad = np.asarray(m.point_data.pop("gradient"))

    # VTK returns Ncomp*3 components. Our 2D results often store Velocity as (u,v),
    # so gradient has 6 components (2x3). Expand to 9 (3x3) by padding the 3rd row.
    if grad.ndim != 2:
        raise ValueError(f"Unexpected gradient array shape: {grad.shape}")

    if grad.shape[1] == 9:
        grad9 = grad
    elif grad.shape[1] == 6:
        g = grad.reshape(-1, 2, 3)
        g3 = np.zeros((g.shape[0], 3, 3), dtype=g.dtype)
        g3[:, 0:2, :] = g
        grad9 = g3.reshape(-1, 9)
    else:
        raise ValueError(f"Unexpected gradient component count: {grad.shape[1]}")

    m.point_data["GradU"] = grad9
    return m

def _line_segments(lines: pv.PolyData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (i0, i1, ds) for each VTK_LINE cell in a PolyData."""
    conn = lines.lines.reshape(-1, 3)
    if conn.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    if not np.all(conn[:, 0] == 2):
        raise ValueError("Expected only 2-point line cells (VTK_LINE).")
    i0 = conn[:, 1]
    i1 = conn[:, 2]
    p0 = lines.points[i0]
    p1 = lines.points[i1]
    ds = np.linalg.norm(p1 - p0, axis=1)
    return i0, i1, ds

def _line_flux_2d(
    mesh: pv.DataSet,
    lines: pv.PolyData,
    *,
    normal_xy: tuple[float, float],
    vel_name: str = "Velocity",
) -> float:
    """
    Compute Q = ∫ u·n ds along a line boundary (2D, outward normal in xy).
    """
    mesh = _ensure_point_arrays(mesh, vel_name)
    samp = lines.sample(mesh)
    u = _as_3d_vectors(np.asarray(samp.point_data[vel_name]))

    i0, i1, ds = _line_segments(lines)
    if ds.size == 0:
        return 0.0

    nx, ny = normal_xy
    udotn = u[:, 0] * nx + u[:, 1] * ny
    qseg = 0.5 * (udotn[i0] + udotn[i1]) * ds
    return float(np.sum(qseg))

def _recirculation_length_centerline(
    mesh: pv.DataSet,
    *,
    vel_name: str,
    cyl_center: tuple[float, float, float],
    D: float,
    n_samples: int = 800,
) -> float:
    """
    Estimate recirculation length Lr behind the cylinder from centerline u_x sign change.

    Lr = x_reattach - (x_c + r), where r = D/2 and x_reattach is the first downstream
    x where u_x crosses from negative to non-negative along y=y_c.
    """
    if n_samples < 10:
        raise ValueError("n_samples must be >= 10")

    mesh = _ensure_point_arrays(mesh, vel_name)
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    cx, cy, cz = cyl_center
    r = 0.5 * float(D)
    x0 = float(cx + r)
    if not (xmin <= x0 <= xmax):
        return float("nan")

    x = np.linspace(x0, float(xmax), int(n_samples))
    pts = np.column_stack([x, np.full_like(x, float(cy)), np.full_like(x, float(cz))])
    probes = pv.PolyData(pts)
    samp = probes.sample(mesh)
    u = _as_3d_vectors(np.asarray(samp.point_data[vel_name]))
    ux = u[:, 0].reshape(-1)

    if ux.size < 2 or not np.any(np.isfinite(ux)):
        return float("nan")

    # Find first negative region, then first crossing back to >=0.
    neg = ux < 0.0
    if not np.any(neg):
        return float("nan")
    i_start = int(np.argmax(neg))
    for i in range(i_start, ux.size - 1):
        if ux[i] < 0.0 and ux[i + 1] >= 0.0:
            x1, x2 = x[i], x[i + 1]
            u1, u2 = ux[i], ux[i + 1]
            if u2 == u1:
                x_cross = x2
            else:
                x_cross = x1 + (0.0 - u1) * (x2 - x1) / (u2 - u1)
            return float(x_cross - x0)

    return float("nan")

def _default_wake_probe_points(
    *,
    cyl_center: tuple[float, float, float],
    D: float,
    bounds: tuple[float, float, float, float, float, float],
) -> list[tuple[float, float, float]]:
    cx, cy, cz = cyl_center
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    candidates = [
        (cx + 2.0 * D, cy, cz),
        (cx + 4.0 * D, cy, cz),
        (cx + 4.0 * D, cy + 0.5 * D, cz),
        (cx + 4.0 * D, cy - 0.5 * D, cz),
    ]
    pts = []
    for x, y, z in candidates:
        if xmin <= x <= xmax and ymin <= y <= ymax:
            pts.append((float(x), float(y), float(z)))
    return pts

def _kinetic_energy_dissipation_and_cfl_factors(
    mesh: pv.DataSet,
    *,
    vel_name: str,
    nu: float,
    cell_area: np.ndarray,
    cell_area_total: float,
) -> tuple[float, float, float, float]:
    """
    Return (K, Phi, cfl_max_factor, cfl_mean_factor) for one timestep.

    - K   = ∫ 0.5 |u|^2 dΩ
    - Phi = ∫ 2ν ε(u):ε(u) dΩ
    - cfl_*_factor are CFL values *without* dt (i.e. |u|/h), where h=sqrt(area).
      Multiply by dt to get CFL.
    """
    if cell_area.size != mesh.n_cells:
        raise ValueError("cell_area does not match mesh.n_cells")
    if cell_area_total <= 0.0 or not np.isfinite(cell_area_total):
        return float("nan"), float("nan"), float("nan"), float("nan")

    mesh = _ensure_point_arrays(mesh, vel_name)
    if "GradU" not in mesh.point_data:
        mesh = _grad_u_at_points(mesh, vel_name)

    u_pt = _as_3d_vectors(np.asarray(mesh.point_data[vel_name]))
    speed2_pt = np.sum(u_pt * u_pt, axis=1)

    GradU = np.asarray(mesh.point_data["GradU"]).reshape(-1, 3, 3)
    eps = 0.5 * (GradU + np.transpose(GradU, (0, 2, 1)))
    eps2_pt = np.sum(eps * eps, axis=(1, 2))

    # Convert to cell averages for integration and CFL.
    mesh.point_data["_metrics_speed2"] = speed2_pt
    mesh.point_data["_metrics_eps2"] = eps2_pt
    cell = mesh.point_data_to_cell_data(pass_point_data=False)

    speed2_cell = np.asarray(cell.cell_data["_metrics_speed2"]).reshape(-1)
    eps2_cell = np.asarray(cell.cell_data["_metrics_eps2"]).reshape(-1)

    K = float(0.5 * np.sum(speed2_cell * cell_area))
    Phi = float(2.0 * float(nu) * np.sum(eps2_cell * cell_area))

    h = np.sqrt(np.asarray(cell_area).reshape(-1))
    h = np.where(h > 0.0, h, np.nan)
    speed_cell = np.sqrt(np.maximum(speed2_cell, 0.0))
    factor = speed_cell / h

    cfl_max_factor = float(np.nanmax(factor))
    cfl_mean_factor = float(np.nansum(factor * cell_area) / float(cell_area_total))

    return K, Phi, cfl_max_factor, cfl_mean_factor

def _line_cell_midpoints_and_ds(lines: pv.PolyData):
    """Return midpoints, segment vectors, and ds for each VTK_LINE cell."""
    conn = lines.lines.reshape(-1, 3)
    assert np.all(conn[:, 0] == 2), "Expected only VTK_LINE cells (2-pt segments)."
    i0 = conn[:, 1]
    i1 = conn[:, 2]
    p0 = lines.points[i0]
    p1 = lines.points[i1]
    mid = 0.5 * (p0 + p1)
    seg = (p1 - p0)
    ds = np.linalg.norm(seg, axis=1)
    return mid, seg, ds

def _outward_normals_for_cylinder(midpoints: np.ndarray, seg: np.ndarray, center_xyz):
    """
    Build outward unit normals in the xy-plane for line segments on a cylinder.
    Normal = rotate90(tangent). Sign chosen so it points away from cylinder center.
    """
    cx, cy, cz = center_xyz
    # tangent in xy
    tx = seg[:, 0]
    ty = seg[:, 1]
    # rotate by +90 deg: (tx, ty) -> (-ty, tx)
    nx = -ty
    ny =  tx
    n = np.stack([nx, ny, np.zeros_like(nx)], axis=1)
    nlen = np.linalg.norm(n[:, :2], axis=1)
    nlen = np.where(nlen == 0, 1.0, nlen)
    n = n / nlen[:, None]

    # flip so it points outward relative to cylinder center
    r = midpoints - np.array([cx, cy, cz])
    s = np.sign(np.sum(n[:, :2] * r[:, :2], axis=1))
    s = np.where(s == 0, 1.0, s)
    n = n * s[:, None]
    return n

def cylinder_force_2d_from_line_integral(
        mesh: pv.DataSet,
        cyl_lines: pv.PolyData,
        *,
        vel_name: str = "Velocity",
        p_name: str = "Pressure",
        nu: float = 1e-3,
        rho: float = 1.0,
        Umean: float = 1.0,
        D: float = 0.1,
        cyl_center=(0.2, 0.2, 0.0),
) -> dict:
    """
    Compute Fx, Fy (per unit depth) and Cd, Cl for one timestep.
    Assumes mesh is planar (z=0) but stored as 3D vectors is fine.
    """
    mesh = _ensure_point_data(mesh, vel_name, p_name)
    if "GradU" not in mesh.point_data:
        mesh = _grad_u_at_points(mesh, vel_name)

    # Sample u, p, GradU onto cylinder boundary points
    samp = cyl_lines.sample(mesh)

    # Build per-line-cell geometry
    mid, seg, ds = _line_cell_midpoints_and_ds(cyl_lines)
    n = _outward_normals_for_cylinder(mid, seg, cyl_center)

    # Pull sampled arrays at boundary points
    u = np.asarray(samp.point_data[vel_name])
    p = np.asarray(samp.point_data[p_name]).reshape(-1)

    GradU = np.asarray(samp.point_data["GradU"]).reshape(-1, 3, 3)  # row-major 3x3

    # Strain rate tensor epsilon at points
    eps = 0.5 * (GradU + np.transpose(GradU, (0, 2, 1)))  # (npts,3,3)

    # Traction at points: t = (-p I + 2 nu eps) n
    # We'll evaluate per line cell by averaging endpoint tractions.
    # First compute traction at points:
    # viscous = 2 nu eps @ n_point  (need n at points; we approximate by nearest cell normal later)
    # Simpler: compute traction per cell using cell normal and average endpoints of stress.

    # Map line cell endpoints to point indices in cyl_lines:
    conn = cyl_lines.lines.reshape(-1, 3)
    i0 = conn[:, 1]
    i1 = conn[:, 2]

    # Cell normals (outward) computed at midpoints; use same normal for both endpoints of that segment
    ncell = n

    # Stress at endpoints
    # sigma = -p I + 2 nu eps
    I = np.eye(3)[None, :, :]
    sigma0 = (-p[i0])[:, None, None] * I + (2.0 * nu) * eps[i0]
    sigma1 = (-p[i1])[:, None, None] * I + (2.0 * nu) * eps[i1]

    # Traction vectors at endpoints: sigma @ ncell
    t0 = np.einsum("nij,nj->ni", sigma0, ncell)
    t1 = np.einsum("nij,nj->ni", sigma1, ncell)
    tcell = 0.5 * (t0 + t1)  # (nlines,3)

    # Line integral: sum(t * ds)
    F = np.sum(tcell * ds[:, None], axis=0)  # (3,)
    Fx, Fy = float(F[0]), float(F[1])

    Cd = 2.0 * Fx / (rho * (Umean**2) * D)
    Cl = 2.0 * Fy / (rho * (Umean**2) * D)

    return {"Fx": Fx, "Fy": Fy, "Cd": Cd, "Cl": Cl}

def pressure_drop(mesh: pv.DataSet, *, p_name="Pressure",
                  a1=(0.15, 0.2, 0.0), a2=(0.25, 0.2, 0.0)) -> float:
    mesh = _ensure_point_data(mesh, vel_name="Velocity", p_name=p_name)  # vel_name unused here
    probes = pv.PolyData(np.array([a1, a2], dtype=float))
    samp = probes.sample(mesh)
    p = np.asarray(samp.point_data[p_name]).reshape(-1)
    return float(p[0] - p[1])

def estimate_frequency_from_peaks(t: np.ndarray, y: np.ndarray) -> float:
    """Return f from peak-to-peak intervals (simple, no scipy)."""
    # crude peak detection: y[i-1] < y[i] > y[i+1]
    peaks = [i for i in range(1, len(y)-1) if (y[i] > y[i-1] and y[i] > y[i+1])]
    if len(peaks) < 2:
        return float("nan")
    T = np.diff(t[peaks])
    T = T[T > 0]
    if len(T) == 0:
        return float("nan")
    return 1.0 / float(np.mean(T))

def analyze_vortex_shedding_series(
        pvd_file: str,
        cyl_vtp: str,
        *,
        vel_name="Velocity",
        p_name="Pressure",
        div_name="Divergence",
        vort_name="Vorticity",
        nu=1e-3,
        rho=1.0,
        Umean=1.0,
        D=0.1,
        cyl_center=(0.2, 0.2, 0.0),
        transient_fraction=0.5,  # discard first half as transient (tune as needed)
):
    pvd_path = Path(pvd_file)
    series = _load_pvd_series(pvd_path)

    cyl_lines = pv.read(cyl_vtp)
    # Ensure boundary is only lines (VTK_LINE)
    if cyl_lines.n_lines == 0:
        raise ValueError("Cylinder boundary file has no line cells.")
    if cyl_lines.lines.size % 3 != 0 or not np.all(cyl_lines.lines.reshape(-1,3)[:,0] == 2):
        raise ValueError("Cylinder boundary must contain only 2-point line cells (no polylines).")

    # Optional boundaries for additional diagnostics (auto-located relative to cyl_vtp).
    surfaces_dir = Path(cyl_vtp).resolve().parent
    boundary_paths = {
        "left": surfaces_dir / "left_line.vtp",
        "right": surfaces_dir / "right_line.vtp",
        "top": surfaces_dir / "top_line.vtp",
        "bottom": surfaces_dir / "bottom_line.vtp",
    }
    boundary_normals = {
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
        "top": (0.0, 1.0),
        "bottom": (0.0, -1.0),
    }
    boundaries: dict[str, pv.PolyData] = {}
    for name, path in boundary_paths.items():
        if path.exists():
            boundaries[name] = pv.read(str(path))

    # Will be initialized after reading the first timestep (needs bounds).
    cell_area = None
    cell_area_total = None
    probe_points = None

    t_list, step_list, Cd_list, Cl_list, dp_list = [], [], [], [], []
    div_max_list, div_l2_list = [], []
    Lr_list = []
    Qin_list, Qout_list, Qtop_list, Qbottom_list = [], [], [], []
    eps_total_list, eps_io_list = [], []
    K_list, Phi_list = [], []
    cfl_max_factor_list, cfl_mean_factor_list = [], []
    uy_probes_list = None
    wz_probes_list = None

    for t_pvd, f in series:
        m = pv.read(str(f))
        _require_solution_fields(m, src=f, vel_name=vel_name, p_name=p_name)
        t_mesh = _time_from_mesh_field_data(m, src=f)
        if t_mesh is not None and not np.isclose(t_mesh, t_pvd, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Time mismatch for {f}: pvd={t_pvd}, FieldData={t_mesh}")
        t = float(t_pvd)

        step_list.append(_extract_step_from_result_name(f))

        # Ensure required arrays are on points for sampling operations.
        m = _ensure_point_arrays(m, vel_name, p_name)

        if cell_area is None:
            area_mesh = m.compute_cell_sizes(length=False, area=True, volume=False)
            cell_area = np.asarray(area_mesh.cell_data["Area"]).reshape(-1)
            cell_area_total = float(np.sum(cell_area))
            probe_points = _default_wake_probe_points(cyl_center=cyl_center, D=float(D), bounds=m.bounds)
            if probe_points:
                uy_probes_list = [[] for _ in range(len(probe_points))]
                wz_probes_list = [[] for _ in range(len(probe_points))]

        div_max, div_l2 = _divergence_max_and_l2(
            m,
            div_name=div_name,
            vel_name=vel_name,
            cell_area=cell_area,
            cell_area_total=cell_area_total,
        )

        # Core integrals for "solver health" style metrics.
        if cell_area is not None and cell_area_total is not None:
            K, Phi, cfl_max_fac, cfl_mean_fac = _kinetic_energy_dissipation_and_cfl_factors(
                m,
                vel_name=vel_name,
                nu=float(nu),
                cell_area=cell_area,
                cell_area_total=cell_area_total,
            )
        else:
            K, Phi, cfl_max_fac, cfl_mean_fac = float("nan"), float("nan"), float("nan"), float("nan")

        res = cylinder_force_2d_from_line_integral(
            m, cyl_lines,
            vel_name=vel_name, p_name=p_name,
            nu=nu, rho=rho, Umean=Umean, D=D,
            cyl_center=cyl_center,
        )
        dp = pressure_drop(m, p_name=p_name)

        # Recirculation length on centerline.
        Lr = _recirculation_length_centerline(m, vel_name=vel_name, cyl_center=cyl_center, D=float(D))

        # Wake probes (u_y and ω_z).
        if probe_points:
            probes = pv.PolyData(np.array(probe_points, dtype=float))
            samp = probes.sample(m)
            u = _as_3d_vectors(np.asarray(samp.point_data[vel_name]))
            uy = u[:, 1].reshape(-1)
            for i, val in enumerate(uy):
                uy_probes_list[i].append(float(val))

            if vort_name in m.point_data:
                vort = _as_3d_vectors(np.asarray(samp.point_data[vort_name]))
                wz = vort[:, 2].reshape(-1)
            else:
                # Compute ω_z from GradU if solver did not provide vorticity.
                if "GradU" not in m.point_data:
                    m = _grad_u_at_points(m, vel_name)
                GradU = np.asarray(m.point_data["GradU"]).reshape(-1, 3, 3)
                m.point_data["_metrics_wz"] = (GradU[:, 1, 0] - GradU[:, 0, 1]).reshape(-1)
                samp_wz = probes.sample(m)
                wz = np.asarray(samp_wz.point_data["_metrics_wz"]).reshape(-1)
            for i, val in enumerate(wz):
                wz_probes_list[i].append(float(val))

        # Flux imbalance diagnostics.
        if boundaries:
            Qin = _line_flux_2d(m, boundaries["left"], normal_xy=boundary_normals["left"], vel_name=vel_name) if "left" in boundaries else float("nan")
            Qout = _line_flux_2d(m, boundaries["right"], normal_xy=boundary_normals["right"], vel_name=vel_name) if "right" in boundaries else float("nan")
            Qtop = _line_flux_2d(m, boundaries["top"], normal_xy=boundary_normals["top"], vel_name=vel_name) if "top" in boundaries else float("nan")
            Qbottom = _line_flux_2d(m, boundaries["bottom"], normal_xy=boundary_normals["bottom"], vel_name=vel_name) if "bottom" in boundaries else float("nan")

            Q_total = Qin + Qout + Qtop + Qbottom
            denom = float(abs(Qin)) if np.isfinite(Qin) and abs(Qin) > 0.0 else float("nan")
            eps_total = float(abs(Q_total) / denom) if np.isfinite(denom) else float("nan")
            eps_io = float(abs(Qin + Qout) / denom) if np.isfinite(denom) else float("nan")
        else:
            Qin = Qout = Qtop = Qbottom = eps_total = eps_io = float("nan")

        t_list.append(t)
        Cd_list.append(res["Cd"])
        Cl_list.append(res["Cl"])
        dp_list.append(dp)
        div_max_list.append(div_max)
        div_l2_list.append(div_l2)
        Lr_list.append(Lr)
        Qin_list.append(Qin)
        Qout_list.append(Qout)
        Qtop_list.append(Qtop)
        Qbottom_list.append(Qbottom)
        eps_total_list.append(eps_total)
        eps_io_list.append(eps_io)
        K_list.append(K)
        Phi_list.append(Phi)
        cfl_max_factor_list.append(cfl_max_fac)
        cfl_mean_factor_list.append(cfl_mean_fac)

    t = np.array(t_list)
    steps = np.array([s if s is not None else -1 for s in step_list], dtype=int)
    Cd = np.array(Cd_list)
    Cl = np.array(Cl_list)
    dp = np.array(dp_list)
    div_max = np.array(div_max_list)
    div_l2 = np.array(div_l2_list)
    Lr = np.array(Lr_list)
    Qin = np.array(Qin_list)
    Qout = np.array(Qout_list)
    Qtop = np.array(Qtop_list)
    Qbottom = np.array(Qbottom_list)
    eps_total = np.array(eps_total_list)
    eps_io = np.array(eps_io_list)
    K = np.array(K_list)
    Phi = np.array(Phi_list)
    cfl_max_fac = np.array(cfl_max_factor_list)
    cfl_mean_fac = np.array(cfl_mean_factor_list)

    # Drop transient portion
    k0 = int(transient_fraction * len(t))
    tt, Cd2, Cl2, dp2 = t[k0:], Cd[k0:], Cl[k0:], dp[k0:]
    div_max2, div_l22 = div_max[k0:], div_l2[k0:]
    Lr2 = Lr[k0:]
    Qin2, Qout2 = Qin[k0:], Qout[k0:]
    Qtop2, Qbottom2 = Qtop[k0:], Qbottom[k0:]
    eps_total2, eps_io2 = eps_total[k0:], eps_io[k0:]
    K2, Phi2 = K[k0:], Phi[k0:]
    cfl_max_fac2, cfl_mean_fac2 = cfl_max_fac[k0:], cfl_mean_fac[k0:]

    # Metrics
    metrics = {
        "Cd_mean": float(np.mean(Cd2)),
        "Cd_min": float(np.min(Cd2)),
        "Cd_max": float(np.max(Cd2)),
        "Cd_amp": float(0.5 * (np.max(Cd2) - np.min(Cd2))),
        "Cl_mean": float(np.mean(Cl2)),
        "Cl_min": float(np.min(Cl2)),
        "Cl_max": float(np.max(Cl2)),
        "Cl_amp": float(0.5 * (np.max(Cl2) - np.min(Cl2))),
        "dp_mean": float(np.mean(dp2)),
        "div_max_mean": float(np.mean(div_max2)),
        "div_max_max": float(np.max(div_max2)),
        "div_l2_mean": float(np.mean(div_l22)),
        "div_l2_max": float(np.max(div_l22)),
    }

    # Shedding period stability (from successive peaks in C_L).
    periods = _periods_from_peaks(tt, Cl2)
    if periods.size > 0:
        T_mean = float(np.mean(periods))
        T_std = float(np.std(periods))
        T_cv = float(T_std / T_mean) if T_mean > 0.0 else float("nan")
        f_shed = float(1.0 / T_mean) if T_mean > 0.0 else float("nan")
    else:
        T_mean = T_std = T_cv = f_shed = float("nan")

    metrics["T_mean"] = T_mean
    metrics["T_std"] = T_std
    metrics["T_cv"] = T_cv
    metrics["n_cycles"] = int(periods.size)
    metrics["f_shed"] = float(f_shed)
    metrics["St"] = float(f_shed * D / Umean) if np.isfinite(f_shed) else float("nan")

    # Lift/drag frequency relationship via FFT (C_D ~ 2× f_L for clean shedding).
    fL_fft, ampL_fft = _dominant_frequency_fft(tt, Cl2)
    fD_fft, ampD_fft = _dominant_frequency_fft(tt, Cd2)
    metrics["fL_fft"] = float(fL_fft)
    metrics["fD_fft"] = float(fD_fft)
    metrics["fD_over_fL_fft"] = float(fD_fft / fL_fft) if np.isfinite(fD_fft) and np.isfinite(fL_fft) and fL_fft > 0.0 else float("nan")
    cd_amp_2fL = _amplitude_at_frequency_fft(tt, Cd2, 2.0 * float(fL_fft))
    metrics["Cd_amp_at_2fL_fft_over_dom"] = float(cd_amp_2fL / ampD_fft) if np.isfinite(cd_amp_2fL) and np.isfinite(ampD_fft) and ampD_fft > 0.0 else float("nan")

    # Recirculation length behind cylinder.
    metrics["Lr_mean"] = float(np.nanmean(Lr2)) if Lr2.size else float("nan")
    metrics["Lr_min"] = float(np.nanmin(Lr2)) if Lr2.size else float("nan")
    metrics["Lr_max"] = float(np.nanmax(Lr2)) if Lr2.size else float("nan")

    # Wake probe amplitudes (u_y and ω_z) at a few downstream points.
    if probe_points and uy_probes_list is not None and wz_probes_list is not None:
        metrics["wake_probes"] = [list(p) for p in probe_points]
        uy = np.array([np.asarray(v, dtype=float) for v in uy_probes_list])  # (nprobe, nt)
        wz = np.array([np.asarray(v, dtype=float) for v in wz_probes_list])
        uy2 = uy[:, k0:]
        wz2 = wz[:, k0:]
        for i in range(uy2.shape[0]):
            metrics[f"uy_probe{i}_rms"] = float(np.sqrt(np.mean(uy2[i] * uy2[i])))
            metrics[f"uy_probe{i}_amp"] = float(0.5 * (np.max(uy2[i]) - np.min(uy2[i])))
            metrics[f"wz_probe{i}_rms"] = float(np.sqrt(np.mean(wz2[i] * wz2[i])))
            metrics[f"wz_probe{i}_amp"] = float(0.5 * (np.max(wz2[i]) - np.min(wz2[i])))
    else:
        metrics["wake_probes"] = []

    # Mass conservation diagnostics (flux imbalance).
    metrics["Qin_mean"] = float(np.nanmean(Qin2)) if Qin2.size else float("nan")
    metrics["Qout_mean"] = float(np.nanmean(Qout2)) if Qout2.size else float("nan")
    metrics["eps_total_mean"] = float(np.nanmean(eps_total2)) if eps_total2.size else float("nan")
    metrics["eps_total_max"] = float(np.nanmax(eps_total2)) if eps_total2.size else float("nan")
    metrics["eps_io_mean"] = float(np.nanmean(eps_io2)) if eps_io2.size else float("nan")
    metrics["eps_io_max"] = float(np.nanmax(eps_io2)) if eps_io2.size else float("nan")

    # Energy budgets.
    metrics["K_mean"] = float(np.nanmean(K2)) if K2.size else float("nan")
    metrics["K_min"] = float(np.nanmin(K2)) if K2.size else float("nan")
    metrics["K_max"] = float(np.nanmax(K2)) if K2.size else float("nan")
    metrics["Phi_mean"] = float(np.nanmean(Phi2)) if Phi2.size else float("nan")
    metrics["Phi_min"] = float(np.nanmin(Phi2)) if Phi2.size else float("nan")
    metrics["Phi_max"] = float(np.nanmax(Phi2)) if Phi2.size else float("nan")

    # dt and time-resolution metrics.
    dt_est = _estimate_dt_from_steps(t, steps) if np.all(steps >= 0) else None
    if dt_est is None:
        dt_est = float(np.median(np.diff(t))) if t.size >= 2 else float("nan")
    metrics["dt_est"] = float(dt_est)
    metrics["dt_over_T"] = float(dt_est / T_mean) if np.isfinite(dt_est) and np.isfinite(T_mean) and T_mean > 0.0 else float("nan")
    metrics["steps_per_cycle"] = float(T_mean / dt_est) if np.isfinite(dt_est) and np.isfinite(T_mean) and dt_est > 0.0 else float("nan")

    # CFL estimates (needs dt; use the factors accumulated per timestep).
    cfl_max = cfl_max_fac * float(dt_est) if np.isfinite(dt_est) else np.full_like(cfl_max_fac, np.nan, dtype=float)
    cfl_mean = cfl_mean_fac * float(dt_est) if np.isfinite(dt_est) else np.full_like(cfl_mean_fac, np.nan, dtype=float)
    cfl_max2 = cfl_max[k0:]
    cfl_mean2 = cfl_mean[k0:]
    metrics["cfl_max_mean"] = float(np.nanmean(cfl_max2)) if cfl_max2.size else float("nan")
    metrics["cfl_max_max"] = float(np.nanmax(cfl_max2)) if cfl_max2.size else float("nan")
    metrics["cfl_mean_mean"] = float(np.nanmean(cfl_mean2)) if cfl_mean2.size else float("nan")
    metrics["cfl_mean_max"] = float(np.nanmax(cfl_mean2)) if cfl_mean2.size else float("nan")

    return metrics, (t, Cd, Cl, dp, div_max, div_l2)
