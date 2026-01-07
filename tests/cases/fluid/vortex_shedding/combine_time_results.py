import argparse
import glob
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import pyvista as pv


def _read_dt_from_solver_xml(solver_xml: Path) -> Optional[float]:
    try:
        root = ET.parse(solver_xml).getroot()
    except Exception:
        return None

    # Typical location for this repo's solver.xml
    elem = root.find(".//GeneralSimulationParameters/Time_step_size")
    if elem is None or elem.text is None:
        return None
    try:
        return float(elem.text.strip())
    except Exception:
        return None


def _extract_time_from_mesh_field_data(path: Path) -> Optional[float]:
    """
    Return the time value stored in the mesh field data if available.
    Looks for common keys: 'TimeValue' (svMultiPhysics/VTK) and 'time' (generic).
    """
    try:
        mesh = pv.read(str(path))
    except Exception:
        return None

    for key in ("TimeValue", "time"):
        try:
            if key in mesh.field_data:
                values = mesh.field_data[key]
                if hasattr(values, "shape") and values.size > 0:
                    return float(values.flat[0])
                return float(values)
        except Exception:
            continue

    return None


def _extract_step_from_name(path: Path) -> int:
    """
    Extract integer step index from a file name like ``result_002.vtu``.
    """
    match = re.search(r"result_(\d+)\.(?:vtu|vtp)$", path.name)
    if not match:
        return 0
    return int(match.group(1))


def _inject_timevalue_field_data(*, src: Path, dst: Path, time_value: float) -> None:
    """
    Write a copy of `src` to `dst`, ensuring a FieldData array:
      FieldData/TimeValue = [time_value]

    Notes:
    - This modifies only the XML header; appended binary data offsets remain valid.
    - VTK's PVD format stores time in the DataSet 'timestep' attribute, but adding
      TimeValue to the VTU is convenient for per-file Python post-processing.
    """
    root = ET.parse(src).getroot()
    ug = root.find("UnstructuredGrid")
    if ug is None:
        raise RuntimeError(f"Unexpected VTU structure (no UnstructuredGrid): {src}")

    # Ensure FieldData exists and is before the first Piece for readability.
    field_data = ug.find("FieldData")
    if field_data is None:
        field_data = ET.Element("FieldData")
        ug.insert(0, field_data)

    time_array = None
    for child in field_data.findall("DataArray"):
        if child.get("Name") == "TimeValue":
            time_array = child
            break
    if time_array is None:
        time_array = ET.SubElement(field_data, "DataArray")

    time_array.set("type", "Float64")
    time_array.set("Name", "TimeValue")
    time_array.set("NumberOfTuples", "1")
    time_array.set("format", "ascii")
    time_array.text = f"{time_value:.17g}"

    ET.ElementTree(root).write(dst, xml_declaration=True, encoding="UTF-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a .pvd time series from svMultiPhysics result_*.vtu files.")
    parser.add_argument("--results-dir", default="1-procs", help="Directory containing result_*.vtu files.")
    parser.add_argument("--prefix", default="result_", help="Result file prefix (default: result_).")
    parser.add_argument("--ext", default="vtu", choices=("vtu", "vtp"), help="Result file extension.")
    parser.add_argument("--dt", type=float, default=None, help="Time step size; overrides reading from solver.xml.")
    parser.add_argument("--solver-xml", default="solver.xml", help="Path to solver.xml (used to auto-detect dt).")
    parser.add_argument("--out", default="timeseries.pvd", help="Output .pvd filename (written inside results-dir).")
    parser.add_argument(
        "--write-timevalue",
        action="store_true",
        help="Write copies of VTU files with FieldData/TimeValue populated and reference those in the .pvd.",
    )
    parser.add_argument(
        "--timevalue-dir",
        default="with_timevalue",
        help="Subdirectory (inside results-dir) to write timevalue-patched copies when --write-timevalue is set.",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent

    def resolve_input_path(path_str: str) -> Path:
        """
        Resolve a user-provided path.
        - Absolute paths are used as-is.
        - Relative paths are first interpreted relative to CWD, then relative to this script.
        """
        path = Path(path_str)
        if path.is_absolute():
            return path
        cand_cwd = cwd / path
        if cand_cwd.exists():
            return cand_cwd.resolve()
        cand_script = script_dir / path
        if cand_script.exists():
            return cand_script.resolve()
        return cand_cwd.resolve()

    results_dir = resolve_input_path(args.results_dir)

    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    solver_xml = resolve_input_path(args.solver_xml)

    dt = args.dt if args.dt is not None else _read_dt_from_solver_xml(solver_xml)

    pattern = str(results_dir / f"{args.prefix}*.{args.ext}")
    files = sorted((Path(p) for p in glob.glob(pattern)), key=_extract_step_from_name)
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")

    # Build a VTK collection file (.pvd) that references each time point.
    root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ET.SubElement(root, "Collection")

    if args.write_timevalue:
        patched_dir = results_dir / args.timevalue_dir
        patched_dir.mkdir(parents=True, exist_ok=True)
    else:
        patched_dir = None

    for src in files:
        step = _extract_step_from_name(src)
        if dt is not None:
            time_value = dt * float(step)
        else:
            time_value = _extract_time_from_mesh_field_data(src)
            if time_value is None:
                time_value = float(step)

        if patched_dir is not None:
            dst = patched_dir / src.name
            _inject_timevalue_field_data(src=src, dst=dst, time_value=time_value)
            rel_path = os.path.relpath(dst, start=results_dir)
        else:
            rel_path = src.name

        ET.SubElement(collection, "DataSet", timestep=f"{time_value:.17g}", part="0", file=rel_path)

    out_path = results_dir / args.out
    ET.ElementTree(root).write(out_path, xml_declaration=True, encoding="UTF-8")
    print(f"Created '{out_path}' with {len(files)} time steps (dt={dt}).")


if __name__ == "__main__":
    main()
