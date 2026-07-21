#!/usr/bin/env python3
"""Create a PVD collection for result_*.vtu or result_*.pvtu files."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def result_step(path: Path, prefix: str) -> int:
    match = re.match(rf"{re.escape(prefix)}_(\d+)\.p?vtu$", path.name)
    return int(match.group(1)) if match else -1


def output_results(case_dir: Path, prefix: str) -> list[Path]:
    return sorted(
        [
            *case_dir.glob(f"{prefix}_*.vtu"),
            *case_dir.glob(f"{prefix}_*.pvtu"),
        ],
        key=lambda path: result_step(path, prefix),
    )


def solver_time_step(case_dir: Path) -> float:
    solver_xml = case_dir / "solver.xml"
    if not solver_xml.exists():
        return 1.0
    root = ET.parse(solver_xml).getroot()
    value = root.findtext("GeneralSimulationParameters/Time_step_size")
    if value is None:
        return 1.0
    return float(value.strip())


def collate_time_series(
    case_dir: Path,
    *,
    prefix: str = "result",
    output_name: str = "result.pvd",
    force: bool = False,
) -> dict[str, Any]:
    case_dir = Path(case_dir)
    pvd_path = case_dir / output_name
    results = output_results(case_dir, prefix)
    if pvd_path.exists() and not force:
        return {
            "generated": False,
            "reason": "pvd_exists",
            "path": str(pvd_path),
            "result_count": len(results),
        }
    if not results:
        return {
            "generated": False,
            "reason": "no_results",
            "path": str(pvd_path),
            "result_count": 0,
        }

    dt = solver_time_step(case_dir)
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for result in results:
        step = result_step(result, prefix)
        time = dt * step if step >= 0 else 0.0
        lines.append(
            f'    <DataSet timestep="{time:.16g}" group="" part="0" file="{result.name}"/>'
        )
    lines.extend(["  </Collection>", "</VTKFile>", ""])
    pvd_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "generated": True,
        "path": str(pvd_path),
        "result_count": len(results),
        "first_result": results[0].name,
        "last_result": results[-1].name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--prefix", default="result")
    parser.add_argument("--output-name", default="result.pvd")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            collate_time_series(
                args.case_dir,
                prefix=args.prefix,
                output_name=args.output_name,
                force=args.force,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
