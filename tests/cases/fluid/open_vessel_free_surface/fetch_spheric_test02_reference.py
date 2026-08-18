#!/usr/bin/env python3
"""Fetch and summarize official SPHERIC Test02 experimental histories."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any


SOURCE_URL = (
    "https://9449af45-2363-44f6-8b03-03b6b7c2aee5.usrfiles.com/archives/"
    "9449af_6ce73873d6d04315853e481c337dbfca.zip"
)
WORKBOOK_MEMBER = "test_case_2_exp_data.xls"
DESCRIPTION_MEMBER = "test_case_2_v1p1.pdf"
GEOMETRY_MEMBER = "test_case_2.f"
TARGET_TRACES = ["H4", "H2", "P1", "P3", "P5", "P7"]


def read_zip_bytes(zip_file: Path | None, url: str) -> tuple[bytes, dict[str, Any]]:
    if zip_file is not None:
        data = zip_file.read_bytes()
        return data, {"source": str(zip_file), "content_length_bytes": len(data)}

    request = urllib.request.Request(url, headers={"User-Agent": "svMultiPhysics-validation"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
        headers = dict(response.headers.items())
    return data, {
        "source": url,
        "content_length_bytes": len(data),
        "source_last_modified": headers.get("Last-Modified"),
    }


def workbook_rows(workbook_bytes: bytes) -> tuple[list[str], list[list[float]]]:
    try:
        import xlrd
    except ImportError as error:
        raise RuntimeError(
            "SPHERIC Test02 workbook parsing requires xlrd for the legacy .xls file"
        ) from error

    book = xlrd.open_workbook(file_contents=workbook_bytes)
    sheet = book.sheet_by_name("Experimental_data")
    headers = [str(sheet.cell_value(0, col)).strip() for col in range(sheet.ncols)]
    keep = [index for index, name in enumerate(headers) if name]
    clean_headers = [headers[index].replace(" (s)", "").replace(" (Pa)", "").replace(" (m)", "") for index in keep]
    rows: list[list[float]] = []
    for row_index in range(1, sheet.nrows):
        values = []
        for col in keep:
            value = sheet.cell_value(row_index, col)
            if value == "":
                break
            values.append(float(value))
        if len(values) == len(keep):
            rows.append(values)
    return clean_headers, rows


def column_summary(headers: list[str], rows: list[list[float]], name: str) -> dict[str, float]:
    index = headers.index(name)
    values = [row[index] for row in rows]
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def write_csv(path: Path, headers: list[str], rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def build_summary(
    headers: list[str],
    rows: list[list[float]],
    zip_info: dict[str, Any],
    workbook_size: int,
    csv_output: Path | None,
) -> dict[str, Any]:
    time_index = headers.index("Time")
    times = [row[time_index] for row in rows]
    dt_values = [right - left for left, right in zip(times, times[1:])]
    pressure_names = [name for name in headers if name.startswith("P")]
    height_names = [name for name in headers if name.startswith("H")]
    return {
        "source": zip_info,
        "workbook_member": WORKBOOK_MEMBER,
        "description_member": DESCRIPTION_MEMBER,
        "geometry_member": GEOMETRY_MEMBER,
        "workbook_size_bytes": workbook_size,
        "csv_output": str(csv_output) if csv_output is not None else None,
        "table": {
            "rows": len(rows),
            "headers": headers,
            "time_start_s": times[0],
            "time_end_s": times[-1],
            "dt_min_s": min(dt_values),
            "dt_max_s": max(dt_values),
            "dt_median_s": statistics.median(dt_values),
            "pressure_pa": {
                name: column_summary(headers, rows, name) for name in pressure_names
            },
            "height_m": {
                name: column_summary(headers, rows, name) for name in height_names
            },
        },
        "literature_geometry_m": {
            "tank_length": 3.22,
            "tank_width": 1.0,
            "tank_height": 1.0,
            "initial_column_length": 1.228,
            "initial_column_x_min": 1.992,
            "initial_column_x_max": 3.22,
            "initial_column_height": 0.55,
            "obstacle_x_min": 0.6635,
            "obstacle_x_max": 0.8245,
            "obstacle_flow_direction_length": 0.161,
            "obstacle_lateral_width": 0.403,
            "obstacle_z_min": 0.2985,
            "obstacle_z_max": 0.7015,
            "obstacle_height": 0.161,
            "height_probe_x_positions": {
                "H1": 0.496,
                "H2": 0.992,
                "H3": 1.488,
                "H4": 2.632,
            },
        },
        "recommended_validation_traces": TARGET_TRACES,
        "status": "reference_series_fetchable",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-file", type=Path)
    parser.add_argument("--url", default=SOURCE_URL)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    zip_bytes, zip_info = read_zip_bytes(args.zip_file, args.url)
    with zipfile.ZipFile(BytesIO(zip_bytes)) as archive:
        workbook_bytes = archive.read(WORKBOOK_MEMBER)
    headers, rows = workbook_rows(workbook_bytes)
    if args.csv_output is not None:
        write_csv(args.csv_output, headers, rows)
    summary = build_summary(headers, rows, zip_info, len(workbook_bytes), args.csv_output)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
