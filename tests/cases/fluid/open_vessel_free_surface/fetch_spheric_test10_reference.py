#!/usr/bin/env python3
"""Fetch SPHERIC Test10 pressure/motion reference files from the official ZIP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import struct
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TEST10_ZIP_URL = (
    "https://9449af45-2363-44f6-8b03-03b6b7c2aee5.usrfiles.com/archives/"
    "9449af_05f4aa0eaa1d4164a1024d07c562592a.zip"
)
DEFAULT_MEMBER = "SPHERIC_TestCase10/data_files/lateral_water_1x.txt"
TAIL_READ_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ZipEntry:
    name: str
    method: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


def request_headers(url: str) -> dict[str, str]:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        return {key.lower(): value for key, value in response.headers.items()}


def fetch_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end}"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def parse_central_directory(url: str, archive_size: int) -> list[ZipEntry]:
    tail_size = min(TAIL_READ_BYTES, archive_size)
    tail_start = archive_size - tail_size
    tail = fetch_range(url, tail_start, archive_size - 1)
    eocd_offset = tail.rfind(b"PK\x05\x06")
    if eocd_offset < 0:
        raise RuntimeError("could not locate ZIP end-of-central-directory record")

    eocd = tail[eocd_offset:eocd_offset + 22]
    if len(eocd) != 22:
        raise RuntimeError("truncated ZIP end-of-central-directory record")
    (
        _signature,
        _disk,
        _central_directory_disk,
        _entries_on_disk,
        total_entries,
        central_directory_size,
        central_directory_offset,
        comment_length,
    ) = struct.unpack("<4s4H2LH", eocd)
    if comment_length:
        raise RuntimeError("ZIP comments are not expected for the Test10 archive")

    directory = fetch_range(
        url,
        central_directory_offset,
        central_directory_offset + central_directory_size - 1,
    )
    entries: list[ZipEntry] = []
    offset = 0
    while offset < len(directory):
        if directory[offset:offset + 4] != b"PK\x01\x02":
            raise RuntimeError(f"invalid central-directory signature at {offset}")
        fields = struct.unpack("<4s6H3L5H2L", directory[offset:offset + 46])
        (
            _signature,
            _version_made_by,
            _version_needed,
            _flags,
            method,
            _mtime,
            _mdate,
            _crc,
            compressed_size,
            uncompressed_size,
            filename_length,
            extra_length,
            comment_length,
            _disk_start,
            _internal_attributes,
            _external_attributes,
            local_header_offset,
        ) = fields
        name_start = offset + 46
        name_end = name_start + filename_length
        name = directory[name_start:name_end].decode("utf-8")
        entries.append(
            ZipEntry(
                name=name,
                method=method,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                local_header_offset=local_header_offset,
            )
        )
        offset = name_end + extra_length + comment_length

    if len(entries) != total_entries:
        raise RuntimeError(
            f"central-directory entry count mismatch: {len(entries)} != {total_entries}"
        )
    return entries


def extract_entry(url: str, entry: ZipEntry) -> bytes:
    local = fetch_range(url, entry.local_header_offset, entry.local_header_offset + 4095)
    if local[:4] != b"PK\x03\x04":
        raise RuntimeError(f"invalid local-header signature for {entry.name}")
    fields = struct.unpack("<4s5H3L2H", local[:30])
    filename_length = fields[-2]
    extra_length = fields[-1]
    data_start = entry.local_header_offset + 30 + filename_length + extra_length
    compressed = fetch_range(
        url,
        data_start,
        data_start + entry.compressed_size - 1,
    )
    if entry.method == 0:
        data = compressed
    elif entry.method == 8:
        data = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError(f"unsupported ZIP compression method {entry.method}")
    if len(data) != entry.uncompressed_size:
        raise RuntimeError(
            f"extracted size mismatch for {entry.name}: "
            f"{len(data)} != {entry.uncompressed_size}"
        )
    return data


def parse_pressure_motion_summary(data: bytes) -> dict[str, Any]:
    text = data.decode("latin1")
    rows = csv.DictReader(text.splitlines(), delimiter="\t")
    times: list[float] = []
    pressures: list[float] = []
    positions: list[float] = []
    velocities: list[float] = []
    accelerations: list[float] = []
    for row in rows:
        times.append(float(row["Time[s]"]))
        pressures.append(float(row["Pressure[mbar]"]))
        positions.append(float(row["Position_smooth_splines [deg]"]))
        velocities.append(float(row["Velocity[deg\\s]"]))
        accelerations.append(float(row["Aceleration[deg\\s2]"]))

    if not times:
        raise RuntimeError("reference table contains no data rows")

    dt_values = [right - left for left, right in zip(times[:-1], times[1:])]
    finite_dt = [value for value in dt_values if math.isfinite(value)]
    return {
        "rows": len(times),
        "time_start_s": times[0],
        "time_end_s": times[-1],
        "dt_min_s": min(finite_dt) if finite_dt else None,
        "dt_max_s": max(finite_dt) if finite_dt else None,
        "dt_median_s": statistics.median(finite_dt) if finite_dt else None,
        "pressure_min_mbar": min(pressures),
        "pressure_max_mbar": max(pressures),
        "pressure_mean_mbar": statistics.fmean(pressures),
        "roll_position_min_deg": min(positions),
        "roll_position_max_deg": max(positions),
        "roll_velocity_min_deg_per_s": min(velocities),
        "roll_velocity_max_deg_per_s": max(velocities),
        "roll_acceleration_min_deg_per_s2": min(accelerations),
        "roll_acceleration_max_deg_per_s2": max(accelerations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a SPHERIC Test10 reference member from the official ZIP "
            "using HTTP range requests."
        )
    )
    parser.add_argument("--url", default=TEST10_ZIP_URL)
    parser.add_argument("--member", default=DEFAULT_MEMBER)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()

    headers = request_headers(args.url)
    archive_size_text = headers.get("content-length")
    if archive_size_text is None:
        raise RuntimeError("server did not provide archive Content-Length")
    archive_size = int(archive_size_text)
    entries = parse_central_directory(args.url, archive_size)
    by_name = {entry.name: entry for entry in entries}
    entry = by_name.get(args.member)
    if entry is None:
        available = "\n".join(sorted(by_name))
        raise RuntimeError(f"member not found: {args.member}\navailable:\n{available}")

    data = extract_entry(args.url, entry)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)

    summary = {
        "source_url": args.url,
        "source_content_length_bytes": archive_size,
        "source_last_modified": headers.get("last-modified"),
        "member": entry.name,
        "compressed_size_bytes": entry.compressed_size,
        "uncompressed_size_bytes": entry.uncompressed_size,
        "output": str(args.output) if args.output is not None else None,
        "table": parse_pressure_motion_summary(data),
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
