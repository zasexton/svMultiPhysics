#!/usr/bin/env python3
"""Fetch and verify the published transient capillary-rise histories."""

from __future__ import annotations

import argparse
import csv
import hashlib
from io import BytesIO, StringIO
import json
import math
from pathlib import Path
import sys
from typing import Any
import urllib.request
import zipfile


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[4]
DEFAULT_REGISTRY = (
    REPOSITORY_ROOT
    / "tests"
    / "cases"
    / "fluid"
    / "free_surface_wp5_capillary_rise_reference.json"
)
USER_AGENT = "svMultiPhysics-free-surface-validation"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def md5_bytes(payload: bytes) -> str:
    return hashlib.md5(payload, usedforsecurity=False).hexdigest()


def load_registry(path: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "reference_id",
        "status",
        "verified_date",
        "citation",
        "access",
        "archive",
        "model_applicability",
        "comparison_contract",
        "selected_series",
        "selected_convergence_records",
    }
    if set(registry) != required:
        raise ValueError("capillary-rise reference registry keys changed")
    if registry["schema_version"] != 1:
        raise ValueError("unsupported capillary-rise reference schema")
    if registry["status"] != "PINNED_SOURCE_AND_RAW_SERIES":
        raise ValueError("capillary-rise reference status is not source-pinned")
    if registry["access"].get("license") != "CC-BY-NC-4.0":
        raise ValueError("capillary-rise reference license changed")
    series = registry["selected_series"]
    if not isinstance(series, list) or len(series) != 4:
        raise ValueError("capillary-rise registry must select four series")
    member_names: set[str] = set()
    output_names: set[str] = set()
    methods: set[str] = set()
    for entry in series:
        if not isinstance(entry, dict) or set(entry) != {
            "method",
            "member",
            "output_name",
            "size_bytes",
            "row_count",
            "sha256",
        }:
            raise ValueError("capillary-rise series contract is malformed")
        if not isinstance(entry["size_bytes"], int) or entry["size_bytes"] <= 0:
            raise ValueError("capillary-rise series size is invalid")
        if not isinstance(entry["row_count"], int) or entry["row_count"] <= 1:
            raise ValueError("capillary-rise series row count is invalid")
        if len(entry["sha256"]) != 64:
            raise ValueError("capillary-rise series hash is invalid")
        if Path(entry["member"]).is_absolute() or ".." in Path(
            entry["member"]
        ).parts:
            raise ValueError("capillary-rise archive member is unsafe")
        if Path(entry["output_name"]).name != entry["output_name"]:
            raise ValueError("capillary-rise output name is unsafe")
        member_names.add(entry["member"])
        output_names.add(entry["output_name"])
        methods.add(entry["method"])
    if len(member_names) != 4 or len(output_names) != 4 or len(methods) != 4:
        raise ValueError("capillary-rise selected series are not unique")
    convergence_records = registry["selected_convergence_records"]
    if not isinstance(convergence_records, list) or len(convergence_records) != 4:
        raise ValueError("capillary-rise registry must select four convergence records")
    for entry in convergence_records:
        if not isinstance(entry, dict) or set(entry) != {
            "method",
            "member",
            "output_name",
            "size_bytes",
            "resolution_count",
            "sha256",
        }:
            raise ValueError("capillary-rise convergence contract is malformed")
        if entry["method"] not in methods:
            raise ValueError("capillary-rise convergence method is unknown")
        if entry["member"] in member_names:
            raise ValueError("capillary-rise archive member is duplicated")
        if entry["output_name"] in output_names:
            raise ValueError("capillary-rise output name is duplicated")
        if Path(entry["member"]).is_absolute() or ".." in Path(
            entry["member"]
        ).parts:
            raise ValueError("capillary-rise convergence member is unsafe")
        if Path(entry["output_name"]).name != entry["output_name"]:
            raise ValueError("capillary-rise convergence output name is unsafe")
        if not isinstance(entry["size_bytes"], int) or entry["size_bytes"] <= 0:
            raise ValueError("capillary-rise convergence size is invalid")
        if (
            not isinstance(entry["resolution_count"], int)
            or entry["resolution_count"] < 3
        ):
            raise ValueError("capillary-rise convergence count is invalid")
        if len(entry["sha256"]) != 64:
            raise ValueError("capillary-rise convergence hash is invalid")
        member_names.add(entry["member"])
        output_names.add(entry["output_name"])
    if len(member_names) != 8 or len(output_names) != 8:
        raise ValueError("capillary-rise selected archive files are not unique")
    return registry


def read_archive(
    archive_file: Path | None,
    url: str,
) -> tuple[bytes, dict[str, Any]]:
    if archive_file is not None:
        payload = archive_file.read_bytes()
        return payload, {
            "source": str(archive_file.resolve()),
            "source_last_modified": None,
        }

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
        last_modified = response.headers.get("Last-Modified")
    return payload, {
        "source": url,
        "source_last_modified": last_modified,
    }


def verify_archive(payload: bytes, archive_contract: dict[str, Any]) -> None:
    if len(payload) != archive_contract["size_bytes"]:
        raise ValueError("capillary-rise archive size does not match the registry")
    if sha256_bytes(payload) != archive_contract["sha256"]:
        raise ValueError("capillary-rise archive SHA-256 does not match the registry")
    if md5_bytes(payload) != archive_contract["repository_md5"]:
        raise ValueError("capillary-rise archive repository checksum does not match")


def parse_curve(payload: bytes) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("capillary-rise series is not UTF-8 text") from error
    rows: list[tuple[float, float]] = []
    for row_number, row in enumerate(csv.reader(StringIO(text)), start=1):
        if len(row) != 2:
            raise ValueError(
                f"capillary-rise series row {row_number} does not have two columns"
            )
        try:
            time_s = float(row[0])
            height_mm = float(row[1])
        except ValueError as error:
            raise ValueError(
                f"capillary-rise series row {row_number} is not numeric"
            ) from error
        if not math.isfinite(time_s) or not math.isfinite(height_mm):
            raise ValueError(
                f"capillary-rise series row {row_number} is not finite"
            )
        if rows and time_s < rows[-1][0]:
            raise ValueError(
                f"capillary-rise series time decreases at row {row_number}"
            )
        rows.append((time_s, height_mm))
    if len(rows) < 2:
        raise ValueError("capillary-rise series has fewer than two rows")
    duplicate_times = sum(
        current[0] == previous[0]
        for previous, current in zip(rows, rows[1:])
    )
    heights = [row[1] for row in rows]
    summary = {
        "row_count": len(rows),
        "time_start_s": rows[0][0],
        "time_end_s": rows[-1][0],
        "height_min_mm": min(heights),
        "height_max_mm": max(heights),
        "duplicate_time_count": duplicate_times,
    }
    return rows, summary


def parse_convergence_record(payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("capillary-rise convergence record is not UTF-8") from error
    parsed: list[list[float]] = []
    for row_number, row in enumerate(csv.reader(StringIO(text)), start=1):
        try:
            values = [float(value) for value in row]
        except ValueError as error:
            raise ValueError(
                f"capillary-rise convergence row {row_number} is not numeric"
            ) from error
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError(
                f"capillary-rise convergence row {row_number} is invalid"
            )
        parsed.append(values)
    if len(parsed) != 3 or len(parsed[0]) < 3:
        raise ValueError("capillary-rise convergence record must have three rows")
    if any(len(row) != len(parsed[0]) for row in parsed):
        raise ValueError("capillary-rise convergence rows have unequal lengths")
    resolutions, maximum_errors, integrated_errors = parsed
    if any(right <= left for left, right in zip(resolutions, resolutions[1:])):
        raise ValueError("capillary-rise convergence resolutions do not increase")
    if any(value < 0.0 for value in maximum_errors + integrated_errors):
        raise ValueError("capillary-rise convergence error is negative")
    return {
        "resolution_count": len(resolutions),
        "minimum_cells_per_half_gap": resolutions[0],
        "maximum_compared_cells_per_half_gap": resolutions[-1],
        "finest_compared_maximum_height_error_mm": maximum_errors[-1],
        "finest_compared_integrated_height_error": integrated_errors[-1],
    }


def extract_selected_series(
    payload: bytes,
    registry: dict[str, Any],
) -> tuple[
    dict[str, bytes],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    extracted: dict[str, bytes] = {}
    summaries: list[dict[str, Any]] = []
    convergence_summaries: list[dict[str, Any]] = []
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        archive_names = set(archive.namelist())
        for contract in registry["selected_series"]:
            member = contract["member"]
            if member not in archive_names:
                raise ValueError(
                    f"capillary-rise archive member is missing: {member}"
                )
            member_payload = archive.read(member)
            if len(member_payload) != contract["size_bytes"]:
                raise ValueError(
                    f"capillary-rise member size changed: {member}"
                )
            member_sha256 = sha256_bytes(member_payload)
            if member_sha256 != contract["sha256"]:
                raise ValueError(
                    f"capillary-rise member hash changed: {member}"
                )
            _, curve_summary = parse_curve(member_payload)
            if curve_summary["row_count"] != contract["row_count"]:
                raise ValueError(
                    f"capillary-rise member row count changed: {member}"
                )
            output_name = contract["output_name"]
            extracted[output_name] = member_payload
            summaries.append(
                {
                    "method": contract["method"],
                    "archive_member": member,
                    "output_name": output_name,
                    "sha256": member_sha256,
                    **curve_summary,
                }
            )
        for contract in registry["selected_convergence_records"]:
            member = contract["member"]
            if member not in archive_names:
                raise ValueError(
                    f"capillary-rise convergence member is missing: {member}"
                )
            member_payload = archive.read(member)
            if len(member_payload) != contract["size_bytes"]:
                raise ValueError(
                    f"capillary-rise convergence member size changed: {member}"
                )
            member_sha256 = sha256_bytes(member_payload)
            if member_sha256 != contract["sha256"]:
                raise ValueError(
                    f"capillary-rise convergence member hash changed: {member}"
                )
            convergence_summary = parse_convergence_record(member_payload)
            if convergence_summary["resolution_count"] != contract[
                "resolution_count"
            ]:
                raise ValueError(
                    f"capillary-rise convergence count changed: {member}"
                )
            output_name = contract["output_name"]
            extracted[output_name] = member_payload
            convergence_summaries.append(
                {
                    "method": contract["method"],
                    "archive_member": member,
                    "output_name": output_name,
                    "sha256": member_sha256,
                    **convergence_summary,
                }
            )
    return extracted, summaries, convergence_summaries


def write_outputs(
    output_directory: Path,
    extracted: dict[str, bytes],
    summary: dict[str, Any],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=False)
    checksums: list[str] = []
    for output_name in sorted(extracted):
        payload = extracted[output_name]
        destination = output_directory / output_name
        destination.write_bytes(payload)
        checksums.append(f"{sha256_bytes(payload)}  {output_name}")
    summary_path = output_directory / "reference_manifest.json"
    summary_payload = (json.dumps(summary, indent=2) + "\n").encode("utf-8")
    summary_path.write_bytes(summary_payload)
    checksums.append(
        f"{sha256_bytes(summary_payload)}  {summary_path.name}"
    )
    (output_directory / "checksums.txt").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )


def build_summary(
    registry: dict[str, Any],
    source: dict[str, Any],
    series: list[dict[str, Any]],
    convergence_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "outcome": "PASS",
        "reference_id": registry["reference_id"],
        "verified_date": registry["verified_date"],
        "source": source,
        "citation": registry["citation"],
        "license": registry["access"]["license"],
        "archive": registry["archive"],
        "model_applicability": registry["model_applicability"],
        "comparison_contract": registry["comparison_contract"],
        "series": series,
        "convergence_records": convergence_records,
    }


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--archive-file", type=Path)
    parser.add_argument("--url")
    parser.add_argument("--output-directory", type=Path)
    options = parser.parse_args(arguments)

    registry = load_registry(options.registry)
    url = options.url or registry["access"]["url"]
    payload, source = read_archive(options.archive_file, url)
    verify_archive(payload, registry["archive"])
    extracted, series, convergence_records = extract_selected_series(
        payload, registry
    )
    source.update(
        {
            "content_length_bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "repository_md5": md5_bytes(payload),
        }
    )
    summary = build_summary(registry, source, series, convergence_records)
    if options.output_directory is not None:
        write_outputs(options.output_directory, extracted, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
