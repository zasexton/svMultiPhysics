#!/usr/bin/env python3
"""Validate the central free-surface qualification campaign contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any


CAMPAIGN_IDS = [f"Q{index}" for index in range(8)]
REQUIRED_ARTIFACT_FILES = {
    "benchmarks.json",
    "checksums.json",
    "dependencies.json",
    "manifest.json",
    "metrics.json",
    "provenance.json",
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
METRIC_NAME = re.compile(r"^[a-z0-9_]+(?:\.[a-z0-9_]+)+$")
RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
WORK_PACKAGE_ID = re.compile(r"^WP-[0-9]+$")


class ValidationError(ValueError):
    """Raised when qualification evidence violates the campaign contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValidationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValidationError(f"expected a regular JSON file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read JSON document {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"JSON document must be an object: {path}")
    return value


def require_exact_keys(
    value: Any, expected: set[str], context: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{context} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValidationError(
            f"{context} keys differ; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return value


def require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{context} must be a nonempty string")
    return value


def require_string_list(
    value: Any, context: str, *, allow_empty: bool = False
) -> list[str]:
    if not isinstance(value, list):
        raise ValidationError(f"{context} must be a list")
    if not allow_empty and not value:
        raise ValidationError(f"{context} must not be empty")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValidationError(f"{context} must contain nonempty strings")
    if len(value) != len(set(value)):
        raise ValidationError(f"{context} must not contain duplicates")
    return value


def require_hex(value: Any, pattern: re.Pattern[str], context: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValidationError(f"{context} has an invalid hexadecimal digest")
    return value


def require_relative_path(value: Any, context: str) -> str:
    path_text = require_nonempty_string(value, context)
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValidationError(f"{context} must be a normalized relative path")
    if path.as_posix() != path_text:
        raise ValidationError(f"{context} must use normalized POSIX separators")
    return path_text


def _validate_reference(value: Any, context: str) -> None:
    reference = require_exact_keys(
        value,
        {
            "kind",
            "status",
            "title",
            "locator",
            "version",
            "data_units",
            "uncertainty",
            "limitations",
        },
        context,
    )
    if reference["kind"] not in {
        "contract_section",
        "analytic",
        "published",
        "reference_dataset",
    }:
        raise ValidationError(f"{context}.kind is unsupported")
    if reference["status"] not in {
        "PINNED",
        "CONTRACT_ONLY",
        "REFERENCE_UNRESOLVED",
    }:
        raise ValidationError(f"{context}.status is unsupported")
    for key in (
        "title",
        "locator",
        "version",
        "data_units",
        "uncertainty",
        "limitations",
    ):
        require_nonempty_string(reference[key], f"{context}.{key}")
    if (
        reference["status"] == "REFERENCE_UNRESOLVED"
        and reference["limitations"] == "not_applicable"
    ):
        raise ValidationError(
            f"{context} must explain why its reference remains unresolved"
        )


def _validate_child_program(
    value: Any, context: str, source_root: Path | None
) -> None:
    child = require_exact_keys(
        value,
        {
            "id",
            "role",
            "registration_state",
            "registry_path",
            "registry_sha256",
            "runner_path",
            "runner_sha256",
        },
        context,
    )
    child_id = require_nonempty_string(child["id"], f"{context}.id")
    if IDENTIFIER.fullmatch(child_id) is None:
        raise ValidationError(f"{context}.id is not a valid identifier")
    if child["role"] not in {"prerequisite_only", "campaign_evidence"}:
        raise ValidationError(f"{context}.role is unsupported")
    if child["registration_state"] == "UNRESOLVED":
        for key in (
            "registry_path",
            "registry_sha256",
            "runner_path",
            "runner_sha256",
        ):
            if child[key] is not None:
                raise ValidationError(
                    f"{context}.{key} must be null while registration is unresolved"
                )
        return
    if child["registration_state"] != "REGISTERED":
        raise ValidationError(f"{context}.registration_state is unsupported")
    registry_path = require_relative_path(
        child["registry_path"], f"{context}.registry_path"
    )
    runner_path = require_relative_path(
        child["runner_path"], f"{context}.runner_path"
    )
    registry_hash = require_hex(
        child["registry_sha256"], HEX64, f"{context}.registry_sha256"
    )
    runner_hash = require_hex(
        child["runner_sha256"], HEX64, f"{context}.runner_sha256"
    )
    if registry_path == runner_path:
        raise ValidationError(f"{context} registry and runner paths must differ")
    if source_root is None:
        return
    for path_text, expected_hash, label in (
        (registry_path, registry_hash, "registry"),
        (runner_path, runner_hash, "runner"),
    ):
        path = source_root / path_text
        if not path.is_file() or path.is_symlink():
            raise ValidationError(f"{context} {label} is not a regular file: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValidationError(
                f"{context} {label} hash mismatch: "
                f"expected={expected_hash}, actual={actual_hash}"
            )


def expected_metric_names(
    registry: dict[str, Any], campaign: dict[str, Any]
) -> list[str]:
    names: list[str] = []
    for set_id in campaign["metric_sets"]:
        names.extend(registry["metric_sets"][set_id])
    if len(names) != len(set(names)):
        raise ValidationError(
            f"{campaign['id']} selected metric sets contain duplicate metrics"
        )
    return names


def load_campaign_registry(
    path: Path, source_root: Path | None = None
) -> dict[str, Any]:
    registry = load_json(path)
    require_exact_keys(
        registry,
        {
            "schema_version",
            "registry_id",
            "disposition",
            "artifact_contract",
            "accepted_step_metric_sets",
            "metric_sets",
            "campaigns",
        },
        "campaign registry",
    )
    if registry["schema_version"] != 1:
        raise ValidationError("unsupported campaign registry schema")
    if registry["registry_id"] != "free_surface_qualification_campaign_v1":
        raise ValidationError("unexpected campaign registry identifier")
    if registry["disposition"] not in {
        "UNRESOLVED_CONTROL_SLICE",
        "ACTIVE",
    }:
        raise ValidationError("unsupported campaign registry disposition")

    artifact_contract = require_exact_keys(
        registry["artifact_contract"],
        {
            "root",
            "directory_pattern",
            "required_files",
            "hash_algorithm",
        },
        "artifact contract",
    )
    require_relative_path(artifact_contract["root"], "artifact contract root")
    if (
        artifact_contract["directory_pattern"]
        != "{campaign_id_lower}/{run_id}"
    ):
        raise ValidationError("unsupported artifact directory pattern")
    required_files = require_string_list(
        artifact_contract["required_files"], "artifact required files"
    )
    if set(required_files) != REQUIRED_ARTIFACT_FILES:
        raise ValidationError("artifact required files do not match the schema")
    if artifact_contract["hash_algorithm"] != "sha256":
        raise ValidationError("artifact hash algorithm must be sha256")

    metric_sets = registry["metric_sets"]
    if not isinstance(metric_sets, dict) or not metric_sets:
        raise ValidationError("metric_sets must be a nonempty object")
    all_metrics: set[str] = set()
    for set_id, metrics_value in metric_sets.items():
        if IDENTIFIER.fullmatch(set_id) is None:
            raise ValidationError(f"invalid metric set identifier: {set_id}")
        metrics = require_string_list(
            metrics_value, f"metric set {set_id}"
        )
        for metric in metrics:
            if METRIC_NAME.fullmatch(metric) is None:
                raise ValidationError(f"invalid metric name: {metric}")
            if metric in all_metrics:
                raise ValidationError(
                    f"metric appears in more than one set: {metric}"
                )
            all_metrics.add(metric)

    baseline_sets = require_string_list(
        registry["accepted_step_metric_sets"],
        "accepted step metric sets",
    )
    missing_sets = sorted(set(baseline_sets) - set(metric_sets))
    if missing_sets:
        raise ValidationError(
            f"accepted step metric sets are undefined: {missing_sets}"
        )

    campaigns = registry["campaigns"]
    if not isinstance(campaigns, list):
        raise ValidationError("campaigns must be a list")
    if [item.get("id") for item in campaigns if isinstance(item, dict)] != CAMPAIGN_IDS:
        raise ValidationError("campaigns must contain Q0 through Q7 in order")

    global_child_ids: set[str] = set()
    for sequence, value in enumerate(campaigns):
        context = f"campaign Q{sequence}"
        campaign = require_exact_keys(
            value,
            {
                "id",
                "sequence",
                "depends_on",
                "work_package_dependencies",
                "state",
                "scope",
                "child_programs",
                "metric_sets",
                "benchmarks",
                "unresolved_exits",
            },
            context,
        )
        campaign_id = CAMPAIGN_IDS[sequence]
        if campaign["id"] != campaign_id or campaign["sequence"] != sequence:
            raise ValidationError(f"{context} sequence does not match its identifier")
        dependencies = require_string_list(
            campaign["depends_on"],
            f"{context}.depends_on",
            allow_empty=sequence == 0,
        )
        if dependencies != CAMPAIGN_IDS[:sequence]:
            raise ValidationError(
                f"{context} must depend on every earlier campaign in order"
            )
        work_packages = require_string_list(
            campaign["work_package_dependencies"],
            f"{context}.work_package_dependencies",
        )
        if any(WORK_PACKAGE_ID.fullmatch(item) is None for item in work_packages):
            raise ValidationError(
                f"{context} has an invalid work-package dependency"
            )
        if campaign["state"] not in {"UNRESOLVED", "READY"}:
            raise ValidationError(f"{context}.state is unsupported")

        scope = require_exact_keys(
            campaign["scope"],
            {
                "claim",
                "evidence_class",
                "qualifies_only",
                "excludes",
                "capability_envelope",
            },
            f"{context}.scope",
        )
        if scope["claim"] != campaign_id:
            raise ValidationError(f"{context} claim must match the campaign")
        if scope["evidence_class"] != "campaign_qualification":
            raise ValidationError(f"{context} has an invalid evidence class")
        if scope["qualifies_only"] != [campaign_id]:
            raise ValidationError(f"{context} may qualify only itself")
        expected_exclusions = [
            item for item in CAMPAIGN_IDS if item != campaign_id
        ]
        if scope["excludes"] != expected_exclusions:
            raise ValidationError(
                f"{context} must explicitly exclude every other campaign"
            )
        require_nonempty_string(
            scope["capability_envelope"],
            f"{context}.scope.capability_envelope",
        )

        children = campaign["child_programs"]
        if not isinstance(children, list) or not children:
            raise ValidationError(f"{context}.child_programs must not be empty")
        campaign_child_ids: set[str] = set()
        has_campaign_evidence = False
        for index, child in enumerate(children):
            child_context = f"{context}.child_programs[{index}]"
            _validate_child_program(child, child_context, source_root)
            child_id = child["id"]
            if child_id in campaign_child_ids or child_id in global_child_ids:
                raise ValidationError(f"duplicate child program id: {child_id}")
            campaign_child_ids.add(child_id)
            global_child_ids.add(child_id)
            has_campaign_evidence = (
                has_campaign_evidence
                or child["role"] == "campaign_evidence"
            )
        if not has_campaign_evidence:
            raise ValidationError(
                f"{context} lacks a campaign-evidence child program"
            )

        selected_metric_sets = require_string_list(
            campaign["metric_sets"], f"{context}.metric_sets"
        )
        undefined = sorted(set(selected_metric_sets) - set(metric_sets))
        if undefined:
            raise ValidationError(
                f"{context} selects undefined metric sets: {undefined}"
            )
        if not set(baseline_sets).issubset(selected_metric_sets):
            raise ValidationError(
                f"{context} omits accepted-step metric sets"
            )
        if len(selected_metric_sets) <= len(baseline_sets):
            raise ValidationError(
                f"{context} must add campaign-specific expected metrics"
            )
        expected_metric_names(registry, campaign)

        benchmarks = campaign["benchmarks"]
        if not isinstance(benchmarks, list) or not benchmarks:
            raise ValidationError(f"{context}.benchmarks must not be empty")
        benchmark_ids: set[str] = set()
        for index, benchmark_value in enumerate(benchmarks):
            benchmark_context = f"{context}.benchmarks[{index}]"
            benchmark = require_exact_keys(
                benchmark_value,
                {"id", "description", "reference"},
                benchmark_context,
            )
            benchmark_id = require_nonempty_string(
                benchmark["id"], f"{benchmark_context}.id"
            )
            if IDENTIFIER.fullmatch(benchmark_id) is None:
                raise ValidationError(
                    f"{benchmark_context}.id is not a valid identifier"
                )
            if benchmark_id in benchmark_ids:
                raise ValidationError(
                    f"{context} has duplicate benchmark identifiers"
                )
            benchmark_ids.add(benchmark_id)
            require_nonempty_string(
                benchmark["description"],
                f"{benchmark_context}.description",
            )
            _validate_reference(
                benchmark["reference"],
                f"{benchmark_context}.reference",
            )

        unresolved_exits = require_string_list(
            campaign["unresolved_exits"],
            f"{context}.unresolved_exits",
            allow_empty=campaign["state"] == "READY",
        )
        if campaign["state"] == "UNRESOLVED" and not unresolved_exits:
            raise ValidationError(
                f"{context} must list its unresolved exits"
            )
        if campaign["state"] == "READY":
            if unresolved_exits:
                raise ValidationError(
                    f"{context} cannot be ready with unresolved exits"
                )
            if any(
                child["registration_state"] != "REGISTERED"
                for child in children
            ):
                raise ValidationError(
                    f"{context} cannot be ready with unresolved child programs"
                )
            if any(
                benchmark["reference"]["status"] == "REFERENCE_UNRESOLVED"
                for benchmark in benchmarks
            ):
                raise ValidationError(
                    f"{context} cannot be ready with unresolved references"
                )
    return registry


def _campaign_by_id(
    registry: dict[str, Any], campaign_id: str
) -> dict[str, Any]:
    if campaign_id not in CAMPAIGN_IDS:
        raise ValidationError(f"unknown campaign id: {campaign_id}")
    return registry["campaigns"][int(campaign_id[1:])]


def _require_artifact_header(
    document: dict[str, Any],
    document_name: str,
    campaign_id: str,
    run_id: str,
    source_commit: str,
    source_tree: str,
) -> None:
    if document["artifact_schema_version"] != 1:
        raise ValidationError(f"{document_name} has an unsupported schema")
    if document["campaign_id"] != campaign_id:
        raise ValidationError(f"{document_name} campaign id is incoherent")
    if document["run_id"] != run_id:
        raise ValidationError(f"{document_name} run id is incoherent")
    if document["source_commit"] != source_commit:
        raise ValidationError(f"{document_name} source commit is incoherent")
    if document["source_tree"] != source_tree:
        raise ValidationError(f"{document_name} source tree is incoherent")


def _validate_json_value(value: Any, context: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{context} must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{context}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValidationError(f"{context} has an invalid object key")
            _validate_json_value(item, f"{context}.{key}")
        return
    raise ValidationError(f"{context} is not a supported JSON value")


def validate_campaign_artifact(
    registry_path: Path,
    artifact_directory: Path,
    source_root: Path,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    if registry_path.is_symlink():
        raise ValidationError("campaign registry path must not be a symlink")
    registry_path = registry_path.resolve()
    registry = load_campaign_registry(registry_path, source_root)
    if artifact_directory.is_symlink():
        raise ValidationError("artifact path must be a regular directory")
    artifact_directory = artifact_directory.resolve()
    if not artifact_directory.is_dir():
        raise ValidationError("artifact path must be a regular directory")

    artifact_root_path = source_root / registry["artifact_contract"]["root"]
    if artifact_root_path.is_symlink():
        raise ValidationError("registered artifact root must not be a symlink")
    artifact_root = artifact_root_path.resolve()
    try:
        artifact_root.relative_to(source_root)
    except ValueError as error:
        raise ValidationError(
            "registered artifact root is outside the source root"
        ) from error
    try:
        relative_artifact = artifact_directory.relative_to(artifact_root)
    except ValueError as error:
        raise ValidationError(
            "artifact directory is outside the registered artifact root"
        ) from error
    if len(relative_artifact.parts) != 2:
        raise ValidationError(
            "artifact directory must have campaign/run layout"
        )

    entries = list(artifact_directory.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValidationError("artifact directory may contain only regular files")
    actual_files = {entry.name for entry in entries}
    if actual_files != REQUIRED_ARTIFACT_FILES:
        raise ValidationError(
            "artifact file layout differs; "
            f"missing={sorted(REQUIRED_ARTIFACT_FILES - actual_files)}, "
            f"unexpected={sorted(actual_files - REQUIRED_ARTIFACT_FILES)}"
        )

    documents = {
        name: load_json(artifact_directory / name)
        for name in REQUIRED_ARTIFACT_FILES
    }
    manifest = require_exact_keys(
        documents["manifest.json"],
        {
            "artifact_schema_version",
            "campaign_id",
            "run_id",
            "outcome",
            "promotion_requested",
            "claims",
            "evidence_scope",
            "campaign_registry_sha256",
            "source_commit",
            "source_tree",
            "child_programs",
        },
        "manifest",
    )
    campaign_id = manifest["campaign_id"]
    campaign = _campaign_by_id(registry, campaign_id)
    run_id = require_nonempty_string(manifest["run_id"], "manifest.run_id")
    if RUN_ID.fullmatch(run_id) is None:
        raise ValidationError("manifest.run_id is invalid")
    if relative_artifact.parts != (campaign_id.lower(), run_id):
        raise ValidationError(
            "artifact directory does not match its campaign and run id"
        )
    if manifest["artifact_schema_version"] != 1:
        raise ValidationError("manifest has an unsupported schema")
    if manifest["outcome"] not in {"PASS", "FAIL", "UNRESOLVED"}:
        raise ValidationError("manifest outcome is unsupported")
    if not isinstance(manifest["promotion_requested"], bool):
        raise ValidationError("manifest promotion_requested must be boolean")
    claims = require_string_list(
        manifest["claims"], "manifest.claims", allow_empty=True
    )
    expected_claims = (
        campaign["scope"]["qualifies_only"]
        if manifest["promotion_requested"]
        else []
    )
    if claims != expected_claims:
        raise ValidationError(
            "manifest claims exceed or differ from the registered scope"
        )
    if manifest["evidence_scope"] != campaign["scope"]:
        raise ValidationError("manifest evidence scope differs from the registry")
    registry_hash = sha256_file(registry_path)
    require_hex(
        manifest["campaign_registry_sha256"],
        HEX64,
        "manifest.campaign_registry_sha256",
    )
    if manifest["campaign_registry_sha256"] != registry_hash:
        raise ValidationError("manifest campaign registry hash mismatch")
    source_commit = require_hex(
        manifest["source_commit"], HEX40, "manifest.source_commit"
    )
    source_tree = require_hex(
        manifest["source_tree"], HEX40, "manifest.source_tree"
    )

    registered_children = {
        child["id"]: child
        for child in campaign["child_programs"]
        if child["registration_state"] == "REGISTERED"
    }
    child_records = manifest["child_programs"]
    if not isinstance(child_records, list):
        raise ValidationError("manifest.child_programs must be a list")
    if [
        item.get("id") for item in child_records if isinstance(item, dict)
    ] != list(registered_children):
        raise ValidationError(
            "manifest child programs differ from registered child programs"
        )
    for index, record_value in enumerate(child_records):
        context = f"manifest.child_programs[{index}]"
        record = require_exact_keys(
            record_value,
            {
                "id",
                "role",
                "outcome",
                "registry_sha256",
                "runner_sha256",
                "source_commit",
                "source_tree",
            },
            context,
        )
        specification = registered_children[record["id"]]
        if record["role"] != specification["role"]:
            raise ValidationError(f"{context} role differs from the registry")
        if record["outcome"] not in {"PASS", "FAIL", "UNRESOLVED"}:
            raise ValidationError(f"{context} outcome is unsupported")
        for key in ("registry_sha256", "runner_sha256"):
            require_hex(record[key], HEX64, f"{context}.{key}")
            if record[key] != specification[key]:
                raise ValidationError(
                    f"{context} {key} differs from the registered hash"
                )
        if (
            record["source_commit"] != source_commit
            or record["source_tree"] != source_tree
        ):
            raise ValidationError(
                f"{context} source revision is incoherent"
            )

    provenance = require_exact_keys(
        documents["provenance.json"],
        {
            "artifact_schema_version",
            "campaign_id",
            "run_id",
            "source_commit",
            "source_tree",
            "dirty_tree_sha256",
            "compiler",
            "libraries_sha256",
            "build_options_sha256",
            "machine",
            "mpi_ranks",
            "threads",
            "mesh_sha256",
            "reference_data_sha256",
            "dimensional_parameters_sha256",
            "nondimensional_groups_sha256",
            "acceptance_thresholds_sha256",
        },
        "provenance",
    )
    _require_artifact_header(
        provenance,
        "provenance",
        campaign_id,
        run_id,
        source_commit,
        source_tree,
    )
    for key in (
        "dirty_tree_sha256",
        "libraries_sha256",
        "build_options_sha256",
        "mesh_sha256",
        "reference_data_sha256",
        "dimensional_parameters_sha256",
        "nondimensional_groups_sha256",
        "acceptance_thresholds_sha256",
    ):
        require_hex(provenance[key], HEX64, f"provenance.{key}")
    compiler = require_exact_keys(
        provenance["compiler"], {"id", "version"}, "provenance.compiler"
    )
    require_nonempty_string(compiler["id"], "provenance.compiler.id")
    require_nonempty_string(
        compiler["version"], "provenance.compiler.version"
    )
    machine = require_exact_keys(
        provenance["machine"],
        {"system", "release", "architecture"},
        "provenance.machine",
    )
    for key in ("system", "release", "architecture"):
        require_nonempty_string(machine[key], f"provenance.machine.{key}")
    for key in ("mpi_ranks", "threads"):
        if (
            isinstance(provenance[key], bool)
            or not isinstance(provenance[key], int)
            or provenance[key] <= 0
        ):
            raise ValidationError(f"provenance.{key} must be positive")

    dependencies = require_exact_keys(
        documents["dependencies.json"],
        {
            "artifact_schema_version",
            "campaign_id",
            "run_id",
            "source_commit",
            "source_tree",
            "campaign_dependencies",
            "work_package_dependencies",
        },
        "dependencies",
    )
    _require_artifact_header(
        dependencies,
        "dependencies",
        campaign_id,
        run_id,
        source_commit,
        source_tree,
    )
    campaign_dependency_records = dependencies["campaign_dependencies"]
    if not isinstance(campaign_dependency_records, list):
        raise ValidationError(
            "dependencies.campaign_dependencies must be a list"
        )
    if [
        item.get("id")
        for item in campaign_dependency_records
        if isinstance(item, dict)
    ] != campaign["depends_on"]:
        raise ValidationError(
            "campaign dependency evidence differs from the registry order"
        )
    for index, record_value in enumerate(campaign_dependency_records):
        context = f"dependencies.campaign_dependencies[{index}]"
        record = require_exact_keys(
            record_value,
            {
                "id",
                "outcome",
                "promotion_status",
                "source_commit",
                "source_tree",
            },
            context,
        )
        if record["outcome"] not in {"PASS", "FAIL", "UNRESOLVED"}:
            raise ValidationError(f"{context} outcome is unsupported")
        if record["promotion_status"] not in {
            "QUALIFIED",
            "NOT_QUALIFIED",
        }:
            raise ValidationError(f"{context} promotion status is unsupported")
        if (
            record["source_commit"] != source_commit
            or record["source_tree"] != source_tree
        ):
            raise ValidationError(
                f"{context} source revision is incoherent"
            )

    work_package_records = dependencies["work_package_dependencies"]
    if not isinstance(work_package_records, list):
        raise ValidationError(
            "dependencies.work_package_dependencies must be a list"
        )
    if [
        item.get("id")
        for item in work_package_records
        if isinstance(item, dict)
    ] != campaign["work_package_dependencies"]:
        raise ValidationError(
            "work-package dependency evidence differs from the registry"
        )
    for index, record_value in enumerate(work_package_records):
        context = f"dependencies.work_package_dependencies[{index}]"
        record = require_exact_keys(
            record_value,
            {
                "id",
                "outcome",
                "evidence_class",
                "source_commit",
                "source_tree",
            },
            context,
        )
        if record["outcome"] not in {"PASS", "FAIL", "UNRESOLVED"}:
            raise ValidationError(f"{context} outcome is unsupported")
        if record["evidence_class"] != "prerequisite_only":
            raise ValidationError(
                f"{context} must remain prerequisite-only evidence"
            )
        if (
            record["source_commit"] != source_commit
            or record["source_tree"] != source_tree
        ):
            raise ValidationError(
                f"{context} source revision is incoherent"
            )

    benchmark_document = require_exact_keys(
        documents["benchmarks.json"],
        {
            "artifact_schema_version",
            "campaign_id",
            "run_id",
            "source_commit",
            "source_tree",
            "benchmarks",
        },
        "benchmarks",
    )
    _require_artifact_header(
        benchmark_document,
        "benchmarks",
        campaign_id,
        run_id,
        source_commit,
        source_tree,
    )
    benchmark_records = benchmark_document["benchmarks"]
    if not isinstance(benchmark_records, list):
        raise ValidationError("benchmarks.benchmarks must be a list")
    expected_benchmark_ids = [
        benchmark["id"] for benchmark in campaign["benchmarks"]
    ]
    if [
        item.get("id") for item in benchmark_records if isinstance(item, dict)
    ] != expected_benchmark_ids:
        raise ValidationError(
            "benchmark evidence is incomplete or unexpectedly ordered"
        )
    metric_names = expected_metric_names(registry, campaign)
    for index, record_value in enumerate(benchmark_records):
        context = f"benchmarks.benchmarks[{index}]"
        record = require_exact_keys(
            record_value,
            {"id", "outcome", "reference", "observed_metric_names"},
            context,
        )
        specification = campaign["benchmarks"][index]
        if record["outcome"] not in {"PASS", "FAIL", "UNRESOLVED"}:
            raise ValidationError(f"{context} outcome is unsupported")
        if record["reference"] != specification["reference"]:
            raise ValidationError(
                f"{context} reference metadata differs from the registry"
            )
        if record["observed_metric_names"] != metric_names:
            raise ValidationError(
                f"{context} does not declare every expected metric"
            )

    metric_document = require_exact_keys(
        documents["metrics.json"],
        {
            "artifact_schema_version",
            "campaign_id",
            "run_id",
            "source_commit",
            "source_tree",
            "metrics",
        },
        "metrics",
    )
    _require_artifact_header(
        metric_document,
        "metrics",
        campaign_id,
        run_id,
        source_commit,
        source_tree,
    )
    metric_records = metric_document["metrics"]
    if not isinstance(metric_records, dict):
        raise ValidationError("metrics.metrics must be an object")
    if set(metric_records) != set(metric_names):
        raise ValidationError(
            "metric evidence differs from the complete expected set; "
            f"missing={sorted(set(metric_names) - set(metric_records))}, "
            f"unexpected={sorted(set(metric_records) - set(metric_names))}"
        )
    for metric_name in metric_names:
        context = f"metrics.metrics.{metric_name}"
        record = require_exact_keys(
            metric_records[metric_name],
            {"value", "unit", "acceptance", "passed"},
            context,
        )
        _validate_json_value(record["value"], f"{context}.value")
        require_nonempty_string(record["unit"], f"{context}.unit")
        acceptance = require_exact_keys(
            record["acceptance"],
            {"relation", "target"},
            f"{context}.acceptance",
        )
        if acceptance["relation"] not in {
            "recorded",
            "equal",
            "less_equal",
            "greater_equal",
            "between",
        }:
            raise ValidationError(
                f"{context}.acceptance.relation is unsupported"
            )
        _validate_json_value(
            acceptance["target"], f"{context}.acceptance.target"
        )
        if not isinstance(record["passed"], bool):
            raise ValidationError(f"{context}.passed must be boolean")

    checksum_document = require_exact_keys(
        documents["checksums.json"],
        {"artifact_schema_version", "algorithm", "files"},
        "checksums",
    )
    if checksum_document["artifact_schema_version"] != 1:
        raise ValidationError("checksums has an unsupported schema")
    if checksum_document["algorithm"] != "sha256":
        raise ValidationError("checksums algorithm must be sha256")
    checksummed_files = REQUIRED_ARTIFACT_FILES - {"checksums.json"}
    if (
        not isinstance(checksum_document["files"], dict)
        or set(checksum_document["files"]) != checksummed_files
    ):
        raise ValidationError(
            "checksums must cover every artifact file except itself"
        )
    for name in sorted(checksummed_files):
        expected_hash = require_hex(
            checksum_document["files"][name],
            HEX64,
            f"checksums.files.{name}",
        )
        actual_hash = sha256_file(artifact_directory / name)
        if expected_hash != actual_hash:
            raise ValidationError(f"artifact checksum mismatch for {name}")

    blockers: list[str] = []
    if campaign["state"] != "READY":
        blockers.append(f"{campaign_id} registry state is unresolved")
    blockers.extend(
        f"unresolved exit: {item}" for item in campaign["unresolved_exits"]
    )
    unresolved_children = [
        child["id"]
        for child in campaign["child_programs"]
        if child["registration_state"] != "REGISTERED"
    ]
    blockers.extend(
        f"unresolved child program: {item}" for item in unresolved_children
    )
    if manifest["outcome"] != "PASS":
        blockers.append("campaign outcome is not PASS")
    if any(
        record["outcome"] != "PASS" for record in child_records
    ):
        blockers.append("one or more registered child programs did not pass")
    campaign_evidence_records = [
        record for record in child_records if record["role"] == "campaign_evidence"
    ]
    if not campaign_evidence_records:
        blockers.append(
            "prerequisite-only evidence cannot promote a campaign"
        )
    elif any(record["outcome"] != "PASS" for record in campaign_evidence_records):
        blockers.append("campaign-evidence child program did not pass")
    if any(
        record["outcome"] != "PASS"
        or record["promotion_status"] != "QUALIFIED"
        for record in campaign_dependency_records
    ):
        blockers.append("one or more campaign dependencies are not qualified")
    if any(record["outcome"] != "PASS" for record in work_package_records):
        blockers.append("one or more work-package dependencies did not pass")
    if any(record["outcome"] != "PASS" for record in benchmark_records):
        blockers.append("one or more benchmarks did not pass")
    if any(
        benchmark["reference"]["status"] == "REFERENCE_UNRESOLVED"
        for benchmark in campaign["benchmarks"]
    ):
        blockers.append("one or more benchmark references remain unresolved")
    if any(not record["passed"] for record in metric_records.values()):
        blockers.append("one or more expected metrics did not pass")

    if manifest["promotion_requested"] and blockers:
        raise ValidationError(
            "campaign promotion refused: " + "; ".join(blockers)
        )
    return {
        "campaign_id": campaign_id,
        "run_id": run_id,
        "artifact_outcome": manifest["outcome"],
        "promotion_requested": manifest["promotion_requested"],
        "promotion_allowed": not blockers,
        "promotion_blockers": blockers,
        "expected_metric_count": len(metric_names),
        "benchmark_count": len(benchmark_records),
        "registered_child_count": len(child_records),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).with_name(
            "free_surface_qualification_campaign_registry.json"
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--artifact", type=Path)
    arguments = parser.parse_args(argv)
    try:
        registry = load_campaign_registry(
            arguments.registry.resolve(),
            arguments.source_root.resolve(),
        )
        if arguments.artifact is None:
            report = {
                "registry_id": registry["registry_id"],
                "disposition": registry["disposition"],
                "campaign_states": {
                    campaign["id"]: campaign["state"]
                    for campaign in registry["campaigns"]
                },
                "unresolved_exit_count": sum(
                    len(campaign["unresolved_exits"])
                    for campaign in registry["campaigns"]
                ),
                "promotion_claimed": False,
            }
        else:
            report = validate_campaign_artifact(
                arguments.registry.resolve(),
                arguments.artifact.resolve(),
                arguments.source_root.resolve(),
            )
    except ValidationError as error:
        print(json.dumps({"outcome": "FAIL", "diagnostic": str(error)}))
        return 2
    print(json.dumps({"outcome": "PASS", "report": report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
