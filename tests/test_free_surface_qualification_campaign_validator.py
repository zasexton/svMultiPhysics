import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = (
    ROOT
    / "tests"
    / "cases"
    / "fluid"
    / "validate_free_surface_qualification_campaign.py"
)
REGISTRY_PATH = VALIDATOR_PATH.with_name(
    "free_surface_qualification_campaign_registry.json"
)
SOURCE_COMMIT = "1" * 40
SOURCE_TREE = "2" * 40
SHA_VALUE = "3" * 64


def load_validator():
    specification = importlib.util.spec_from_file_location(
        "free_surface_qualification_campaign_validator",
        VALIDATOR_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def make_registry(source_root, *, ready=False):
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry["artifact_contract"]["root"] = "artifacts"

    child_directory = source_root / "children"
    child_directory.mkdir(parents=True)
    wp0_registry = child_directory / "wp0_registry.json"
    wp0_runner = child_directory / "wp0_runner.py"
    wp0_registry.write_text('{"matrix": "wp0"}\n', encoding="utf-8")
    wp0_runner.write_text('print("wp0")\n', encoding="utf-8")

    q0 = registry["campaigns"][0]
    wp0 = q0["child_programs"][0]
    wp0.update(
        {
            "registry_path": "children/wp0_registry.json",
            "registry_sha256": file_hash(wp0_registry),
            "runner_path": "children/wp0_runner.py",
            "runner_sha256": file_hash(wp0_runner),
        }
    )
    if ready:
        q0_registry = child_directory / "q0_registry.json"
        q0_runner = child_directory / "q0_runner.py"
        q0_registry.write_text('{"matrix": "q0"}\n', encoding="utf-8")
        q0_runner.write_text('print("q0")\n', encoding="utf-8")
        q0["child_programs"][1].update(
            {
                "registration_state": "REGISTERED",
                "registry_path": "children/q0_registry.json",
                "registry_sha256": file_hash(q0_registry),
                "runner_path": "children/q0_runner.py",
                "runner_sha256": file_hash(q0_runner),
            }
        )
        q0["state"] = "READY"
        q0["unresolved_exits"] = []

    registry_path = source_root / "campaign_registry.json"
    write_json(registry_path, registry)
    return registry_path


def refresh_checksums(artifact_directory):
    files = {}
    for path in artifact_directory.iterdir():
        if path.name != "checksums.json":
            files[path.name] = file_hash(path)
    write_json(
        artifact_directory / "checksums.json",
        {
            "artifact_schema_version": 1,
            "algorithm": "sha256",
            "files": files,
        },
    )


def make_artifact(
    source_root,
    registry_path,
    *,
    promotion_requested=False,
    artifact_outcome="PASS",
):
    validator = load_validator()
    registry = validator.load_campaign_registry(registry_path, source_root)
    campaign = registry["campaigns"][0]
    run_id = "q0-run-001"
    artifact_directory = source_root / "artifacts" / "q0" / run_id
    artifact_directory.mkdir(parents=True)

    registered_children = [
        child
        for child in campaign["child_programs"]
        if child["registration_state"] == "REGISTERED"
    ]
    child_records = [
        {
            "id": child["id"],
            "role": child["role"],
            "outcome": "PASS",
            "registry_sha256": child["registry_sha256"],
            "runner_sha256": child["runner_sha256"],
            "source_commit": SOURCE_COMMIT,
            "source_tree": SOURCE_TREE,
        }
        for child in registered_children
    ]
    metric_names = validator.expected_metric_names(registry, campaign)
    metrics = {
        name: {
            "value": 0,
            "unit": "recorded",
            "acceptance": {"relation": "recorded", "target": None},
            "passed": True,
        }
        for name in metric_names
    }
    manifest = {
        "artifact_schema_version": 1,
        "campaign_id": "Q0",
        "run_id": run_id,
        "outcome": artifact_outcome,
        "promotion_requested": promotion_requested,
        "claims": ["Q0"] if promotion_requested else [],
        "evidence_scope": copy.deepcopy(campaign["scope"]),
        "campaign_registry_sha256": file_hash(registry_path),
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "child_programs": child_records,
    }
    provenance = {
        "artifact_schema_version": 1,
        "campaign_id": "Q0",
        "run_id": run_id,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "dirty_tree_sha256": SHA_VALUE,
        "compiler": {"id": "fixture", "version": "1"},
        "libraries_sha256": SHA_VALUE,
        "build_options_sha256": SHA_VALUE,
        "machine": {
            "system": "fixture",
            "release": "1",
            "architecture": "fixture",
        },
        "mpi_ranks": 1,
        "threads": 1,
        "mesh_sha256": SHA_VALUE,
        "reference_data_sha256": SHA_VALUE,
        "dimensional_parameters_sha256": SHA_VALUE,
        "nondimensional_groups_sha256": SHA_VALUE,
        "acceptance_thresholds_sha256": SHA_VALUE,
    }
    dependencies = {
        "artifact_schema_version": 1,
        "campaign_id": "Q0",
        "run_id": run_id,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "campaign_dependencies": [],
        "work_package_dependencies": [
            {
                "id": work_package,
                "outcome": "PASS",
                "evidence_class": "prerequisite_only",
                "source_commit": SOURCE_COMMIT,
                "source_tree": SOURCE_TREE,
            }
            for work_package in campaign["work_package_dependencies"]
        ],
    }
    benchmarks = {
        "artifact_schema_version": 1,
        "campaign_id": "Q0",
        "run_id": run_id,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "benchmarks": [
            {
                "id": benchmark["id"],
                "outcome": "PASS",
                "reference": copy.deepcopy(benchmark["reference"]),
                "observed_metric_names": metric_names,
            }
            for benchmark in campaign["benchmarks"]
        ],
    }
    metric_document = {
        "artifact_schema_version": 1,
        "campaign_id": "Q0",
        "run_id": run_id,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "metrics": metrics,
    }
    for name, document in (
        ("manifest.json", manifest),
        ("provenance.json", provenance),
        ("dependencies.json", dependencies),
        ("benchmarks.json", benchmarks),
        ("metrics.json", metric_document),
    ):
        write_json(artifact_directory / name, document)
    refresh_checksums(artifact_directory)
    return artifact_directory


def mutate_document(artifact_directory, name, mutation):
    path = artifact_directory / name
    document = json.loads(path.read_text(encoding="utf-8"))
    mutation(document)
    write_json(path, document)
    refresh_checksums(artifact_directory)


def test_central_registry_is_strict_and_explicitly_unresolved():
    validator = load_validator()
    registry = validator.load_campaign_registry(REGISTRY_PATH, ROOT)
    assert [campaign["id"] for campaign in registry["campaigns"]] == [
        f"Q{index}" for index in range(8)
    ]
    assert all(
        campaign["state"] == "UNRESOLVED"
        and campaign["unresolved_exits"]
        for campaign in registry["campaigns"]
    )
    q0 = registry["campaigns"][0]
    assert q0["scope"]["qualifies_only"] == ["Q0"]
    assert any(
        child["role"] == "prerequisite_only"
        and child["registration_state"] == "REGISTERED"
        for child in q0["child_programs"]
    )
    assert any(
        child["role"] == "campaign_evidence"
        and child["registration_state"] == "UNRESOLVED"
        for child in q0["child_programs"]
    )
    metric_names = validator.expected_metric_names(registry, q0)
    assert "volume.raw_components" in metric_names
    assert "energy.accepted_balance_residual" in metric_names
    assert "geometry.validation_maxima" in metric_names


def test_q4_reference_is_pinned_without_promoting_the_campaign():
    validator = load_validator()
    registry = validator.load_campaign_registry(REGISTRY_PATH, ROOT)
    q4 = registry["campaigns"][4]
    reference = q4["benchmarks"][0]["reference"]

    assert q4["id"] == "Q4"
    assert q4["state"] == "UNRESOLVED"
    assert reference["kind"] == "reference_dataset"
    assert reference["status"] == "PINNED"
    assert "10.1016/j.apm.2020.04.020" in reference["locator"]
    assert "10.25534/tudatalib-173" in reference["locator"]
    assert reference["version"] == (
        "comparison contract gruending_2020_omega1_intercode_envelope_v1"
    )
    assert reference["data_units"] == (
        "time in seconds and rise height in millimetres"
    )
    assert "pointwise half-range" in reference["uncertainty"]
    assert "candidate numerical uncertainty" in reference["uncertainty"]
    assert "does not execute the candidate" in reference["limitations"]
    assert q4["unresolved_exits"][-1] == (
        "capillary_rise_candidate_refinement_and_uncertainty_comparison_not_executed"
    )


def test_record_only_prerequisite_artifact_cannot_promote(tmp_path):
    validator = load_validator()
    registry_path = make_registry(tmp_path)
    artifact = make_artifact(tmp_path, registry_path)
    report = validator.validate_campaign_artifact(
        registry_path, artifact, tmp_path
    )
    assert report["promotion_requested"] is False
    assert report["promotion_allowed"] is False
    assert any(
        "prerequisite-only evidence" in item
        for item in report["promotion_blockers"]
    )


def test_complete_ready_campaign_artifact_can_promote(tmp_path):
    validator = load_validator()
    registry_path = make_registry(tmp_path, ready=True)
    artifact = make_artifact(
        tmp_path, registry_path, promotion_requested=True
    )
    report = validator.validate_campaign_artifact(
        registry_path, artifact, tmp_path
    )
    assert report["promotion_allowed"] is True
    assert report["promotion_blockers"] == []


@pytest.mark.parametrize("defect", ["missing", "unexpected", "symlink"])
def test_artifact_layout_defects_are_rejected(tmp_path, defect):
    validator = load_validator()
    registry_path = make_registry(tmp_path)
    artifact = make_artifact(tmp_path, registry_path)
    if defect == "missing":
        (artifact / "metrics.json").unlink()
    elif defect == "unexpected":
        (artifact / "extra.json").write_text("{}\n", encoding="utf-8")
    else:
        (artifact / "manifest-link.json").symlink_to("manifest.json")
    with pytest.raises(validator.ValidationError, match="artifact"):
        validator.validate_campaign_artifact(
            registry_path, artifact, tmp_path
        )


def test_artifact_directory_symlink_is_rejected(tmp_path):
    validator = load_validator()
    registry_path = make_registry(tmp_path)
    artifact = make_artifact(tmp_path, registry_path)
    artifact_link = tmp_path / "artifact-link"
    artifact_link.symlink_to(artifact, target_is_directory=True)
    with pytest.raises(validator.ValidationError, match="regular directory"):
        validator.validate_campaign_artifact(
            registry_path, artifact_link, tmp_path
        )


def test_boolean_parallelism_provenance_is_rejected(tmp_path):
    validator = load_validator()
    registry_path = make_registry(tmp_path)
    artifact = make_artifact(tmp_path, registry_path)
    mutate_document(
        artifact,
        "provenance.json",
        lambda document: document.update({"mpi_ranks": True}),
    )
    with pytest.raises(validator.ValidationError, match="must be positive"):
        validator.validate_campaign_artifact(
            registry_path, artifact, tmp_path
        )


def test_source_tree_mismatch_is_rejected_before_promotion(tmp_path):
    validator = load_validator()
    registry_path = make_registry(tmp_path)
    artifact = make_artifact(tmp_path, registry_path)
    mutate_document(
        artifact,
        "metrics.json",
        lambda document: document.update({"source_tree": "4" * 40}),
    )
    with pytest.raises(validator.ValidationError, match="source tree is incoherent"):
        validator.validate_campaign_artifact(
            registry_path, artifact, tmp_path
        )


def test_child_registry_hash_mismatch_is_rejected(tmp_path):
    validator = load_validator()
    registry_path = make_registry(tmp_path)
    artifact = make_artifact(tmp_path, registry_path)

    def change_hash(document):
        document["child_programs"][0]["registry_sha256"] = "4" * 64

    mutate_document(artifact, "manifest.json", change_hash)
    with pytest.raises(validator.ValidationError, match="registered hash"):
        validator.validate_campaign_artifact(
            registry_path, artifact, tmp_path
        )


def test_registry_rejects_campaign_dependency_reordering(tmp_path):
    validator = load_validator()
    registry_path = make_registry(tmp_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["campaigns"][3]["depends_on"] = ["Q1", "Q0", "Q2"]
    write_json(registry_path, registry)
    with pytest.raises(validator.ValidationError, match="earlier campaign"):
        validator.load_campaign_registry(registry_path, tmp_path)


def test_manifest_cannot_request_a_claim_outside_registered_scope(tmp_path):
    validator = load_validator()
    registry_path = make_registry(tmp_path, ready=True)
    artifact = make_artifact(
        tmp_path, registry_path, promotion_requested=True
    )

    def widen_claims(document):
        document["claims"] = ["Q0", "Q1"]

    mutate_document(artifact, "manifest.json", widen_claims)
    with pytest.raises(validator.ValidationError, match="claims exceed"):
        validator.validate_campaign_artifact(
            registry_path, artifact, tmp_path
        )


def test_registry_rejects_incomplete_reference_metadata(tmp_path):
    validator = load_validator()
    registry_path = make_registry(tmp_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    del registry["campaigns"][2]["benchmarks"][0]["reference"]["uncertainty"]
    write_json(registry_path, registry)
    with pytest.raises(validator.ValidationError, match="reference.*keys differ"):
        validator.load_campaign_registry(registry_path, tmp_path)


def test_artifact_rejects_missing_expected_metric(tmp_path):
    validator = load_validator()
    registry_path = make_registry(tmp_path)
    artifact = make_artifact(tmp_path, registry_path)

    def remove_metric(document):
        document["metrics"].pop(next(iter(document["metrics"])))

    mutate_document(artifact, "metrics.json", remove_metric)
    with pytest.raises(validator.ValidationError, match="complete expected set"):
        validator.validate_campaign_artifact(
            registry_path, artifact, tmp_path
        )


def test_prerequisite_only_evidence_is_refused_for_promotion(tmp_path):
    validator = load_validator()
    registry_path = make_registry(tmp_path)
    artifact = make_artifact(
        tmp_path, registry_path, promotion_requested=True
    )
    with pytest.raises(
        validator.ValidationError,
        match="prerequisite-only evidence cannot promote",
    ):
        validator.validate_campaign_artifact(
            registry_path, artifact, tmp_path
        )


def test_checksum_mismatch_is_rejected(tmp_path):
    validator = load_validator()
    registry_path = make_registry(tmp_path)
    artifact = make_artifact(tmp_path, registry_path)
    manifest = json.loads(
        (artifact / "manifest.json").read_text(encoding="utf-8")
    )
    write_json(artifact / "manifest.json", manifest)
    with (artifact / "manifest.json").open("a", encoding="utf-8") as output:
        output.write(" ")
    with pytest.raises(validator.ValidationError, match="checksum mismatch"):
        validator.validate_campaign_artifact(
            registry_path, artifact, tmp_path
        )
