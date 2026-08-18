import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FLUID_CASES = ROOT / "tests" / "cases" / "fluid"
RUNNER_PATH = (
    FLUID_CASES
    / "run_free_surface_wp3_wp7_nitsche_coercivity_qualification_v2.py"
)
MATRIX_PATH = (
    FLUID_CASES
    / "free_surface_wp3_wp7_nitsche_coercivity_qualification_matrix_v2.json"
)


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp3_wp7_nitsche_coercivity_qualification_runner_v2",
        RUNNER_PATH,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def matrix_document():
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def valid_draft_document():
    return matrix_document()


def valid_frozen_document(commit_length=40):
    matrix = valid_draft_document()
    matrix["status"] = "FROZEN_BEFORE_EXECUTION"
    matrix["implementation_source_commit"] = "a" * commit_length
    matrix["source_inventory_hash_status"] = "FROZEN"
    matrix["runner_sha256"] = "e" * 64
    matrix["draft_promotion_contract"].update(
        {
            "current_state": "FROZEN_BEFORE_EXECUTION",
            "source_hashes_frozen": True,
            "qualification_evidence_executed": False,
            "validate_only_allowed": True,
            "execution_allowed": True,
            "required_execution_state": "FROZEN_BEFORE_EXECUTION",
        }
    )
    derivation = matrix["matching_derivation"]
    if derivation not in {
        entry["path"] for entry in matrix["implementation_sources"]
    }:
        matrix["implementation_sources"].append(
            {
                "path": derivation,
                "sha256": "b" * 64,
                "role": "frozen matching derivation",
            }
        )
    return matrix


def trace_records(runner, gap=None):
    if gap is None:
        gap = -runner.EXPECTED_CASE_AXES["sample_comparison_tolerance"]
    cases = []
    ordinal = 0
    wet_lower_bound = 1.0 - math.sqrt(4.0 / 12.0)
    eigensolver_tolerance = 1.0e-12
    for orientation in runner.EXPECTED_CASE_AXES["orientations"]:
        for scale in runner.EXPECTED_CASE_AXES["affine_mesh_scales"]:
            for side in runner.EXPECTED_CASE_AXES["active_sides"]:
                for fraction in runner.EXPECTED_CASE_AXES["wall_fractions"]:
                    dry = fraction == 0.0
                    upper_bound = 0.0 if dry else 4.0
                    ratio = upper_bound / 12.0
                    cases.append(
                        {
                            "case_id": f"trace-case-{ordinal:03d}",
                            "orientation": orientation,
                            "active_side": side,
                            "mesh_scale": scale,
                            "target_wall_fraction": fraction,
                            "certificate_digest": ordinal + 1,
                            "aggregation_digest": ordinal + 101,
                            "cut_context_revision": 1,
                            "snapshot_revision": 2,
                            "source_value_revision": 3,
                            "form_binding_digest": ordinal + 201,
                            "source_formulation_record_index": 0,
                            "form_binding_source_match": True,
                            "boundary_rule_count": 0 if dry else 1,
                            "patch_count": 0 if dry else 1,
                            "trace_upper_bound": upper_bound,
                            "effective_penalty_multiplier": 12.0,
                            "trace_to_penalty_ratio": ratio,
                            "grouped_symmetric_ratio": ratio,
                            "finite_sample_energy_lower_bound": (
                                1.0 if dry else wet_lower_bound
                            ),
                            "minimum_generalized_eigenvalue": (
                                None
                                if dry
                                else (
                                    wet_lower_bound
                                    + eigensolver_tolerance
                                    + gap
                                )
                            ),
                            "eigensolver_tolerance": (
                                None if dry else eigensolver_tolerance
                            ),
                            "sampled_eigenvalue_gap": None if dry else gap,
                            "deterministic": True,
                            "revision_match": True,
                        }
                    )
                    ordinal += 1
    summary = {
        "case_count": 108,
        "wet_case_count": 96,
        "dry_case_count": 12,
        "deterministic_case_count": 108,
        "revision_match_case_count": 108,
        "maximum_trace_upper_bound": 4.0,
        "minimum_finite_sample_energy_lower_bound": wet_lower_bound,
        "minimum_sampled_eigenvalue_gap": gap,
        "method_coercivity_lower_bound": None,
        "uniform_bound_status": "UNFROZEN_NO_BOUND_INVENTED",
        "accepted_claim": "joint_low_level_prerequisite",
    }
    return cases, summary


def trace_stdout(runner, cases, summary):
    lines = [
        runner.TRACE_CASE_PREFIX + json.dumps(case, sort_keys=True)
        for case in cases
    ]
    lines.append(
        runner.TRACE_SUMMARY_PREFIX + json.dumps(summary, sort_keys=True)
    )
    return "\n".join(lines)


def test_draft_contract_accepts_structural_validation():
    runner = load_runner()
    matrix = valid_draft_document()

    assert runner.validate_v2_contract(matrix) is matrix


def test_draft_validate_only_is_accepted_and_execution_is_rejected(
    monkeypatch,
    capsys,
):
    runner = load_runner()
    matrix = valid_draft_document()
    monkeypatch.setattr(runner, "load_registry", lambda path: matrix)
    source_observation = {
        "inventory_count": 2,
        "matching_count": 1,
        "drift_count": 1,
        "missing_count": 0,
        "all_match": False,
        "records": [],
    }
    monkeypatch.setattr(
        runner,
        "observe_implementation_sources",
        lambda registry: source_observation,
    )

    assert runner.main(["--validate-only"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "DRAFT_UNEXECUTED"
    assert result["execution_ready"] is False
    assert result["validation_scope"] == (
        "draft_structure_and_dependency_validation"
    )
    assert result["implementation_source_observation"] == source_observation
    assert result["group_count"] == 4
    assert result["test_count"] == 20
    assert result["outcome"] == "DRAFT_SOURCE_DRIFT"

    with pytest.raises(ValueError, match="full execution requires promotion"):
        runner.main([])


def test_draft_source_observation_reports_each_hash(tmp_path):
    runner = load_runner()
    current = tmp_path / "current.cpp"
    changed = tmp_path / "changed.cpp"
    current.write_bytes(b"current\n")
    changed.write_bytes(b"changed\n")
    registry = {
        "implementation_sources": [
            {
                "path": "current.cpp",
                "sha256": hashlib.sha256(b"current\n").hexdigest(),
            },
            {
                "path": "changed.cpp",
                "sha256": hashlib.sha256(b"prior\n").hexdigest(),
            },
            {
                "path": "missing.cpp",
                "sha256": hashlib.sha256(b"missing\n").hexdigest(),
            },
        ]
    }

    observation = runner.observe_implementation_sources(
        registry,
        tmp_path,
    )

    assert observation["inventory_count"] == 3
    assert observation["matching_count"] == 1
    assert observation["drift_count"] == 2
    assert observation["missing_count"] == 1
    assert observation["all_match"] is False
    assert [record["path"] for record in observation["records"]] == [
        "current.cpp",
        "changed.cpp",
        "missing.cpp",
    ]
    assert observation["records"][0]["matches_draft_observation"] is True
    assert observation["records"][1]["observed_sha256"] == hashlib.sha256(
        b"changed\n"
    ).hexdigest()
    assert observation["records"][2]["observed_sha256"] is None


@pytest.mark.parametrize("commit_length", [40, 64])
def test_frozen_contract_accepts_lowercase_commit_and_frozen_sources(
    commit_length,
):
    runner = load_runner()
    matrix = valid_frozen_document(commit_length)

    assert runner.validate_v2_contract(matrix) is matrix


@pytest.mark.parametrize(
    "commit",
    [
        None,
        "a" * 39,
        "a" * 41,
        "A" * 40,
        "g" * 40,
    ],
)
def test_frozen_contract_rejects_invalid_implementation_commit(commit):
    runner = load_runner()
    matrix = valid_frozen_document()
    matrix["implementation_source_commit"] = commit

    with pytest.raises(ValueError, match="lowercase hexadecimal digest"):
        runner.validate_v2_contract(matrix)


def test_frozen_contract_rejects_unfrozen_source_inventory():
    runner = load_runner()
    matrix = valid_frozen_document()
    matrix["source_inventory_hash_status"] = "DRAFT_OBSERVED_NOT_FROZEN"

    with pytest.raises(ValueError, match="intentionally frozen hashes"):
        runner.validate_v2_contract(matrix)

    matrix = valid_frozen_document()
    matrix["runner_sha256"] = runner.RUNNER_SHA256_ZERO_SENTINEL
    with pytest.raises(ValueError, match="must lock the exact runner bytes"):
        runner.validate_v2_contract(matrix)

    matrix = valid_draft_document()
    matrix["runner_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="must remain the zero sentinel"):
        runner.validate_v2_contract(matrix)


def test_frozen_dependencies_bind_inventory_to_an_ancestor_commit(
    monkeypatch,
    tmp_path,
):
    runner = load_runner()
    matrix = {
        "status": "FROZEN_BEFORE_EXECUTION",
        "parent_artifacts": [],
        "implementation_sources": [],
        "implementation_source_commit": "a" * 40,
    }

    def valid_git_bytes(root, *arguments):
        if arguments[:2] == ("rev-parse", "--verify"):
            return b"a" * 40 + b"\n"
        if arguments[:2] == ("merge-base", "--is-ancestor"):
            return b""
        raise AssertionError(arguments)

    monkeypatch.setattr(
        runner.strict_runner,
        "git_bytes",
        valid_git_bytes,
    )
    monkeypatch.setattr(
        runner,
        "sha256_file",
        lambda path: runner.EXPECTED_SHARED_RUNNER_SHA256,
    )

    runner.validate_frozen_dependencies(matrix, tmp_path)

    def nonancestor_git_bytes(root, *arguments):
        if arguments[:2] == ("rev-parse", "--verify"):
            return b"a" * 40 + b"\n"
        if arguments[:2] == ("merge-base", "--is-ancestor"):
            raise subprocess.CalledProcessError(1, ["git", *arguments])
        raise AssertionError(arguments)

    monkeypatch.setattr(
        runner.strict_runner,
        "git_bytes",
        nonancestor_git_bytes,
    )
    with pytest.raises(ValueError, match="is not an ancestor of HEAD"):
        runner.validate_frozen_dependencies(matrix, tmp_path)

    runner_bytes = b"#!/usr/bin/env python3\n"
    runner_digest = hashlib.sha256(runner_bytes).hexdigest()
    matrix_bytes = (
        '{\n  "runner_sha256": "' + runner_digest + '"\n}\n'
    ).encode()
    matrix["runner_sha256"] = runner_digest
    matrix_path = tmp_path / runner.EXPECTED_MATRIX_PATH
    runner_path = tmp_path / runner.EXPECTED_PROPOSED_RUNNER
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_bytes(matrix_bytes)
    runner_path.write_bytes(runner_bytes)
    bundle_commit = "c" * 40
    committed = {
        runner.EXPECTED_MATRIX_PATH: matrix_bytes,
        runner.EXPECTED_PROPOSED_RUNNER: runner_bytes,
    }

    def bundle_git_bytes(root, *arguments):
        if arguments == (
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
        ):
            return bundle_commit.encode() + b"\n"
        if arguments[0] == "show":
            specification = arguments[1]
            commit, relative_path = specification.split(":", 1)
            assert commit == bundle_commit
            return committed[relative_path]
        raise AssertionError(arguments)

    monkeypatch.setattr(
        runner.strict_runner,
        "git_bytes",
        bundle_git_bytes,
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_NORMALIZED_REGISTRY_SHA256",
        hashlib.sha256(
            runner.normalized_registry_bytes(matrix_bytes)
        ).hexdigest(),
    )
    alternate_matrix_bytes = matrix_bytes.replace(
        runner_digest.encode(),
        b"f" * 64,
    )
    assert runner.normalized_registry_bytes(alternate_matrix_bytes) == (
        runner.normalized_registry_bytes(matrix_bytes)
    )
    multiple_runner_fields = (
        json.dumps(
            {
                "runner_sha256": runner_digest,
                "nested": {"runner_sha256": runner_digest},
            },
            indent=2,
        )
        + "\n"
    ).encode()
    with pytest.raises(ValueError, match="exactly one raw runner_sha256"):
        runner.normalized_registry_bytes(multiple_runner_fields)
    with pytest.raises(ValueError, match="exactly one raw runner_sha256"):
        runner.normalized_registry_bytes(
            matrix_bytes.replace(
                runner_digest.encode(),
                runner_digest.upper().encode(),
            )
        )
    binding = runner.validate_frozen_qualification_bundle(
        matrix,
        matrix_path,
        tmp_path,
        runner_path,
    )
    assert binding["qualification_bundle_commit"] == bundle_commit
    assert binding["implementation_source_commit"] == "a" * 40
    assert binding["artifacts"] == [
        {
            "role": "matrix",
            "path": runner.EXPECTED_MATRIX_PATH,
            "sha256": hashlib.sha256(matrix_bytes).hexdigest(),
            "normalized_sha256": (
                runner.EXPECTED_NORMALIZED_REGISTRY_SHA256
            ),
        },
        {
            "role": "runner",
            "path": runner.EXPECTED_PROPOSED_RUNNER,
            "sha256": hashlib.sha256(runner_bytes).hexdigest(),
        },
    ]
    monkeypatch.setattr(
        runner,
        "_frozen_qualification_bundle_binding",
        binding,
    )
    provenance = {}
    runner._inject_claim_boundary(provenance)
    assert provenance["qualification_bundle_binding"] == binding

    matrix["runner_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="does not match the frozen registry"):
        runner.validate_frozen_qualification_bundle(
            matrix,
            matrix_path,
            tmp_path,
            runner_path,
        )
    matrix["runner_sha256"] = runner_digest
    changed_runner_bytes = b"changed runner bytes\n"
    runner_path.write_bytes(changed_runner_bytes)
    committed[runner.EXPECTED_PROPOSED_RUNNER] = changed_runner_bytes
    with pytest.raises(ValueError, match="does not match the matrix"):
        runner.validate_frozen_qualification_bundle(
            matrix,
            matrix_path,
            tmp_path,
            runner_path,
        )

    committed[runner.EXPECTED_PROPOSED_RUNNER] = runner_bytes
    with pytest.raises(ValueError, match="runner differs from its HEAD blob"):
        runner.validate_frozen_qualification_bundle(
            matrix,
            matrix_path,
            tmp_path,
            runner_path,
        )


def test_frozen_dependencies_bind_each_inventory_blob_to_the_commit(
    monkeypatch,
    tmp_path,
):
    runner = load_runner()
    payload = b"frozen source bytes\n"
    source_path = tmp_path / "source.cpp"
    source_path.write_bytes(payload)
    matrix = {
        "status": "FROZEN_BEFORE_EXECUTION",
        "parent_artifacts": [],
        "implementation_sources": [
            {
                "path": "source.cpp",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
        "implementation_source_commit": "a" * 40,
    }

    def git_bytes(root, *arguments):
        if arguments[:2] == ("rev-parse", "--verify"):
            return b"a" * 40 + b"\n"
        if arguments[:2] == ("merge-base", "--is-ancestor"):
            return b""
        if arguments[0] == "show":
            return payload
        raise AssertionError(arguments)

    monkeypatch.setattr(runner.strict_runner, "git_bytes", git_bytes)
    original_sha256_file = runner.sha256_file
    monkeypatch.setattr(
        runner,
        "sha256_file",
        lambda path: (
            runner.EXPECTED_SHARED_RUNNER_SHA256
            if path == runner.SHARED_RUNNER_PATH
            else original_sha256_file(path)
        ),
    )
    runner.validate_frozen_dependencies(matrix, tmp_path)

    def changed_git_bytes(root, *arguments):
        if arguments[0] == "show":
            return b"changed source bytes\n"
        return git_bytes(root, *arguments)

    monkeypatch.setattr(
        runner.strict_runner,
        "git_bytes",
        changed_git_bytes,
    )
    with pytest.raises(
        ValueError,
        match="differs from its frozen commit: source.cpp",
    ):
        runner.validate_frozen_dependencies(matrix, tmp_path)


def test_status_contract_rejects_unknown_state_and_inconsistent_promotion():
    runner = load_runner()
    matrix = valid_draft_document()
    matrix["status"] = "EXECUTED"
    with pytest.raises(ValueError, match="must be DRAFT_UNEXECUTED or"):
        runner.validate_v2_contract(matrix)

    matrix = valid_frozen_document()
    matrix["draft_promotion_contract"]["execution_allowed"] = False
    with pytest.raises(ValueError, match="inconsistent.*execution_allowed"):
        runner.validate_v2_contract(matrix)

    matrix = valid_draft_document()
    matrix["qualification_bundle_binding"]["runner_sha256_source"] = (
        "matrix"
    )
    with pytest.raises(ValueError, match="bundle binding contract changed"):
        runner.validate_v2_contract(matrix)

    matrix = valid_draft_document()
    matrix["implementation_sources"].append(
        {
            "path": runner.EXPECTED_PROPOSED_RUNNER,
            "sha256": "d" * 64,
            "role": "invalid circular runner lock",
        }
    )
    with pytest.raises(
        ValueError,
        match="must use the reciprocal qualification-bundle binding",
    ):
        runner.validate_v2_contract(matrix)


def test_frozen_execution_uses_only_the_four_declared_binaries(
    monkeypatch,
):
    runner = load_runner()
    matrix = valid_frozen_document()
    observed = {}
    monkeypatch.setattr(runner, "load_registry", lambda path: matrix)
    monkeypatch.setattr(
        runner,
        "require_execution_resource_preflight",
        lambda source_root, output_directory, build_directories=(): None,
    )

    def delegated_run(arguments, binaries, **options):
        observed["status"] = runner.strict_runner.EXPECTED_MATRIX_STATUS
        observed["arguments"] = arguments
        observed["binaries"] = binaries
        observed["options"] = options
        return 17

    monkeypatch.setattr(
        runner.strict_runner,
        "run_qualification",
        delegated_run,
    )
    original_status = runner.strict_runner.EXPECTED_MATRIX_STATUS

    assert runner.main(
        [
            "--math-binary",
            "math",
            "--assembly-binary",
            "assembly",
            "--assembly-mpi-binary",
            "assembly-mpi",
            "--physics-binary",
            "physics",
            "--output",
            "unused",
        ]
    ) == 17
    assert observed["status"] == "FROZEN_BEFORE_EXECUTION"
    assert observed["binaries"] == {
        "math": Path("math"),
        "assembly": Path("assembly"),
        "assembly_mpi": Path("assembly-mpi"),
        "physics": Path("physics"),
    }
    assert observed["options"]["expected_binary_keys"] == {
        "math",
        "assembly",
        "assembly_mpi",
        "physics",
    }
    assert observed["arguments"].build_parallel == 1
    assert runner.strict_runner.EXPECTED_MATRIX_STATUS == original_status


def test_exact_dyadic_math_group_and_source_contract_are_exact():
    runner = load_runner()
    matrix = valid_draft_document()

    runner.validate_v2_contract(matrix)

    assert matrix["status"] == "DRAFT_UNEXECUTED"
    assert len(matrix["groups"]) == 4
    assert sum(len(group["tests"]) for group in matrix["groups"]) == 20
    math_group = matrix["groups"][0]
    assert math_group == {
        "id": runner.EXACT_DYADIC_GROUP_ID,
        "binary": "math",
        "mpi_ranks": 1,
        "gtest_output_copies": 1,
        "tests": list(runner.EXACT_DYADIC_TESTS),
        "execution": {
            "wall_time_seconds": 300,
            "memory_mib": 1024,
            "output_mib": 64,
        },
    }
    assert matrix["build_targets"]["math"] == "test_fe_math"
    assert matrix["build_cmake_homes"]["math"] == (
        "Code/Source/solver/FE"
    )
    trace_contract = matrix["certified_aggregate_trace_contract"]
    assert trace_contract["maximum_exact_retained_quotient_dimension"] == 32
    assert trace_contract["quotient_authority"] == (
        "exact_binary64_dyadic_D_spd_N_psd_and_qD_minus_N_psd"
    )
    assert trace_contract["floating_spectral_role"] == (
        "optional_diagnostics_only"
    )
    assert matrix["certificate_envelope"][
        "hard_exact_retained_quotient_dimension_cap"
    ] == 32
    source_roles = {
        entry["path"]: entry["role"]
        for entry in matrix["implementation_sources"]
    }
    assert {
        path: source_roles[path]
        for path in runner.EXPECTED_EXACT_DYADIC_SOURCE_ROLES
    } == runner.EXPECTED_EXACT_DYADIC_SOURCE_ROLES


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("maximum_exact_retained_quotient_dimension", 31),
        ("quotient_authority", "floating_generalized_eigensolver"),
        ("floating_spectral_role", "qualification_authority"),
    ],
)
def test_exact_dyadic_trace_contract_drift_is_rejected(field, replacement):
    runner = load_runner()
    matrix = valid_draft_document()
    matrix["certified_aggregate_trace_contract"][field] = replacement

    with pytest.raises(
        ValueError,
        match="certified aggregate trace contract changed",
    ):
        runner.validate_v2_contract(matrix)


def test_exact_dyadic_cap_source_group_and_math_target_drift_are_rejected():
    runner = load_runner()

    changed = valid_draft_document()
    changed["certificate_envelope"][
        "hard_exact_retained_quotient_dimension_cap"
    ] = 33
    with pytest.raises(ValueError, match="retained quotient cap changed"):
        runner.validate_v2_contract(changed)

    changed = valid_draft_document()
    changed["implementation_sources"] = [
        entry
        for entry in changed["implementation_sources"]
        if entry["path"]
        != "Code/Source/solver/FE/Math/DenseExactDyadic.cpp"
    ]
    with pytest.raises(
        ValueError,
        match="exact-dyadic implementation source inventory changed",
    ):
        runner.validate_v2_contract(changed)

    changed = valid_draft_document()
    changed["groups"][0]["tests"].pop()
    with pytest.raises(ValueError, match="v2 qualification group changed"):
        runner.validate_v2_contract(changed)

    changed = valid_draft_document()
    changed["build_targets"]["math"] = "test_fe_assembly"
    with pytest.raises(
        ValueError,
        match="build target/CMake-home inventory changed",
    ):
        runner.validate_v2_contract(changed)


def test_group_execution_envelopes_are_exact():
    runner = load_runner()
    matrix = valid_draft_document()
    runner.validate_v2_contract(matrix)

    actual = {
        group["id"]: (
            group["execution"]["wall_time_seconds"],
            group["execution"]["memory_mib"],
            group["execution"]["output_mib"],
        )
        for group in matrix["groups"]
    }
    assert actual == runner.EXPECTED_GROUP_EXECUTION

    matrix["groups"][1]["execution"]["memory_mib"] = 1025
    with pytest.raises(ValueError, match="group execution envelope changed"):
        runner.validate_v2_contract(matrix)


def test_resource_safeguards_are_exact_and_preflight_is_fail_closed(
    monkeypatch,
    tmp_path,
):
    runner = load_runner()
    matrix = valid_draft_document()
    runner.validate_v2_contract(matrix)

    changed = copy.deepcopy(matrix)
    changed["resource_safeguards"]["build_parallel"] = 2
    with pytest.raises(
        ValueError,
        match="resource safeguards changed",
    ):
        runner.validate_v2_contract(changed)

    monkeypatch.setattr(
        runner.strict_runner,
        "host_available_memory_mib",
        lambda: 10239,
    )
    with pytest.raises(
        ValueError,
        match="requires at least 10240 MiB MemAvailable",
    ):
        runner.require_execution_resource_preflight(
            tmp_path,
            tmp_path / "result",
        )

    monkeypatch.setattr(
        runner.strict_runner,
        "host_available_memory_mib",
        lambda: 10240,
    )
    monkeypatch.setattr(
        runner.strict_runner,
        "filesystem_free_mib",
        lambda path: 4095,
    )
    with pytest.raises(
        ValueError,
        match="requires at least 4096 MiB free",
    ):
        runner.require_execution_resource_preflight(
            tmp_path,
            tmp_path / "result",
        )


def test_postprocess_text_reads_are_strictly_bounded(tmp_path):
    runner = load_runner()
    path = tmp_path / "bounded.txt"
    path.write_bytes(b"abcd")

    assert runner.strict_runner.read_text_with_limit(path, 4) == "abcd"

    path.write_bytes(b"abcde")
    with pytest.raises(ValueError, match="exceeds 4-byte parse limit"):
        runner.strict_runner.read_text_with_limit(path, 4)


def test_binary_link_provenance_uses_monitored_bounded_execution(
    monkeypatch,
    tmp_path,
):
    runner = load_runner()
    binary = tmp_path / "test_binary"
    binary.write_bytes(b"binary")
    output_root = tmp_path / "output"
    output_root.mkdir()
    observed = {}

    def monitored(*arguments, **options):
        observed["arguments"] = arguments
        observed["options"] = options
        Path(arguments[3]).write_text(
            "libexample.so => /lib/libexample.so\n",
            encoding="utf-8",
        )
        Path(arguments[4]).write_bytes(b"")
        return {
            "return_code": 0,
            "termination_reason": None,
            "resource_monitoring_outcome": "PASS",
        }

    monkeypatch.setattr(
        runner.strict_runner,
        "run_monitored",
        monitored,
    )
    record = runner.strict_runner.binary_record(
        binary,
        tmp_path,
        output_root,
        "assembly",
    )

    arguments = observed["arguments"]
    assert arguments[0] == ["ldd", str(binary)]
    assert arguments[6] == 60
    assert arguments[7] == 256
    assert arguments[8] == 4
    assert arguments[9] == "direct_serial"
    assert observed["options"] == {}
    assert record["outcome"] == "PASS"
    assert record["linked_libraries"] == [
        "libexample.so => /lib/libexample.so"
    ]


def test_binary_link_provenance_fails_closed_on_process_failure(
    monkeypatch,
    tmp_path,
):
    runner = load_runner()
    binary = tmp_path / "test_binary"
    binary.write_bytes(b"binary")
    output_root = tmp_path / "output"
    output_root.mkdir()

    def monitored(*arguments, **_options):
        Path(arguments[3]).write_bytes(b"")
        Path(arguments[4]).write_text(
            "dependency inspection failed\n",
            encoding="utf-8",
        )
        return {
            "return_code": 1,
            "termination_reason": None,
            "resource_monitoring_outcome": "PASS",
        }

    monkeypatch.setattr(
        runner.strict_runner,
        "run_monitored",
        monitored,
    )
    record = runner.strict_runner.binary_record(
        binary,
        tmp_path,
        output_root,
        "assembly",
    )

    assert record["outcome"] == "FAIL_METHOD"
    assert record["linked_libraries"] == []
    assert (
        record["diagnostic"]
        == "linked_library_provenance_process_failed"
    )


def test_monitored_wrapper_pins_threads_and_runtime_floors(
    monkeypatch,
    tmp_path,
):
    runner = load_runner()
    observed = {}

    def delegated(*arguments, **options):
        observed["arguments"] = arguments
        observed["options"] = options
        return {"outcome": "sentinel"}

    monkeypatch.setattr(runner, "_shared_run_monitored", delegated)
    result = runner.run_monitored(
        ["binary"],
        {"OMP_NUM_THREADS": "9"},
        tmp_path,
        tmp_path / "stdout.txt",
        tmp_path / "stderr.txt",
        tmp_path,
        10,
        64,
        8,
        "direct_serial",
    )

    assert result == {"outcome": "sentinel"}
    environment = observed["arguments"][1]
    assert {
        key: environment[key]
        for key in runner.EXPECTED_RESOURCE_SAFEGUARDS[
            "thread_environment"
        ]
    } == runner.EXPECTED_RESOURCE_SAFEGUARDS[
        "thread_environment"
    ]
    assert observed["options"] == {
        "minimum_host_available_mib": 4096,
        "minimum_filesystem_free_mib": 4096,
        "filesystem_path": tmp_path,
    }


def test_build_phase_cannot_pass_a_resource_monitoring_failure(
    monkeypatch,
    tmp_path,
):
    runner = load_runner()
    build_directory = tmp_path / "build"
    build_directory.mkdir()
    output_root = tmp_path / "output"
    output_root.mkdir()
    observed = {}

    def monitored(*arguments, **options):
        Path(arguments[3]).write_bytes(b"")
        Path(arguments[4]).write_bytes(b"")
        observed["options"] = options
        return {
            "return_code": 0,
            "termination_reason": "filesystem_free_space_floor_breached",
            "termination": None,
            "wall_time_seconds": 0.01,
            "resource_monitoring_outcome": "FAIL_METHOD",
        }

    monkeypatch.setattr(runner, "_shared_run_monitored", monitored)
    result = runner.run_build_phase(
        [
            "cmake",
            "--build",
            str(build_directory),
            "--target",
            "target",
        ],
        tmp_path,
        output_root,
        output_root / "stdout.txt",
        output_root / "stderr.txt",
        10,
    )

    assert result["process_return_code"] == 0
    assert result["return_code"] is None
    assert result["monitored_build_directory"] == str(
        build_directory.resolve()
    )
    assert observed["options"]["filesystem_path"] == (
        build_directory.resolve()
    )
    assert observed["options"]["additional_filesystem_paths"] == (
        output_root,
    )


def test_test_discovery_uses_the_same_monitored_resource_contract(
    monkeypatch,
    tmp_path,
):
    runner = load_runner()
    output_root = tmp_path / "output"
    output_root.mkdir()
    binary = tmp_path / "test_binary"
    observed = {}

    def monitored(*arguments, **options):
        Path(arguments[3]).write_text(
            "Suite.\n  Case\n",
            encoding="utf-8",
        )
        Path(arguments[4]).write_bytes(b"")
        observed["options"] = options
        return {
            "return_code": 0,
            "termination_reason": None,
            "resource_monitoring_outcome": "PASS",
        }

    monkeypatch.setattr(runner, "_shared_run_monitored", monitored)
    discover = runner.monitored_test_discovery(
        tmp_path,
        output_root,
    )

    assert discover(binary) == {"Suite.Case"}
    assert observed["options"]["minimum_host_available_mib"] == 4096
    assert observed["options"]["minimum_filesystem_free_mib"] == 4096
    assert observed["options"]["filesystem_path"] == binary.parent
    discovery_path = observed["options"][
        "additional_filesystem_paths"
    ][0]
    assert discovery_path.parent == output_root / "test_discovery"


def test_runtime_gates_are_an_exact_one_to_one_test_map():
    runner = load_runner()
    matrix = valid_draft_document()
    runner.validate_v2_contract(matrix)

    tests = [gate["test"] for gate in matrix["runtime_gates"]]
    assert len(tests) == 20
    assert len(set(tests)) == 20

    duplicate = copy.deepcopy(matrix)
    duplicate["runtime_gates"][1]["test"] = (
        duplicate["runtime_gates"][0]["test"]
    )
    with pytest.raises(ValueError, match="duplicate runtime gate test"):
        runner.validate_v2_contract(duplicate)

    unknown = copy.deepcopy(matrix)
    unknown["runtime_gates"][0]["test"] = "UnknownSuite.UnknownTest"
    with pytest.raises(ValueError, match="runtime gate test map changed"):
        runner.validate_v2_contract(unknown)

    missing = copy.deepcopy(matrix)
    missing["runtime_gates"].pop()
    with pytest.raises(ValueError, match="exactly one entry"):
        runner.validate_v2_contract(missing)


def test_structured_output_field_lists_must_be_exact_and_unique():
    runner = load_runner()
    matrix = valid_draft_document()
    runner.validate_v2_contract(matrix)

    duplicate = copy.deepcopy(matrix)
    duplicate["structured_output_contract"]["case_required_fields"].append(
        "target_wall_fraction"
    )
    with pytest.raises(
        ValueError,
        match=(
            "case_required_fields contains duplicate field: "
            "target_wall_fraction"
        ),
    ):
        runner.validate_v2_contract(duplicate)

    missing = copy.deepcopy(matrix)
    missing["structured_output_contract"]["summary_required_fields"].remove(
        "accepted_claim"
    )
    with pytest.raises(ValueError, match="summary_required_fields changed"):
        runner.validate_v2_contract(missing)

    extra = copy.deepcopy(matrix)
    extra["structured_output_contract"]["case_required_fields"].append(
        "unregistered_field"
    )
    with pytest.raises(ValueError, match="case_required_fields changed"):
        runner.validate_v2_contract(extra)


def test_trace_parser_accepts_gap_at_negative_comparison_tolerance():
    runner = load_runner()
    cases, summary = trace_records(runner)

    evidence = runner.parse_trace_evidence(
        trace_stdout(runner, cases, summary)
    )

    assert evidence["outcome"] == "PASS"
    assert evidence["observed_case_count"] == 108
    assert evidence["wet_case_count"] == 96
    assert evidence["dry_case_count"] == 12
    assert evidence["minimum_sampled_eigenvalue_gap"] == -1.0e-11


def test_trace_parser_rejects_gap_below_comparison_tolerance():
    runner = load_runner()
    cases, summary = trace_records(runner, gap=-1.0001e-11)

    with pytest.raises(ValueError, match="below the comparison tolerance"):
        runner.parse_trace_evidence(trace_stdout(runner, cases, summary))


def test_trace_parser_rejects_inconsistent_gap_and_negative_tolerance():
    runner = load_runner()
    cases, summary = trace_records(runner)
    cases[1]["sampled_eigenvalue_gap"] = 0.0
    with pytest.raises(ValueError, match="gap is inconsistent"):
        runner.parse_trace_evidence(trace_stdout(runner, cases, summary))

    cases, summary = trace_records(runner)
    cases[1]["eigensolver_tolerance"] = -1.0e-12
    with pytest.raises(ValueError, match="tolerance must be nonnegative"):
        runner.parse_trace_evidence(trace_stdout(runner, cases, summary))


def test_trace_parser_requires_unique_case_ids_and_exact_axes():
    runner = load_runner()
    cases, summary = trace_records(runner)
    cases[1]["case_id"] = cases[0]["case_id"]
    with pytest.raises(ValueError, match="duplicate trace case_id"):
        runner.parse_trace_evidence(trace_stdout(runner, cases, summary))

    cases, summary = trace_records(runner)
    cases[1]["target_wall_fraction"] = cases[0]["target_wall_fraction"]
    with pytest.raises(ValueError, match="duplicate trace case axes"):
        runner.parse_trace_evidence(trace_stdout(runner, cases, summary))


def test_trace_parser_requires_exact_case_and_summary_fields():
    runner = load_runner()
    cases, summary = trace_records(runner)
    del cases[0]["revision_match"]
    with pytest.raises(ValueError, match="trace case 0 fields changed"):
        runner.parse_trace_evidence(trace_stdout(runner, cases, summary))

    cases, summary = trace_records(runner)
    cases[0]["unregistered_field"] = 1
    with pytest.raises(ValueError, match="trace case 0 fields changed"):
        runner.parse_trace_evidence(trace_stdout(runner, cases, summary))

    cases, summary = trace_records(runner)
    summary["unregistered_field"] = 1
    with pytest.raises(ValueError, match="trace summary fields changed"):
        runner.parse_trace_evidence(trace_stdout(runner, cases, summary))


def test_trace_parser_requires_form_binding_provenance():
    runner = load_runner()
    cases, summary = trace_records(runner)
    cases[0]["form_binding_digest"] = 0
    with pytest.raises(ValueError, match="form-binding digest must be positive"):
        runner.parse_trace_evidence(trace_stdout(runner, cases, summary))

    cases, summary = trace_records(runner)
    cases[0]["source_formulation_record_index"] = -1
    with pytest.raises(
        ValueError,
        match="source formulation record index must be nonnegative",
    ):
        runner.parse_trace_evidence(trace_stdout(runner, cases, summary))

    cases, summary = trace_records(runner)
    cases[0]["form_binding_source_match"] = False
    with pytest.raises(
        ValueError,
        match="form binding does not match its source formulation",
    ):
        runner.parse_trace_evidence(trace_stdout(runner, cases, summary))


def test_trace_json_records_reject_duplicate_object_keys():
    runner = load_runner()
    stdout = (
        runner.TRACE_CASE_PREFIX
        + '{"case_id":"first","case_id":"second"}'
    )

    with pytest.raises(ValueError, match="duplicate JSON key: case_id"):
        runner._json_records(stdout, runner.TRACE_CASE_PREFIX)
