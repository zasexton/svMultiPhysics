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
    / "run_free_surface_wp3_wp7_nitsche_coercivity_qualification_v3.py"
)
MATRIX_PATH = (
    FLUID_CASES
    / "free_surface_wp3_wp7_nitsche_coercivity_qualification_matrix_v3.json"
)
V2_RUNNER_PATH = (
    FLUID_CASES
    / "run_free_surface_wp3_wp7_nitsche_coercivity_qualification_v2.py"
)
V2_TEST_PATH = (
    ROOT
    / "tests"
    / "test_free_surface_wp3_wp7_nitsche_coercivity_qualification_runner_v2.py"
)


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp3_wp7_nitsche_coercivity_qualification_runner_v3",
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


def git_bytes(repository, *arguments, input_bytes=None):
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout


def initialize_repository(repository):
    repository.mkdir(parents=True)
    git_bytes(repository, "init", "--quiet")
    git_bytes(repository, "config", "user.name", "Zachary Sexton")
    git_bytes(
        repository,
        "config",
        "user.email",
        "zsexton@stanford.edu",
    )


def write_repository_file(repository, relative_path, contents):
    path = repository / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def commit_paths(repository, message, paths):
    git_bytes(repository, "add", "--", *paths)
    git_bytes(repository, "commit", "--quiet", "-m", message)
    return git_bytes(repository, "rev-parse", "HEAD").decode().strip()


def prepare_bundle_repository(
    runner,
    tmp_path,
    monkeypatch,
    *,
    commit_bundle=True,
):
    repository = tmp_path / "repository"
    initialize_repository(repository)
    write_repository_file(repository, "baseline.txt", b"source baseline\n")
    source_commit = commit_paths(
        repository,
        "source baseline",
        ["baseline.txt"],
    )

    runner_bytes = b"frozen runner bytes\n"
    focused_test_bytes = b"frozen focused test bytes\n"
    runner_digest = hashlib.sha256(runner_bytes).hexdigest()
    matrix_bytes = (
        json.dumps({"runner_sha256": runner_digest}, sort_keys=True, indent=2)
        + "\n"
    ).encode()
    bundle_bytes = {
        runner.EXPECTED_MATRIX_PATH: matrix_bytes,
        runner.EXPECTED_PROPOSED_RUNNER: runner_bytes,
        runner.EXPECTED_FOCUSED_TEST_PATH: focused_test_bytes,
    }
    for relative_path, contents in bundle_bytes.items():
        write_repository_file(repository, relative_path, contents)
    monkeypatch.setattr(
        runner,
        "EXPECTED_NORMALIZED_REGISTRY_SHA256",
        hashlib.sha256(
            runner.normalized_registry_bytes(matrix_bytes)
        ).hexdigest(),
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_FOCUSED_TEST_SHA256",
        hashlib.sha256(focused_test_bytes).hexdigest(),
    )
    registry = {
        "status": runner.EXECUTABLE_MATRIX_STATUS,
        "implementation_source_commit": source_commit,
        "runner_sha256": runner_digest,
        "qualification_bundle_binding": copy.deepcopy(
            runner.EXPECTED_FROZEN_QUALIFICATION_BUNDLE_BINDING
        ),
    }
    bundle_commit = None
    if commit_bundle:
        bundle_commit = commit_paths(
            repository,
            "frozen bundle",
            list(bundle_bytes),
        )
    return {
        "repository": repository,
        "source_commit": source_commit,
        "bundle_commit": bundle_commit,
        "bundle_bytes": bundle_bytes,
        "registry": registry,
    }


def trace_records(runner, gap=None):
    if gap is None:
        gap = -runner.EXPECTED_CASE_AXES["sample_comparison_tolerance"]
    cases = []
    ordinal = 0
    eigensolver_tolerance = 1.0e-12
    for orientation in runner.EXPECTED_CASE_AXES["orientations"]:
        for scale in runner.EXPECTED_CASE_AXES["affine_mesh_scales"]:
            for side in runner.EXPECTED_CASE_AXES["active_sides"]:
                for fraction in runner.EXPECTED_CASE_AXES["wall_fractions"]:
                    dry = fraction == 0.0
                    upper_bound = 0.0 if dry else 4.0
                    ratio = upper_bound / runner.TRACE_PENALTY_GAMMA
                    quotient_patch_count = 0 if dry else 1
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
                            "exact_common_kernel_metadata_valid": True,
                            "exact_common_kernel_quotient_patch_count": (
                                quotient_patch_count
                            ),
                            "trace_upper_bound": upper_bound,
                            "effective_penalty_multiplier": (
                                runner.TRACE_PENALTY_GAMMA
                            ),
                            "required_minimum_energy_ratio": (
                                runner.METHOD_ENERGY_FLOOR
                            ),
                            "trace_to_penalty_ratio": ratio,
                            "grouped_symmetric_ratio": ratio,
                            "finite_sample_energy_lower_bound": (
                                runner.METHOD_ENERGY_FLOOR
                            ),
                            "minimum_generalized_eigenvalue": (
                                None
                                if dry
                                else (
                                    runner.METHOD_ENERGY_FLOOR
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
        "exact_common_kernel_metadata_valid_case_count": 108,
        "exact_common_kernel_quotient_patch_count": 96,
        "maximum_trace_upper_bound": 4.0,
        "minimum_finite_sample_energy_lower_bound": (
            runner.METHOD_ENERGY_FLOOR
        ),
        "minimum_sampled_eigenvalue_gap": gap,
        "method_coercivity_lower_bound": runner.METHOD_ENERGY_FLOOR,
        "uniform_bound_status": runner.UNIFORM_BOUND_STATUS,
        "accepted_claim": runner.ACCEPTED_CLAIM,
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


def set_wet_case_ratio(runner, cases, summary, ratio):
    case = next(
        item for item in cases if item["target_wall_fraction"] != 0.0
    )
    case["trace_upper_bound"] = runner.TRACE_PENALTY_GAMMA * ratio
    case["trace_to_penalty_ratio"] = ratio
    case["grouped_symmetric_ratio"] = ratio
    summary["maximum_trace_upper_bound"] = max(
        item["trace_upper_bound"] for item in cases
    )


def test_checked_in_frozen_bundle_hashes_and_v2_parent_bytes_are_exact():
    runner = load_runner()
    matrix = matrix_document()

    assert runner.normalized_registry_sha256(MATRIX_PATH) == (
        runner.EXPECTED_NORMALIZED_REGISTRY_SHA256
    )
    assert matrix["status"] == runner.EXPECTED_CHECKED_IN_MATRIX_STATUS
    assert runner.EXPECTED_CHECKED_IN_MATRIX_STATUS == (
        "FROZEN_BEFORE_EXECUTION"
    )
    assert matrix["runner_sha256"] == hashlib.sha256(
        RUNNER_PATH.read_bytes()
    ).hexdigest()
    assert matrix["runner_sha256"] != "0" * 64
    assert matrix["qualification_bundle_binding"] == (
        runner.EXPECTED_FROZEN_QUALIFICATION_BUNDLE_BINDING
    )
    assert matrix["qualification_bundle_binding"][
        "exact_bundle_commit_blobs_required"
    ] == [
        runner.EXPECTED_MATRIX_PATH,
        runner.EXPECTED_PROPOSED_RUNNER,
        runner.EXPECTED_FOCUSED_TEST_PATH,
    ]
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == (
        runner.EXPECTED_FOCUSED_TEST_SHA256
    )
    for binding in (
        runner.EXPECTED_DRAFT_QUALIFICATION_BUNDLE_BINDING,
        runner.EXPECTED_FROZEN_QUALIFICATION_BUNDLE_BINDING,
    ):
        assert binding["bundle_commit_must_have_exactly_one_parent"] is True
        assert binding[
            "bundle_commit_parent_must_equal_implementation_source_commit"
        ] is True
        assert binding[
            "bundle_commit_changed_paths_must_equal_exact_bundle_commit_blobs_required"
        ] is True
        assert binding[
            "bundle_commit_blobs_must_match_checked_out_frozen_bytes"
        ] is True
        assert binding["validation_HEAD_must_descend_from_bundle_commit"] is True
        assert binding["execution_HEAD_must_equal_bundle_commit"] is True
    assert len(matrix["implementation_sources"]) == 62
    for safeguard in (
        "source_root_must_equal_runner_repository_root",
        "source_worktree_requires_detached_head",
        "source_worktree_requires_external_git_common_directory",
        "source_worktree_requires_zero_ignored_paths",
        "execution_HEAD_must_equal_bundle_commit",
        "historical_validation_uses_recorded_implementation_source_commit",
        "python_bytecode_writes_disabled",
        "cmake_configure_uses_fresh",
        "cmake_configure_requires_exact_source_and_build_arguments",
        "cmake_configure_rejects_unrecognized_source_homes",
        "cmake_configure_rejects_nonexact_cache_definitions",
    ):
        assert matrix["resource_safeguards"][safeguard] is True
        assert runner.EXPECTED_RESOURCE_SAFEGUARDS[safeguard] is True
    expected_fresh_definitions = {
        source_home: list(definitions)
        for source_home, definitions in (
            runner.EXPECTED_FRESH_CONFIGURE_DEFINITIONS.items()
        )
    }
    assert {
        source_home: len(definitions)
        for source_home, definitions in expected_fresh_definitions.items()
    } == {
        "Code/Source/solver/FE": 19,
        "Code/Source/solver/Physics": 21,
    }
    boost_include_definition = (
        "-DBoost_INCLUDE_DIR="
        "/share/software/user/open/boost/1.90.0/include"
    )
    for definitions in expected_fresh_definitions.values():
        assert definitions.count(boost_include_definition) == 1
    eigen_package_definition = (
        "-DEigen3_DIR="
        "/share/software/user/open/eigen/3.4.0/share/eigen3/cmake"
    )
    for definitions in expected_fresh_definitions.values():
        assert definitions.count(eigen_package_definition) == 1
    assert matrix["resource_safeguards"][
        "cmake_fresh_configure_definitions_by_source_home"
    ] == expected_fresh_definitions
    assert runner.EXPECTED_RESOURCE_SAFEGUARDS[
        "cmake_fresh_configure_definitions_by_source_home"
    ] == expected_fresh_definitions
    assert matrix["draft_promotion_contract"]["promotion_requirements"] == (
        runner.EXPECTED_PROMOTION_REQUIREMENTS
    )
    assert hashlib.sha256(V2_RUNNER_PATH.read_bytes()).hexdigest() == (
        "8d995e70e77e27e3e9b7c15401150cc34e673ecd273d88bc1f6f55af60ac245d"
    )
    assert hashlib.sha256(V2_TEST_PATH.read_bytes()).hexdigest() == (
        "20f20a3ed1b7e83e047041566d96f4e0ad09e11cda63eb176d3ac933169316b1"
    )


def test_parent_runner_is_hashed_before_its_module_is_executed():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    hash_position = source.index(
        "hashlib.sha256(V2_RUNNER_PATH.read_bytes()).hexdigest()"
    )
    execution_position = source.index("specification.loader.exec_module(module)")

    assert hash_position < execution_position


def test_bytecode_writes_are_disabled_before_parent_import():
    runner = load_runner()
    source = RUNNER_PATH.read_text(encoding="utf-8")

    assignment_position = source.index("sys.dont_write_bytecode = True")
    parent_import_position = source.index("_parent = _load_v2_runner()")
    assert assignment_position < parent_import_position
    assert runner.sys.dont_write_bytecode is True


def test_frozen_contract_locks_sources_and_bundle_hashes():
    runner = load_runner()
    matrix = matrix_document()

    assert runner.validate_v3_contract(matrix) is matrix
    assert matrix["implementation_source_commit"] == (
        runner.EXPECTED_IMPLEMENTATION_SOURCE_COMMIT
    )
    assert matrix["source_inventory_hash_status"] == "FROZEN"
    assert matrix["draft_promotion_contract"]["source_hashes_frozen"] is True
    assert (
        matrix["draft_promotion_contract"][
            "qualification_bundle_hashes_frozen"
        ]
        is True
    )
    assert matrix["draft_promotion_contract"]["execution_allowed"] is True


def test_contract_accepts_only_the_reciprocal_binding_after_promotion():
    runner = load_runner()
    matrix = matrix_document()
    matrix["status"] = runner.EXECUTABLE_MATRIX_STATUS
    matrix["status_reason"] = runner.EXPECTED_STATUS_REASONS[
        runner.EXECUTABLE_MATRIX_STATUS
    ]
    matrix["runner_sha256"] = "1" * 64
    matrix["qualification_bundle_binding"] = (
        runner.EXPECTED_FROZEN_QUALIFICATION_BUNDLE_BINDING
    )
    promotion = matrix["draft_promotion_contract"]
    promotion["current_state"] = runner.EXECUTABLE_MATRIX_STATUS
    promotion["qualification_bundle_hashes_frozen"] = True
    promotion["execution_allowed"] = True

    assert runner.validate_v3_contract(matrix) is matrix

    matrix["qualification_bundle_binding"] = (
        runner.EXPECTED_DRAFT_QUALIFICATION_BUNDLE_BINDING
    )
    with pytest.raises(ValueError, match="inconsistent with lifecycle state"):
        runner.validate_v3_contract(matrix)


def test_shared_execution_identity_is_synchronized_to_v3():
    runner = load_runner()

    assert runner.strict_runner.SCRIPT_PATH == RUNNER_PATH
    assert runner.strict_runner.DEFAULT_REGISTRY == MATRIX_PATH
    assert runner.strict_runner.EXPECTED_MATRIX_ID == runner.EXPECTED_MATRIX_ID
    assert runner._parent.EXPECTED_MATRIX_STATUS == runner.EXPECTED_MATRIX_STATUS
    assert runner.strict_runner.EXPECTED_MATRIX_STATUS == (
        runner.EXPECTED_CHECKED_IN_MATRIX_STATUS
    )
    assert runner.strict_runner.EXPECTED_WORK_PACKAGE == (
        runner.EXPECTED_WORK_PACKAGE
    )
    assert runner._parent.EXPECTED_CHECKED_IN_MATRIX_STATUS == (
        runner.EXPECTED_CHECKED_IN_MATRIX_STATUS
    )
    assert runner._parent.EXPECTED_QUALIFICATION_BUNDLE_BINDING == (
        runner.EXPECTED_QUALIFICATION_BUNDLE_BINDINGS[
            runner.EXPECTED_CHECKED_IN_MATRIX_STATUS
        ]
    )


@pytest.mark.parametrize(
    "raw_commit",
    [
        b"a" * 40,
        b" " + b"a" * 40 + b"\n",
        b"a" * 40 + b" \n",
        b"a" * 40 + b"\r\n",
        b"a" * 40 + b"\n\n",
        b"a" * 40 + b"\n" + b"b" * 40 + b"\n",
    ],
)
@pytest.mark.parametrize("resolver", ["recorded", "head"])
def test_commit_resolution_rejects_malformed_git_output(
    tmp_path,
    monkeypatch,
    raw_commit,
    resolver,
):
    runner = load_runner()
    monkeypatch.setattr(
        runner._parent.strict_runner,
        "git_bytes",
        lambda *args: raw_commit,
    )

    with pytest.raises(ValueError, match="commit output is malformed"):
        if resolver == "recorded":
            runner._resolved_commit(
                tmp_path,
                "a" * 40,
                "implementation source commit",
            )
        else:
            runner._current_head_commit(tmp_path)


def test_execution_wrapper_replaces_the_inherited_record_title(monkeypatch):
    runner = load_runner()
    observed = {}

    def fake_run(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return "sentinel"

    monkeypatch.setattr(runner, "_shared_run_qualification", fake_run)
    assert runner._run_v3_qualification(
        "argument",
        record_title=runner.PARENT_RECORD_TITLE,
    ) == "sentinel"
    assert observed["args"] == ("argument",)
    assert observed["kwargs"]["record_title"] == runner.V3_RECORD_TITLE

    with pytest.raises(RuntimeError, match="record title changed"):
        runner._run_v3_qualification(record_title="unexpected")


@pytest.mark.parametrize("with_descendant", [False, True])
def test_frozen_bundle_resolves_the_same_canonical_commit_at_descendants(
    tmp_path,
    monkeypatch,
    with_descendant,
):
    runner = load_runner()
    prepared = prepare_bundle_repository(runner, tmp_path, monkeypatch)
    repository = prepared["repository"]
    bundle_commit = prepared["bundle_commit"]
    if with_descendant:
        write_repository_file(repository, "later.txt", b"later record\n")
        commit_paths(repository, "later record", ["later.txt"])
    validation_head = git_bytes(repository, "rev-parse", "HEAD").decode().strip()

    binding = runner.validate_frozen_qualification_bundle(
        prepared["registry"],
        repository / runner.EXPECTED_MATRIX_PATH,
        repository,
        repository / runner.EXPECTED_PROPOSED_RUNNER,
    )

    assert binding["qualification_bundle_commit"] == bundle_commit
    assert binding["validation_head_commit"] == validation_head
    assert binding["bundle_parent_commit"] == prepared["source_commit"]
    assert binding["authority"] == runner.EXPECTED_FROZEN_BUNDLE_AUTHORITY
    assert binding["bundle_changed_paths"] == sorted(
        prepared["bundle_bytes"]
    )
    assert [artifact["role"] for artifact in binding["artifacts"]] == [
        "matrix",
        "runner",
        "focused_test",
    ]
    assert all(
        artifact["git_object_type"] == "blob"
        and artifact["git_mode"] == "100644"
        for artifact in binding["artifacts"]
    )


def test_validate_only_at_a_descendant_is_historical_not_execution_ready(
    tmp_path,
    monkeypatch,
):
    runner = load_runner()
    prepared = prepare_bundle_repository(runner, tmp_path, monkeypatch)
    repository = prepared["repository"]
    write_repository_file(repository, "later.txt", b"later record\n")
    descendant = commit_paths(repository, "later record", ["later.txt"])
    binding = runner.validate_frozen_qualification_bundle(
        prepared["registry"],
        repository / runner.EXPECTED_MATRIX_PATH,
        repository,
        repository / runner.EXPECTED_PROPOSED_RUNNER,
    )
    runner._parent._frozen_qualification_bundle_binding = binding
    registry = {
        "matrix_id": runner.EXPECTED_MATRIX_ID,
        "status": runner.EXECUTABLE_MATRIX_STATUS,
        "implementation_source_commit": prepared["source_commit"],
        "implementation_sources": [],
        "groups": [],
        "quantitative_evidence": [],
    }

    summary = runner.validate_only_summary(registry, runner.ACCEPTED_CLAIM)

    assert summary["qualification_bundle_commit"] == prepared["bundle_commit"]
    assert summary["validation_head_commit"] == descendant
    assert summary["execution_ready"] is False
    assert summary["validation_scope"] == "frozen_historical_validation"
    assert summary["outcome"] == "PASS_FROZEN_VALIDATION"


def test_frozen_bundle_rejects_zero_and_multiple_canonical_candidates(
    tmp_path,
    monkeypatch,
):
    runner = load_runner()
    zero = prepare_bundle_repository(
        runner,
        tmp_path / "zero",
        monkeypatch,
        commit_bundle=False,
    )
    with pytest.raises(ValueError, match="zero canonical bundle candidates"):
        runner.validate_frozen_qualification_bundle(
            zero["registry"],
            zero["repository"] / runner.EXPECTED_MATRIX_PATH,
            zero["repository"],
            zero["repository"] / runner.EXPECTED_PROPOSED_RUNNER,
        )

    multiple = prepare_bundle_repository(
        runner,
        tmp_path / "multiple",
        monkeypatch,
    )
    repository = multiple["repository"]
    primary_branch = git_bytes(
        repository,
        "rev-parse",
        "--abbrev-ref",
        "HEAD",
    ).decode().strip()
    git_bytes(
        repository,
        "checkout",
        "--quiet",
        "-b",
        "alternate",
        multiple["source_commit"],
    )
    for relative_path, contents in multiple["bundle_bytes"].items():
        write_repository_file(repository, relative_path, contents)
    commit_paths(
        repository,
        "alternate frozen bundle",
        list(multiple["bundle_bytes"]),
    )
    git_bytes(repository, "checkout", "--quiet", primary_branch)
    git_bytes(
        repository,
        "merge",
        "--quiet",
        "--no-ff",
        "alternate",
        "-m",
        "merge frozen bundles",
    )
    with pytest.raises(ValueError, match="observed 2"):
        runner.validate_frozen_qualification_bundle(
            multiple["registry"],
            repository / runner.EXPECTED_MATRIX_PATH,
            repository,
            repository / runner.EXPECTED_PROPOSED_RUNNER,
        )


def test_frozen_bundle_rejects_non_descendant_validation_head(
    tmp_path,
    monkeypatch,
):
    runner = load_runner()
    prepared = prepare_bundle_repository(runner, tmp_path, monkeypatch)
    repository = prepared["repository"]
    tree = git_bytes(repository, "rev-parse", "HEAD^{tree}").decode().strip()
    unrelated_commit = git_bytes(
        repository,
        "commit-tree",
        tree,
        input_bytes=b"unrelated root\n",
    ).decode().strip()
    git_bytes(repository, "checkout", "--quiet", "--detach", unrelated_commit)

    with pytest.raises(ValueError, match="validation HEAD must descend"):
        runner.validate_frozen_qualification_bundle(
            prepared["registry"],
            repository / runner.EXPECTED_MATRIX_PATH,
            repository,
            repository / runner.EXPECTED_PROPOSED_RUNNER,
        )


def test_frozen_bundle_rejects_committed_blob_drift(
    tmp_path,
    monkeypatch,
):
    runner = load_runner()
    prepared = prepare_bundle_repository(runner, tmp_path, monkeypatch)
    repository = prepared["repository"]
    changed_focused_bytes = b"later focused test bytes\n"
    write_repository_file(
        repository,
        runner.EXPECTED_FOCUSED_TEST_PATH,
        changed_focused_bytes,
    )
    commit_paths(
        repository,
        "later focused test",
        [runner.EXPECTED_FOCUSED_TEST_PATH],
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_FOCUSED_TEST_SHA256",
        hashlib.sha256(changed_focused_bytes).hexdigest(),
    )

    with pytest.raises(ValueError, match="frozen blob drift"):
        runner.validate_frozen_qualification_bundle(
            prepared["registry"],
            repository / runner.EXPECTED_MATRIX_PATH,
            repository,
            repository / runner.EXPECTED_PROPOSED_RUNNER,
        )


@pytest.mark.parametrize("invalid_case", ["bad_parent", "wrong_paths"])
def test_frozen_bundle_rejects_bad_parent_or_path_topology(
    tmp_path,
    monkeypatch,
    invalid_case,
):
    runner = load_runner()
    prepared = prepare_bundle_repository(
        runner,
        tmp_path,
        monkeypatch,
        commit_bundle=False,
    )
    repository = prepared["repository"]
    if invalid_case == "bad_parent":
        write_repository_file(repository, "intermediate.txt", b"intermediate\n")
        commit_paths(repository, "intermediate", ["intermediate.txt"])
        commit_paths(
            repository,
            "late frozen bundle",
            list(prepared["bundle_bytes"]),
        )
    else:
        write_repository_file(repository, "unexpected.txt", b"unexpected\n")
        commit_paths(
            repository,
            "bundle with extra path",
            [*prepared["bundle_bytes"], "unexpected.txt"],
        )

    with pytest.raises(
        ValueError,
        match="zero canonical bundle candidates.*changed paths",
    ):
        runner.validate_frozen_qualification_bundle(
            prepared["registry"],
            repository / runner.EXPECTED_MATRIX_PATH,
            repository,
            repository / runner.EXPECTED_PROPOSED_RUNNER,
        )


def test_frozen_source_validation_uses_recorded_commit_blobs(
    tmp_path,
):
    runner = load_runner()
    repository = tmp_path / "repository"
    initialize_repository(repository)
    source_path = "Code/Source/solver/FE/example.cpp"
    recorded_bytes = b"recorded source\n"
    write_repository_file(repository, source_path, recorded_bytes)
    source_commit = commit_paths(repository, "recorded source", [source_path])
    write_repository_file(repository, source_path, b"later source drift\n")
    commit_paths(repository, "later source drift", [source_path])
    registry = {
        "status": runner.EXECUTABLE_MATRIX_STATUS,
        "implementation_source_commit": source_commit,
        "parent_artifacts": [],
        "implementation_sources": [
            {
                "path": source_path,
                "sha256": hashlib.sha256(recorded_bytes).hexdigest(),
            }
        ],
    }

    runner.validate_frozen_dependencies(registry, repository)
    observation = runner.observe_implementation_sources(registry, repository)
    assert observation["observation_commit"] == source_commit
    assert observation["all_match"] is True
    assert observation["records"][0]["observed_sha256"] == (
        hashlib.sha256(recorded_bytes).hexdigest()
    )

    registry["implementation_sources"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="recorded commit"):
        runner.validate_frozen_dependencies(registry, repository)


def test_contract_locks_exact_ordered_implementation_source_roles():
    runner = load_runner()
    matrix = matrix_document()
    observed = tuple(
        (entry["path"], entry["role"])
        for entry in matrix["implementation_sources"]
    )

    assert len(observed) == 62
    assert observed == runner.EXPECTED_IMPLEMENTATION_SOURCE_ROLES

    missing = copy.deepcopy(matrix)
    missing["implementation_sources"] = [
        entry
        for entry in missing["implementation_sources"]
        if entry["path"] != "Code/Source/solver/FE/Systems/SystemSetup.h"
    ]
    with pytest.raises(ValueError, match="implementation source manifest changed"):
        runner.validate_v3_contract(missing)

    changed_role = copy.deepcopy(matrix)
    changed_role["implementation_sources"][0]["role"] = "changed role"
    with pytest.raises(ValueError, match="implementation source manifest changed"):
        runner.validate_v3_contract(changed_role)

    reordered = copy.deepcopy(matrix)
    reordered["implementation_sources"][0:2] = reversed(
        reordered["implementation_sources"][0:2]
    )
    with pytest.raises(ValueError, match="implementation source manifest changed"):
        runner.validate_v3_contract(reordered)


def test_source_preflight_rejects_a_noncanonical_root_before_delegation(
    tmp_path,
    monkeypatch,
):
    runner = load_runner()

    def unexpected_delegate(*args, **kwargs):
        pytest.fail("resource delegate must not run")

    monkeypatch.setattr(
        runner,
        "_parent_require_execution_resource_preflight",
        unexpected_delegate,
    )
    with pytest.raises(ValueError, match="must equal the runner repository root"):
        runner.require_execution_resource_preflight(
            tmp_path,
            tmp_path / "output",
        )


def test_source_preflight_rejects_an_attached_head(tmp_path, monkeypatch):
    runner = load_runner()

    monkeypatch.setattr(
        runner._parent.strict_runner,
        "git_bytes",
        lambda *args: b"refs/heads/topic\n",
    )
    with pytest.raises(ValueError, match="must use a detached HEAD"):
        runner._validate_execution_source_worktree(tmp_path, tmp_path)


def test_source_preflight_rejects_an_internal_common_directory(
    tmp_path,
    monkeypatch,
):
    runner = load_runner()
    bundle_commit = "b" * 40
    runner._parent._frozen_qualification_bundle_binding = {
        "qualification_bundle_commit": bundle_commit,
    }

    def git_bytes(source_root, *arguments):
        assert source_root == tmp_path.resolve()
        if arguments[0] == "symbolic-ref":
            raise runner._parent.subprocess.CalledProcessError(1, arguments)
        if arguments == (
            runner.GIT_NO_REPLACE_OBJECTS,
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
        ):
            return f"{bundle_commit}\n".encode()
        assert arguments == ("rev-parse", "--git-common-dir")
        return b".git\n"

    monkeypatch.setattr(runner._parent.strict_runner, "git_bytes", git_bytes)
    with pytest.raises(ValueError, match="external Git common directory"):
        runner._validate_execution_source_worktree(tmp_path, tmp_path)


def test_source_preflight_rejects_an_unknown_head_state(
    tmp_path,
    monkeypatch,
):
    runner = load_runner()

    def git_bytes(source_root, *arguments):
        raise runner._parent.subprocess.CalledProcessError(128, arguments)

    monkeypatch.setattr(runner._parent.strict_runner, "git_bytes", git_bytes)
    with pytest.raises(ValueError, match="HEAD state is unavailable"):
        runner._validate_execution_source_worktree(tmp_path, tmp_path)


def test_detached_external_source_preflight_delegates_resource_checks(
    tmp_path,
    monkeypatch,
):
    runner = load_runner()
    observed = {}
    common_directory = tmp_path / "external-common"
    output_directory = tmp_path / "output"
    build_directories = (tmp_path / "build",)
    bundle_commit = "b" * 40
    runner._parent._frozen_qualification_bundle_binding = {
        "qualification_bundle_commit": bundle_commit,
    }

    def git_bytes(source_root, *arguments):
        assert source_root == runner.REPOSITORY_ROOT.resolve()
        if arguments[0] == "symbolic-ref":
            raise runner._parent.subprocess.CalledProcessError(1, arguments)
        if arguments == (
            runner.GIT_NO_REPLACE_OBJECTS,
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
        ):
            return f"{bundle_commit}\n".encode()
        assert arguments == ("rev-parse", "--git-common-dir")
        return f"{common_directory}\n".encode()

    def delegated(source_root, output, builds):
        observed["arguments"] = (source_root, output, builds)

    monkeypatch.setattr(runner._parent.strict_runner, "git_bytes", git_bytes)
    monkeypatch.setattr(
        runner,
        "_parent_require_execution_resource_preflight",
        delegated,
    )
    runner.require_execution_resource_preflight(
        runner.REPOSITORY_ROOT,
        output_directory,
        build_directories,
    )
    assert observed["arguments"] == (
        runner.REPOSITORY_ROOT,
        output_directory,
        build_directories,
    )


def test_execution_preflight_rejects_a_descendant_of_the_bundle_commit(
    tmp_path,
    monkeypatch,
):
    runner = load_runner()
    prepared = prepare_bundle_repository(runner, tmp_path, monkeypatch)
    repository = prepared["repository"]
    write_repository_file(repository, "later.txt", b"later record\n")
    descendant = commit_paths(repository, "later record", ["later.txt"])
    git_bytes(repository, "checkout", "--quiet", "--detach", descendant)
    runner._parent._frozen_qualification_bundle_binding = {
        "qualification_bundle_commit": prepared["bundle_commit"],
    }

    with pytest.raises(ValueError, match="must equal the canonical bundle"):
        runner._validate_execution_source_worktree(repository, repository)


def test_untracked_scan_includes_the_whole_resolved_source_root(
    tmp_path,
    monkeypatch,
):
    runner = load_runner()
    source_root = tmp_path / "source"
    source_root.mkdir()
    allowed_output_root = tmp_path / "output"
    existing_ignored_root = tmp_path / "existing"
    observed = {}

    def delegated(source, allowed, ignored):
        observed["arguments"] = (source, allowed, ignored)
        return {"sentinel": True}

    monkeypatch.setattr(
        runner,
        "_shared_untracked_source_record",
        delegated,
    )
    result = runner.untracked_source_record(
        source_root,
        allowed_output_root,
        (existing_ignored_root, source_root),
    )

    forwarded_source, forwarded_output, forwarded_ignored = observed["arguments"]
    assert result == {"sentinel": True}
    assert forwarded_source == source_root.resolve()
    assert forwarded_output == allowed_output_root
    assert existing_ignored_root in forwarded_ignored
    assert forwarded_ignored.count(source_root.resolve()) == 1
    assert runner.strict_runner.untracked_source_record is (
        runner.untracked_source_record
    )


def test_build_wrapper_routes_exact_locked_fresh_configures(
    monkeypatch,
    tmp_path,
):
    runner = load_runner()
    observed_commands = []

    def delegated(command, *args):
        observed_commands.append(command)
        return {"command": command}

    monkeypatch.setattr(runner, "_parent_run_build_phase", delegated)
    common_arguments = (
        runner.REPOSITORY_ROOT,
        tmp_path / "output",
        tmp_path / "stdout.txt",
        tmp_path / "stderr.txt",
        60,
    )
    expected_configures = []
    for ordinal, (relative_home, definitions) in enumerate(
        runner.EXPECTED_FRESH_CONFIGURE_DEFINITIONS.items(),
        start=1,
    ):
        source_home = (runner.REPOSITORY_ROOT / relative_home).resolve()
        build_home = tmp_path / f"build-{ordinal}"
        configure = [
            "cmake",
            "-S",
            str(source_home),
            "-B",
            str(build_home),
        ]
        original = list(configure)
        result = runner.run_build_phase(configure, *common_arguments)
        expected = [
            "cmake",
            "--fresh",
            *definitions,
            "-S",
            str(source_home),
            "-B",
            str(build_home),
        ]
        expected_configures.append(expected)
        assert configure == original
        assert result["command"] == expected
        assert result["command"].count("--fresh") == 1
        for definition in definitions:
            assert result["command"].count(definition) == 1

    already_locked = list(expected_configures[0])
    runner.run_build_phase(already_locked, *common_arguments)
    build = ["cmake", "--build", "build", "--parallel", "1"]
    runner.run_build_phase(build, *common_arguments)
    assert observed_commands[:2] == expected_configures
    assert observed_commands[2] == expected_configures[0]
    assert observed_commands[2].count("--fresh") == 1
    assert observed_commands[3] == build
    assert runner._parent.run_build_phase is runner.run_build_phase
    assert runner.strict_runner.run_build_phase is runner.run_build_phase


@pytest.mark.parametrize(
    "route_case",
    ["unknown_home", "missing_build", "duplicate_source", "extra_option"],
)
def test_build_wrapper_rejects_unrecognized_or_ambiguous_configure_routes(
    monkeypatch,
    tmp_path,
    route_case,
):
    runner = load_runner()
    source_home = (
        runner.REPOSITORY_ROOT / "Code/Source/solver/FE"
    ).resolve()
    build_home = tmp_path / "build"
    commands = {
        "unknown_home": [
            "cmake",
            "-S",
            str(tmp_path / "unknown"),
            "-B",
            str(build_home),
        ],
        "missing_build": ["cmake", "-S", str(source_home)],
        "duplicate_source": [
            "cmake",
            "-S",
            str(source_home),
            "-S",
            str(source_home),
            "-B",
            str(build_home),
        ],
        "extra_option": [
            "cmake",
            "-S",
            str(source_home),
            "-B",
            str(build_home),
            "-G",
            "Unix Makefiles",
        ],
    }

    def unexpected_delegate(*args, **kwargs):
        pytest.fail("invalid configure command must not be delegated")

    monkeypatch.setattr(
        runner,
        "_parent_run_build_phase",
        unexpected_delegate,
    )
    with pytest.raises(ValueError, match="CMake configure"):
        runner.run_build_phase(
            commands[route_case],
            runner.REPOSITORY_ROOT,
            tmp_path / "output",
            tmp_path / "stdout.txt",
            tmp_path / "stderr.txt",
            60,
        )


@pytest.mark.parametrize(
    "caller_definitions",
    [
        ["-DFE_WITH_MESH=OFF"],
        ["-DFE_WITH_MESH:BOOL=ON"],
        ["-DUNLOCKED_OPTION=ON"],
        ["-DFE_WITH_MESH=ON", "-DFE_WITH_MESH=ON"],
        ["-D", "FE_WITH_MESH=ON"],
    ],
)
def test_build_wrapper_rejects_nonexact_cache_definitions(
    monkeypatch,
    tmp_path,
    caller_definitions,
):
    runner = load_runner()
    source_home = (
        runner.REPOSITORY_ROOT / "Code/Source/solver/FE"
    ).resolve()
    command = [
        "cmake",
        "-S",
        str(source_home),
        "-B",
        str(tmp_path / "build"),
        *caller_definitions,
    ]

    def unexpected_delegate(*args, **kwargs):
        pytest.fail("nonexact configure command must not be delegated")

    monkeypatch.setattr(
        runner,
        "_parent_run_build_phase",
        unexpected_delegate,
    )
    with pytest.raises(ValueError, match="CMake configure"):
        runner.run_build_phase(
            command,
            runner.REPOSITORY_ROOT,
            tmp_path / "output",
            tmp_path / "stdout.txt",
            tmp_path / "stderr.txt",
            60,
        )


def test_build_wrapper_rejects_mixed_build_and_configure_route(
    monkeypatch,
    tmp_path,
):
    runner = load_runner()

    def unexpected_delegate(*args, **kwargs):
        pytest.fail("mixed CMake route must not be delegated")

    monkeypatch.setattr(
        runner,
        "_parent_run_build_phase",
        unexpected_delegate,
    )
    with pytest.raises(ValueError, match="route is ambiguous"):
        runner.run_build_phase(
            ["cmake", "--build", "build", "-S", "source"],
            runner.REPOSITORY_ROOT,
            tmp_path / "output",
            tmp_path / "stdout.txt",
            tmp_path / "stderr.txt",
            60,
        )


def test_contract_locks_five_groups_thirty_three_tests_and_eight_properties():
    runner = load_runner()
    matrix = matrix_document()

    assert matrix["gates"] == runner.EXPECTED_GATES
    assert len(matrix["groups"]) == 5
    assert sum(len(group["tests"]) for group in matrix["groups"]) == 33
    assert len(matrix["quantitative_evidence"]) == 8
    assert [group["id"] for group in matrix["groups"]] == list(
        runner.EXPECTED_GROUP_TESTS
    )
    production = next(
        group
        for group in matrix["groups"]
        if group["id"] == runner.PRODUCTION_GROUP_ID
    )
    assert tuple(production["tests"]) == runner.PRODUCTION_TESTS


def test_contract_locks_floor_safe_cap_and_direct_upper_limit():
    runner = load_runner()
    matrix = matrix_document()

    assert runner.METHOD_ENERGY_FLOOR == 0.25
    assert runner.SAFE_GROUPED_SYMMETRIC_RATIO_CAP.hex() == (
        "0x1.1fffffffffffep-1"
    )
    upper_limit = (
        runner.TRACE_PENALTY_GAMMA
        * runner.SAFE_GROUPED_SYMMETRIC_RATIO_CAP
    )
    assert upper_limit.hex() == "0x1.afffffffffffdp+2"
    assert matrix["case_axes"]["required_minimum_energy_ratio"] == 0.25
    assert matrix["case_axes"]["downward_safe_group_ratio_cap"] == (
        runner.SAFE_GROUPED_SYMMETRIC_RATIO_CAP
    )
    maximum_property = next(
        item
        for item in matrix["quantitative_evidence"]
        if item["property"].endswith("maximum_upper_bound")
    )
    assert maximum_property["relation"] == "less_than_or_equal"
    assert maximum_property["threshold"] == upper_limit


def test_contract_locks_production_scope_and_digest_boundary():
    runner = load_runner()
    matrix = matrix_document()
    contract = matrix["certified_aggregate_trace_contract"]
    envelope = matrix["certificate_envelope"]

    assert "production Navier-Stokes viscous/Nitsche subform" in (
        matrix["qualification_scope"]
    )
    assert "generic FE gate is conditional" in matrix["qualification_scope"]
    assert contract["coercive_bulk_energy_authority"] == (
        "production_Navier-Stokes_module-supplied_viscous_K"
    )
    assert contract["generic_FE_gate_scope"] == (
        "conditional_on_an_installed_caller-supplied_coercive_bulk_form"
    )
    assert contract["floor_bits_excluded_from_digests"] == [
        "emitted_route_form_binding_digest",
        "exact_aggregate_trace_certificate_digest",
    ]
    assert envelope["floor_bits_bind_policy_signature"] is True
    assert envelope["floor_bits_bind_current_certificate_cache_digest"] is True
    assert envelope["floor_bits_bind_emitted_route_form_binding_digest"] is False
    assert envelope["floor_bits_bind_exact_trace_certificate_digest"] is False
    assert runner.validate_v3_contract(matrix) is matrix


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda matrix: matrix.__setitem__(
                "implementation_source_commit", "a" * 40
            ),
            "implementation source commit changed",
        ),
        (
            lambda matrix: matrix["draft_promotion_contract"].__setitem__(
                "qualification_bundle_hashes_frozen",
                not matrix["draft_promotion_contract"][
                    "qualification_bundle_hashes_frozen"
                ],
            ),
            "qualification_bundle_hashes_frozen",
        ),
        (
            lambda matrix: matrix.__setitem__(
                "method_coercivity_lower_bound", 0.5
            ),
            "method energy floor changed",
        ),
        (
            lambda matrix: matrix["certificate_envelope"].__setitem__(
                "floor_bits_bind_exact_trace_certificate_digest", True
            ),
            "aggregate trace contract changed|digest exclusions changed|"
            "certificate envelope changed",
        ),
        (
            lambda matrix: matrix.__setitem__(
                "status_reason", "stale lifecycle statement"
            ),
            "status reason changed",
        ),
        (
            lambda matrix: matrix["draft_promotion_contract"].__setitem__(
                "promotion_requirements", ["out-of-order hash construction"]
            ),
            "promotion requirements changed",
        ),
    ],
)
def test_contract_rejects_lifecycle_floor_and_digest_drift(mutation, message):
    runner = load_runner()
    matrix = matrix_document()
    mutation(matrix)

    with pytest.raises(ValueError, match=message):
        runner.validate_v3_contract(matrix)


def test_validate_only_reports_the_frozen_prerequisite_preflight(capsys):
    runner = load_runner()

    assert runner.main(["--validate-only"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["status"] == runner.EXECUTABLE_MATRIX_STATUS
    assert result["execution_ready"] is False
    assert result["validation_scope"] == "frozen_historical_validation"
    assert result["group_count"] == 5
    assert result["test_count"] == 33
    assert result["quantitative_evidence_gate_count"] == 8
    assert result["method_coercivity_lower_bound"] == 0.25
    assert result["uniform_bound_status"] == runner.UNIFORM_BOUND_STATUS
    assert result["requested_claim"] == runner.ACCEPTED_CLAIM
    assert result["outcome"] == "PASS_FROZEN_VALIDATION"


def test_trace_parser_accepts_exact_floor_for_wet_and_dry_cases():
    runner = load_runner()
    cases, summary = trace_records(runner)

    evidence = runner.parse_trace_evidence(
        trace_stdout(runner, cases, summary)
    )

    assert evidence["outcome"] == "PASS"
    assert evidence["observed_case_count"] == 108
    assert evidence["wet_case_count"] == 96
    assert evidence["dry_case_count"] == 12
    assert evidence["minimum_finite_sample_energy_lower_bound"] == 0.25
    assert evidence["method_coercivity_lower_bound"] == 0.25
    assert evidence["uniform_bound_status"] == runner.UNIFORM_BOUND_STATUS
    assert evidence["requested_claim"] == runner.ACCEPTED_CLAIM
    assert {case["required_minimum_energy_ratio"] for case in evidence["cases"]} == {
        0.25
    }
    assert {
        case["finite_sample_energy_lower_bound"] for case in evidence["cases"]
    } == {0.25}


@pytest.mark.parametrize("case_index", [0, 1])
@pytest.mark.parametrize(
    "field",
    [
        "required_minimum_energy_ratio",
        "finite_sample_energy_lower_bound",
    ],
)
def test_trace_parser_rejects_nonexact_floor_for_dry_and_wet_cases(
    case_index,
    field,
):
    runner = load_runner()
    cases, summary = trace_records(runner)
    cases[case_index][field] = math.nextafter(0.25, 1.0)

    with pytest.raises(ValueError, match="energy floor is not exact"):
        runner.parse_trace_evidence(trace_stdout(runner, cases, summary))


def test_trace_parser_accepts_safe_cap_and_rejects_its_next_value():
    runner = load_runner()
    cases, summary = trace_records(runner, gap=0.0)
    set_wet_case_ratio(
        runner,
        cases,
        summary,
        runner.SAFE_GROUPED_SYMMETRIC_RATIO_CAP,
    )

    evidence = runner.parse_trace_evidence(
        trace_stdout(runner, cases, summary)
    )
    assert evidence["maximum_trace_upper_bound"].hex() == (
        "0x1.afffffffffffdp+2"
    )

    cases, summary = trace_records(runner, gap=0.0)
    set_wet_case_ratio(
        runner,
        cases,
        summary,
        math.nextafter(runner.SAFE_GROUPED_SYMMETRIC_RATIO_CAP, math.inf),
    )
    with pytest.raises(ValueError, match="exceeds the direct safe risk cap"):
        runner.parse_trace_evidence(trace_stdout(runner, cases, summary))


def test_trace_parser_rejects_tolerant_overcap_representation_mismatches():
    runner = load_runner()
    cases, summary = trace_records(runner, gap=0.0)
    case = next(
        item for item in cases if item["target_wall_fraction"] != 0.0
    )
    case["grouped_symmetric_ratio"] = (
        runner.SAFE_GROUPED_SYMMETRIC_RATIO_CAP
    )
    case["trace_to_penalty_ratio"] = float.fromhex(
        "0x1.2000000000001p-1"
    )
    case["trace_upper_bound"] = float.fromhex("0x1.b000000000002p+2")
    summary["maximum_trace_upper_bound"] = case["trace_upper_bound"]

    with pytest.raises(ValueError, match="reported trace ratio exceeds"):
        runner.parse_trace_evidence(trace_stdout(runner, cases, summary))

    cases, summary = trace_records(runner, gap=0.0)
    set_wet_case_ratio(
        runner,
        cases,
        summary,
        runner.SAFE_GROUPED_SYMMETRIC_RATIO_CAP,
    )
    case = next(
        item for item in cases if item["target_wall_fraction"] != 0.0
    )
    case["trace_upper_bound"] = math.nextafter(
        runner.DIRECT_SAFE_TRACE_UPPER_BOUND_LIMIT,
        math.inf,
    )
    summary["maximum_trace_upper_bound"] = case["trace_upper_bound"]
    with pytest.raises(ValueError, match="trace upper bound exceeds"):
        runner.parse_trace_evidence(trace_stdout(runner, cases, summary))


def test_trace_parser_requires_exact_penalty_and_capped_summary():
    runner = load_runner()
    cases, summary = trace_records(runner, gap=0.0)
    set_wet_case_ratio(
        runner,
        cases,
        summary,
        runner.SAFE_GROUPED_SYMMETRIC_RATIO_CAP,
    )
    case = next(
        item for item in cases if item["target_wall_fraction"] != 0.0
    )
    case["effective_penalty_multiplier"] = math.nextafter(
        runner.TRACE_PENALTY_GAMMA,
        0.0,
    )
    with pytest.raises(ValueError, match="effective penalty is not exact"):
        runner.parse_trace_evidence(trace_stdout(runner, cases, summary))

    cases, summary = trace_records(runner, gap=0.0)
    set_wet_case_ratio(
        runner,
        cases,
        summary,
        runner.SAFE_GROUPED_SYMMETRIC_RATIO_CAP,
    )
    summary["maximum_trace_upper_bound"] = math.nextafter(
        runner.DIRECT_SAFE_TRACE_UPPER_BOUND_LIMIT,
        math.inf,
    )
    with pytest.raises(ValueError, match="summary maximum upper bound"):
        runner.parse_trace_evidence(trace_stdout(runner, cases, summary))


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "minimum_finite_sample_energy_lower_bound",
            math.nextafter(0.25, 1.0),
            "summary minimum energy floor is not exact",
        ),
        (
            "method_coercivity_lower_bound",
            math.nextafter(0.25, 0.0),
            "summary method energy floor is not exact",
        ),
        (
            "uniform_bound_status",
            "OPEN",
            "summary uniform-bound status changed",
        ),
        (
            "accepted_claim",
            "joint_low_level_prerequisite",
            "summary accepted claim changed",
        ),
    ],
)
def test_trace_parser_rejects_summary_floor_status_and_claim_drift(
    field,
    replacement,
    message,
):
    runner = load_runner()
    cases, summary = trace_records(runner)
    summary[field] = replacement

    with pytest.raises(ValueError, match=message):
        runner.parse_trace_evidence(trace_stdout(runner, cases, summary))


def test_trace_parser_retains_the_explicit_floor_binding_boundary():
    runner = load_runner()
    cases, summary = trace_records(runner)

    evidence = runner.parse_trace_evidence(
        trace_stdout(runner, cases, summary)
    )

    assert evidence["floor_binding_boundary"] == {
        "collective_policy_signature": True,
        "certificate_cache_digest": True,
        "emitted_route_digest": False,
        "exact_certificate_digest": False,
    }


def test_trace_parser_requires_exact_fields_unique_ids_and_valid_gaps():
    runner = load_runner()
    cases, summary = trace_records(runner)
    del cases[0]["required_minimum_energy_ratio"]
    with pytest.raises(ValueError, match="trace case 0 fields changed"):
        runner.parse_trace_evidence(trace_stdout(runner, cases, summary))

    cases, summary = trace_records(runner)
    cases[1]["case_id"] = cases[0]["case_id"]
    with pytest.raises(ValueError, match="duplicate trace case_id"):
        runner.parse_trace_evidence(trace_stdout(runner, cases, summary))

    cases, summary = trace_records(runner)
    cases[1]["sampled_eigenvalue_gap"] = 0.0
    with pytest.raises(ValueError, match="gap is inconsistent"):
        runner.parse_trace_evidence(trace_stdout(runner, cases, summary))


def test_trace_json_records_reject_duplicate_object_keys():
    runner = load_runner()
    stdout = (
        runner.TRACE_CASE_PREFIX
        + '{"case_id":"first","case_id":"second"}'
    )

    with pytest.raises(ValueError, match="duplicate JSON key: case_id"):
        runner._json_records(stdout, runner.TRACE_CASE_PREFIX)
