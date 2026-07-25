import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT / "tests" / "cases" / "fluid" / "run_free_surface_q0_harness_qualification.py"
)
MATRIX_PATH = RUNNER_PATH.with_name("free_surface_q0_harness_qualification_matrix.json")


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_q0_harness_qualification_runner",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def raw_matrix():
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def write_pytest_junit(
    runner,
    path,
    node_ids,
    statuses=None,
    declared_overrides=None,
):
    statuses = statuses or {}
    counts = {"failures": 0, "errors": 0, "skipped": 0}
    status_to_count = {
        "failure": "failures",
        "error": "errors",
        "skipped": "skipped",
    }
    for status in statuses.values():
        counts[status_to_count[status]] += 1
    attributes = {
        "tests": str(len(node_ids)),
        **{key: str(value) for key, value in counts.items()},
    }
    attributes.update(
        {key: str(value) for key, value in (declared_overrides or {}).items()}
    )
    root = ET.Element("testsuites", {"name": "pytest tests"})
    suite = ET.SubElement(root, "testsuite", {"name": "pytest", **attributes})
    for node_id in node_ids:
        classname, name = runner.pytest_junit_identity(node_id)
        testcase = ET.SubElement(
            suite,
            "testcase",
            {"classname": classname, "name": name},
        )
        status = statuses.get(node_id)
        if status is not None:
            ET.SubElement(testcase, status)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def test_q0_matrix_bytes_and_canonical_path_are_frozen(tmp_path):
    runner = load_runner()
    digest = hashlib.sha256(MATRIX_PATH.read_bytes()).hexdigest()

    assert digest == runner.EXPECTED_MATRIX_SHA256
    assert runner.load_matrix(MATRIX_PATH)["status"] == (
        "FROZEN_PREREQUISITE_NONCLOSURE"
    )
    copy_path = tmp_path / MATRIX_PATH.name
    copy_path.write_bytes(MATRIX_PATH.read_bytes())
    with pytest.raises(ValueError, match="canonical frozen matrix path"):
        runner.load_matrix(copy_path)


def test_q0_matrix_rejects_symbolic_link_alias_before_resolution(tmp_path):
    runner = load_runner()
    alias = tmp_path / MATRIX_PATH.name
    alias.symlink_to(MATRIX_PATH)

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        runner.load_matrix(alias)


def test_q0_matrix_rejects_single_byte_drift(tmp_path, monkeypatch):
    runner = load_runner()
    mutated = bytearray(MATRIX_PATH.read_bytes())
    mutated[0] ^= 1
    path = tmp_path / MATRIX_PATH.name
    path.write_bytes(mutated)
    monkeypatch.setattr(runner, "DEFAULT_MATRIX", path)

    with pytest.raises(ValueError, match="frozen matrix bytes changed"):
        runner.load_matrix(path)


def test_q0_matrix_has_no_duplicate_json_keys():
    runner = load_runner()

    json.loads(
        MATRIX_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=runner.reject_duplicate_keys,
    )


def test_q0_source_definitions_and_open_campaign_state_are_exact():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    records = runner.validate_source_definitions(matrix, ROOT)
    registry = json.loads(
        (
            ROOT
            / "tests"
            / "cases"
            / "fluid"
            / "free_surface_qualification_campaign_registry.json"
        ).read_text(encoding="utf-8")
    )
    q0 = registry["campaigns"][0]

    assert len(records) == 20
    assert q0["id"] == "Q0"
    assert q0["state"] == "UNRESOLVED"
    assert q0["child_programs"][1]["id"] == "q0_campaign_execution"
    assert q0["child_programs"][1]["registration_state"] == "UNRESOLVED"


def test_q0_contract_freezes_only_control_prerequisites():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)

    assert len(matrix["gtest_group"]["tests"]) == 24
    assert len(matrix["pytest_group"]["tests"]) == 44
    assert {entry["id"] for entry in matrix["open_exits"]} == (
        runner.EXPECTED_OPEN_EXITS
    )
    assert matrix["qualification_disposition"] == runner.EXPECTED_DISPOSITION
    assert (
        matrix["qualification_disposition"]["wp0_invalid_input_matrix_ci_registered"]
        is True
    )
    assert matrix["qualification_disposition"]["q0_closed"] is False
    assert (
        matrix["qualification_disposition"]["audit_q0_checkbox_may_be_checked"] is False
    )


@pytest.mark.parametrize(
    "claim",
    [
        "q0_closure",
        "future_q0_closure",
        "q0_qualification",
        "q0_campaign_pass",
        "physical_gate_ready",
    ],
)
def test_q0_rejects_every_closing_or_promoting_claim(claim):
    runner = load_runner()

    with pytest.raises(ValueError, match="outside this matrix"):
        runner.parse_control(["--requested-claim", claim])


def test_q0_rejects_unknown_prerequisite_claim():
    runner = load_runner()

    with pytest.raises(ValueError, match="unsupported Q0 requested claim"):
        runner.parse_control(["--requested-claim", "unregistered_prerequisite"])


def test_q0_contract_rejects_source_escape():
    runner = load_runner()
    matrix = copy.deepcopy(raw_matrix())
    matrix["source_definitions"][0]["path"] = "../outside.md"

    with pytest.raises(ValueError, match="source-definition path"):
        runner.validate_matrix_contract(matrix)


def test_q0_source_hash_drift_is_rejected(monkeypatch):
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    real_read_stable_bytes = runner.read_stable_bytes
    first_path = (ROOT / matrix["source_definitions"][0]["path"]).resolve()

    def changed_bytes(path):
        if path.resolve() == first_path:
            return b"changed source bytes"
        return real_read_stable_bytes(path)

    monkeypatch.setattr(runner, "read_stable_bytes", changed_bytes)
    with pytest.raises(ValueError, match="source-definition hash changed"):
        runner.validate_source_definitions(matrix, ROOT)


@pytest.mark.parametrize("symbolic_component", ["parent", "leaf"])
def test_q0_source_definition_rejects_every_symbolic_link_component(
    tmp_path,
    symbolic_component,
):
    runner = load_runner()
    source_root = tmp_path / "source"
    real_parent = source_root / "real"
    real_parent.mkdir(parents=True)
    real_file = real_parent / "definition.txt"
    real_file.write_text("required fragment\n", encoding="utf-8")

    if symbolic_component == "parent":
        (source_root / "linked").symlink_to(real_parent, target_is_directory=True)
        relative = Path("linked/definition.txt")
    else:
        (source_root / "definition.txt").symlink_to(real_file)
        relative = Path("definition.txt")
    matrix = {
        "source_definitions": [
            {
                "id": "definition",
                "role": "test",
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(real_file.read_bytes()).hexdigest(),
                "required_fragments": ["required fragment"],
            }
        ]
    }

    with pytest.raises(ValueError, match="symbolic-link path component"):
        runner.validate_source_definitions(matrix, source_root)


def test_q0_canonical_source_root_rejects_symbolic_link_alias(tmp_path):
    runner = load_runner()
    alias = tmp_path / "repository-alias"
    alias.symlink_to(ROOT, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        runner.canonical_source_root(alias)


def test_q0_wp0_ctest_and_hosted_ci_registration_chain_is_exact():
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)

    record = runner.validate_wp0_ci_registration(matrix, ROOT)

    assert record == {
        "ctest_name": "Physics_FreeSurfaceConfiguration_WP0",
        "test_count": 24,
        "workflow_triggers": ["pull_request", "push"],
        "workflow_jobs": ["test-ubuntu", "test-macos"],
        "hosted_execution_archived": False,
        "outcome": "REGISTERED_AWAITING_HOSTED_EXECUTION",
    }


def test_q0_wp0_ci_registration_rejects_q0_matrix_inventory_drift():
    runner = load_runner()
    matrix = copy.deepcopy(runner.load_matrix(MATRIX_PATH))
    matrix["gtest_group"]["tests"][0] = (
        "NavierStokesLegacyBCs.UnregisteredConfigurationCase"
    )

    with pytest.raises(ValueError, match="Q0 and WP-0 frozen test inventories differ"):
        runner.validate_wp0_ci_registration(matrix, ROOT)


@pytest.mark.parametrize(
    ("relative_path", "old", "new", "diagnostic"),
    [
        (
            "Code/Source/solver/Physics/CMakeLists.txt",
            (
                "NavierStokesLegacyBCs."
                "FittedFreeSurfacePrescribedTangentialMeshPolicyTranslation"
            ),
            "NavierStokesLegacyBCs.UnregisteredConfigurationCase",
            "dedicated CTest inventory differs",
        ),
        (
            ".github/workflows/tests.yml",
            "  push:\n",
            "",
            "must run for push and pull_request",
        ),
        (
            ".github/actions/test-ubuntu/action.yml",
            "ctest --verbose",
            "ctest --verbose -R Physics_Tests",
            "must invoke one unfiltered",
        ),
        (
            ".github/actions/test-macos/action.yml",
            "ctest --verbose",
            "ctest --verbose -R Physics_Tests",
            "must invoke one unfiltered",
        ),
    ],
)
def test_q0_wp0_ci_registration_rejects_chain_drift(
    monkeypatch,
    relative_path,
    old,
    new,
    diagnostic,
):
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    changed_path = (ROOT / relative_path).resolve()
    real_read_stable_bytes = runner.read_stable_bytes
    original = real_read_stable_bytes(changed_path)
    changed = original.decode("utf-8").replace(old, new, 1).encode("utf-8")
    assert changed != original

    def drifted_bytes(path):
        if path.resolve() == changed_path:
            return changed
        return real_read_stable_bytes(path)

    monkeypatch.setattr(runner, "read_stable_bytes", drifted_bytes)
    with pytest.raises(ValueError, match=diagnostic):
        runner.validate_wp0_ci_registration(matrix, ROOT)


@pytest.mark.parametrize("symbolic_component", ["parent", "leaf"])
def test_q0_source_state_rejects_untracked_symbolic_link_components(
    tmp_path,
    monkeypatch,
    symbolic_component,
):
    runner = load_runner()
    source_root = tmp_path / "source"
    source_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "definition.txt"
    outside_file.write_text("external\n", encoding="utf-8")
    if symbolic_component == "parent":
        (source_root / "linked").symlink_to(outside, target_is_directory=True)
        untracked_name = "linked/definition.txt"
    else:
        (source_root / "definition.txt").symlink_to(outside_file)
        untracked_name = "definition.txt"

    def fake_git_bytes(_source_root, *arguments):
        if arguments == ("rev-parse", "HEAD"):
            return b"1" * 40 + b"\n"
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return b"2" * 40 + b"\n"
        if arguments == ("ls-files", "--others", "--exclude-standard", "-z"):
            return untracked_name.encode() + b"\0"
        return b""

    monkeypatch.setattr(runner, "git_bytes", fake_git_bytes)

    with pytest.raises(ValueError, match="symbolic-link path component"):
        runner.source_state(source_root)


def test_q0_rejects_duplicate_gtest_identifiers_in_listing_and_results(
    monkeypatch,
):
    runner = load_runner()
    duplicate_listing = subprocess.CompletedProcess(
        args=["test-binary", "--gtest_list_tests"],
        returncode=0,
        stdout="Suite.\n  Case\n  Case\n",
        stderr="",
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: duplicate_listing,
    )

    with pytest.raises(ValueError, match="duplicate listed GoogleTest identifier"):
        runner.listed_gtests(Path("test-binary"))

    document = {
        "testsuites": [
            {
                "name": "Suite",
                "testsuite": [
                    {"name": "Case", "status": "RUN"},
                    {"name": "Case", "status": "RUN"},
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="duplicate GoogleTest result identifier"):
        runner.flatten_gtests(document)


def test_q0_pytest_execution_uses_only_the_exact_frozen_node_ids(tmp_path):
    runner = load_runner()
    matrix = runner.load_matrix(MATRIX_PATH)
    command = runner.pytest_execution_command(matrix, tmp_path / "pytest.xml")

    assert command[:4] == [sys.executable, "-m", "pytest", "-q"]
    assert command[4] == f"--junitxml={tmp_path / 'pytest.xml'}"
    assert command[5:] == matrix["pytest_group"]["tests"]
    assert len(command[5:]) == 44


def test_q0_pytest_junit_requires_exact_unique_passing_inventory(tmp_path):
    runner = load_runner()
    expected = runner.load_matrix(MATRIX_PATH)["pytest_group"]["tests"]
    path = tmp_path / "pytest.xml"
    write_pytest_junit(runner, path, expected)

    inventory = runner.parse_pytest_junit(path, expected)

    assert inventory["passed"] is True
    assert inventory["observed_tests"] == sorted(expected)
    assert inventory["observed_test_count"] == 44
    assert inventory["unique_observed_test_count"] == 44
    assert inventory["passed_test_count"] == 44
    assert not inventory["missing_tests"]
    assert not inventory["unexpected_tests"]
    assert not inventory["duplicate_tests"]
    assert not inventory["skipped_tests"]
    assert not inventory["failed_tests"]
    assert not inventory["error_tests"]
    assert not inventory["declared_total_mismatches"]


@pytest.mark.parametrize(
    ("defect", "expected_field"),
    [
        ("missing", "missing_tests"),
        ("unexpected", "unexpected_tests"),
        ("duplicate", "duplicate_tests"),
        ("skipped", "skipped_tests"),
        ("failure", "failed_tests"),
        ("error", "error_tests"),
        ("declared_total", "declared_total_mismatches"),
    ],
)
def test_q0_pytest_junit_inventory_defects_fail_closed(
    tmp_path,
    defect,
    expected_field,
):
    runner = load_runner()
    expected = runner.load_matrix(MATRIX_PATH)["pytest_group"]["tests"]
    observed = list(expected)
    statuses = {}
    declared_overrides = {}
    if defect == "missing":
        observed.pop()
    elif defect == "unexpected":
        observed[-1] = "tests/test_unexpected_q0_case.py::test_unexpected"
    elif defect == "duplicate":
        observed.append(observed[0])
    elif defect in {"skipped", "failure", "error"}:
        statuses[observed[0]] = defect
    else:
        declared_overrides["tests"] = len(observed) + 1

    path = tmp_path / "pytest.xml"
    write_pytest_junit(
        runner,
        path,
        observed,
        statuses=statuses,
        declared_overrides=declared_overrides,
    )
    inventory = runner.parse_pytest_junit(path, expected)

    assert inventory["passed"] is False
    assert inventory[expected_field]


def test_q0_pytest_junit_rejects_missing_or_invalid_suite_totals(tmp_path):
    runner = load_runner()
    expected = runner.load_matrix(MATRIX_PATH)["pytest_group"]["tests"]
    path = tmp_path / "pytest.xml"
    write_pytest_junit(runner, path, expected)
    root = ET.parse(path).getroot()
    suite = next(element for element in root.iter() if element.tag == "testsuite")
    del suite.attrib["errors"]
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

    with pytest.raises(ValueError, match="missing 'errors' total"):
        runner.parse_pytest_junit(path, expected)

    write_pytest_junit(
        runner,
        path,
        expected,
        declared_overrides={"errors": "not-an-integer"},
    )
    with pytest.raises(ValueError, match="invalid 'errors' total"):
        runner.parse_pytest_junit(path, expected)


def test_q0_pytest_junit_cannot_hide_an_error_in_an_empty_suite(tmp_path):
    runner = load_runner()
    expected = runner.load_matrix(MATRIX_PATH)["pytest_group"]["tests"]
    path = tmp_path / "pytest.xml"
    write_pytest_junit(runner, path, expected)
    root = ET.parse(path).getroot()
    ET.SubElement(
        root,
        "testsuite",
        {
            "name": "collection",
            "tests": "0",
            "failures": "0",
            "errors": "1",
            "skipped": "0",
        },
    )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

    inventory = runner.parse_pytest_junit(path, expected)

    assert inventory["passed"] is False
    assert inventory["declared_total_mismatches"] == ["suite[1].errors"]


def test_q0_linked_library_manifest_is_address_independent_and_hashes_bytes(
    tmp_path,
):
    runner = load_runner()
    library = tmp_path / "libqualification.so.1.0"
    library.write_bytes(b"library-one")
    library_alias = tmp_path / "libqualification.so.1"
    library_alias.symlink_to(library)
    loader = tmp_path / "ld-qualification.so"
    loader.write_bytes(b"loader-one")
    first_output = (
        "linux-vdso.so.1 (0x00007fff00000000)\n"
        f"libqualification.so.1 => {library_alias} (0x0000700000000000)\n"
        f"{loader} (0x0000710000000000)\n"
    )
    second_output = (
        f"{loader} (0x0000990000000000)\n"
        f"libqualification.so.1 => {library_alias} (0x0000880000000000)\n"
        "linux-vdso.so.1 (0x0000770000000000)\n"
    )

    first = runner.linked_library_manifest(first_output)
    second = runner.linked_library_manifest(second_output)

    assert first == second
    assert first["linkage"] == "dynamic"
    assert first["virtual_dependencies"] == ["linux-vdso.so.1"]
    assert {record["resolved_path"] for record in first["libraries"]} == {
        str(library.resolve()),
        str(loader.resolve()),
    }
    assert {record["sha256"] for record in first["libraries"]} == {
        hashlib.sha256(library.read_bytes()).hexdigest(),
        hashlib.sha256(loader.read_bytes()).hexdigest(),
    }
    assert "0x" not in json.dumps(first, sort_keys=True)

    previous_manifest = first["manifest_sha256"]
    library.write_bytes(b"library-two")
    changed = runner.linked_library_manifest(first_output)
    assert changed["manifest_sha256"] != previous_manifest


def test_q0_linked_library_manifest_rejects_missing_or_duplicate_records(
    tmp_path,
):
    runner = load_runner()
    library = tmp_path / "libqualification.so"
    library.write_bytes(b"library")

    with pytest.raises(ValueError, match="was not found"):
        runner.linked_library_manifest("libqualification.so => not found\n")
    with pytest.raises(ValueError, match="duplicate linked-library record"):
        runner.linked_library_manifest(
            f"libqualification.so => {library} (0x01)\n"
            f"libqualification.so => {library} (0x02)\n"
        )


def write_cmake_cache(
    path,
    build_directory,
    source_directory,
    project_name="svMultiPhysicsPhysics",
    build_tests="ON",
):
    path.write_text(
        "\n".join(
            [
                f"CMAKE_CACHEFILE_DIR:INTERNAL={build_directory}",
                f"CMAKE_HOME_DIRECTORY:INTERNAL={source_directory}",
                f"CMAKE_PROJECT_NAME:STATIC={project_name}",
                f"PHYSICS_BUILD_TESTS:BOOL={build_tests}",
                "CMAKE_BUILD_TYPE:STRING=Debug",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_q0_cmake_cache_snapshot_requires_exact_physics_build_identity(tmp_path):
    runner = load_runner()
    build = tmp_path / "build"
    build.mkdir()
    binary = build / "test_physics"
    binary.write_bytes(b"binary")
    cache = build / "CMakeCache.txt"
    source = ROOT / "Code" / "Source" / "solver" / "Physics"
    write_cmake_cache(cache, build.resolve(), source.resolve())

    snapshot = runner.cmake_cache_snapshot(binary)

    assert snapshot["build_directory"] == str(build.resolve())
    assert snapshot["source_directory"] == str(source.resolve())
    assert snapshot["project_name"] == "svMultiPhysicsPhysics"
    assert snapshot["sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "defect",
    ["missing", "build_directory", "source_directory", "project", "tests_off"],
)
def test_q0_cmake_cache_identity_defects_fail_closed(tmp_path, defect):
    runner = load_runner()
    build = tmp_path / "build"
    build.mkdir()
    binary = build / "test_physics"
    binary.write_bytes(b"binary")
    cache = build / "CMakeCache.txt"
    if defect == "missing":
        with pytest.raises(ValueError, match="does not have a CMake cache"):
            runner.cmake_cache_snapshot(binary)
        return

    source = ROOT / "Code" / "Source" / "solver" / "Physics"
    declared_build = ROOT if defect == "build_directory" else build.resolve()
    declared_source = ROOT if defect == "source_directory" else source.resolve()
    project = "different-project" if defect == "project" else "svMultiPhysicsPhysics"
    build_tests = "OFF" if defect == "tests_off" else "ON"
    write_cmake_cache(
        cache,
        declared_build,
        declared_source,
        project_name=project,
        build_tests=build_tests,
    )

    with pytest.raises(ValueError, match="CMake .* identity changed"):
        runner.cmake_cache_snapshot(binary)


@pytest.mark.parametrize(
    "changed_field",
    ["binary", "cache", "libraries"],
)
def test_q0_build_provenance_drift_fails_closed(changed_field):
    runner = load_runner()
    discovery = {"physics_binary_sha256": "1" * 64}
    before = {
        "binary_sha256": "1" * 64,
        "cmake_cache": {"sha256": "2" * 64},
        "linked_library_manifest_sha256": "3" * 64,
    }
    after = copy.deepcopy(before)
    if changed_field == "binary":
        after["binary_sha256"] = "4" * 64
    elif changed_field == "cache":
        after["cmake_cache"]["sha256"] = "4" * 64
    else:
        after["linked_library_manifest_sha256"] = "4" * 64

    with pytest.raises(
        RuntimeError,
        match="binary, CMake cache, or linked-library provenance changed",
    ):
        runner.require_unchanged_build_provenance(discovery, before, after)


def test_q0_build_provenance_must_match_discovery_binary_hash():
    runner = load_runner()
    build = {
        "binary_sha256": "1" * 64,
        "cmake_cache": {"sha256": "2" * 64},
        "linked_library_manifest_sha256": "3" * 64,
    }

    with pytest.raises(RuntimeError, match="different test-binary bytes"):
        runner.require_unchanged_build_provenance(
            {"physics_binary_sha256": "4" * 64},
            build,
            build,
        )


def test_q0_execution_controls_distinguish_requests_from_enforced_limits():
    runner = load_runner()
    execution = {
        "mpi_ranks": 1,
        "threads": 2,
        "wall_time_seconds": 3,
        "memory_mib": 4,
        "output_mib": 5,
    }

    record = runner.execution_control_record(execution)

    assert record == {
        "requested_parallelism": {"mpi_ranks": 1, "threads": 2},
        "enforced_resource_limits": {
            "address_space_mib": 4,
            "wall_time_seconds": 3,
            "output_mib": 5,
        },
    }


def test_q0_cli_rejects_closure_before_artifact_or_binary_validation(
    tmp_path,
):
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--requested-claim",
            "q0_closure",
            "--physics-binary",
            str(tmp_path / "missing-physics"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "outside this matrix" in result.stderr
    assert not output.exists()


def test_q0_validate_only_reports_explicit_nonclosure():
    result = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--validate-only"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)

    assert summary["outcome"] == "PASS_PREREQUISITE_NONCLOSURE"
    assert (
        summary["matrix_sha256"] == hashlib.sha256(MATRIX_PATH.read_bytes()).hexdigest()
    )
    assert summary["source_definition_count"] == 20
    assert summary["gtest_count"] == 24
    assert summary["pytest_count"] == 44
    assert summary["open_exit_count"] == 8
    assert summary["wp0_invalid_input_matrix_ci_registered"] is True
    assert summary["q0_campaign_execution_registered"] is False
    assert summary["q0_complete_artifact_archived"] is False
    assert summary["q0_closed"] is False
    assert summary["audit_q0_checkbox_may_be_checked"] is False
