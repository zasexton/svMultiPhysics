#!/usr/bin/env python3
"""Focused contract tests for the WP-3 sharp-boundary V6 runner."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    REPOSITORY_ROOT
    / "tests/cases/fluid/"
    "run_free_surface_wp3_sharp_boundary_qualification_v6.py"
)
MATRIX_PATH = (
    REPOSITORY_ROOT
    / "tests/cases/fluid/"
    "free_surface_wp3_sharp_boundary_qualification_matrix_v6.json"
)


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "_focused_wp3_sharp_boundary_v6",
        RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the focused qualification runner")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


class SharpBoundaryV6ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        runner.strict_runner.run_gtest_group = runner._base_run_gtest_group
        runner._active_test_execution = None
        self.delta = runner.parse_json_document(MATRIX_PATH)
        self.registry = runner.validate_matrix_contract(
            copy.deepcopy(self.delta)
        )

    def tearDown(self) -> None:
        runner.strict_runner.run_gtest_group = runner._base_run_gtest_group
        runner._active_test_execution = None

    @staticmethod
    def _binaries() -> dict[str, Path]:
        return {
            key: Path("/scratch/qualification") / f"test_{key}"
            for key in runner.EXPECTED_BINARY_KEYS
        }

    def test_effective_matrix_changes_only_the_v6_execution_contract(
        self,
    ) -> None:
        self.assertEqual(self.registry["schema_version"], 6)
        self.assertEqual(self.registry["matrix_id"], runner.EXPECTED_MATRIX_ID)
        self.assertEqual(len(self.registry["implementation_sources"]), 47)
        self.assertEqual(len(self.registry["groups"]), 13)
        self.assertEqual(
            sum(len(group["tests"]) for group in self.registry["groups"]),
            80,
        )
        self.assertEqual(len(self.registry["quantitative_evidence"]), 85)
        self.assertEqual(
            sum(
                len(group.get("recorded_properties", []))
                for group in self.registry["groups"]
            ),
            70,
        )
        self.assertEqual(
            self.registry["resource_safeguards"],
            runner._BASE_REGISTRY["resource_safeguards"],
        )
        changed = {
            key
            for key in set(self.registry) & set(runner._BASE_REGISTRY)
            if self.registry[key] != runner._BASE_REGISTRY[key]
        }
        self.assertEqual(
            changed,
            {
                "focused_contract_test",
                "implementation_source_commit",
                "implementation_sources",
                "matrix_id",
                "parent_artifacts",
                "proposed_runner",
                "qualification_bundle_binding",
                "runner_sha256",
                "schema_version",
                "status_reason",
            },
        )
        self.assertEqual(
            set(self.registry) - set(runner._BASE_REGISTRY),
            {"test_execution_contract"},
        )

    def test_parent_and_source_dependencies_are_hash_locked(self) -> None:
        runner.validate_frozen_dependencies(
            self.registry, REPOSITORY_ROOT
        )
        observation = runner.observe_implementation_sources(
            self.registry, REPOSITORY_ROOT
        )
        self.assertTrue(observation["all_match"])
        self.assertEqual(observation["inventory_count"], 47)
        self.assertEqual(observation["drift_count"], 0)
        for path, expected in (
            (runner.PARENT_MATRIX_PATH, runner.EXPECTED_PARENT_MATRIX_SHA256),
            (runner.PARENT_RUNNER_PATH, runner.EXPECTED_PARENT_RUNNER_SHA256),
            (
                runner.PARENT_FOCUSED_TEST_PATH,
                runner.EXPECTED_PARENT_FOCUSED_TEST_SHA256,
            ),
            (
                runner.EXECUTION_HELPER_PATH,
                runner.EXPECTED_EXECUTION_HELPER_SHA256,
            ),
            (
                runner.EXECUTION_HELPER_TEST_PATH,
                runner.EXPECTED_EXECUTION_HELPER_TEST_SHA256,
            ),
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), expected
                )

    def test_execution_contract_matches_discovery_and_group_routes(
        self,
    ) -> None:
        execution_contract = self.registry["test_execution_contract"]
        discovery_contract = self.registry["test_discovery_contract"]
        for field in (
            "mpi_single_rank_arguments",
            "mpi_single_rank_binary_keys",
            "direct_binary_keys",
        ):
            self.assertEqual(
                execution_contract[field], discovery_contract[field]
            )
        physics_ranks = {
            group["mpi_ranks"]
            for group in self.registry["groups"]
            if group["binary"] == "physics"
        }
        self.assertIn(1, physics_ranks)
        self.assertTrue(any(ranks > 1 for ranks in physics_ranks))
        execution = runner.create_test_execution(
            self._binaries(),
            self.registry,
            Path("/opt/mpi/bin/mpiexec"),
        )
        self.assertEqual(execution.contract(), execution_contract)

    def test_discovery_creation_installs_the_execution_router(self) -> None:
        binaries = self._binaries()
        launcher = Path("/opt/mpi/bin/mpiexec")
        discovery = runner.create_test_discovery(
            binaries,
            self.registry,
            launcher,
        )
        self.assertEqual(
            discovery.contract(), self.registry["test_discovery_contract"]
        )
        self.assertIs(
            runner.strict_runner.run_gtest_group,
            runner._active_test_execution,
        )
        self.assertEqual(
            runner._active_test_execution.contract(),
            self.registry["test_execution_contract"],
        )

    def test_execution_contract_mutations_are_rejected(self) -> None:
        mutations = (
            ("mpi_single_rank_arguments", ["-n", "1"]),
            ("mpi_single_rank_binary_keys", ["physics"]),
            ("direct_binary_keys", ["geometry"]),
            (
                "inherited_scheduler_process_count_policy",
                "inherit_scheduler_process_count",
            ),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.delta)
                changed["test_execution_contract"][field] = value
                with self.assertRaisesRegex(
                    ValueError, "test execution contract changed"
                ):
                    runner.validate_matrix_contract(changed)
        changed = copy.deepcopy(self.delta)
        changed["test_execution_contract"]["mpi_single_rank_monitoring"][
            "required_simultaneous_process_samples"
        ] = 1
        with self.assertRaisesRegex(
            ValueError, "test execution contract changed"
        ):
            runner.validate_matrix_contract(changed)

    def test_execution_contract_is_injected_into_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "build.json"
            runner.write_json(path, {"outcome": "PASS"})
            document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            document["test_execution_contract"],
            runner.EXPECTED_TEST_EXECUTION_CONTRACT,
        )
        self.assertEqual(
            document["binary_link_provenance_policy"],
            runner._parent.EXPECTED_BINARY_LINK_PROVENANCE_POLICY,
        )
        self.assertEqual(
            document["implementation_source_commit"],
            runner.EXPECTED_IMPLEMENTATION_SOURCE_COMMIT,
        )

    def test_resource_safeguard_mutation_is_rejected(self) -> None:
        changed = copy.deepcopy(self.delta)
        changed["resource_safeguards"]["qualification_job_tasks"] = 9
        with self.assertRaisesRegex(
            ValueError, "qualification resource safeguards changed"
        ):
            runner.validate_matrix_contract(changed)

    def test_duplicate_json_key_is_rejected(self) -> None:
        malformed = (
            b'{"runner_sha256":"'
            + b"0" * 64
            + b'","runner_sha256":"'
            + b"0" * 64
            + b'"}'
        )
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            runner.normalized_registry_bytes(malformed)

    def test_validate_only_summary_keeps_later_work_open(self) -> None:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            return_code = runner.main(["--validate-only"])
        self.assertEqual(return_code, 0)
        summary = json.loads(stream.getvalue())
        self.assertEqual(summary["group_count"], 13)
        self.assertEqual(summary["test_count"], 80)
        self.assertEqual(summary["quantitative_evidence_gate_count"], 85)
        self.assertEqual(summary["recorded_property_gate_count"], 70)
        self.assertEqual(summary["open_outcomes"]["wp7"], "OPEN")
        self.assertEqual(summary["open_outcomes"]["q1"], "OPEN")
        self.assertEqual(
            summary["test_execution_contract"],
            runner.EXPECTED_TEST_EXECUTION_CONTRACT,
        )

    def test_bundle_binding_matches_lifecycle_state(self) -> None:
        status = self.delta["status"]
        if status == runner.DRAFT_MATRIX_STATUS:
            self.assertEqual(
                self.delta["qualification_bundle_binding"],
                runner._draft_bundle_binding(),
            )
            self.assertEqual(
                self.delta["runner_sha256"],
                runner.RUNNER_SHA256_ZERO_SENTINEL,
            )
            self.assertEqual(
                runner.EXPECTED_NORMALIZED_REGISTRY_SHA256,
                runner.RUNNER_SHA256_ZERO_SENTINEL,
            )
            self.assertEqual(
                runner.EXPECTED_FOCUSED_TEST_SHA256,
                runner.RUNNER_SHA256_ZERO_SENTINEL,
            )
            return

        self.assertEqual(status, runner.EXECUTABLE_MATRIX_STATUS)
        self.assertEqual(
            self.delta["qualification_bundle_binding"],
            runner._frozen_bundle_binding(),
        )
        self.assertEqual(
            runner.normalized_registry_sha256(MATRIX_PATH),
            runner.EXPECTED_NORMALIZED_REGISTRY_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(RUNNER_PATH.read_bytes()).hexdigest(),
            self.delta["runner_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            runner.EXPECTED_FOCUSED_TEST_SHA256,
        )

    def test_frozen_bundle_history_when_promoted(self) -> None:
        if self.delta["status"] == runner.DRAFT_MATRIX_STATUS:
            with self.assertRaisesRegex(ValueError, "requires a frozen"):
                runner.validate_frozen_qualification_bundle(self.registry)
            return
        binding = runner.validate_frozen_qualification_bundle(
            self.registry,
            MATRIX_PATH,
            REPOSITORY_ROOT,
            RUNNER_PATH,
        )
        self.assertEqual(binding["binding_schema_version"], 6)
        self.assertEqual(
            binding["bundle_parent_commit"],
            runner.EXPECTED_IMPLEMENTATION_SOURCE_COMMIT,
        )
        self.assertEqual(
            binding["bundle_changed_paths"],
            sorted(runner.EXPECTED_BUNDLE_PATHS),
        )
        self.assertEqual(len(binding["artifacts"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
