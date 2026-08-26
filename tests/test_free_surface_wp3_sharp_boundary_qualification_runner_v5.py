#!/usr/bin/env python3
"""Focused contract tests for the WP-3 sharp-boundary V5 runner."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    REPOSITORY_ROOT
    / "tests/cases/fluid/"
    "run_free_surface_wp3_sharp_boundary_qualification_v5.py"
)
MATRIX_PATH = (
    REPOSITORY_ROOT
    / "tests/cases/fluid/"
    "free_surface_wp3_sharp_boundary_qualification_matrix_v5.json"
)


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "_focused_wp3_sharp_boundary_v5",
        RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the focused qualification runner")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


class SharpBoundaryV5ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.delta = runner.parse_json_document(MATRIX_PATH)
        self.registry = runner.validate_matrix_contract(
            copy.deepcopy(self.delta)
        )

    def test_effective_matrix_preserves_the_v4_scientific_contract(self) -> None:
        self.assertEqual(self.registry["schema_version"], 5)
        self.assertEqual(self.registry["matrix_id"], runner.EXPECTED_MATRIX_ID)
        self.assertEqual(len(self.registry["implementation_sources"]), 45)
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
        self.assertEqual(self.registry["gates"], runner.EXPECTED_GATES)
        self.assertEqual(
            self.registry["test_discovery_contract"],
            runner.EXPECTED_TEST_DISCOVERY_CONTRACT,
        )

    def test_parent_and_source_dependencies_are_hash_locked(self) -> None:
        runner.validate_frozen_dependencies(
            self.registry, REPOSITORY_ROOT
        )
        observation = runner.observe_implementation_sources(
            self.registry, REPOSITORY_ROOT
        )
        self.assertTrue(observation["all_match"])
        self.assertEqual(observation["inventory_count"], 45)
        self.assertEqual(observation["drift_count"], 0)
        self.assertEqual(
            hashlib.sha256(runner.PARENT_MATRIX_PATH.read_bytes()).hexdigest(),
            runner.EXPECTED_PARENT_MATRIX_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(runner.PARENT_RUNNER_PATH.read_bytes()).hexdigest(),
            runner.EXPECTED_PARENT_RUNNER_SHA256,
        )

    def test_binary_provenance_policy_is_explicit_and_bounded(self) -> None:
        policy = runner.EXPECTED_BINARY_LINK_PROVENANCE_POLICY
        self.assertEqual(policy["command"], "ldd")
        self.assertEqual(policy["timeout_seconds"], 60)
        self.assertEqual(policy["address_and_resident_memory_mib"], 1024)
        self.assertEqual(policy["output_mib"], 4)
        self.assertEqual(
            policy["maximum_qualification_job_memory_fraction"], 0.05
        )
        self.assertEqual(
            runner.strict_runner.BINARY_LINK_PROVENANCE_MEMORY_MIB,
            policy["address_and_resident_memory_mib"],
        )
        self.assertEqual(
            self.registry["resource_safeguards"][
                "binary_link_provenance_policy"
            ],
            policy,
        )

    def test_binary_provenance_policy_mutation_is_rejected(self) -> None:
        changed = copy.deepcopy(self.delta)
        changed["resource_safeguards"][
            "binary_link_provenance_policy"
        ]["address_and_resident_memory_mib"] = 2048
        with self.assertRaisesRegex(
            ValueError, "binary provenance resource policy changed"
        ):
            runner.validate_matrix_contract(changed)

    def test_binary_record_uses_and_records_the_v5_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "test_binary"
            binary.write_bytes(b"binary")
            output = root / "output"
            output.mkdir()
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

            with mock.patch.object(
                runner.strict_runner, "run_monitored", monitored
            ):
                record = runner.binary_record(
                    binary, root, output, "application"
                )

        arguments = observed["arguments"]
        self.assertEqual(arguments[0], ["ldd", str(binary)])
        self.assertEqual(arguments[6], 60)
        self.assertEqual(arguments[7], 1024)
        self.assertEqual(arguments[8], 4)
        self.assertEqual(arguments[9], "direct_serial")
        self.assertEqual(observed["options"], {})
        self.assertEqual(record["outcome"], "PASS")
        self.assertEqual(
            record["linked_library_provenance_policy"],
            runner.EXPECTED_BINARY_LINK_PROVENANCE_POLICY,
        )

    def test_virtual_mapping_reproduces_old_limit_and_passes_new_limit(
        self,
    ) -> None:
        command = [
            sys.executable,
            "-c",
            (
                "import mmap; "
                "region=mmap.mmap(-1,600*1024*1024); "
                "region.close()"
            ),
        ]
        environment = os.environ.copy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_root = root / "old"
            new_root = root / "new"
            old_root.mkdir()
            new_root.mkdir()
            old = runner.strict_runner.run_monitored(
                command,
                environment,
                REPOSITORY_ROOT,
                old_root / "stdout.txt",
                old_root / "stderr.txt",
                old_root,
                20,
                256,
                4,
                "direct_serial",
            )
            new = runner.strict_runner.run_monitored(
                command,
                environment,
                REPOSITORY_ROOT,
                new_root / "stdout.txt",
                new_root / "stderr.txt",
                new_root,
                20,
                1024,
                4,
                "direct_serial",
            )

        self.assertNotEqual(old["return_code"], 0)
        self.assertEqual(old["termination_reason"], None)
        self.assertEqual(new["return_code"], 0)
        self.assertEqual(new["termination_reason"], None)
        self.assertEqual(new["resource_monitoring_outcome"], "PASS")

    def test_policy_is_injected_into_evidence_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "build.json"
            runner.write_json(path, {"outcome": "PASS"})
            document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            document["binary_link_provenance_policy"],
            runner.EXPECTED_BINARY_LINK_PROVENANCE_POLICY,
        )
        self.assertEqual(
            document["implementation_source_commit"],
            runner.EXPECTED_IMPLEMENTATION_SOURCE_COMMIT,
        )

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
            summary["binary_link_provenance_policy"],
            runner.EXPECTED_BINARY_LINK_PROVENANCE_POLICY,
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
        self.assertEqual(binding["binding_schema_version"], 5)
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
