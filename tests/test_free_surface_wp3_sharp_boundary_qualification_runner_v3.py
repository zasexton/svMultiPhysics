#!/usr/bin/env python3
"""Focused contract tests for the WP-3 sharp-boundary V3 runner."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    REPOSITORY_ROOT
    / "tests/cases/fluid/"
    "run_free_surface_wp3_sharp_boundary_qualification_v3.py"
)
MATRIX_PATH = (
    REPOSITORY_ROOT
    / "tests/cases/fluid/"
    "free_surface_wp3_sharp_boundary_qualification_matrix_v3.json"
)


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "_focused_wp3_sharp_boundary_v3",
        RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the focused qualification runner")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


class SharpBoundaryV3ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = runner.parse_json_document(MATRIX_PATH)

    def test_matrix_contract_has_complete_frozen_inventory(self) -> None:
        validated = runner.validate_matrix_contract(
            copy.deepcopy(self.registry)
        )
        self.assertEqual(validated["schema_version"], 3)
        self.assertEqual(validated["matrix_id"], runner.EXPECTED_MATRIX_ID)
        self.assertEqual(validated["gates"], runner.EXPECTED_GATES)
        self.assertEqual(len(validated["implementation_sources"]), 43)
        self.assertEqual(len(validated["groups"]), 13)
        self.assertEqual(
            sum(len(group["tests"]) for group in validated["groups"]),
            80,
        )
        self.assertEqual(
            len(
                {
                    test
                    for group in validated["groups"]
                    for test in group["tests"]
                }
            ),
            80,
        )
        self.assertEqual(len(validated["quantitative_evidence"]), 85)
        self.assertEqual(
            sum(
                len(group.get("recorded_properties", []))
                for group in validated["groups"]
            ),
            70,
        )
        for group in validated["groups"]:
            self.assertEqual(
                group["gtest_output_copies"],
                runner.EXPECTED_GTEST_OUTPUT_COPIES[group["binary"]],
            )

    def test_group_and_evidence_digests_are_independent_locks(self) -> None:
        self.assertEqual(
            runner._canonical_sha256(self.registry["groups"]),
            runner.EXPECTED_GROUPS_SHA256,
        )
        self.assertEqual(
            runner._canonical_sha256(
                self.registry["quantitative_evidence"]
            ),
            runner.EXPECTED_QUANTITATIVE_EVIDENCE_SHA256,
        )
        self.assertEqual(
            runner._canonical_sha256(self.registry["closure_contract"]),
            runner.EXPECTED_CLOSURE_CONTRACT_SHA256,
        )
        self.assertEqual(
            runner._canonical_sha256(
                self.registry["implementation_sources"]
            ),
            runner.EXPECTED_IMPLEMENTATION_SOURCES_SHA256,
        )

    def test_operator_inventory_covers_all_supported_routes(self) -> None:
        operators = {
            entry["operator"]
            for entry in self.registry["operator_disposition_contract"]
        }
        self.assertEqual(operators, runner.EXPECTED_OPERATOR_NAMES)
        self.assertEqual(len(operators), 12)
        for entry in self.registry["operator_disposition_contract"]:
            self.assertEqual(
                entry["cut_active_disposition"],
                "generated_active_boundary",
            )
            self.assertEqual(
                entry["dry_face_disposition"],
                "exact_zero",
            )
            self.assertEqual(
                entry["missing_sharp_domain_disposition"],
                "hard_error",
            )

    def test_critical_end_to_end_selectors_are_frozen(self) -> None:
        tests = {
            test
            for group in self.registry["groups"]
            for test in group["tests"]
        }
        required = {
            (
                "ApplicationDriverLevelSetWorkflows."
                "NativeCertifiedManufacturedChannelTracksSharpBoundaryWork"
            ),
            (
                "ApplicationDriverLevelSetWorkflowsMPI."
                "NativeCertifiedManufacturedChannelIsRepartitionIndependent"
            ),
            (
                "FreeSurfaceSharpBoundaryOperators."
                "AdditionalPspgBoundaryFormsUseGeneratedWetWallMeasure"
            ),
            (
                "FreeSurfaceSharpBoundaryOperatorsMPI."
                "AdditionalPspgBoundaryFormsArePartitionIndependent"
            ),
            (
                "GeneratedBoundaryAggregateTraceCertificate."
                "FullActiveBoundaryInAggregateUsesClosedCellPatch"
            ),
            (
                "DenseLinearAlgebra."
                "ExactFactorizedBoundUsesLinearPositiveScaleMultipliers"
            ),
            (
                "LevelSetInterfaceBuilder."
                "PublishesAlignedFacetFromRequestedParentSideExactlyOnce"
            ),
            (
                "GeneratedActiveBoundaryDomain."
                "SnapshotPrunesBoundaryTraceWithoutRetainedParentVolume"
            ),
            (
                "LevelSetInterfaceLifecycle."
                "QuadraturePolicyKeyChangesWithBackendOptions"
            ),
            "LevelSetRestart.CapturesFieldAndGeneratedInterfaceRecords",
        }
        self.assertTrue(required.issubset(tests))
        self.assertNotIn(
            (
                "FreeSurfaceSharpBoundaryOperators."
                "NitscheTraceScalingProducesFiniteSampledMargins"
            ),
            tests,
        )

    def test_source_manifest_matches_recorded_commit(self) -> None:
        runner.validate_frozen_dependencies(
            self.registry,
            REPOSITORY_ROOT,
        )
        observation = runner.observe_implementation_sources(
            self.registry,
            REPOSITORY_ROOT,
        )
        self.assertTrue(observation["all_match"])
        self.assertEqual(observation["inventory_count"], 43)
        self.assertEqual(observation["drift_count"], 0)
        self.assertEqual(observation["missing_count"], 0)

    def test_normalization_changes_only_runner_hash_value(self) -> None:
        raw = MATRIX_PATH.read_bytes()
        matches = list(runner._RUNNER_SHA256_FIELD_PATTERN.finditer(raw))
        self.assertEqual(len(matches), 1)
        start, end = matches[0].span(2)
        alternate = raw[:start] + b"1" * 64 + raw[end:]
        self.assertEqual(
            runner.normalized_registry_bytes(raw),
            runner.normalized_registry_bytes(alternate),
        )
        whitespace_mutation = raw + b"\n"
        self.assertNotEqual(
            hashlib.sha256(
                runner.normalized_registry_bytes(raw)
            ).hexdigest(),
            hashlib.sha256(
                runner.normalized_registry_bytes(whitespace_mutation)
            ).hexdigest(),
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

    def test_group_selector_mutation_is_rejected(self) -> None:
        changed = copy.deepcopy(self.registry)
        changed["groups"][0]["tests"][0] = (
            "GeneratedActiveBoundaryDomain.RejectsNonfiniteRequestScalars"
        )
        with self.assertRaisesRegex(
            ValueError,
            "qualification group contract changed",
        ):
            runner.validate_matrix_contract(changed)

    def test_operator_disposition_mutation_is_rejected(self) -> None:
        changed = copy.deepcopy(self.registry)
        changed["operator_disposition_contract"][0][
            "cut_active_disposition"
        ] = "physical_boundary"
        with self.assertRaisesRegex(
            ValueError,
            "operator disposition contract changed",
        ):
            runner.validate_matrix_contract(changed)

    def test_factorized_gate_mutation_is_rejected(self) -> None:
        changed = copy.deepcopy(self.registry)
        for entry in changed["quantitative_evidence"]:
            if entry["property"] == (
                "native_channel_maximum_factorized_input_dimension"
            ):
                entry["threshold"] = 33
                break
        else:
            self.fail("factorized dimension gate is missing")
        with self.assertRaisesRegex(
            ValueError,
            "quantitative evidence contract changed",
        ):
            runner.validate_matrix_contract(changed)

    def test_fresh_configure_route_is_exact_and_rejects_drift(self) -> None:
        runner.validate_matrix_contract(copy.deepcopy(self.registry))
        cmake = "/share/software/user/open/cmake/3.31.4/bin/cmake"
        for relative_home, definitions in (
            runner._active_fresh_configure_definitions.items()
        ):
            command = [
                cmake,
                "-S",
                str(REPOSITORY_ROOT / relative_home),
                "-B",
                f"/scratch/users/zsexton/focused-{len(relative_home)}",
            ]
            locked = runner._locked_fresh_configure_command(
                command,
                REPOSITORY_ROOT,
            )
            self.assertEqual(locked[0], cmake)
            self.assertEqual(locked[1], "--fresh")
            self.assertEqual(
                tuple(locked[2 : 2 + len(definitions)]),
                definitions,
            )
        with self.assertRaisesRegex(ValueError, "conflicts with locked"):
            runner._locked_fresh_configure_command(
                [
                    cmake,
                    "-S",
                    str(REPOSITORY_ROOT / "Code"),
                    "-B",
                    "/scratch/users/zsexton/focused-bad",
                    "-DCMAKE_BUILD_TYPE=Debug",
                ],
                REPOSITORY_ROOT,
            )
        with self.assertRaisesRegex(ValueError, "not recognized"):
            runner._locked_fresh_configure_command(
                [
                    cmake,
                    "-S",
                    str(REPOSITORY_ROOT / "tests"),
                    "-B",
                    "/scratch/users/zsexton/focused-unknown",
                ],
                REPOSITORY_ROOT,
            )

    def test_scheduler_allocation_contract_rejects_resource_drift(self) -> None:
        safeguards = self.registry["resource_safeguards"]
        environment = {
            "SLURM_JOB_ACCOUNT": "amarsden",
            "SLURM_JOB_PARTITION": "amarsden",
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_NTASKS": "8",
            "SLURM_CPUS_PER_TASK": "1",
            "SLURM_CPUS_ON_NODE": "8",
            "SLURM_MEM_PER_NODE": "20480",
        }
        observed = runner._validate_scheduler_allocation(
            safeguards, environment
        )
        self.assertEqual(observed["nodes"], 1)
        self.assertEqual(observed["tasks"], 8)
        self.assertEqual(observed["memory_mib"], 20480)
        mutations = {
            "SLURM_JOB_ACCOUNT": "different",
            "SLURM_JOB_PARTITION": "different",
            "SLURM_JOB_NUM_NODES": "2",
            "SLURM_NTASKS": "1",
            "SLURM_CPUS_PER_TASK": "2",
            "SLURM_CPUS_ON_NODE": "7",
            "SLURM_MEM_PER_NODE": "40960",
        }
        for name, value in mutations.items():
            changed = dict(environment)
            changed[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                runner._validate_scheduler_allocation(safeguards, changed)

    def test_claim_boundary_rejects_broader_outcomes(self) -> None:
        claim, validate_only, remaining = runner.requested_claim(
            ["--validate-only"]
        )
        self.assertEqual(claim, runner.ACCEPTED_CLAIM)
        self.assertTrue(validate_only)
        self.assertEqual(remaining, [])
        for rejected in runner.EXPECTED_CLOSURE_REQUEST_POLICY[
            "rejected_claims"
        ]:
            with self.assertRaisesRegex(ValueError, "outside this matrix"):
                runner.requested_claim(
                    ["--requested-claim", rejected, "--validate-only"]
                )

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

    def test_bundle_binding_matches_lifecycle_state(self) -> None:
        status = self.registry["status"]
        if status == runner.DRAFT_MATRIX_STATUS:
            self.assertEqual(
                self.registry["qualification_bundle_binding"],
                runner._draft_bundle_binding(),
            )
            self.assertEqual(
                self.registry["runner_sha256"],
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
        else:
            self.assertEqual(status, runner.EXECUTABLE_MATRIX_STATUS)
            self.assertEqual(
                self.registry["qualification_bundle_binding"],
                runner._frozen_bundle_binding(),
            )
            self.assertEqual(
                runner.normalized_registry_sha256(MATRIX_PATH),
                runner.EXPECTED_NORMALIZED_REGISTRY_SHA256,
            )
            self.assertEqual(
                hashlib.sha256(RUNNER_PATH.read_bytes()).hexdigest(),
                self.registry["runner_sha256"],
            )
            self.assertEqual(
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                runner.EXPECTED_FOCUSED_TEST_SHA256,
            )

    def test_frozen_bundle_history_when_promoted(self) -> None:
        if self.registry["status"] == runner.DRAFT_MATRIX_STATUS:
            with self.assertRaisesRegex(ValueError, "requires a frozen"):
                runner.validate_frozen_qualification_bundle(self.registry)
            return
        binding = runner.validate_frozen_qualification_bundle(
            self.registry,
            MATRIX_PATH,
            REPOSITORY_ROOT,
            RUNNER_PATH,
        )
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
