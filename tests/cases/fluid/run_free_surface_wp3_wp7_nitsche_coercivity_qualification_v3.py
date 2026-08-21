#!/usr/bin/env python3
"""Run the V3 accepted-state symmetric-Nitsche prerequisite matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode = True


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIRECTORY = SCRIPT_PATH.parent
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_REGISTRY = SCRIPT_PATH.with_name(
    "free_surface_wp3_wp7_nitsche_coercivity_qualification_matrix_v3.json"
)
V2_RUNNER_PATH = SCRIPT_PATH.with_name(
    "run_free_surface_wp3_wp7_nitsche_coercivity_qualification_v2.py"
)
SHARED_RUNNER_PATH = SCRIPT_PATH.with_name(
    "run_free_surface_wp2_geometry_qualification.py"
)
V2_RUNNER_SHA256 = (
    "8d995e70e77e27e3e9b7c15401150cc34e673ecd273d88bc1f6f55af60ac245d"
)
SHARED_RUNNER_SHA256 = (
    "5387dd19618139aeee45bb6f3c77f27fd8b26ce28713d221a866e1eea4662037"
)


def _load_v2_runner() -> Any:
    if hashlib.sha256(V2_RUNNER_PATH.read_bytes()).hexdigest() != (
        V2_RUNNER_SHA256
    ):
        raise RuntimeError("V2 qualification parent bytes changed")
    if hashlib.sha256(SHARED_RUNNER_PATH.read_bytes()).hexdigest() != (
        SHARED_RUNNER_SHA256
    ):
        raise RuntimeError("shared qualification base bytes changed")
    specification = importlib.util.spec_from_file_location(
        "_free_surface_wp3_wp7_trace_v3_parent",
        V2_RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the V2 qualification parent")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


_parent = _load_v2_runner()
_parent_validate_contract = _parent.validate_v2_contract
_parent_parse_trace_evidence = _parent.parse_trace_evidence
_parent_write_json = _parent.write_json
_parent_main = _parent.main
_parent_observe_implementation_sources = (
    _parent.observe_implementation_sources
)
_parent_require_execution_resource_preflight = (
    _parent.require_execution_resource_preflight
)
_parent_run_build_phase = _parent.run_build_phase
_shared_untracked_source_record = (
    _parent.strict_runner.untracked_source_record
)
_V2_TRACE_CASE_FIELDS = frozenset(_parent.EXPECTED_TRACE_CASE_FIELDS)
_V2_TRACE_SUMMARY_FIELDS = frozenset(_parent.EXPECTED_TRACE_SUMMARY_FIELDS)

EXPECTED_NORMALIZED_REGISTRY_SHA256 = (
    "0384998c268353f3ffb32b9cc24567d6be0695bcd8ec4082665ebc1dd67a60c4"
)
RUNNER_SHA256_ZERO_SENTINEL = "0" * 64
EXPECTED_SHARED_RUNNER_SHA256 = (
    SHARED_RUNNER_SHA256
)
EXPECTED_V2_PARENT_SHA256 = {
    (
        "tests/cases/fluid/"
        "free_surface_wp3_wp7_nitsche_coercivity_qualification_matrix_v2.json"
    ): "a35818da1455763e630e243baf16b852a6d6c6b7a2112b72e517e6181d4fb407",
    (
        "tests/cases/fluid/"
        "run_free_surface_wp3_wp7_nitsche_coercivity_qualification_v2.py"
    ): V2_RUNNER_SHA256,
    (
        "Documentation/"
        "free_surface_wp3_wp7_symmetric_nitsche_coercivity_method_v2.md"
    ): "3b951a76a0694a9b6d1bc5fa859f036dc9173ee40fe41ca43abe3b27f101fd39",
    (
        "Documentation/qualification_logs/"
        "free_surface_wp3_wp7_nitsche_coercivity_v2_20260818_e9ae9f82/"
        "checksums.txt"
    ): "b251010050632b8cf63002356add2d192cce20522e03e9cd35bcf5b16f1f1ef6",
}

EXPECTED_MATRIX_ID = (
    "free_surface_wp3_wp7_symmetric_nitsche_"
    "accepted_state_floor_prerequisite_v3"
)
EXPECTED_MATRIX_STATUS = "DRAFT_UNEXECUTED"
EXECUTABLE_MATRIX_STATUS = "FROZEN_BEFORE_EXECUTION"
EXPECTED_CHECKED_IN_MATRIX_STATUS = EXECUTABLE_MATRIX_STATUS
ALLOWED_MATRIX_STATUSES = {
    EXPECTED_MATRIX_STATUS,
    EXECUTABLE_MATRIX_STATUS,
}
EXPECTED_DRAFT_SOURCE_HASH_STATUS = "DRAFT_OBSERVED_NOT_FROZEN"
EXPECTED_FROZEN_SOURCE_HASH_STATUS = "FROZEN"
EXPECTED_IMPLEMENTATION_SOURCE_COMMIT = (
    "9ee2b96da03a6e350f580a4bcc9ac098ae979eb5"
)
EXPECTED_WORK_PACKAGE = "WP-3/WP-7"
EXPECTED_MATCHING_DERIVATION = (
    "Documentation/"
    "free_surface_wp3_wp7_symmetric_nitsche_coercivity_method_v3.md"
)
EXPECTED_PROPOSED_RUNNER = (
    "tests/cases/fluid/"
    "run_free_surface_wp3_wp7_nitsche_coercivity_qualification_v3.py"
)
EXPECTED_MATRIX_PATH = (
    "tests/cases/fluid/"
    "free_surface_wp3_wp7_nitsche_coercivity_qualification_matrix_v3.json"
)
EXPECTED_FOCUSED_TEST_PATH = (
    "tests/"
    "test_free_surface_wp3_wp7_nitsche_coercivity_qualification_runner_v3.py"
)
EXPECTED_FOCUSED_TEST_SHA256 = (
    "a711e17e11122fb3700f2cdd48aeefe7e795d1dff3172dfea57e05682be4314a"
)
EXPECTED_BUNDLE_COMMIT_RESOLUTION = (
    "unique_direct_child_of_implementation_source_commit_on_validation_"
    "HEAD_ancestry_matching_exact_paths_and_frozen_blobs"
)
EXPECTED_FROZEN_BUNDLE_AUTHORITY = (
    "reciprocal_SHA256_plus_canonical_bundle_commit_history"
)
EXPECTED_DRAFT_QUALIFICATION_BUNDLE_BINDING = {
    "authority": (
        "PREFREEZE_NO_RECIPROCAL_AUTHORITY_UNTIL_V3_HASH_LOCKS_FINALIZED"
    ),
    "matrix_sha256_source": (
        "future_v3_runner_embedded_normalized_matrix_SHA256"
    ),
    "matrix_hash_normalization": (
        "replace_the_unique_runner_sha256_64_lowercase_hex_JSON_value_with_"
        "64_ASCII_zero_digits"
    ),
    "runner_sha256_source": (
        "matrix_runner_sha256_zero_sentinel_until_v3_runner_freeze"
    ),
    "focused_test_sha256_source": (
        "future_v3_runner_embedded_focused_test_sha256"
    ),
    "bundle_commit_resolution": EXPECTED_BUNDLE_COMMIT_RESOLUTION,
    "exact_bundle_commit_blobs_required": [
        EXPECTED_MATRIX_PATH,
        EXPECTED_PROPOSED_RUNNER,
        EXPECTED_FOCUSED_TEST_PATH,
    ],
    "bundle_commit_must_have_exactly_one_parent": True,
    "bundle_commit_parent_must_equal_implementation_source_commit": True,
    "bundle_commit_changed_paths_must_equal_exact_bundle_commit_blobs_required": (
        True
    ),
    "bundle_commit_blobs_must_match_checked_out_frozen_bytes": True,
    "validation_HEAD_must_descend_from_bundle_commit": True,
    "execution_HEAD_must_equal_bundle_commit": True,
}
EXPECTED_FROZEN_QUALIFICATION_BUNDLE_BINDING = {
    "authority": EXPECTED_FROZEN_BUNDLE_AUTHORITY,
    "matrix_sha256_source": "runner_embedded_normalized_matrix_SHA256",
    "matrix_hash_normalization": (
        "replace_the_unique_runner_sha256_64_lowercase_hex_JSON_value_with_"
        "64_ASCII_zero_digits"
    ),
    "runner_sha256_source": "matrix_runner_sha256",
    "focused_test_sha256_source": (
        "runner_embedded_focused_test_sha256"
    ),
    "bundle_commit_resolution": EXPECTED_BUNDLE_COMMIT_RESOLUTION,
    "exact_bundle_commit_blobs_required": [
        EXPECTED_MATRIX_PATH,
        EXPECTED_PROPOSED_RUNNER,
        EXPECTED_FOCUSED_TEST_PATH,
    ],
    "bundle_commit_must_have_exactly_one_parent": True,
    "bundle_commit_parent_must_equal_implementation_source_commit": True,
    "bundle_commit_changed_paths_must_equal_exact_bundle_commit_blobs_required": (
        True
    ),
    "bundle_commit_blobs_must_match_checked_out_frozen_bytes": True,
    "validation_HEAD_must_descend_from_bundle_commit": True,
    "execution_HEAD_must_equal_bundle_commit": True,
}
EXPECTED_QUALIFICATION_BUNDLE_BINDINGS = {
    EXPECTED_MATRIX_STATUS: EXPECTED_DRAFT_QUALIFICATION_BUNDLE_BINDING,
    EXECUTABLE_MATRIX_STATUS: EXPECTED_FROZEN_QUALIFICATION_BUNDLE_BINDING,
}
EXPECTED_STATUS_REASONS = {
    EXPECTED_MATRIX_STATUS: (
        "The implementation inventory is frozen at the recorded clean source "
        "commit, but the V3 runner and reciprocal matrix/runner SHA-256 locks "
        "have not yet been finalized. No V3 qualification evidence has been "
        "executed."
    ),
    EXECUTABLE_MATRIX_STATUS: (
        "The implementation inventory is frozen at the recorded clean source "
        "commit, and the reciprocal V3 matrix/runner SHA-256 locks are "
        "finalized for a clean qualification execution HEAD. No V3 "
        "qualification evidence has been executed."
    ),
}
EXPECTED_PROMOTION_REQUIREMENTS = [
    "author and review the exact V3 runner against this matrix",
    "revalidate discovery of all 33 named tests and the disabled-test "
    "execution route",
    "finalize the frozen lifecycle fields and reciprocal qualification-bundle "
    "contract while retaining the zero runner SHA-256 sentinel for normalized "
    "matrix hashing",
    "compute and embed the normalized finalized matrix SHA-256 in the V3 "
    "runner",
    "compute the finalized V3 runner SHA-256, replace the matrix zero "
    "sentinel, and verify that the normalized matrix digest is unchanged",
    "commit the exact V3 matrix, runner, and focused test bytes together in "
    "the unique three-path bundle commit whose sole parent is the "
    "implementation source commit",
    "resolve that immutable bundle commit on validation HEAD ancestry and "
    "require the detached execution HEAD to equal it exactly",
    "validate frozen implementation sources from their recorded commit blobs "
    "without rebinding historical validation to later working-tree bytes",
    "execute from the canonical source root in an isolated detached linked "
    "worktree whose common Git directory is external to that source root",
    "require zero ignored source paths, suppress Python bytecode writes, and "
    "freshly configure CMake with the exact locked per-source-home cache "
    "definitions before the serialized clean build",
    "execute all five groups without failure, error, skip, or disabled result",
    "bind structured evidence and final provenance to the reciprocally locked "
    "normalized matrix hash, exact runner hash, qualification bundle commit, "
    "and frozen source hashes",
]

METHOD_ENERGY_FLOOR = 0.25
SAFE_GROUPED_SYMMETRIC_RATIO_CAP = float.fromhex(
    "0x1.1fffffffffffep-1"
)
ACCEPTED_CLAIM = "accepted_state_coercivity_policy_prerequisite"
UNIFORM_BOUND_STATUS = "ENFORCED_ACCEPTED_STATE_FLOOR"

EXPECTED_SCOPE = (
    "Exact aggregate-trace certification plus a predeclared symmetric-Nitsche "
    "energy floor c*=1/4 for every accepted current state of the supported "
    "production Navier-Stokes viscous/Nitsche subform, whose module supplies "
    "the bulk viscous energy K. The generic FE gate is conditional on an "
    "installed caller-supplied coercive bulk form and does not independently "
    "prove that bulk hypothesis. This matrix does not close FSR-16, FSR-07, "
    "WP-3, WP-7, or Q1, and does not prove that every cut or mesh-family state "
    "will be accepted."
)
EXPECTED_CLOSURE_REQUEST_POLICY = {
    "accepted_claim": ACCEPTED_CLAIM,
    "rejected_claims": [
        "fsr16_closure",
        "fsr07_closure",
        "wp3_closure",
        "wp7_closure",
        "wp3_wp7_joint_closure",
        "q1_closure",
        "unconditional_cut_and_mesh_family_acceptance",
    ],
    "diagnostic": (
        "For the production Navier-Stokes viscous/Nitsche subform whose module "
        "supplies K, the exact current-state gate proves a uniform positive "
        "energy floor only for accepted states. The generic FE gate remains "
        "conditional on a caller-supplied coercive bulk form. Neither path "
        "supplies an acceptance or penalty-existence theorem, the broader "
        "operator and element-family envelope, mixed stability, convergence, "
        "conditioning, solver robustness, or physical campaign evidence "
        "required for closure."
    ),
}
EXPECTED_DISPOSITION = {
    "fsr16_closed": False,
    "fsr07_closed": False,
    "wp3_closed": False,
    "wp7_closed": False,
    "q1_closed": False,
    "accepted_state_uniform_coercivity_floor_is_qualification_target": True,
    "matrix_itself_asserts_executed_qualification_evidence": False,
    "unconditional_cut_and_mesh_family_acceptance_established": False,
}
EXPECTED_OPEN_OUTCOMES = copy.deepcopy(_parent.EXPECTED_OPEN_OUTCOMES)
EXPECTED_MODEL_ENVELOPE = (
    "production_Navier-Stokes_viscous_plus_generated-boundary_symmetric-"
    "Nitsche_subform_with_module-supplied_bulk_K_on_reference-frame_affine_"
    "P1_product_velocity_linear_triangles_or_tetrahedra_constant_positive_"
    "finite_viscosity_current_trace-eligible_small-cut_aggregation_and_"
    "predeclared_energy_floor;generic_FE_gate_is_conditional_on_an_installed_"
    "caller-supplied_coercive_bulk_form"
)
EXPECTED_METHOD_LIMITATIONS = [
    (
        "The positive value c*=1/4 targets a uniform contract over accepted "
        "current states of the production Navier-Stokes viscous/Nitsche "
        "subform whose module supplies K. The generic FE gate remains "
        "conditional on a caller-supplied coercive bulk form, and no pre-"
        "execution matrix entry is qualification evidence or an existence "
        "theorem guaranteeing that every cut or mesh-family state will pass "
        "the fixed floor gate."
    ),
    (
        "Quadrilateral and hexahedral Q1, higher order, non-affine geometry, "
        "spatially variable or constitutive viscosity, and current-frame "
        "geometry are not certified."
    ),
    (
        "The supported sharp route is generated-boundary symmetric velocity "
        "Nitsche on one physical marker with finalized trace-eligible "
        "aggregation; traction, pressure flux, outflow, physical Robin, "
        "Navier slip, and other boundary operators remain outside the "
        "certificate."
    ),
    (
        "The two-rank evidence covers one exact cross-owner aggregate plus "
        "policy-mismatch and distributed-subfloor failures. It does not "
        "qualify four-or-more ranks, repartition sweeps, or rank-count "
        "invariance."
    ),
    (
        "The 108-case diagnostic verifies the declared direct risk gate and "
        "compares the accepted floor with a finite sampled generalized "
        "eigenspectrum. It does not establish mixed velocity-pressure "
        "stability, conditioning, convergence, or an acceptance theorem "
        "beyond the exercised states."
    ),
    (
        "The formulation installer verifies one exact canonical generated-"
        "interface symmetric-gradient route anchor and publishes its sealed "
        "effective penalty, while the policy floor is validated and installed "
        "separately. Certification consumes those values and never selects or "
        "mutates the penalty. Floor bits bind the policy signature and current "
        "certificate-cache digest but do not bind the emitted route/form-"
        "binding digest or exact aggregate-trace certificate digest. The "
        "binding-metadata digest is not a digest of the entire mixed assembled "
        "form, and this path must not rescale the physical Navier-slip "
        "coefficient mu/slip_length."
    ),
]
EXPECTED_CASE_AXES = copy.deepcopy(_parent.EXPECTED_CASE_AXES)
EXPECTED_CASE_AXES.update(
    {
        "required_minimum_energy_ratio": METHOD_ENERGY_FLOOR,
        "downward_safe_group_ratio_cap": (
            SAFE_GROUPED_SYMMETRIC_RATIO_CAP
        ),
        "downward_safe_trace_upper_bound_limit_hex": (
            "0x1.afffffffffffdp+2"
        ),
        "downward_safe_trace_upper_bound_limit": (
            12.0 * SAFE_GROUPED_SYMMETRIC_RATIO_CAP
        ),
    }
)

EXPECTED_TRACE_CONTRACT = copy.deepcopy(_parent.EXPECTED_TRACE_CONTRACT)
EXPECTED_TRACE_CONTRACT.update(
    {
        "symmetric_acceptance": (
            "R_op<=downward_safe_binary64_cap_for_(1-c)^2_with_"
            "predeclared_c_in_(0,1)"
        ),
        "finite_space_energy_ratio_lower_bound": (
            "exact_configured_floor_c_after_the_direct_outward-risk_gate"
        ),
        "unsymmetric_contract": (
            "revision_bound_continuity_diagnostic_with_exact_zero_policy_"
            "floor_and_no_symmetric_energy_bound"
        ),
        "cache_binding": (
            "floor_bits_bind_the_policy_signature_and_current_certificate-"
            "cache_digest_together_with_the_current_cut-context_snapshot_"
            "source-value_affine-constraint_and_aggregation_revisions"
        ),
        "floor_bits_excluded_from_digests": [
            "emitted_route_form_binding_digest",
            "exact_aggregate_trace_certificate_digest",
        ],
        "coercive_bulk_energy_authority": (
            "production_Navier-Stokes_module-supplied_viscous_K"
        ),
        "generic_FE_gate_scope": (
            "conditional_on_an_installed_caller-supplied_coercive_bulk_form"
        ),
        "predeclared_method_energy_floor": METHOD_ENERGY_FLOOR,
        "method_floor_safe_risk_cap_hex": "0x1.1fffffffffffep-1",
        "method_floor_safe_risk_cap": SAFE_GROUPED_SYMMETRIC_RATIO_CAP,
    }
)

TRACE_TEST = (
    "FreeSurfaceCutStability."
    "DISABLED_SymmetricNitscheAggregateTraceCertificateMatrixV3"
)
TRACE_GROUP_ID = (
    "symmetric_nitsche_accepted_state_floor_108_case_diagnostic"
)
TRACE_CASE_PREFIX = "WP3_WP7_NITSCHE_TRACE_V3_CASE "
TRACE_SUMMARY_PREFIX = "WP3_WP7_NITSCHE_TRACE_V3_SUMMARY "
TRACE_EVIDENCE_ARTIFACT = "aggregate_trace_certificate_evidence.json"
TRACE_PENALTY_GAMMA = 12.0
DIRECT_SAFE_TRACE_UPPER_BOUND_LIMIT = (
    TRACE_PENALTY_GAMMA * SAFE_GROUPED_SYMMETRIC_RATIO_CAP
)
EXPECTED_TRACE_CASE_COUNT = 108
EXPECTED_TRACE_WET_CASE_COUNT = 96
EXPECTED_TRACE_DRY_CASE_COUNT = 12

EXACT_DYADIC_RETAINED_QUOTIENT_DIMENSION_CAP = (
    _parent.EXACT_DYADIC_RETAINED_QUOTIENT_DIMENSION_CAP
)
EXACT_DYADIC_GROUP_ID = _parent.EXACT_DYADIC_GROUP_ID
EXACT_DYADIC_TESTS = _parent.EXACT_DYADIC_TESTS
EXPECTED_EXACT_DYADIC_SOURCE_ROLES = copy.deepcopy(
    _parent.EXPECTED_EXACT_DYADIC_SOURCE_ROLES
)
EXPECTED_IMPLEMENTATION_SOURCE_ROLES = (
    (
        "Code/Source/solver/FE/Analysis/"
        "GeneratedBoundaryAggregateTraceCertificate.h",
        "public certificate envelope and result contract",
    ),
    (
        "Code/Source/solver/FE/Analysis/"
        "GeneratedBoundaryAggregateTraceCertificate.cpp",
        "collective patch construction and deterministic generalized trace "
        "certification",
    ),
    (
        "Code/Source/solver/FE/Assembly/BackgroundEntityMeasures.h",
        "shared physical background-entity measure contract",
    ),
    (
        "Code/Source/solver/FE/Assembly/BackgroundEntityMeasures.cpp",
        "shared physical background-entity measure implementation",
    ),
    (
        "Code/Source/solver/FE/Assembly/CutIntegrationContext.h",
        "generated-domain rules, content revision, and snapshot binding "
        "contract",
    ),
    (
        "Code/Source/solver/FE/Basis/LagrangeBasis.h",
        "affine-P1 basis value and gradient contract",
    ),
    (
        "Code/Source/solver/FE/Basis/LagrangeBasis.cpp",
        "affine-P1 basis evaluation used by exact Gram-factor formation",
    ),
    (
        "Code/Source/solver/FE/Geometry/CutQuadrature.h",
        "generated cut-quadrature rule, provenance, and validity contract",
    ),
    (
        "Code/Source/solver/FE/Geometry/CutQuadrature.cpp",
        "generated cut-quadrature construction and validation",
    ),
    (
        "Code/Source/solver/FE/Interfaces/FreeSurfaceGeometrySnapshot.h",
        "authoritative free-surface geometry snapshot contract",
    ),
    (
        "Code/Source/solver/FE/Interfaces/FreeSurfaceGeometrySnapshot.cpp",
        "authoritative free-surface geometry snapshot implementation",
    ),
    (
        "Code/Source/solver/FE/Interfaces/LevelSetInterfaceDomain.h",
        "revision-bound generated interface and volume-domain contract",
    ),
    (
        "Code/Source/solver/FE/Interfaces/LevelSetInterfaceBuilder.h",
        "linear level-set cut-cell construction contract",
    ),
    (
        "Code/Source/solver/FE/Interfaces/LevelSetInterfaceBuilder.cpp",
        "linear level-set interface and active-volume construction",
    ),
    (
        "Code/Source/solver/FE/Interfaces/"
        "GeneratedInterfaceBoundaryIntersectionDomain.h",
        "generated interface and physical-boundary intersection contract",
    ),
    (
        "Code/Source/solver/FE/Interfaces/"
        "GeneratedInterfaceBoundaryIntersectionDomain.cpp",
        "generated boundary-intersection construction and validation",
    ),
    (
        "Code/Source/solver/FE/Interfaces/GeneratedActiveBoundaryDomain.h",
        "generated sharp active-boundary domain and partition contract",
    ),
    (
        "Code/Source/solver/FE/Interfaces/GeneratedActiveBoundaryDomain.cpp",
        "generated sharp active-boundary clipping and partition validation",
    ),
    (
        "Code/Source/solver/FE/LevelSet/LevelSetCellEvaluator.h",
        "current scalar and reference-gradient evaluator contract",
    ),
    (
        "Code/Source/solver/FE/LevelSet/LevelSetCellEvaluator.cpp",
        "current level-set cell value and gradient evaluation",
    ),
    (
        "Code/Source/solver/FE/LevelSet/"
        "LevelSetImplicitCutQuadratureBackend.h",
        "implicit cut-quadrature dispatch contract",
    ),
    (
        "Code/Source/solver/FE/LevelSet/"
        "LevelSetImplicitCutQuadratureBackend.cpp",
        "production implicit cut-quadrature dispatch and fallback "
        "implementation",
    ),
    (
        "Code/Source/solver/FE/LevelSet/LevelSetInterfaceLifecycle.h",
        "revision-bound generated level-set lifecycle contract",
    ),
    (
        "Code/Source/solver/FE/LevelSet/LevelSetInterfaceLifecycle.cpp",
        "current-source generated-interface lifecycle and publication",
    ),
    (
        "Code/Source/solver/FE/Geometry/CutQuadratureMapping.h",
        "reference-to-physical cut-quadrature mapping contract",
    ),
    (
        "Code/Source/solver/FE/Geometry/CutQuadratureMapping.cpp",
        "reference-to-physical cut-quadrature mapping implementation",
    ),
    (
        "Code/Source/solver/FE/Math/DenseLinearAlgebra.h",
        "floating diagnostics plus exact dense and factorized binary64-dyadic "
        "generalized-bound contracts",
    ),
    (
        "Code/Source/solver/FE/Math/DenseExactDyadic.cpp",
        "authoritative exact binary64-dyadic dense and factorized "
        "positive-Gram quotient certification",
    ),
    (
        "Code/Source/solver/FE/Math/DenseLinearAlgebra.cpp",
        "dense generalized eigenvalue certification implementation",
    ),
    (
        "Code/Source/solver/FE/Tests/Unit/Math/test_DenseLinearAlgebra.cpp",
        "floating, exact-dense, and exact-factorized generalized-bound "
        "evidence",
    ),
    (
        "Code/Source/solver/FE/Constraints/AffineConstraints.h",
        "closed tangent-row and layout-revision contract",
    ),
    (
        "Code/Source/solver/FE/Constraints/AffineConstraints.cpp",
        "affine closure, tangent application, and revision implementation",
    ),
    (
        "Code/Source/solver/FE/Constraints/SmallCutAggregationConstraint.h",
        "trace-eligible finalized aggregation report schema",
    ),
    (
        "Code/Source/solver/FE/Constraints/SmallCutAggregationConstraint.cpp",
        "canonical closed-tangent aggregation report production",
    ),
    (
        "Code/Source/solver/FE/Forms/BoundaryConditions.h",
        "opaque generated-boundary symmetric-gradient Nitsche route binding",
    ),
    (
        "Code/Source/solver/FE/Forms/FormExpr.h",
        "bound route expression identity contract",
    ),
    (
        "Code/Source/solver/FE/Forms/FormExpr.cpp",
        "bound route expression identity implementation",
    ),
    (
        "Code/Source/solver/FE/Analysis/FormulationRecord.h",
        "installed-form provenance record contract",
    ),
    (
        "Code/Source/solver/FE/Systems/FESystem.h",
        "accepted-state floor policy, certificate record, cache, and public "
        "lifecycle interfaces",
    ),
    (
        "Code/Source/solver/FE/Systems/FESystem.cpp",
        "eager exact certification, outward operator grouping, rounding-safe "
        "floor gate, digest validation, revision validation, and collective "
        "agreement",
    ),
    (
        "Code/Source/solver/FE/Systems/FormsInstaller.h",
        "form-bound policy installation request including the predeclared "
        "symmetric energy floor",
    ),
    (
        "Code/Source/solver/FE/Systems/FormsInstaller.cpp",
        "exact route-anchor and floor validation with transactional policy "
        "installation",
    ),
    (
        "Code/Source/solver/FE/Systems/SystemSetup.h",
        "system setup input and lifecycle entry contract",
    ),
    (
        "Code/Source/solver/FE/Systems/SystemSetup.cpp",
        "setup and finalized constraint-refresh certification lifecycle",
    ),
    (
        "Code/Source/solver/FE/Systems/SystemAssembly.h",
        "current-certificate assembly-preflight entry contract",
    ),
    (
        "Code/Source/solver/FE/Systems/SystemAssembly.cpp",
        "assembly preflight requiring current certificates before output "
        "mutation",
    ),
    (
        "Code/Source/solver/FE/Assembly/StandardAssembler.h",
        "standard assembly route contract",
    ),
    (
        "Code/Source/solver/FE/Assembly/StandardAssembler.cpp",
        "standard assembly route implementation",
    ),
    (
        "Code/Source/solver/FE/Forms/FormKernels.h",
        "production sharp-boundary Nitsche form-kernel contract",
    ),
    (
        "Code/Source/solver/FE/Forms/FormKernels.cpp",
        "production sharp-boundary Nitsche form kernels",
    ),
    (
        "Code/Source/solver/Physics/Formulations/NavierStokes/"
        "NavierStokesBCFactories.h",
        "production velocity-Nitsche boundary form and penalty coefficient "
        "construction",
    ),
    (
        "Code/Source/solver/Physics/Formulations/NavierStokes/"
        "IncompressibleNavierStokesVMSModule.h",
        "production generated-boundary energy-floor option and Nitsche "
        "qualification contract",
    ),
    (
        "Code/Source/solver/Physics/Formulations/NavierStokes/"
        "IncompressibleNavierStokesVMSModule.cpp",
        "production validation, registration, and effective-configuration "
        "provenance for generated-boundary energy-floor policies",
    ),
    (
        "Code/Source/solver/FE/Tests/Unit/Assembly/"
        "test_GeneratedBoundaryAggregateTraceCertificate.cpp",
        "serial analytic, accepted/subfloor, adjacent-ULP, operator-group, "
        "invalidation, and unsymmetric evidence",
    ),
    (
        "Code/Source/solver/FE/Tests/Unit/Assembly/"
        "test_GeneratedBoundaryAggregateTraceCertificateMPI.cpp",
        "exact two-rank policy-consensus, distributed-subfloor, cross-owner "
        "acceptance, and assembly-preflight evidence",
    ),
    (
        "Code/Source/solver/Physics/Tests/Unit/"
        "test_FreeSurfaceCutStability.cpp",
        "108-case direct-gate, accepted-floor, exact-certificate, and "
        "sampled-spectrum diagnostic",
    ),
    (
        "Code/Source/solver/Physics/Tests/Unit/"
        "test_MovingDomainPhysics.cpp",
        "production floor validation, policy binding, and "
        "effective-configuration provenance evidence",
    ),
    (
        "Code/Source/solver/FE/CMakeLists.txt",
        "serial and MPI certificate test build routing",
    ),
    (
        "Code/Source/solver/Physics/CMakeLists.txt",
        "physics diagnostic test build routing",
    ),
    (
        "Documentation/"
        "free_surface_wp3_wp7_symmetric_nitsche_coercivity_method_v3.md",
        "matching accepted-state energy-floor derivation, rounding-safe gate, "
        "and claim boundary",
    ),
    (
        "tests/cases/fluid/run_free_surface_wp2_geometry_qualification.py",
        "resource-bounded generic qualification execution base",
    ),
    (
        "tests/cases/fluid/"
        "run_free_surface_wp3_wp7_nitsche_coercivity_qualification_v2.py",
        "inherited fail-closed qualification validation and execution "
        "machinery imported by the V3 wrapper",
    ),
)
EXPECTED_CERTIFICATE_ENVELOPE = copy.deepcopy(
    _parent.EXPECTED_CERTIFICATE_ENVELOPE
)
EXPECTED_CERTIFICATE_ENVELOPE.update(
    {
        "floor_bits_bind_policy_signature": True,
        "floor_bits_bind_current_certificate_cache_digest": True,
        "floor_bits_bind_emitted_route_form_binding_digest": False,
        "floor_bits_bind_exact_trace_certificate_digest": False,
        "operator_group_risk_outward_rounded": True,
        "assembly_recomputes_direct_floor_gate": True,
    }
)
EXPECTED_BUILD_TARGETS = copy.deepcopy(_parent.EXPECTED_BUILD_TARGETS)
EXPECTED_BUILD_CMAKE_HOMES = copy.deepcopy(
    _parent.EXPECTED_BUILD_CMAKE_HOMES
)
EXPECTED_FRESH_CONFIGURE_DEFINITIONS = {
    "Code/Source/solver/FE": (
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_TESTING=ON",
        "-DBoost_INCLUDE_DIR=/share/software/user/open/boost/1.90.0/include",
        "-DEigen3_DIR=/share/software/user/open/eigen/3.4.0/share/eigen3/cmake",
        "-DFE_BUILD_TESTS=ON",
        "-DFE_USE_SYSTEM_GTEST=OFF",
        "-DFE_ENABLE_ASSEMBLY=ON",
        "-DFE_WITH_MESH=ON",
        "-DFE_ENABLE_MPI=ON",
        "-DFE_ENABLE_METIS=OFF",
        "-DFE_ENABLE_PARMETIS=OFF",
        "-DMESH_ENABLE_MPI=ON",
        "-DMESH_ENABLE_METIS=ON",
        "-DMESH_ENABLE_PARMETIS=ON",
        "-DMESH_ENABLE_VTK=OFF",
        "-DMESH_ENABLE_EIGEN=ON",
        "-DUSE_SYSTEM_EIGEN=ON",
        "-DMESH_BUILD_TESTS=OFF",
        "-DMESH_BUILD_SHARED=OFF",
    ),
    "Code/Source/solver/Physics": (
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_TESTING=ON",
        "-DBoost_INCLUDE_DIR=/share/software/user/open/boost/1.90.0/include",
        "-DEigen3_DIR=/share/software/user/open/eigen/3.4.0/share/eigen3/cmake",
        "-DPHYSICS_BUILD_TESTS=ON",
        "-DPHYSICS_USE_SYSTEM_GTEST=OFF",
        "-DPHYSICS_WITH_MESH=ON",
        "-DFE_BUILD_TESTS=OFF",
        "-DFE_ENABLE_ASSEMBLY=ON",
        "-DFE_WITH_MESH=ON",
        "-DFE_ENABLE_MPI=ON",
        "-DFE_ENABLE_METIS=OFF",
        "-DFE_ENABLE_PARMETIS=OFF",
        "-DMESH_ENABLE_MPI=ON",
        "-DMESH_ENABLE_METIS=ON",
        "-DMESH_ENABLE_PARMETIS=ON",
        "-DMESH_ENABLE_VTK=OFF",
        "-DMESH_ENABLE_EIGEN=ON",
        "-DUSE_SYSTEM_EIGEN=ON",
        "-DMESH_BUILD_TESTS=OFF",
        "-DMESH_BUILD_SHARED=OFF",
    ),
}
EXPECTED_RESOURCE_SAFEGUARDS = copy.deepcopy(
    _parent.EXPECTED_RESOURCE_SAFEGUARDS
)
EXPECTED_RESOURCE_SAFEGUARDS.update(
    {
        "source_root_must_equal_runner_repository_root": True,
        "source_worktree_requires_detached_head": True,
        "source_worktree_requires_external_git_common_directory": True,
        "source_worktree_requires_zero_ignored_paths": True,
        "execution_HEAD_must_equal_bundle_commit": True,
        "historical_validation_uses_recorded_implementation_source_commit": (
            True
        ),
        "python_bytecode_writes_disabled": True,
        "cmake_configure_uses_fresh": True,
        "cmake_configure_requires_exact_source_and_build_arguments": True,
        "cmake_configure_rejects_unrecognized_source_homes": True,
        "cmake_configure_rejects_nonexact_cache_definitions": True,
        "cmake_fresh_configure_definitions_by_source_home": {
            source_home: list(definitions)
            for source_home, definitions in (
                EXPECTED_FRESH_CONFIGURE_DEFINITIONS.items()
            )
        },
    }
)

SERIAL_TRACE_TESTS = (
    "GeneratedBoundaryAggregateTraceCertificate."
    "FormBindingRequiresExactlyOneRouteAnchorBeforeMutation",
    "GeneratedBoundaryAggregateTraceCertificate."
    "FullActiveUnitTriangleHasAnalyticBoundFour",
    "GeneratedBoundaryAggregateTraceCertificate."
    "RootedCutSquareCertifiesActualAggregateProlongation",
    "GeneratedBoundaryAggregateTraceCertificate."
    "RootlessAggregateSupportIsRejected",
    "GeneratedBoundaryAggregateTraceCertificate."
    "ImportedGeneratedDomainsWithoutAuthoritativeSnapshotFailClosed",
    "GeneratedBoundaryAggregateTraceCertificate."
    "ScalarFieldIsRejectedAsAnUnsupportedTraceSpace",
    "GeneratedBoundaryAggregateTraceCertificate."
    "SymmetricPolicyRejectsAnInsufficientConfiguredPenalty",
    "GeneratedBoundaryAggregateTraceCertificate."
    "SymmetricPolicyRejectsAPositiveButSubfloorEnergyRatio",
    "GeneratedBoundaryAggregateTraceCertificate."
    "SymmetricEnergyFloorAcceptsTheSafeCapAndRejectsItsNextUlp",
    "GeneratedBoundaryAggregateTraceCertificate."
    "SymmetricPoliciesApplyTheEnergyFloorToTheirOperatorLevelGroup",
    "GeneratedBoundaryAggregateTraceCertificate."
    "UnsymmetricPolicyRetainsTheBoundWithoutACoercivityThreshold",
)
MPI_TRACE_TESTS = (
    "GeneratedBoundaryAggregateTraceCertificateMPI."
    "PolicyFloorMismatchFailsCollectivelyBeforeCertificatePublication",
    "GeneratedBoundaryAggregateTraceCertificateMPI."
    "SubfloorCertificateFailsCollectivelyWithoutPartialPublication",
    "GeneratedBoundaryAggregateTraceCertificateMPI."
    "RootedCrossRankAggregateHasAnalyticBoundThirtyTwoOverSeventyNine",
)
PRODUCTION_GROUP_ID = (
    "generated_boundary_nitsche_energy_floor_production_serial"
)
PRODUCTION_TESTS = (
    "MovingDomainPhysics."
    "GeneratedBoundaryNitscheRouteRegistersItsTracePolicy",
    "MovingDomainPhysics."
    "NavierStokesEffectiveConfigurationSnapshotExpandsBoundaryDefaults",
)
EXPECTED_GROUP_TESTS = {
    EXACT_DYADIC_GROUP_ID: ("math", 1, 1, EXACT_DYADIC_TESTS),
    "aggregate_trace_certificate_serial": (
        "assembly",
        1,
        1,
        SERIAL_TRACE_TESTS,
    ),
    "aggregate_trace_certificate_exact_two_rank_mpi": (
        "assembly_mpi",
        2,
        2,
        MPI_TRACE_TESTS,
    ),
    PRODUCTION_GROUP_ID: ("physics", 1, 1, PRODUCTION_TESTS),
    TRACE_GROUP_ID: ("physics", 1, 1, (TRACE_TEST,)),
}
EXPECTED_GROUP_EXECUTION = {
    EXACT_DYADIC_GROUP_ID: (300, 1024, 64),
    "aggregate_trace_certificate_serial": (600, 1024, 64),
    "aggregate_trace_certificate_exact_two_rank_mpi": (600, 1024, 64),
    PRODUCTION_GROUP_ID: (600, 1024, 64),
    TRACE_GROUP_ID: (3600, 1024, 64),
}
EXPECTED_CLOSURE_CONTRACT = [
    {
        "claim": (
            "exact_binary64_dyadic_dense_and_factorized_spd_quotient_"
            "prerequisite"
        ),
        "evidence": list(EXACT_DYADIC_TESTS),
    },
    {
        "claim": (
            "serial_conditional_FE_energy_floor_gate_and_aggregate_trace_"
            "prerequisite"
        ),
        "evidence": list(SERIAL_TRACE_TESTS),
    },
    {
        "claim": (
            "exact_two_rank_floor_consensus_rejection_and_acceptance_"
            "prerequisite"
        ),
        "evidence": list(MPI_TRACE_TESTS),
    },
    {
        "claim": (
            "production_floor_policy_and_configuration_provenance_"
            "prerequisite"
        ),
        "evidence": list(PRODUCTION_TESTS),
    },
    {
        "claim": (
            "finite_108_case_accepted_floor_and_sampled_spectrum_"
            "consistency_prerequisite"
        ),
        "evidence": [TRACE_TEST],
    },
]
EXPECTED_GATES = {
    "expected_group_count": 5,
    "expected_distinct_test_count": 33,
    "expected_quantitative_evidence_count": 8,
    "expected_failures": 0,
    "expected_errors": 0,
    "expected_disabled": 0,
    "expected_skipped": 0,
}
EXPECTED_QUANTITATIVE_EVIDENCE = {
    (TRACE_TEST, "wp3_wp7_nitsche_trace_v3_case_count"): (
        "integer",
        "equal",
        108,
    ),
    (TRACE_TEST, "wp3_wp7_nitsche_trace_v3_maximum_upper_bound"): (
        "real",
        "less_than_or_equal",
        TRACE_PENALTY_GAMMA * SAFE_GROUPED_SYMMETRIC_RATIO_CAP,
    ),
    (
        TRACE_TEST,
        "wp3_wp7_nitsche_trace_v3_minimum_finite_sample_lower_bound",
    ): ("real", "equal", METHOD_ENERGY_FLOOR),
    (
        TRACE_TEST,
        "wp3_wp7_nitsche_trace_v3_minimum_sampled_eigenvalue_gap",
    ): ("real", "greater_than_or_equal", -1.0e-11),
    (
        TRACE_TEST,
        "wp3_wp7_nitsche_trace_v3_exact_common_kernel_quotient_patch_count",
    ): ("integer", "greater_than", 0),
    (
        TRACE_TEST,
        "wp3_wp7_nitsche_trace_v3_method_coercivity_lower_bound",
    ): ("real", "equal", METHOD_ENERGY_FLOOR),
    (
        TRACE_TEST,
        "wp3_wp7_nitsche_trace_v3_uniform_bound_status",
    ): ("string", "equal", UNIFORM_BOUND_STATUS),
    (
        TRACE_TEST,
        "wp3_wp7_nitsche_trace_v3_accepted_claim",
    ): ("string", "equal", ACCEPTED_CLAIM),
}

EXPECTED_TRACE_CASE_FIELDS = set(_V2_TRACE_CASE_FIELDS) | {
    "required_minimum_energy_ratio"
}
EXPECTED_TRACE_IDENTITY_FIELDS = set(_parent.EXPECTED_TRACE_IDENTITY_FIELDS)
EXPECTED_TRACE_SUMMARY_FIELDS = set(_V2_TRACE_SUMMARY_FIELDS)


def _synchronize_parent_contract() -> None:
    values = {
        "SCRIPT_PATH": SCRIPT_PATH,
        "SCRIPT_DIRECTORY": SCRIPT_DIRECTORY,
        "REPOSITORY_ROOT": REPOSITORY_ROOT,
        "DEFAULT_REGISTRY": DEFAULT_REGISTRY,
        "SHARED_RUNNER_PATH": SHARED_RUNNER_PATH,
        "EXPECTED_NORMALIZED_REGISTRY_SHA256": (
            EXPECTED_NORMALIZED_REGISTRY_SHA256
        ),
        "RUNNER_SHA256_ZERO_SENTINEL": RUNNER_SHA256_ZERO_SENTINEL,
        "EXPECTED_SHARED_RUNNER_SHA256": EXPECTED_SHARED_RUNNER_SHA256,
        "EXPECTED_V1_PARENT_SHA256": EXPECTED_V2_PARENT_SHA256,
        "EXPECTED_MATRIX_ID": EXPECTED_MATRIX_ID,
        "EXPECTED_MATRIX_STATUS": EXPECTED_MATRIX_STATUS,
        "EXPECTED_CHECKED_IN_MATRIX_STATUS": (
            EXPECTED_CHECKED_IN_MATRIX_STATUS
        ),
        "EXECUTABLE_MATRIX_STATUS": EXECUTABLE_MATRIX_STATUS,
        "ALLOWED_MATRIX_STATUSES": ALLOWED_MATRIX_STATUSES,
        "EXPECTED_DRAFT_SOURCE_HASH_STATUS": (
            EXPECTED_DRAFT_SOURCE_HASH_STATUS
        ),
        "EXPECTED_FROZEN_SOURCE_HASH_STATUS": (
            EXPECTED_FROZEN_SOURCE_HASH_STATUS
        ),
        "EXPECTED_WORK_PACKAGE": EXPECTED_WORK_PACKAGE,
        "EXPECTED_MATCHING_DERIVATION": EXPECTED_MATCHING_DERIVATION,
        "EXPECTED_PROPOSED_RUNNER": EXPECTED_PROPOSED_RUNNER,
        "EXPECTED_MATRIX_PATH": EXPECTED_MATRIX_PATH,
        "EXPECTED_QUALIFICATION_BUNDLE_BINDING": (
            EXPECTED_QUALIFICATION_BUNDLE_BINDINGS[
                EXPECTED_CHECKED_IN_MATRIX_STATUS
            ]
        ),
        "EXPECTED_SCOPE": EXPECTED_SCOPE,
        "EXPECTED_CLOSURE_REQUEST_POLICY": EXPECTED_CLOSURE_REQUEST_POLICY,
        "EXPECTED_DISPOSITION": EXPECTED_DISPOSITION,
        "EXPECTED_OPEN_OUTCOMES": EXPECTED_OPEN_OUTCOMES,
        "EXPECTED_CASE_AXES": EXPECTED_CASE_AXES,
        "EXPECTED_TRACE_CONTRACT": EXPECTED_TRACE_CONTRACT,
        "TRACE_TEST": TRACE_TEST,
        "TRACE_GROUP_ID": TRACE_GROUP_ID,
        "TRACE_CASE_PREFIX": TRACE_CASE_PREFIX,
        "TRACE_SUMMARY_PREFIX": TRACE_SUMMARY_PREFIX,
        "TRACE_EVIDENCE_ARTIFACT": TRACE_EVIDENCE_ARTIFACT,
        "TRACE_PENALTY_GAMMA": TRACE_PENALTY_GAMMA,
        "EXPECTED_TRACE_CASE_COUNT": EXPECTED_TRACE_CASE_COUNT,
        "EXPECTED_TRACE_WET_CASE_COUNT": EXPECTED_TRACE_WET_CASE_COUNT,
        "EXPECTED_TRACE_DRY_CASE_COUNT": EXPECTED_TRACE_DRY_CASE_COUNT,
        "EXACT_DYADIC_RETAINED_QUOTIENT_DIMENSION_CAP": (
            EXACT_DYADIC_RETAINED_QUOTIENT_DIMENSION_CAP
        ),
        "EXACT_DYADIC_GROUP_ID": EXACT_DYADIC_GROUP_ID,
        "EXACT_DYADIC_TESTS": EXACT_DYADIC_TESTS,
        "EXPECTED_EXACT_DYADIC_SOURCE_ROLES": (
            EXPECTED_EXACT_DYADIC_SOURCE_ROLES
        ),
        "EXPECTED_CERTIFICATE_ENVELOPE": EXPECTED_CERTIFICATE_ENVELOPE,
        "EXPECTED_BUILD_TARGETS": EXPECTED_BUILD_TARGETS,
        "EXPECTED_BUILD_CMAKE_HOMES": EXPECTED_BUILD_CMAKE_HOMES,
        "EXPECTED_RESOURCE_SAFEGUARDS": EXPECTED_RESOURCE_SAFEGUARDS,
        "EXPECTED_GROUP_TESTS": EXPECTED_GROUP_TESTS,
        "EXPECTED_GROUP_EXECUTION": EXPECTED_GROUP_EXECUTION,
        "EXPECTED_GATES": EXPECTED_GATES,
        "EXPECTED_QUANTITATIVE_EVIDENCE": EXPECTED_QUANTITATIVE_EVIDENCE,
        "EXPECTED_TRACE_CASE_FIELDS": EXPECTED_TRACE_CASE_FIELDS,
        "EXPECTED_TRACE_IDENTITY_FIELDS": EXPECTED_TRACE_IDENTITY_FIELDS,
        "EXPECTED_TRACE_SUMMARY_FIELDS": EXPECTED_TRACE_SUMMARY_FIELDS,
    }
    for name, value in values.items():
        setattr(_parent, name, value)


_synchronize_parent_contract()


def _synchronize_shared_contract() -> None:
    shared = _parent.strict_runner
    shared.SCRIPT_PATH = SCRIPT_PATH
    shared.DEFAULT_REGISTRY = DEFAULT_REGISTRY
    shared.EXPECTED_MATRIX_ID = EXPECTED_MATRIX_ID
    shared.EXPECTED_MATRIX_STATUS = EXPECTED_CHECKED_IN_MATRIX_STATUS
    shared.EXPECTED_WORK_PACKAGE = EXPECTED_WORK_PACKAGE
    shared.__doc__ = __doc__


_synchronize_shared_contract()


GIT_NO_REPLACE_OBJECTS = "--no-replace-objects"


def _validated_commit_digest(value: str, label: str) -> str:
    if (
        len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} has an invalid commit digest")
    return value


def _exact_commit_output(raw: bytes, label: str) -> str:
    if (
        not raw.endswith(b"\n")
        or raw.count(b"\n") != 1
        or b"\r" in raw
    ):
        raise ValueError(f"{label} commit output is malformed")
    try:
        value = raw[:-1].decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} commit output is malformed") from error
    try:
        return _validated_commit_digest(value, label)
    except ValueError as error:
        raise ValueError(f"{label} commit output is malformed") from error


def _resolved_commit(
    repository_root: Path,
    revision: str,
    label: str,
) -> str:
    _validated_commit_digest(revision, label)
    try:
        raw = _parent.strict_runner.git_bytes(
            repository_root,
            GIT_NO_REPLACE_OBJECTS,
            "rev-parse",
            "--verify",
            f"{revision}^{{commit}}",
        )
    except (
        OSError,
        _parent.subprocess.CalledProcessError,
    ) as error:
        raise ValueError(f"{label} is not an available commit") from error
    resolved = _exact_commit_output(raw, label)
    if resolved != revision:
        raise ValueError(f"{label} did not resolve exactly")
    return revision


def _current_head_commit(repository_root: Path) -> str:
    try:
        raw = _parent.strict_runner.git_bytes(
            repository_root,
            GIT_NO_REPLACE_OBJECTS,
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
        )
    except (
        OSError,
        _parent.subprocess.CalledProcessError,
    ) as error:
        raise ValueError("validation HEAD is not an available commit") from error
    return _exact_commit_output(raw, "validation HEAD")


def _require_ancestor(
    repository_root: Path,
    ancestor: str,
    descendant: str,
    diagnostic: str,
) -> None:
    try:
        _parent.strict_runner.git_bytes(
            repository_root,
            GIT_NO_REPLACE_OBJECTS,
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        )
    except _parent.subprocess.CalledProcessError as error:
        if error.returncode == 1:
            raise ValueError(diagnostic) from error
        raise ValueError("qualification ancestry is unavailable") from error
    except OSError as error:
        raise ValueError("qualification ancestry is unavailable") from error


def _validate_execution_source_worktree(
    source_root: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, str]:
    resolved_source_root = source_root.resolve()
    resolved_repository_root = repository_root.resolve()
    if resolved_source_root != resolved_repository_root:
        raise ValueError(
            "qualification source root must equal the runner repository root"
        )

    try:
        _parent.strict_runner.git_bytes(
            resolved_source_root,
            "symbolic-ref",
            "-q",
            "HEAD",
        )
    except _parent.subprocess.CalledProcessError as error:
        if error.returncode != 1:
            raise ValueError(
                "qualification source worktree HEAD state is unavailable"
            ) from error
    except OSError as error:
        raise ValueError(
            "qualification source worktree HEAD state is unavailable"
        ) from error
    else:
        raise ValueError(
            "qualification source worktree must use a detached HEAD"
        )

    binding = _parent._frozen_qualification_bundle_binding
    if not isinstance(binding, dict):
        raise ValueError(
            "qualification bundle binding is unavailable for execution"
        )
    bundle_commit = binding.get("qualification_bundle_commit")
    if not isinstance(bundle_commit, str):
        raise ValueError(
            "qualification bundle commit is unavailable for execution"
        )
    _validated_commit_digest(bundle_commit, "qualification bundle commit")
    execution_head = _current_head_commit(resolved_source_root)
    if execution_head != bundle_commit:
        raise ValueError(
            "qualification execution HEAD must equal the canonical bundle "
            "commit"
        )

    try:
        common_directory_text = (
            _parent.strict_runner.git_bytes(
                resolved_source_root,
                "rev-parse",
                "--git-common-dir",
            )
            .decode("utf-8")
            .strip()
        )
    except (
        OSError,
        _parent.subprocess.CalledProcessError,
        UnicodeDecodeError,
    ) as error:
        raise ValueError(
            "qualification source Git common directory is unavailable"
        ) from error
    if not common_directory_text:
        raise ValueError(
            "qualification source Git common directory is empty"
        )
    common_directory = Path(common_directory_text)
    if not common_directory.is_absolute():
        common_directory = resolved_source_root / common_directory
    common_directory = common_directory.resolve()
    if _parent.strict_runner.path_is_within(
        common_directory,
        resolved_source_root,
    ):
        raise ValueError(
            "qualification source worktree requires an external Git common "
            "directory"
        )
    return {
        "source_root": str(resolved_source_root),
        "git_common_directory": str(common_directory),
        "qualification_bundle_commit": bundle_commit,
        "execution_head_commit": execution_head,
    }


def require_execution_resource_preflight(
    source_root: Path,
    output_directory: Path,
    build_directories: tuple[Path, ...] = (),
) -> None:
    _validate_execution_source_worktree(source_root)
    _parent_require_execution_resource_preflight(
        source_root,
        output_directory,
        build_directories,
    )


def untracked_source_record(
    source_root: Path,
    allowed_output_root: Path | None = None,
    ignored_source_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    resolved_source_root = source_root.resolve()
    whole_root_ignored = tuple(
        root
        for root in ignored_source_roots
        if root.resolve() != resolved_source_root
    ) + (resolved_source_root,)
    return _shared_untracked_source_record(
        resolved_source_root,
        allowed_output_root,
        whole_root_ignored,
    )


def _cmake_cache_definition_name(argument: str) -> str:
    if argument == "-D" or not argument.startswith("-D"):
        raise ValueError(
            "CMake configure cache definitions must use joined -DNAME=VALUE "
            "arguments"
        )
    name_with_type, separator, _value = argument[2:].partition("=")
    name = name_with_type.partition(":")[0]
    if not separator or not name:
        raise ValueError("CMake configure cache definition is malformed")
    return name


def _locked_fresh_configure_command(
    command: list[str],
    source_root: Path,
) -> list[str]:
    if not command:
        raise ValueError("CMake configure command is empty")
    if command.count("-S") != 1 or command.count("-B") != 1:
        raise ValueError(
            "CMake configure command must use exactly one -S and one -B"
        )
    if command.count("--fresh") > 1:
        raise ValueError("CMake configure command repeats --fresh")

    caller_definition_indices: set[int] = set()
    caller_definitions: dict[str, str] = {}
    for index, argument in enumerate(command):
        if argument == "-D":
            raise ValueError(
                "CMake configure cache definitions must use joined "
                "-DNAME=VALUE arguments"
            )
        if not argument.startswith("-D"):
            continue
        name = _cmake_cache_definition_name(argument)
        if name in caller_definitions:
            raise ValueError(
                f"CMake configure cache definition is ambiguous: {name}"
            )
        caller_definitions[name] = argument
        caller_definition_indices.add(index)

    structural_command = [
        argument
        for index, argument in enumerate(command)
        if index not in caller_definition_indices and argument != "--fresh"
    ]
    if (
        len(structural_command) != 5
        or structural_command[1] != "-S"
        or structural_command[3] != "-B"
    ):
        raise ValueError(
            "CMake configure command must contain only the exact -S and -B "
            "route"
        )

    resolved_source_root = source_root.resolve()
    source_home = Path(structural_command[2])
    if not source_home.is_absolute():
        source_home = resolved_source_root / source_home
    source_home = source_home.resolve()
    selected_home: str | None = None
    locked_definitions: tuple[str, ...] | None = None
    for relative_home, definitions in (
        EXPECTED_FRESH_CONFIGURE_DEFINITIONS.items()
    ):
        if source_home == (resolved_source_root / relative_home).resolve():
            selected_home = relative_home
            locked_definitions = definitions
            break
    if selected_home is None or locked_definitions is None:
        raise ValueError("CMake configure source home is not recognized")

    locked_by_name = {
        _cmake_cache_definition_name(definition): definition
        for definition in locked_definitions
    }
    for name, argument in caller_definitions.items():
        expected = locked_by_name.get(name)
        if expected is None:
            raise ValueError(
                f"CMake configure cache definition is not locked for "
                f"{selected_home}: {name}"
            )
        if argument != expected:
            raise ValueError(
                f"CMake configure cache definition conflicts with the "
                f"locked value for {selected_home}: {name}"
            )

    return [
        structural_command[0],
        "--fresh",
        *locked_definitions,
        *structural_command[1:],
    ]


def run_build_phase(
    command: list[str],
    source_root: Path,
    output_root: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    exact_command = list(command)
    if "--build" in exact_command:
        if (
            "-S" in exact_command
            or "-B" in exact_command
            or "--fresh" in exact_command
            or any(
                argument == "-D" or argument.startswith("-D")
                for argument in exact_command
            )
        ):
            raise ValueError("CMake build/configure command route is ambiguous")
    else:
        exact_command = _locked_fresh_configure_command(
            exact_command,
            source_root,
        )
    return _parent_run_build_phase(
        exact_command,
        source_root,
        output_root,
        stdout_path,
        stderr_path,
        timeout_seconds,
    )


def _validate_v3_status_contract(registry: dict[str, Any]) -> None:
    status = registry.get("status")
    if status not in ALLOWED_MATRIX_STATUSES:
        raise ValueError("accepted-state matrix has an invalid lifecycle state")
    if registry.get("implementation_source_commit") != (
        EXPECTED_IMPLEMENTATION_SOURCE_COMMIT
    ):
        raise ValueError("frozen implementation source commit changed")
    if registry.get("source_inventory_hash_status") != "FROZEN":
        raise ValueError("implementation source inventory must remain frozen")
    if registry.get("status_reason") != EXPECTED_STATUS_REASONS[status]:
        raise ValueError("accepted-state matrix status reason changed")
    if registry.get("qualification_bundle_binding") != (
        EXPECTED_QUALIFICATION_BUNDLE_BINDINGS[status]
    ):
        raise ValueError(
            "qualification bundle binding is inconsistent with lifecycle state"
        )

    promotion = registry.get("draft_promotion_contract")
    if not isinstance(promotion, dict):
        raise ValueError("draft promotion contract is missing")
    expected = {
        "current_state": status,
        "source_hashes_frozen": True,
        "qualification_bundle_hashes_frozen": (
            status == EXECUTABLE_MATRIX_STATUS
        ),
        "qualification_evidence_executed": False,
        "validate_only_allowed": True,
        "execution_allowed": status == EXECUTABLE_MATRIX_STATUS,
        "required_execution_state": EXECUTABLE_MATRIX_STATUS,
    }
    for field, value in expected.items():
        if promotion.get(field) != value:
            raise ValueError(
                f"draft promotion contract is inconsistent with {status}: "
                f"{field}"
            )
    requirements = promotion.get("promotion_requirements")
    if requirements != EXPECTED_PROMOTION_REQUIREMENTS:
        raise ValueError("draft promotion requirements changed")

    runner_digest = registry.get("runner_sha256")
    if (
        not isinstance(runner_digest, str)
        or len(runner_digest) != 64
        or any(character not in "0123456789abcdef" for character in runner_digest)
    ):
        raise ValueError(
            "runner_sha256 must be a 64-character lowercase hexadecimal digest"
        )
    if status == EXPECTED_MATRIX_STATUS:
        if runner_digest != RUNNER_SHA256_ZERO_SENTINEL:
            raise ValueError("draft runner_sha256 must remain the zero sentinel")
    elif runner_digest == RUNNER_SHA256_ZERO_SENTINEL:
        raise ValueError("frozen runner_sha256 must lock the exact runner bytes")


def validate_v3_contract(registry: dict[str, Any]) -> dict[str, Any]:
    """Validate the exact V3 matrix while reusing the V2 structural gates."""
    if registry.get("schema_version") != 3:
        raise ValueError("unsupported accepted-state qualification schema")
    implementation_sources = registry.get("implementation_sources")
    if not isinstance(implementation_sources, list) or any(
        not isinstance(entry, dict) for entry in implementation_sources
    ):
        raise ValueError("accepted-state implementation source manifest changed")
    observed_source_roles = tuple(
        (entry.get("path"), entry.get("role"))
        for entry in implementation_sources
    )
    if observed_source_roles != EXPECTED_IMPLEMENTATION_SOURCE_ROLES:
        raise ValueError("accepted-state implementation source manifest changed")
    _validate_v3_status_contract(registry)
    if registry.get("method_coercivity_lower_bound") != METHOD_ENERGY_FLOOR:
        raise ValueError("accepted-state method energy floor changed")
    if registry.get("uniform_bound_status") != UNIFORM_BOUND_STATUS:
        raise ValueError("accepted-state uniform-bound status changed")
    if registry.get("qualification_scope") != EXPECTED_SCOPE:
        raise ValueError("accepted-state qualification scope changed")
    if registry.get("closure_request_policy") != (
        EXPECTED_CLOSURE_REQUEST_POLICY
    ):
        raise ValueError("accepted-state claim boundary changed")
    if registry.get("model_envelope") != EXPECTED_MODEL_ENVELOPE:
        raise ValueError("accepted-state model envelope changed")
    if registry.get("method_limitations") != EXPECTED_METHOD_LIMITATIONS:
        raise ValueError("accepted-state method limitations changed")
    if registry.get("closure_contract") != EXPECTED_CLOSURE_CONTRACT:
        raise ValueError("accepted-state closure contract changed")
    contract = registry.get("certified_aggregate_trace_contract")
    if contract != EXPECTED_TRACE_CONTRACT:
        raise ValueError("accepted-state aggregate trace contract changed")
    if contract.get("floor_bits_excluded_from_digests") != [
        "emitted_route_form_binding_digest",
        "exact_aggregate_trace_certificate_digest",
    ]:
        raise ValueError("accepted-state floor digest boundary changed")
    envelope = registry.get("certificate_envelope")
    if (
        not isinstance(envelope, dict)
        or envelope.get(
            "floor_bits_bind_emitted_route_form_binding_digest"
        )
        is not False
        or envelope.get("floor_bits_bind_exact_trace_certificate_digest")
        is not False
    ):
        raise ValueError("accepted-state floor digest exclusions changed")

    adapted = copy.deepcopy(registry)
    adapted["schema_version"] = 2
    adapted["method_coercivity_lower_bound"] = None
    adapted["uniform_bound_status"] = "UNFROZEN_NO_BOUND_INVENTED"
    if adapted["status"] == EXPECTED_MATRIX_STATUS:
        adapted["implementation_source_commit"] = None
        adapted["source_inventory_hash_status"] = (
            EXPECTED_DRAFT_SOURCE_HASH_STATUS
        )
        adapted["draft_promotion_contract"]["source_hashes_frozen"] = False
    adapted["draft_promotion_contract"].pop(
        "qualification_bundle_hashes_frozen",
        None,
    )
    saved_binding = _parent.EXPECTED_QUALIFICATION_BUNDLE_BINDING
    try:
        _parent.EXPECTED_QUALIFICATION_BUNDLE_BINDING = (
            EXPECTED_QUALIFICATION_BUNDLE_BINDINGS[registry["status"]]
        )
        _parent_validate_contract(adapted)
    finally:
        _parent.EXPECTED_QUALIFICATION_BUNDLE_BINDING = saved_binding
    return registry


validate_v2_contract = validate_v3_contract


def _ancestry_records(
    repository_root: Path,
    source_commit: str,
    validation_head: str,
) -> list[tuple[str, tuple[str, ...]]]:
    try:
        raw = _parent.strict_runner.git_bytes(
            repository_root,
            GIT_NO_REPLACE_OBJECTS,
            "rev-list",
            "--parents",
            "--ancestry-path",
            f"{source_commit}..{validation_head}",
        )
    except (OSError, _parent.subprocess.CalledProcessError) as error:
        raise ValueError(
            "qualification bundle ancestry record is unavailable"
        ) from error
    if not raw or not raw.endswith(b"\n") or b"\r" in raw or b"\t" in raw:
        raise ValueError("qualification bundle ancestry record is malformed")
    try:
        lines = raw[:-1].decode("ascii").split("\n")
    except UnicodeDecodeError as error:
        raise ValueError(
            "qualification bundle ancestry record is malformed"
        ) from error
    if not lines or any(not line for line in lines):
        raise ValueError("qualification bundle ancestry record is malformed")
    records: list[tuple[str, tuple[str, ...]]] = []
    observed_commits: set[str] = set()
    digest_length = len(source_commit)
    for line in lines:
        fields = line.split(" ")
        if " ".join(fields) != line or not fields:
            raise ValueError(
                "qualification bundle ancestry record is malformed"
            )
        for field in fields:
            _validated_commit_digest(
                field,
                "qualification bundle ancestry entry",
            )
            if len(field) != digest_length:
                raise ValueError(
                    "qualification bundle ancestry record mixes digest lengths"
                )
        commit, *parents = fields
        if commit in observed_commits:
            raise ValueError(
                "qualification bundle ancestry record repeats a commit"
            )
        observed_commits.add(commit)
        records.append((commit, tuple(parents)))
    if records[0][0] != validation_head or validation_head not in observed_commits:
        raise ValueError("qualification bundle ancestry record is malformed")
    return records


def _commit_changed_paths(
    repository_root: Path,
    commit: str,
) -> list[tuple[str, str]]:
    try:
        raw = _parent.strict_runner.git_bytes(
            repository_root,
            GIT_NO_REPLACE_OBJECTS,
            "diff-tree",
            "--no-commit-id",
            "-r",
            "--no-renames",
            "--name-status",
            "-z",
            commit,
            "--",
        )
    except (OSError, _parent.subprocess.CalledProcessError) as error:
        raise ValueError(
            "qualification bundle changed-path record is unavailable"
        ) from error
    if not raw:
        return []
    if not raw.endswith(b"\0"):
        raise ValueError(
            "qualification bundle changed-path record is malformed"
        )
    fields = raw[:-1].split(b"\0")
    if len(fields) % 2 != 0:
        raise ValueError(
            "qualification bundle changed-path record is malformed"
        )
    records: list[tuple[str, str]] = []
    observed_paths: set[str] = set()
    for index in range(0, len(fields), 2):
        try:
            status = fields[index].decode("ascii")
            path = fields[index + 1].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                "qualification bundle changed-path record is malformed"
            ) from error
        if not status or not path or any(character.isspace() for character in status):
            raise ValueError(
                "qualification bundle changed-path record is malformed"
            )
        if path in observed_paths:
            raise ValueError(
                "qualification bundle changed-path record repeats a path"
            )
        observed_paths.add(path)
        records.append((status, path))
    return records


def _commit_regular_blob(
    repository_root: Path,
    commit: str,
    relative_path: str,
    label: str,
    *,
    allowed_modes: frozenset[str] = frozenset({"100644"}),
) -> tuple[bytes, dict[str, str]]:
    try:
        raw_entry = _parent.strict_runner.git_bytes(
            repository_root,
            GIT_NO_REPLACE_OBJECTS,
            "ls-tree",
            "-z",
            commit,
            "--",
            relative_path,
        )
    except (OSError, _parent.subprocess.CalledProcessError) as error:
        raise ValueError(f"{label} tree entry is unavailable") from error
    if not raw_entry or not raw_entry.endswith(b"\0"):
        raise ValueError(f"{label} tree entry is missing or malformed")
    entries = raw_entry[:-1].split(b"\0")
    if len(entries) != 1 or entries[0].count(b"\t") != 1:
        raise ValueError(f"{label} tree entry is ambiguous or malformed")
    raw_header, raw_path = entries[0].split(b"\t", 1)
    try:
        header = raw_header.decode("ascii")
        observed_path = raw_path.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} tree entry is malformed") from error
    header_fields = header.split(" ")
    if len(header_fields) != 3 or " ".join(header_fields) != header:
        raise ValueError(f"{label} tree entry is malformed")
    mode, object_type, object_id = header_fields
    _validated_commit_digest(object_id, f"{label} blob object")
    if (
        observed_path != relative_path
        or mode not in allowed_modes
        or object_type != "blob"
    ):
        raise ValueError(f"{label} must be a regular frozen blob")
    try:
        blob = _parent.strict_runner.git_bytes(
            repository_root,
            GIT_NO_REPLACE_OBJECTS,
            "cat-file",
            "blob",
            object_id,
        )
    except (OSError, _parent.subprocess.CalledProcessError) as error:
        raise ValueError(f"{label} blob is unavailable") from error
    return blob, {
        "git_mode": mode,
        "git_object_type": object_type,
        "git_blob_id": object_id,
    }


def _canonical_bundle_working_bytes(
    registry: dict[str, Any],
    matrix_path: Path,
    repository_root: Path,
    runner_path: Path,
) -> tuple[list[tuple[str, str, bytes]], str]:
    expected_paths = {
        "matrix": EXPECTED_MATRIX_PATH,
        "runner": EXPECTED_PROPOSED_RUNNER,
        "focused_test": EXPECTED_FOCUSED_TEST_PATH,
    }
    actual_paths = {
        "matrix": matrix_path,
        "runner": runner_path,
        "focused_test": repository_root / EXPECTED_FOCUSED_TEST_PATH,
    }
    working: list[tuple[str, str, bytes]] = []
    for role in ("matrix", "runner", "focused_test"):
        relative_path = expected_paths[role]
        expected_path = (repository_root / relative_path).resolve()
        actual_path = actual_paths[role]
        if actual_path.resolve() != expected_path:
            raise ValueError(
                f"qualification bundle {role} path is not canonical"
            )
        if actual_path.is_symlink() or not actual_path.is_file():
            raise ValueError(
                f"qualification bundle {role} is not a regular file"
            )
        working.append((role, relative_path, actual_path.read_bytes()))

    matrix_bytes = working[0][2]
    runner_bytes = working[1][2]
    focused_test_bytes = working[2][2]
    normalized_digest = hashlib.sha256(
        normalized_registry_bytes(matrix_bytes)
    ).hexdigest()
    if normalized_digest != EXPECTED_NORMALIZED_REGISTRY_SHA256:
        raise ValueError(
            "qualification bundle matrix does not match the runner's "
            "embedded normalized SHA-256"
        )
    try:
        matrix_document = json.loads(
            matrix_bytes.decode("utf-8"),
            object_pairs_hook=_parent._reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("qualification bundle matrix is malformed") from error
    matrix_runner_digest = matrix_document.get("runner_sha256")
    if matrix_runner_digest != registry.get("runner_sha256"):
        raise ValueError(
            "qualification bundle matrix runner_sha256 does not match the "
            "frozen registry"
        )
    if hashlib.sha256(runner_bytes).hexdigest() != matrix_runner_digest:
        raise ValueError(
            "qualification bundle runner does not match the matrix "
            "runner_sha256"
        )
    if hashlib.sha256(focused_test_bytes).hexdigest() != (
        EXPECTED_FOCUSED_TEST_SHA256
    ):
        raise ValueError(
            "qualification bundle focused test does not match the runner's "
            "embedded SHA-256"
        )
    return working, normalized_digest


def validate_frozen_qualification_bundle(
    registry: dict[str, Any],
    matrix_path: Path = DEFAULT_REGISTRY,
    repository_root: Path = REPOSITORY_ROOT,
    runner_path: Path = SCRIPT_PATH,
) -> dict[str, Any]:
    if registry.get("status") != EXECUTABLE_MATRIX_STATUS:
        raise ValueError(
            "qualification bundle binding requires a frozen matrix"
        )
    declared_binding = registry.get("qualification_bundle_binding")
    if (
        declared_binding is not None
        and declared_binding != EXPECTED_FROZEN_QUALIFICATION_BUNDLE_BINDING
    ):
        raise ValueError("qualification bundle exact path contract changed")
    source_commit = registry.get("implementation_source_commit")
    if not isinstance(source_commit, str):
        raise ValueError("implementation source commit is missing")
    _resolved_commit(
        repository_root,
        source_commit,
        "implementation source commit",
    )
    validation_head = _current_head_commit(repository_root)
    _require_ancestor(
        repository_root,
        source_commit,
        validation_head,
        "validation HEAD must descend from the implementation source commit",
    )
    if validation_head == source_commit:
        raise ValueError(
            "qualification history contains zero canonical bundle candidates"
        )
    ancestry = _ancestry_records(
        repository_root,
        source_commit,
        validation_head,
    )
    direct_children = [
        commit
        for commit, parents in ancestry
        if parents == (source_commit,)
    ]
    if not direct_children:
        raise ValueError(
            "qualification history contains zero direct-child bundle "
            "candidates"
        )

    working, normalized_digest = _canonical_bundle_working_bytes(
        registry,
        matrix_path,
        repository_root,
        runner_path,
    )
    exact_paths = tuple(
        EXPECTED_FROZEN_QUALIFICATION_BUNDLE_BINDING[
            "exact_bundle_commit_blobs_required"
        ]
    )
    exact_path_candidates: list[str] = []
    matching_candidates: list[
        tuple[str, dict[str, dict[str, str]]]
    ] = []
    blob_drift_candidates: list[str] = []
    for candidate in direct_children:
        changed_records = _commit_changed_paths(repository_root, candidate)
        if (
            len(changed_records) != len(exact_paths)
            or any(status != "A" for status, _path in changed_records)
            or {path for _status, path in changed_records} != set(exact_paths)
        ):
            continue
        exact_path_candidates.append(candidate)
        metadata_by_path: dict[str, dict[str, str]] = {}
        candidate_matches = True
        for _role, relative_path, working_bytes in working:
            committed_bytes, metadata = _commit_regular_blob(
                repository_root,
                candidate,
                relative_path,
                f"qualification bundle {relative_path}",
            )
            metadata_by_path[relative_path] = metadata
            if committed_bytes != working_bytes:
                candidate_matches = False
        if candidate_matches:
            matching_candidates.append((candidate, metadata_by_path))
        else:
            blob_drift_candidates.append(candidate)

    if len(matching_candidates) != 1:
        if not matching_candidates and blob_drift_candidates:
            raise ValueError(
                "qualification history contains zero canonical bundle "
                "candidates because the exact-path candidate has frozen "
                "blob drift"
            )
        if not matching_candidates and direct_children and not exact_path_candidates:
            raise ValueError(
                "qualification history contains zero canonical bundle "
                "candidates because direct-child changed paths do not equal "
                "the exact required paths"
            )
        raise ValueError(
            "qualification history must contain exactly one canonical bundle "
            f"candidate; observed {len(matching_candidates)}"
        )

    bundle_commit, metadata_by_path = matching_candidates[0]
    _require_ancestor(
        repository_root,
        bundle_commit,
        validation_head,
        "validation HEAD must descend from the canonical bundle commit",
    )
    artifacts: list[dict[str, Any]] = []
    for role, relative_path, working_bytes in working:
        artifact: dict[str, Any] = {
            "role": role,
            "path": relative_path,
            "sha256": hashlib.sha256(working_bytes).hexdigest(),
            **metadata_by_path[relative_path],
        }
        if role == "matrix":
            artifact["normalized_sha256"] = normalized_digest
        artifacts.append(artifact)
    return {
        "binding_schema_version": 2,
        "authority": EXPECTED_FROZEN_BUNDLE_AUTHORITY,
        "bundle_commit_resolution": EXPECTED_BUNDLE_COMMIT_RESOLUTION,
        "qualification_bundle_commit": bundle_commit,
        "validation_head_commit": validation_head,
        "implementation_source_commit": source_commit,
        "bundle_parent_commit": source_commit,
        "bundle_changed_paths": sorted(exact_paths),
        "normalized_matrix_sha256_embedded_in_runner": normalized_digest,
        "runner_sha256_from_matrix": registry["runner_sha256"],
        "focused_test_sha256_from_runner": EXPECTED_FOCUSED_TEST_SHA256,
        "artifacts": artifacts,
    }


def load_registry(path: Path) -> dict[str, Any]:
    _parent._frozen_qualification_bundle_binding = None
    if not _parent._normalized_registry_digest_is_frozen():
        raise RuntimeError(
            "accepted-state normalized matrix SHA-256 is not frozen"
        )
    if path.resolve() != DEFAULT_REGISTRY.resolve():
        raise ValueError("qualification requires the canonical V3 matrix")
    if normalized_registry_sha256(path) != (
        EXPECTED_NORMALIZED_REGISTRY_SHA256
    ):
        raise ValueError("accepted-state normalized matrix bytes changed")
    parsed_registry = _parent.parse_json_document(path)
    if parsed_registry.get("status") != EXPECTED_CHECKED_IN_MATRIX_STATUS:
        raise ValueError(
            "accepted-state checked-in matrix lifecycle state changed"
        )
    registry = validate_v3_contract(parsed_registry)
    validate_frozen_dependencies(registry)
    if registry["status"] == EXECUTABLE_MATRIX_STATUS:
        _parent._frozen_qualification_bundle_binding = (
            validate_frozen_qualification_bundle(registry, path)
        )
    return registry


def validate_frozen_dependencies(
    registry: dict[str, Any],
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    if registry.get("status") != EXECUTABLE_MATRIX_STATUS:
        frozen_inventory = copy.deepcopy(registry)
        frozen_inventory["status"] = EXECUTABLE_MATRIX_STATUS
        _parent.validate_frozen_dependencies(
            frozen_inventory,
            repository_root,
        )
        return

    parent_only_inventory = copy.deepcopy(registry)
    parent_only_inventory["status"] = EXPECTED_MATRIX_STATUS
    _parent.validate_frozen_dependencies(
        parent_only_inventory,
        repository_root,
    )
    source_commit = registry.get("implementation_source_commit")
    if not isinstance(source_commit, str):
        raise ValueError("implementation source commit is missing")
    _resolved_commit(
        repository_root,
        source_commit,
        "implementation source commit",
    )
    validation_head = _current_head_commit(repository_root)
    _require_ancestor(
        repository_root,
        source_commit,
        validation_head,
        "validation HEAD must descend from the implementation source commit",
    )
    for entry in registry["implementation_sources"]:
        committed_bytes, _metadata = _commit_regular_blob(
            repository_root,
            source_commit,
            entry["path"],
            f"implementation source {entry['path']}",
            allowed_modes=frozenset({"100644", "100755"}),
        )
        if hashlib.sha256(committed_bytes).hexdigest() != entry["sha256"]:
            raise ValueError(
                "implementation source differs from its recorded commit: "
                f"{entry['path']}"
            )


def observe_implementation_sources(
    registry: dict[str, Any],
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    if registry.get("status") != EXECUTABLE_MATRIX_STATUS:
        return _parent_observe_implementation_sources(
            registry,
            repository_root,
        )
    source_commit = registry.get("implementation_source_commit")
    records: list[dict[str, Any]] = []
    for entry in registry["implementation_sources"]:
        observed_sha256: str | None = None
        if isinstance(source_commit, str):
            try:
                committed_bytes, _metadata = _commit_regular_blob(
                    repository_root,
                    source_commit,
                    entry["path"],
                    f"implementation source {entry['path']}",
                    allowed_modes=frozenset({"100644", "100755"}),
                )
            except ValueError:
                pass
            else:
                observed_sha256 = hashlib.sha256(committed_bytes).hexdigest()
        matches = observed_sha256 == entry["sha256"]
        records.append(
            {
                "path": entry["path"],
                "expected_sha256": entry["sha256"],
                "observed_sha256": observed_sha256,
                "matches_recorded_source": matches,
                "matches_draft_observation": matches,
            }
        )
    matching_count = sum(
        record["matches_recorded_source"] for record in records
    )
    missing_count = sum(
        record["observed_sha256"] is None for record in records
    )
    return {
        "observation_authority": "recorded_implementation_source_commit",
        "observation_commit": source_commit,
        "inventory_count": len(records),
        "matching_count": matching_count,
        "drift_count": len(records) - matching_count,
        "missing_count": missing_count,
        "all_match": matching_count == len(records),
        "records": records,
    }


def _v3_records(stdout: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = _parent._json_records(stdout, TRACE_CASE_PREFIX)
    summaries = _parent._json_records(stdout, TRACE_SUMMARY_PREFIX)
    if len(cases) != EXPECTED_TRACE_CASE_COUNT:
        raise ValueError(
            "accepted-state trace evidence must contain exactly "
            f"{EXPECTED_TRACE_CASE_COUNT} case records"
        )
    if len(summaries) != 1:
        raise ValueError(
            "accepted-state trace evidence must contain one summary record"
        )
    return cases, summaries[0]


def _validate_v3_floor_fields(
    cases: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    for ordinal, case in enumerate(cases):
        label = f"trace case {ordinal}"
        if set(case) != EXPECTED_TRACE_CASE_FIELDS:
            missing = sorted(EXPECTED_TRACE_CASE_FIELDS - set(case))
            extra = sorted(set(case) - EXPECTED_TRACE_CASE_FIELDS)
            raise ValueError(
                f"{label} fields changed: missing={missing}, extra={extra}"
            )
        required_floor = _parent._finite_real(
            case.get("required_minimum_energy_ratio"),
            f"{label} required minimum energy ratio",
        )
        stored_floor = _parent._finite_real(
            case.get("finite_sample_energy_lower_bound"),
            f"{label} finite-sample lower bound",
        )
        grouped_ratio = _parent._finite_real(
            case.get("grouped_symmetric_ratio"),
            f"{label} grouped symmetric ratio",
        )
        trace_ratio = _parent._finite_real(
            case.get("trace_to_penalty_ratio"),
            f"{label} trace-to-penalty ratio",
        )
        upper_bound = _parent._finite_real(
            case.get("trace_upper_bound"),
            f"{label} trace upper bound",
        )
        penalty = _parent._finite_real(
            case.get("effective_penalty_multiplier"),
            f"{label} effective penalty multiplier",
        )
        if required_floor != METHOD_ENERGY_FLOOR:
            raise ValueError(f"{label} required energy floor is not exact")
        if stored_floor != METHOD_ENERGY_FLOOR:
            raise ValueError(f"{label} stored energy floor is not exact")
        if grouped_ratio > SAFE_GROUPED_SYMMETRIC_RATIO_CAP:
            raise ValueError(f"{label} exceeds the direct safe risk cap")
        if trace_ratio > SAFE_GROUPED_SYMMETRIC_RATIO_CAP:
            raise ValueError(
                f"{label} reported trace ratio exceeds the direct safe risk cap"
            )
        if upper_bound > DIRECT_SAFE_TRACE_UPPER_BOUND_LIMIT:
            raise ValueError(
                f"{label} trace upper bound exceeds the direct safe limit"
            )
        if penalty != TRACE_PENALTY_GAMMA:
            raise ValueError(f"{label} effective penalty is not exact")
        if upper_bound / penalty > SAFE_GROUPED_SYMMETRIC_RATIO_CAP:
            raise ValueError(
                f"{label} direct trace-to-penalty quotient exceeds the safe cap"
            )

        fraction = _parent._finite_real(
            case.get("target_wall_fraction"),
            f"{label} wall fraction",
        )
        if fraction == 0.0:
            continue
        minimum_eigenvalue = _parent._finite_real(
            case.get("minimum_generalized_eigenvalue"),
            f"{label} minimum generalized eigenvalue",
        )
        tolerance = _parent._finite_real(
            case.get("eigensolver_tolerance"),
            f"{label} eigensolver tolerance",
        )
        gap = _parent._finite_real(
            case.get("sampled_eigenvalue_gap"),
            f"{label} sampled eigenvalue gap",
        )
        if not _parent._close(
            gap,
            minimum_eigenvalue - tolerance - METHOD_ENERGY_FLOOR,
        ):
            raise ValueError(f"{label} sampled eigenvalue gap is inconsistent")

    if set(summary) != EXPECTED_TRACE_SUMMARY_FIELDS:
        missing = sorted(EXPECTED_TRACE_SUMMARY_FIELDS - set(summary))
        extra = sorted(set(summary) - EXPECTED_TRACE_SUMMARY_FIELDS)
        raise ValueError(
            f"trace summary fields changed: missing={missing}, extra={extra}"
        )
    minimum_floor = _parent._finite_real(
        summary.get("minimum_finite_sample_energy_lower_bound"),
        "trace summary minimum finite-sample lower bound",
    )
    method_floor = _parent._finite_real(
        summary.get("method_coercivity_lower_bound"),
        "trace summary method coercivity lower bound",
    )
    if minimum_floor != METHOD_ENERGY_FLOOR:
        raise ValueError("trace summary minimum energy floor is not exact")
    if method_floor != METHOD_ENERGY_FLOOR:
        raise ValueError("trace summary method energy floor is not exact")
    maximum_upper_bound = _parent._finite_real(
        summary.get("maximum_trace_upper_bound"),
        "trace summary maximum trace upper bound",
    )
    if maximum_upper_bound > DIRECT_SAFE_TRACE_UPPER_BOUND_LIMIT:
        raise ValueError("trace summary maximum upper bound exceeds the safe limit")
    if summary.get("uniform_bound_status") != UNIFORM_BOUND_STATUS:
        raise ValueError("trace summary uniform-bound status changed")
    if summary.get("accepted_claim") != ACCEPTED_CLAIM:
        raise ValueError("trace summary accepted claim changed")


def _parent_compatible_trace_stdout(
    cases: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    adapted_cases: list[dict[str, Any]] = []
    for case in cases:
        adapted = copy.deepcopy(case)
        del adapted["required_minimum_energy_ratio"]
        grouped_ratio = float(adapted["grouped_symmetric_ratio"])
        legacy_lower = 1.0 - math.sqrt(grouped_ratio)
        adapted["finite_sample_energy_lower_bound"] = legacy_lower
        if adapted["target_wall_fraction"] != 0.0:
            adapted["minimum_generalized_eigenvalue"] = (
                legacy_lower
                + float(adapted["eigensolver_tolerance"])
                + float(adapted["sampled_eigenvalue_gap"])
            )
        adapted_cases.append(adapted)

    adapted_summary = copy.deepcopy(summary)
    adapted_summary["minimum_finite_sample_energy_lower_bound"] = min(
        float(case["finite_sample_energy_lower_bound"])
        for case in adapted_cases
    )
    adapted_summary["method_coercivity_lower_bound"] = None
    adapted_summary["uniform_bound_status"] = "UNFROZEN_NO_BOUND_INVENTED"
    adapted_summary["accepted_claim"] = "joint_low_level_prerequisite"
    lines = [
        TRACE_CASE_PREFIX + json.dumps(case, sort_keys=True)
        for case in adapted_cases
    ]
    lines.append(
        TRACE_SUMMARY_PREFIX + json.dumps(adapted_summary, sort_keys=True)
    )
    return "\n".join(lines)


def parse_trace_evidence(stdout: str) -> dict[str, Any]:
    cases, summary = _v3_records(stdout)
    _validate_v3_floor_fields(cases, summary)
    adapted_stdout = _parent_compatible_trace_stdout(cases, summary)

    saved_case_fields = _parent.EXPECTED_TRACE_CASE_FIELDS
    saved_summary_fields = _parent.EXPECTED_TRACE_SUMMARY_FIELDS
    try:
        _parent.EXPECTED_TRACE_CASE_FIELDS = set(_V2_TRACE_CASE_FIELDS)
        _parent.EXPECTED_TRACE_SUMMARY_FIELDS = set(
            _V2_TRACE_SUMMARY_FIELDS
        )
        evidence = _parent_parse_trace_evidence(adapted_stdout)
    finally:
        _parent.EXPECTED_TRACE_CASE_FIELDS = saved_case_fields
        _parent.EXPECTED_TRACE_SUMMARY_FIELDS = saved_summary_fields

    cases_by_id = {case["case_id"]: copy.deepcopy(case) for case in cases}
    evidence["cases"] = [
        cases_by_id[case["case_id"]] for case in evidence["cases"]
    ]
    evidence["summary_record"] = copy.deepcopy(summary)
    evidence["requested_claim"] = ACCEPTED_CLAIM
    evidence["method_coercivity_lower_bound"] = METHOD_ENERGY_FLOOR
    evidence["uniform_bound_status"] = UNIFORM_BOUND_STATUS
    evidence["minimum_finite_sample_energy_lower_bound"] = (
        METHOD_ENERGY_FLOOR
    )
    evidence["direct_safe_grouped_ratio_cap"] = (
        SAFE_GROUPED_SYMMETRIC_RATIO_CAP
    )
    evidence["floor_binding_boundary"] = {
        "collective_policy_signature": True,
        "certificate_cache_digest": True,
        "emitted_route_digest": False,
        "exact_certificate_digest": False,
    }
    return evidence


def evaluate_trace_evidence(stdout: str) -> dict[str, Any]:
    try:
        return parse_trace_evidence(stdout)
    except (
        json.JSONDecodeError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ) as error:
        observed_case_count = sum(
            line.startswith(TRACE_CASE_PREFIX)
            for line in stdout.splitlines()
        )
        return {
            "artifact_schema_version": 1,
            "matrix_id": EXPECTED_MATRIX_ID,
            "requested_claim": ACCEPTED_CLAIM,
            "method_coercivity_lower_bound": METHOD_ENERGY_FLOOR,
            "uniform_bound_status": UNIFORM_BOUND_STATUS,
            "penalty_gamma": TRACE_PENALTY_GAMMA,
            "expected_case_count": EXPECTED_TRACE_CASE_COUNT,
            "observed_case_count": observed_case_count,
            "diagnostics": [str(error)],
            "outcome": "FAIL_METHOD",
        }


def _inject_claim_boundary(value: dict[str, Any]) -> None:
    value["qualification_scope"] = EXPECTED_SCOPE
    value["requested_claim"] = ACCEPTED_CLAIM
    value["qualification_disposition"] = copy.deepcopy(
        EXPECTED_DISPOSITION
    )
    value["open_outcomes"] = copy.deepcopy(EXPECTED_OPEN_OUTCOMES)
    value["method_coercivity_lower_bound"] = METHOD_ENERGY_FLOOR
    value["uniform_bound_status"] = UNIFORM_BOUND_STATUS
    value["certified_aggregate_trace_contract"] = copy.deepcopy(
        EXPECTED_TRACE_CONTRACT
    )
    value["v2_parent_artifacts"] = [
        {"path": path, "sha256": digest}
        for path, digest in EXPECTED_V2_PARENT_SHA256.items()
    ]
    if _parent._frozen_qualification_bundle_binding is not None:
        value["qualification_bundle_binding"] = copy.deepcopy(
            _parent._frozen_qualification_bundle_binding
        )


def write_json(path: Path, value: Any) -> None:
    _parent_write_json(path, value)


def write_text(path: Path, value: str) -> None:
    if path.name == "record.md":
        evidence = _parent._load_trace_evidence(path.parent)
        value += (
            "\n## Accepted-state symmetric-Nitsche prerequisite\n\n"
            + EXPECTED_SCOPE
            + "\n\n"
            "The accepted-state floor is exactly `0.25` for the supported "
            "production subform. The emitted route digest and exact "
            "certificate digest do not bind that floor; policy-signature and "
            "certificate-cache provenance do. FSR-16, FSR-07, WP-3, WP-7, "
            "and Q1 remain open.\n"
        )
        if evidence is not None:
            value += (
                "\n- Aggregate-trace evidence: "
                f"**{evidence.get('outcome', 'INVALID')}**\n"
                f"- Accepted cases: {evidence.get('observed_case_count')}\n"
                "- Maximum trace upper bound: "
                f"{evidence.get('maximum_trace_upper_bound')}\n"
                "- Minimum accepted-state energy floor: "
                f"{evidence.get('minimum_finite_sample_energy_lower_bound')}\n"
                "- Minimum conservative sampled eigenvalue gap: "
                f"{evidence.get('minimum_sampled_eigenvalue_gap')}\n"
            )
    _parent._shared_write_text(path, value)


PARENT_RECORD_TITLE = (
    "WP-3/WP-7 certified aggregate-trace prerequisite qualification record"
)
V3_RECORD_TITLE = (
    "WP-3/WP-7 accepted-state coercivity-policy prerequisite qualification "
    "record"
)
_shared_run_qualification = _parent.strict_runner.run_qualification


def _run_v3_qualification(*args: Any, **kwargs: Any) -> Any:
    if kwargs.get("record_title") != PARENT_RECORD_TITLE:
        raise RuntimeError("inherited qualification record title changed")
    kwargs["record_title"] = V3_RECORD_TITLE
    return _shared_run_qualification(*args, **kwargs)


def requested_claim(arguments: list[str]) -> tuple[str, bool, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--requested-claim", default=ACCEPTED_CLAIM)
    parser.add_argument("--validate-only", action="store_true")
    parsed, remaining = parser.parse_known_args(arguments)
    claim = parsed.requested_claim
    rejected = set(EXPECTED_CLOSURE_REQUEST_POLICY["rejected_claims"])
    if claim in rejected:
        raise ValueError(
            f"requested claim {claim!r} is outside this matrix: "
            f"{EXPECTED_CLOSURE_REQUEST_POLICY['diagnostic']}"
        )
    if claim != ACCEPTED_CLAIM:
        raise ValueError(
            f"unsupported V3 requested claim {claim!r}; "
            f"expected {ACCEPTED_CLAIM!r}"
        )
    return claim, parsed.validate_only, remaining


def validate_only_summary(
    registry: dict[str, Any],
    claim: str,
) -> dict[str, Any]:
    source_observation = observe_implementation_sources(registry)
    draft = registry["status"] == EXPECTED_MATRIX_STATUS
    binding = _parent._frozen_qualification_bundle_binding
    bundle_commit = (
        binding.get("qualification_bundle_commit")
        if isinstance(binding, dict)
        else None
    )
    validation_head = (
        binding.get("validation_head_commit")
        if isinstance(binding, dict)
        else None
    )
    at_bundle_commit = (
        not draft
        and isinstance(bundle_commit, str)
        and validation_head == bundle_commit
    )
    return {
        "matrix_id": registry["matrix_id"],
        "status": registry["status"],
        "execution_ready": at_bundle_commit,
        "validation_scope": (
            "draft_structure_and_dependency_validation"
            if draft
            else (
                "frozen_execution_preflight"
                if at_bundle_commit
                else "frozen_historical_validation"
            )
        ),
        "qualification_bundle_commit": bundle_commit,
        "validation_head_commit": validation_head,
        "execution_HEAD_must_equal_bundle_commit": True,
        "implementation_source_observation": source_observation,
        "requested_claim": claim,
        "group_count": len(registry["groups"]),
        "test_count": sum(len(group["tests"]) for group in registry["groups"]),
        "quantitative_evidence_gate_count": len(
            registry["quantitative_evidence"]
        ),
        "expected_trace_case_count": EXPECTED_TRACE_CASE_COUNT,
        "method_coercivity_lower_bound": METHOD_ENERGY_FLOOR,
        "uniform_bound_status": UNIFORM_BOUND_STATUS,
        "qualification_disposition": copy.deepcopy(EXPECTED_DISPOSITION),
        "open_outcomes": copy.deepcopy(EXPECTED_OPEN_OUTCOMES),
        "closure_outcome": (
            "OPEN_ACCEPTED_STATE_COERCIVITY_POLICY_PREREQUISITE"
        ),
        "outcome": (
            "PASS_DRAFT_STRUCTURE_ONLY"
            if draft and source_observation["all_match"]
            else (
                "DRAFT_SOURCE_DRIFT"
                if draft
                else (
                    "PASS_FROZEN_VALIDATION"
                    if source_observation["all_match"]
                    else "FROZEN_SOURCE_DRIFT"
                )
            )
        ),
    }


def main(arguments: list[str] | None = None) -> int:
    _parent.load_registry = globals()["load_registry"]
    _parent.requested_claim = globals()["requested_claim"]
    _parent.validate_only_summary = globals()["validate_only_summary"]
    saved_run_qualification = _parent.strict_runner.run_qualification
    try:
        _parent.strict_runner.run_qualification = _run_v3_qualification
        return _parent_main(arguments)
    finally:
        _parent.strict_runner.run_qualification = saved_run_qualification


sha256_file = _parent.sha256_file
normalized_registry_bytes = _parent.normalized_registry_bytes
normalized_registry_sha256 = _parent.normalized_registry_sha256
parse_json_document = _parent.parse_json_document
coerce_quantitative_value = _parent.coerce_quantitative_value
run_monitored = _parent.run_monitored
monitored_test_discovery = _parent.monitored_test_discovery
strict_runner = _parent.strict_runner
_json_records = _parent._json_records
_finite_real = _parent._finite_real
_integer = _parent._integer
_close = _parent._close

_parent.validate_v2_contract = validate_v3_contract
_parent.validate_frozen_qualification_bundle = (
    validate_frozen_qualification_bundle
)
_parent.load_registry = load_registry
_parent.parse_trace_evidence = parse_trace_evidence
_parent.evaluate_trace_evidence = evaluate_trace_evidence
_parent._inject_claim_boundary = _inject_claim_boundary
_parent.write_json = write_json
_parent.write_text = write_text
_parent.requested_claim = requested_claim
_parent.validate_only_summary = validate_only_summary
_parent.observe_implementation_sources = observe_implementation_sources
_parent.require_execution_resource_preflight = (
    require_execution_resource_preflight
)
_parent.run_build_phase = run_build_phase
strict_runner.load_registry = load_registry
strict_runner.write_json = write_json
strict_runner.write_text = write_text
strict_runner.untracked_source_record = untracked_source_record
strict_runner.run_build_phase = run_build_phase


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
