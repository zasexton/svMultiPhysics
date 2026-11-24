/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Core/EnvironmentNames.h"

#include "Core/OptionSchema.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace environment {

namespace {

// The catalogue: every environment name the solver reads, grouped by owning
// service.  A name is read through the registry only; a reader that is
// deleted takes its entry with it.
std::span<const NameDeclaration> solverCatalogue() noexcept
{
    [[maybe_unused]] constexpr auto Diagnostic = NameKind::Diagnostic;
    [[maybe_unused]] constexpr auto Performance = NameKind::Performance;
    [[maybe_unused]] constexpr auto TestHook = NameKind::TestHook;
    [[maybe_unused]] constexpr auto NumericalOverride = NameKind::NumericalOverride;
    [[maybe_unused]] constexpr auto Flag = ValueType::Flag;
    [[maybe_unused]] constexpr auto Integer = ValueType::Integer;
    [[maybe_unused]] constexpr auto Real = ValueType::Real;
    [[maybe_unused]] constexpr auto Text = ValueType::Text;
    [[maybe_unused]] constexpr auto List = ValueType::List;
    [[maybe_unused]] constexpr auto Reference = RunStatus::Reference;
    [[maybe_unused]] constexpr auto NonReference = RunStatus::NonReference;

    // Owning services.
    constexpr const char* kDriver = "Application/Core/ApplicationDriver";
    constexpr const char* kCutConfiguration = "Application/Core/LevelSetCutConfiguration";
    constexpr const char* kSimulationBuilder = "Application/Core/SimulationBuilder";
    constexpr const char* kAnalysis = "FE/Analysis";
    constexpr const char* kCertificate = "FE/Analysis/ConstraintRangeCertificate";
    constexpr const char* kRowObserver = "FE/Analysis/RowContributionObserver";
    constexpr const char* kAssembly = "FE/Assembly";
    constexpr const char* kEigen = "FE/Backends/Eigen";
    constexpr const char* kFsils = "FE/Backends/FSILS";
    constexpr const char* kConstraints = "FE/Constraints";
    constexpr const char* kAggregation = "FE/Constraints/SmallCutAggregationConstraint";
    constexpr const char* kCore = "FE/Core";
    constexpr const char* kJit = "FE/Forms/JIT";
    constexpr const char* kInterfaces = "FE/Interfaces";
    constexpr const char* kLevelSetExtension = "FE/LevelSet/Extension";
    constexpr const char* kLevelSetGeometry = "FE/LevelSet/LevelSetGeometryService";
    constexpr const char* kLevelSetMaintenance = "FE/LevelSet/Maintenance";
    constexpr const char* kSystems = "FE/Systems";
    constexpr const char* kBoundaryReduction = "FE/Systems/BoundaryReductionService";
    constexpr const char* kFESystem = "FE/Systems/FESystem";
    constexpr const char* kFormsInstaller = "FE/Systems/FormsInstaller";
    constexpr const char* kSystemAssembly = "FE/Systems/SystemAssembly";
    constexpr const char* kSystemSetup = "FE/Systems/SystemSetup";
    constexpr const char* kGeneratedState = "FE/TimeStepping/GeneratedState";
    constexpr const char* kNewton = "FE/TimeStepping/NewtonSolver";
    constexpr const char* kTimeLoop = "FE/TimeStepping/TimeLoop";
    constexpr const char* kPhysicsJit = "Physics/Core/JITRuntimePolicy";

    static constexpr NameDeclaration kSolverNames[] = {
        // Application/Core/ApplicationDriver: the solve setup and the step loop.
        {"SVMP_CUT_RULE_DUMP", Diagnostic, kDriver, Flag, "off", Reference,
         "dumps every retained generated cut-volume rule per accepted step"},
        {"SVMP_DISABLE_CUT_CONTEXT_REINSTALL", Performance, kDriver, Flag, "off", Reference,
         "rebuilds a replaced cut context at the final candidate gate instead of reinstalling it"},
        {"SVMP_DISABLE_POSTACCEPT_CUT_REUSE", Performance, kDriver, Flag, "off", Reference,
         "rebuilds the cut geometry of the post-accept maintenance baseline and validator"},
        {"SVMP_JACOBIAN_REBUILD_PERIOD", NumericalOverride, kDriver, Integer, "1", NonReference,
         "rebuilds the Newton Jacobian every n iterations"},
        {"SVMP_NEWTON_ACCEPT_INEXACT_LINEAR", NumericalOverride, kDriver, Flag, "off", NonReference,
         "accepts Newton updates from linear solves that missed their tolerance"},
        {"SVMP_NEWTON_LINE_SEARCH", NumericalOverride, kDriver, Flag, "on with active cut or advection requests",
         NonReference, "Newton backtracking line search"},
        {"SVMP_NEWTON_LINE_SEARCH_ALPHA_MIN", NumericalOverride, kDriver, Real, "the Newton options", NonReference,
         "smallest line-search step"},
        {"SVMP_NEWTON_LINE_SEARCH_C1", NumericalOverride, kDriver, Real, "the Newton options", NonReference,
         "sufficient-decrease constant of the line search"},
        {"SVMP_NEWTON_LINE_SEARCH_FAIL_ON_NO_REDUCTION", NumericalOverride, kDriver, Flag, "the Newton options",
         NonReference, "fails a Newton iteration whose line search reduces nothing (the deck key wins)"},
        {"SVMP_NEWTON_LINE_SEARCH_MAX_ITERATIONS", NumericalOverride, kDriver, Integer, "the Newton options",
         NonReference, "line-search trials per Newton iteration (the deck key wins)"},
        {"SVMP_NEWTON_LINE_SEARCH_SHRINK", NumericalOverride, kDriver, Real, "the Newton options", NonReference,
         "step reduction per line-search trial"},
        {"SVMP_NEWTON_PTC", NumericalOverride, kDriver, Flag, "off", NonReference,
         "pseudo-transient continuation in Newton"},
        {"SVMP_NEWTON_PTC_ACTIVATE_ON_LINEAR_FAILURE", NumericalOverride, kDriver, Flag, "the Newton options",
         NonReference, "starts pseudo-transient continuation only after a failed linear solve"},
        {"SVMP_NEWTON_PTC_GAMMA_DROP_TOLERANCE", NumericalOverride, kDriver, Real, "the Newton options",
         NonReference, "continuation scaling below which it is dropped"},
        {"SVMP_NEWTON_PTC_GAMMA_GROWTH", NumericalOverride, kDriver, Real, "the Newton options", NonReference,
         "growth of the continuation scaling on linear-solve retries"},
        {"SVMP_NEWTON_PTC_GAMMA_INITIAL", NumericalOverride, kDriver, Real, "the Newton options", NonReference,
         "initial continuation scaling"},
        {"SVMP_NEWTON_PTC_GAMMA_MAX", NumericalOverride, kDriver, Real, "the Newton options", NonReference,
         "largest continuation scaling"},
        {"SVMP_NEWTON_PTC_MAX_LINEAR_RETRIES", NumericalOverride, kDriver, Integer, "the Newton options",
         NonReference, "linear-solve retries per Newton iteration under continuation"},
        {"SVMP_NEWTON_PTC_UPDATE_FROM_RESIDUAL_RATIO", NumericalOverride, kDriver, Flag, "the Newton options",
         NonReference, "updates the continuation scaling from the residual ratio"},
        {"SVMP_PDE_EXTENSION_FACTORIZATION", Performance, kDriver, Text, "lu_colamd", NonReference,
         "direct factorization of the PDE velocity extension (lu_colamd, ldlt_amd, mumps, mumps_serial)"},
        {"SVMP_TIMELOOP_LAST_STEP_ABSORB_FRACTION", NumericalOverride, kDriver, Real,
         "1e-2 with a step controller, otherwise 0", NonReference,
         "remainder, as a fraction of the step, absorbed into the last step"},
        {"SVMP_VTK_OUTPUT_FINAL_TIME", Diagnostic, kDriver, Flag, "off", Reference,
         "also writes VTK output at the final time"},
        {"SVMP_WET_VOLUME_DRIFT_WARNING", Diagnostic, kDriver, Real, "1e-3", Reference,
         "relative wet-volume drift above which a warning is logged"},

        // Application: the simulation builder and the cut configuration.
        {"SVMP_CUT_RETENTION_FORCE", TestHook, kCutConfiguration, Text, "the request's retention", NonReference,
         "forces the retained generated cut-volume sides (active_and_inactive, active_only)"},
        {"SVMP_DOF_NUMBERING", NumericalOverride, kSimulationBuilder, Text,
         "Morton with FSILS, otherwise sequential", NonReference,
         "DOF numbering (rcm, cuthill-mckee, morton, hilbert, sequential, interleaved, block)"},
        {"SVMP_FSILS_BLOCKSCHUR_MOMENTUM_HAT", NumericalOverride, kSimulationBuilder, Text,
         "ilu-k", NonReference,
         "replaces the BlockSchur momentum approximation (diag-k, blockdiag-k, ilu-k, asm-k)"},
        {"SVMP_FSILS_BLOCKSCHUR_SCHUR_PC", NumericalOverride, kSimulationBuilder, Text,
         "algebraic-shat", NonReference,
         "replaces the BlockSchur Schur preconditioner (diag-l, blockdiag-l, ilu-l, algebraic-shat)"},
        {"SVMP_FSILS_LOCALITY_ORDERING", NumericalOverride, kSimulationBuilder, Text, "morton", NonReference,
         "FSILS node-locality numbering (off, none, interleaved, auto, morton, hilbert, rcm)"},
        {"SVMP_FSILS_REJECT_FILL_BEYOND_HALO", NumericalOverride, kSimulationBuilder, Flag, "off", NonReference,
         "rejects constraint fill columns beyond the ghost layers"},

        // FE/Analysis: the constraint-range certificate and the row-contribution
        // observers.  The six certificate gates (D61 item 7) mark a run as
        // non-reference.
        {"SVMP_ANALYSIS_SCAN_SELF_CHECK", Diagnostic, kAnalysis, Flag, "the first 256 reads", Reference,
         "checks every direct matrix read of the analysis scan against getEntry and throws on a "
         "difference"},
        {"SVMP_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", Diagnostic, kCertificate, Flag, "off", Reference,
         "second spelling of SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC"},
        {"SVMP_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC_EVERY_ASSEMBLY", Performance, kCertificate, Flag,
         "off", Reference,
         "second spelling of SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC_EVERY_ASSEMBLY"},
        {"SVMP_FREE_SURFACE_CONSTANT_PRESSURE_KKT_MAX_RELATIVE_DISTANCE", NumericalOverride, kCertificate, Real,
         "no gate", NonReference,
         "second spelling of SVMP_NS_FREE_SURFACE_CONSTANT_PRESSURE_KKT_MAX_RELATIVE_DISTANCE"},
        {"SVMP_FREE_SURFACE_EQUILIBRIUM_DIAGNOSTIC", Diagnostic, kCertificate, Flag, "off", Reference,
         "fourth spelling of SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC"},
        {"SVMP_FREE_SURFACE_PRESSURE_REPRESENTABILITY_MAX_RELATIVE_DISTANCE", NumericalOverride, kCertificate,
         Real, "no gate", NonReference,
         "second spelling of SVMP_NS_FREE_SURFACE_PRESSURE_REPRESENTABILITY_MAX_RELATIVE_DISTANCE"},
        {"SVMP_FREE_SURFACE_STATIC_COMPATIBLE_PRESSURE_INITIALIZER", NumericalOverride, kCertificate, Flag,
         "off", NonReference,
         "second spelling of SVMP_NS_FREE_SURFACE_STATIC_COMPATIBLE_PRESSURE_INITIALIZER"},
        {"SVMP_NEWTON_FIELD_RESIDUAL_DIAGNOSTIC", Diagnostic, kRowObserver, Flag, "off", Reference,
         "logs the residual of one field per Newton assembly"},
        {"SVMP_NEWTON_FIELD_RESIDUAL_FIELD", Diagnostic, kRowObserver, Text, "Pressure", Reference,
         "field of the field-residual log"},
        {"SVMP_NEWTON_FIELD_RESIDUAL_SAMPLE_DOFS", Diagnostic, kRowObserver, List, "none", Reference,
         "global DOFs sampled by the field-residual log"},
        {"SVMP_NEWTON_MATRIX_SUPPORT_SAMPLE_DOFS", Diagnostic, kRowObserver, List, "none", Reference,
         "global DOFs whose matrix support is logged"},
        {"SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", Diagnostic, kCertificate, Flag, "off",
         Reference,
         "logs the constraint-range certificate (representability and constant-field KKT distances); "
         "the first set spelling of the four decides"},
        {"SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC_EVERY_ASSEMBLY", Performance, kCertificate, Flag,
         "off", Reference, "recomputes the certificate's pair matrix and least-squares solve for every sample"},
        {"SVMP_NS_FREE_SURFACE_CONSTANT_PRESSURE_KKT_MAX_RELATIVE_DISTANCE", NumericalOverride, kCertificate,
         Real, "no gate", NonReference,
         "accepts a converged step only if the constant-field KKT distance is at most this value"},
        {"SVMP_NS_FREE_SURFACE_EQUILIBRIUM_DIAGNOSTIC", Diagnostic, kCertificate, Flag, "off", Reference,
         "third spelling of SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC"},
        {"SVMP_NS_FREE_SURFACE_PRESSURE_REPRESENTABILITY_MAX_RELATIVE_DISTANCE", NumericalOverride, kCertificate,
         Real, "no gate", NonReference,
         "accepts a converged step only if the representability distance is at most this value"},
        {"SVMP_NS_FREE_SURFACE_STATIC_COMPATIBLE_PRESSURE_INITIALIZER", NumericalOverride, kCertificate, Flag,
         "off", NonReference, "replaces the initial guess by the static-compatible least-squares correction"},
        {"SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC", Diagnostic, kRowObserver, Flag, "off", Reference,
         "logs the declared row-contribution operators per assembly, and the field-residual log"},
        {"SVMP_NS_PRESSURE_ROW_CONTRIBUTION_MATRIX_DIAGNOSTIC", Diagnostic, kRowObserver, Flag, "off",
         Reference, "adds the matrix support of the sampled rows to the row-contribution log"},
        {"SVMP_NS_PRESSURE_ROW_CONTRIBUTION_MATRIX_SUMMARY_DIAGNOSTIC", Diagnostic, kRowObserver, Flag, "off",
         Reference, "adds matrix summaries of the row-contribution operators"},
        {"SVMP_NS_PRESSURE_ROW_CONTRIBUTION_MATRIX_SUMMARY_OPS", Diagnostic, kRowObserver, List,
         "every operator", Reference, "operators of the matrix summaries"},
        {"SVMP_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC", Diagnostic, kRowObserver, Flag, "off", Reference,
         "second spelling of SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC"},
        {"SVMP_PRESSURE_ROW_CONTRIBUTION_MATRIX_DIAGNOSTIC", Diagnostic, kRowObserver, Flag, "off", Reference,
         "second spelling of SVMP_NS_PRESSURE_ROW_CONTRIBUTION_MATRIX_DIAGNOSTIC"},
        {"SVMP_PRESSURE_ROW_CONTRIBUTION_MATRIX_SUMMARY_DIAGNOSTIC", Diagnostic, kRowObserver, Flag, "off",
         Reference, "second spelling of SVMP_NS_PRESSURE_ROW_CONTRIBUTION_MATRIX_SUMMARY_DIAGNOSTIC"},
        {"SVMP_PRESSURE_ROW_CONTRIBUTION_MATRIX_SUMMARY_OPS", Diagnostic, kRowObserver, List,
         "every operator", Reference, "second spelling of SVMP_NS_PRESSURE_ROW_CONTRIBUTION_MATRIX_SUMMARY_OPS"},
        {"SVMP_PRESSURE_ROW_CONTRIBUTION_SAMPLE_DOFS", Diagnostic, kRowObserver, List, "none", Reference,
         "second spelling of SVMP_NEWTON_FIELD_RESIDUAL_SAMPLE_DOFS; setting it also adds the matrix "
         "support log"},

        // FE/Assembly and FE/Systems/SystemAssembly: assembly and cut-cell numerics.
        {"SVMP_ASSEMBLY_SPIN_US", Performance, kAssembly, Integer, "0", Reference,
         "busy-wait of a threaded-assembly worker before it blocks, in microseconds"},
        {"SVMP_ASSEMBLY_THREAD_TIMING", Diagnostic, kAssembly, Flag, "off", Reference,
         "prints the wall time of each assembly loop and of the threaded pipeline"},
        {"SVMP_ASSEMBLY_TIMING", Diagnostic, kAssembly, Flag, "off", Reference,
         "assembly phase timers; also the cut-volume and interior-face timers"},
        {"SVMP_CUT_VOLUME_BASIS_CACHE_MAX_ENTRIES", Performance, kAssembly, Integer, "0 (off)", Reference,
         "entries of the cut-volume basis tabulation cache"},
        {"SVMP_CUT_VOLUME_EPOCH_CACHE_MAX_MB", Performance, kAssembly, Integer, "1024", Reference,
         "memory budget of the cut-volume cache kept across assemblies, in MB"},
        {"SVMP_CUT_VOLUME_TIMING", Diagnostic, kAssembly, Flag, "off", Reference,
         "prints the cut-volume assembly timing"},
        {"SVMP_DISABLE_CUT_VOLUME_FUSION", Performance, kSystemAssembly, Flag, "off", NonReference,
         "assembles each cut-volume term in its own pass (changes the summation order)"},
        {"SVMP_FE_COMPARE_MONOLITHIC_COMPILED", Diagnostic, kAssembly, Flag, "off", Reference,
         "compares compiled monolithic cell kernels with the interpreted ones and throws on a mismatch"},
        {"SVMP_FE_COMPARE_MONOLITHIC_MAX_CELLS", Diagnostic, kAssembly, Integer, "1", Reference,
         "cells compared by SVMP_FE_COMPARE_MONOLITHIC_COMPILED"},
        {"SVMP_FE_COMPARE_MONOLITHIC_TOL", Diagnostic, kAssembly, Real, "1e-11", Reference,
         "tolerance of SVMP_FE_COMPARE_MONOLITHIC_COMPILED"},
        {"SVMP_FE_CUT_VOLUME_ROW_PROVENANCE_DIAGNOSTIC", Diagnostic, kSystemAssembly, Flag, "off", Reference,
         "logs which cut-volume rules contribute to each row"},
        {"SVMP_FE_CUT_VOLUME_ROW_PROVENANCE_MAX_RULES", Diagnostic, kSystemAssembly, Integer, "0 (unlimited)",
         Reference, "caps the rules of the row provenance log"},
        {"SVMP_FE_CUT_VOLUME_ROW_PROVENANCE_OPERATOR", Diagnostic, kSystemAssembly, List, "all operators",
         Reference, "operator tags of the row provenance log, separated by | or ,"},
        {"SVMP_FE_DISABLE_DIRICHLET_FAST_PATH", Performance, kAssembly, Flag, "off", NonReference,
         "sends cells whose DOFs are all Dirichlet through the general constraint distributor"},
        {"SVMP_FE_DISABLE_MONOLITHIC_COMPILED_DISPATCH", Performance, kAssembly, Flag, "off", Reference,
         "vetoes SVMP_FE_ENABLE_MONOLITHIC_COMPILED_DISPATCH"},
        {"SVMP_FE_ENABLE_MONOLITHIC_COMPILED_DISPATCH", Performance, kAssembly, Flag, "off", NonReference,
         "JIT-compiled monolithic coupled cell kernels (changes round-off)"},
        {"SVMP_GHOST_TRACE_ROW_DOF", Diagnostic, kAssembly, Integer, "off", Reference,
         "traces the ghost contributions sent to this global row DOF"},
        {"SVMP_INTERIOR_FACE_TIMING", Diagnostic, kAssembly, Flag, "off", Reference,
         "prints the interior-face assembly timing"},
        {"SVMP_MIN_GENERATED_CUT_VOLUME_FRACTION", NumericalOverride, kAssembly, Real, "1e-8", NonReference,
         "volume fraction below which a generated cut sub-volume rule is pruned"},
        {"SVMP_MONO_AUX_TRACE_DOF_GATHER", Diagnostic, kAssembly, Flag, "off", Reference,
         "prints the norms of functional-assembler coefficient gathers"},
        {"SVMP_PARALLEL_ASSEMBLER_TRACE", Diagnostic, kAssembly, Flag, "off", Reference,
         "prints per-rank ghost exchange statistics"},

        // FE/Backends/Eigen: the Eigen matrix backend.
        {"SVMP_EIGEN_COUNT_DROPPED", Diagnostic, kEigen, Flag, "off", Reference,
         "logs samples of matrix writes outside the sparsity pattern"},
        {"SVMP_EIGEN_STRICT_PATTERN", Diagnostic, kEigen, Flag, "off", Reference,
         "throws on a matrix write outside the sparsity pattern instead of dropping it"},
        {"SVMP_FE_EIGEN_FACTOR_DIAGNOSTICS", Diagnostic, kEigen, Flag, "off", Reference,
         "prints the direct-solver factorization diagnostics; the PDE velocity extension then "
         "factorizes on every call"},

        // FE/Backends/FSILS: the FSILS linear-solver backend and its solvers.
        {"SVMP_DISABLE_REDUCED_FACE_CACHE_ADD_BC_MUL", Performance, kFsils, Flag, "off", NonReference,
         "turns off the face-cache products of reduced coupled-boundary updates in add_bc_mul "
         "(changes round-off)"},
        {"SVMP_FSILS_ADD_BC_MUL_DUMP_MAX_CALLS", Diagnostic, kFsils, Integer, "80", Reference,
         "caps the reduced add_bc_mul dumps (negative: unlimited)"},
        {"SVMP_FSILS_ADD_BC_MUL_DUMP_PREFIX", Diagnostic, kFsils, Text, "off", Reference,
         "dumps the reduced-update products applied in add_bc_mul under this prefix"},
        {"SVMP_FSILS_ASSUME_TRANSPOSE_SADDLE", NumericalOverride, kFsils, Flag, "off", NonReference,
         "overwrites the constraint block D with -G^T before the BlockSchur solve"},
        {"SVMP_FSILS_BLOCKSCHUR_CONSTRAINT_MEAN_DOMINANCE", Diagnostic, kFsils, Real, "1", Reference,
         "threshold of the trace line on a dominant constraint mean in the BlockSchur solution"},
        {"SVMP_FSILS_BLOCKSCHUR_FORCE_SCHUR_BICGSTAB", NumericalOverride, kFsils, Flag, "off", NonReference,
         "forces BiCGStab for the Schur Krylov solve"},
        {"SVMP_FSILS_BLOCKSCHUR_FORCE_SCHUR_GMRES", NumericalOverride, kFsils, Flag, "off", NonReference,
         "forces GMRES for the Schur Krylov solve"},
        {"SVMP_FSILS_BLOCKSCHUR_FORCE_SCHUR_PC", NumericalOverride, kFsils, Text, "the configured Schur preconditioner",
         NonReference, "replaces the Schur preconditioner (diag-l, blockdiag-l, ilu-l, algebraic-shat)"},
        {"SVMP_FSILS_BLOCKSCHUR_MIN_OUTER_ITERS", NumericalOverride, kFsils, Integer, "0", NonReference,
         "minimum BlockSchur outer iterations before convergence is accepted"},
        {"SVMP_FSILS_BLOCKSCHUR_MOMINV_DROP_SCORE", NumericalOverride, kFsils, Real, "0 (no drop)", NonReference,
         "drop score of off-diagonal blocks of the sparse momentum-inverse approximation"},
        {"SVMP_FSILS_BLOCKSCHUR_MOMINV_MAX_OFFDIAG", NumericalOverride, kFsils, Integer, "-1 (unlimited)",
         NonReference, "off-diagonal blocks kept per row of the momentum-inverse approximation"},
        {"SVMP_FSILS_BLOCKSCHUR_OUTER_CAP", NumericalOverride, kFsils, Text,
         "50 (200 with coupled updates; unlimited with the coupled outer path)", NonReference,
         "caps the BlockSchur outer iterations (a positive integer, or none, unlimited, off)"},
        {"SVMP_FSILS_BLOCKSCHUR_SCHUR_FIXED_ITERS", NumericalOverride, kFsils, Integer, "0 (off)", NonReference,
         "forces at least this many Schur BiCGStab iterations"},
        {"SVMP_FSILS_BLOCKSCHUR_SCHUR_INIT_PRECOND_RHS", NumericalOverride, kFsils, Flag, "off", NonReference,
         "starts the Schur Krylov solve from the preconditioned right-hand side"},
        {"SVMP_FSILS_BLOCKSCHUR_SCHUR_PC_REUSE_SOLVES", Performance, kFsils, Integer, "1", Reference,
         "Schur solves on a cached Schur preconditioner before a forced rebuild (-1: never, 0: every "
         "solve)"},
        {"SVMP_FSILS_BLOCKSCHUR_SHAT_GUARANTEED_OFFDIAG", NumericalOverride, kFsils, Integer, "2", NonReference,
         "off-diagonal blocks always kept per row when the Schur approximations are pruned"},
        {"SVMP_FSILS_BLOCKSCHUR_SHAT_MAX_OFFDIAG", NumericalOverride, kFsils, Integer, "-1 (unlimited)",
         NonReference, "off-diagonal blocks kept per row of the algebraic Schur approximation"},
        {"SVMP_FSILS_BLOCKSCHUR_SHAT_OPERATOR_REFINE_OMEGA", NumericalOverride, kFsils, Real, "0.5", NonReference,
         "damping of the refinement steps of the algebraic Schur preconditioner"},
        {"SVMP_FSILS_BLOCKSCHUR_SHAT_OPERATOR_REFINE_STEPS", NumericalOverride, kFsils, Integer, "1", NonReference,
         "refinement steps of the algebraic Schur preconditioner"},
        {"SVMP_FSILS_BLOCKSCHUR_SHAT_ROW_DROP_SCORE", NumericalOverride, kFsils, Real, "0 (keep all)", NonReference,
         "drop score of off-diagonal entries of the algebraic Schur approximation"},
        {"SVMP_FSILS_BLOCKSCHUR_TRUE_RESIDUAL_RETRY_MAX_ITER", NumericalOverride, kFsils, Integer, "2000",
         NonReference, "iteration cap of the tightened BlockSchur true-residual retry"},
        {"SVMP_FSILS_BLOCKSCHUR_TRUE_RESIDUAL_RETRY_REL_SCALE", NumericalOverride, kFsils, Real, "1e-3",
         NonReference, "factor on the relative tolerance of the BlockSchur true-residual retry"},
        {"SVMP_FSILS_DISABLE_BLOCKSCHUR_STAGE_SCALING", NumericalOverride, kFsils, Flag, "off", NonReference,
         "vetoes the stage scaling of SVMP_FSILS_ENABLE_BLOCKSCHUR_STAGE_SCALING"},
        {"SVMP_FSILS_DISABLE_BLOCKSCHUR_TRUE_RESIDUAL_RETRY", NumericalOverride, kFsils, Flag, "off", NonReference,
         "vetoes the BlockSchur true-residual retry"},
        {"SVMP_FSILS_DISABLE_LOW_RANK_RESIDUAL_POLISH", NumericalOverride, kFsils, Flag, "off (polish on)",
         NonReference, "turns off the low-rank residual polish after a solve"},
        {"SVMP_FSILS_DISABLE_MULTI_FACE_LEGACY_GMRES", NumericalOverride, kFsils, Flag, "off", NonReference,
         "uses BiCGStab instead of GMRES in the face-only Schur path with several coupled faces"},
        {"SVMP_FSILS_DISABLE_SCHUR_GLOBAL_MEAN_COARSE", NumericalOverride, kFsils, Flag, "off", NonReference,
         "vetoes SVMP_FSILS_SCHUR_GLOBAL_MEAN_COARSE"},
        {"SVMP_FSILS_DISABLE_SCHUR_PARTITION_COARSE", NumericalOverride, kFsils, Flag, "off", NonReference,
         "vetoes SVMP_FSILS_SCHUR_PARTITION_COARSE"},
        {"SVMP_FSILS_DISABLE_TRANSPOSE_SADDLE", NumericalOverride, kFsils, Flag, "off", NonReference,
         "vetoes SVMP_FSILS_ASSUME_TRANSPOSE_SADDLE"},
        {"SVMP_FSILS_DUMP_PREPARED_COL_COMPONENT", Diagnostic, kFsils, Integer, "-1 (all)", Reference,
         "column component of the prepared-row dump"},
        {"SVMP_FSILS_DUMP_PREPARED_ROW_COMPONENT", Diagnostic, kFsils, Integer, "-1 (off)", Reference,
         "row component of the prepared-row dump"},
        {"SVMP_FSILS_DUMP_PREPARED_ROW_GLOBAL_NODE", Diagnostic, kFsils, Integer, "-1 (off)", Reference,
         "global node of the prepared-row dump"},
        {"SVMP_FSILS_DUMP_PREPARED_ROW_PREFIX", Diagnostic, kFsils, Text, "off", Reference,
         "output prefix of the prepared-row dump"},
        {"SVMP_FSILS_DUMP_SYSTEM_MAX", Diagnostic, kFsils, Integer, "16", Reference,
         "linear systems dumped"},
        {"SVMP_FSILS_DUMP_SYSTEM_PREFIX", Diagnostic, kFsils, Text, "off", Reference,
         "dumps serial linear systems (matrix, right-hand side, solution) under this prefix"},
        {"SVMP_FSILS_DUMP_SYSTEM_SKIP", Diagnostic, kFsils, Integer, "0", Reference,
         "solves skipped before the system dumps start"},
        {"SVMP_FSILS_ENABLE_BLOCKSCHUR_STAGE_SCALING", NumericalOverride, kFsils, Flag, "off", NonReference,
         "scales momentum rows by the stage time step and constraint columns by its inverse"},
        {"SVMP_FSILS_ENABLE_BLOCKSCHUR_TRUE_RESIDUAL_RETRY", NumericalOverride, kFsils, Flag, "off", NonReference,
         "retries a BlockSchur solve with tighter tolerances when its true residual check fails"},
        {"SVMP_FSILS_ENABLE_MPI_NATIVE_FACE_RANK_ONE", NumericalOverride, kFsils, Flag, "off", NonReference,
         "allows the native-face rank-one coupled-boundary treatment under MPI"},
        {"SVMP_FSILS_FULL_SPMV_HALO", Performance, kFsils, Flag, "off", Reference,
         "keeps the full owned-row halo for matrix-vector products"},
        {"SVMP_FSILS_GMRES_ADAPTIVE", NumericalOverride, kFsils, Flag, "off", NonReference,
         "stagnation stop and adaptive restart in vector GMRES"},
        {"SVMP_FSILS_GMRES_BASIS_PANEL", Performance, kFsils, Flag, "off", NonReference,
         "contiguous Krylov basis panel in vector GMRES (changes round-off)"},
        {"SVMP_FSILS_GMRES_BASIS_PANEL_MAX_MB", Performance, kFsils, Integer, "1024", Reference,
         "memory cap of the GMRES basis panel (0: unlimited)"},
        {"SVMP_FSILS_GMRES_BASIS_PANEL_MIN_SD", Performance, kFsils, Integer, "16", Reference,
         "smallest restart length that uses the GMRES basis panel"},
        {"SVMP_FSILS_GMRES_DEBUG_TRACE", Diagnostic, kFsils, Flag, "off", Reference,
         "prints the first-cycle Arnoldi norms of vector GMRES"},
        {"SVMP_FSILS_GMRES_DGKS_RATIO", NumericalOverride, kFsils, Real, "0.5", NonReference,
         "reorthogonalization ratio of the mgs criterion"},
        {"SVMP_FSILS_GMRES_EXPERIMENTAL_ADAPTIVE", NumericalOverride, kFsils, Flag, "off", NonReference,
         "same as SVMP_FSILS_GMRES_ADAPTIVE"},
        {"SVMP_FSILS_GMRES_FULL_LENGTH_KRYLOV", Performance, kFsils, Flag, "off", Reference,
         "full-length (ghost-including) Krylov loops in vector GMRES"},
        {"SVMP_FSILS_GMRES_NO_ADAPTIVE", NumericalOverride, kFsils, Flag, "off", NonReference,
         "forces the adaptive GMRES controls off"},
        {"SVMP_FSILS_GMRES_ORTH", NumericalOverride, kFsils, Text, "the CGS2 criterion", NonReference,
         "reorthogonalization criterion (mgs, modified, modified-gram-schmidt)"},
        {"SVMP_FSILS_GMRES_OWNED_KRYLOV_CHECK", Diagnostic, kFsils, Flag, "off", Reference,
         "runs every vector GMRES solve with both Krylov loop lengths and throws unless they agree"},
        {"SVMP_FSILS_GMRES_PROFILE", Diagnostic, kFsils, Flag, "SVMP_FSILS_GMRES_RECYCLE_VERBOSE", Reference,
         "prints the vector GMRES timing profile"},
        {"SVMP_FSILS_GMRES_RECYCLE_ADAPTIVE", NumericalOverride, kFsils, Flag, "off", NonReference,
         "adaptive keep or drop of recycled Krylov vectors"},
        {"SVMP_FSILS_GMRES_RECYCLE_DROP_MIN_K", NumericalOverride, kFsils, Integer, "0", NonReference,
         "smallest recycle space kept by the adaptive drop"},
        {"SVMP_FSILS_GMRES_RECYCLE_DROP_REL", NumericalOverride, kFsils, Real, "0.05", NonReference,
         "relative drop threshold of recycled Krylov vectors"},
        {"SVMP_FSILS_GMRES_RECYCLE_DROP_TOL", NumericalOverride, kFsils, Real, "1e-3", NonReference,
         "normalized drop threshold of recycled Krylov vectors"},
        {"SVMP_FSILS_GMRES_RECYCLE_DROP_WINDOW", NumericalOverride, kFsils, Integer, "3", NonReference,
         "low-score restarts before a recycled vector is dropped"},
        {"SVMP_FSILS_GMRES_RECYCLE_K", NumericalOverride, kFsils, Integer, "0 (off)", NonReference,
         "size of the recycled Krylov subspace of vector GMRES"},
        {"SVMP_FSILS_GMRES_RECYCLE_REPLACE_N", NumericalOverride, kFsils, Integer, "0 (about half)", NonReference,
         "recycled vectors replaced per update when the space is full"},
        {"SVMP_FSILS_GMRES_RECYCLE_SCORE_ALPHA", NumericalOverride, kFsils, Real, "0.8", NonReference,
         "smoothing of the recycled-vector scores"},
        {"SVMP_FSILS_GMRES_RECYCLE_UPDATE", NumericalOverride, kFsils, Text, "each restart", NonReference,
         "when the recycle space is updated (off, once, each)"},
        {"SVMP_FSILS_GMRES_RECYCLE_VERBOSE", Diagnostic, kFsils, Flag, "off", Reference,
         "verbose recycle log; also the default of SVMP_FSILS_GMRES_PROFILE"},
        {"SVMP_FSILS_GMRES_REORTH", NumericalOverride, kFsils, Text,
         "selective for MPI vector GMRES with scaling, otherwise off", NonReference,
         "reorthogonalization in GMRES (on, selective, auto, off, none)"},
        {"SVMP_FSILS_GMRES_SD", NumericalOverride, kFsils, Integer, "the Krylov space dimension", NonReference,
         "GMRES restart length"},
        {"SVMP_FSILS_HALO_TRACE", Diagnostic, kFsils, Flag, "off", Reference,
         "logs the owned-row halo plan"},
        {"SVMP_FSILS_MATRIX_DIAGNOSTICS", Diagnostic, kFsils, Flag, "off", Reference,
         "logs prepared-matrix diagnostics per solve"},
        {"SVMP_FSILS_MATRIX_DIAGNOSTICS_EVERY_N", Diagnostic, kFsils, Integer, "1", Reference,
         "logs the matrix diagnostics every n-th solve"},
        {"SVMP_FSILS_MATRIX_DIAGNOSTICS_MAX_RECORDS", Diagnostic, kFsils, Integer, "unlimited", Reference,
         "caps the matrix diagnostic records"},
        {"SVMP_FSILS_MATRIX_LOCALITY_PROFILE", Diagnostic, kFsils, Flag, "off", Reference,
         "logs the column-locality profile of a new matrix"},
        {"SVMP_FSILS_MATRIX_VIEW_TRACE", Diagnostic, kFsils, Flag, "off", Reference,
         "logs matrix-view assembly counts and norms"},
        {"SVMP_FSILS_MEAN_MODE_ROWS", Diagnostic, kFsils, Integer, "off (8 rows when on)", Reference,
         "adds the largest rows to the constraint mean-mode trace"},
        {"SVMP_FSILS_NATIVE_FACE_EXACT_PRE", NumericalOverride, kFsils, Flag, "off", NonReference,
         "exact dense coupling of native coupled faces in the preconditioned boundary operator"},
        {"SVMP_FSILS_NS_DUMP_MAX_SOLVES", Diagnostic, kFsils, Integer, "1", Reference,
         "BlockSchur solves dumped (negative: unlimited)"},
        {"SVMP_FSILS_NS_DUMP_PREFIX", Diagnostic, kFsils, Text, "off", Reference,
         "dumps BlockSchur stage vectors under this prefix"},
        {"SVMP_FSILS_NS_FULL_COLUMN_COMPARE", Diagnostic, kFsils, Flag, "off", Reference,
         "compares first-iteration block columns in the BlockSchur trace"},
        {"SVMP_FSILS_NS_ORACLE_BLOCK_COMPARE_FILE", Diagnostic, kFsils, Text, "off", Reference,
         "compares the block apply with a direct product on an oracle mode from this file"},
        {"SVMP_FSILS_NS_SOLVER_TRACE", Diagnostic, kFsils, Flag, "off", Reference,
         "traces the BlockSchur solver"},
        {"SVMP_FSILS_NS_TRACE", Diagnostic, kFsils, Flag, "off", Reference,
         "traces the BlockSchur solver"},
        {"SVMP_FSILS_PRECOND_REDUCED_DUMP_MAX_CALLS", Diagnostic, kFsils, Integer, "4", Reference,
         "caps the reduced-update preconditioner dumps (negative: unlimited)"},
        {"SVMP_FSILS_PRECOND_REDUCED_DUMP_PREFIX", Diagnostic, kFsils, Text, "off", Reference,
         "dumps the diagonal preconditioner scaling of reduced updates under this prefix"},
        {"SVMP_FSILS_REDUCED_SCHUR_FORCE_GT_SYNC", NumericalOverride, kFsils, Flag, "off", NonReference,
         "forces a halo sync of the G^T columns of the reduced Schur correction"},
        {"SVMP_FSILS_SCHUR_DUMP_MAX_ITERS", Diagnostic, kFsils, Integer, "1", Reference,
         "Schur Krylov iterations dumped"},
        {"SVMP_FSILS_SCHUR_DUMP_MAX_SOLVES", Diagnostic, kFsils, Integer, "1", Reference,
         "Schur solves dumped (negative: unlimited)"},
        {"SVMP_FSILS_SCHUR_DUMP_PREFIX", Diagnostic, kFsils, Text, "off", Reference,
         "dumps the Schur right-hand side, iterates and probes under this prefix"},
        {"SVMP_FSILS_SCHUR_GLOBAL_MEAN_COARSE", NumericalOverride, kFsils, Flag, "off", NonReference,
         "adds a global-mean coarse correction to the Schur preconditioner"},
        {"SVMP_FSILS_SCHUR_PARTITION_COARSE", NumericalOverride, kFsils, Flag, "off", NonReference,
         "adds per-partition coarse modes to the Schur preconditioner (rank-dependent)"},
        {"SVMP_FSILS_SKIP_POST_SOLVE_COMM", NumericalOverride, kFsils, Flag, "off", NonReference,
         "skips the halo syncs after nested BlockSchur solves"},
        {"SVMP_FSILS_SOLVE_VECTOR_TRACE", Diagnostic, kFsils, Flag, "off", Reference,
         "traces per-rank right-hand side and solution statistics"},
        {"SVMP_FSILS_SPMV_PREFETCH", Performance, kFsils, Flag, "on", Reference,
         "software prefetch in the vector matrix-vector product"},
        {"SVMP_FSILS_TRACE", Diagnostic, kFsils, Flag, "off", Reference,
         "FSILS backend and BlockSchur trace lines"},
        {"SVMP_FSILS_TRACE_ADD_ROW_DOF", Diagnostic, kFsils, Integer, "off", Reference,
         "traces every matrix write to this row DOF"},
        {"SVMP_FSILS_TRACE_EXPLICIT_BLOCK_CORRECTIONS", Diagnostic, kFsils, Flag, "off", Reference,
         "traces the explicit low-rank block corrections"},
        {"SVMP_FSILS_TRACE_EXPLICIT_BLOCK_CORRECTIONS_LIMIT", Diagnostic, kFsils, Integer, "6", Reference,
         "caps the explicit-correction trace records"},
        {"SVMP_FSILS_TRACE_NATIVE_FACE_EXACT_PRE", Diagnostic, kFsils, Flag, "off", Reference,
         "prints the dense native-face preconditioner coupling"},
        {"SVMP_FSILS_TRACE_REDUCED_SCHUR_GT_SYNC", Diagnostic, kFsils, Flag, "off", Reference,
         "traces the G^T halo state of the reduced Schur correction"},
        {"SVMP_FSILS_TRACE_SCHUR_OPERATOR_PROBE", Diagnostic, kFsils, Flag, "off", Reference,
         "probes the face-only Schur operator once"},
        {"SVMP_FSILS_TRACE_SCHUR_PRECONDITIONER_PROBES", Diagnostic, kFsils, Flag, "off", Reference,
         "probes the Schur preconditioner on coarse and low-rank vectors"},
        {"SVMP_FSILS_TRACE_SCHUR_SETUP_TIMING", Diagnostic, kFsils, Flag, "off", Reference,
         "traces the Schur preconditioner setup and matrix-view diagnostics"},
        {"SVMP_FSILS_TRACE_SCHUR_SETUP_TIMING_ALL_RANKS", Diagnostic, kFsils, Flag, "off", Reference,
         "extends the Schur setup traces to every rank"},
        {"SVMP_FSILS_TRACE_SYNC_ORDER", Diagnostic, kFsils, Flag, "off", Reference,
         "traces the halo sync order of the BlockSchur outer loop"},
        {"SVMP_FSILS_VALIDATE_ADD_MATRIX", Diagnostic, kFsils, Flag, "off", Reference,
         "checks cached-slot element matrix writes"},
        {"SVMP_FSILS_VALIDATE_BLOCK_LOOKUP", Diagnostic, kFsils, Flag, "off", Reference,
         "checks the block lookup tables of a new matrix"},
        {"SVMP_FSILS_VECTOR_VIEW_TRACE", Diagnostic, kFsils, Flag, "off", Reference,
         "logs vector-view assembly counts and norms"},

        // FE/Constraints: constraint construction, small-cut aggregation and the
        // parallel constraint exchange.
        {"SVMP_ACTIVE_SIDE_LOCAL_SUPPORT", NumericalOverride, kConstraints, Flag, "off", NonReference,
         "marks active-side DOF support rank-locally instead of by owner-routed marks"},
        {"SVMP_ACTIVE_SIDE_SUPPORT_CHECK", Diagnostic, kConstraints, Flag, "off", Reference,
         "compares local and owner-routed active-side support marks and throws on a difference"},
        {"SVMP_AGGREGATION_ALLOW_UNAGGREGATED", NumericalOverride, kAggregation, Flag, "off", NonReference,
         "leaves small-cut candidates free when their island has no root (0 or 1)"},
        {"SVMP_AGGREGATION_DUMP", Diagnostic, kAggregation, Flag, "off", Reference,
         "logs probe vertex coordinates and turns off aggregation reuse"},
        {"SVMP_AGGREGATION_LINEAR_EXTENSION", NumericalOverride, kAggregation, Flag, "off", NonReference,
         "restricts the aggregation extension weights to the linear corner basis (0 or 1)"},
        {"SVMP_AGGREGATION_MAX_LINES", NumericalOverride, kAggregation, Integer, "unlimited", NonReference,
         "caps the aggregation lines of a refresh"},
        {"SVMP_AGGREGATION_SLAVE_ALL_CUT", NumericalOverride, kAggregation, Flag, "off", NonReference,
         "makes every cut-cell DOF an aggregation candidate (0 or 1)"},
        {"SVMP_AGGREGATION_TRACE_PROPOSALS", Diagnostic, kAggregation, Flag, "off", Reference,
         "logs every aggregation root proposal and its outcome"},
        {"SVMP_CONTACT_POINT_VELOCITY_TRACE", Diagnostic, kConstraints, Flag, "off", Reference,
         "logs the contact-point samples after each constraint update"},
        {"SVMP_DISABLE_SMALL_CUT_AGGREGATION_REUSE", Performance, kAggregation, Flag, "off", Reference,
         "recomputes an unchanged aggregation refresh instead of reusing it"},
        {"SVMP_GAUGE_VALIDATE", Diagnostic, kConstraints, Flag, "off", Reference,
         "prints the gauge report and checks the inferred null-space vectors"},
        {"SVMP_LEVEL_SET_ACTIVE_CONSTRAINT_SAMPLE_DOFS", Diagnostic, kConstraints, List, "none", Reference,
         "field-local DOFs whose active-side support and constraint status are logged"},
        {"SVMP_PARALLEL_CONSTRAINTS_ALLGATHER", Performance, kConstraints, Flag, "off", Reference,
         "resolves constraint lines by an all-gather instead of the owner-routed exchange"},
        {"SVMP_PARALLEL_CONSTRAINTS_CHECK", Diagnostic, kConstraints, Flag, "off", Reference,
         "repeats the constraint resolution by an all-gather and throws unless the lines agree"},
        {"SVMP_TRACE_STRONG_DIRICHLET", Diagnostic, kConstraints, Flag, "off", Reference,
         "traces strong Dirichlet constraint events"},

        // FE/Core: the logger and the solver-wide traces and self-checks.
        {"FE_LOG_CONSOLE", Diagnostic, kCore, Flag, "on", Reference,
         "prints log records on the console (false or 0 turns it off)"},
        {"FE_LOG_FILE", Diagnostic, kCore, Text, "none", Reference,
         "also writes the log to this file"},
        {"FE_LOG_LEVEL", Diagnostic, kCore, Text, "INFO", Reference,
         "lowest level logged (DEBUG, INFO, WARNING, ERROR, CRITICAL, OFF)"},
        {"FE_LOG_SHOW_RANK", Diagnostic, kCore, Flag, "on", Reference,
         "prefixes log records with the rank (false or 0 turns it off)"},
        {"FE_LOG_SHOW_TIME", Diagnostic, kCore, Flag, "on", Reference,
         "prefixes log records with the time (false or 0 turns it off)"},
        {"SVMP_FE_KERNEL_TRACE", Diagnostic, kCore, List, "no channels", Reference,
         "kernel trace channels (all, selection, specialization, assembly, capability)"},
        {"SVMP_GEOMETRY_THREADS_SELF_CHECK", Diagnostic, kCore, Flag, "off", Reference,
         "rebuilds threaded cut geometry on one thread and throws unless the two agree"},
        {"SVMP_JIT_TRACE_SPECIALIZATION", Diagnostic, kCore, Flag, "off", Reference,
         "traces JIT kernel specialization decisions"},
        {"SVMP_MPI_COLLECTIVE_TRACE", Diagnostic, kCore, Flag, "off", Reference,
         "prints a line before and after each traced MPI collective"},
        {"SVMP_OOP_SOLVER_TRACE", Diagnostic, kCore, Flag, "off", Reference,
         "verbose solver trace (setup, Newton, linear solvers, constraints, assembly) on every rank"},

        // FE/Forms/JIT: the LLVM JIT of form kernels.
        {"SVMP_CACHE_PROFILE", Performance, kJit, Text, "the detected cache sizes", NonReference,
         "cache sizes the JIT tiles for (L1d:N,L1i:N,L2:N,L3:N in bytes)"},
        {"SVMP_DUMP_DEP_MASK", Diagnostic, kJit, Flag, "off", Reference,
         "prints the dependency-mask breakdown of each lowered kernel"},
        {"SVMP_JIT_BASIS_BAKING", Performance, kJit, Flag, "the JIT options (on)", NonReference,
         "bakes basis values into specialized cell kernels"},
        {"SVMP_JIT_CACHE_DIAGNOSTICS", Diagnostic, kJit, Flag, "the JIT options (off)", Reference,
         "logs JIT kernel cache hits, misses and keys"},
        {"SVMP_JIT_CACHE_DIR", Performance, kJit, Text, "$HOME/.cache/svMultiPhysics/jit_cache", Reference,
         "directory of the on-disk JIT object cache"},
        {"SVMP_JIT_CPU", Performance, kJit, Text, "host", NonReference,
         "JIT target CPU (host, x86-64, x86-64-v2, x86-64-v3, x86-64-v4)"},
        {"SVMP_JIT_DISABLE_MARKED_INTERIOR_FACE", Performance, kJit, Flag, "off", NonReference,
         "uses the interpreted kernel on marked interior faces"},
        {"SVMP_JIT_FP_CONTRACT", Performance, kJit, Text, "default", NonReference,
         "floating-point contraction of JIT code (on, off, default)"},
        {"SVMP_JIT_OPT_LEVEL", Performance, kJit, Integer, "the JIT options", NonReference,
         "LLVM optimization level of JIT kernels (0 to 3)"},
        {"SVMP_JIT_TARGET_AWARE", Performance, kJit, Flag, "on for AVX-512 builds, otherwise off", NonReference,
         "target-aware LLVM pass pipeline"},
        {"SVMP_JIT_TELEMETRY", Diagnostic, kJit, Flag, "off", Reference,
         "prints JIT code-generation telemetry"},
        {"SVMP_SIMD_BATCH", Performance, kJit, Flag, "the JIT options (on)", NonReference,
         "cross-element SIMD batching of linear JIT kernels (0 turns it off)"},

        // FE/Interfaces: interface producers.
        {"SVMP_PRODUCER_ARITHMETIC_EXACT", Performance, kInterfaces, Flag, "off", Reference,
         "computes the interface-producer interval arithmetic exactly, without the filtered fast path"},
        {"SVMP_PRODUCER_ARITHMETIC_SELF_CHECK", Diagnostic, kInterfaces, Flag, "off", Reference,
         "runs the filtered and the exact producer arithmetic and aborts on a difference"},

        // FE/LevelSet: the generated cut geometry, the PDE velocity extension and
        // level-set maintenance.
        {"SVMP_AUDIT_GHOST_LAYERS", Diagnostic, kLevelSetGeometry, Flag, "off", Reference,
         "prints per-rank histograms of cut cells and fragments by ghost layer"},
        {"SVMP_AUDIT_WITHHOLD_GHOST_CUT_LAYERS", TestHook, kLevelSetGeometry, Integer, "off", NonReference,
         "removes the generated cut data of ghost cells beyond this layer"},
        {"SVMP_CUT_MEMORY_REPORT", Diagnostic, kLevelSetGeometry, Flag, "off", Reference,
         "logs the memory of the generated cut geometry at each rebuild phase"},
        {"SVMP_DISABLE_ACTIVE_CUT_REFRESH_CACHE", Performance, kLevelSetGeometry, Flag, "off", Reference,
         "rebuilds the installed cut context instead of keeping, resyncing or reinstalling it"},
        {"SVMP_DISABLE_CUT_CONTEXT_RESYNC", Performance, kLevelSetGeometry, Flag, "off", Reference,
         "rebuilds the cut context when only the field layout changed"},
        {"SVMP_DISABLE_INCREMENTAL_REBUILD", Performance, kLevelSetGeometry, Flag, "off", Reference,
         "rebuilds the cut geometry of every cell instead of reusing unchanged cells"},
        {"SVMP_DISABLE_VERTEX_FIELD_FAST_PATH", Performance, kLevelSetGeometry, Flag, "off", NonReference,
         "evaluates vertex fields by point location instead of nodal reads (changes round-off)"},
        {"SVMP_INCREMENTAL_REBUILD_SELF_CHECK", Diagnostic, kLevelSetGeometry, Flag, "off", Reference,
         "rebuilds each cut snapshot from scratch too and throws unless the two agree"},
        {"SVMP_KEEP_DRY_CELL_POINTS", Performance, kLevelSetGeometry, Flag, "off", Reference,
         "keeps every quadrature point of full cells in the cut snapshot"},
        {"SVMP_RETAIN_RESTORED_CUT_CONTEXT", Performance, kLevelSetGeometry, Flag, "off", Reference,
         "keeps the attempt-entry cut context for a rollback to reinstall"},
        {"SVMP_PDE_EXTENSION_CACHE_ENTRIES", Performance, kLevelSetExtension, Integer,
         "1, or 3 for distributed component solves and multi-rank MUMPS", Reference,
         "cached dry-region factorizations of the PDE velocity extension (1 to 8)"},
        {"SVMP_PDE_EXTENSION_DUMP", Diagnostic, kLevelSetExtension, Text, "off", Reference,
         "writes the extension solution of the first calls under this prefix"},
        {"SVMP_PDE_EXTENSION_DUMP_CALLS", Diagnostic, kLevelSetExtension, Integer, "2", Reference,
         "calls written by SVMP_PDE_EXTENSION_DUMP"},
        {"SVMP_PDE_EXTENSION_FAIL_COMPONENT", TestHook, kLevelSetExtension, Integer, "off", NonReference,
         "makes the solve of this velocity component throw"},
        {"SVMP_PDE_EXTENSION_MUMPS_DIAGNOSTICS", Diagnostic, kLevelSetExtension, Flag, "off", Reference,
         "logs every MUMPS factorization of the extension"},
        {"SVMP_PDE_EXTENSION_REPLICATED_SOLVES", Performance, kLevelSetExtension, Flag, "off", Reference,
         "solves every velocity component on every rank"},
        {"SVMP_PDE_EXTENSION_SELF_CHECK", Diagnostic, kLevelSetExtension, Flag, "off", Reference,
         "repeats each cached or distributed extension solve uncached and throws unless they agree"},
        {"SVMP_REDISTANCING_TRIGGER_CELLS", NumericalOverride, kLevelSetMaintenance, Real, "1", NonReference,
         "interface motion, in cut-cell edges, that triggers interface-preserving redistancing"},
        {"SVMP_TRACE_LEVEL_SET_ADVECTION", Diagnostic, kLevelSetExtension, Flag, "off", Reference,
         "logs the velocity-extension updates of level-set advection, with every MUMPS factorization"},

        // FE/Systems: system setup, form installation and boundary reductions.
        {"SVMP_ASSEMBLY_GHOST_POLICY", NumericalOverride, kSystemSetup, Text, "reverse scatter", NonReference,
         "assembly ghost policy (owned or ownedrowsonly, reverse or reversescatter)"},
        {"SVMP_FE_FORM_BLOCK_DIAGNOSTICS", Diagnostic, kFormsInstaller, Flag, "off", Reference,
         "logs the coupled block dependencies of installed forms"},
        {"SVMP_MONO_AUX_TRACE_DOFS", Diagnostic, kBoundaryReduction, List, "none", Reference,
         "global DOFs whose values are printed by the boundary reductions"},
        {"SVMP_SPARSITY_FULL_REBUILD", Performance, kSystemSetup, Flag, "off", Reference,
         "rebuilds the active sparsity fully when the constraint structure changes"},
        {"SVMP_SPARSITY_REFRESH_CHECK", Diagnostic, kSystemSetup, Flag, "off", Reference,
         "compares the replayed sparsity with a full rebuild and throws on a difference"},
        {"SVMP_TRACE_DOF_PERMUTATION", Diagnostic, kSystemSetup, Flag, "off", Reference,
         "logs why the nodal DOF permutation was or was not built"},
        {"SVMP_MONO_AUX_TRACE", Diagnostic, kSystems, Flag, "off", Reference,
         "traces the monolithic auxiliary coupling (vector gathers, boundary functionals, their assembly)"},

        {"SVMP_DISABLE_AUX_DIRECT_COUPLING", NumericalOverride, kFESystem, Flag, "off", NonReference,
         "drops the direct output-input coupling terms from the monolithic auxiliary Jacobian"},
        {"SVMP_DISABLE_MPI_NATIVE_RANK1_PROMOTION", NumericalOverride, kFESystem, Flag, "off (promotion on)",
         NonReference, "turns off the promotion of exact auxiliary coupling updates to native rank-one updates"},
        {"SVMP_MONO_DIRECT_TRACE", Diagnostic, kFESystem, Flag, "off", Reference,
         "traces the direct output-input coupling entries of the monolithic auxiliary assembly"},
        {"SVMP_NEGATE_AUX_DIRECT_COUPLING", NumericalOverride, kFESystem, Flag, "off", NonReference,
         "flips the sign of the direct auxiliary coupling terms"},
        {"SVMP_POSITIVE_AUX_DIRECT_COUPLING", NumericalOverride, kFESystem, Flag, "off", NonReference,
         "forces a positive sign of the direct auxiliary coupling terms (wins over the negation)"},

        // FE/TimeStepping: the generated-state schedule, the Newton solver and the
        // time loop.
        {"SVMP_CUT_REFRESH_PER_STEP_ONLY", NumericalOverride, kGeneratedState, Flag, "off", NonReference,
         "refreshes the generated state once per step, without the outer fixed point"},
        {"SVMP_SYNC_LINE_SEARCH_TRIALS", NumericalOverride, kGeneratedState, Flag,
         "on when generated state defines the residual", NonReference,
         "regenerates the generated state at every line-search trial (0 freezes it for comparisons)"},
        {"SVMP_DEBUG_FIRST_LINEAR_VECTOR_DUMP_PREFIX", Diagnostic, kNewton, Text, "off", Reference,
         "writes the right-hand side and increment of the first Newton linear solve under this prefix"},
        {"SVMP_DEBUG_LINEAR_PROBE_DUMP", Diagnostic, kNewton, Flag, "off", Reference,
         "logs probe products of the first Newton linear system"},
        {"SVMP_DEBUG_LINEAR_SOLVE_COMPONENT_NORMS", Diagnostic, kNewton, Flag, "off", Reference,
         "logs the component norms of each Newton increment"},
        {"SVMP_DEBUG_LINEAR_SOLVE_COMPONENT_NORMS_MAX_NEWTON_IT", Diagnostic, kNewton, Integer, "-1 (all)",
         Reference, "last Newton iteration of the component-norm log"},
        {"SVMP_DEBUG_LINEAR_SOLVE_HISTORY", Diagnostic, kNewton, Flag, "off", Reference,
         "logs one line per Newton linear solve"},
        {"SVMP_DEBUG_LINEAR_SOLVE_HISTORY_MAX_CALLS", Diagnostic, kNewton, Integer, "-1 (unlimited)", Reference,
         "linear solves logged per Newton solve"},
        {"SVMP_DIRECT_ONLY_OUTLET_JACOBIAN_REBUILD_PERIOD", NumericalOverride, kNewton, Integer, "1",
         NonReference, "Jacobian rebuild period with direct-only coupled-outlet updates"},
        {"SVMP_DIRECT_ONLY_OUTLET_JACOBIAN_SCALE", NumericalOverride, kNewton, Real, "1", NonReference,
         "scale of the direct-only coupled-outlet updates"},
        {"SVMP_DISABLE_LOCAL_CONDENSED_RECOVERY", NumericalOverride, kNewton, Flag, "off", NonReference,
         "skips the recovery of locally condensed auxiliary unknowns"},
        {"SVMP_DISABLE_PREDICTED_CERTIFICATE_RESIDUAL_FIRST", Performance, kNewton, Flag, "off", Reference,
         "assembles residual and Jacobian together on refreshes predicted to pass the zero-update "
         "certificate"},
        {"SVMP_DISTRIBUTE_CONSTRAINTS_INTO_HISTORY", NumericalOverride, kNewton, Flag, "off", NonReference,
         "stamps current constraint values into every history state at state syncs"},
        {"SVMP_FE_ANALYSIS_LOG", Diagnostic, kNewton, Text, "unset", Reference,
         "FE system analysis report at setup and after the first tangent (any value, trace, full)"},
        {"SVMP_FE_JACOBIAN_CHECK", Diagnostic, kNewton, Flag, "off", Reference,
         "checks the Jacobian against residual differences at one Newton iteration"},
        {"SVMP_FE_JACOBIAN_CHECK_COMPONENTS", Diagnostic, kNewton, List, "all", Reference,
         "components of the Jacobian check direction"},
        {"SVMP_FE_JACOBIAN_CHECK_COMPONENT_SWEEPS", Diagnostic, kNewton, List, "one sweep", Reference,
         "component groups of the Jacobian check, one sweep each"},
        {"SVMP_FE_JACOBIAN_CHECK_IT", Diagnostic, kNewton, Integer, "0", Reference,
         "Newton iteration of the Jacobian check"},
        {"SVMP_FE_JACOBIAN_CHECK_SCHEME", Diagnostic, kNewton, Text, "forward", Reference,
         "difference scheme of the Jacobian check (forward, central)"},
        {"SVMP_FE_JACOBIAN_CHECK_STEP", Diagnostic, kNewton, Real, "1e-7", Reference,
         "relative step of the Jacobian check"},
        {"SVMP_FORCE_EXPLICIT_RANK_ONE", NumericalOverride, kNewton, Flag, "off", NonReference,
         "assembles low-rank auxiliary coupling updates into the Jacobian"},
        {"SVMP_FORCE_EXPLICIT_RANK_ONE_AFTER_REL_RES", NumericalOverride, kNewton, Real, "never", NonReference,
         "relative residual below which the direct-only outlet updates are assembled explicitly"},
        {"SVMP_FSILS_POST_SOLVE_SYNC", NumericalOverride, kNewton, Text, "off", NonReference,
         "refreshes the ghost entries of the Newton increment after each FSILS solve (update, ghost)"},
        {"SVMP_FSILS_TIGHTEN_DIRECT_ONLY_AFTER_REL_RES", NumericalOverride, kNewton, Real, "never",
         NonReference, "relative residual below which the direct-only outlet solve is tightened"},
        {"SVMP_FSILS_TIGHTEN_DIRECT_ONLY_INNER_REL_TOL", NumericalOverride, kNewton, Real, "1e-6", NonReference,
         "inner relative tolerance of the tightened direct-only outlet solve"},
        {"SVMP_LINEAR_SOLVE_MEMORY_DIAGNOSTICS", Diagnostic, kNewton, Flag, "off", Reference,
         "logs memory use around each Newton linear solve"},
        {"SVMP_MAX_CONDENSED_AUX_SIZE", NumericalOverride, kNewton, Integer, "64", NonReference,
         "largest auxiliary block condensed into reduced field updates"},
        {"SVMP_NEWTON_ASSEMBLY_DIAGNOSTICS", Diagnostic, kNewton, Flag, "off", Reference,
         "logs per-assembly statistics of Newton assemblies"},
        {"SVMP_NEWTON_DIRECTION_CHECK", Diagnostic, kNewton, Flag, "off", Reference,
         "logs the linear residual of each Newton increment"},
        {"SVMP_NEWTON_LINE_SEARCH_FIRST_DIRECT_ONLY_REDUCED", NumericalOverride, kNewton, Flag, "off",
         NonReference, "line search on the first Newton iteration with direct-only reduced updates"},
        {"SVMP_NEWTON_MATRIX_SUPPORT_DIAGNOSTIC", Diagnostic, kNewton, Flag, "off", Reference,
         "logs the Jacobian support of the sampled DOFs before each linear solve"},
        {"SVMP_NEWTON_VERIFY_ACCEPTED_REFRESH_SKIP", Diagnostic, kNewton, Flag, "off", Reference,
         "reassembles a reused line-search residual and throws unless it is identical"},
        {"SVMP_NO_MPC_STATE_DISTRIBUTE", NumericalOverride, kNewton, Flag, "off", NonReference,
         "skips distributing master-bearing constraint state into the solution and rate history"},
        {"SVMP_PRESERVE_GROUPED_ALGEBRAIC_DIRECT_ONLY", NumericalOverride, kNewton, Flag, "off", NonReference,
         "keeps algebraic direct-only auxiliary couplings grouped"},
        {"SVMP_PURE_ALGEBRAIC_BORDERED_RECOVERY", NumericalOverride, kNewton, Flag, "off", NonReference,
         "recovers increments of pure algebraic auxiliary blocks from the explicit bordered system"},
        {"SVMP_RESIDUAL_FIRST_SELF_CHECK", Diagnostic, kNewton, Flag, "off", Reference,
         "also assembles residual and Jacobian together on residual-first iterations and logs whether they "
         "agree"},
        {"SVMP_TIMELOOP_INITIALIZATION_DIAGNOSTICS", Diagnostic, kTimeLoop, Flag, "off", Reference,
         "logs the constraints and norms of the initialization solves"},
        {"SVMP_ZERO_CONSTRAINED_RATES", NumericalOverride, kTimeLoop, Flag, "off", NonReference,
         "zeros the rates at constrained DOFs after every rate update"},

        // Physics/Core: the JIT policy of physics modules.
        {"SVMP_FE_JIT_ENABLE", Performance, kPhysicsJit, Flag, "the module options (on)", NonReference,
         "second spelling of SVMP_OOP_JIT_ENABLE"},
        {"SVMP_FE_JIT_SPECIALIZATION_ENABLE", Performance, kPhysicsJit, Flag, "the module options (on)",
         NonReference, "second spelling of SVMP_OOP_JIT_SPECIALIZATION_ENABLE"},
        {"SVMP_OOP_JIT_ENABLE", Performance, kPhysicsJit, Flag, "the module options (on)", NonReference,
         "LLVM JIT form kernels, off for the interpreted kernels (after the deck and module options)"},
        {"SVMP_OOP_JIT_SPECIALIZATION_ENABLE", Performance, kPhysicsJit, Flag, "the module options (on)",
         NonReference, "specialized JIT kernels (after the deck and module options)"},
    };
    return kSolverNames;
}

[[nodiscard]] bool validName(std::string_view name) noexcept
{
    return !name.empty() && std::all_of(name.begin(), name.end(), [](char ch) {
        return (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_';
    });
}

[[nodiscard]] bool nonEmpty(const char* text) noexcept
{
    return text != nullptr && text[0] != '\0';
}

[[nodiscard]] bool lessByName(const NameDeclaration* lhs, const NameDeclaration* rhs) noexcept
{
    return std::string_view(lhs->name) < std::string_view(rhs->name);
}

} // namespace

const char* toString(NameKind kind) noexcept
{
    switch (kind) {
    case NameKind::Diagnostic: return "diagnostic";
    case NameKind::Performance: return "performance";
    case NameKind::TestHook: return "test_hook";
    case NameKind::NumericalOverride: return "numerical_override";
    }
    return "unknown";
}

const char* toString(ValueType type) noexcept
{
    switch (type) {
    case ValueType::Flag: return "flag";
    case ValueType::Integer: return "integer";
    case ValueType::Real: return "real";
    case ValueType::Text: return "text";
    case ValueType::List: return "list";
    }
    return "unknown";
}

EnvironmentNameRegistry::EnvironmentNameRegistry(std::span<const NameDeclaration> declarations)
{
    by_name_.reserve(declarations.size());
    for (const auto& declaration : declarations) {
        if (declaration.name == nullptr || !validName(declaration.name)) {
            throw std::logic_error(
                "EnvironmentNameRegistry: a declared name must be nonempty and spelled with "
                "upper-case letters, digits and underscores");
        }
        if (!nonEmpty(declaration.owner) || !nonEmpty(declaration.default_value) ||
            !nonEmpty(declaration.controls)) {
            throw std::logic_error(std::string("EnvironmentNameRegistry: '") + declaration.name +
                                   "' needs an owner, a default value and what it controls");
        }
        by_name_.push_back(&declaration);
    }
    std::sort(by_name_.begin(), by_name_.end(), lessByName);
    const auto duplicate = std::adjacent_find(
        by_name_.begin(), by_name_.end(), [](const NameDeclaration* lhs, const NameDeclaration* rhs) {
            return std::string_view(lhs->name) == std::string_view(rhs->name);
        });
    if (duplicate != by_name_.end()) {
        throw std::logic_error(std::string("EnvironmentNameRegistry: '") + (*duplicate)->name +
                               "' is declared twice");
    }
}

const EnvironmentNameRegistry& EnvironmentNameRegistry::solver()
{
    static const EnvironmentNameRegistry registry(solverCatalogue());
    return registry;
}

const NameDeclaration* EnvironmentNameRegistry::find(std::string_view name) const noexcept
{
    const auto it = std::lower_bound(
        by_name_.begin(), by_name_.end(), name,
        [](const NameDeclaration* declaration, std::string_view key) {
            return std::string_view(declaration->name) < key;
        });
    if (it == by_name_.end() || std::string_view((*it)->name) != name) {
        return nullptr;
    }
    return *it;
}

const NameDeclaration& EnvironmentNameRegistry::declaration(std::string_view name) const
{
    if (const auto* declaration = find(name)) {
        return *declaration;
    }
    throw std::logic_error("EnvironmentNameRegistry: the environment name '" + std::string(name) +
                           "' is not declared");
}

const char* EnvironmentNameRegistry::value(std::string_view name) const
{
    return std::getenv(declaration(name).name);
}

bool EnvironmentNameRegistry::isSet(std::string_view name) const
{
    return value(name) != nullptr;
}

bool EnvironmentNameRegistry::enabled(std::string_view name) const
{
    const auto& declared = declaration(name);
    if (declared.type != ValueType::Flag) {
        throw std::logic_error(std::string("EnvironmentNameRegistry: '") + declared.name +
                               "' is not declared as a flag");
    }
    const char* text = std::getenv(declared.name);
    if (text == nullptr || text[0] == '\0') {
        return false;
    }
    std::string lowered(text);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return !(lowered == "0" || lowered == "false" || lowered == "off" || lowered == "no");
}

std::vector<NameUse> EnvironmentNameRegistry::setNames() const
{
    std::vector<NameUse> uses;
    for (const auto* declaration : by_name_) {
        if (const char* text = std::getenv(declaration->name)) {
            uses.push_back(NameUse{.declaration = declaration, .value = text});
        }
    }
    return uses;
}

std::string EnvironmentNameRegistry::configurationRecord() const
{
    namespace json = options::json;
    std::string names;
    bool reference_run = true;
    for (const auto& use : setNames()) {
        const auto& declared = *use.declaration;
        if (declared.kind == NameKind::Diagnostic) {
            continue;
        }
        reference_run = reference_run && !declared.marksNonReference();
        if (!names.empty()) {
            names += ',';
        }
        names += "{\"name\":" + json::quote(declared.name) +
                 ",\"value\":" + json::quote(use.value) +
                 ",\"kind\":" + json::quote(toString(declared.kind)) +
                 ",\"value_type\":" + json::quote(toString(declared.type)) +
                 ",\"owner\":" + json::quote(declared.owner) +
                 ",\"marks_non_reference\":" + json::boolean(declared.marksNonReference()) + "}";
    }
    if (names.empty()) {
        return {};
    }
    return std::string("\"environment_names\":{\"reference_run\":") + json::boolean(reference_run) +
           ",\"set\":[" + names + "]}";
}

const char* value(std::string_view name)
{
    return EnvironmentNameRegistry::solver().value(name);
}

bool isSet(std::string_view name)
{
    return EnvironmentNameRegistry::solver().isSet(name);
}

bool enabled(std::string_view name)
{
    return EnvironmentNameRegistry::solver().enabled(name);
}

std::string configurationRecord()
{
    return EnvironmentNameRegistry::solver().configurationRecord();
}

} // namespace environment
} // namespace FE
} // namespace svmp
