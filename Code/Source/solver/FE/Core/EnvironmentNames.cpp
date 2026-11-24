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
    constexpr const char* kEigen = "FE/Backends/Eigen";
    constexpr const char* kFsils = "FE/Backends/FSILS";

    static constexpr NameDeclaration kSolverNames[] = {
        // FE/Backends/Eigen: the Eigen matrix backend.
        {"SVMP_EIGEN_COUNT_DROPPED", Diagnostic, kEigen, Flag, "off", Reference,
         "logs samples of matrix writes outside the sparsity pattern"},
        {"SVMP_EIGEN_STRICT_PATTERN", Diagnostic, kEigen, Flag, "off", Reference,
         "throws on a matrix write outside the sparsity pattern instead of dropping it"},

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
