#include "Application/Core/SimulationBuilder.h"

#include "Application/Core/LevelSetCutConfiguration.h"
#include "Application/Core/OopMpiLog.h"
#include "Application/Translators/EquationTranslator.h"
#include "FE/Interfaces/LevelSetInterfaceDomain.h"
#include "Application/Translators/MeshTranslator.h"

#include "FE/Backends/Interfaces/BackendFactory.h"
#include "FE/Backends/Interfaces/BackendKind.h"
#include "FE/Backends/Interfaces/GenericVector.h"
#include "FE/Backends/Interfaces/LinearSolver.h"
#include "FE/Backends/Utils/BackendOptions.h"
#include "FE/Assembly/CompositeMeshAccess.h"
#include "FE/Dofs/DofHandler.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Systems/FESystem.h"
#include "FE/TimeStepping/TimeHistory.h"
#include "Mesh/Mesh.h"
#include "Physics/Core/PhysicsModule.h"
#include "Parameters.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

std::string lower_copy(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

bool oopStateTraceEnabled()
{
  const char* env = std::getenv("SVMP_OOP_SOLVER_TRACE");
  if (env == nullptr) {
    return false;
  }
  const auto v = lower_copy(env);
  return !(v == "0" || v == "false" || v == "off" || v == "no");
}

void traceStateVectorFields(const svmp::FE::systems::FESystem& system,
                            svmp::FE::backends::GenericVector& vector,
                            const char* label)
{
  if (!oopStateTraceEnabled()) {
    return;
  }
  const auto view = vector.createAssemblyView();
  if (!view) {
    return;
  }
  const auto& fields = system.fieldMap();
  std::ostringstream oss;
  oss << "[svMultiPhysics::Application] state_vector diagnostic=state_vector_fields"
      << " label='" << label << "'";
  for (std::size_t field = 0; field < fields.numFields(); ++field) {
    const auto& rec = fields.getField(field);
    const auto range = fields.getFieldDofRange(field);
    double sq_norm = 0.0;
    double sum = 0.0;
    double min_value = std::numeric_limits<double>::infinity();
    double max_value = -std::numeric_limits<double>::infinity();
    std::uint64_t count = 0;
    for (svmp::FE::GlobalIndex dof = range.first; dof < range.second; ++dof) {
      const double value = static_cast<double>(view->getVectorEntry(dof));
      sq_norm += value * value;
      sum += value;
      min_value = std::min(min_value, value);
      max_value = std::max(max_value, value);
      ++count;
    }
    const double mean = count > 0 ? sum / static_cast<double>(count) : 0.0;
    oss << " [" << rec.name
        << " dofs=" << count
        << " norm=" << std::sqrt(std::max(0.0, sq_norm))
        << " mean=" << mean
        << " min=" << (count > 0 ? min_value : 0.0)
        << " max=" << (count > 0 ? max_value : 0.0)
        << "]";
  }
  application::core::oopCout() << oss.str() << std::endl;
}

svmp::FE::backends::SolverMethod toSolverMethod(const std::string& legacy_type)
{
  using svmp::FE::backends::SolverMethod;
  const auto v = lower_copy(legacy_type);

  if (v.empty()) {
    return SolverMethod::Direct;
  }
  if (v == "direct" || v == "eigen" || v == "sparse-lu" || v == "sparselu") {
    return SolverMethod::Direct;
  }
  if (v == "cg" || v == "conjugate-gradient") {
    return SolverMethod::CG;
  }
  if (v == "gmres") {
    return SolverMethod::GMRES;
  }
  if (v == "pgmres" || v == "ksppgmres") {
    return SolverMethod::PGMRES;
  }
  if (v == "bicg" || v == "bicgs" || v == "bi-conjugate-gradient") {
    return SolverMethod::BiCGSTAB;
  }
  if (v == "ns" || v == "bi-partitioned" || v == "bpn" || v == "bipn") {
    return SolverMethod::BlockSchur;
  }

  throw std::runtime_error("[svMultiPhysics::Application] Unsupported linear solver type '" + legacy_type +
                           "' for the new solver. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the "
                           "legacy solver.");
}

svmp::FE::backends::PreconditionerType toPreconditioner(const std::string& legacy_prec)
{
  using svmp::FE::backends::PreconditionerType;
  const auto v = lower_copy(legacy_prec);

  if (v.empty() || v == "none") {
    return PreconditionerType::None;
  }
  if (v == "fsils") {
    // Legacy FSILS "fsils" preconditioner corresponds to the built-in diagonal scaling.
    return PreconditionerType::Diagonal;
  }
  if (v == "row-column-scaling" || v == "petsc-rcs") {
    return PreconditionerType::RowColumnScaling;
  }
  if (v == "petsc-jacobi" || v == "trilinos-diagonal" || v == "trilinos-blockjacobi") {
    return PreconditionerType::Diagonal;
  }
  if (v == "trilinos-ilu" || v == "trilinos-ilut" || v == "trilinos-ic" || v == "trilinos-ict") {
    return PreconditionerType::ILU;
  }
  if (v == "trilinos-ml") {
    return PreconditionerType::AMG;
  }

  return PreconditionerType::None;
}

svmp::FE::backends::FsilsBlockSchurSchurPreconditioner
toFsilsBlockSchurPreconditioner(const std::string& value)
{
  using svmp::FE::backends::FsilsBlockSchurSchurPreconditioner;
  const auto v = lower_copy(value);

  if (v.empty() || v == "diag-l" || v == "diagl" || v == "diag") {
    return FsilsBlockSchurSchurPreconditioner::DiagL;
  }
  if (v == "blockdiag-l" || v == "blockdiagl" || v == "blockdiag") {
    return FsilsBlockSchurSchurPreconditioner::BlockDiagL;
  }
  if (v == "ilu-l" || v == "ilul") {
    return FsilsBlockSchurSchurPreconditioner::ILUL;
  }
  if (v == "algebraic-shat" || v == "algebraic" || v == "shat") {
    return FsilsBlockSchurSchurPreconditioner::AlgebraicSchur;
  }

  throw std::runtime_error("[svMultiPhysics::Application] Unsupported FSILS BlockSchur Schur preconditioner '" +
                           value + "'.");
}

bool hasUnfittedLevelSetFreeSurface(const EquationParameters& eq)
{
  if (!eq.type.defined() || lower_copy(eq.type.value()) != "fluid") {
    return false;
  }

  auto implementation_matches = [](const BoundaryConditionParameters& bc,
                                   const std::string& key) {
    const auto it = bc.extra_string_params.find(key);
    return it != bc.extra_string_params.end() &&
           it->second.defined() &&
           lower_copy(it->second.value()) == "unfittedlevelset";
  };

  for (const auto* bc : eq.boundary_conditions) {
    if (bc == nullptr || !bc->type.defined()) {
      continue;
    }
    if (lower_copy(bc->type.value()) != "free_surface") {
      continue;
    }
    if (implementation_matches(*bc, "Implementation") ||
        implementation_matches(*bc, "Free_surface_implementation") ||
        implementation_matches(*bc, "FreeSurfaceImplementation")) {
      return true;
    }
  }
  return false;
}

svmp::FE::backends::FsilsBlockSchurMomentumApproximation
toFsilsBlockSchurMomentumApproximation(const std::string& value)
{
  using svmp::FE::backends::FsilsBlockSchurMomentumApproximation;
  const auto v = lower_copy(value);

  if (v.empty() || v == "diag-k" || v == "diagk" || v == "diag") {
    return FsilsBlockSchurMomentumApproximation::DiagK;
  }
  if (v == "blockdiag-k" || v == "blockdiagk" || v == "blockdiag") {
    return FsilsBlockSchurMomentumApproximation::BlockDiagK;
  }
  if (v == "ilu-k" || v == "iluk" || v == "ilu") {
    return FsilsBlockSchurMomentumApproximation::ILUK;
  }
  if (v == "asm-k" || v == "asm") {
    return FsilsBlockSchurMomentumApproximation::ASM;
  }

  throw std::runtime_error("[svMultiPhysics::Application] Unsupported FSILS BlockSchur momentum approximation '" +
                           value + "'.");
}

const EquationParameters* selectSolverEquation(const Parameters& params);

svmp::FE::backends::BackendKind selectBackend(const Parameters& params)
{
  using svmp::FE::backends::BackendKind;

  const EquationParameters* eq = selectSolverEquation(params);

  if (!eq || !eq->linear_solver.linear_algebra.defined()) {
    return BackendKind::FSILS;
  }

  const auto type = lower_copy(eq->linear_solver.linear_algebra.type.value());
  if (type == "fsils") {
    return BackendKind::FSILS;
  }
  if (type == "petsc") {
    return BackendKind::PETSc;
  }
  if (type == "trilinos") {
    return BackendKind::Trilinos;
  }
  if (type == "eigen") {
    return BackendKind::Eigen;
  }

  return BackendKind::FSILS;
}

bool applyFsilsLocalityOrderingEnv(svmp::FE::dofs::DofDistributionOptions& options)
{
  const char* env = std::getenv("SVMP_FSILS_LOCALITY_ORDERING");
  if (env == nullptr || env[0] == '\0') {
    return false;
  }

  const auto v = lower_copy(std::string(env));
  using S = svmp::FE::dofs::DofNumberingStrategy;
  if (v == "off" || v == "none" || v == "interleaved") {
    options.numbering = S::Interleaved;
    return true;
  }
  if (v == "auto" || v == "morton") {
    options.numbering = S::Morton;
    return true;
  }
  if (v == "hilbert") {
    options.numbering = S::Hilbert;
    return true;
  }
  if (v == "rcm" || v == "cuthill-mckee") {
    options.numbering = S::CuthillMcKee;
    return true;
  }

  throw std::runtime_error("[svMultiPhysics::Application] Unsupported SVMP_FSILS_LOCALITY_ORDERING='" +
                           std::string(env) + "'. Supported values: off, auto, morton, hilbert, rcm.");
}

std::string codim1StorageName(svmp::MeshCodim1StorageMode mode)
{
  switch (mode) {
    case svmp::MeshCodim1StorageMode::None:
      return "none";
    case svmp::MeshCodim1StorageMode::BoundaryOnly:
      return "boundary-only";
    case svmp::MeshCodim1StorageMode::Full:
      return "full";
    case svmp::MeshCodim1StorageMode::Explicit:
      return "explicit";
  }
  return "unknown";
}

bool firstEquationTypeIs(const Parameters& params, const std::string& equation_type)
{
  for (const auto* e : params.equation_parameters) {
    if (!e) {
      continue;
    }
    return e->type.defined() && lower_copy(e->type.value()) == equation_type;
  }
  return false;
}

const EquationParameters* firstEquation(const Parameters& params)
{
  for (const auto* e : params.equation_parameters) {
    if (e) {
      return e;
    }
  }
  return nullptr;
}

const EquationParameters* selectSolverEquation(const Parameters& params)
{
  const auto* first = firstEquation(params);
  if (first == nullptr) {
    return nullptr;
  }

  if (first->type.defined() && lower_copy(first->type.value()) == "level_set") {
    for (const auto* e : params.equation_parameters) {
      if (e && e->type.defined() && lower_copy(e->type.value()) == "fluid") {
        return e;
      }
    }
  }

  return first;
}

svmp::FE::geometry::CutIntegrationSide toCutIntegrationSide(
    application::core::LevelSetActiveSide side) noexcept
{
  return side == application::core::LevelSetActiveSide::Positive
             ? svmp::FE::geometry::CutIntegrationSide::Positive
             : svmp::FE::geometry::CutIntegrationSide::Negative;
}

struct EquationCellDomainRestriction {
  application::core::ActiveCutVolumeRequest request{};
  svmp::FE::systems::FESystem::FormCellDomainRestriction restriction{};
};

std::vector<EquationCellDomainRestriction>
equationCellDomainRestrictions(
    const svmp::FE::systems::FESystem& system,
    const EquationParameters& equation)
{
  std::vector<EquationCellDomainRestriction> restrictions;
  const auto requests = application::core::activeCutVolumeRequests(equation);
  for (const auto& request : requests) {
    if (request.origin != application::core::ActiveCutVolumeRequestOrigin::Equation) {
      continue;
    }

    const auto field_id = system.findFieldByName(request.level_set_field_name);
    const int marker =
        application::core::requireResolvedActiveCutVolumeInterfaceMarker(
            system, request);

    const std::string equation_type =
        equation.type.defined() ? equation.type.value() : std::string{};
    restrictions.push_back(
        EquationCellDomainRestriction{
            .request = request,
            .restriction =
                svmp::FE::systems::FESystem::FormCellDomainRestriction{
                    .interface_marker = marker,
                    .side = toCutIntegrationSide(request.active_side),
                    .level_set_field = field_id,
                    .isovalue = static_cast<svmp::FE::Real>(request.isovalue),
                    .enable_level_set_shape_tangent =
                        field_id != svmp::FE::INVALID_FIELD_ID,
                    .diagnostic =
                        "equation_level_level_set_cut_domain equation_type='" +
                        equation_type + "' domain_id='" + request.domain_id +
                        "' level_set_field='" +
                        request.level_set_field_name + "'"}});
  }
  return restrictions;
}

svmp::FE::backends::SolverOptions translateSolverOptions(const Parameters& params,
                                                         svmp::FE::backends::BackendKind backend_kind)
{
  using svmp::FE::backends::SolverOptions;

  SolverOptions opts{};

  const EquationParameters* eq = selectSolverEquation(params);

  if (!eq) {
    return opts;
  }

  if (eq->linear_solver.type.defined()) {
    opts.method = toSolverMethod(eq->linear_solver.type.value());
  }

  if (eq->linear_solver.tolerance.defined()) {
    opts.rel_tol = static_cast<svmp::FE::Real>(eq->linear_solver.tolerance.value());
  }
  if (eq->linear_solver.absolute_tolerance.defined()) {
    opts.abs_tol = static_cast<svmp::FE::Real>(eq->linear_solver.absolute_tolerance.value());
  } else {
    // Match legacy defaults (LinearSolverParameters::Absolute_tolerance) when not explicitly set.
    opts.abs_tol = static_cast<svmp::FE::Real>(1.0e-10);
  }
  if (eq->linear_solver.max_iterations.defined()) {
    opts.max_iter = eq->linear_solver.max_iterations.value();
  }
  if (eq->linear_solver.krylov_space_dimension.defined()) {
    opts.krylov_dim = eq->linear_solver.krylov_space_dimension.value();
  }

  if (eq->linear_solver.linear_algebra.defined()) {
    const auto& la = eq->linear_solver.linear_algebra;
    opts.preconditioner = toPreconditioner(la.preconditioner.value());
    opts.fsils_use_rcs = (opts.preconditioner == svmp::FE::backends::PreconditionerType::RowColumnScaling);

    if (backend_kind == svmp::FE::backends::BackendKind::Trilinos && la.configuration_file.defined()) {
      opts.trilinos_xml_file = la.configuration_file.value();
    }
  }

  const bool is_oop_ustruct =
      eq->type.defined() && lower_copy(eq->type.value()) == "ustruct";
  const bool is_fsils_gmres_family =
      backend_kind == svmp::FE::backends::BackendKind::FSILS &&
      (opts.method == svmp::FE::backends::SolverMethod::GMRES ||
       opts.method == svmp::FE::backends::SolverMethod::PGMRES ||
       opts.method == svmp::FE::backends::SolverMethod::FGMRES);
  const bool has_diagonal_like_scaling =
      opts.preconditioner == svmp::FE::backends::PreconditionerType::None ||
      opts.preconditioner == svmp::FE::backends::PreconditionerType::Diagonal;
  if (is_fsils_gmres_family &&
      hasUnfittedLevelSetFreeSurface(*eq) &&
      has_diagonal_like_scaling) {
    opts.preconditioner = svmp::FE::backends::PreconditionerType::RowColumnScaling;
    opts.fsils_use_rcs = true;
  }
  if (is_fsils_gmres_family && hasUnfittedLevelSetFreeSurface(*eq)) {
    opts.fsils_residual_check_policy =
        svmp::FE::backends::FsilsResidualCheckPolicy::Always;
  }
  if (backend_kind == svmp::FE::backends::BackendKind::FSILS &&
      is_oop_ustruct &&
      opts.method != svmp::FE::backends::SolverMethod::Direct &&
      opts.method != svmp::FE::backends::SolverMethod::BlockSchur &&
      has_diagonal_like_scaling) {
    opts.preconditioner = svmp::FE::backends::PreconditionerType::RowColumnScaling;
    opts.fsils_use_rcs = true;
  }

  // FSILS GMRES legacy semantics:
  // - <Max_iterations> is the outer restart count (RI.mItr),
  // - <Krylov_space_dimension> is the restart length (RI.sD, default 50).
  //
  // The FE linear-solver contract uses `max_iter` as a total Krylov-step budget, so convert:
  //   max_iter_total = mItr * (sD + 1)
  // and ensure krylov_dim is populated even if the XML omitted it (to match legacy defaults).
  if (backend_kind == svmp::FE::backends::BackendKind::FSILS &&
      (opts.method == svmp::FE::backends::SolverMethod::GMRES ||
       opts.method == svmp::FE::backends::SolverMethod::PGMRES ||
       opts.method == svmp::FE::backends::SolverMethod::FGMRES) &&
      eq->linear_solver.max_iterations.defined()) {
    // FSILS GMRES default: sD = 250 (see fsils_ls_create in ls.cpp).
    // The old hardcoded default of 50 was too small, causing restarted GMRES
    // to stagnate on ill-conditioned systems (e.g. 2nd Newton iteration of
    // Navier-Stokes with diagonal preconditioning).
    const int legacy_restart_len = eq->linear_solver.krylov_space_dimension.defined()
                                      ? eq->linear_solver.krylov_space_dimension.value()
                                      : 250;
    const int restart_len = std::max(0, legacy_restart_len);
    opts.krylov_dim = restart_len;

    using i64 = long long;
    const i64 outer = static_cast<i64>(eq->linear_solver.max_iterations.value());
    const i64 per_restart = static_cast<i64>(restart_len) + 1LL;
    const i64 total = outer * per_restart;
    if (total > static_cast<i64>(std::numeric_limits<int>::max())) {
      opts.max_iter = std::numeric_limits<int>::max();
    } else if (total > 0) {
      opts.max_iter = static_cast<int>(total);
    }
  }

  // Explicit saddle-point block names: env vars or programmatic SolverOptions.
  // SVMP_MOMENTUM_BLOCK=<name> and SVMP_CONSTRAINT_BLOCK=<name> override the heuristic.
  if (opts.momentum_block_name.empty()) {
    const char* env = std::getenv("SVMP_MOMENTUM_BLOCK");
    if (env && env[0]) opts.momentum_block_name = env;
  }
  if (opts.constraint_block_name.empty()) {
    const char* env = std::getenv("SVMP_CONSTRAINT_BLOCK");
    if (env && env[0]) opts.constraint_block_name = env;
  }

  // FSILS BlockSchur sub-solver knobs: pass GM/CG sub-solver controls through.
  if (backend_kind == svmp::FE::backends::BackendKind::FSILS &&
      opts.method == svmp::FE::backends::SolverMethod::BlockSchur) {
    opts.fsils_blockschur_gm_max_iter = eq->linear_solver.ns_gm_max_iterations.value();
    opts.fsils_blockschur_cg_max_iter = eq->linear_solver.ns_cg_max_iterations.value();
    opts.fsils_blockschur_gm_rel_tol = static_cast<svmp::FE::Real>(eq->linear_solver.ns_gm_tolerance.value());
    opts.fsils_blockschur_cg_rel_tol = static_cast<svmp::FE::Real>(eq->linear_solver.ns_cg_tolerance.value());
    opts.fsils_blockschur_min_outer_iterations =
        eq->linear_solver.ns_min_outer_iterations.value();
    opts.fsils_blockschur_schur_preconditioner =
        toFsilsBlockSchurPreconditioner(eq->linear_solver.ns_schur_preconditioner.value());
    opts.fsils_blockschur_momentum_approximation =
        toFsilsBlockSchurMomentumApproximation(eq->linear_solver.ns_momentum_approximation.value());
    opts.fsils_blockschur_use_coupled_outer_fgmres =
        eq->linear_solver.ns_use_coupled_outer_fgmres.value();

    const char* schur_pc_env = std::getenv("SVMP_FSILS_BLOCKSCHUR_SCHUR_PC");
    if (schur_pc_env && schur_pc_env[0]) {
      opts.fsils_blockschur_schur_preconditioner = toFsilsBlockSchurPreconditioner(schur_pc_env);
    }
    const char* momentum_hat_env = std::getenv("SVMP_FSILS_BLOCKSCHUR_MOMENTUM_HAT");
    if (momentum_hat_env && momentum_hat_env[0]) {
      opts.fsils_blockschur_momentum_approximation =
          toFsilsBlockSchurMomentumApproximation(momentum_hat_env);
    }
  }

  return opts;
}

std::shared_ptr<const svmp::FE::backends::DofPermutation> build_fsils_dof_permutation(const svmp::FE::systems::FESystem& system,
                                                                                      int dof_per_node,
                                                                                      const svmp::FE::dofs::DofDistributionOptions& dof_options)
{
  using svmp::FE::GlobalIndex;
  using svmp::FE::backends::DofPermutation;
  (void)dof_options;

  if (dof_per_node <= 0) {
    return {};
  }

  const GlobalIndex total_dofs = system.dofHandler().getNumDofs();
  if (total_dofs <= 0) {
    return {};
  }

  // Derive node-block permutation from the field map (requires equal-order fields).
  const auto& fmap = system.fieldMap();
  const auto n_fields = fmap.numFields();
  if (n_fields == 0) {
    return {};
  }

  GlobalIndex n_nodes = -1;
  int expected_dof_per_node = 0;
  for (std::size_t f = 0; f < n_fields; ++f) {
    const auto& field = fmap.getField(f);
    expected_dof_per_node += field.n_components;
    if (field.n_components <= 0) {
      throw std::runtime_error("[svMultiPhysics::Application] FSILS backend requires component-wise fields.");
    }
    if (field.n_dofs % field.n_components != 0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] FSILS backend requires field DOF counts divisible by the number of components.");
    }
    const GlobalIndex n_per_component = field.n_dofs / field.n_components;
    if (n_nodes < 0) {
      n_nodes = n_per_component;
    } else if (n_nodes != n_per_component) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] FSILS backend requires all fields to have the same number of DOFs per "
          "component (e.g., equal-order nodal spaces).");
    }
  }

  if (expected_dof_per_node != dof_per_node) {
    throw std::runtime_error("[svMultiPhysics::Application] FSILS backend dof_per_node mismatch.");
  }
  if (n_nodes <= 0) {
    return {};
  }
  if (total_dofs != static_cast<GlobalIndex>(dof_per_node) * n_nodes) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] FSILS backend requires total DOFs == dof_per_node * n_nodes; check field spaces.");
  }

  // In MPI runs the application does not have globally complete node ownership/coordinate information,
  // so any owner/spatial-based global permutation would be rank-dependent and incorrect. Use the
  // canonical node ordering (node_id == local_dof index) for a deterministic, globally consistent
  // node-block permutation.
  std::vector<GlobalIndex> node_to_backend(static_cast<std::size_t>(n_nodes), svmp::FE::INVALID_GLOBAL_INDEX);
  for (GlobalIndex node = 0; node < n_nodes; ++node) {
    node_to_backend[static_cast<std::size_t>(node)] = node;
  }

  auto perm = std::make_shared<DofPermutation>();
  perm->forward.assign(static_cast<std::size_t>(total_dofs), svmp::FE::INVALID_GLOBAL_INDEX);
  perm->inverse.assign(static_cast<std::size_t>(total_dofs), svmp::FE::INVALID_GLOBAL_INDEX);

  for (GlobalIndex node = 0; node < n_nodes; ++node) {
    const GlobalIndex backend_node = node_to_backend[static_cast<std::size_t>(node)];
    if (backend_node < 0) {
      throw std::runtime_error("[svMultiPhysics::Application] FSILS permutation: missing backend node id.");
    }
    int comp_offset = 0;
    for (std::size_t f = 0; f < n_fields; ++f) {
      const auto& field = fmap.getField(f);
      for (svmp::FE::LocalIndex c = 0; c < field.n_components; ++c) {
        const GlobalIndex fe_dof = fmap.componentToGlobal(f, c, node);
        const GlobalIndex fs_dof =
            backend_node * static_cast<GlobalIndex>(dof_per_node) + static_cast<GlobalIndex>(comp_offset);
        perm->forward[static_cast<std::size_t>(fe_dof)] = fs_dof;
        perm->inverse[static_cast<std::size_t>(fs_dof)] = fe_dof;
        ++comp_offset;
      }
    }
  }

  for (std::size_t i = 0; i < perm->forward.size(); ++i) {
    if (perm->forward[i] == svmp::FE::INVALID_GLOBAL_INDEX) {
      throw std::runtime_error("[svMultiPhysics::Application] FSILS backend DOF permutation is incomplete.");
    }
  }
  for (std::size_t i = 0; i < perm->inverse.size(); ++i) {
    if (perm->inverse[i] == svmp::FE::INVALID_GLOBAL_INDEX) {
      throw std::runtime_error("[svMultiPhysics::Application] FSILS backend DOF permutation is incomplete.");
    }
  }

  return perm;
}

} // namespace

namespace application {
namespace core {

SimulationComponents::SimulationComponents() = default;

SimulationComponents::SimulationComponents(SimulationComponents&&) noexcept = default;

SimulationComponents& SimulationComponents::operator=(SimulationComponents&&) noexcept = default;

SimulationComponents::~SimulationComponents() = default;

SimulationBuilder::SimulationBuilder(const Parameters& params)
  : params_(params)
{
}

SimulationComponents SimulationBuilder::build()
{
  oopCout() << "[svMultiPhysics::Application] SimulationBuilder: loadMeshes()" << std::endl;
  loadMeshes();

  oopCout() << "[svMultiPhysics::Application] SimulationBuilder: createFESystem()" << std::endl;
  createFESystem();

  oopCout() << "[svMultiPhysics::Application] SimulationBuilder: createPhysicsModules()" << std::endl;
  createPhysicsModules();

  oopCout() << "[svMultiPhysics::Application] SimulationBuilder: setupSystem()" << std::endl;
  setupSystem();

  oopCout() << "[svMultiPhysics::Application] SimulationBuilder: createSolvers()" << std::endl;
  createSolvers();

  oopCout() << "[svMultiPhysics::Application] SimulationBuilder: allocateHistory()" << std::endl;
  allocateHistory();

  return std::move(components_);
}

void SimulationBuilder::loadMeshes()
{
  components_.mesh_collection.clear();
  components_.meshes.clear();
  components_.primary_mesh.reset();
  components_.primary_mesh_name.clear();

  const auto declared =
      static_cast<int>(std::count_if(params_.mesh_parameters.begin(), params_.mesh_parameters.end(),
                                     [](const auto* p) { return p != nullptr; }));
  oopCout() << "[svMultiPhysics::Application] SimulationBuilder: meshes declared=" << declared << std::endl;

  for (const auto* mesh_params : params_.mesh_parameters) {
    if (!mesh_params) {
      continue;
    }

    const auto mesh_name = mesh_params->name.value();
    if (mesh_name.empty()) {
      throw std::runtime_error("[svMultiPhysics::Application] <Add_mesh> is missing required name attribute.");
    }

    if (components_.meshes.count(mesh_name) != 0) {
      throw std::runtime_error("[svMultiPhysics::Application] Duplicate <Add_mesh name=\"" + mesh_name +
                               "\"> detected.");
    }

    const auto mesh_path = mesh_params->mesh_file_path.value();
    oopCout() << "[svMultiPhysics::Application]   Loading mesh '" << mesh_name << "' path='" << mesh_path << "'"
              << " faces=" << static_cast<int>(mesh_params->face_parameters.size());
    if (mesh_params->domain_id.defined()) {
      oopCout() << " domain_id=" << mesh_params->domain_id.value();
    }
    oopCout() << std::endl;

    auto mesh = application::translators::MeshTranslator::loadMesh(*mesh_params);
    auto participant = MeshParticipant::fromLoadedMesh(*mesh_params, mesh);
    components_.mesh_collection.addParticipant(std::move(participant));
    components_.meshes.emplace(mesh_name, mesh);

    if (mesh) {
      oopCout() << "[svMultiPhysics::Application]   Mesh '" << mesh_name << "': dim=" << mesh->dim()
                << " local_vertices=" << mesh->n_vertices() << " local_cells=" << mesh->n_cells()
                << " local_faces=" << mesh->n_faces() << std::endl;
    }

    if (!components_.primary_mesh) {
      components_.primary_mesh = mesh;
      components_.primary_mesh_name = mesh_name;
    }
  }

  oopCout() << "[svMultiPhysics::Application] SimulationBuilder: meshes loaded="
            << static_cast<int>(components_.meshes.size()) << " primary='" << components_.primary_mesh_name << "'"
            << std::endl;
}

void SimulationBuilder::createFESystem()
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  throw std::runtime_error(
      "[svMultiPhysics::Application] The new OOP solver path requires FE_WITH_MESH (SVMP_FE_WITH_MESH=1). "
      "Rebuild with FE_WITH_MESH enabled or set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
#else
  if (!components_.primary_mesh) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] No mesh was loaded; cannot create FE system for the new solver. "
        "Check your <Add_mesh> section or set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  const auto& loaded_participants = components_.mesh_collection.participants();
  if (loaded_participants.size() <= 1u) {
    svmp::FE::systems::MeshParticipantInfo participant_info{};
    participant_info.name = components_.primary_mesh_name.empty() ? "mesh" : components_.primary_mesh_name;
    if (!loaded_participants.empty()) {
      participant_info.name = loaded_participants.front().name;
      participant_info.domain_id = loaded_participants.front().domain_id;
    }

    components_.fe_system = std::make_unique<svmp::FE::systems::FESystem>(
        components_.primary_mesh, std::move(participant_info), svmp::Configuration::Reference);
    oopCout() << "[svMultiPhysics::Application] SimulationBuilder: created FE system from primary mesh '"
              << components_.primary_mesh_name << "'" << std::endl;
    return;
  }

  std::vector<svmp::FE::assembly::CompositeMeshParticipant> composite_participants;
  composite_participants.reserve(loaded_participants.size());
  for (const auto& participant : loaded_participants) {
    if (!participant.mesh) {
      throw std::runtime_error("[svMultiPhysics::Application] Mesh participant '" + participant.name +
                               "' has no loaded mesh.");
    }
    composite_participants.push_back(svmp::FE::assembly::CompositeMeshParticipant{
        .name = participant.name,
        .mesh = participant.mesh.get(),
        .domain_id = participant.domain_id});
  }

  auto composite_access = std::make_shared<svmp::FE::assembly::CompositeMeshAccess>(
      std::move(composite_participants));
  components_.fe_system = std::make_unique<svmp::FE::systems::FESystem>(std::move(composite_access));
  oopCout() << "[svMultiPhysics::Application] SimulationBuilder: created FE system from "
            << static_cast<int>(loaded_participants.size()) << " mesh participants" << std::endl;
#endif
}

void SimulationBuilder::createPhysicsModules()
{
  if (!components_.fe_system) {
    throw std::runtime_error("[svMultiPhysics::Application] createPhysicsModules() called before createFESystem().");
  }

  const auto declared =
      static_cast<int>(std::count_if(params_.equation_parameters.begin(), params_.equation_parameters.end(),
                                     [](const auto* p) { return p != nullptr; }));
  oopCout() << "[svMultiPhysics::Application] SimulationBuilder: equations declared=" << declared << std::endl;

  components_.physics_modules.clear();
  for (const auto* eq_params : params_.equation_parameters) {
    if (!eq_params) {
      continue;
    }

    oopCout() << "[svMultiPhysics::Application]   Translating equation type='" << eq_params->type.value() << "'"
              << " domains=" << static_cast<int>(eq_params->domains.size())
              << " bcs=" << static_cast<int>(eq_params->boundary_conditions.size()) << std::endl;

    const auto equation_cut_restrictions =
        equationCellDomainRestrictions(*components_.fe_system, *eq_params);
    std::vector<svmp::FE::systems::FESystem::FormCellDomainRestriction>
        cell_restrictions;
    cell_restrictions.reserve(equation_cut_restrictions.size());
    for (const auto& cut_restriction : equation_cut_restrictions) {
      cell_restrictions.push_back(cut_restriction.restriction);
    }
    if (!cell_restrictions.empty()) {
      oopCout() << "[svMultiPhysics::Application]   Applying equation-level cut-domain cell restriction count="
                << static_cast<int>(cell_restrictions.size()) << std::endl;
    }
    auto cell_restriction_scope =
        components_.fe_system->scopedFormInstallCellDomainRestrictions(
            std::move(cell_restrictions));
    auto module = application::translators::EquationTranslator::createModule(
        *eq_params, *components_.fe_system, components_.meshes);
    cell_restriction_scope.restore();
    for (const auto& cut_restriction : equation_cut_restrictions) {
      application::core::validateEquationLevelCutVolumeConsumer(
          *components_.fe_system,
          cut_restriction.request,
          cut_restriction.restriction.interface_marker);
    }
    components_.physics_modules.push_back(std::move(module));
  }

  if (components_.physics_modules.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] No equations were provided via <Add_equation>; the new solver requires at least "
        "one equation. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }
}

void SimulationBuilder::setupSystem()
{
  if (!components_.fe_system) {
    throw std::runtime_error("[svMultiPhysics::Application] setupSystem() called before createFESystem().");
  }

  // All module registrations must occur before setup finalizes dof maps/sparsity/etc.
  svmp::FE::systems::SetupOptions setup_opts{};
  {
    const auto comm = svmp::MeshComm::world();
    setup_opts.dof_options.my_rank = comm.rank();
    setup_opts.dof_options.world_size = comm.size();
#if FE_HAS_MPI
    setup_opts.dof_options.mpi_comm = comm.native();
#endif
  }

  // FSILS is a point-block backend: its sparse graph is node-blocked, with
  // field components carried inside each block.  Default to Morton node
  // ordering for locality; keep SVMP_FSILS_LOCALITY_ORDERING and
  // SVMP_DOF_NUMBERING as explicit overrides for benchmarking/debugging.
  const bool fsils_backend = selectBackend(params_) == svmp::FE::backends::BackendKind::FSILS;
  if (fsils_backend) {
    if (applyFsilsLocalityOrderingEnv(setup_opts.dof_options)) {
      oopCout() << "[svMultiPhysics::Application] SVMP_FSILS_LOCALITY_ORDERING="
                << std::getenv("SVMP_FSILS_LOCALITY_ORDERING")
                << " -> FSILS locality numbering override applied" << std::endl;
    } else if (std::getenv("SVMP_DOF_NUMBERING") == nullptr) {
      setup_opts.dof_options.numbering = svmp::FE::dofs::DofNumberingStrategy::Morton;
      oopCout() << "[svMultiPhysics::Application] FSILS backend: defaulting locality numbering to Morton"
                << std::endl;
    }
  }
  if (fsils_backend) {
    setup_opts.use_backend_row_ownership_for_assembly = true;
    if (setup_opts.dof_options.world_size > 1) {
      setup_opts.retain_serial_sparsity = false;
      oopCout() << "[svMultiPhysics::Application] FSILS backend: retaining distributed sparsity only in MPI"
                << std::endl;
    }
  }

  // Allow env var override for DOF numbering strategy
  if (const char* env = std::getenv("SVMP_DOF_NUMBERING"); env != nullptr) {
    const std::string val = lower_copy(std::string(env));
    using S = svmp::FE::dofs::DofNumberingStrategy;
    if (val == "rcm" || val == "cuthill-mckee") {
      setup_opts.dof_options.numbering = S::CuthillMcKee;
    } else if (val == "morton") {
      setup_opts.dof_options.numbering = S::Morton;
    } else if (val == "hilbert") {
      setup_opts.dof_options.numbering = S::Hilbert;
    } else if (val == "sequential") {
      setup_opts.dof_options.numbering = S::Sequential;
    } else if (val == "interleaved") {
      setup_opts.dof_options.numbering = S::Interleaved;
    } else if (val == "block") {
      setup_opts.dof_options.numbering = S::Block;
    }
    oopCout() << "[svMultiPhysics::Application] SVMP_DOF_NUMBERING=" << env
              << " -> numbering override applied" << std::endl;
  }

  components_.fe_system->setup(setup_opts);
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
  if (components_.fe_system->hasSingleNativeMesh()) {
    const auto prescribed_values_from_mesh =
        components_.fe_system->syncPrescribedVertexFieldsFromMeshFields();
    if (prescribed_values_from_mesh > 0u) {
      oopCout() << "[svMultiPhysics::Application] SimulationBuilder: prescribed vertex field values synchronized from mesh="
                << prescribed_values_from_mesh << std::endl;
    }
  }
#endif
  oopCout() << "[svMultiPhysics::Application] SimulationBuilder: FE storage plan "
            << components_.fe_system->setupStoragePlan().summary() << std::endl;
  if (components_.primary_mesh) {
    const auto& base = components_.primary_mesh->base();
    oopCout() << "[svMultiPhysics::Application] SimulationBuilder: mesh storage materialized codim1="
              << codim1StorageName(base.codim1_storage_mode())
              << " faces=" << base.n_faces()
              << " edges=" << base.n_edges() << std::endl;
  }

  // Materialize deployed auxiliary models (RCR, resistance BCs, etc.) into
  // the AuxiliaryStateManager so their output buffers are allocated and
  // steppers are ready.  Must happen after setup() (mesh/DOF info needed)
  // and before the time loop uses auxiliary output values.
  components_.fe_system->finalizeAuxiliaryLayout();

  const auto n_dofs = components_.fe_system->dofHandler().getNumDofs();
  {
    const auto& part = components_.fe_system->dofHandler().getPartition();
    const bool owned_contiguous = part.locallyOwned().contiguousRange().has_value();
    oopCout() << "[svMultiPhysics::Application] SimulationBuilder: DOF partition global=" << part.globalSize()
              << " owned=" << part.localOwnedSize() << " ghost=" << part.ghostSize()
              << " relevant=" << part.localRelevantSize() << " owned_contiguous=" << (owned_contiguous ? "true" : "false")
              << std::endl;
  }
  oopCout() << "[svMultiPhysics::Application] SimulationBuilder: FE system setup complete; ndofs=" << n_dofs
            << " constraints=" << components_.fe_system->constraints().numConstraints() << std::endl;

  const auto& fmap = components_.fe_system->fieldMap();
  oopCout() << "[svMultiPhysics::Application] SimulationBuilder: fields=" << static_cast<int>(fmap.numFields())
            << std::endl;
  for (std::size_t i = 0; i < fmap.numFields(); ++i) {
    const auto& f = fmap.getField(i);
    oopCout() << "[svMultiPhysics::Application]   field[" << i << "] name='" << f.name
              << "' components=" << f.n_components << " dofs=" << f.n_dofs << " offset=" << f.dof_offset
              << std::endl;
  }
}

void SimulationBuilder::createSolvers()
{
  if (!components_.fe_system || !components_.fe_system->isSetup()) {
    throw std::runtime_error("[svMultiPhysics::Application] createSolvers() requires setupSystem() to run first.");
  }

  const auto backend_kind = selectBackend(params_);
  oopCout() << "[svMultiPhysics::Application] SimulationBuilder: backend="
            << svmp::FE::backends::backendKindToString(backend_kind) << std::endl;

  svmp::FE::backends::BackendFactory::CreateOptions create_options{};
  create_options.dof_per_node = 1;
  {
    int dof_per_node = 0;
    const auto& fmap = components_.fe_system->fieldMap();
    for (std::size_t i = 0; i < fmap.numFields(); ++i) {
      dof_per_node += fmap.getField(i).n_components;
    }
    if (dof_per_node > 0) {
      create_options.dof_per_node = dof_per_node;
    }
  }

  oopCout() << "[svMultiPhysics::Application] SimulationBuilder: dof_per_node=" << create_options.dof_per_node
            << std::endl;

#if FE_HAS_MPI
  {
    const auto comm = svmp::MeshComm::world();
    create_options.mpi_comm = comm.native();
  }
#endif

  if (backend_kind == svmp::FE::backends::BackendKind::FSILS && create_options.dof_per_node > 1) {
    oopCout() << "[svMultiPhysics::Application] SimulationBuilder: configuring FSILS DOF permutation." << std::endl;
    svmp::FE::dofs::DofDistributionOptions dof_options{};
    {
      const auto comm = svmp::MeshComm::world();
      dof_options.my_rank = comm.rank();
      dof_options.world_size = comm.size();
#if FE_HAS_MPI
      dof_options.mpi_comm = comm.native();
#endif
    }

    // Prefer the permutation built by the FE system setup (MPI-safe and consistent with any
    // node-interleaved distributed sparsity / backend indexing).
    create_options.dof_permutation = components_.fe_system->dofPermutation();
    if (!create_options.dof_permutation || create_options.dof_permutation->empty()) {
      if (dof_options.world_size > 1) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] FSILS requires a node-interleaved DOF permutation in MPI runs, but the FE "
            "system did not provide one. This typically indicates distributed sparsity/permutation setup failed.");
      }
      create_options.dof_permutation =
          build_fsils_dof_permutation(*components_.fe_system, create_options.dof_per_node, dof_options);
    }

    oopCout() << "[svMultiPhysics::Application] SimulationBuilder: FSILS DOF permutation="
              << (create_options.dof_permutation ? "enabled" : "disabled") << std::endl;
  }

  components_.backend = svmp::FE::backends::BackendFactory::create(backend_kind, create_options);
  if (!components_.backend) {
    throw std::runtime_error("[svMultiPhysics::Application] Failed to create FE backend factory.");
  }

  auto solver_options = translateSolverOptions(params_, backend_kind);

  // Populate block layout from field metadata — available for any multi-field system.
  {
    const auto& fmap = components_.fe_system->fieldMap();
    if (fmap.numFields() > 0) {
      svmp::FE::backends::BlockLayout layout{};
      int offset = 0;
      for (std::size_t i = 0; i < fmap.numFields(); ++i) {
        const auto& f = fmap.getField(i);
        const int ncomp = static_cast<int>(f.n_components);
        layout.blocks.push_back({f.name, offset, ncomp});
        offset += ncomp;
      }

      const bool is_ustruct_blockschur =
          backend_kind == svmp::FE::backends::BackendKind::FSILS &&
          solver_options.method == svmp::FE::backends::SolverMethod::BlockSchur &&
          firstEquationTypeIs(params_, "ustruct");
      if (is_ustruct_blockschur) {
        const auto* displacement = layout.findBlock("Displacement");
        const auto* velocity = layout.findBlock("Velocity");
        const auto* pressure = layout.findBlock("Pressure");
        if (displacement && velocity && pressure &&
            displacement->n_components == velocity->n_components &&
            pressure->n_components == 1 &&
            displacement->start_component == 0 &&
            velocity->start_component == displacement->n_components &&
            pressure->start_component == displacement->n_components + velocity->n_components) {
          svmp::FE::backends::BlockLayout grouped_layout{};
          grouped_layout.blocks.push_back({
              "UstructKinematicMomentum",
              displacement->start_component,
              displacement->n_components + velocity->n_components,
              svmp::FE::backends::BlockRole::PrimaryField});
          grouped_layout.blocks.push_back({
              pressure->name,
              pressure->start_component,
              pressure->n_components,
              svmp::FE::backends::BlockRole::ConstraintField});
          grouped_layout.momentum_block = 0;
          grouped_layout.constraint_block = 1;
          layout = std::move(grouped_layout);
        }
      }

      const bool is_level_set_fluid_blockschur =
          backend_kind == svmp::FE::backends::BackendKind::FSILS &&
          solver_options.method == svmp::FE::backends::SolverMethod::BlockSchur &&
          firstEquationTypeIs(params_, "level_set");
      if (is_level_set_fluid_blockschur) {
        const auto* level_set = layout.findBlock("phi");
        const auto* velocity = layout.findBlock("Velocity");
        const auto* pressure = layout.findBlock("Pressure");
        if (level_set && velocity && pressure &&
            level_set->n_components == 1 &&
            velocity->n_components > 1 &&
            pressure->n_components == 1 &&
            level_set->start_component == 0 &&
            velocity->start_component == level_set->n_components &&
            pressure->start_component == level_set->n_components + velocity->n_components) {
          svmp::FE::backends::BlockLayout grouped_layout{};
          grouped_layout.blocks.push_back({
              "LevelSetVelocity",
              level_set->start_component,
              level_set->n_components + velocity->n_components,
              svmp::FE::backends::BlockRole::PrimaryField});
          grouped_layout.blocks.push_back({
              pressure->name,
              pressure->start_component,
              pressure->n_components,
              svmp::FE::backends::BlockRole::ConstraintField});
          grouped_layout.momentum_block = 0;
          grouped_layout.constraint_block = 1;
          layout = std::move(grouped_layout);
        }
      }

      // Identify saddle-point block pair when the solver/backend can benefit from
      // explicit momentum/constraint roles.  For FSILS this is also required by
      // native outlet-coupled GMRES solves, not just the BlockSchur path.
      const bool should_identify_saddle_point =
          solver_options.method == svmp::FE::backends::SolverMethod::BlockSchur ||
          !solver_options.momentum_block_name.empty() ||
          !solver_options.constraint_block_name.empty() ||
          (backend_kind == svmp::FE::backends::BackendKind::FSILS &&
           layout.blocks.size() == 2u);
      if (should_identify_saddle_point) {
        // Prefer explicit block names from solver configuration.
        if (!solver_options.momentum_block_name.empty() || !solver_options.constraint_block_name.empty()) {
          for (int bi = 0; bi < static_cast<int>(layout.blocks.size()); ++bi) {
            if (!layout.momentum_block && layout.blocks[static_cast<std::size_t>(bi)].name == solver_options.momentum_block_name) {
              layout.momentum_block = bi;
            }
            if (!layout.constraint_block && layout.blocks[static_cast<std::size_t>(bi)].name == solver_options.constraint_block_name) {
              layout.constraint_block = bi;
            }
          }
        }

        // Fall back to heuristic: first multi-component field = field-A (momentum),
        // first single-component field = field-B (constraint).
        if (!layout.momentum_block || !layout.constraint_block) {
          for (int bi = 0; bi < static_cast<int>(layout.blocks.size()); ++bi) {
            if (!layout.momentum_block && layout.blocks[static_cast<std::size_t>(bi)].n_components > 1) {
              layout.momentum_block = bi;
            } else if (!layout.constraint_block && layout.blocks[static_cast<std::size_t>(bi)].n_components == 1) {
              layout.constraint_block = bi;
            }
          }
        }
      }

      solver_options.block_layout = std::move(layout);
    }
  }

  oopCout() << "[svMultiPhysics::Application] SimulationBuilder: linear solver method="
            << svmp::FE::backends::solverMethodToString(solver_options.method)
            << " preconditioner=" << svmp::FE::backends::preconditionerToString(solver_options.preconditioner)
            << " rel_tol=" << solver_options.rel_tol << " abs_tol=" << solver_options.abs_tol
            << " max_iter=" << solver_options.max_iter;
  if (solver_options.block_layout) {
    oopCout() << " block_layout=[";
    for (std::size_t i = 0; i < solver_options.block_layout->blocks.size(); ++i) {
      if (i > 0) oopCout() << ", ";
      const auto& blk = solver_options.block_layout->blocks[i];
      oopCout() << blk.name << "(" << blk.start_component << ":" << blk.n_components << ")";
    }
    oopCout() << "]";
    if (solver_options.block_layout->hasSaddlePoint()) {
      oopCout() << " saddle_point=(" << *solver_options.block_layout->momentum_block
                << "," << *solver_options.block_layout->constraint_block << ")";
    }
  }
  oopCout() << std::endl;

  components_.fe_system->setAnalysisSolverOptions(solver_options);
  const char* analysis_log = std::getenv("SVMP_FE_ANALYSIS_LOG");
  const auto analysis_log_mode = analysis_log ? lower_copy(analysis_log) : std::string{};
  const bool mpi_setup_analysis_requested = !analysis_log_mode.empty();
  const bool parallel_run = svmp::MeshComm::world().is_parallel();
  if (!parallel_run || mpi_setup_analysis_requested) {
    if (parallel_run && mpi_setup_analysis_requested) {
      oopCout() << "[svMultiPhysics::Application] SVMP_FE_ANALYSIS_LOG requested during MPI setup; "
                << "running setup analysis before FSILS priming." << std::endl;
    }
    const auto& report = components_.fe_system->analysisReport();
    if (analysis_log_mode == "full") {
      report.print(oopCout());
    } else {
      report.printApplicationLog(oopCout());
      if (analysis_log_mode == "trace") {
        report.printTraceLog(oopCout(), components_.fe_system->latestAnalysisSummaries());
      }
    }
  } else if (oopTraceEnabled()) {
    oopCout() << "[svMultiPhysics::Application] SimulationBuilder: skipping implicit setup analysis report in MPI; "
              << "set SVMP_FE_ANALYSIS_LOG to request it." << std::endl;
  }

  components_.linear_solver = components_.backend->createLinearSolver(solver_options);
  if (!components_.linear_solver) {
    throw std::runtime_error("[svMultiPhysics::Application] Failed to create FE linear solver.");
  }

  // Prime backend layout (notably FSILS) by creating the system matrix once.
  if (backend_kind == svmp::FE::backends::BackendKind::FSILS) {
    const auto* dist = components_.fe_system->distributedSparsityIfAvailable("equations");
    if (dist) {
      oopCout() << "[svMultiPhysics::Application] SimulationBuilder: priming FSILS matrix layout; distributed rows="
                << dist->globalRows() << " cols=" << dist->globalCols()
                << " local_nnz=" << dist->getLocalNnz() << std::endl;
      (void)components_.backend->createMatrix(*dist);
    } else {
      const auto& pat = components_.fe_system->sparsity("equations");
      oopCout() << "[svMultiPhysics::Application] SimulationBuilder: priming FSILS matrix layout; pattern rows="
                << pat.numRows() << " cols=" << pat.numCols() << " nnz=" << pat.getNnz() << std::endl;
      (void)components_.backend->createMatrix(pat);
    }
  }
}

void SimulationBuilder::allocateHistory()
{
  if (!components_.fe_system || !components_.fe_system->isSetup()) {
    throw std::runtime_error("[svMultiPhysics::Application] allocateHistory() requires setupSystem() to run first.");
  }
  if (!components_.backend) {
    throw std::runtime_error("[svMultiPhysics::Application] allocateHistory() requires createSolvers() to run first.");
  }

  const auto ndofs = components_.fe_system->dofHandler().getNumDofs();
  const int history_depth = 2;
  oopCout() << "[svMultiPhysics::Application] SimulationBuilder: allocating TimeHistory ndofs=" << ndofs
            << " depth=" << history_depth << std::endl;

  auto history = svmp::FE::timestepping::TimeHistory::allocate(*components_.backend, ndofs, history_depth,
                                                               /*allocate_second_order_state=*/false);
  history.setTime(0.0);
  double dt = params_.general_simulation_parameters.time_step_size.value();
  if (!(dt > 0.0)) {
    dt = 1.0;
    oopCout() << "[svMultiPhysics::Application] Time_step_size is not set or <= 0; using dt=1.0." << std::endl;
  }
  history.setDt(dt);
  history.setPrevDt(dt);
  history.setStepIndex(0);
  history.primeDtHistory(dt);

  for (const auto& module : components_.physics_modules) {
    if (module) {
      module->applyInitialConditions(*components_.fe_system, history.u());
      traceStateVectorFields(*components_.fe_system, history.u(), "after_module_initial_conditions");
    }
  }
  for (int k = 1; k <= history.historyDepth(); ++k) {
    history.uPrevK(k).copyFrom(history.u());
  }
  traceStateVectorFields(*components_.fe_system, history.u(), "after_history_copy");
  history.updateGhosts();
  traceStateVectorFields(*components_.fe_system, history.u(), "after_history_update_ghosts");

  components_.time_history = std::make_unique<svmp::FE::timestepping::TimeHistory>(std::move(history));
  oopCout() << "[svMultiPhysics::Application] SimulationBuilder: TimeHistory initialized time="
            << components_.time_history->time() << " dt=" << components_.time_history->dt()
            << " step=" << components_.time_history->stepIndex() << std::endl;
}

} // namespace core
} // namespace application
