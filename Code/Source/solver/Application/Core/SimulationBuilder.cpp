#include "Application/Core/SimulationBuilder.h"

#include "Application/Translators/EquationTranslator.h"
#include "Application/Translators/MeshTranslator.h"

#include "FE/Backends/Interfaces/BackendFactory.h"
#include "FE/Backends/Interfaces/BackendKind.h"
#include "FE/Backends/Interfaces/LinearSolver.h"
#include "FE/Backends/Utils/BackendOptions.h"
#include "FE/Systems/FESystem.h"
#include "FE/TimeStepping/TimeHistory.h"
#include "Mesh/Mesh.h"
#include "Physics/Core/PhysicsModule.h"
#include "Parameters.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>

namespace {

std::string lower_copy(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

svmp::FE::backends::SolverMethod toSolverMethod(const std::string& legacy_type)
{
  using svmp::FE::backends::SolverMethod;
  const auto v = lower_copy(legacy_type);

  if (v.empty()) {
    return SolverMethod::Direct;
  }
  if (v == "cg" || v == "conjugate-gradient") {
    return SolverMethod::CG;
  }
  if (v == "gmres") {
    return SolverMethod::GMRES;
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

svmp::FE::backends::BackendKind selectBackend(const Parameters& params)
{
  using svmp::FE::backends::BackendKind;

  const EquationParameters* eq = nullptr;
  for (const auto* e : params.equation_parameters) {
    if (e) {
      eq = e;
      break;
    }
  }

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

  return BackendKind::FSILS;
}

svmp::FE::backends::SolverOptions translateSolverOptions(const Parameters& params,
                                                         svmp::FE::backends::BackendKind backend_kind)
{
  using svmp::FE::backends::SolverOptions;

  SolverOptions opts{};

  const EquationParameters* eq = nullptr;
  for (const auto* e : params.equation_parameters) {
    if (e) {
      eq = e;
      break;
    }
  }

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
  }
  if (eq->linear_solver.max_iterations.defined()) {
    opts.max_iter = eq->linear_solver.max_iterations.value();
  }

  if (eq->linear_solver.linear_algebra.defined()) {
    const auto& la = eq->linear_solver.linear_algebra;
    opts.preconditioner = toPreconditioner(la.preconditioner.value());
    opts.fsils_use_rcs = (opts.preconditioner == svmp::FE::backends::PreconditionerType::RowColumnScaling);

    if (backend_kind == svmp::FE::backends::BackendKind::Trilinos && la.configuration_file.defined()) {
      opts.trilinos_xml_file = la.configuration_file.value();
    }
  }

  return opts;
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
  std::cout << "[svMultiPhysics::Application] SimulationBuilder: loadMeshes()" << std::endl;
  loadMeshes();

  std::cout << "[svMultiPhysics::Application] SimulationBuilder: createFESystem()" << std::endl;
  createFESystem();

  std::cout << "[svMultiPhysics::Application] SimulationBuilder: createPhysicsModules()" << std::endl;
  createPhysicsModules();

  std::cout << "[svMultiPhysics::Application] SimulationBuilder: setupSystem()" << std::endl;
  setupSystem();

  std::cout << "[svMultiPhysics::Application] SimulationBuilder: createSolvers()" << std::endl;
  createSolvers();

  std::cout << "[svMultiPhysics::Application] SimulationBuilder: allocateHistory()" << std::endl;
  allocateHistory();

  return std::move(components_);
}

void SimulationBuilder::loadMeshes()
{
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

    auto mesh = application::translators::MeshTranslator::loadMesh(*mesh_params);
    components_.meshes.emplace(mesh_name, mesh);

    if (!components_.primary_mesh) {
      components_.primary_mesh = mesh;
      components_.primary_mesh_name = mesh_name;
    }
  }
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

  auto mesh = svmp::create_mesh(components_.primary_mesh);
  components_.fe_system = std::make_unique<svmp::FE::systems::FESystem>(std::move(mesh), svmp::Configuration::Reference);
#endif
}

void SimulationBuilder::createPhysicsModules()
{
  if (!components_.fe_system) {
    throw std::runtime_error("[svMultiPhysics::Application] createPhysicsModules() called before createFESystem().");
  }

  components_.physics_modules.clear();
  for (const auto* eq_params : params_.equation_parameters) {
    if (!eq_params) {
      continue;
    }

    auto module = application::translators::EquationTranslator::createModule(*eq_params, *components_.fe_system,
                                                                             components_.meshes);
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
  components_.fe_system->setup();
}

void SimulationBuilder::createSolvers()
{
  if (!components_.fe_system || !components_.fe_system->isSetup()) {
    throw std::runtime_error("[svMultiPhysics::Application] createSolvers() requires setupSystem() to run first.");
  }

  const auto backend_kind = selectBackend(params_);

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

  components_.backend = svmp::FE::backends::BackendFactory::create(backend_kind, create_options);
  if (!components_.backend) {
    throw std::runtime_error("[svMultiPhysics::Application] Failed to create FE backend factory.");
  }

  const auto solver_options = translateSolverOptions(params_, backend_kind);
  components_.linear_solver = components_.backend->createLinearSolver(solver_options);
  if (!components_.linear_solver) {
    throw std::runtime_error("[svMultiPhysics::Application] Failed to create FE linear solver.");
  }

  // Prime backend layout (notably FSILS) by creating the system matrix once.
  if (backend_kind == svmp::FE::backends::BackendKind::FSILS) {
    const auto* dist = components_.fe_system->distributedSparsityIfAvailable("jacobian");
    if (dist) {
      (void)components_.backend->createMatrix(*dist);
    } else {
      (void)components_.backend->createMatrix(components_.fe_system->sparsity("jacobian"));
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

  auto history = svmp::FE::timestepping::TimeHistory::allocate(*components_.backend, ndofs, history_depth,
                                                               /*allocate_second_order_state=*/false);
  history.setTime(0.0);
  double dt = params_.general_simulation_parameters.time_step_size.value();
  if (!(dt > 0.0)) {
    dt = 1.0;
    std::cout << "[svMultiPhysics::Application] Time_step_size is not set or <= 0; using dt=1.0." << std::endl;
  }
  history.setDt(dt);
  history.setPrevDt(dt);
  history.setStepIndex(0);
  history.primeDtHistory(dt);

  components_.time_history = std::make_unique<svmp::FE::timestepping::TimeHistory>(std::move(history));
}

} // namespace core
} // namespace application
