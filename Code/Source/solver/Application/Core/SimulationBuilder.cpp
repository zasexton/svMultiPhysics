#include "Application/Core/SimulationBuilder.h"

#include "Application/Translators/EquationTranslator.h"
#include "Application/Translators/MeshTranslator.h"

#include "FE/Backends/Interfaces/BackendFactory.h"
#include "FE/Backends/Interfaces/BackendKind.h"
#include "FE/Backends/Interfaces/LinearSolver.h"
#include "FE/Backends/Utils/BackendOptions.h"
#include "FE/Dofs/EntityDofMap.h"
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

std::shared_ptr<const svmp::FE::backends::DofPermutation> build_fsils_dof_permutation(const svmp::FE::systems::FESystem& system,
                                                                                      int dof_per_node)
{
  using svmp::FE::GlobalIndex;
  using svmp::FE::backends::DofPermutation;

  if (dof_per_node <= 0) {
    return {};
  }

  const GlobalIndex total_dofs = system.dofHandler().getNumDofs();
  if (total_dofs <= 0) {
    return {};
  }

  // Prefer the merged entity map (robust against component-layout assumptions).
  if (const auto* emap = system.dofHandler().getEntityDofMap()) {
    const GlobalIndex n_vertices = emap->numVertices();
    if (n_vertices > 0 && total_dofs == static_cast<GlobalIndex>(dof_per_node) * n_vertices) {
      auto perm = std::make_shared<DofPermutation>();
      perm->forward.assign(static_cast<std::size_t>(total_dofs), svmp::FE::INVALID_GLOBAL_INDEX);
      perm->inverse.assign(static_cast<std::size_t>(total_dofs), svmp::FE::INVALID_GLOBAL_INDEX);

      for (GlobalIndex v = 0; v < n_vertices; ++v) {
        const auto vdofs = emap->getVertexDofs(v);
        if (vdofs.size() != static_cast<std::size_t>(dof_per_node)) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] FSILS backend requires dof_per_node DOFs per vertex for nodal systems.");
        }
        for (std::size_t c = 0; c < vdofs.size(); ++c) {
          const GlobalIndex fe_dof = vdofs[c];
          const GlobalIndex fs_dof = v * static_cast<GlobalIndex>(dof_per_node) + static_cast<GlobalIndex>(c);
          if (fe_dof < 0 || fe_dof >= total_dofs) {
            throw std::runtime_error("[svMultiPhysics::Application] FSILS backend permutation encountered invalid DOF.");
          }
          perm->forward[static_cast<std::size_t>(fe_dof)] = fs_dof;
          perm->inverse[static_cast<std::size_t>(fs_dof)] = fe_dof;
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
  }

  // Fallback: derive node-block permutation from the field map (requires equal-order fields).
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

  auto perm = std::make_shared<DofPermutation>();
  perm->forward.assign(static_cast<std::size_t>(total_dofs), svmp::FE::INVALID_GLOBAL_INDEX);
  perm->inverse.assign(static_cast<std::size_t>(total_dofs), svmp::FE::INVALID_GLOBAL_INDEX);

  for (GlobalIndex node = 0; node < n_nodes; ++node) {
    int comp_offset = 0;
    for (std::size_t f = 0; f < n_fields; ++f) {
      const auto& field = fmap.getField(f);
      for (svmp::FE::LocalIndex c = 0; c < field.n_components; ++c) {
        const GlobalIndex fe_dof = fmap.componentToGlobal(f, c, node);
        const GlobalIndex fs_dof =
            node * static_cast<GlobalIndex>(dof_per_node) + static_cast<GlobalIndex>(comp_offset);
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
  const auto declared =
      static_cast<int>(std::count_if(params_.mesh_parameters.begin(), params_.mesh_parameters.end(),
                                     [](const auto* p) { return p != nullptr; }));
  std::cout << "[svMultiPhysics::Application] SimulationBuilder: meshes declared=" << declared << std::endl;

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
    std::cout << "[svMultiPhysics::Application]   Loading mesh '" << mesh_name << "' path='" << mesh_path << "'"
              << " faces=" << static_cast<int>(mesh_params->face_parameters.size());
    if (mesh_params->domain_id.defined()) {
      std::cout << " domain_id=" << mesh_params->domain_id.value();
    }
    std::cout << std::endl;

    auto mesh = application::translators::MeshTranslator::loadMesh(*mesh_params);
    components_.meshes.emplace(mesh_name, mesh);

    if (mesh) {
      std::cout << "[svMultiPhysics::Application]   Mesh '" << mesh_name << "': dim=" << mesh->dim()
                << " vertices=" << mesh->n_vertices() << " cells=" << mesh->n_cells()
                << " faces=" << mesh->n_faces() << std::endl;
    }

    if (!components_.primary_mesh) {
      components_.primary_mesh = mesh;
      components_.primary_mesh_name = mesh_name;
    }
  }

  std::cout << "[svMultiPhysics::Application] SimulationBuilder: meshes loaded="
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

  components_.fe_system =
      std::make_unique<svmp::FE::systems::FESystem>(components_.primary_mesh, svmp::Configuration::Reference);
  std::cout << "[svMultiPhysics::Application] SimulationBuilder: created FE system from primary mesh '"
            << components_.primary_mesh_name << "'" << std::endl;
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
  std::cout << "[svMultiPhysics::Application] SimulationBuilder: equations declared=" << declared << std::endl;

  components_.physics_modules.clear();
  for (const auto* eq_params : params_.equation_parameters) {
    if (!eq_params) {
      continue;
    }

    std::cout << "[svMultiPhysics::Application]   Translating equation type='" << eq_params->type.value() << "'"
              << " domains=" << static_cast<int>(eq_params->domains.size())
              << " bcs=" << static_cast<int>(eq_params->boundary_conditions.size()) << std::endl;

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
  svmp::FE::systems::SetupOptions setup_opts{};
  {
    const auto comm = svmp::MeshComm::world();
    setup_opts.dof_options.my_rank = comm.rank();
    setup_opts.dof_options.world_size = comm.size();
#if FE_HAS_MPI
    setup_opts.dof_options.mpi_comm = comm.native();
#endif
  }
  components_.fe_system->setup(setup_opts);

  const auto n_dofs = components_.fe_system->dofHandler().getNumDofs();
  std::cout << "[svMultiPhysics::Application] SimulationBuilder: FE system setup complete; ndofs=" << n_dofs
            << " constraints=" << components_.fe_system->constraints().numConstraints() << std::endl;

  const auto& fmap = components_.fe_system->fieldMap();
  std::cout << "[svMultiPhysics::Application] SimulationBuilder: fields=" << static_cast<int>(fmap.numFields())
            << std::endl;
  for (std::size_t i = 0; i < fmap.numFields(); ++i) {
    const auto& f = fmap.getField(i);
    std::cout << "[svMultiPhysics::Application]   field[" << i << "] name='" << f.name
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
  std::cout << "[svMultiPhysics::Application] SimulationBuilder: backend="
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

  std::cout << "[svMultiPhysics::Application] SimulationBuilder: dof_per_node=" << create_options.dof_per_node
            << std::endl;

  if (backend_kind == svmp::FE::backends::BackendKind::FSILS && create_options.dof_per_node > 1) {
    std::cout << "[svMultiPhysics::Application] SimulationBuilder: building FSILS DOF permutation." << std::endl;
    create_options.dof_permutation = build_fsils_dof_permutation(*components_.fe_system, create_options.dof_per_node);
    std::cout << "[svMultiPhysics::Application] SimulationBuilder: FSILS DOF permutation="
              << (create_options.dof_permutation ? "enabled" : "disabled") << std::endl;
  }

  components_.backend = svmp::FE::backends::BackendFactory::create(backend_kind, create_options);
  if (!components_.backend) {
    throw std::runtime_error("[svMultiPhysics::Application] Failed to create FE backend factory.");
  }

  const auto solver_options = translateSolverOptions(params_, backend_kind);
  std::cout << "[svMultiPhysics::Application] SimulationBuilder: linear solver method="
            << svmp::FE::backends::solverMethodToString(solver_options.method)
            << " preconditioner=" << svmp::FE::backends::preconditionerToString(solver_options.preconditioner)
            << " rel_tol=" << solver_options.rel_tol << " abs_tol=" << solver_options.abs_tol
            << " max_iter=" << solver_options.max_iter << std::endl;
  components_.linear_solver = components_.backend->createLinearSolver(solver_options);
  if (!components_.linear_solver) {
    throw std::runtime_error("[svMultiPhysics::Application] Failed to create FE linear solver.");
  }

  // Prime backend layout (notably FSILS) by creating the system matrix once.
  if (backend_kind == svmp::FE::backends::BackendKind::FSILS) {
    const auto& pat = components_.fe_system->sparsity("jacobian");
    std::cout << "[svMultiPhysics::Application] SimulationBuilder: priming FSILS matrix layout; pattern rows="
              << pat.numRows() << " cols=" << pat.numCols() << " nnz=" << pat.getNnz() << std::endl;
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
  std::cout << "[svMultiPhysics::Application] SimulationBuilder: allocating TimeHistory ndofs=" << ndofs
            << " depth=" << history_depth << std::endl;

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
  std::cout << "[svMultiPhysics::Application] SimulationBuilder: TimeHistory initialized time="
            << components_.time_history->time() << " dt=" << components_.time_history->dt()
            << " step=" << components_.time_history->stepIndex() << std::endl;
}

} // namespace core
} // namespace application
