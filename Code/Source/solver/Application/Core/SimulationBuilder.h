#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Application/Core/MeshCollection.h"
#include "Mesh/Mesh.h"

class Parameters;

namespace svmp {
namespace FE {
namespace systems {
class FESystem;
} // namespace systems

namespace interfaces {
class FreeSurfaceGeometrySnapshotCache;
} // namespace interfaces

namespace backends {
class BackendFactory;
class LinearSolver;
struct BlockLayout;
enum class BackendKind : std::uint8_t;
enum class SolverMethod : std::uint8_t;
} // namespace backends

namespace timestepping {
class TimeHistory;
} // namespace timestepping
} // namespace FE

namespace Physics {
class PhysicsModule;
} // namespace Physics
} // namespace svmp

namespace application {
namespace core {

struct SimulationComponents {
  MeshCollection mesh_collection{};
  std::map<std::string, std::shared_ptr<svmp::Mesh>> meshes{};
  std::shared_ptr<svmp::Mesh> primary_mesh{};
  std::string primary_mesh_name{};

  std::unique_ptr<svmp::FE::systems::FESystem> fe_system{};
  std::unique_ptr<svmp::FE::interfaces::FreeSurfaceGeometrySnapshotCache>
      free_surface_geometry_snapshot_cache{};
  std::vector<std::unique_ptr<svmp::Physics::PhysicsModule>> physics_modules{};

  std::unique_ptr<svmp::FE::backends::BackendFactory> backend{};
  std::unique_ptr<svmp::FE::backends::LinearSolver> linear_solver{};

  std::unique_ptr<svmp::FE::timestepping::TimeHistory> time_history{};

  SimulationComponents();
  SimulationComponents(SimulationComponents&&) noexcept;
  SimulationComponents& operator=(SimulationComponents&&) noexcept;
  ~SimulationComponents();

  SimulationComponents(const SimulationComponents&) = delete;
  SimulationComponents& operator=(const SimulationComponents&) = delete;
};

namespace detail {

/**
 * Validate cross-equation Physics dependencies before predeclaring fields.
 *
 * Kept as one testable ordering boundary: every Navier--Stokes fitted
 * surface/contact capability is checked before future wet-extension velocity
 * ownership is allowed to mutate the FE system.
 */
void preflightAndPreRegisterPhysicsModuleDependencies(
    const Parameters& params,
    SimulationComponents& components);

/**
 * Form the exact two-block FSILS partition for the monolithic material-
 * interface/two-fluid unknown layout.  Returns false without mutation when
 * the declarations or field layout do not match the declared contract.
 */
bool groupTwoFluidMaterialInterfaceFsilsLayout(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::backends::BlockLayout& layout);

/**
 * Enforce the solver envelope for a declared material-interface/two-fluid
 * pair and install its exact two-block layout.  Systems without either
 * declaration are left unchanged.
 */
void requireTwoFluidMaterialInterfaceSolverLayout(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::backends::BackendKind backend_kind,
    svmp::FE::backends::SolverMethod solver_method,
    svmp::FE::backends::BlockLayout& layout);

} // namespace detail

class SimulationBuilder {
public:
  explicit SimulationBuilder(const Parameters& params);
  SimulationComponents build();

private:
  void loadMeshes();
  void createFESystem();
  void createPhysicsModules();
  void setupSystem();
  void createSolvers();
  void allocateHistory();

  const Parameters& params_;
  SimulationComponents components_{};
};

} // namespace core
} // namespace application
