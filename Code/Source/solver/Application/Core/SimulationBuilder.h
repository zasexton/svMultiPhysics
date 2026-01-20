#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

class Parameters;

namespace svmp {
class MeshBase;

namespace FE {
namespace systems {
class FESystem;
} // namespace systems

namespace backends {
class BackendFactory;
class LinearSolver;
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
  std::map<std::string, std::shared_ptr<svmp::MeshBase>> meshes{};
  std::shared_ptr<svmp::MeshBase> primary_mesh{};
  std::string primary_mesh_name{};

  std::unique_ptr<svmp::FE::systems::FESystem> fe_system{};
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
