/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceRunSupport.h"

#include "FE/Assembly/Assembler.h"
#include "FE/Backends/Interfaces/GenericVector.h"
#include "FE/Geometry/MappingFactory.h"
#include "FE/Math/Vector.h"
#include "FE/Systems/FESystem.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef MESH_HAS_MPI
#include <mpi.h>
#endif

namespace svmp::Physics::formulations::navier_stokes::free_surface::detail {

svmp::MeshComm activeFESystemCommunicator(
    const svmp::FE::systems::FESystem& system)
{
#if defined(MESH_HAS_MPI) && FE_HAS_MPI
  return svmp::MeshComm(system.activeMpiCommunicator());
#else
  (void)system;
  return svmp::MeshComm::self();
#endif
}

svmp::FE::systems::CollectiveCommunicator feCollectiveCommunicator(
    const svmp::MeshComm& comm)
{
#if defined(MESH_HAS_MPI) && FE_HAS_MPI
  return svmp::FE::systems::CollectiveCommunicator(
      comm.native(), static_cast<int>(comm.rank()), comm.size());
#else
  (void)comm;
  return {};
#endif
}

double globalSumDouble(double local, const svmp::MeshComm& comm)
{
#ifdef MESH_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized && comm.size() > 1) {
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm.native());
    return global;
  }
#else
  (void)comm;
#endif

  return local;
}

double globalMinDouble(double local, const svmp::MeshComm& comm)
{
  return svmp::FE::systems::collectiveMinDouble(
      local, feCollectiveCommunicator(comm));
}

double globalMaxDouble(double local, const svmp::MeshComm& comm)
{
  return svmp::FE::systems::collectiveMaxDouble(
      local, feCollectiveCommunicator(comm));
}

std::pair<std::uint64_t, std::uint64_t> globalMinMaxUint64(
    std::uint64_t local,
    const svmp::MeshComm& comm)
{
  return svmp::FE::systems::collectiveMinMaxUint64(
      local, feCollectiveCommunicator(comm));
}

std::size_t globalSumSize(std::size_t local, const svmp::MeshComm& comm)
{
  return svmp::FE::systems::collectiveSumSize(
      local, feCollectiveCommunicator(comm));
}

bool globalAnyBool(bool local, const svmp::MeshComm& comm)
{
  return svmp::FE::systems::collectiveAnyTrue(
      local, feCollectiveCommunicator(comm));
}

void requireCollectivePhasePreparation(
    const std::exception_ptr& local_failure,
    const svmp::MeshComm& comm,
    std::string_view phase)
{
  svmp::FE::systems::requireCollectivePhasePreparation(
      local_failure, feCollectiveCommunicator(comm), phase);
}

CutTopologyFingerprintMultiset collectiveCutTopologyFingerprintMultiset(
    const CutTopologyFingerprintMultiset& local,
    const svmp::MeshComm& comm)
{
  auto global = local;
#ifdef MESH_HAS_MPI
  if (comm.is_parallel()) {
#ifdef MPI_UINT64_T
    const MPI_Datatype fingerprint_type = MPI_UINT64_T;
#else
    const MPI_Datatype fingerprint_type = MPI_UNSIGNED_LONG_LONG;
#endif
    MPI_Allreduce(local.sums.data(),
                  global.sums.data(),
                  static_cast<int>(local.sums.size()),
                  fingerprint_type,
                  MPI_SUM,
                  comm.native());
    MPI_Allreduce(local.xors.data(),
                  global.xors.data(),
                  static_cast<int>(local.xors.size()),
                  fingerprint_type,
                  MPI_BXOR,
                  comm.native());
  }
#else
  if (comm.is_parallel()) {
    throw std::runtime_error(
        "Cut-topology multiset fingerprinting requires MPI in a parallel run.");
  }
#endif
  return global;
}

std::shared_ptr<svmp::FE::geometry::GeometryMapping> createCellGeometryMapping(
    const svmp::FE::assembly::IMeshAccess& mesh,
    svmp::FE::GlobalIndex cell)
{
  if (cell < 0 || cell >= mesh.numCells()) {
    return nullptr;
  }

  std::vector<std::array<svmp::FE::Real, 3>> coords;
  mesh.getCellCoordinates(cell, coords);
  if (coords.empty()) {
    return nullptr;
  }

  std::vector<svmp::FE::math::Vector<svmp::FE::Real, 3>> nodes;
  nodes.reserve(coords.size());
  for (const auto& coord : coords) {
    svmp::FE::math::Vector<svmp::FE::Real, 3> node{};
    node[0] = coord[0];
    node[1] = coord[1];
    node[2] = coord[2];
    nodes.push_back(node);
  }

  svmp::FE::geometry::MappingRequest map_request;
  map_request.element_type = mesh.getCellType(cell);
  map_request.geometry_order = mesh.getCellGeometryOrder(cell);
  map_request.use_affine = map_request.geometry_order <= 1;
  return svmp::FE::geometry::MappingFactory::create(map_request, nodes);
}

std::uint64_t levelSetMaintenanceAlgebraicRevision(
    std::span<const svmp::FE::Real> state) noexcept
{
  std::uint64_t revision = 1469598103934665603ull;
  constexpr std::uint64_t prime = 1099511628211ull;
  const auto mix = [&](std::uint64_t value) {
    for (std::size_t byte = 0u; byte < sizeof(value); ++byte) {
      revision ^= (value >> (byte * 8u)) & 0xffu;
      revision *= prime;
    }
  };
  mix(static_cast<std::uint64_t>(state.size()));
  for (const auto value : state) {
    std::uint64_t bits = 0u;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    mix(bits);
  }
  return revision == 0u ? 1u : revision;
}

std::uint64_t collectiveLevelSetMaintenanceAlgebraicRevision(
    std::span<const svmp::FE::Real> state,
    const svmp::MeshComm& comm)
{
  const auto revision = levelSetMaintenanceAlgebraicRevision(state);
  const auto [minimum, maximum] =
      globalMinMaxUint64(revision, comm);
  if (minimum != maximum) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set maintenance algebraic state revision differs across the FE communicator.");
  }
  return revision;
}

std::vector<svmp::FE::Real>
captureFeOrderedVectorCollectively(
    svmp::FE::backends::GenericVector& solution,
    const svmp::MeshComm& comm)
{
  return svmp::FE::systems::captureFeOrderedVectorCollectively(
      solution, feCollectiveCommunicator(comm));
}

std::vector<svmp::FE::Real> gatherFeOrderedSolution(
    svmp::FE::backends::GenericVector& solution,
    const svmp::MeshComm& comm)
{
  return captureFeOrderedVectorCollectively(solution, comm);
}

std::vector<svmp::FE::Real> gatherFeOrderedSolution(
    const svmp::FE::systems::SystemStateView& state,
    const svmp::MeshComm& comm)
{
  if (state.u_vector != nullptr) {
    auto* solution =
        const_cast<svmp::FE::backends::GenericVector*>(state.u_vector);
    return gatherFeOrderedSolution(*solution, comm);
  }
  if (!state.u.empty()) {
    return std::vector<svmp::FE::Real>(state.u.begin(), state.u.end());
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Could not collectively gather FE-ordered state values.");
}

std::string step_reject_reason_to_string(svmp::FE::timestepping::StepRejectReason r)
{
  using svmp::FE::timestepping::StepRejectReason;
  switch (r) {
    case StepRejectReason::NonlinearSolveFailed:
      return "NonlinearSolveFailed";
    case StepRejectReason::ErrorTooLarge:
      return "ErrorTooLarge";
    case StepRejectReason::CutTopologyChanged:
      return "CutTopologyChanged";
  }
  return std::to_string(static_cast<int>(r));
}

} // namespace svmp::Physics::formulations::navier_stokes::free_surface::detail
