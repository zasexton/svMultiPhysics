/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#pragma once

/**
 * @file
 * @brief Support shared by the free-surface record and stage code.
 *
 * FreeSurfaceSystemHandle names the FE system of a run.  The functions in
 * namespace detail are copies of the Application driver's communicator,
 * revision-hash and cell-geometry helpers with unchanged bodies, so the
 * moved free-surface code issues the same collectives in the same order and
 * computes the same revisions.  The driver keeps the originals for its own
 * code; both go once FE provides these helpers.
 */

#include "FE/Core/Types.h"
#include "FE/Systems/SystemCollectives.h"
#include "FE/TimeStepping/StepController.h"
#include "Mesh/Core/MeshComm.h"

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp::FE {
namespace assembly {
class IMeshAccess;
}
namespace geometry {
class GeometryMapping;
}
namespace backends {
class GenericVector;
}
namespace systems {
class FESystem;
struct SystemStateView;
}
} // namespace svmp::FE

namespace svmp::Physics::formulations::navier_stokes::free_surface {

/**
 * @brief The FE system of a free-surface run.
 *
 * Refers to the owning pointer of the run's FE system and converts
 * implicitly from any object with an fe_system member of that type (the
 * Application's simulation components), so callers pass their components
 * as before.  Valid while the referenced pointer is.
 */
struct FreeSurfaceSystemHandle {
  template <class Components>
    requires std::convertible_to<
        decltype((std::declval<Components&>().fe_system)),
        const std::unique_ptr<svmp::FE::systems::FESystem>&>
  FreeSurfaceSystemHandle(Components& components) noexcept
      : fe_system(components.fe_system)
  {
  }

  explicit FreeSurfaceSystemHandle(
      const std::unique_ptr<svmp::FE::systems::FESystem>& system) noexcept
      : fe_system(system)
  {
  }

  const std::unique_ptr<svmp::FE::systems::FESystem>& fe_system;
};

namespace detail {

svmp::MeshComm activeFESystemCommunicator(
    const svmp::FE::systems::FESystem& system);

svmp::FE::systems::CollectiveCommunicator feCollectiveCommunicator(
    const svmp::MeshComm& comm);

double globalSumDouble(double local, const svmp::MeshComm& comm);

double globalMinDouble(double local, const svmp::MeshComm& comm);

double globalMaxDouble(double local, const svmp::MeshComm& comm);

std::pair<std::uint64_t, std::uint64_t> globalMinMaxUint64(
    std::uint64_t local,
    const svmp::MeshComm& comm);

std::size_t globalSumSize(std::size_t local, const svmp::MeshComm& comm);

bool globalAnyBool(bool local, const svmp::MeshComm& comm);

void requireCollectivePhasePreparation(
    const std::exception_ptr& local_failure,
    const svmp::MeshComm& comm,
    std::string_view phase);

constexpr std::uint64_t kCutContextHashOffset = 1469598103934665603ull;

constexpr std::uint64_t kCutContextHashPrime = 1099511628211ull;

inline void mixCutContextHash(std::uint64_t& h, std::uint64_t value) noexcept
{
  h ^= value;
  h *= kCutContextHashPrime;
}

inline void mixCutContextHash(std::uint64_t& h, const std::string& value) noexcept
{
  for (const char c : value) {
    mixCutContextHash(h, static_cast<unsigned char>(c));
  }
  mixCutContextHash(h, 0xffu);
}

inline std::uint64_t avalancheCutTopologyFingerprint(
    std::uint64_t value) noexcept
{
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

struct CutTopologyFingerprintMultiset {
  // This is a bounded probabilistic fingerprint, not an exact set-equality
  // certificate.  Count plus two independently mixed sums and xors makes the
  // reduction insensitive to rule/rank order while retaining multiplicity.
  std::array<std::uint64_t, 3> sums{{0u, 0u, 0u}};
  std::array<std::uint64_t, 2> xors{{0u, 0u}};

  void insert(std::uint64_t fingerprint) noexcept
  {
    fingerprint = fingerprint == 0u ? 1u : fingerprint;
    const auto mixed = avalancheCutTopologyFingerprint(fingerprint);
    ++sums[0];
    sums[1] += fingerprint;
    sums[2] += mixed;
    xors[0] ^= fingerprint;
    xors[1] ^= mixed;
  }

  void merge(const CutTopologyFingerprintMultiset& other) noexcept
  {
    for (std::size_t i = 0u; i < sums.size(); ++i) {
      sums[i] += other.sums[i];
    }
    for (std::size_t i = 0u; i < xors.size(); ++i) {
      xors[i] ^= other.xors[i];
    }
  }
};

CutTopologyFingerprintMultiset collectiveCutTopologyFingerprintMultiset(
    const CutTopologyFingerprintMultiset& local,
    const svmp::MeshComm& comm);

std::shared_ptr<svmp::FE::geometry::GeometryMapping> createCellGeometryMapping(
    const svmp::FE::assembly::IMeshAccess& mesh,
    svmp::FE::GlobalIndex cell);

std::uint64_t levelSetMaintenanceAlgebraicRevision(
    std::span<const svmp::FE::Real> state) noexcept;

std::uint64_t collectiveLevelSetMaintenanceAlgebraicRevision(
    std::span<const svmp::FE::Real> state,
    const svmp::MeshComm& comm);

std::vector<svmp::FE::Real>
captureFeOrderedVectorCollectively(
    svmp::FE::backends::GenericVector& solution,
    const svmp::MeshComm& comm);

std::vector<svmp::FE::Real> gatherFeOrderedSolution(
    svmp::FE::backends::GenericVector& solution,
    const svmp::MeshComm& comm);

std::vector<svmp::FE::Real> gatherFeOrderedSolution(
    const svmp::FE::systems::SystemStateView& state,
    const svmp::MeshComm& comm);

std::string step_reject_reason_to_string(svmp::FE::timestepping::StepRejectReason r);

} // namespace detail

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
