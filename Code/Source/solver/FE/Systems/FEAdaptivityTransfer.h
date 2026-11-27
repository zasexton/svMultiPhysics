/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_FEADAPTIVITYTRANSFER_H
#define SVMP_FE_SYSTEMS_FEADAPTIVITYTRANSFER_H

#include "Core/FEConfig.h"
#include "Core/Types.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Mesh/Adaptivity/RefinementDelta.h"
#include "Mesh/Core/MeshBase.h"
#endif

namespace svmp {
namespace FE {
namespace systems {

class CollectiveCommunicator;

enum class FEFieldTransferMethod : std::uint8_t {
    Interpolate,
    Conservative
};

struct FEFieldTransferOptions {
    FEFieldTransferMethod method{FEFieldTransferMethod::Interpolate};
    bool require_all_vertices{true};
    Real conservation_tolerance{1.0e-12};
};

struct FEFieldTransferDiagnostics {
    bool success{true};
    std::size_t values_transferred{0};
    Real conservation_error{0.0};
    std::vector<std::string> diagnostics{};
};

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
/**
 * @brief Transfer entity-major nodal field values across a Mesh refinement delta.
 *
 * The arrays hold components per vertex in mesh order. Conservative transfer
 * preserves each component's finite-element integral on the physical mesh.
 */
FEFieldTransferDiagnostics transferNodalFieldByVertexProvenance(
    const svmp::MeshBase& old_mesh,
    const svmp::MeshBase& new_mesh,
    const svmp::RefinementDelta& delta,
    int components,
    std::span<const Real> old_vertex_values,
    std::span<Real> new_vertex_values,
    const FEFieldTransferOptions& options = {});

/// Distributed transfer counts each global cell once, including on overlapping
/// partitions. By default each mesh supplies its own active geometry.
FEFieldTransferDiagnostics transferNodalFieldByVertexProvenance(
    const svmp::MeshBase& old_mesh, const svmp::MeshBase& new_mesh,
    const svmp::RefinementDelta& delta, int components,
    std::span<const Real> old_vertex_values, std::span<Real> new_vertex_values,
    const FEFieldTransferOptions& options, const CollectiveCommunicator& comm,
    std::optional<svmp::Configuration> configuration = {});
#endif

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FEADAPTIVITYTRANSFER_H
