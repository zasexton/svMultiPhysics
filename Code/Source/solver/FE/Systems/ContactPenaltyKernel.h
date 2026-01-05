/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_CONTACT_PENALTY_KERNEL_H
#define SVMP_FE_SYSTEMS_CONTACT_PENALTY_KERNEL_H

#include "Core/Types.h"
#include "Systems/GlobalKernel.h"

namespace svmp {
namespace FE {
namespace systems {

struct PenaltyContactConfig {
    FieldId field{INVALID_FIELD_ID};
    int slave_marker{0};
    int master_marker{0};

    // Maximum distance for candidate search (must be > 0).
    Real search_radius{0.0};

    // Contact activates when the current distance is strictly less than this.
    Real activation_distance{0.0};

    // Penalty coefficient (spring stiffness).
    Real penalty{1.0};

    // For each slave vertex, couple only to the nearest master vertex.
    bool only_nearest{true};
};

/**
 * @brief Simple point-to-point penalty "contact" global kernel
 *
 * This kernel is intended as infrastructure for global, search-driven operator
 * terms. It couples vertices on a "slave" boundary marker to vertices on a
 * "master" boundary marker via a repulsive penalty when their distance is
 * below `activation_distance`.
 *
 * Notes:
 * - The kernel interprets the specified field's vertex DOFs as a displacement
 *   vector of length `mesh.dimension()`, applied to the mesh coordinates.
 * - Uses Systems::ISearchAccess for neighborhood queries (Mesh-backed by default).
 */
class PenaltyPointContactKernel final : public GlobalKernel {
public:
    explicit PenaltyPointContactKernel(PenaltyContactConfig cfg);

    [[nodiscard]] std::string name() const override { return "PenaltyPointContactKernel"; }

    void addSparsityCouplings(const FESystem& system,
                              sparsity::SparsityPattern& pattern) const override;

    [[nodiscard]] assembly::AssemblyResult assemble(const FESystem& system,
                                                    const AssemblyRequest& request,
                                                    const SystemStateView& state,
                                                    assembly::GlobalSystemView* matrix_out,
                                                    assembly::GlobalSystemView* vector_out) override;

private:
    PenaltyContactConfig cfg_;
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_CONTACT_PENALTY_KERNEL_H
