/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_SURFACE_CONTACT_KERNEL_H
#define SVMP_FE_SYSTEMS_SURFACE_CONTACT_KERNEL_H

#include "Core/Types.h"
#include "Systems/GlobalKernel.h"

namespace svmp {
namespace FE {
namespace systems {

struct PenaltySurfaceContactConfig {
    FieldId field{INVALID_FIELD_ID};
    int slave_marker{0};
    int master_marker{0};

    // Maximum distance for closest-point search (must be > 0).
    Real search_radius{0.0};

    // Contact activates when the current distance is strictly less than this.
    Real activation_distance{0.0};

    // Penalty coefficient (traction stiffness).
    Real penalty{1.0};

    // Quadrature order used on slave boundary faces (<= 0 selects a recommended order).
    int quadrature_order{0};

    // Optional: track a per-quadrature-point "contact activation count" on slave faces.
    // This demonstrates history/state storage on boundary faces for global kernels.
    bool track_contact_count{false};

    // Upper bound used when allocating persistent boundary-face state storage.
    // Ignored unless track_contact_count is enabled (<= 0 defaults to a conservative value).
    LocalIndex max_state_qpts{0};
};

/**
 * @brief Simple surface-to-surface penalty contact global kernel
 *
 * Integrates a penalty traction over slave boundary faces, using a marker-filtered
 * closest-point projection onto the master boundary marker via `systems::ISearchAccess`.
 *
 * Notes:
 * - The specified field is interpreted as a displacement vector of length `mesh.dimension()`.
 * - The closest-point projection is performed on the mesh geometry provided to ISearchAccess
 *   (typically reference coordinates); deformation is approximated by interpolating the
 *   displacement field at the projected point.
 */
class PenaltySurfaceContactKernel final : public GlobalKernel {
public:
    explicit PenaltySurfaceContactKernel(PenaltySurfaceContactConfig cfg);

    [[nodiscard]] std::string name() const override { return "PenaltySurfaceContactKernel"; }

    [[nodiscard]] GlobalStateSpec globalStateSpec() const noexcept override;

    void addSparsityCouplings(const FESystem& system,
                              sparsity::SparsityPattern& pattern) const override;

    [[nodiscard]] assembly::AssemblyResult assemble(const FESystem& system,
                                                    const AssemblyRequest& request,
                                                    const SystemStateView& state,
                                                    assembly::GlobalSystemView* matrix_out,
                                                    assembly::GlobalSystemView* vector_out) override;

private:
    PenaltySurfaceContactConfig cfg_;
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_SURFACE_CONTACT_KERNEL_H
