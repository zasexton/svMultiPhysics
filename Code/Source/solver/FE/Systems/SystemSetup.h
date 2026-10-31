/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_SYSTEMSETUP_H
#define SVMP_FE_SYSTEMS_SYSTEMSETUP_H

#include "Core/Types.h"
#include "Dofs/DofHandler.h"

#include <functional>
#include <memory>
#include <optional>
#include <vector>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Mesh/Core/MeshTypes.h"
#include "Mesh/Mesh.h"
#endif

namespace svmp {
namespace FE {
namespace backends {
struct BlockLayout;
} // namespace backends

namespace systems {

class FESystem;

/// What deriveBlockLayout forms beyond one block per unknown field.
struct BlockLayoutRequest {
    /// Merge the primary fields into one block and the constraint fields into
    /// another, when they are two contiguous runs, primary first, and one run
    /// has several fields (a two-way Schur-complement solve).
    bool split_by_role{false};
    /// Name the momentum and constraint blocks of a layout that is not merged:
    /// the first primary field and the first constraint field.
    bool name_saddle_pair{false};
};

/**
 * @brief The solver block layout of the unknowns of a set-up system
 *
 * One block per field of fieldMap(), in DOF order, named by the field, with
 * no role.  With split_by_role, a merged block of several fields takes the
 * first FieldSpec::solver_block its fields state, or else their names joined
 * by '+', and the role PrimaryField or ConstraintField; momentum_block and
 * constraint_block are 0 and 1.  A layout whose constraint fields do not
 * follow all its primary fields is not merged.  A field that states no
 * solver role is primary.
 */
[[nodiscard]] backends::BlockLayout deriveBlockLayout(const FESystem& system,
                                                      const BlockLayoutRequest& request);

struct SetupInputs {
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    std::shared_ptr<const svmp::Mesh> mesh{};
    svmp::Configuration coord_cfg{svmp::Configuration::Reference};
#endif

    std::optional<dofs::MeshTopologyInfo> topology_override{};

    std::function<std::optional<std::vector<GlobalIndex>>(int marker, FieldId field)>
        boundary_dofs_override{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_SYSTEMSETUP_H
