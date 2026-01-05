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
namespace systems {

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
