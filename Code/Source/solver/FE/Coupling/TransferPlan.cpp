/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/TransferPlan.h"

namespace svmp {
namespace FE {
namespace coupling {

const char* toString(CouplingTransferKind kind) noexcept
{
    switch (kind) {
    case CouplingTransferKind::Unspecified:
        return "unspecified";
    case CouplingTransferKind::Identity:
        return "identity";
    case CouplingTransferKind::InterfacePointwiseInterpolation:
        return "interface_pointwise_interpolation";
    case CouplingTransferKind::InterfaceConservativeProjection:
        return "interface_conservative_projection";
    case CouplingTransferKind::InterfaceMortar:
        return "interface_mortar";
    case CouplingTransferKind::DriverOwned:
        return "driver_owned";
    }
    return "unknown";
}

const char* toString(CouplingInterfaceFramePolicy policy) noexcept
{
    switch (policy) {
    case CouplingInterfaceFramePolicy::None:
        return "none";
    case CouplingInterfaceFramePolicy::SourceToTargetVector:
        return "source_to_target_vector";
    case CouplingInterfaceFramePolicy::SourceToTargetRank2Tensor:
        return "source_to_target_rank2_tensor";
    }
    return "unknown";
}

bool isInterfaceTransferKind(CouplingTransferKind kind) noexcept
{
    return kind == CouplingTransferKind::InterfacePointwiseInterpolation ||
           kind == CouplingTransferKind::InterfaceConservativeProjection ||
           kind == CouplingTransferKind::InterfaceMortar;
}

} // namespace coupling
} // namespace FE
} // namespace svmp
