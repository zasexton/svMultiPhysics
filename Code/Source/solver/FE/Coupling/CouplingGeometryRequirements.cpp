/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingGeometryRequirements.h"

namespace svmp {
namespace FE {
namespace coupling {

const char* toString(CouplingGeometryTerminalQuantity quantity) noexcept
{
    switch (quantity) {
    case CouplingGeometryTerminalQuantity::MeshDisplacement:
        return "mesh_displacement";
    case CouplingGeometryTerminalQuantity::Coordinate:
        return "coordinate";
    case CouplingGeometryTerminalQuantity::ReferenceCoordinate:
        return "reference_coordinate";
    case CouplingGeometryTerminalQuantity::CurrentCoordinate:
        return "current_coordinate";
    case CouplingGeometryTerminalQuantity::PreviousCoordinate:
        return "previous_coordinate";
    case CouplingGeometryTerminalQuantity::ReferencePhysicalCoordinate:
        return "reference_physical_coordinate";
    case CouplingGeometryTerminalQuantity::Jacobian:
        return "jacobian";
    case CouplingGeometryTerminalQuantity::JacobianInverse:
        return "jacobian_inverse";
    case CouplingGeometryTerminalQuantity::JacobianDeterminant:
        return "jacobian_determinant";
    case CouplingGeometryTerminalQuantity::CurrentJacobian:
        return "current_jacobian";
    case CouplingGeometryTerminalQuantity::ReferenceJacobian:
        return "reference_jacobian";
    case CouplingGeometryTerminalQuantity::CurrentJacobianDeterminant:
        return "current_jacobian_determinant";
    case CouplingGeometryTerminalQuantity::ReferenceJacobianDeterminant:
        return "reference_jacobian_determinant";
    case CouplingGeometryTerminalQuantity::Normal:
        return "normal";
    case CouplingGeometryTerminalQuantity::CurrentNormal:
        return "current_normal";
    case CouplingGeometryTerminalQuantity::ReferenceNormal:
        return "reference_normal";
    case CouplingGeometryTerminalQuantity::CurrentMeasure:
        return "current_measure";
    case CouplingGeometryTerminalQuantity::ReferenceMeasure:
        return "reference_measure";
    case CouplingGeometryTerminalQuantity::SurfaceJacobian:
        return "surface_jacobian";
    case CouplingGeometryTerminalQuantity::CellDiameter:
        return "cell_diameter";
    case CouplingGeometryTerminalQuantity::CellVolume:
        return "cell_volume";
    case CouplingGeometryTerminalQuantity::FacetArea:
        return "facet_area";
    case CouplingGeometryTerminalQuantity::CellDomainId:
        return "cell_domain_id";
    }
    return "unknown";
}

} // namespace coupling
} // namespace FE
} // namespace svmp
