/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_SYSTEMCONSTRAINTS_H
#define SVMP_FE_SYSTEMS_SYSTEMCONSTRAINTS_H

#include "Core/Types.h"

#include "Constraints/Constraint.h"
#include "Constraints/DirichletBC.h"

#include "Dofs/DofHandler.h"

#include <memory>
#include <string_view>
#include <vector>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Mesh/Mesh.h"
#endif

namespace svmp {
namespace FE {
namespace systems {

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

std::vector<GlobalIndex> boundaryDofsByMarker(const svmp::Mesh& mesh,
                                              const dofs::DofHandler& dof_handler,
                                              int boundary_marker);

std::vector<GlobalIndex> boundaryDofsByFaceSet(const svmp::Mesh& mesh,
                                               const dofs::DofHandler& dof_handler,
                                               std::string_view face_set_name);

std::unique_ptr<constraints::Constraint>
makeDirichletConstantByMarker(const svmp::Mesh& mesh,
                              const dofs::DofHandler& dof_handler,
                              int boundary_marker,
                              double value,
                              const constraints::DirichletBCOptions& opts = {});

std::unique_ptr<constraints::Constraint>
makeDirichletConstantByFaceSet(const svmp::Mesh& mesh,
                               const dofs::DofHandler& dof_handler,
                               std::string_view face_set_name,
                               double value,
                               const constraints::DirichletBCOptions& opts = {});

#endif // SVMP_FE_WITH_MESH

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_SYSTEMCONSTRAINTS_H
