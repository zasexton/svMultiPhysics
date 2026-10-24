/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_VERTEX_FIELD_INITIALIZATION_H
#define SVMP_FE_SYSTEMS_VERTEX_FIELD_INITIALIZATION_H

/**
 * @file VertexFieldInitialization.h
 * @brief Initial values of an FE field from the same-named vertex field of
 *        the system's native mesh.
 */

#include "Core/Types.h"

#include <cstddef>
#include <string>

namespace svmp {
namespace FE {
namespace backends {
class GenericVector;
}
namespace systems {

class FESystem;

/**
 * @brief Set the coefficients of @p field_id in @p u0 from the vertex mesh
 *        field of the same name (FESystem::projectMeshVertexValuesToFieldCoefficients).
 *
 * Nothing happens (0 is returned) without a field id, a native mesh, or a
 * vertex mesh field of the field's name.  Otherwise the mesh field must be
 * Float64 with at least the field's components and an entry per mesh vertex,
 * the field must have vertex DOF metadata, and every coefficient must be
 * assigned; a violation throws std::runtime_error with a message that starts
 * with @p message_prefix.
 *
 * @return the number of coefficients set
 */
std::size_t initializeFieldFromMeshVertexField(
    const FESystem& system,
    backends::GenericVector& u0,
    FieldId field_id,
    const std::string& message_prefix);

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_VERTEX_FIELD_INITIALIZATION_H
