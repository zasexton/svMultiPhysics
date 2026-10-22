/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/NavierStokesFields.h"

#include "FE/Constraints/VertexDirichletConstraint.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Systems/FESystem.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

#include <cstddef>
#include <utility>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

[[nodiscard]] bool
isSupportedGeneratedBoundaryTraceSpace(
    const FE::spaces::FunctionSpace& space,
    int dimension) noexcept
{
    const auto element_type = space.element_type();
    return space.space_type() ==
               FE::spaces::SpaceType::Product &&
           space.continuity() == FE::Continuity::C0 &&
           space.polynomial_order() == 1 &&
           space.value_dimension() == dimension &&
           !space.element().basis().is_vector_valued() &&
           space.element().basis().size() ==
               static_cast<std::size_t>(dimension + 1) &&
           ((dimension == 2 &&
             element_type == FE::ElementType::Triangle3) ||
            (dimension == 3 &&
             element_type == FE::ElementType::Tetra4));
}

std::vector<FE::constraints::VertexDirichletValue> nodePressureValues(
    const IncompressibleNavierStokesVMSOptions& options)
{
    std::vector<FE::constraints::VertexDirichletValue> values;
    values.reserve(options.node_pressure_constraints.values.size());
    for (const auto& in : options.node_pressure_constraints.values) {
        values.push_back(FE::constraints::VertexDirichletValue{in.node_id, in.pressure});
    }
    return values;
}

FE::constraints::VertexIdMode nodePressureIdMode(
    const IncompressibleNavierStokesVMSOptions& options)
{
    FE::constraints::VertexIdMode mode = FE::constraints::VertexIdMode::GlobalVertexGid;
    switch (options.node_pressure_constraints.id_type) {
    case IncompressibleNavierStokesVMSOptions::NodePressureConstraintIdType::GlobalVertexGid:
        mode = FE::constraints::VertexIdMode::GlobalVertexGid;
        break;
    case IncompressibleNavierStokesVMSOptions::NodePressureConstraintIdType::LocalVertexId:
        mode = FE::constraints::VertexIdMode::LocalVertexId;
        break;
    }
    return mode;
}

NavierStokesFieldIds registerNavierStokesFields(
    FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options,
    NavierStokesPreflight& preflight)
{
    auto& u_spec = preflight.u_spec;
    auto& p_spec = preflight.p_spec;
    auto& body_force_spec = preflight.body_force_spec;
    auto& semi_implicit_reference_spec = preflight.semi_implicit_reference_spec;

    FE::systems::FieldBinding fields(
        system, "IncompressibleNavierStokesVMSModule::registerOn");
    const FE::FieldId u_id = fields.unknown(std::move(u_spec), "velocity");
    const FE::FieldId p_id = fields.unknown(std::move(p_spec), "pressure");

    FE::FieldId body_force_field_id = FE::INVALID_FIELD_ID;
    if (body_force_spec.has_value()) {
        body_force_field_id = fields.prescribed(
            std::move(*body_force_spec),
            options.auto_register_body_force_field,
            "momentum source");
    }
    if (semi_implicit_reference_spec.has_value()) {
        (void)fields.prescribed(
            std::move(*semi_implicit_reference_spec),
            /*auto_register=*/true,
            "semi-implicit reference velocity");
    }

    return NavierStokesFieldIds{
        .velocity = u_id,
        .pressure = p_id,
        .body_force = body_force_field_id,
    };
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
