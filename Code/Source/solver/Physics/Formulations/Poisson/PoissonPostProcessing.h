#ifndef SVMP_PHYSICS_FORMULATIONS_POISSON_POSTPROCESSING_H
#define SVMP_PHYSICS_FORMULATIONS_POISSON_POSTPROCESSING_H

/**
 * @file PoissonPostProcessing.h
 * @brief Formulation-local derived result registration helpers for Poisson/Darcy.
 */

#include "FE/Forms/Vocabulary.h"
#include "FE/PostProcessing/DerivedResultBuilder.h"
#include "FE/Systems/FESystem.h"
#include "Physics/Formulations/Poisson/PoissonModule.h"

namespace svmp {
namespace Physics {
namespace formulations {
namespace poisson {
namespace post {

inline void registerDarcyFlux(FE::systems::FESystem& system,
                              FE::FieldId pressure_id,
                              const FE::spaces::FunctionSpace& pressure_space,
                              const PoissonOptions& options)
{
    using namespace svmp::FE::forms;
    using namespace svmp::FE::post;

    auto p = StateField(pressure_id, pressure_space, options.field_name);
    auto K = FormExpr::constant(options.diffusion);
    auto flux = -K * grad(p);
    const int dim = pressure_space.topological_dimension();

    system.addDerivedResult(
        DerivedResultBuilder("Darcy_flux")
            .scope(DerivedResultScope::Cell)
            .policy(DerivedResultPolicy::CellAverage)
            .shape(FE::systems::FEQuantityShape::vector(dim))
            .expression(flux)
            .referencedField(pressure_id)
            .build());

    system.addDerivedResult(
        DerivedResultBuilder("Darcy_flux_node")
            .scope(DerivedResultScope::Vertex)
            .policy(DerivedResultPolicy::PatchAverage)
            .shape(FE::systems::FEQuantityShape::vector(dim))
            .expression(flux)
            .referencedField(pressure_id)
            .build());
}

inline void registerPostProcessing(FE::systems::FESystem& system,
                                   FE::FieldId field_id,
                                   const FE::spaces::FunctionSpace& space,
                                   const PoissonOptions& options)
{
    if (options.register_darcy_flux_output) {
        registerDarcyFlux(system, field_id, space, options);
    }
}

} // namespace post
} // namespace poisson
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_POISSON_POSTPROCESSING_H
