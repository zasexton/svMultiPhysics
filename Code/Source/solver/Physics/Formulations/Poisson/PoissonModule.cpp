/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/Poisson/PoissonModule.h"

#include "Physics/Formulations/Poisson/PoissonBCFactories.h"
#include "Physics/Formulations/Poisson/PoissonPostProcessing.h"

#include "FE/Constraints/VertexDirichletConstraint.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace svmp {
namespace Physics {
namespace formulations {
namespace poisson {

PoissonModule::PoissonModule(std::shared_ptr<const FE::spaces::FunctionSpace> space,
                             PoissonOptions options)
    : space_(std::move(space))
    , options_(std::move(options))
{
}

void PoissonModule::registerOn(FE::systems::FESystem& system) const
{
    if (!space_) {
        throw std::invalid_argument("PoissonModule::registerOn: null space");
    }

    FE::systems::FieldSpec spec;
    spec.name = options_.field_name;
    spec.space = space_;
    const FE::FieldId u_id = system.addField(std::move(spec));
    post::registerPostProcessing(system, u_id, *space_, options_);

    system.addOperator("equations");

    if (!options_.node_dirichlet.values.empty()) {
        std::vector<FE::constraints::VertexDirichletValue> values;
        values.reserve(options_.node_dirichlet.values.size());
        for (const auto& in : options_.node_dirichlet.values) {
            values.push_back(FE::constraints::VertexDirichletValue{in.node_id, in.value});
        }

        FE::constraints::VertexIdMode mode = FE::constraints::VertexIdMode::GlobalVertexGid;
        switch (options_.node_dirichlet.id_type) {
        case NodeIdType::GlobalVertexGid:
            mode = FE::constraints::VertexIdMode::GlobalVertexGid;
            break;
        default:
            throw std::invalid_argument("PoissonModule::registerOn: unsupported node Dirichlet id type");
        }

        system.addSystemConstraint(
            std::make_unique<FE::constraints::VertexDirichletConstraint>(u_id, std::move(values), mode));
    }

    using namespace svmp::FE::forms;

    auto u = StateField(u_id, *space_, options_.field_name);
    auto v = TestField(u_id, *space_, "v");

    const auto k = FormExpr::constant(options_.diffusion);
    const auto f = FormExpr::constant(options_.source);

    const auto integrand = k * inner(grad(u), grad(v)) - f * v;
    auto residual = integrand.dx();

    if (!options_.coupled_neumann_rcr.empty()) {
        setBoundaryReductionCompilerOptions(system, u_id, options_.jit_policy);
    }

    FE::systems::BoundaryConditionManager bc_manager;
    bc_manager.install(options_.neumann, Factories::toNaturalBC);
    bc_manager.install(options_.robin, Factories::toRobinBC);
    bc_manager.install(options_.dirichlet, [&](const auto& bc) { return Factories::toEssentialBC(bc, options_.field_name); });
    bc_manager.install(options_.dirichlet_weak, [&](const auto& bc) {
        return Factories::toNitscheBC(bc,
                                      k,
                                      options_.nitsche_gamma,
                                      options_.nitsche_symmetric,
                                      options_.nitsche_scale_with_p);
    });
    bc_manager.install(options_.coupled_neumann_rcr, [&](const auto& bc) {
        return Factories::toWindkesselBC(bc, system, u);
    });

    bc_manager.applyAll(system, residual, u, v, u_id);

    auto install_opts = physicsInstallOptions(options_.jit_policy);
    install_opts.compiler_options.use_symbolic_tangent = true;

    (void)FE::systems::installFormulation(system, "equations", {u_id}, residual, install_opts);
}

} // namespace poisson
} // namespace formulations
} // namespace Physics
} // namespace svmp
