/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/Poisson/PoissonModule.h"

#include "Physics/Formulations/Poisson/PoissonBCFactories.h"

#include "FE/Forms/Vocabulary.h"
#include "FE/Forms/WeakForm.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"

#include <stdexcept>
#include <string>
#include <utility>

#ifndef SVMP_FE_ENABLE_LLVM_JIT
#define SVMP_FE_ENABLE_LLVM_JIT 0
#endif

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

    system.addOperator("equations");

    using namespace svmp::FE::forms;

    auto u = FormExpr::trialFunction(*space_, options_.field_name);
    auto v = FormExpr::testFunction(*space_, "v");

    const auto k = FormExpr::constant(options_.diffusion);
    const auto f = FormExpr::constant(options_.source);

    const auto integrand = k * inner(grad(u), grad(v)) - f * v;
    auto residual = integrand.dx();

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
        return Factories::toWindkesselBC(bc, u_id, *space_, options_.field_name);
    });

    bc_manager.validate();

    auto strong_constraints = bc_manager.getStrongConstraints(u_id);
    bc_manager.apply(system, residual, u, v, u_id);

    FE::forms::WeakForm weak_form;
    weak_form.residual = residual;
    weak_form.strong_constraints = std::move(strong_constraints);

    FE::systems::FormInstallOptions install_opts{};
#if SVMP_FE_ENABLE_LLVM_JIT
    // Enable LLVM JIT fast path when available. For nonlinear/coupled cases, we
    // also prefer symbolic tangents because the JIT backend does not accelerate
    // the dual-number (NonlinearFormKernel) path yet.
    install_opts.compiler_options.jit.enable = true;
    install_opts.compiler_options.use_symbolic_tangent = true;
#endif

    // Register the weak form under the unified "equations" operator tag.
    // The underlying kernel can produce either vector or matrix outputs
    // depending on the AssemblyRequest (and may auto-select an optimized
    // LinearFormKernel when the residual is affine in the TrialFunction).
    FE::systems::installWeakForm(system, {"equations"}, u_id, u_id, weak_form, install_opts);
}

} // namespace poisson
} // namespace formulations
} // namespace Physics
} // namespace svmp
