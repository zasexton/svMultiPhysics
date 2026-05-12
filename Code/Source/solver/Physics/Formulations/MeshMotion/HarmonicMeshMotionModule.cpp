/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/MeshMotion/HarmonicMeshMotionModule.h"

#include "Physics/Formulations/MeshMotion/MeshMotionBCFactories.h"

#include "FE/Forms/Vocabulary.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/MeshDisplacementBinding.h"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace svmp {
namespace Physics {
namespace formulations {
namespace mesh_motion {

HarmonicMeshMotionModule::HarmonicMeshMotionModule(
    std::shared_ptr<const FE::spaces::FunctionSpace> displacement_space,
    HarmonicMeshMotionOptions options)
    : displacement_space_(std::move(displacement_space))
    , options_(std::move(options))
{
}

void HarmonicMeshMotionModule::registerOn(FE::systems::FESystem& system) const
{
    if (!displacement_space_) {
        throw std::invalid_argument("HarmonicMeshMotionModule::registerOn: null displacement_space");
    }
    const int dim = displacement_space_->value_dimension();
    if (dim < 1 || dim > 3) {
        throw std::invalid_argument(
            "HarmonicMeshMotionModule::registerOn: displacement space must have 1..3 components");
    }
    if (displacement_space_->field_type() != FE::FieldType::Vector) {
        throw std::invalid_argument(
            "HarmonicMeshMotionModule::registerOn: displacement space must be vector-valued");
    }

    const auto literal_real =
        [](const HarmonicMeshMotionOptions::ScalarValue& value) -> std::optional<FE::Real> {
            if (const auto* real = std::get_if<FE::Real>(&value)) {
                return *real;
            }
            return std::nullopt;
        };
    const auto validate_positive_literal =
        [&](const HarmonicMeshMotionOptions::ScalarValue& value, const char* name) {
            if (const auto real = literal_real(value)) {
                if (!(*real > 0.0)) {
                    throw std::invalid_argument(
                        std::string("HarmonicMeshMotionModule::registerOn: ") +
                        name + " must be positive");
                }
            }
        };

    validate_positive_literal(options_.kappa, "kappa");
    HarmonicMeshMotionOptions::ScalarValue effective_kappa = options_.kappa;
    if (options_.stiffness) {
        validate_positive_literal(*options_.stiffness, "stiffness");

        const auto kappa_literal = literal_real(options_.kappa);
        const auto stiffness_literal = literal_real(*options_.stiffness);
        if (kappa_literal && stiffness_literal) {
            constexpr FE::Real default_kappa = 1.0;
            if (std::abs(*kappa_literal - default_kappa) <= 0.0) {
                effective_kappa = *options_.stiffness;
            } else if (std::abs(*kappa_literal - *stiffness_literal) <= 0.0) {
                effective_kappa = options_.kappa;
            } else {
                throw std::invalid_argument(
                    "HarmonicMeshMotionModule::registerOn: both kappa and deprecated "
                    "stiffness were set; use kappa only or set matching literal values");
            }
        } else {
            throw std::invalid_argument(
                "HarmonicMeshMotionModule::registerOn: both kappa and deprecated "
                "stiffness were set; use kappa only or set matching literal values");
        }
    }

    const auto binding = FE::systems::resolveMeshDisplacementBinding(
        system,
        FE::systems::MeshDisplacementBindingOptions{
            true,
            dim,
            options_.field_name,
            displacement_space_,
            options_.auto_register_field,
            options_.bind_as_mesh_displacement});

    if (!system.hasOperator(options_.operator_tag)) {
        system.addOperator(options_.operator_tag);
    }

    using namespace svmp::FE::forms;
    const auto d_id = binding.displacement_field;
    const auto& V = *binding.space;
    const auto d_mesh = StateField(d_id, V, "d_mesh");
    const auto psi = TestField(d_id, V, "psi");
    const auto kappa = FE::forms::bc::toScalarExpr(effective_kappa, "mesh_motion_kappa");
    auto residual = (kappa * inner(grad(d_mesh), grad(psi))).dx();

    FE::systems::BoundaryConditionManager bc_manager;
    bc_manager.install(options_.natural, [&](const auto& bc) {
        return Factories::toVectorNaturalBC(
            bc, dim, "HarmonicMeshMotionModule::registerOn natural");
    });
    bc_manager.install(options_.robin, [&](const auto& bc) {
        return Factories::toVectorRobinBC(
            bc, dim, "HarmonicMeshMotionModule::registerOn robin");
    });
    bc_manager.install(options_.normal_constraint, [&](const auto& bc) {
        return Factories::toNormalConstraintBC(
            bc, "HarmonicMeshMotionModule::registerOn normal_constraint");
    });
    for (const auto& bc : options_.tangential_policy) {
        switch (bc.policy) {
        case TangentialMeshPolicy::Free:
        case TangentialMeshPolicy::SmoothingOnly:
            break;
        case TangentialMeshPolicy::Prescribed:
            bc_manager.add(Factories::toTangentialConstraintBC(
                bc, dim, "HarmonicMeshMotionModule::registerOn tangential_policy"));
            break;
        }
    }
    bc_manager.install(options_.dirichlet, [&](const auto& bc) {
        return Factories::toVectorEssentialBC(
            bc, dim, "HarmonicMeshMotionModule::registerOn dirichlet", "d_mesh");
    });
    bc_manager.applyAll(system, residual, d_mesh, psi, d_id);

    auto install = physicsInstallOptions(options_.jit_policy);
    install.compiler_options.geometry_tangent_path = options_.tangent_path;
    install.compiler_options.use_symbolic_tangent =
        options_.tangent_path != FE::forms::GeometryTangentPath::ADReference;

    (void)FE::systems::installFormulation(system, options_.operator_tag, {d_id}, residual, install);
}

} // namespace mesh_motion
} // namespace formulations
} // namespace Physics
} // namespace svmp
