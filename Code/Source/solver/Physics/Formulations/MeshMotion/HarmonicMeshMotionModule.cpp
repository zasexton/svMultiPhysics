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

#include <algorithm>
#include <cmath>
#include <optional>
#include <set>
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
    if (!options_.normal_constraint.empty() &&
        !system.meshNormalBoundaryConstraintHistory().empty()) {
        throw std::invalid_argument(
            "HarmonicMeshMotionModule::registerOn: normal constraints "
            "cannot be installed after accepted normal-constraint history "
            "has begun");
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

    const auto bound_displacement = system.meshMotionField(
        FE::systems::MeshMotionFieldRole::Displacement);
    if (!options_.normal_constraint.empty() &&
        !bound_displacement.has_value() &&
        !options_.bind_as_mesh_displacement) {
        throw std::invalid_argument(
            "HarmonicMeshMotionModule::registerOn: normal constraints "
            "require the target field to be bound as mesh displacement");
    }
    auto existing_displacement = bound_displacement;
    if (!existing_displacement.has_value()) {
        const auto named = system.findFieldByName(options_.field_name);
        if (named != FE::INVALID_FIELD_ID) {
            existing_displacement = named;
        }
    }
    std::set<int> normal_markers;
    std::vector<FE::forms::FormExpr> normal_targets;
    normal_targets.reserve(options_.normal_constraint.size());
    for (const auto& bc : options_.normal_constraint) {
        const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(
            bc,
            "HarmonicMeshMotionModule::registerOn normal_constraint");
        if (!normal_markers.insert(marker).second) {
            throw std::invalid_argument(
                "HarmonicMeshMotionModule::registerOn: duplicate normal "
                "constraint marker " +
                std::to_string(marker));
        }
        if (bc.quantity != NormalConstraintQuantity::Displacement &&
            bc.quantity != NormalConstraintQuantity::Velocity) {
            throw std::invalid_argument(
                "HarmonicMeshMotionModule::registerOn: unknown normal "
                "constraint quantity");
        }
        if (!Factories::isFinitePositiveScalarLiteral(bc.penalty)) {
            throw std::invalid_argument(
                "HarmonicMeshMotionModule::registerOn: normal constraint "
                "penalty must be a finite positive literal");
        }
        if (bc.quantity == NormalConstraintQuantity::Velocity) {
            if (!Factories::isFinitePositiveScalarLiteral(
                    bc.velocity_time_scale)) {
                throw std::invalid_argument(
                    "HarmonicMeshMotionModule::registerOn: normal "
                    "constraint velocity time scale must be a finite "
                    "positive literal");
            }
        }
        auto target = Factories::effectiveNormalConstraintTarget(
            bc,
            dim,
            "HarmonicMeshMotionModule::registerOn normal_constraint");
        const auto* target_node = target.node();
        if (target_node == nullptr || target.hasTest() ||
            FE::forms::bc::detail::
                hasForbiddenPrescribedValueDependency(*target_node)) {
            throw std::invalid_argument(
                "HarmonicMeshMotionModule::registerOn: normal constraint "
                "target must be independent of FE state and variational "
                "geometry");
        }
        normal_targets.push_back(std::move(target));
        if (!existing_displacement.has_value()) {
            continue;
        }
        const auto existing = std::find_if(
            system.meshNormalBoundaryConstraints().begin(),
            system.meshNormalBoundaryConstraints().end(),
            [&](const auto& declaration) {
                return declaration.mesh_displacement_field ==
                           *existing_displacement &&
                       declaration.boundary_marker == marker;
            });
        if (existing != system.meshNormalBoundaryConstraints().end()) {
            throw FE::InvalidArgumentException(
                "HarmonicMeshMotionModule::registerOn: boundary " +
                std::to_string(marker) +
                " already has normal mesh-motion owner '" +
                existing->owner_component + "'");
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

    std::vector<FE::systems::MeshNormalBoundaryConstraintDeclaration>
        normal_declarations;
    normal_declarations.reserve(options_.normal_constraint.size());
    for (std::size_t index = 0u;
         index < options_.normal_constraint.size();
         ++index) {
        const auto& bc = options_.normal_constraint[index];
        const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(
            bc,
            "HarmonicMeshMotionModule::registerOn normal_constraint");
        normal_declarations.push_back(
            FE::systems::MeshNormalBoundaryConstraintDeclaration{
                .mesh_displacement_field = binding.displacement_field,
                .boundary_marker = marker,
                .quantity = FE::systems::MeshNormalBoundaryQuantity::
                    DisplacementTrace,
                .target_kind =
                    bc.quantity == NormalConstraintQuantity::Displacement
                        ? FE::systems::MeshNormalBoundaryTargetKind::
                              PrescribedDisplacement
                        : FE::systems::MeshNormalBoundaryTargetKind::
                              TimeScaledPrescribedVelocity,
                .target_expression = std::move(normal_targets[index]),
                .enforcement_kind =
                    FE::analysis::EnforcementKind::WeakPenalty,
                .related_velocity_field = FE::INVALID_FIELD_ID,
                .owner_component = "HarmonicMeshMotionModule",
            });
    }
    for (const auto& bc : options_.tangential_policy) {
        const auto system_policy = [&]() {
            switch (bc.policy) {
            case TangentialMeshPolicy::Free:
                return FE::systems::MeshTangentialBoundaryPolicy::Free;
            case TangentialMeshPolicy::SmoothingOnly:
                return FE::systems::MeshTangentialBoundaryPolicy::SmoothingOnly;
            case TangentialMeshPolicy::Prescribed:
                return FE::systems::MeshTangentialBoundaryPolicy::Prescribed;
            }
            throw std::invalid_argument(
                "HarmonicMeshMotionModule::registerOn: unknown tangential policy");
        }();
        system.declareMeshTangentialBoundaryPolicy(
            FE::systems::MeshTangentialBoundaryPolicyDeclaration{
                .mesh_displacement_field = binding.displacement_field,
                .boundary_marker = bc.boundary_marker,
                .policy = system_policy,
                .owner_component = "HarmonicMeshMotionModule",
            });
    }

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
            bc,
            dim,
            "HarmonicMeshMotionModule::registerOn normal_constraint");
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
    bc_manager.applyAll(
        system,
        residual,
        d_mesh,
        psi,
        d_id,
        options_.operator_tag);

    auto install = physicsInstallOptions(options_.jit_policy);
    install.compiler_options.geometry_tangent_path = options_.tangent_path;
    install.compiler_options.use_symbolic_tangent =
        options_.tangent_path != FE::forms::GeometryTangentPath::ADReference;

    (void)FE::systems::installFormulation(system, options_.operator_tag, {d_id}, residual, install);

    for (auto& declaration : normal_declarations) {
        const auto marker = declaration.boundary_marker;
        system.declareMeshNormalBoundaryConstraint(
            std::move(declaration));
        system.bindMeshNormalBoundaryConstraintConsumer(
            binding.displacement_field,
            marker,
            options_.operator_tag,
            "TraceRobinBC on marker " + std::to_string(marker));
    }
}

} // namespace mesh_motion
} // namespace formulations
} // namespace Physics
} // namespace svmp
