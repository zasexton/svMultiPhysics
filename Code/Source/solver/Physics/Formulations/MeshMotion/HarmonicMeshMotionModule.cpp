/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/MeshMotion/HarmonicMeshMotionModule.h"

#include "Physics/Formulations/MeshMotion/MeshMotionBoundaryPreparation.h"

#include "FE/Forms/Vocabulary.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormulationBuilder.h"
#include "FE/Systems/MeshBoundaryKinematics.h"
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

    auto normal_preparation = BoundaryPreparation::prepareNormalConstraints(
        system,
        options_.field_name,
        options_.bind_as_mesh_displacement,
        dim,
        options_.normal_constraint,
        "HarmonicMeshMotionModule::registerOn");

    // The normal-rate relations on the target displacement (fitted MeshNitsche
    // free surfaces, MeshBoundaryKinematics.h): validate before any system
    // mutation.  They constrain the normal mesh velocity, so the operator must
    // act on the velocity as well (a displacement operator would leave a
    // per-step P1 flux defect that accumulates), and the combined boundary row
    // must be coercive, gamma_N > 2 kappa.
    FE::systems::requireNormalRateRelationsConsumable(
        system,
        FE::systems::NormalRateRelationConsumer{
            .field = normal_preparation.candidate_displacement,
            .own_normal_markers = normal_preparation.markers,
            .flux_of_rate = options_.quantity == HarmonicQuantity::Velocity,
            .literal_kappa = literal_real(effective_kappa),
            .context = "HarmonicMeshMotionModule::registerOn",
            .rate_flux_hint = "set the harmonic quantity to velocity (Harmonic_quantity=velocity)",
        });

    const auto binding = FE::systems::resolveMeshDisplacementBinding(
        system,
        FE::systems::MeshDisplacementBindingOptions{
            true,
            dim,
            options_.field_name,
            displacement_space_,
            options_.auto_register_field,
            options_.bind_as_mesh_displacement});

    auto normal_declarations = BoundaryPreparation::pendingNormalDeclarations(
        binding.displacement_field,
        options_.normal_constraint,
        normal_preparation,
        "HarmonicMeshMotionModule",
        "HarmonicMeshMotionModule::registerOn");
    BoundaryPreparation::declareTangentialPolicies(
        system,
        binding.displacement_field,
        options_.tangential_policy,
        "HarmonicMeshMotionModule",
        "HarmonicMeshMotionModule::registerOn");

    FE::systems::FormulationBuilder formulation(system, options_.operator_tag, {});

    using namespace svmp::FE::forms;
    const auto d_id = binding.displacement_field;
    const auto& V = *binding.space;
    const auto d_mesh = StateField(d_id, V, "d_mesh");
    const auto psi = TestField(d_id, V, "psi");
    const auto kappa = FE::forms::bc::toScalarExpr(effective_kappa, "mesh_motion_kappa");
    // The harmonic operator acts on the displacement or on the mesh velocity.
    const auto q_mesh =
        options_.quantity == HarmonicQuantity::Velocity ? dt(d_mesh) : d_mesh;
    auto residual = (kappa * inner(grad(q_mesh), grad(psi))).dx();

    // Nitsche consistency on the boundaries of the normal-rate relations: the
    // relation's row penalizes the normal mismatch; removing this operator's
    // own normal flux kappa ((grad q) n . n) there makes the boundary equation
    // reduce to the relation, so the enforcement error does not depend on the
    // penalty.  The tangential flux stays natural (the Free tangential policy).
    residual = FE::systems::addNormalRateRelationConsistency(
        system, binding.displacement_field, residual, kappa, grad(q_mesh), psi);

    FE::systems::BoundaryConditionManager bc_manager;
    BoundaryPreparation::populateBoundaryConditions(
        bc_manager, options_, dim, "HarmonicMeshMotionModule::registerOn");
    bc_manager.applyAll(
        system,
        residual,
        d_mesh,
        psi,
        d_id,
        options_.operator_tag);

    formulation.options() = physicsInstallOptions(options_.jit_policy);
    formulation.options().compiler_options.geometry_tangent_path = options_.tangent_path;
    formulation.symbolicTangent(
        options_.tangent_path != FE::forms::GeometryTangentPath::ADReference);

    (void)formulation.install({d_id}, residual);

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
