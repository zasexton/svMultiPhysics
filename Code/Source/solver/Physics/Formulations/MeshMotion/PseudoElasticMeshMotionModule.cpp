/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/MeshMotion/PseudoElasticMeshMotionModule.h"

#include "Physics/Formulations/MeshMotion/MeshMotionBoundaryPreparation.h"

#include "FE/Forms/Vocabulary.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormulationBuilder.h"
#include "FE/Systems/MeshDisplacementBinding.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace svmp {
namespace Physics {
namespace formulations {
namespace mesh_motion {

PseudoElasticMeshMotionModule::PseudoElasticMeshMotionModule(
    std::shared_ptr<const FE::spaces::FunctionSpace> displacement_space,
    PseudoElasticMeshMotionOptions options)
    : displacement_space_(std::move(displacement_space))
    , options_(std::move(options))
{
}

void PseudoElasticMeshMotionModule::registerOn(FE::systems::FESystem& system) const
{
    if (!displacement_space_) {
        throw std::invalid_argument(
            "PseudoElasticMeshMotionModule::registerOn: null displacement_space");
    }
    const int dim = displacement_space_->value_dimension();
    if (dim < 1 || dim > 3) {
        throw std::invalid_argument(
            "PseudoElasticMeshMotionModule::registerOn: displacement space must have 1..3 components");
    }
    if (displacement_space_->field_type() != FE::FieldType::Vector) {
        throw std::invalid_argument(
            "PseudoElasticMeshMotionModule::registerOn: displacement space must be vector-valued");
    }
    if (!options_.normal_constraint.empty() &&
        !system.meshNormalBoundaryConstraintHistory().empty()) {
        throw std::invalid_argument(
            "PseudoElasticMeshMotionModule::registerOn: normal constraints "
            "cannot be installed after accepted normal-constraint history "
            "has begun");
    }
    for (const auto& declaration : system.meshNormalBoundaryConstraints()) {
        if (declaration.requires_mesh_flux_consistency) {
            throw std::invalid_argument(
                "PseudoElasticMeshMotionModule::registerOn: boundary " +
                std::to_string(declaration.boundary_marker) +
                " requires the mesh-motion Nitsche consistency term of a "
                "consistent kinematic owner (fitted MeshNitsche); only the "
                "harmonic mesh motion provides it");
        }
    }

    const auto validate_positive_literal =
        [](const PseudoElasticMeshMotionOptions::ScalarValue& value, const char* name) {
            if (const auto* real = std::get_if<FE::Real>(&value)) {
                if (!(*real > 0.0)) {
                    throw std::invalid_argument(
                        std::string("PseudoElasticMeshMotionModule::registerOn: ") +
                        name + " must be positive");
                }
            }
        };
    validate_positive_literal(options_.lambda_mesh, "lambda_mesh");
    validate_positive_literal(options_.mu_mesh, "mu_mesh");

    auto normal_preparation = BoundaryPreparation::prepareNormalConstraints(
        system,
        options_.field_name,
        options_.bind_as_mesh_displacement,
        dim,
        options_.normal_constraint,
        "PseudoElasticMeshMotionModule::registerOn");

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
        "PseudoElasticMeshMotionModule",
        "PseudoElasticMeshMotionModule::registerOn");
    BoundaryPreparation::declareTangentialPolicies(
        system,
        binding.displacement_field,
        options_.tangential_policy,
        "PseudoElasticMeshMotionModule",
        "PseudoElasticMeshMotionModule::registerOn");

    FE::systems::FormulationBuilder formulation(system, options_.operator_tag, {});

    using namespace svmp::FE::forms;
    const auto d_id = binding.displacement_field;
    const auto& V = *binding.space;
    const auto d_mesh = StateField(d_id, V, "d_mesh");
    const auto psi = TestField(d_id, V, "psi");
    const auto lambda_mesh =
        FE::forms::bc::toScalarExpr(options_.lambda_mesh, "mesh_motion_lambda");
    const auto mu_mesh =
        FE::forms::bc::toScalarExpr(options_.mu_mesh, "mesh_motion_mu");

    const auto eps_d = sym(grad(d_mesh));
    const auto eps_psi = sym(grad(psi));
    const auto I = FormExpr::identity(dim);
    const auto sigma_mesh =
        FormExpr::constant(2.0) * mu_mesh * eps_d +
        lambda_mesh * trace(eps_d) * I;

    auto residual = inner(sigma_mesh, eps_psi).dx();

    FE::systems::BoundaryConditionManager bc_manager;
    BoundaryPreparation::populateBoundaryConditions(
        bc_manager, options_, dim, "PseudoElasticMeshMotionModule::registerOn");
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
