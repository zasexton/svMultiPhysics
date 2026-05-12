/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/MeshMotion/HarmonicMeshMotionModule.h"

#include "FE/Forms/StandardBCs.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/MeshDisplacementBinding.h"

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace mesh_motion {

namespace {

[[nodiscard]] std::vector<FE::forms::FormExpr>
vectorComponents(const std::array<HarmonicMeshMotionOptions::ScalarValue, 3>& value,
                 int dim,
                 const std::string& name_prefix,
                 int marker)
{
    std::vector<FE::forms::FormExpr> values;
    values.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        values.push_back(FE::forms::bc::toScalarExpr(
            value[static_cast<std::size_t>(d)],
            name_prefix + "_" + std::to_string(marker) + "_c" + std::to_string(d)));
    }
    return values;
}

[[nodiscard]] FE::forms::FormExpr
vectorValue(const std::array<HarmonicMeshMotionOptions::ScalarValue, 3>& value,
            int dim,
            const std::string& name_prefix,
            int marker)
{
    return FE::forms::FormExpr::asVector(vectorComponents(value, dim, name_prefix, marker));
}

[[nodiscard]] std::optional<FE::Real>
literalReal(const HarmonicMeshMotionOptions::ScalarValue& value)
{
    if (const auto* real = std::get_if<FE::Real>(&value)) {
        return *real;
    }
    return std::nullopt;
}

void validatePositiveLiteral(const HarmonicMeshMotionOptions::ScalarValue& value,
                             const char* name)
{
    if (const auto real = literalReal(value)) {
        if (!(*real > 0.0)) {
            throw std::invalid_argument(
                std::string("HarmonicMeshMotionModule::registerOn: ") +
                name + " must be positive");
        }
    }
}

[[nodiscard]] HarmonicMeshMotionOptions::ScalarValue
effectiveKappa(const HarmonicMeshMotionOptions& options)
{
    validatePositiveLiteral(options.kappa, "kappa");
    if (!options.stiffness) {
        return options.kappa;
    }

    validatePositiveLiteral(*options.stiffness, "stiffness");

    const auto kappa_literal = literalReal(options.kappa);
    const auto stiffness_literal = literalReal(*options.stiffness);
    if (kappa_literal && stiffness_literal) {
        constexpr FE::Real default_kappa = 1.0;
        if (std::abs(*kappa_literal - default_kappa) <= 0.0) {
            return *options.stiffness;
        }
        if (std::abs(*kappa_literal - *stiffness_literal) <= 0.0) {
            return options.kappa;
        }
    }

    throw std::invalid_argument(
        "HarmonicMeshMotionModule::registerOn: both kappa and deprecated "
        "stiffness were set; use kappa only or set matching literal values");
}

} // namespace

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
    const auto binding = FE::systems::resolveMeshDisplacementBinding(
        system,
        FE::systems::MeshDisplacementBindingOptions{
            true,
            dim,
            options_.field_name,
            displacement_space_,
            options_.auto_register_field,
            options_.bind_as_mesh_displacement});

    system.addOperator(options_.operator_tag);

    using namespace svmp::FE::forms;
    const auto d_id = binding.displacement_field;
    const auto& V = *binding.space;
    const auto d_mesh = StateField(d_id, V, "d_mesh");
    const auto psi = TestField(d_id, V, "psi");
    const auto kappa = FE::forms::bc::toScalarExpr(effectiveKappa(options_), "mesh_motion_kappa");
    auto residual = (kappa * inner(grad(d_mesh), grad(psi))).dx();

    FE::systems::BoundaryConditionManager bc_manager;
    for (const auto& bc : options_.natural) {
        const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(
            bc, "HarmonicMeshMotionModule::registerOn natural");
        bc_manager.add(std::make_unique<FE::forms::bc::ReservedBC>(marker));

        const auto g_mesh = vectorValue(bc.value, dim, "mesh_motion_natural", marker);
        residual = residual + (FormExpr::constant(-1.0) * inner(g_mesh, psi)).ds(marker);
    }
    for (const auto& bc : options_.robin) {
        const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(
            bc, "HarmonicMeshMotionModule::registerOn robin");
        bc_manager.add(std::make_unique<FE::forms::bc::ReservedBC>(marker));

        const auto alpha = FE::forms::bc::toScalarExpr(
            bc.alpha, "mesh_motion_robin_alpha_" + std::to_string(marker));
        const auto d_target = vectorValue(bc.target, dim, "mesh_motion_robin_target", marker);
        residual = residual + (alpha * inner(d_mesh - d_target, psi)).ds(marker);
    }
    bc_manager.install(options_.dirichlet, [&](const auto& bc) {
        const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(
            bc, "HarmonicMeshMotionModule::registerOn dirichlet");
        return std::make_unique<FE::forms::bc::EssentialBC>(
            marker,
            vectorComponents(bc.value, dim, "mesh_displacement", marker),
            "d_mesh");
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
