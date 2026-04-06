#ifndef SVMP_FE_AUXILIARY_OPERATOR_BUILDER_H
#define SVMP_FE_AUXILIARY_OPERATOR_BUILDER_H

/**
 * @file AuxiliaryOperatorBuilder.h
 * @brief Declarative builder for auxiliary operators (nonlocal couplings).
 *
 * `AuxiliaryOperator` is the advanced public API for genuinely nonlocal
 * coupling graphs and custom mixed sparse operators.  Local or per-entity
 * monolithic models stay on the `AuxiliaryModel` + `use(model)` workflow;
 * `AuxiliaryOperator` is for cases that cannot be expressed as local
 * per-entity residuals.
 *
 * ## Canonical builder surface
 *
 * ```cpp
 * auto op = AuxiliaryOperatorBuilder("my_coupling")
 *     .source("aux_block_A")
 *     .target("aux_block_B")
 *     .topology(AuxiliaryCouplingTopology::Sparse)
 *     .residual(my_residual_callback)
 *     .jacobian(my_jacobian_callback)
 *     .mass(my_mass_callback)       // optional
 *     .transfer(my_transfer_cb)     // optional
 *     .derivatives(policy)          // optional
 *     .build();
 * ```
 *
 * ## Operator types
 *
 * | Type               | Source        | Target        |
 * |--------------------|--------------|---------------|
 * | AuxSelf            | aux block    | same block    |
 * | AuxToAux           | aux block    | aux block     |
 * | FieldToAux         | FE field     | aux block     |
 * | AuxToField         | aux block    | FE field      |
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Auxiliary/AuxiliaryStateTypes.h"
#include "Auxiliary/AuxiliaryCouplingGraph.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  Coupling topology
// ---------------------------------------------------------------------------

enum class AuxiliaryCouplingTopology : std::uint8_t {
    /// Dense coupling (every source entity couples to every target entity).
    Dense,

    /// Sparse coupling with explicit connectivity.
    Sparse,

    /// Point-to-point coupling (1-to-1 entity mapping).
    PointToPoint,

    /// Reduction coupling (many-to-one, e.g., boundary → global).
    Reduction
};

// ---------------------------------------------------------------------------
//  Operator callbacks
// ---------------------------------------------------------------------------

/**
 * @brief Context for operator residual/Jacobian evaluation.
 */
struct AuxiliaryOperatorContext {
    Real time{0.0};
    Real dt{0.0};

    /// Source block data.
    std::span<const Real> source_data{};
    std::size_t source_entity_count{0};
    int source_stride{0};

    /// Target block data.
    std::span<const Real> target_data{};
    std::size_t target_entity_count{0};
    int target_stride{0};

    /// Optional coupling metadata / user data.
    const void* user_data{nullptr};
};

/**
 * @brief Callback for operator residual contribution.
 *
 * Writes residual contributions into `output` (size = target entities × target stride).
 */
using AuxiliaryOperatorResidualFn = std::function<void(
    const AuxiliaryOperatorContext& ctx,
    std::span<Real> output)>;

/**
 * @brief Callback for operator Jacobian contribution.
 *
 * Writes Jacobian entries.  Layout depends on topology.
 */
using AuxiliaryOperatorJacobianFn = std::function<void(
    const AuxiliaryOperatorContext& ctx,
    std::span<Real> jacobian)>;

/**
 * @brief Callback for operator mass-like contribution (optional).
 */
using AuxiliaryOperatorMassFn = std::function<void(
    const AuxiliaryOperatorContext& ctx,
    std::span<Real> mass)>;

/**
 * @brief Callback for operator transfer/prolongation (optional).
 */
using AuxiliaryOperatorTransferFn = std::function<void(
    const AuxiliaryOperatorContext& ctx,
    std::span<const Real> input,
    std::span<Real> output)>;

// ---------------------------------------------------------------------------
//  Built operator descriptor
// ---------------------------------------------------------------------------

/**
 * @brief Fully configured auxiliary operator descriptor.
 *
 * Created by `AuxiliaryOperatorBuilder::build()`.
 */
struct AuxiliaryOperatorDescriptor {
    std::string name{};

    AuxiliaryCouplingType coupling_type{AuxiliaryCouplingType::AuxSelf};
    AuxiliaryCouplingTopology topology{AuxiliaryCouplingTopology::Dense};

    std::string source_name{};
    std::string target_name{};

    AuxiliaryOperatorResidualFn residual_fn{};
    AuxiliaryOperatorJacobianFn jacobian_fn{};
    AuxiliaryOperatorMassFn mass_fn{};
    AuxiliaryOperatorTransferFn transfer_fn{};

    AuxiliaryDerivativePolicy derivative_policy{};
    bool has_derivative_policy{false};
};

// ---------------------------------------------------------------------------
//  Builder
// ---------------------------------------------------------------------------

/**
 * @brief Fluent builder for auxiliary operators.
 */
class AuxiliaryOperatorBuilder {
public:
    explicit AuxiliaryOperatorBuilder(std::string name)
        : name_(std::move(name))
    {
        FE_THROW_IF(name_.empty(), InvalidArgumentException,
                    "AuxiliaryOperatorBuilder: empty operator name");
    }

    AuxiliaryOperatorBuilder& source(std::string source_name)
    {
        source_ = std::move(source_name);
        return *this;
    }

    AuxiliaryOperatorBuilder& target(std::string target_name)
    {
        target_ = std::move(target_name);
        return *this;
    }

    AuxiliaryOperatorBuilder& topology(AuxiliaryCouplingTopology topo)
    {
        topology_ = topo;
        return *this;
    }

    AuxiliaryOperatorBuilder& residual(AuxiliaryOperatorResidualFn fn)
    {
        residual_fn_ = std::move(fn);
        return *this;
    }

    AuxiliaryOperatorBuilder& jacobian(AuxiliaryOperatorJacobianFn fn)
    {
        jacobian_fn_ = std::move(fn);
        return *this;
    }

    AuxiliaryOperatorBuilder& mass(AuxiliaryOperatorMassFn fn)
    {
        mass_fn_ = std::move(fn);
        return *this;
    }

    AuxiliaryOperatorBuilder& transfer(AuxiliaryOperatorTransferFn fn)
    {
        transfer_fn_ = std::move(fn);
        return *this;
    }

    AuxiliaryOperatorBuilder& derivatives(AuxiliaryDerivativePolicy policy)
    {
        derivative_policy_ = policy;
        has_derivative_policy_ = true;
        return *this;
    }

    /**
     * @brief Build the operator descriptor.
     *
     * Infers coupling type from source/target:
     * - Same name → AuxSelf
     * - Both auxiliary → AuxToAux
     * - Source starts with "field:" → FieldToAux
     * - Target starts with "field:" → AuxToField
     */
    [[nodiscard]] AuxiliaryOperatorDescriptor build() const
    {
        FE_THROW_IF(source_.empty(), InvalidArgumentException,
                    "AuxiliaryOperatorBuilder::build: source not set");
        FE_THROW_IF(target_.empty(), InvalidArgumentException,
                    "AuxiliaryOperatorBuilder::build: target not set");
        FE_THROW_IF(!residual_fn_, InvalidArgumentException,
                    "AuxiliaryOperatorBuilder::build: residual callback not set");

        AuxiliaryOperatorDescriptor desc;
        desc.name = name_;
        desc.source_name = source_;
        desc.target_name = target_;
        desc.topology = topology_;
        desc.residual_fn = residual_fn_;
        desc.jacobian_fn = jacobian_fn_;
        desc.mass_fn = mass_fn_;
        desc.transfer_fn = transfer_fn_;
        desc.derivative_policy = derivative_policy_;
        desc.has_derivative_policy = has_derivative_policy_;

        // Infer coupling type.
        const bool src_is_field = source_.substr(0, 6) == "field:";
        const bool tgt_is_field = target_.substr(0, 6) == "field:";

        // Reject field-to-field before any other classification.
        if (src_is_field && tgt_is_field) {
            throw std::invalid_argument(
                "AuxiliaryOperatorBuilder: both source '" + source_ +
                "' and target '" + target_ + "' are FE field references. "
                "Use the FE assembly pipeline for field-to-field coupling.");
        }

        if (source_ == target_) {
            desc.coupling_type = AuxiliaryCouplingType::AuxSelf;
        } else if (src_is_field && !tgt_is_field) {
            desc.coupling_type = AuxiliaryCouplingType::FieldToAux;
        } else if (!src_is_field && tgt_is_field) {
            desc.coupling_type = AuxiliaryCouplingType::AuxToField;
        } else {
            desc.coupling_type = AuxiliaryCouplingType::AuxToAux;
        }

        return desc;
    }

private:
    std::string name_{};
    std::string source_{};
    std::string target_{};
    AuxiliaryCouplingTopology topology_{AuxiliaryCouplingTopology::Dense};
    AuxiliaryOperatorResidualFn residual_fn_{};
    AuxiliaryOperatorJacobianFn jacobian_fn_{};
    AuxiliaryOperatorMassFn mass_fn_{};
    AuxiliaryOperatorTransferFn transfer_fn_{};
    AuxiliaryDerivativePolicy derivative_policy_{};
    bool has_derivative_policy_{false};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_AUXILIARY_OPERATOR_BUILDER_H
