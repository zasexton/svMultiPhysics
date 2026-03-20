#ifndef SVMP_FE_FORMS_STANDARD_BCS_H
#define SVMP_FE_FORMS_STANDARD_BCS_H

/**
 * @file StandardBCs.h
 * @brief Standard physics-agnostic boundary condition implementations
 */

#include "Forms/BoundaryCondition.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace bc {

/**
 * @brief Marker-only boundary condition used for conflict validation
 *
 * This BC reserves a boundary marker in BoundaryConditionManager::validate()
 * without contributing weak-form terms or strong constraints.
 */
class ReservedBC final : public BoundaryCondition {
public:
    explicit ReservedBC(int boundary_marker)
        : boundary_marker_(boundary_marker)
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument("ReservedBC: boundary_marker must be >= 0");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }

    void contributeToResidual(FormExpr& /*residual*/,
                              const FormExpr& /*u*/,
                              const FormExpr& /*v*/) const override
    {
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId /*field_id*/, const systems::FESystem* /*system*/) const override
    {
        return {};  // Reserved BC — no mathematical constraint
    }

private:
    int boundary_marker_{-1};
};

/**
 * @brief Natural (Neumann/traction) BC: adds -∫ flux · v ds(marker)
 */
class NaturalBC : public BoundaryCondition {
public:
    NaturalBC(int boundary_marker, FormExpr flux)
        : boundary_marker_(boundary_marker)
        , flux_(std::move(flux))
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument("NaturalBC: boundary_marker must be >= 0");
        }
        if (!flux_.isValid()) {
            throw std::invalid_argument("NaturalBC: invalid flux expression");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& /*u*/,
                              const FormExpr& v) const override
    {
        residual = residual - inner(flux_, v).ds(boundary_marker_);
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.boundary_marker = boundary_marker_;
        d.trace_kind = analysis::TraceKind::Flux;
        d.enforcement_kind = analysis::EnforcementKind::WeakConsistent;
        d.anchors_constant_mode = false;
        d.anchors_rigid_body_translation = false;
        d.anchors_rigid_body_rotation = false;
        d.source = "NaturalBC on marker " + std::to_string(boundary_marker_);
        return {d};
    }

private:
protected:
    int boundary_marker_{-1};
    FormExpr flux_{};
};

/**
 * @brief Robin BC: adds ∫ alpha * (u · v) ds(marker) - ∫ rhs · v ds(marker)
 */
class RobinBC : public BoundaryCondition {
public:
    RobinBC(int boundary_marker, FormExpr alpha, FormExpr rhs)
        : boundary_marker_(boundary_marker)
        , alpha_(std::move(alpha))
        , rhs_(std::move(rhs))
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument("RobinBC: boundary_marker must be >= 0");
        }
        if (!alpha_.isValid()) {
            throw std::invalid_argument("RobinBC: invalid alpha expression");
        }
        if (!rhs_.isValid()) {
            throw std::invalid_argument("RobinBC: invalid rhs expression");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& u,
                              const FormExpr& v) const override
    {
        residual = residual + (alpha_ * inner(u, v)).ds(boundary_marker_) - inner(rhs_, v).ds(boundary_marker_);
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.boundary_marker = boundary_marker_;
        d.trace_kind = analysis::TraceKind::Mixed;
        d.enforcement_kind = analysis::EnforcementKind::WeakPenalty;
        d.anchors_constant_mode = true;
        d.anchors_rigid_body_translation = true;
        d.anchors_rigid_body_rotation = false;
        d.source = "RobinBC on marker " + std::to_string(boundary_marker_);
        return {d};
    }

private:
protected:
    int boundary_marker_{-1};
    FormExpr alpha_{};
    FormExpr rhs_{};
};

/**
 * @brief Essential (Dirichlet) BC: declares u = g on ds(marker)
 *
 * For vector-valued fields, this returns one StrongDirichlet per component.
 */
class EssentialBC final : public BoundaryCondition {
public:
    EssentialBC(int boundary_marker, FormExpr value, std::string symbol = "u")
        : boundary_marker_(boundary_marker)
        , value_components_{std::move(value)}
        , symbol_(std::move(symbol))
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument("EssentialBC: boundary_marker must be >= 0");
        }
        if (!value_components_.front().isValid()) {
            throw std::invalid_argument("EssentialBC: invalid value expression");
        }
    }

    EssentialBC(int boundary_marker, std::vector<FormExpr> value_components, std::string symbol = "u")
        : boundary_marker_(boundary_marker)
        , value_components_(std::move(value_components))
        , symbol_(std::move(symbol))
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument("EssentialBC: boundary_marker must be >= 0");
        }
        if (value_components_.empty()) {
            throw std::invalid_argument("EssentialBC: empty value component list");
        }
        for (const auto& c : value_components_) {
            if (!c.isValid()) {
                throw std::invalid_argument("EssentialBC: invalid value component expression");
            }
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }

    void contributeToResidual(FormExpr& /*residual*/,
                              const FormExpr& /*u*/,
                              const FormExpr& /*v*/) const override
    {
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId field_id) const override
    {
        std::vector<StrongDirichlet> out;
        out.reserve(value_components_.size());

        if (value_components_.size() == 1u) {
            out.push_back(strongDirichlet(field_id, boundary_marker_, value_components_.front(), symbol_));
            return out;
        }

        for (std::size_t comp = 0; comp < value_components_.size(); ++comp) {
            out.push_back(strongDirichlet(field_id,
                                          boundary_marker_,
                                          value_components_[comp],
                                          symbol_,
                                          static_cast<int>(comp)));
        }
        return out;
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        std::vector<analysis::BoundaryConditionDescriptor> descs;

        if (value_components_.size() <= 1u) {
            // Scalar field or single-component vector: field-wide descriptor
            analysis::BoundaryConditionDescriptor d;
            d.primary_variable = analysis::VariableKey::field(field_id);
            d.component = -1;
            d.boundary_marker = boundary_marker_;
            d.trace_kind = analysis::TraceKind::Value;
            d.enforcement_kind = analysis::EnforcementKind::Strong;
            d.anchors_constant_mode = true;
            d.anchors_rigid_body_translation = true;
            d.anchors_rigid_body_rotation = false;
            d.source = "EssentialBC on marker " + std::to_string(boundary_marker_);
            descs.push_back(std::move(d));
        } else {
            // Multi-component: emit one descriptor per component so that
            // partially constrained vector fields are correctly analyzed.
            for (std::size_t comp = 0; comp < value_components_.size(); ++comp) {
                analysis::BoundaryConditionDescriptor d;
                d.primary_variable = analysis::VariableKey::field(field_id, static_cast<int>(comp));
                d.component = static_cast<int>(comp);
                d.boundary_marker = boundary_marker_;
                d.trace_kind = analysis::TraceKind::Value;
                d.enforcement_kind = analysis::EnforcementKind::Strong;
                d.anchors_constant_mode = true;
                d.anchors_rigid_body_translation = true;
                d.anchors_rigid_body_rotation = false;
                d.source = "EssentialBC comp " + std::to_string(comp) +
                           " on marker " + std::to_string(boundary_marker_);
                descs.push_back(std::move(d));
            }
        }
        return descs;
    }

private:
    int boundary_marker_{-1};
    std::vector<FormExpr> value_components_{};
    std::string symbol_{"u"};
};

} // namespace bc
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_STANDARD_BCS_H
