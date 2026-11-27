/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

#include "Assembly/Assembler.h"
#include "Assembly/AssemblyContext.h"

#include <functional>
#include <map>
#include <span>
#include <utility>
#include <vector>

namespace svmp::FE::forms::cut_surface {

// Indexed by the trial basis function and then the retained quadrature point.
using Table = std::vector<std::vector<assembly::CutSurfaceSampleVariation>>;

[[nodiscard]] Table differentiateRule(const assembly::IMeshAccess& mesh,
                                      GlobalIndex cell,
                                      const geometry::CutQuadratureRule& rule,
                                      const spaces::FunctionSpace& level_set,
                                      std::span<const Real> coefficients,
                                      const spaces::FunctionSpace& trial);

// Integral of the reference outward-normal speed, divided by the parent reference measure.
[[nodiscard]] std::vector<Real> referenceFractionDerivative(const geometry::CutQuadratureRule& surface,
    ElementType type, GlobalIndex cell, const spaces::FunctionSpace& field, std::span<const Real> coefficients,
    const spaces::FunctionSpace& trial, Real signed_inverse_parent_measure);

[[nodiscard]] Table implicitNormalSpeed(const assembly::AssemblyContext& context, FieldId field);

// Each active field builds its geometry table once per assembly of a retained rule.
class Binding final : public assembly::CutSurfaceVariationProvider {
public:
    using Build = std::function<Table(FieldId)>;

    Binding(assembly::AssemblyContext& context, Build build, bool positive_side)
        : context_(context), build_(std::move(build)), positive_side_(positive_side)
    {
        context_.setCutSurfaceVariations(this);
    }

    Binding(assembly::AssemblyContext& context, FieldId field, std::vector<Real> derivative)
        : context_(context), positive_side_(false), fraction_field_(field), fraction_derivative_(std::move(derivative))
    {
        context_.setCutSurfaceVariations(this);
    }

    assembly::CutFractionDerivativeView fractionDerivative() const noexcept override
    {
        return {fraction_field_, fraction_derivative_};
    }

    ~Binding() override { context_.setCutSurfaceVariations(nullptr); }
    Binding(const Binding&) = delete;
    Binding& operator=(const Binding&) = delete;

    const assembly::CutSurfaceSampleVariation& sample(
        FieldId field, LocalIndex trial_dof, LocalIndex quadrature_point) const override
    {
        return table(field).at(trial_dof).at(quadrature_point);
    }

    std::span<const assembly::CutSurfaceSampleVariation> column(FieldId field, LocalIndex trial_dof) const override
    {
        return table(field).at(trial_dof);
    }

private:
    const Table& table(FieldId field) const
    {
        auto table = tables_.find(field);
        if (table == tables_.end()) {
            auto variations = build_(field);
            if (positive_side_) {
                for (auto& column : variations) {
                    for (auto& sample : column) {
                        for (auto& component : sample.normal) {
                            component = -component;
                        }
                    }
                }
            }
            table = tables_.emplace(field, std::move(variations)).first;
        }
        return table->second;
    }

    assembly::AssemblyContext& context_;
    Build build_;
    bool positive_side_;
    FieldId fraction_field_{INVALID_FIELD_ID};
    std::vector<Real> fraction_derivative_{};
    mutable std::map<FieldId, Table> tables_;
};

} // namespace svmp::FE::forms::cut_surface
