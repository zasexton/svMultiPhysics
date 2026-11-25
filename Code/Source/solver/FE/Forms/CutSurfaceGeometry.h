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

// Each active field builds its geometry table once per assembly of a retained rule.
class Binding final : public assembly::CutSurfaceVariationProvider {
public:
    using Build = std::function<Table(FieldId)>;

    Binding(assembly::AssemblyContext& context, Build build, bool positive_side)
        : context_(context), build_(std::move(build)), positive_side_(positive_side)
    {
        context_.setCutSurfaceVariations(this);
    }

    ~Binding() override { context_.setCutSurfaceVariations(nullptr); }
    Binding(const Binding&) = delete;
    Binding& operator=(const Binding&) = delete;

    const assembly::CutSurfaceSampleVariation& sample(
        FieldId field, LocalIndex trial_dof, LocalIndex quadrature_point) const override
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
        return table->second.at(trial_dof).at(quadrature_point);
    }

private:
    assembly::AssemblyContext& context_;
    Build build_;
    bool positive_side_;
    mutable std::map<FieldId, Table> tables_;
};

} // namespace svmp::FE::forms::cut_surface
