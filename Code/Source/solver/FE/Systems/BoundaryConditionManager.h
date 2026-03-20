#ifndef SVMP_FE_SYSTEMS_BOUNDARY_CONDITION_MANAGER_H
#define SVMP_FE_SYSTEMS_BOUNDARY_CONDITION_MANAGER_H

/**
 * @file BoundaryConditionManager.h
 * @brief Systems-side manager for physics-agnostic boundary conditions
 */

#include "Forms/BoundaryCondition.h"
#include "Systems/FESystem.h"
#include "Systems/SystemConstraint.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

namespace detail {

class BoundaryConditionAffineConstraint final : public ISystemConstraint {
public:
    BoundaryConditionAffineConstraint(FieldId field_id,
                                      std::vector<std::unique_ptr<forms::bc::BoundaryCondition>> bcs)
        : field_id_(field_id)
        , bcs_(std::move(bcs))
    {
    }

    void apply(const FESystem& /*system*/, constraints::AffineConstraints& constraints) override
    {
        for (const auto& bc : bcs_) {
            if (!bc) {
                throw std::invalid_argument("BoundaryConditionAffineConstraint::apply: null boundary condition");
            }
            bc->addAffineConstraints(constraints, field_id_);
        }
    }

    bool updateValues(const FESystem& /*system*/,
                      constraints::AffineConstraints& /*constraints*/,
                      double /*time*/,
                      double /*dt*/) override
    {
        return false;
    }

    [[nodiscard]] bool isTimeDependent() const noexcept override { return false; }

private:
    FieldId field_id_{INVALID_FIELD_ID};
    std::vector<std::unique_ptr<forms::bc::BoundaryCondition>> bcs_{};
};

} // namespace detail

class BoundaryConditionManager {
public:
    void add(std::unique_ptr<forms::bc::BoundaryCondition> bc)
    {
        if (!bc) {
            throw std::invalid_argument("BoundaryConditionManager::add: null boundary condition");
        }
        bcs_.push_back(std::move(bc));
    }

    template <typename OptionContainer, typename Factory>
    void install(const OptionContainer& options, Factory&& factory)
    {
        for (const auto& opt : options) {
            add(factory(opt));
        }
    }

    void validate() const
    {
        std::unordered_map<int, const forms::bc::BoundaryCondition*> seen;
        for (const auto& bc : bcs_) {
            if (!bc) {
                throw std::invalid_argument("BoundaryConditionManager::validate: null boundary condition");
            }
            const int marker = bc->boundaryMarker();
            if (marker < 0) {
                continue;
            }

            auto [it, inserted] = seen.emplace(marker, bc.get());
            if (!inserted) {
                throw std::invalid_argument("BoundaryConditionManager::validate: multiple boundary conditions target boundary_marker " +
                                            std::to_string(marker));
            }
        }
    }

    [[nodiscard]] std::vector<forms::bc::StrongDirichlet> getStrongConstraints(FieldId field_id) const
    {
        std::vector<forms::bc::StrongDirichlet> out;
        for (const auto& bc : bcs_) {
            if (!bc) {
                throw std::invalid_argument("BoundaryConditionManager::getStrongConstraints: null boundary condition");
            }

            auto strong = bc->getStrongConstraints(field_id);
            for (auto& c : strong) {
                out.push_back(std::move(c));
            }
        }
        return out;
    }

    void apply(FESystem& system,
               forms::FormExpr& residual,
               const forms::FormExpr& u,
               const forms::FormExpr& v,
               FieldId field_id)
    {
        for (const auto& bc : bcs_) {
            if (!bc) {
                throw std::invalid_argument("BoundaryConditionManager::apply: null boundary condition");
            }
            bc->setup(system, field_id);
        }

        // Collect analysis metadata (BC descriptors) from BCs BEFORE they are moved.
        // The BCs will be consumed (moved into BoundaryConditionAffineConstraint)
        // at the end of this method, so we must query analysisMetadata() now.
        // Also lower each descriptor into normalized ContributionDescriptors,
        // and derive gauge anchoring evidence from descriptors.
        {
            auto& reg = system.gaugeRegistry();
            const auto& rec = system.fieldRecord(field_id);
            const int n_comp = rec.components;

            for (const auto& bc : bcs_) {
                if (!bc) continue;
                const int marker = bc->boundaryMarker();
                auto descs = bc->analysisMetadata(field_id, &system);
                for (auto& d : descs) {
                    // Lower to ContributionDescriptors for analysis pipeline
                    auto contributions = analysis::lowerBCDescriptor(d);
                    for (auto& c : contributions) {
                        system.addContribution(std::move(c));
                    }

                    // Derive gauge anchoring evidence from descriptor
                    const std::string src = "BC on boundary " + std::to_string(marker);
                    for (auto family : {gauge::NullspaceModeFamily::ScalarConstant,
                                        gauge::NullspaceModeFamily::ComponentwiseConstant,
                                        gauge::NullspaceModeFamily::KernelOfSymGrad}) {
                        const bool is_multicomp_family =
                            (family == gauge::NullspaceModeFamily::ComponentwiseConstant ||
                             family == gauge::NullspaceModeFamily::KernelOfSymGrad);

                        auto verdict = analysis::descriptorToVerdict(d, family);
                        if (verdict == gauge::AnchoringVerdict::Unknown) continue;

                        if (is_multicomp_family && n_comp > 1) {
                            for (int comp = 0; comp < n_comp; ++comp) {
                                gauge::AnchoringEvidence ev;
                                ev.field = field_id;
                                ev.component = comp;
                                ev.family = family;
                                ev.verdict = verdict;
                                ev.source = src;
                                ev.boundary_marker = marker;
                                reg.addAnchoring(std::move(ev));
                            }
                        } else {
                            gauge::AnchoringEvidence ev;
                            ev.field = field_id;
                            ev.family = family;
                            ev.verdict = verdict;
                            ev.source = src;
                            ev.boundary_marker = marker;
                            reg.addAnchoring(std::move(ev));
                        }
                    }

                    system.addBoundaryConditionDescriptor(std::move(d));
                }
            }
        }

        for (const auto& bc : bcs_) {
            if (!bc) {
                throw std::invalid_argument("BoundaryConditionManager::apply: null boundary condition");
            }
            bc->contributeToResidual(residual, u, v);
        }

        // Persist any advanced affine constraints for setup-time lowering.
        if (!bcs_.empty()) {
            system.addSystemConstraint(
                std::make_unique<detail::BoundaryConditionAffineConstraint>(field_id, std::move(bcs_)));
        }
    }

    void apply(FESystem& system,
               forms::FormExpr& residual,
               const forms::FormExpr& u,
               const forms::FormExpr& v,
               FieldId field_id,
               std::vector<forms::bc::StrongDirichlet>& constraints)
    {
        auto strong = getStrongConstraints(field_id);
        for (auto& c : strong) {
            constraints.push_back(std::move(c));
        }

        apply(system, residual, u, v, field_id);
    }

private:
    std::vector<std::unique_ptr<forms::bc::BoundaryCondition>> bcs_{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_BOUNDARY_CONDITION_MANAGER_H
