#ifndef SVMP_FE_SYSTEMS_BOUNDARY_CONDITION_MANAGER_H
#define SVMP_FE_SYSTEMS_BOUNDARY_CONDITION_MANAGER_H

/**
 * @file BoundaryConditionManager.h
 * @brief Systems-side manager for physics-agnostic boundary conditions
 */

#include "Analysis/FormExprScanner.h"
#include "Forms/BoundaryCondition.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Constraints/SystemConstraint.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <sstream>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

namespace detail {

class BoundaryConditionAffineConstraint final : public constraints::ISystemConstraint {
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

    [[nodiscard]] SetupStorageRequirements storageRequirements() const noexcept override
    {
        return {};
    }

private:
    FieldId field_id_{INVALID_FIELD_ID};
    std::vector<std::unique_ptr<forms::bc::BoundaryCondition>> bcs_{};
};

} // namespace detail

/**
 * @brief Systems-side manager for boundary conditions on a single field
 *
 * Collects BoundaryCondition objects, validates for marker conflicts, and
 * applies them in one call. Handles all BC types uniformly: strong Dirichlet,
 * weak Neumann/Robin, Nitsche, coupled, etc.
 *
 * Usage (single-field or primary field of a coupled system):
 * @code
 *   BoundaryConditionManager bc_manager;
 *   bc_manager.add(std::make_unique<EssentialBC>(1, ...));
 *   bc_manager.add(std::make_unique<NaturalBC>(2, ...));
 *   bc_manager.applyAll(system, residual, u, v, u_id);
 * @endcode
 *
 * Usage (secondary field with strong BCs only, e.g., pressure Dirichlet):
 * @code
 *   BoundaryConditionManager p_bc_manager;
 *   p_bc_manager.add(std::make_unique<EssentialBC>(3, ...));
 *   p_bc_manager.applyAll(system, p_id);
 * @endcode
 */
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
        struct MarkerBucket {
            const forms::bc::BoundaryCondition* exclusive{nullptr};
            std::size_t shared_count{0};
        };

        std::unordered_map<std::string, MarkerBucket> seen;
        for (const auto& bc : bcs_) {
            if (!bc) {
                throw std::invalid_argument("BoundaryConditionManager::validate: null boundary condition");
            }
            const int marker = bc->targetMarker();
            if (marker < 0) {
                continue;
            }

            std::ostringstream oss;
            oss << static_cast<int>(bc->targetDomain()) << ":" << marker;
            auto& bucket = seen[oss.str()];

            if (bc->allowsMarkerSharing()) {
                if (bucket.exclusive) {
                    throw std::invalid_argument(
                        "BoundaryConditionManager::validate: weak boundary condition targets "
                        + bc->targetDescription() +
                        " but that target is already reserved by an exclusive condition");
                }
                ++bucket.shared_count;
                continue;
            }

            if (bucket.exclusive || bucket.shared_count > 0u) {
                throw std::invalid_argument(
                    "BoundaryConditionManager::validate: multiple incompatible boundary conditions target "
                    + bc->targetDescription());
            }
            bucket.exclusive = bc.get();
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
            bc->installSystemConstraints(system, field_id);
        }

        // Collect analysis metadata (BC descriptors) from BCs BEFORE they are moved.
        // The BCs will be consumed (moved into BoundaryConditionAffineConstraint)
        // at the end of this method, so we must query analysisMetadata() now.
        // Also lower each descriptor into normalized ContributionDescriptors,
        // and derive gauge anchoring evidence from descriptors.
        collectAnalysisMetadata_(system, field_id);

        for (const auto& bc : bcs_) {
            if (!bc) {
                throw std::invalid_argument("BoundaryConditionManager::apply: null boundary condition");
            }
            bc->contributeToResidual(residual, u, v);
        }

        if (residual.isValid() && residual.node()) {
            const auto scan = analysis::scanFormExpr(*residual.node());
            if (!scan.boundary_functional_names.empty() ||
                !scan.auxiliary_state_names.empty()) {
                throw std::invalid_argument(
                    "BoundaryConditionManager::apply: legacy coupled-boundary placeholders are no longer supported. "
                    "Register FE-backed inputs with boundaryIntegral(...) and couple boundary terms through deployed "
                    "AuxiliaryOutput(...) expressions instead. Auxiliary outputs remain symbolic here and are "
                    "resolved by installFormulation() so metadata can retain stable output references for "
                    "monolithic coupling.");
            }
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

    // ====================================================================
    // One-call convenience: validate + apply all BCs + install strong BCs
    // ====================================================================

    /**
     * @brief Apply all boundary conditions for a single field
     *
     * This is the recommended single entry point for boundary conditions.
     * It handles all BC types uniformly:
     *   1. validate() — check for conflicting markers
     *   2. Setup each BC, collect analysis metadata
     *   3. Call contributeToResidual() for weak terms (Neumann, Robin, Nitsche, etc.)
     *   4. installStrongDirichlet() — install strong Dirichlet constraints
     *   5. Persist affine constraints for setup-time lowering
     *
     * After this call, the residual is ready for installFormulation() and
     * the system has all BC constraints installed.
     *
     * @param system     FESystem to install constraints into
     * @param residual   Residual expression (modified in-place for weak BCs)
     * @param u          State field symbol
     * @param v          Test field symbol
     * @param field_id   FieldId for strong-constraint extraction
     */
    void applyAll(FESystem& system,
                  forms::FormExpr& residual,
                  const forms::FormExpr& u,
                  const forms::FormExpr& v,
                  FieldId field_id)
    {
        validate();

        auto strong = getStrongConstraints(field_id);
        apply(system, residual, u, v, field_id);

        if (!strong.empty()) {
            installStrongDirichlet(system, strong);
        }
    }

    /**
     * @brief Apply BCs for a field that has no residual equation of its own
     *
     * For coupled multi-field systems where a secondary field (e.g., pressure)
     * has BCs but no independent residual equation, this overload avoids the
     * need for a state/test symbol pair.
     *
     * All BCs must be strong/algebraic (hasWeakTerms() == false). If any BC
     * reports weak terms, the call throws *before mutating the system* — the
     * caller must use the full applyAll(system, residual, u, v, field_id)
     * overload for fields with weak BCs.
     *
     * Typical usage:
     * @code
     *   vel_bc_manager.applyAll(system, residual, u, v, u_id);
     *   pres_bc_manager.applyAll(system, p_id);
     *   installFormulation(system, "equations", {u_id, p_id}, residual);
     * @endcode
     *
     * @param system     FESystem to install constraints into
     * @param field_id   FieldId for strong-constraint extraction
     * @throws std::invalid_argument if any BC has weak residual terms
     */
    void applyAll(FESystem& system, FieldId field_id)
    {
        // validate() is read-only (checks bcs_ for marker conflicts).
        validate();

        // Reject weak BCs before any system mutation (setup, metadata, constraints).
        for (const auto& bc : bcs_) {
            if (!bc) continue;
            if (bc->hasWeakTerms()) {
                throw std::invalid_argument(
                    "BoundaryConditionManager::applyAll(system, field_id): BC on "
                    + bc->targetDescription()
                    + " has weak residual terms but no residual expression was provided. "
                      "Use applyAll(system, residual, u, v, field_id) for fields with "
                      "Neumann, Robin, or Nitsche BCs.");
            }
        }

        // Safe to proceed — all BCs are strong/algebraic only.
        for (const auto& bc : bcs_) {
            if (!bc) continue;
            bc->setup(system, field_id);
            bc->installSystemConstraints(system, field_id);
        }

        collectAnalysisMetadata_(system, field_id);

        auto strong = getStrongConstraints(field_id);

        // Persist affine constraints for setup-time lowering.
        if (!bcs_.empty()) {
            system.addSystemConstraint(
                std::make_unique<detail::BoundaryConditionAffineConstraint>(field_id, std::move(bcs_)));
        }

        if (!strong.empty()) {
            installStrongDirichlet(system, strong);
        }
    }

private:
    std::vector<std::unique_ptr<forms::bc::BoundaryCondition>> bcs_{};

    /// Shared implementation: collect analysis metadata and gauge anchoring
    /// evidence from all BCs.  Called by both apply() and applyAll(field_id).
    void collectAnalysisMetadata_(FESystem& system, FieldId field_id)
    {
        auto& reg = system.gaugeRegistry();
        const auto& rec = system.fieldRecord(field_id);
        const int n_comp = rec.components;

        for (const auto& bc : bcs_) {
            if (!bc) continue;
            auto descs = bc->analysisMetadata(field_id, &system);
            for (auto& d : descs) {
                auto contributions = analysis::lowerBCDescriptor(d);
                for (auto& c : contributions) {
                    system.addContribution(std::move(c));
                }

                std::string src = "BC on ";
                if (d.domain == analysis::DomainKind::InterfaceFace && d.interface_marker >= 0) {
                    src += "interface " + std::to_string(d.interface_marker);
                } else if (d.boundary_marker >= 0) {
                    src += "boundary " + std::to_string(d.boundary_marker);
                } else {
                    src += bc->targetDescription();
                }

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
                            ev.boundary_marker = d.boundary_marker;
                            reg.addAnchoring(std::move(ev));
                        }
                    } else {
                        gauge::AnchoringEvidence ev;
                        ev.field = field_id;
                        ev.family = family;
                        ev.verdict = verdict;
                        ev.source = src;
                        ev.boundary_marker = d.boundary_marker;
                        reg.addAnchoring(std::move(ev));
                    }
                }

                system.addBoundaryConditionDescriptor(std::move(d));
            }
        }
    }
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_BOUNDARY_CONDITION_MANAGER_H
