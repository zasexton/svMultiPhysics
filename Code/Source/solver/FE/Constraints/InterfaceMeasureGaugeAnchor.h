/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_INTERFACEMEASUREGAUGEANCHOR_H
#define SVMP_FE_CONSTRAINTS_INTERFACEMEASUREGAUGEANCHOR_H

/**
 * @file InterfaceMeasureGaugeAnchor.h
 * @brief A natural condition on a generated interface anchors a constant mode
 *
 * A traction (or flux) prescribed on a generated interface fixes the constant
 * mode of a field, but only while the interface has a positive measure.  The
 * anchor records the anchoring evidence in the gauge registry, once per field
 * and source, and checks the measure of the interface rules of the cut
 * context, summed over the ranks, at setup (when a context exists), at every
 * assembly (a global kernel) and at every cut-context update (a callback).
 * No gauge constraint is inserted when the measure vanishes; the solve is
 * rejected instead.
 */

#include "Core/Types.h"
#include "Systems/GlobalKernel.h"

#include <memory>
#include <string>
#include <string_view>

namespace svmp {
namespace FE {
namespace assembly {
class CutIntegrationContext;
} // namespace assembly

namespace systems {
class FESystem;
} // namespace systems

namespace constraints {

class InterfaceMeasureGaugeKernel;

/// Texts of an anchor.  The defaults are generic; a module passes its own so
/// that its messages, kernel and callback names stay its own.
struct InterfaceMeasureGaugeAnchorTexts {
    std::string owner{"InterfaceMeasureGaugeAnchor"};  ///< message prefix
    std::string anchor_source{"Natural condition on a generated interface anchors a constant mode"};
    std::string kernel_name{"InterfaceMeasureGaugeAnchor"};        ///< "<kernel_name>:<marker>"
    std::string callback_name{"interface_measure_gauge_anchor"};   ///< "<callback_name>:<marker>"
    std::string requirement{
        "interface-measure gauge anchor requires a positive generated interface measure before a solve"};
    std::string preflight_phase{"embedded_interface_measure_preflight"};
    std::string preflight_rejection{"another communicator rank rejected embedded interface measure preflight"};
    std::string context_phase{"interface_measure_gauge_anchor_context"};
    std::string null_context_accepted{"interface-measure gauge anchor null context was unexpectedly accepted"};
    std::string gauge_policy{"reject_zero_interface_no_dynamic_gauge_insertion"};
};

struct InterfaceMeasureGaugeAnchor {
    FieldId field{INVALID_FIELD_ID};   ///< its ScalarConstant mode is anchored
    int interface_marker{-1};
    std::string level_set_field_name{};
    std::string domain_id{};
    InterfaceMeasureGaugeAnchorTexts texts{};
    /// The global kernel; a module may pass one of its own type, whose name
    /// the kernel manifest shows.  Null: an InterfaceMeasureGaugeKernel.
    std::shared_ptr<InterfaceMeasureGaugeKernel> kernel{};
};

/// The global kernel of an anchor: checks the measure at setup (when a cut
/// context exists), at every assembly, and on validateContextUpdate.
class InterfaceMeasureGaugeKernel : public systems::GlobalKernel {
public:
    explicit InterfaceMeasureGaugeKernel(const InterfaceMeasureGaugeAnchor& anchor);

    /// "<kernel_name>:<interface marker>".
    [[nodiscard]] std::string name() const override;

    void addSparsityCouplings(const systems::FESystem& system,
                              sparsity::SparsityPattern& pattern) const override;

    [[nodiscard]] assembly::AssemblyResult assemble(const systems::FESystem& system,
                                                    const systems::AssemblyRequest& request,
                                                    const systems::SystemStateView& state,
                                                    assembly::GlobalSystemView* matrix_out,
                                                    assembly::GlobalSystemView* vector_out) override;

    void validateContextUpdate(const systems::FESystem& system,
                               const assembly::CutIntegrationContext* context) const;

private:
    void validate(const systems::FESystem& system,
                  const assembly::CutIntegrationContext* context,
                  std::string_view stage) const;

    int interface_marker_{-1};
    std::string level_set_field_name_{};
    std::string domain_id_{};
    InterfaceMeasureGaugeAnchorTexts texts_{};
};

/**
 * @brief Records the anchoring evidence and adds the measure checks.
 *
 * The global kernel is added to @p operator_tag.  The measure check throws
 * std::runtime_error with "<owner>: <requirement> validation_stage=<stage>
 * marker=... level_set_field='...' domain_id='...' cut_context=present|missing
 * global_interface_rules=... global_interface_quadrature_points=...
 * global_stored_measure=... global_quadrature_weight_measure=...
 * invalid_measure=true|false gauge_policy=<gauge_policy>".
 */
void anchorGaugeOnInterfaceMeasure(systems::FESystem& system,
                                   const std::string& operator_tag,
                                   const InterfaceMeasureGaugeAnchor& anchor);

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_INTERFACEMEASUREGAUGEANCHOR_H
