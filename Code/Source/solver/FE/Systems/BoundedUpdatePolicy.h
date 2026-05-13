/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_BOUNDEDUPDATEPOLICY_H
#define SVMP_FE_SYSTEMS_BOUNDEDUPDATEPOLICY_H

#include "Systems/StateAdmissibility.h"

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

enum class BoundedUpdatePolicyKind : std::uint8_t {
    None,
    CheckOnly,
    ClampCellAverage,
    RejectStepOnViolation
};

enum class BoundedUpdateDecision : std::uint8_t {
    Accepted,
    AcceptedWithModification,
    Rejected
};

using BoundedUpdateSamplesCallback =
    std::function<void(std::vector<StateAdmissibilitySample>& samples)>;
using BoundedUpdateCandidateFilter =
    std::function<bool(std::span<const StateAdmissibilitySample> samples)>;

struct BoundedUpdatePolicy {
    BoundedUpdatePolicyKind kind{BoundedUpdatePolicyKind::None};
    StateAdmissibilityDescriptor admissibility{};
    BoundedUpdateSamplesCallback pre_update_callback{};
    BoundedUpdateSamplesCallback post_update_callback{};
    BoundedUpdateCandidateFilter candidate_state_filter{};
    StateAdmissibilityResidualCallback admissibility_residual_callback{};
};

struct BoundedUpdateResult {
    BoundedUpdateDecision decision{BoundedUpdateDecision::Accepted};
    bool modified{false};
    StateAdmissibilityResult before;
    StateAdmissibilityResult after;
    std::string message;
};

class BoundedUpdateController {
public:
    static BoundedUpdateResult apply(BoundedUpdatePolicy policy,
                                     std::vector<StateAdmissibilitySample>& samples);

    static BoundedUpdateResult applyToSpan(BoundedUpdatePolicy policy,
                                           std::span<Real> values,
                                           std::string label = {});

private:
    static bool clampSamples(const StateAdmissibilityDescriptor& descriptor,
                             std::vector<StateAdmissibilitySample>& samples);
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_BOUNDEDUPDATEPOLICY_H
