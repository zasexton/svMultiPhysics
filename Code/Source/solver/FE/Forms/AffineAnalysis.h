#ifndef SVMP_FE_FORMS_AFFINE_ANALYSIS_H
#define SVMP_FE_FORMS_AFFINE_ANALYSIS_H

/**
 * @file AffineAnalysis.h
 * @brief Utilities to validate/split residual forms that are affine in the TrialFunction
 */

#include "Forms/FormExpr.h"

#include <optional>
#include <string>

namespace svmp {
namespace FE {
namespace forms {

struct AffineResidualSplit {
    // a(u,v): bilinear form (must contain both test and trial functions)
    FormExpr bilinear{};

    // L(v): linear (trial-independent) form; may be invalid if there is no constant part
    FormExpr linear{};
};

struct AffineResidualOptions {
    // Conservative default: transient `dt(...)` terms are not yet lowered for LinearFormKernel.
    bool allow_time_derivatives{false};

    // Conservative default: DG interior-face residual lowering is not yet handled by LinearFormKernel.
    bool allow_interior_face_terms{false};
};

/**
 * @brief Attempt to split a residual form F(u;v) into affine parts a(u,v) + L(v)
 *
 * Returns std::nullopt if the residual is not provably affine in the active TrialFunction
 * (or uses constructs not supported by the affine splitter).
 *
 * When provided, `reason_out` is populated with a short diagnostic on failure.
 */
[[nodiscard]] std::optional<AffineResidualSplit> trySplitAffineResidual(
    const FormExpr& residual_form,
    const AffineResidualOptions& options = {},
    std::string* reason_out = nullptr);

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_AFFINE_ANALYSIS_H

