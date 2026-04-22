#ifndef SVMP_FE_POSTPROCESSING_DERIVED_RESULT_EVALUATOR_H
#define SVMP_FE_POSTPROCESSING_DERIVED_RESULT_EVALUATOR_H

/**
 * @file DerivedResultEvaluator.h
 * @brief Evaluator for FormExpr-based derived result fields.
 */

#include "PostProcessing/DerivedResultOutput.h"
#include "PostProcessing/DerivedResultRegistry.h"
#include "Systems/SystemState.h"

namespace svmp {

class MeshBase;

namespace FE {

namespace systems {
class FESystem;
}

namespace post {

class DerivedResultEvaluator {
public:
    DerivedResultEvaluator(const systems::FESystem& system,
                           const systems::SystemStateView& state);

    void evaluateToMeshField(MeshBase& mesh,
                             const DerivedResultDefinition& def,
                             const DerivedResultOutputOptions& options = {}) const;

private:
    const systems::FESystem& system_;
    const systems::SystemStateView& state_;
};

} // namespace post
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_POSTPROCESSING_DERIVED_RESULT_EVALUATOR_H
