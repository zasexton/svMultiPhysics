#ifndef SVMP_FE_POSTPROCESSING_DERIVED_RESULT_EVALUATOR_H
#define SVMP_FE_POSTPROCESSING_DERIVED_RESULT_EVALUATOR_H

/**
 * @file DerivedResultEvaluator.h
 * @brief Evaluator for FormExpr-based derived result fields.
 */

#include "PostProcessing/DerivedResultOutput.h"
#include "PostProcessing/DerivedResultRegistry.h"
#include "Systems/SystemState.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
    struct CellAverageCache {
        std::size_t components{0};
        std::vector<double> values{};
    };

    const systems::FESystem& system_;
    const systems::SystemStateView& state_;
    mutable std::unordered_set<std::string> reusable_cell_average_keys_{};
    mutable std::unordered_map<std::string, CellAverageCache> cell_average_cache_{};
};

} // namespace post
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_POSTPROCESSING_DERIVED_RESULT_EVALUATOR_H
