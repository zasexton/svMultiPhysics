#ifndef SVMP_FE_POSTPROCESSING_DERIVED_RESULT_OUTPUT_H
#define SVMP_FE_POSTPROCESSING_DERIVED_RESULT_OUTPUT_H

/**
 * @file DerivedResultOutput.h
 * @brief Mesh-field output helpers for derived result evaluation.
 */

#include "PostProcessing/DerivedResultTypes.h"

#include <cstddef>
#include <span>
#include <string_view>

namespace svmp {

class MeshBase;

namespace FE {
namespace post {

enum class DerivedResultOverwritePolicy : std::uint8_t {
    ReplaceCompatible,
    ReplaceAny,
    Reject
};

struct DerivedResultOutputOptions {
    DerivedResultOverwritePolicy overwrite{DerivedResultOverwritePolicy::ReplaceCompatible};
};

[[nodiscard]] FieldHandle ensureDerivedResultField(
    MeshBase& mesh,
    EntityKind kind,
    std::string_view name,
    std::size_t components,
    DerivedResultOverwritePolicy overwrite);

[[nodiscard]] std::span<double> derivedResultFieldData(MeshBase& mesh,
                                                       const FieldHandle& handle,
                                                       std::size_t components);

} // namespace post
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_POSTPROCESSING_DERIVED_RESULT_OUTPUT_H
