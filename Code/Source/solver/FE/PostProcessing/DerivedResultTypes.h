#ifndef SVMP_FE_POSTPROCESSING_DERIVED_RESULT_TYPES_H
#define SVMP_FE_POSTPROCESSING_DERIVED_RESULT_TYPES_H

/**
 * @file DerivedResultTypes.h
 * @brief Physics-agnostic metadata for FormExpr-based derived output fields.
 */

#include "Core/Types.h"
#include "Forms/FormExpr.h"
#include "Mesh/Core/MeshTypes.h"
#include "Systems/FEQuantityDefinition.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace post {

enum class DerivedResultScope : std::uint8_t {
    Vertex,
    Edge,
    Face,
    BoundaryFace,
    Cell,
    QuadraturePoint
};

enum class DerivedResultPolicy : std::uint8_t {
    PointValue,
    CellCentroid,
    CellAverage,
    FaceCentroid,
    FaceAverage,
    PatchAverage,
    L2Projection,
    EdgeAverage,
    QuadratureValue,
    ProjectToCell,
    ProjectToVertex
};

struct DerivedResultDefinition {
    std::string name{};
    DerivedResultScope scope{DerivedResultScope::Cell};
    DerivedResultPolicy policy{DerivedResultPolicy::CellAverage};
    systems::FEQuantityShape shape{systems::FEQuantityShape::scalar()};
    forms::FormExpr expression{};
    std::vector<FieldId> referenced_fields{};
    std::optional<int> marker{};
    bool enabled{true};
};

[[nodiscard]] std::string_view toString(DerivedResultScope scope) noexcept;
[[nodiscard]] std::string_view toString(DerivedResultPolicy policy) noexcept;
[[nodiscard]] std::size_t componentCount(const systems::FEQuantityShape& shape) noexcept;
[[nodiscard]] bool hasDirectMeshField(DerivedResultScope scope) noexcept;
[[nodiscard]] std::optional<EntityKind> meshEntityKind(DerivedResultScope scope) noexcept;

} // namespace post
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_POSTPROCESSING_DERIVED_RESULT_TYPES_H
