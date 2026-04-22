#include "PostProcessing/DerivedResultTypes.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Mesh/Core/MeshTypes.h"
#endif

namespace svmp {
namespace FE {
namespace post {

std::string_view toString(DerivedResultScope scope) noexcept
{
    switch (scope) {
    case DerivedResultScope::Vertex: return "Vertex";
    case DerivedResultScope::Edge: return "Edge";
    case DerivedResultScope::Face: return "Face";
    case DerivedResultScope::BoundaryFace: return "BoundaryFace";
    case DerivedResultScope::Cell: return "Cell";
    case DerivedResultScope::QuadraturePoint: return "QuadraturePoint";
    }
    return "Unknown";
}

std::string_view toString(DerivedResultPolicy policy) noexcept
{
    switch (policy) {
    case DerivedResultPolicy::PointValue: return "PointValue";
    case DerivedResultPolicy::CellCentroid: return "CellCentroid";
    case DerivedResultPolicy::CellAverage: return "CellAverage";
    case DerivedResultPolicy::FaceCentroid: return "FaceCentroid";
    case DerivedResultPolicy::FaceAverage: return "FaceAverage";
    case DerivedResultPolicy::PatchAverage: return "PatchAverage";
    case DerivedResultPolicy::L2Projection: return "L2Projection";
    case DerivedResultPolicy::EdgeAverage: return "EdgeAverage";
    case DerivedResultPolicy::QuadratureValue: return "QuadratureValue";
    case DerivedResultPolicy::ProjectToCell: return "ProjectToCell";
    case DerivedResultPolicy::ProjectToVertex: return "ProjectToVertex";
    }
    return "Unknown";
}

std::size_t componentCount(const systems::FEQuantityShape& shape) noexcept
{
    return static_cast<std::size_t>(shape.components > 0 ? shape.components : 1);
}

bool hasDirectMeshField(DerivedResultScope scope) noexcept
{
    return scope == DerivedResultScope::Vertex ||
           scope == DerivedResultScope::Edge ||
           scope == DerivedResultScope::Face ||
           scope == DerivedResultScope::BoundaryFace ||
           scope == DerivedResultScope::Cell;
}

std::optional<EntityKind> meshEntityKind(DerivedResultScope scope) noexcept
{
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    switch (scope) {
    case DerivedResultScope::Vertex: return EntityKind::Vertex;
    case DerivedResultScope::Edge: return EntityKind::Edge;
    case DerivedResultScope::Face:
    case DerivedResultScope::BoundaryFace:
        return EntityKind::Face;
    case DerivedResultScope::Cell: return EntityKind::Volume;
    case DerivedResultScope::QuadraturePoint:
        break;
    }
#else
    (void)scope;
#endif
    return std::nullopt;
}

} // namespace post
} // namespace FE
} // namespace svmp
