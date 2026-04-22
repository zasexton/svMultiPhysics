#include "PostProcessing/DerivedResultOutput.h"

#include "Core/FEException.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Core/MeshTypes.h"
#endif

#include <algorithm>

namespace svmp {
namespace FE {
namespace post {

FieldHandle ensureDerivedResultField(MeshBase& mesh,
                                     EntityKind kind,
                                     std::string_view name,
                                     std::size_t components,
                                     DerivedResultOverwritePolicy overwrite)
{
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    FE_THROW_IF(components == 0, InvalidArgumentException,
                "ensureDerivedResultField: components must be positive");

    const std::string field_name(name);
    if (mesh.has_field(kind, field_name)) {
        auto handle = mesh.field_handle(kind, field_name);
        const bool compatible = mesh.field_type(handle) == FieldScalarType::Float64 &&
                                mesh.field_components(handle) == components;
        if (compatible) {
            return handle;
        }

        FE_THROW_IF(overwrite == DerivedResultOverwritePolicy::Reject ||
                        overwrite == DerivedResultOverwritePolicy::ReplaceCompatible,
                    InvalidArgumentException,
                    "Derived result field '" + field_name +
                        "' already exists with incompatible type or component count");

        mesh.remove_field(handle);
    }

    return mesh.attach_field(kind, field_name, FieldScalarType::Float64, components);
#else
    (void)mesh;
    (void)kind;
    (void)name;
    (void)components;
    (void)overwrite;
    FE_THROW(NotImplementedException,
             "Derived result mesh-field output requires FE_WITH_MESH");
#endif
}

std::span<double> derivedResultFieldData(MeshBase& mesh,
                                         const FieldHandle& handle,
                                         std::size_t components)
{
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    FE_THROW_IF(components == 0, InvalidArgumentException,
                "derivedResultFieldData: components must be positive");
    auto* data = static_cast<double*>(mesh.field_data(handle));
    FE_THROW_IF(data == nullptr, InvalidArgumentException,
                "derivedResultFieldData: null field data for '" + handle.name + "'");

    const auto n_values = mesh.field_entity_count(handle) * components;
    auto out = std::span<double>(data, n_values);
    std::fill(out.begin(), out.end(), 0.0);
    return out;
#else
    (void)mesh;
    (void)handle;
    (void)components;
    FE_THROW(NotImplementedException,
             "Derived result mesh-field output requires FE_WITH_MESH");
#endif
}

} // namespace post
} // namespace FE
} // namespace svmp
