#ifndef SVMP_FE_SYSTEMS_MESHDISPLACEMENTBINDING_H
#define SVMP_FE_SYSTEMS_MESHDISPLACEMENTBINDING_H

#include "Core/Types.h"

#include <memory>
#include <string>

namespace svmp::FE {
namespace spaces {
class FunctionSpace;
}

namespace systems {

class FESystem;

struct MeshDisplacementBindingOptions {
    bool enabled{true};
    int dimension{0};
    std::string field_name{"mesh_displacement"};
    std::shared_ptr<const spaces::FunctionSpace> space{};
    bool auto_register_field{true};
    bool bind_as_mesh_displacement{true};
};

struct MeshDisplacementBinding {
    bool enabled{false};
    FieldId displacement_field{INVALID_FIELD_ID};
    std::shared_ptr<const spaces::FunctionSpace> space{};
};

[[nodiscard]] MeshDisplacementBinding resolveMeshDisplacementBinding(
    FESystem& system, MeshDisplacementBindingOptions options);

} // namespace systems
} // namespace svmp::FE

#endif // SVMP_FE_SYSTEMS_MESHDISPLACEMENTBINDING_H
