#include "Systems/MeshDisplacementBinding.h"

#include "Core/FEException.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"
#include "Systems/FieldRegistry.h"

#include <stdexcept>
#include <utility>

namespace svmp::FE::systems {
namespace {

void validateVectorSpace(const std::shared_ptr<const spaces::FunctionSpace>& space,
                         int dimension,
                         const char* label)
{
    if (!space) {
        throw std::invalid_argument(std::string("resolveMeshDisplacementBinding: null ") +
                                    label + " space");
    }
    if (space->field_type() != FieldType::Vector) {
        throw std::invalid_argument(std::string("resolveMeshDisplacementBinding: ") +
                                    label + " space must be vector-valued");
    }
    if (space->value_dimension() != dimension) {
        throw std::invalid_argument(std::string("resolveMeshDisplacementBinding: ") +
                                    label + " space dimension must match mesh dimension");
    }
}

FieldId findBoundOrNamedDisplacement(const FESystem& system, const std::string& field_name)
{
    if (const auto bound = system.meshMotionField(MeshMotionFieldRole::Displacement)) {
        return *bound;
    }
    return system.findFieldByName(field_name);
}

void validateDisplacementRecord(const FieldRecord& record, int dimension)
{
    if (record.source_kind != FieldSourceKind::Unknown) {
        throw std::invalid_argument(
            "resolveMeshDisplacementBinding: mesh displacement field '" + record.name +
            "' must be an Unknown");
    }
    validateVectorSpace(record.space, dimension, "mesh displacement");
    if (record.components != dimension) {
        throw std::invalid_argument(
            "resolveMeshDisplacementBinding: mesh displacement field '" + record.name +
            "' component count must match mesh dimension");
    }
}

} // namespace

MeshDisplacementBinding resolveMeshDisplacementBinding(FESystem& system,
                                                       MeshDisplacementBindingOptions options)
{
    if (!options.enabled) {
        return {};
    }
    if (options.dimension <= 0 || options.dimension > 3) {
        throw std::invalid_argument(
            "resolveMeshDisplacementBinding: mesh dimension must be in [1, 3]");
    }
    if (options.field_name.empty()) {
        throw std::invalid_argument(
            "resolveMeshDisplacementBinding: mesh displacement field name cannot be empty");
    }
    if (options.space) {
        validateVectorSpace(options.space, options.dimension, "mesh displacement");
    }

    auto displacement_id = findBoundOrNamedDisplacement(system, options.field_name);
    if (displacement_id == INVALID_FIELD_ID) {
        if (!options.auto_register_field) {
            throw std::invalid_argument(
                "resolveMeshDisplacementBinding: mesh displacement field '" +
                options.field_name + "' is not registered");
        }
        validateVectorSpace(options.space, options.dimension, "mesh displacement");

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
        displacement_id = system.addMeshDisplacementUnknown(options.field_name,
                                                            std::move(options.space),
                                                            options.dimension);
#else
        FieldSpec spec;
        spec.name = options.field_name;
        spec.space = std::move(options.space);
        spec.components = options.dimension;
        displacement_id = system.addField(std::move(spec));
#endif
    }

    const auto& record = system.fieldRecord(displacement_id);
    validateDisplacementRecord(record, options.dimension);

    if (options.bind_as_mesh_displacement) {
        system.bindMeshMotionField(MeshMotionFieldRole::Displacement, displacement_id);
    }

    return MeshDisplacementBinding{true, displacement_id, record.space};
}

} // namespace svmp::FE::systems
