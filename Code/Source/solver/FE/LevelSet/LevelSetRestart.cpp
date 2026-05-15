#include "LevelSet/LevelSetRestart.h"

#include <stdexcept>
#include <utility>

namespace svmp::FE::level_set {
namespace {

[[nodiscard]] FieldId resolveScalarField(
    const systems::FESystem& system,
    const std::string& field_name,
    const std::string& context)
{
    if (field_name.empty()) {
        throw std::invalid_argument(context + " requires a non-empty field name");
    }
    const auto field = system.findFieldByName(field_name);
    if (field == INVALID_FIELD_ID) {
        throw std::invalid_argument(context + " references unknown field '" + field_name + "'");
    }
    const auto& rec = system.fieldRecord(field);
    if (rec.components != 1 || !rec.space || rec.space->value_dimension() != 1) {
        throw std::invalid_argument(context + " field '" + field_name + "' must be scalar");
    }
    return field;
}

void setDiagnostic(std::string* diagnostic, std::string value)
{
    if (diagnostic != nullptr) {
        *diagnostic = std::move(value);
    }
}

} // namespace

LevelSetFieldRestartRecord captureLevelSetFieldRestartRecord(
    const systems::FESystem& system,
    const LevelSetTransportOptions& options,
    std::uint64_t value_revision)
{
    const auto field = resolveScalarField(
        system,
        options.level_set.field_name,
        "level-set restart field");
    const auto& rec = system.fieldRecord(field);
    const auto& dofs = system.fieldDofHandler(field);

    LevelSetFieldRestartRecord record;
    record.field_name = options.level_set.field_name;
    record.field_id = field;
    record.source = options.level_set.source;
    record.auto_register_field = options.level_set.auto_register_field;
    record.components = rec.components;
    record.dof_offset = system.fieldDofOffset(field);
    record.dof_count = dofs.getNumDofs();
    record.value_revision = value_revision;
    return record;
}

LevelSetGeneratedInterfaceRestartRecord
captureLevelSetGeneratedInterfaceRestartRecord(
    const systems::FESystem& system,
    const LevelSetGeneratedInterfaceOptions& options,
    const LevelSetGeneratedInterfaceResult& result)
{
    if (result.interface_marker < 0) {
        throw std::invalid_argument(
            "generated-interface restart record requires a valid interface marker");
    }
    const auto field = resolveScalarField(
        system,
        options.level_set_field_name,
        "generated-interface restart record");

    LevelSetGeneratedInterfaceRestartRecord record;
    record.level_set_field_name = options.level_set_field_name;
    record.level_set_field_id = field;
    record.domain_id = options.domain_id;
    record.requested_interface_marker = options.requested_interface_marker;
    record.interface_marker = result.interface_marker;
    record.isovalue = options.isovalue;
    record.tolerance = options.tolerance;
    record.quadrature_order = options.quadrature_order;
    record.interface_quadrature_order = options.interface_quadrature_order;
    record.volume_quadrature_order = options.volume_quadrature_order;
    record.keep_degenerate_fragments = options.keep_degenerate_fragments;
    record.value_revision = result.value_revision;
    record.mesh_geometry_revision = result.domain.request().mesh_geometry_revision;
    record.mesh_topology_revision = result.domain.request().mesh_topology_revision;
    record.ownership_revision = result.domain.request().ownership_revision;
    record.summary = result.summary;
    return record;
}

LevelSetGeneratedInterfaceOptions
optionsFromLevelSetGeneratedInterfaceRestartRecord(
    const LevelSetGeneratedInterfaceRestartRecord& record)
{
    if (record.interface_marker < 0) {
        throw std::invalid_argument(
            "generated-interface restart record cannot restore an invalid marker");
    }
    LevelSetGeneratedInterfaceOptions options;
    options.level_set_field_name = record.level_set_field_name;
    options.domain_id = record.domain_id;
    options.requested_interface_marker = record.interface_marker;
    options.isovalue = record.isovalue;
    options.tolerance = record.tolerance;
    options.quadrature_order = record.quadrature_order;
    options.interface_quadrature_order = record.interface_quadrature_order;
    options.volume_quadrature_order = record.volume_quadrature_order;
    options.keep_degenerate_fragments = record.keep_degenerate_fragments;
    return options;
}

bool levelSetGeneratedInterfaceRestartRecordMatches(
    const systems::FESystem& system,
    const LevelSetGeneratedInterfaceRestartRecord& record,
    std::string* diagnostic)
{
    const auto field = system.findFieldByName(record.level_set_field_name);
    if (field == INVALID_FIELD_ID) {
        setDiagnostic(diagnostic, "level-set restart field is not registered");
        return false;
    }
    const auto& rec = system.fieldRecord(field);
    if (rec.components != 1 || !rec.space || rec.space->value_dimension() != 1) {
        setDiagnostic(diagnostic, "level-set restart field is not scalar");
        return false;
    }
    if (record.interface_marker < 0) {
        setDiagnostic(diagnostic, "generated-interface restart marker is invalid");
        return false;
    }
    if (record.domain_id.empty()) {
        setDiagnostic(diagnostic, "generated-interface restart domain id is empty");
        return false;
    }
    if (!(record.tolerance > 0.0)) {
        setDiagnostic(diagnostic, "generated-interface restart tolerance is invalid");
        return false;
    }
    if (record.quadrature_order < 0) {
        setDiagnostic(diagnostic, "generated-interface restart quadrature order is invalid");
        return false;
    }
    if (record.interface_quadrature_order < -1) {
        setDiagnostic(diagnostic, "generated-interface restart interface quadrature order is invalid");
        return false;
    }
    if (record.volume_quadrature_order < -1) {
        setDiagnostic(diagnostic, "generated-interface restart volume quadrature order is invalid");
        return false;
    }

    const auto& mesh = system.meshAccess();
    if (record.mesh_topology_revision != 0u &&
        record.mesh_topology_revision != mesh.topologyRevision()) {
        setDiagnostic(diagnostic, "generated-interface restart topology revision differs");
        return false;
    }
    if (record.mesh_geometry_revision != 0u &&
        record.mesh_geometry_revision != mesh.geometryRevision()) {
        setDiagnostic(diagnostic, "generated-interface restart geometry revision differs");
        return false;
    }
    if (record.ownership_revision != 0u &&
        record.ownership_revision != mesh.ownershipRevision()) {
        setDiagnostic(diagnostic, "generated-interface restart ownership revision differs");
        return false;
    }

    setDiagnostic(diagnostic, {});
    return true;
}

} // namespace svmp::FE::level_set
