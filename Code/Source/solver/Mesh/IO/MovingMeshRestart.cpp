/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "MovingMeshRestart.h"

#include "../Motion/MotionFields.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace svmp {
namespace moving_mesh_restart {
namespace {

constexpr const char* kMagic = "SVMP_MOVING_MESH_RESTART";

std::string configuration_to_string(Configuration cfg)
{
  switch (cfg) {
    case Configuration::Reference: return "reference";
    case Configuration::Current: return "current";
    case Configuration::Deformed: return "current";
  }
  return "reference";
}

Configuration configuration_from_string(const std::string& value)
{
  if (value == "reference" || value == "Reference") return Configuration::Reference;
  if (value == "current" || value == "Current" || value == "deformed" || value == "Deformed") {
    return Configuration::Current;
  }
  throw std::runtime_error("moving mesh restart: unknown active configuration '" + value + "'");
}

EntityKind entity_kind_from_int(int value)
{
  if (value < static_cast<int>(EntityKind::Vertex) || value > static_cast<int>(EntityKind::Volume)) {
    throw std::runtime_error("moving mesh restart: invalid entity kind");
  }
  return static_cast<EntityKind>(value);
}

FieldScalarType field_scalar_type_from_int(int value)
{
  if (value < static_cast<int>(FieldScalarType::Int32) || value > static_cast<int>(FieldScalarType::Custom)) {
    throw std::runtime_error("moving mesh restart: invalid field scalar type");
  }
  return static_cast<FieldScalarType>(value);
}

FieldIntent field_intent_from_int(int value)
{
  if (value < static_cast<int>(FieldIntent::ReadOnly) || value > static_cast<int>(FieldIntent::Temporary)) {
    throw std::runtime_error("moving mesh restart: invalid field intent");
  }
  return static_cast<FieldIntent>(value);
}

FieldGhostPolicy ghost_policy_from_int(int value)
{
  if (value < static_cast<int>(FieldGhostPolicy::None) || value > static_cast<int>(FieldGhostPolicy::Accumulate)) {
    throw std::runtime_error("moving mesh restart: invalid field ghost policy");
  }
  return static_cast<FieldGhostPolicy>(value);
}

void expect_tag(std::istream& in, const char* expected)
{
  std::string tag;
  if (!(in >> tag)) {
    throw std::runtime_error(std::string("moving mesh restart: expected tag '") + expected + "' before EOF");
  }
  if (tag != expected) {
    throw std::runtime_error("moving mesh restart: expected tag '" + std::string(expected) +
                             "', found '" + tag + "'");
  }
}

template <typename T>
void write_vector(std::ostream& out, const char* tag, const std::vector<T>& values)
{
  out << tag << ' ' << values.size();
  for (const auto& value : values) {
    out << ' ' << value;
  }
  out << '\n';
}

template <typename T>
std::vector<T> read_vector(std::istream& in, const char* tag)
{
  expect_tag(in, tag);
  std::size_t count = 0;
  in >> count;
  std::vector<T> values(count);
  for (auto& value : values) {
    in >> value;
  }
  if (!in) {
    throw std::runtime_error(std::string("moving mesh restart: malformed vector block '") + tag + "'");
  }
  return values;
}

void write_shapes(std::ostream& out, const char* tag, const std::vector<CellShape>& shapes)
{
  out << tag << ' ' << shapes.size() << '\n';
  for (const auto& shape : shapes) {
    out << static_cast<int>(shape.family) << ' '
        << shape.num_corners << ' '
        << shape.order << ' '
        << (shape.is_mixed_order ? 1 : 0) << ' '
        << shape.num_faces_hint << ' '
        << shape.num_edges_hint << '\n';
  }
}

std::vector<CellShape> read_shapes(std::istream& in, const char* tag)
{
  expect_tag(in, tag);
  std::size_t count = 0;
  in >> count;
  std::vector<CellShape> shapes(count);
  for (auto& shape : shapes) {
    int family = 0;
    int mixed = 0;
    in >> family
       >> shape.num_corners
       >> shape.order
       >> mixed
       >> shape.num_faces_hint
       >> shape.num_edges_hint;
    shape.family = static_cast<CellFamily>(family);
    shape.is_mixed_order = (mixed != 0);
  }
  if (!in) {
    throw std::runtime_error(std::string("moving mesh restart: malformed shape block '") + tag + "'");
  }
  return shapes;
}

void write_pair_vector(std::ostream& out,
                       const char* tag,
                       const std::vector<std::array<index_t, 2>>& values)
{
  out << tag << ' ' << values.size() << '\n';
  for (const auto& value : values) {
    out << value[0] << ' ' << value[1] << '\n';
  }
}

std::vector<std::array<index_t, 2>> read_pair_vector(std::istream& in, const char* tag)
{
  expect_tag(in, tag);
  std::size_t count = 0;
  in >> count;
  std::vector<std::array<index_t, 2>> values(count);
  for (auto& value : values) {
    in >> value[0] >> value[1];
  }
  if (!in) {
    throw std::runtime_error(std::string("moving mesh restart: malformed pair block '") + tag + "'");
  }
  return values;
}

void write_revision_state(std::ostream& out, const MeshRevisionState& revisions)
{
  out << "revision_state "
      << revisions.geometry << ' '
      << revisions.reference_geometry << ' '
      << revisions.current_geometry << ' '
      << revisions.topology << ' '
      << revisions.ownership << ' '
      << revisions.numbering << ' '
      << revisions.field_layout << ' '
      << revisions.labels << ' '
      << revisions.active_configuration << '\n';
}

MeshRevisionState read_revision_state(std::istream& in)
{
  expect_tag(in, "revision_state");
  MeshRevisionState revisions{};
  in >> revisions.geometry
     >> revisions.reference_geometry
     >> revisions.current_geometry
     >> revisions.topology
     >> revisions.ownership
     >> revisions.numbering
     >> revisions.field_layout
     >> revisions.labels
     >> revisions.active_configuration;
  if (!in) {
    throw std::runtime_error("moving mesh restart: malformed revision_state block");
  }
  return revisions;
}

void write_label_registry(std::ostream& out, const MeshBase& mesh)
{
  const auto labels = mesh.list_label_names();
  std::vector<std::pair<label_t, std::string>> sorted(labels.begin(), labels.end());
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });

  out << "label_registry " << sorted.size() << '\n';
  for (const auto& [label, name] : sorted) {
    out << label << ' ' << std::quoted(name) << '\n';
  }
}

void read_label_registry(std::istream& in, MeshBase& mesh)
{
  expect_tag(in, "label_registry");
  std::size_t count = 0;
  in >> count;
  for (std::size_t i = 0; i < count; ++i) {
    label_t label = INVALID_LABEL;
    std::string name;
    in >> label >> std::quoted(name);
    mesh.register_label(name, label);
  }
  if (!in) {
    throw std::runtime_error("moving mesh restart: malformed label_registry block");
  }
}

void write_sets(std::ostream& out, const MeshBase& mesh)
{
  struct SetRecord {
    EntityKind kind;
    std::string name;
    std::vector<index_t> ids;
  };

  std::vector<SetRecord> records;
  for (const auto kind : {EntityKind::Vertex, EntityKind::Edge, EntityKind::Face, EntityKind::Volume}) {
    auto names = mesh.list_sets(kind);
    std::sort(names.begin(), names.end());
    for (const auto& name : names) {
      records.push_back(SetRecord{kind, name, mesh.get_set(kind, name)});
    }
  }

  out << "entity_sets " << records.size() << '\n';
  for (const auto& record : records) {
    out << static_cast<int>(record.kind) << ' ' << std::quoted(record.name) << ' '
        << record.ids.size();
    for (const auto id : record.ids) {
      out << ' ' << id;
    }
    out << '\n';
  }
}

void read_sets(std::istream& in, MeshBase& mesh)
{
  expect_tag(in, "entity_sets");
  std::size_t count = 0;
  in >> count;
  for (std::size_t i = 0; i < count; ++i) {
    int kind_value = 0;
    std::string name;
    std::size_t n_ids = 0;
    in >> kind_value >> std::quoted(name) >> n_ids;
    const auto kind = entity_kind_from_int(kind_value);
    for (std::size_t j = 0; j < n_ids; ++j) {
      index_t id = INVALID_INDEX;
      in >> id;
      mesh.add_to_set(kind, name, id);
    }
  }
  if (!in) {
    throw std::runtime_error("moving mesh restart: malformed entity_sets block");
  }
}

void write_metadata_maps(std::ostream& out, const WriteOptions& options)
{
  out << "motion_backend_state " << options.motion_backend_state.size() << '\n';
  for (const auto& [key, value] : options.motion_backend_state) {
    out << std::quoted(key) << ' ' << std::quoted(value) << '\n';
  }

  out << "adaptivity_provenance " << options.adaptivity_provenance.size() << '\n';
  for (const auto& value : options.adaptivity_provenance) {
    out << std::quoted(value) << '\n';
  }
}

void read_metadata_maps(std::istream& in, Metadata& metadata)
{
  expect_tag(in, "motion_backend_state");
  std::size_t count = 0;
  in >> count;
  for (std::size_t i = 0; i < count; ++i) {
    std::string key;
    std::string value;
    in >> std::quoted(key) >> std::quoted(value);
    metadata.motion_backend_state.emplace(std::move(key), std::move(value));
  }

  expect_tag(in, "adaptivity_provenance");
  in >> count;
  metadata.adaptivity_provenance.resize(count);
  for (auto& value : metadata.adaptivity_provenance) {
    in >> std::quoted(value);
  }

  if (!in) {
    throw std::runtime_error("moving mesh restart: malformed restart metadata block");
  }
}

void write_descriptor(std::ostream& out, const FieldDescriptor* descriptor)
{
  if (!descriptor) {
    out << "field_descriptor 0\n";
    return;
  }

  out << "field_descriptor 1 "
      << static_cast<int>(descriptor->location) << ' '
      << descriptor->components << ' '
      << std::quoted(descriptor->units) << ' '
      << descriptor->unit_scale << ' '
      << (descriptor->time_dependent ? 1 : 0) << ' '
      << static_cast<int>(descriptor->intent) << ' '
      << static_cast<int>(descriptor->ghost_policy) << ' '
      << std::quoted(descriptor->description) << ' '
      << descriptor->component_names.size();
  for (const auto& name : descriptor->component_names) {
    out << ' ' << std::quoted(name);
  }
  out << '\n';
}

bool read_descriptor(std::istream& in, FieldDescriptor& descriptor)
{
  expect_tag(in, "field_descriptor");
  int has_descriptor = 0;
  in >> has_descriptor;
  if (!has_descriptor) {
    return false;
  }

  int location = 0;
  int time_dependent = 0;
  int intent = 0;
  int ghost_policy = 0;
  std::size_t n_component_names = 0;

  in >> location
     >> descriptor.components
     >> std::quoted(descriptor.units)
     >> descriptor.unit_scale
     >> time_dependent
     >> intent
     >> ghost_policy
     >> std::quoted(descriptor.description)
     >> n_component_names;

  descriptor.location = entity_kind_from_int(location);
  descriptor.time_dependent = (time_dependent != 0);
  descriptor.intent = field_intent_from_int(intent);
  descriptor.ghost_policy = ghost_policy_from_int(ghost_policy);
  descriptor.component_names.resize(n_component_names);
  for (auto& name : descriptor.component_names) {
    in >> std::quoted(name);
  }

  if (!in) {
    throw std::runtime_error("moving mesh restart: malformed field descriptor");
  }
  return true;
}

template <typename T>
void write_typed_field_values(std::ostream& out, const void* raw, std::size_t count)
{
  const auto* values = static_cast<const T*>(raw);
  for (std::size_t i = 0; i < count; ++i) {
    out << ' ' << values[i];
  }
}

template <>
void write_typed_field_values<std::uint8_t>(std::ostream& out, const void* raw, std::size_t count)
{
  const auto* values = static_cast<const std::uint8_t*>(raw);
  for (std::size_t i = 0; i < count; ++i) {
    out << ' ' << static_cast<unsigned int>(values[i]);
  }
}

void write_field_values(std::ostream& out,
                        FieldScalarType type,
                        const void* raw,
                        std::size_t n_values,
                        std::size_t n_bytes)
{
  out << "field_values " << n_values;
  switch (type) {
    case FieldScalarType::Int32:
      write_typed_field_values<std::int32_t>(out, raw, n_values);
      break;
    case FieldScalarType::Int64:
      write_typed_field_values<std::int64_t>(out, raw, n_values);
      break;
    case FieldScalarType::Float32:
      write_typed_field_values<float>(out, raw, n_values);
      break;
    case FieldScalarType::Float64:
      write_typed_field_values<double>(out, raw, n_values);
      break;
    case FieldScalarType::UInt8:
      write_typed_field_values<std::uint8_t>(out, raw, n_values);
      break;
    case FieldScalarType::Custom: {
      const auto* bytes = static_cast<const std::uint8_t*>(raw);
      out << " bytes " << n_bytes;
      for (std::size_t i = 0; i < n_bytes; ++i) {
        out << ' ' << static_cast<unsigned int>(bytes[i]);
      }
      break;
    }
  }
  out << '\n';
}

template <typename T>
void read_typed_field_values(std::istream& in, void* raw, std::size_t count)
{
  auto* values = static_cast<T*>(raw);
  for (std::size_t i = 0; i < count; ++i) {
    in >> values[i];
  }
}

template <>
void read_typed_field_values<std::uint8_t>(std::istream& in, void* raw, std::size_t count)
{
  auto* values = static_cast<std::uint8_t*>(raw);
  for (std::size_t i = 0; i < count; ++i) {
    unsigned int value = 0;
    in >> value;
    if (value > std::numeric_limits<std::uint8_t>::max()) {
      throw std::runtime_error("moving mesh restart: UInt8 field value is out of range");
    }
    values[i] = static_cast<std::uint8_t>(value);
  }
}

void read_field_values(std::istream& in,
                       FieldScalarType type,
                       void* raw,
                       std::size_t expected_values,
                       std::size_t expected_bytes)
{
  expect_tag(in, "field_values");
  std::size_t n_values = 0;
  in >> n_values;
  if (n_values != expected_values) {
    throw std::runtime_error("moving mesh restart: field value count does not match mesh entity layout");
  }

  switch (type) {
    case FieldScalarType::Int32:
      read_typed_field_values<std::int32_t>(in, raw, n_values);
      break;
    case FieldScalarType::Int64:
      read_typed_field_values<std::int64_t>(in, raw, n_values);
      break;
    case FieldScalarType::Float32:
      read_typed_field_values<float>(in, raw, n_values);
      break;
    case FieldScalarType::Float64:
      read_typed_field_values<double>(in, raw, n_values);
      break;
    case FieldScalarType::UInt8:
      read_typed_field_values<std::uint8_t>(in, raw, n_values);
      break;
    case FieldScalarType::Custom: {
      std::string bytes_tag;
      std::size_t n_bytes = 0;
      in >> bytes_tag >> n_bytes;
      if (bytes_tag != "bytes" || n_bytes != expected_bytes) {
        throw std::runtime_error("moving mesh restart: custom field byte count mismatch");
      }
      auto* bytes = static_cast<std::uint8_t*>(raw);
      for (std::size_t i = 0; i < n_bytes; ++i) {
        unsigned int value = 0;
        in >> value;
        if (value > std::numeric_limits<std::uint8_t>::max()) {
          throw std::runtime_error("moving mesh restart: custom field byte is out of range");
        }
        bytes[i] = static_cast<std::uint8_t>(value);
      }
      break;
    }
  }

  if (!in) {
    throw std::runtime_error("moving mesh restart: malformed field values");
  }
}

bool should_write_field(const MeshBase& mesh,
                        EntityKind kind,
                        const std::string& name,
                        const WriteOptions& options)
{
  if (!options.include_fields) {
    return false;
  }
  if (options.include_motion_fields) {
    return true;
  }
  return !(kind == EntityKind::Vertex && motion::is_standard_motion_field_name(name));
}

void write_fields(std::ostream& out, const MeshBase& mesh, const WriteOptions& options)
{
  struct FieldRecord {
    EntityKind kind;
    std::string name;
  };

  std::vector<FieldRecord> fields;
  for (const auto kind : {EntityKind::Vertex, EntityKind::Edge, EntityKind::Face, EntityKind::Volume}) {
    auto names = mesh.field_names(kind);
    std::sort(names.begin(), names.end());
    for (const auto& name : names) {
      if (should_write_field(mesh, kind, name, options)) {
        fields.push_back(FieldRecord{kind, name});
      }
    }
  }

  out << "fields " << fields.size() << '\n';
  for (const auto& field : fields) {
    const auto handle = mesh.field_handle(field.kind, field.name);
    const auto type = mesh.field_type(handle);
    const auto components = mesh.field_components(handle);
    const auto entity_count = mesh.field_entity_count(handle);
    const auto bytes_per_entity = mesh.field_bytes_per_entity(handle);
    const auto bytes_per_component = components == 0 ? 0 : bytes_per_entity / components;
    const auto n_values = entity_count * components;
    const auto n_bytes = entity_count * bytes_per_entity;

    out << "field "
        << static_cast<int>(field.kind) << ' '
        << std::quoted(field.name) << ' '
        << static_cast<int>(type) << ' '
        << components << ' '
        << bytes_per_component << ' '
        << entity_count << '\n';
    write_descriptor(out, mesh.field_descriptor(handle));
    write_field_values(out, type, mesh.field_data(handle), n_values, n_bytes);
  }
}

void read_fields(std::istream& in, MeshBase& mesh)
{
  expect_tag(in, "fields");
  std::size_t count = 0;
  in >> count;
  for (std::size_t i = 0; i < count; ++i) {
    expect_tag(in, "field");
    int kind_value = 0;
    int type_value = 0;
    std::string name;
    std::size_t components = 0;
    std::size_t bytes_per_component = 0;
    std::size_t entity_count = 0;

    in >> kind_value
       >> std::quoted(name)
       >> type_value
       >> components
       >> bytes_per_component
       >> entity_count;

    const auto kind = entity_kind_from_int(kind_value);
    const auto type = field_scalar_type_from_int(type_value);

    FieldDescriptor descriptor;
    const bool has_descriptor = read_descriptor(in, descriptor);

    FieldHandle handle;
    if (has_descriptor) {
      if (descriptor.location != kind) {
        throw std::runtime_error("moving mesh restart: field descriptor location mismatch");
      }
      if (descriptor.components != components) {
        throw std::runtime_error("moving mesh restart: field descriptor component mismatch");
      }
      handle = mesh.attach_field(kind, name, type, components,
                                 type == FieldScalarType::Custom ? bytes_per_component : 0);
      mesh.set_field_descriptor(handle, descriptor);
    } else {
      handle = mesh.attach_field(kind, name, type, components,
                                 type == FieldScalarType::Custom ? bytes_per_component : 0);
      if (kind == EntityKind::Vertex && type == FieldScalarType::Float64 &&
          components == static_cast<std::size_t>(mesh.dim()) &&
          motion::is_standard_motion_field_name(name)) {
        const auto role = motion::parse_motion_field_role(name);
        mesh.set_field_descriptor(handle, motion::standard_motion_field_descriptor(role, mesh.dim()));
      }
    }

    if (mesh.field_entity_count(handle) != entity_count) {
      throw std::runtime_error("moving mesh restart: field entity count does not match restored topology");
    }
    read_field_values(in,
                      type,
                      mesh.field_data(handle),
                      entity_count * components,
                      entity_count * mesh.field_bytes_per_entity(handle));
  }

  if (!in) {
    throw std::runtime_error("moving mesh restart: malformed fields block");
  }
}

struct Payload {
  Metadata metadata;
  MeshBase mesh;
};

Payload read_payload(const std::string& path, const ReadOptions& options)
{
  std::ifstream in(path);
  if (!in.good()) {
    throw std::runtime_error("moving mesh restart: cannot open file '" + path + "'");
  }

  expect_tag(in, kMagic);
  expect_tag(in, "version");
  Metadata metadata;
  in >> metadata.version;
  if (!in) {
    throw std::runtime_error("moving mesh restart: malformed version block");
  }
  if (options.require_supported_version && metadata.version != kSupportedVersion) {
    throw std::runtime_error("moving mesh restart: unsupported version " +
                             std::to_string(metadata.version) +
                             "; supported version is " + std::to_string(kSupportedVersion));
  }

  expect_tag(in, "restart_epoch");
  in >> metadata.restart_epoch;

  expect_tag(in, "active_configuration");
  std::string active_configuration;
  in >> active_configuration;
  metadata.active_configuration = configuration_from_string(active_configuration);

  metadata.mesh_revisions = read_revision_state(in);
  read_metadata_maps(in, metadata);

  expect_tag(in, "mesh_dim");
  int dim = 0;
  in >> dim;
  if (dim <= 0 || dim > 3) {
    throw std::runtime_error("moving mesh restart: invalid mesh dimension");
  }

  auto x_ref = read_vector<real_t>(in, "x_ref");
  if (x_ref.size() % static_cast<std::size_t>(dim) != 0) {
    throw std::runtime_error("moving mesh restart: reference coordinate count is not divisible by dimension");
  }

  expect_tag(in, "has_current_coordinates");
  int has_current = 0;
  in >> has_current;
  metadata.has_current_coordinates = (has_current != 0);
  std::vector<real_t> x_cur;
  if (metadata.has_current_coordinates) {
    x_cur = read_vector<real_t>(in, "x_cur");
    if (x_cur.size() != x_ref.size()) {
      throw std::runtime_error("moving mesh restart: current coordinate count does not match reference coordinates");
    }
  }

  auto cell_shapes = read_shapes(in, "cell_shapes");
  auto cell_offsets = read_vector<offset_t>(in, "cell2vertex_offsets");
  auto cell_conn = read_vector<index_t>(in, "cell2vertex");

  auto face_shapes = read_shapes(in, "face_shapes");
  auto face_offsets = read_vector<offset_t>(in, "face2vertex_offsets");
  auto face_conn = read_vector<index_t>(in, "face2vertex");
  auto face2cell = read_pair_vector(in, "face2cell");
  auto edges = read_pair_vector(in, "edge2vertex");

  auto vertex_gids = read_vector<gid_t>(in, "vertex_gids");
  auto cell_gids = read_vector<gid_t>(in, "cell_gids");
  auto face_gids = read_vector<gid_t>(in, "face_gids");
  auto edge_gids = read_vector<gid_t>(in, "edge_gids");

  auto vertex_labels = read_vector<label_t>(in, "vertex_labels");
  auto cell_labels = read_vector<label_t>(in, "cell_labels");
  auto face_labels = read_vector<label_t>(in, "face_labels");
  auto edge_labels = read_vector<label_t>(in, "edge_labels");
  auto refinement_levels = read_vector<std::size_t>(in, "cell_refinement_levels");

  MeshBase mesh(dim);
  mesh.build_from_arrays(dim, x_ref, cell_offsets, cell_conn, cell_shapes);
  if (!face_shapes.empty()) {
    mesh.set_faces_from_arrays(face_shapes, face_offsets, face_conn, face2cell);
  }
  if (!edges.empty()) {
    mesh.set_edges_from_arrays(edges);
  }
  if (!vertex_gids.empty()) mesh.set_vertex_gids(std::move(vertex_gids));
  if (!cell_gids.empty()) mesh.set_cell_gids(std::move(cell_gids));
  if (!face_gids.empty()) mesh.set_face_gids(std::move(face_gids));
  if (!edge_gids.empty()) mesh.set_edge_gids(std::move(edge_gids));

  for (std::size_t i = 0; i < vertex_labels.size(); ++i) {
    if (vertex_labels[i] != INVALID_LABEL) {
      mesh.set_vertex_label(static_cast<index_t>(i), vertex_labels[i]);
    }
  }
  for (std::size_t i = 0; i < cell_labels.size(); ++i) {
    if (cell_labels[i] != INVALID_LABEL) {
      mesh.set_region_label(static_cast<index_t>(i), cell_labels[i]);
    }
  }
  for (std::size_t i = 0; i < face_labels.size(); ++i) {
    if (face_labels[i] != INVALID_LABEL) {
      mesh.set_boundary_label(static_cast<index_t>(i), face_labels[i]);
    }
  }
  for (std::size_t i = 0; i < edge_labels.size(); ++i) {
    if (edge_labels[i] != INVALID_LABEL) {
      mesh.set_edge_label(static_cast<index_t>(i), edge_labels[i]);
    }
  }
  if (!refinement_levels.empty()) {
    mesh.set_cell_refinement_levels(std::move(refinement_levels));
  }

  read_label_registry(in, mesh);
  read_sets(in, mesh);

  mesh.finalize();

  if (metadata.has_current_coordinates) {
    mesh.set_current_coords(x_cur);
  } else if (metadata.active_configuration == Configuration::Current &&
             options.require_current_coordinates_when_active_current) {
    throw std::runtime_error("moving mesh restart: active configuration is current but no current coordinates are stored");
  }

  if (metadata.active_configuration == Configuration::Current) {
    mesh.use_current_configuration();
  } else {
    mesh.use_reference_configuration();
  }

  read_fields(in, mesh);
  expect_tag(in, "end");

  return Payload{metadata, std::move(mesh)};
}

WriteOptions write_options_from_mesh_io(const MeshIOOptions& io_options)
{
  WriteOptions options;
  if (auto it = io_options.kv.find("restart_epoch"); it != io_options.kv.end()) {
    options.restart_epoch = static_cast<std::uint64_t>(std::stoull(it->second));
  }
  if (auto it = io_options.kv.find("include_fields"); it != io_options.kv.end()) {
    options.include_fields = (it->second == "true" || it->second == "1");
  }
  if (auto it = io_options.kv.find("include_motion_fields"); it != io_options.kv.end()) {
    options.include_motion_fields = (it->second == "true" || it->second == "1");
  }

  constexpr const char* backend_prefix = "motion_backend.";
  constexpr std::size_t backend_prefix_len = 15;
  constexpr const char* provenance_prefix = "adaptivity_provenance.";
  constexpr std::size_t provenance_prefix_len = 22;
  std::map<std::string, std::string> provenance_ordered;

  for (const auto& [key, value] : io_options.kv) {
    if (key.rfind(backend_prefix, 0) == 0) {
      options.motion_backend_state.emplace(key.substr(backend_prefix_len), value);
    } else if (key.rfind(provenance_prefix, 0) == 0) {
      provenance_ordered.emplace(key.substr(provenance_prefix_len), value);
    }
  }
  for (const auto& [key, value] : provenance_ordered) {
    (void)key;
    options.adaptivity_provenance.push_back(value);
  }

  return options;
}

void registered_write(const MeshBase& mesh, const MeshIOOptions& io_options)
{
  write(mesh, io_options.path, write_options_from_mesh_io(io_options));
}

MeshBase registered_read(const MeshIOOptions& io_options)
{
  return read(io_options.path);
}

} // namespace

void write(const MeshBase& mesh, const std::string& path, const WriteOptions& options)
{
  std::ofstream out(path);
  if (!out.good()) {
    throw std::runtime_error("moving mesh restart: cannot open file for writing '" + path + "'");
  }

  out << std::setprecision(17);
  out << kMagic << '\n';
  out << "version " << kSupportedVersion << '\n';
  out << "restart_epoch " << options.restart_epoch << '\n';
  out << "active_configuration " << configuration_to_string(mesh.active_configuration()) << '\n';
  write_revision_state(out, mesh.revision_state());
  write_metadata_maps(out, options);

  out << "mesh_dim " << mesh.dim() << '\n';
  write_vector(out, "x_ref", mesh.X_ref());
  out << "has_current_coordinates " << (mesh.has_current_coords() ? 1 : 0) << '\n';
  if (mesh.has_current_coords()) {
    write_vector(out, "x_cur", mesh.X_cur());
  }

  write_shapes(out, "cell_shapes", mesh.cell_shapes());
  write_vector(out, "cell2vertex_offsets", mesh.cell2vertex_offsets());
  write_vector(out, "cell2vertex", mesh.cell2vertex());

  write_shapes(out, "face_shapes", mesh.face_shapes());
  write_vector(out, "face2vertex_offsets", mesh.face2vertex_offsets());
  write_vector(out, "face2vertex", mesh.face2vertex());
  write_pair_vector(out, "face2cell", mesh.face2cell());
  write_pair_vector(out, "edge2vertex", mesh.edge2vertex());

  write_vector(out, "vertex_gids", mesh.vertex_gids());
  write_vector(out, "cell_gids", mesh.cell_gids());
  write_vector(out, "face_gids", mesh.face_gids());
  write_vector(out, "edge_gids", mesh.edge_gids());

  write_vector(out, "vertex_labels", mesh.vertex_label_ids());
  write_vector(out, "cell_labels", mesh.cell_region_ids());
  write_vector(out, "face_labels", mesh.face_boundary_ids());
  write_vector(out, "edge_labels", mesh.edge_label_ids());
  write_vector(out, "cell_refinement_levels", mesh.cell_refinement_levels());

  write_label_registry(out, mesh);
  write_sets(out, mesh);
  write_fields(out, mesh, options);
  out << "end\n";

  if (!out.good()) {
    throw std::runtime_error("moving mesh restart: failed while writing '" + path + "'");
  }
}

MeshBase read(const std::string& path, const ReadOptions& options)
{
  return read_payload(path, options).mesh;
}

Metadata inspect(const std::string& path, const ReadOptions& options)
{
  return read_payload(path, options).metadata;
}

void register_with_mesh()
{
  MeshBase::register_reader("svmp_restart", registered_read);
  MeshBase::register_reader("moving_mesh_restart", registered_read);
  MeshBase::register_reader("mmrst", registered_read);

  MeshBase::register_writer("svmp_restart", registered_write);
  MeshBase::register_writer("moving_mesh_restart", registered_write);
  MeshBase::register_writer("mmrst", registered_write);
}

} // namespace moving_mesh_restart
} // namespace svmp
