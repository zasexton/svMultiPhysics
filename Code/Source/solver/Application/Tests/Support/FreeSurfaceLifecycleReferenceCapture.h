/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#pragma once

#include "Application/Core/FreeSurfaceEnergyLedger.h"
#include "FE/Core/FEConfig.h"
#include "FE/Core/Types.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "FE/Basis/BasisFunction.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Spaces/FunctionSpace.h"
#include "FE/Systems/FESystem.h"
#include "Mesh/Mesh.h"
#endif

#include <gtest/gtest.h>
#if FE_HAS_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <locale>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace application_test::free_surface_lifecycle_capture {

using JsonFields = std::vector<std::pair<std::string, std::string>>;

inline std::string jsonString(std::string_view value) {
  std::ostringstream output;
  output << '"';
  for (const unsigned char character : value) {
    switch (character) {
    case '"':
      output << "\\\"";
      break;
    case '\\':
      output << "\\\\";
      break;
    case '\b':
      output << "\\b";
      break;
    case '\f':
      output << "\\f";
      break;
    case '\n':
      output << "\\n";
      break;
    case '\r':
      output << "\\r";
      break;
    case '\t':
      output << "\\t";
      break;
    default:
      if (character < 0x20u) {
        output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
               << static_cast<unsigned>(character) << std::dec;
      } else {
        output << static_cast<char>(character);
      }
    }
  }
  output << '"';
  return output.str();
}

inline std::string jsonBool(bool value) { return value ? "true" : "false"; }

template <class Integer> std::string jsonInteger(Integer value) {
  return std::to_string(value);
}

inline std::string jsonReal(double value) {
  if (!std::isfinite(value)) {
    throw std::runtime_error("lifecycle reference JSON requires a finite real");
  }
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<double>::max_digits10)
         << value;
  return output.str();
}

inline std::string jsonObservedReal(double value,
                                    std::string_view unavailable_reason) {
  if (std::isfinite(value)) {
    return "{\"status\":\"available\",\"value\":" + jsonReal(value) + "}";
  }
  return "{\"status\":\"unavailable\",\"reason\":" +
         jsonString(unavailable_reason) + ",\"value\":null}";
}

inline std::string
jsonOptionalUnsigned(const std::optional<std::uint64_t> &value,
                     std::string_view absent_reason) {
  if (value.has_value()) {
    return "{\"status\":\"available\",\"value\":" + jsonInteger(*value) + "}";
  }
  return "{\"status\":\"absent\",\"reason\":" + jsonString(absent_reason) +
         ",\"value\":null}";
}

inline std::string jsonObject(const JsonFields &fields) {
  std::ostringstream output;
  output << '{';
  for (std::size_t index = 0u; index < fields.size(); ++index) {
    if (index != 0u) {
      output << ',';
    }
    output << jsonString(fields[index].first) << ':' << fields[index].second;
  }
  output << '}';
  return output.str();
}

inline std::string jsonArray(const std::vector<std::string> &entries) {
  std::ostringstream output;
  output << '[';
  for (std::size_t index = 0u; index < entries.size(); ++index) {
    if (index != 0u) {
      output << ',';
    }
    output << entries[index];
  }
  output << ']';
  return output.str();
}

template <class Real> std::string jsonRealArray(std::span<const Real> values) {
  std::vector<std::string> entries;
  entries.reserve(values.size());
  for (const auto value : values) {
    entries.push_back(jsonReal(static_cast<double>(value)));
  }
  return jsonArray(entries);
}

template <class Integer>
std::string jsonIntegerArray(std::span<const Integer> values) {
  std::vector<std::string> entries;
  entries.reserve(values.size());
  for (const auto value : values) {
    entries.push_back(jsonInteger(value));
  }
  return jsonArray(entries);
}

struct CaptureGate {
  bool enabled{false};
  std::filesystem::path root{};
  std::string source_commit{};
  std::string source_tree{};
  std::string overlay_sha256{};
};

inline CaptureGate captureGate() {
  const char *directory = std::getenv("SVMP_FREE_SURFACE_R0_CAPTURE_DIR");
  if (directory == nullptr || directory[0] == '\0') {
    return {};
  }

  const auto require_lower_hex = [](const char *name, std::size_t length) {
    const char *raw = std::getenv(name);
    const std::string value = raw == nullptr ? std::string{} : raw;
    const auto lower_hex = [](char character) {
      return (character >= '0' && character <= '9') ||
             (character >= 'a' && character <= 'f');
    };
    if (value.size() != length ||
        !std::all_of(value.begin(), value.end(), lower_hex)) {
      throw std::runtime_error(std::string(name) + " must contain exactly " +
                               std::to_string(length) +
                               " lowercase hexadecimal characters when "
                               "lifecycle capture is enabled");
    }
    return value;
  };

#if FE_HAS_MPI
  int initialized = 0;
  if (MPI_Initialized(&initialized) != MPI_SUCCESS || initialized == 0) {
    throw std::runtime_error(
        "Application lifecycle capture requires initialized MPI");
  }
  int finalized = 0;
  if (MPI_Finalized(&finalized) != MPI_SUCCESS || finalized != 0) {
    throw std::runtime_error(
        "Application lifecycle capture requires active MPI");
  }
  int rank_count = 0;
  int rank = -1;
  if (MPI_Comm_size(MPI_COMM_WORLD, &rank_count) != MPI_SUCCESS ||
      MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS) {
    throw std::runtime_error(
        "Application lifecycle capture could not inspect MPI_COMM_WORLD");
  }
  if (rank_count != 1 || rank != 0) {
    throw std::runtime_error("Application lifecycle capture supports exactly "
                             "one initialized MPI rank");
  }

#else
  throw std::runtime_error(
      "Application lifecycle capture unavailable: built without MPI support");
#endif

  return CaptureGate{
      .enabled = true,
      .root = std::filesystem::path(directory),
      .source_commit =
          require_lower_hex("SVMP_FREE_SURFACE_R0_SOURCE_COMMIT", 40u),
      .source_tree = require_lower_hex("SVMP_FREE_SURFACE_R0_SOURCE_TREE", 40u),
      .overlay_sha256 =
          require_lower_hex("SVMP_FREE_SURFACE_R0_OVERLAY_SHA256", 64u),
  };
}

inline void publish(const CaptureGate &gate, std::string_view relative_path,
                    std::string_view test_suite, std::string_view test_name,
                    std::string_view case_scope, const std::string &payload,
                    std::string_view diagnostic_path = {},
                    std::string_view diagnostic_text = {}) {
  if (!gate.enabled) {
    return;
  }
  if (::testing::Test::HasFailure()) {
    throw std::runtime_error("lifecycle reference publication requires "
                             "successful fixture completion");
  }
  const auto envelope =
      jsonObject({
          {"artifact_type",
           jsonString("svmp_free_surface_application_lifecycle")},
          {"schema_version", "1"},
          {"source_gate",
           jsonObject({
               {"source_commit", jsonString(gate.source_commit)},
               {"source_tree", jsonString(gate.source_tree)},
               {"overlay_sha256", jsonString(gate.overlay_sha256)},
               {"overlay_identity",
                jsonString("retained_three_file_manifest_bundle")},
               {"overlay_source_file_count", "3"},
           })},
          {"execution_scope", jsonObject({
                                  {"mpi_initialized", "true"},
                                  {"rank_count", "1"},
                                  {"rank", "0"},
                                  {"layout", jsonString("serial")},
                              })},
          {"test_suite", jsonString(test_suite)},
          {"test_name", jsonString(test_name)},
          {"case_scope", jsonString(case_scope)},
          {"payload", payload},
      }) +
      "\n";

  // Both payloads are complete before any path is created. This bounded
  // publication has one numerical output and at most one diagnostic sidecar.
  struct Output {
    std::filesystem::path destination;
    std::filesystem::path temporary;
    std::string_view text;
    bool temporary_created{false};
    bool final_created{false};
  };
  std::vector<Output> outputs;
  const auto add_output = [&](std::string_view path, std::string_view text) {
    const std::filesystem::path relative(path);
    if (relative.empty() || relative.is_absolute() ||
        std::any_of(relative.begin(), relative.end(),
                    [](const auto &part) { return part == ".."; })) {
      throw std::runtime_error("invalid lifecycle publication relative path");
    }
    const auto destination = (gate.root / relative).lexically_normal();
    outputs.push_back({destination, destination.string() + ".tmp", text});
  };
  add_output(relative_path, envelope);
  if (!diagnostic_path.empty()) {
    add_output(diagnostic_path, diagnostic_text);
  }
  std::set<std::filesystem::path> names;
  for (const auto &output : outputs) {
    for (const auto &path : {output.destination, output.temporary}) {
      if (!names.insert(path).second) {
        throw std::runtime_error("overlapping lifecycle publication paths");
      }
      std::error_code error;
      const auto status = std::filesystem::symlink_status(path, error);
      if ((error && error != std::errc::no_such_file_or_directory) ||
          status.type() != std::filesystem::file_type::not_found) {
        throw std::runtime_error(
            "lifecycle output already exists or cannot be inspected: " +
            path.string());
      }
    }
  }
  try {
    for (auto &output : outputs) {
      std::filesystem::create_directories(output.destination.parent_path());
      std::FILE *file = std::fopen(output.temporary.string().c_str(), "wbx");
      if (file == nullptr) {
        throw std::runtime_error("cannot create lifecycle temporary exclusively");
      }
      output.temporary_created = true;
      const bool write_failed =
          std::fwrite(output.text.data(), 1u, output.text.size(), file) !=
          output.text.size();
      const bool flush_failed = std::fflush(file) != 0;
      const bool close_failed = std::fclose(file) != 0;
      if (write_failed || flush_failed || close_failed) {
        throw std::runtime_error("cannot write and close lifecycle temporary");
      }
    }
    for (auto &output : outputs) {
      std::filesystem::create_hard_link(output.temporary, output.destination);
      output.final_created = true;
    }
  } catch (const std::exception &failure) {
    std::string cleanup_failures;
    for (auto &output : outputs) {
      for (const auto &[created, path] :
           {std::pair{output.final_created, output.destination},
            std::pair{output.temporary_created, output.temporary}}) {
        if (created) {
          std::error_code error;
          std::filesystem::remove(path, error);
          if (error) {
            cleanup_failures += "; cleanup failed for " + path.string() +
                                ": " + error.message();
          }
        }
      }
    }
    throw std::runtime_error(std::string(failure.what()) + cleanup_failures);
  }
  // The complete group is committed. Temporary-link cleanup is best effort:
  // a cleanup error is diagnostic, never a reported fixture/publication failure.
  for (const auto &output : outputs) {
    std::error_code error;
    std::filesystem::remove(output.temporary, error);
    if (error) {
      std::fprintf(stderr, "lifecycle publication committed; temporary retained: %s\n",
                   output.temporary.c_str());
    }
  }
}

struct CanonicalRow {
  std::uint64_t public_fe_row{0u};
  std::uint64_t field_local_row{0u};
  std::string field_name{};
  std::uint64_t entity_gid{0u};
  bool constrained{false};
};

struct ScalarP1VertexMap {
  std::uint64_t global_size{0u};
  std::uint64_t system_dof_layout_revision{0u};
  std::uint64_t constraint_layout_revision{0u};
  std::uint64_t mesh_topology_revision{0u};
  std::uint64_t mesh_numbering_revision{0u};
  std::vector<CanonicalRow> rows{};
};

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
inline ScalarP1VertexMap
captureScalarP1VertexMap(const svmp::FE::systems::FESystem &system,
                         const svmp::Mesh &mesh,
                         std::span<const std::string_view> expected_fields) {
  const auto field_ids = system.unknownFieldIdsInDofMapOrder();
  if (field_ids.size() != expected_fields.size()) {
    throw std::runtime_error("lifecycle reference field count does not match "
                             "its declared scalar P1 map");
  }
  const auto &base = mesh.local_mesh();
  const auto &gids = base.vertex_gids();
  if (gids.size() != base.n_vertices()) {
    throw std::runtime_error(
        "lifecycle reference vertex GID array is incomplete");
  }
  std::set<std::uint64_t> unique_gids;
  for (const auto gid : gids) {
    if (!unique_gids.emplace(static_cast<std::uint64_t>(gid)).second) {
      throw std::runtime_error(
          "lifecycle reference vertex GIDs are not unique");
    }
  }

  ScalarP1VertexMap result{
      .global_size =
          static_cast<std::uint64_t>(system.dofHandler().getNumDofs()),
      .system_dof_layout_revision = system.dofLayoutRevision(),
      .constraint_layout_revision = system.constraintLayoutRevision(),
      .mesh_topology_revision = base.topology_revision(),
      .mesh_numbering_revision = base.numbering_revision(),
  };
  std::set<std::uint64_t> public_rows;
  std::set<std::pair<std::string, std::uint64_t>> canonical_keys;
  for (std::size_t field_index = 0u; field_index < field_ids.size();
       ++field_index) {
    const auto field = field_ids[field_index];
    const auto &record = system.fieldRecord(field);
    if (record.name != expected_fields[field_index] || record.components != 1 ||
        record.source_kind != svmp::FE::systems::FieldSourceKind::Unknown ||
        record.space == nullptr ||
        record.space->space_type() != svmp::FE::spaces::SpaceType::H1 ||
        record.space->field_type() != svmp::FE::FieldType::Scalar ||
        record.space->polynomial_order() != 1 ||
        record.space->element().basis().basis_type() !=
            svmp::FE::BasisType::Lagrange) {
      throw std::runtime_error("lifecycle reference supports only its declared "
                               "scalar nodal P1 unknown fields");
    }
    const auto offset = system.fieldDofOffset(field);
    const auto &handler = system.fieldDofHandler(field);
    if (handler.getNumDofs() !=
        static_cast<svmp::FE::GlobalIndex>(base.n_vertices())) {
      throw std::runtime_error("lifecycle reference scalar P1 field does not "
                               "have one row per vertex");
    }
    const auto &partition = handler.getPartition();
    if (partition.localOwnedSize() != handler.getNumDofs() ||
        partition.ghostSize() != 0) {
      throw std::runtime_error("lifecycle reference scalar P1 map is not fully "
                               "owned without ghosts");
    }
    const auto *entity_map = handler.getEntityDofMap();
    if (entity_map == nullptr || !entity_map->isFinalized()) {
      throw std::runtime_error("lifecycle reference scalar P1 entity map is "
                               "unavailable or unfinalized");
    }
    for (std::size_t vertex = 0u; vertex < base.n_vertices(); ++vertex) {
      const auto dofs =
          entity_map->getVertexDofs(static_cast<svmp::FE::GlobalIndex>(vertex));
      if (dofs.size() != 1u || dofs.front() < 0 ||
          dofs.front() >= handler.getNumDofs()) {
        throw std::runtime_error("lifecycle reference scalar P1 vertex does "
                                 "not own one valid field-local row");
      }
      const auto public_row = offset + dofs.front();
      const auto public_key = static_cast<std::uint64_t>(public_row);
      const auto gid = static_cast<std::uint64_t>(gids[vertex]);
      if (!public_rows.emplace(public_key).second ||
          !canonical_keys.emplace(record.name, gid).second) {
        throw std::runtime_error(
            "lifecycle reference scalar P1 map has duplicate row identity");
      }
      result.rows.push_back(CanonicalRow{
          .public_fe_row = public_key,
          .field_local_row = static_cast<std::uint64_t>(dofs.front()),
          .field_name = record.name,
          .entity_gid = gid,
          .constrained = system.constraints().isConstrained(public_row),
      });
    }
  }
  std::sort(result.rows.begin(), result.rows.end(),
            [](const auto &left, const auto &right) {
              return left.public_fe_row < right.public_fe_row;
            });
  if (result.rows.size() != result.global_size) {
    throw std::runtime_error(
        "lifecycle reference scalar P1 map does not cover every public FE row");
  }
  for (std::size_t row = 0u; row < result.rows.size(); ++row) {
    if (result.rows[row].public_fe_row != row) {
      throw std::runtime_error(
          "lifecycle reference scalar P1 map is not a public-row bijection");
    }
  }
  return result;
}

#endif

inline std::string scalarP1VertexMapJson(const ScalarP1VertexMap &map) {
  std::vector<std::string> rows;
  rows.reserve(map.rows.size());
  for (const auto &row : map.rows) {
    rows.push_back(jsonObject({
        {"public_fe_row", jsonInteger(row.public_fe_row)},
        {"field_local_row", jsonInteger(row.field_local_row)},
        {"field_name", jsonString(row.field_name)},
        {"component", "0"},
        {"entity_kind", jsonString("vertex")},
        {"entity_gid", jsonInteger(row.entity_gid)},
        {"basis_ordinal", "0"},
        {"constrained", jsonBool(row.constrained)},
    }));
  }
  return jsonObject({
      {"global_size", jsonInteger(map.global_size)},
      {"system_dof_layout_revision",
       jsonInteger(map.system_dof_layout_revision)},
      {"constraint_layout_revision",
       jsonInteger(map.constraint_layout_revision)},
      {"mesh_topology_revision", jsonInteger(map.mesh_topology_revision)},
      {"mesh_numbering_revision", jsonInteger(map.mesh_numbering_revision)},
      {"rows", jsonArray(rows)},
  });
}

inline std::string fieldSubmapJson(const ScalarP1VertexMap &map,
                                   std::string_view field_name) {
  std::vector<std::string> rows;
  for (const auto &row : map.rows) {
    if (row.field_name != field_name) {
      continue;
    }
    rows.push_back(jsonObject({
        {"field_local_row", jsonInteger(row.field_local_row)},
        {"public_fe_row", jsonInteger(row.public_fe_row)},
        {"field_name", jsonString(row.field_name)},
        {"component", "0"},
        {"entity_kind", jsonString("vertex")},
        {"entity_gid", jsonInteger(row.entity_gid)},
        {"basis_ordinal", "0"},
    }));
  }
  if (rows.empty()) {
    throw std::runtime_error("lifecycle reference field submap is empty");
  }
  return jsonObject({
      {"field_name", jsonString(field_name)},
      {"ordering", jsonString("field_local_row")},
      {"rows", jsonArray(rows)},
  });
}

template <class Vector>
void validateCompleteOwnedVector(const Vector &vector,
                                 const ScalarP1VertexMap &map,
                                 std::span<const svmp::FE::Real> values) {
  const auto owned = vector.ownedGlobalRows();
  if (static_cast<std::uint64_t>(vector.size()) != map.global_size ||
      values.size() != map.global_size || owned.size() != map.global_size) {
    throw std::runtime_error(
        "lifecycle reference vector does not cover the complete public FE map");
  }
  std::set<std::uint64_t> owned_rows;
  for (const auto row : owned) {
    owned_rows.emplace(static_cast<std::uint64_t>(row));
  }
  for (std::size_t row = 0u; row < map.rows.size(); ++row) {
    if (!owned_rows.contains(row) || !std::isfinite(values[row])) {
      throw std::runtime_error("lifecycle reference vector has an unowned or "
                               "nonfinite public FE row");
    }
  }
}

inline std::string
vectorSnapshotJson(std::string_view checkpoint, std::string_view role,
                   std::span<const svmp::FE::Real> values,
                   std::optional<std::uint64_t> value_revision = std::nullopt) {
  return jsonObject({
      {"checkpoint", jsonString(checkpoint)},
      {"semantic_role", jsonString(role)},
      {"ordering", jsonString("public_fe_row")},
      {"value_revision",
       jsonOptionalUnsigned(value_revision,
                            "observer span has no vector revision getter")},
      {"values", jsonRealArray(values)},
  });
}

template <class Signature>
std::string refreshSignatureJson(const std::optional<Signature> &signature) {
  if (!signature.has_value()) {
    return jsonObject({
        {"status", jsonString("absent")},
        {"reason", jsonString("refresh cache has no signature")},
        {"value", "null"},
    });
  }
  const auto &value = *signature;
  return jsonObject({
      {"status", jsonString("available")},
      {"value",
       jsonObject({
           {"request_policy_key", jsonInteger(value.request_policy_key)},
           {"mesh_geometry_revision",
            jsonInteger(value.mesh_geometry_revision)},
           {"mesh_topology_revision",
            jsonInteger(value.mesh_topology_revision)},
           {"mesh_ownership_revision",
            jsonInteger(value.mesh_ownership_revision)},
           {"mesh_numbering_revision",
            jsonInteger(value.mesh_numbering_revision)},
           {"mesh_field_layout_revision",
            jsonInteger(value.mesh_field_layout_revision)},
           {"mesh_label_revision", jsonInteger(value.mesh_label_revision)},
           {"mesh_active_configuration_epoch",
            jsonInteger(value.mesh_active_configuration_epoch)},
           {"mesh_coordinate_configuration_key",
            jsonInteger(value.mesh_coordinate_configuration_key)},
           {"system_space_revision", jsonInteger(value.system_space_revision)},
           {"system_dof_layout_revision",
            jsonInteger(value.system_dof_layout_revision)},
           {"system_block_layout_revision",
            jsonInteger(value.system_block_layout_revision)},
           {"solution_signature_kind",
            jsonInteger(static_cast<unsigned>(value.solution_signature_kind))},
           {"solution_hash", jsonInteger(value.solution_hash)},
           {"solution_size", jsonInteger(value.solution_size)},
       })},
  });
}

template <class Identity>
std::string phaseGraphIdentityJson(const Identity &identity) {
  return jsonObject({
      {"dimension", jsonInteger(identity.dimension)},
      {"nodes", jsonInteger(identity.nodes)},
      {"edges", jsonInteger(identity.edges)},
      {"geometry_revision", jsonInteger(identity.geometry_revision)},
      {"topology_revision", jsonInteger(identity.topology_revision)},
      {"ownership_revision", jsonInteger(identity.ownership_revision)},
      {"numbering_revision", jsonInteger(identity.numbering_revision)},
      {"dof_layout_revision", jsonInteger(identity.dof_layout_revision)},
      {"content_revision", jsonInteger(identity.content_revision)},
      {"content_revision_scope",
       jsonString("execution_layout_sensitive_diagnostic")},
  });
}

template <class Snapshot>
std::string conservativeCandidateStageJson(const Snapshot &snapshot) {
  std::vector<std::string> requests;
  requests.reserve(snapshot.requests.size());
  for (const auto &request : snapshot.requests) {
    std::vector<std::string> velocity;
    velocity.reserve(request.sampled_nodal_velocity.size());
    for (const auto &value : request.sampled_nodal_velocity) {
      velocity.push_back(jsonRealArray(std::span<const svmp::FE::Real>(value)));
    }
    requests.push_back(jsonObject({
        {"enabled", jsonBool(request.enabled)},
        {"phase_field_name", jsonString(request.phase_field_name)},
        {"velocity_field_name", jsonString(request.velocity_field_name)},
        {"velocity_source",
         jsonInteger(static_cast<unsigned>(request.velocity_source))},
        {"material_interface_marker",
         jsonInteger(request.material_interface_marker)},
        {"graph_identity", phaseGraphIdentityJson(request.graph_identity)},
        {"graph_geometry_revision",
         jsonInteger(request.graph_geometry_revision)},
        {"graph_topology_revision",
         jsonInteger(request.graph_topology_revision)},
        {"graph_ownership_revision",
         jsonInteger(request.graph_ownership_revision)},
        {"graph_numbering_revision",
         jsonInteger(request.graph_numbering_revision)},
        {"sampled_nodal_velocity", jsonArray(velocity)},
    }));
  }
  return jsonObject({
      {"scheme",
       jsonString("backward_euler_explicit_indicator_endpoint_velocity")},
      {"temporal_order", jsonInteger(snapshot.temporal_order)},
      {"prospective_step", jsonInteger(snapshot.prospective_step)},
      {"attempt", jsonInteger(snapshot.attempt)},
      {"step_start_time", jsonReal(snapshot.step_start_time)},
      {"step_end_time", jsonReal(snapshot.step_end_time)},
      {"q_input_time", jsonReal(snapshot.q_input_time)},
      {"velocity_state_time", jsonReal(snapshot.velocity_state_time)},
      {"time_step", jsonReal(snapshot.time_step)},
      {"operator_state_revision",
       jsonInteger(snapshot.operator_state_revision)},
      {"request_schedule_words",
       jsonIntegerArray(
           std::span<const std::uint64_t>(snapshot.request_schedule_words))},
      {"requests", jsonArray(requests)},
  });
}

template <class Ledger>
std::string conservativeMaintenanceLedgerJson(const Ledger &ledger) {
  const auto mismatch = [](const auto &value) {
    return jsonObject({
        {"maximum_nodal_residual", jsonReal(value.maximum_nodal_residual)},
        {"residual_norm", jsonReal(value.residual_norm)},
        {"total_residual", jsonReal(value.total_residual)},
    });
  };
  const auto &transport = ledger.transport_stage;
  const auto &repair = ledger.reinitialization;
  const auto &reconciliation = ledger.reconciliation;
  const auto stage_options = [](const auto &options) {
    return jsonObject({
        {"invariant_tolerance", jsonReal(options.invariant_tolerance)},
        {"component_activity_tolerance",
         jsonReal(options.component_activity_tolerance)},
        {"maximum_courant", jsonReal(options.maximum_courant)},
        {"enforce_courant_limit", jsonBool(options.enforce_courant_limit)},
        {"require_constant_preservation",
         jsonBool(options.require_constant_preservation)},
    });
  };
  std::string provenance = jsonObject({
      {"status", jsonString("absent")},
      {"reason", jsonString("split-stage provenance unavailable")},
      {"value", "null"},
  });
  if (ledger.split_stage_provenance.has_value()) {
    const auto &value = *ledger.split_stage_provenance;
    provenance = jsonObject({
        {"status", jsonString("available")},
        {"value",
         jsonObject({
             {"scheme",
              jsonString(
                  "backward_euler_explicit_indicator_endpoint_velocity")},
             {"transport_mesh_policy", jsonString("fixed_background")},
             {"temporal_order", jsonInteger(value.temporal_order)},
             {"prospective_step", jsonInteger(value.prospective_step)},
             {"attempt", jsonInteger(value.attempt)},
             {"step_start_time", jsonReal(value.step_start_time)},
             {"step_end_time", jsonReal(value.step_end_time)},
             {"q_input_time", jsonReal(value.q_input_time)},
             {"velocity_state_time", jsonReal(value.velocity_state_time)},
             {"time_step", jsonReal(value.time_step)},
             {"operator_state_revision",
              jsonInteger(value.operator_state_revision)},
             {"previous_q_revision", jsonInteger(value.previous_q_revision)},
             {"nodal_velocity_revision",
              jsonInteger(value.nodal_velocity_revision)},
             {"previous_graph_identity",
              phaseGraphIdentityJson(value.previous_graph_identity)},
             {"operator_graph_identity",
              phaseGraphIdentityJson(value.operator_graph_identity)},
             {"final_flux_ledger_digest",
              jsonInteger(value.final_flux_ledger_digest)},
             {"stage_options", stage_options(value.stage_options)},
         })},
    });
  }
  return jsonObject({
      {"reinitialization_due", jsonBool(ledger.reinitialization_due)},
      {"reinitialization_applied", jsonBool(ledger.reinitialization_applied)},
      {"inventory",
       jsonObject({
           {"raw_post_transport_phase_measure",
            jsonReal(ledger.raw_post_transport_phase_measure)},
           {"post_limit_phase_measure",
            jsonReal(ledger.post_limit_phase_measure)},
           {"raw_post_transport_geometry_measure",
            jsonReal(ledger.raw_post_transport_geometry_measure)},
           {"post_reinitialization_geometry_measure",
            jsonReal(ledger.post_reinitialization_geometry_measure)},
           {"post_correction_phase_measure",
            jsonReal(ledger.post_correction_phase_measure)},
           {"post_correction_geometry_measure",
            jsonReal(ledger.post_correction_geometry_measure)},
           {"maximum_nodal_boundary_mass_transfer",
            jsonReal(ledger.maximum_nodal_boundary_mass_transfer)},
           {"boundary_mass_tolerance",
            jsonReal(ledger.boundary_mass_tolerance)},
           {"boundary_flux_policy",
            jsonInteger(static_cast<unsigned>(ledger.boundary_flux_policy))},
       })},
      {"post_reinitialization_mismatch",
       mismatch(ledger.post_reinitialization_mismatch)},
      {"post_correction_mismatch", mismatch(ledger.post_correction_mismatch)},
      {"transport_stage",
       jsonObject({
           {"success", jsonBool(transport.success)},
           {"courant_satisfied", jsonBool(transport.courant_satisfied)},
           {"low_order_coefficients_nonnegative",
            jsonBool(transport.low_order_coefficients_nonnegative)},
           {"strong_form_decomposition_satisfied",
            jsonBool(transport.strong_form_decomposition_satisfied)},
           {"replicated_stage_inputs_satisfied",
            jsonBool(transport.replicated_stage_inputs_satisfied)},
           {"maximum_courant", jsonReal(transport.maximum_courant)},
           {"minimum_low_order_coefficient",
            jsonReal(transport.minimum_low_order_coefficient)},
           {"maximum_strong_form_decomposition_residual",
            jsonReal(transport.maximum_strong_form_decomposition_residual)},
           {"time_step", jsonReal(transport.time_step)},
           {"executed_options", stage_options(transport.executed_options)},
           {"nodal_courant", jsonRealArray(std::span<const svmp::FE::Real>(
                                 transport.nodal_courant))},
           {"physical_boundary_mass_transfer",
            jsonRealArray(std::span<const svmp::FE::Real>(
                transport.physical_boundary_mass_transfer))},
           {"discrete_divergence_mass_source",
            jsonRealArray(std::span<const svmp::FE::Real>(
                transport.discrete_divergence_mass_source))},
           {"diagnostic", jsonString(transport.diagnostic)},
       })},
      {"region_ledger",
       jsonObject({
           {"success", jsonBool(ledger.region_ledger.success)},
           {"all_balances_satisfied",
            jsonBool(ledger.region_ledger.all_balances_satisfied)},
           {"maximum_balance_residual",
            jsonReal(ledger.region_ledger.maximum_balance_residual)},
           {"maximum_flux_reconstruction_residual",
            jsonReal(
                ledger.region_ledger.maximum_flux_reconstruction_residual)},
           {"region_count", jsonInteger(ledger.region_ledger.regions.size())},
           {"diagnostic", jsonString(ledger.region_ledger.diagnostic)},
       })},
      {"reinitialization",
       jsonObject({
           {"success", jsonBool(repair.success)},
           {"converged", jsonBool(repair.converged)},
           {"iterations", jsonInteger(repair.iterations)},
           {"repaired_dofs", jsonInteger(repair.repaired_dofs)},
           {"preserved_dofs", jsonInteger(repair.preserved_dofs)},
           {"max_abs_update", jsonReal(repair.max_abs_update)},
           {"max_distance", jsonReal(repair.max_distance)},
           {"max_interface_displacement",
            jsonReal(repair.max_interface_displacement)},
           {"max_iteration_residual", jsonReal(repair.max_iteration_residual)},
           {"max_signed_distance_error",
            jsonReal(repair.max_signed_distance_error)},
           {"diagnostic", jsonString(repair.diagnostic)},
       })},
      {"reconciliation",
       jsonObject({
           {"success", jsonBool(reconciliation.success)},
           {"target_reached", jsonBool(reconciliation.target_reached)},
           {"limited_by_displacement",
            jsonBool(reconciliation.limited_by_displacement)},
           {"limited_by_topology",
            jsonBool(reconciliation.limited_by_topology)},
           {"iterations", jsonInteger(reconciliation.iterations)},
           {"line_search_evaluations",
            jsonInteger(reconciliation.line_search_evaluations)},
           {"geometry_refresh_requests",
            jsonInteger(reconciliation.geometry_refresh_requests)},
           {"geometry_rebuilds", jsonInteger(reconciliation.geometry_rebuilds)},
           {"rejected_geometry_trials",
            jsonInteger(reconciliation.rejected_geometry_trials)},
           {"initial_residual_norm",
            jsonReal(reconciliation.initial_residual_norm)},
           {"final_residual_norm",
            jsonReal(reconciliation.final_residual_norm)},
           {"maximum_final_nodal_residual",
            jsonReal(reconciliation.maximum_final_nodal_residual)},
           {"final_total_residual",
            jsonReal(reconciliation.final_total_residual)},
           {"diagnostic", jsonString(reconciliation.diagnostic)},
       })},
      {"split_stage_provenance", provenance},
  });
}

template <class Plan>
std::string generalizedAlphaPlanJson(const Plan &plan, std::string_view closure,
                                     std::string_view status,
                                     std::string_view subcase) {
  const auto optional_real = [](const auto &value, std::string_view reason) {
    if (!value.has_value()) {
      return jsonObject({
          {"status", jsonString("absent")},
          {"reason", jsonString(reason)},
          {"value", "null"},
      });
    }
    return jsonObject({
        {"status", jsonString("available")}, {"value", jsonReal(*value)},
    });
  };
  const auto optional_array = [](const auto &value, std::string_view reason) {
    if (!value.has_value()) {
      return jsonObject({
          {"status", jsonString("absent")},
          {"reason", jsonString(reason)},
          {"value", "null"},
      });
    }
    return jsonObject({
        {"status", jsonString("available")},
        {"value", jsonRealArray(std::span<const svmp::FE::Real>(*value))},
    });
  };
  JsonFields fields{
      {"subcase", jsonString(subcase)},
      {"component_identity",
       jsonObject({
           {"kind", jsonString("synthetic_fixture_component")},
           {"names",
            jsonArray({jsonString("component_0"), jsonString("component_1"),
                       jsonString("component_2")})},
           {"mesh_dof_identity", jsonString("not_applicable")},
       })},
      {"closure", jsonString(closure)},
      {"status", jsonString(status)},
      {"scheme",
       jsonObject({
           {"alpha_m", optional_real(plan.scheme.alpha_m,
                                     "rate parameters intentionally omitted")},
           {"alpha_f", jsonReal(plan.scheme.alpha_f)},
           {"gamma", optional_real(plan.scheme.gamma,
                                   "rate parameters intentionally omitted")},
           {"dt", optional_real(plan.scheme.dt,
                                "rate parameters intentionally omitted")},
       })},
      {"requested_stage_state_delta",
       jsonRealArray(
           std::span<const svmp::FE::Real>(plan.requested_stage_state_delta))},
      {"requested_endpoint_state_delta",
       jsonRealArray(std::span<const svmp::FE::Real>(
           plan.requested_endpoint_state_delta))},
      {"implied_prior_state_delta",
       optional_array(plan.implied_prior_state_delta,
                      "plan does not expose an implied prior-state delta")},
      {"identity_tolerance", jsonReal(plan.identity_tolerance)},
      {"max_stage_state_identity_residual",
       jsonReal(plan.max_stage_state_identity_residual)},
      {"max_endpoint_update_identity_residual",
       jsonReal(plan.max_endpoint_update_identity_residual)},
      {"max_stage_rate_identity_residual",
       jsonReal(plan.max_stage_rate_identity_residual)},
      {"requires_separate_geometric_motion_account",
       jsonBool(plan.requires_separate_geometric_motion_account)},
      {"diagnostic", jsonString(plan.diagnostic)},
  };
  if (plan.post_accept.has_value()) {
    const auto &post = *plan.post_accept;
    fields.emplace_back(
        "post_accept",
        jsonObject({
            {"status", jsonString("available")},
            {"u_delta",
             jsonRealArray(std::span<const svmp::FE::Real>(post.u_delta))},
            {"u_prev_delta",
             jsonRealArray(std::span<const svmp::FE::Real>(post.u_prev_delta))},
            {"u_prev2_and_deeper_delta",
             jsonRealArray(std::span<const svmp::FE::Real>(
                 post.u_prev2_and_deeper_delta))},
            {"prior_rate_delta", jsonRealArray(std::span<const svmp::FE::Real>(
                                     post.prior_rate_delta))},
            {"u_dot_delta",
             jsonRealArray(std::span<const svmp::FE::Real>(post.u_dot_delta))},
            {"accepted_stage_rate_delta",
             jsonRealArray(std::span<const svmp::FE::Real>(
                 post.accepted_stage_rate_delta))},
            {"maintained_first_order_u_ddot_unchanged",
             jsonBool(post.maintained_first_order_u_ddot_unchanged)},
        }));
  } else {
    fields.emplace_back(
        "post_accept",
        jsonObject({
            {"status", jsonString("absent")},
            {"reason",
             jsonString("publication plan is not algebraically complete")},
            {"value", "null"},
        }));
  }
  return jsonObject(fields);
}

inline const char *
energyStatusName(application::core::FreeSurfaceEnergyAttemptStatus status) {
  using T = application::core::FreeSurfaceEnergyAttemptStatus;
  switch (status) {
  case T::Trial:
    return "trial";
  case T::Accepted:
    return "accepted";
  case T::Rejected:
    return "rejected";
  }
  throw std::runtime_error("unknown energy attempt status");
}

inline const char *
energyReasonName(application::core::FreeSurfaceEnergyRejectionReason reason) {
  using T = application::core::FreeSurfaceEnergyRejectionReason;
  switch (reason) {
  case T::None:
    return "none";
  case T::NonlinearSolveFailure:
    return "nonlinear_solve_failure";
  case T::StepControllerRejection:
    return "step_controller_rejection";
  case T::PreacceptRejection:
    return "preaccept_rejection";
  case T::TopologyChange:
    return "topology_change";
  case T::MaintenanceRollback:
    return "maintenance_rollback";
  case T::PublicationFailure:
    return "publication_failure";
  }
  throw std::runtime_error("unknown energy rejection reason");
}

inline const char *
energySchemeName(application::core::FreeSurfaceEnergyTemporalScheme scheme) {
  using T = application::core::FreeSurfaceEnergyTemporalScheme;
  switch (scheme) {
  case T::Unspecified:
    return "unspecified";
  case T::BackwardEuler:
    return "backward_euler";
  case T::GeneralizedAlpha:
    return "generalized_alpha_unsupported";
  }
  throw std::runtime_error("unknown energy temporal scheme");
}

inline std::string channelSourceJson(
    const application::core::FreeSurfaceEnergyChannelSource &source) {
  using T = application::core::FreeSurfaceEnergyChannelApplicability;
  const char *applicability = "unspecified";
  if (source.applicability == T::Produced) {
    applicability = "produced";
  } else if (source.applicability == T::NotApplicable) {
    applicability = "not_applicable";
  }
  return jsonObject({
      {"applicability", jsonString(applicability)},
      {"owner",
       source.owner.empty()
           ? jsonObject({{"status", jsonString("absent")}, {"value", "null"}})
           : jsonObject({{"status", jsonString("available")},
                         {"value", jsonString(source.owner)}})},
  });
}

inline std::string
storedEnergyJson(const application::core::FreeSurfaceStoredEnergy &value) {
  using G = application::core::FreeSurfaceGasEnergyApplicability;
  const char *gas = "unspecified";
  if (value.gas_applicability == G::Active) {
    gas = "active";
  } else if (value.gas_applicability == G::NotApplicable) {
    gas = "not_applicable";
  }
  return jsonObject({
      {"kinetic", jsonObservedReal(value.kinetic, "unstaged")},
      {"gravitational", jsonObservedReal(value.gravitational, "unstaged")},
      {"liquid_gas_surface",
       jsonObservedReal(value.liquid_gas_surface, "unstaged")},
      {"solid_liquid_wall",
       jsonObservedReal(value.solid_liquid_wall, "unstaged")},
      {"gas_applicability", jsonString(gas)},
      {"gas_or_compressibility",
       jsonObservedReal(value.gas_or_compressibility, "unstaged")},
  });
}

inline std::string
energyAttemptJson(const application::core::FreeSurfaceEnergyAttempt &attempt) {
  const auto &metadata = attempt.metadata;
  const auto endpoint_revision = [&](std::uint64_t value) {
    if (!attempt.balance_staged) {
      return jsonObject({
          {"status", jsonString("unavailable")},
          {"reason", jsonString("attempt rejected without a staged endpoint balance")},
          {"value", "null"},
      });
    }
    return jsonObject({{"status", jsonString("available")},
                       {"value", jsonInteger(value)}});
  };
  const auto &sources = attempt.channel_sources;
  const auto source_group = [](const auto &group, const auto &names) {
    JsonFields fields;
    for (const auto & [ name, member ] : names) {
      fields.emplace_back(name, channelSourceJson(group.*member));
    }
    return jsonObject(fields);
  };
  return jsonObject({
      {"fixture_kind", jsonString("synthetic_ledger_inputs")},
      {"status", jsonString(energyStatusName(attempt.status))},
      {"rejection_reason",
       jsonString(energyReasonName(attempt.rejection_reason))},
      {"balance_staged", jsonBool(attempt.balance_staged)},
      {"metadata",
       jsonObject({
           {"transaction_id", jsonInteger(metadata.transaction_id)},
           {"step", jsonInteger(metadata.step)},
           {"attempt", jsonInteger(metadata.attempt)},
           {"time_before",
            jsonObservedReal(metadata.time_before, "unstaged metadata")},
           {"time_after",
            jsonObservedReal(metadata.time_after, "unstaged metadata")},
           {"dt", jsonObservedReal(metadata.dt, "unstaged metadata")},
           {"temporal_scheme",
            jsonString(energySchemeName(metadata.temporal_scheme))},
           {"physical_evaluation_time",
            jsonObservedReal(metadata.physical_evaluation_time,
                             "unstaged rejection")},
           {"physical_evaluation_stage_fraction",
            jsonObservedReal(metadata.physical_evaluation_stage_fraction,
                             "unstaged rejection")},
           {"algebraic_state_revision_before",
            jsonInteger(metadata.algebraic_state_revision_before)},
           {"physical_endpoint_algebraic_state_revision",
            endpoint_revision(metadata.physical_endpoint_algebraic_state_revision)},
           {"algebraic_state_revision_after",
            endpoint_revision(metadata.algebraic_state_revision_after)},
           {"snapshot_set_revision_before",
            jsonInteger(metadata.snapshot_set_revision_before)},
           {"physical_endpoint_snapshot_set_revision",
            endpoint_revision(metadata.physical_endpoint_snapshot_set_revision)},
           {"snapshot_set_revision_after",
            endpoint_revision(metadata.snapshot_set_revision_after)},
           {"mesh_topology_set_revision_before",
            jsonInteger(metadata.mesh_topology_set_revision_before)},
           {"physical_endpoint_mesh_topology_set_revision",
            endpoint_revision(metadata.physical_endpoint_mesh_topology_set_revision)},
           {"mesh_topology_set_revision_after",
            endpoint_revision(metadata.mesh_topology_set_revision_after)},
           {"cut_topology_set_revision_before",
            jsonInteger(metadata.cut_topology_set_revision_before)},
           {"physical_endpoint_cut_topology_set_revision",
            endpoint_revision(metadata.physical_endpoint_cut_topology_set_revision)},
           {"cut_topology_set_revision_after",
            endpoint_revision(metadata.cut_topology_set_revision_after)},
           {"extension_map_revision_before",
            jsonOptionalUnsigned(metadata.extension_map_revision_before,
                                 "extension map unavailable")},
           {"physical_endpoint_extension_map_revision",
            jsonOptionalUnsigned(
                metadata.physical_endpoint_extension_map_revision,
                "extension map unavailable")},
           {"extension_map_revision_after",
            jsonOptionalUnsigned(metadata.extension_map_revision_after,
                                 "extension map unavailable")},
       })},
      {"stored_energy",
       jsonObject({
           {"before", storedEnergyJson(attempt.before)},
           {"physical_endpoint_before_maintenance",
            storedEnergyJson(attempt.physical_endpoint_before_maintenance)},
           {"after", storedEnergyJson(attempt.after)},
       })},
      {"dissipation_rate",
       jsonObject({
           {"bulk_viscous",
            jsonObservedReal(attempt.dissipation_rate.bulk_viscous,
                             "unstaged")},
           {"navier_slip",
            jsonObservedReal(attempt.dissipation_rate.navier_slip, "unstaged")},
           {"line_friction",
            jsonObservedReal(attempt.dissipation_rate.line_friction,
                             "unstaged")},
       })},
      {"external_work",
       jsonObject({
           {"pressure",
            jsonObservedReal(attempt.external_work.pressure, "unstaged")},
           {"body_force",
            jsonObservedReal(attempt.external_work.body_force, "unstaged")},
           {"imposed_traction",
            jsonObservedReal(attempt.external_work.imposed_traction,
                             "unstaged")},
           {"open_boundary_flux",
            jsonObservedReal(attempt.external_work.open_boundary_flux,
                             "unstaged")},
       })},
      {"numerical_work",
       jsonObject({
           {"time_discretization",
            jsonObservedReal(attempt.numerical_work.time_discretization,
                             "unstaged")},
           {"kinetic_domain_transport",
            jsonObservedReal(attempt.numerical_work.kinetic_domain_transport,
                             "unstaged")},
           {"gravitational_transport_coupling",
            jsonObservedReal(
                attempt.numerical_work.gravitational_transport_coupling,
                "unstaged")},
           {"convection",
            jsonObservedReal(attempt.numerical_work.convection, "unstaged")},
           {"pressure_continuity",
            jsonObservedReal(attempt.numerical_work.pressure_continuity,
                             "unstaged")},
           {"surface_transport_coupling",
            jsonObservedReal(attempt.numerical_work.surface_transport_coupling,
                             "unstaged")},
           {"weak_boundary",
            jsonObservedReal(attempt.numerical_work.weak_boundary, "unstaged")},
           {"vms_pspg",
            jsonObservedReal(attempt.numerical_work.vms_pspg, "unstaged")},
           {"cut_stabilization",
            jsonObservedReal(attempt.numerical_work.cut_stabilization,
                             "unstaged")},
           {"ghost_penalty",
            jsonObservedReal(attempt.numerical_work.ghost_penalty, "unstaged")},
           {"aggregation",
            jsonObservedReal(attempt.numerical_work.aggregation, "unstaged")},
           {"extension",
            jsonObservedReal(attempt.numerical_work.extension, "unstaged")},
           {"pruning",
            jsonObservedReal(attempt.numerical_work.pruning, "unstaged")},
           {"limiting",
            jsonObservedReal(attempt.numerical_work.limiting, "unstaged")},
           {"redistancing",
            jsonObservedReal(attempt.numerical_work.redistancing, "unstaged")},
           {"local_reconciliation",
            jsonObservedReal(attempt.numerical_work.local_reconciliation,
                             "unstaged")},
           {"global_correction",
            jsonObservedReal(attempt.numerical_work.global_correction,
                             "unstaged")},
       })},
      {"channel_sources",
       jsonObject({
           {"stored",
            source_group(
                sources.stored,
                std::array{
                    std::pair{"kinetic",
                              &application::core::
                                  FreeSurfaceStoredEnergySources::kinetic},
                    std::pair{
                        "gravitational",
                        &application::core::FreeSurfaceStoredEnergySources::
                            gravitational},
                    std::pair{
                        "liquid_gas_surface",
                        &application::core::FreeSurfaceStoredEnergySources::
                            liquid_gas_surface},
                    std::pair{
                        "solid_liquid_wall",
                        &application::core::FreeSurfaceStoredEnergySources::
                            solid_liquid_wall},
                    std::pair{
                        "gas_or_compressibility",
                        &application::core::FreeSurfaceStoredEnergySources::
                            gas_or_compressibility}})},
           {"dissipation",
            source_group(
                sources.dissipation,
                std::array{
                    std::pair{"bulk_viscous",
                              &application::core::
                                  FreeSurfacePhysicalDissipationSources::
                                      bulk_viscous},
                    std::pair{
                        "navier_slip",
                        &application::core::
                            FreeSurfacePhysicalDissipationSources::navier_slip},
                    std::pair{"line_friction",
                              &application::core::
                                  FreeSurfacePhysicalDissipationSources::
                                      line_friction}})},
           {"external",
            source_group(
                sources.external,
                std::array{
                    std::pair{"pressure",
                              &application::core::
                                  FreeSurfaceExternalWorkSources::pressure},
                    std::pair{"body_force",
                              &application::core::
                                  FreeSurfaceExternalWorkSources::body_force},
                    std::pair{
                        "imposed_traction",
                        &application::core::FreeSurfaceExternalWorkSources::
                            imposed_traction},
                    std::pair{
                        "open_boundary_flux",
                        &application::core::FreeSurfaceExternalWorkSources::
                            open_boundary_flux}})},
           {"numerical",
            source_group(
                sources.numerical,
                std::array{
                    std::pair{
                        "time_discretization",
                        &application::core::FreeSurfaceNumericalWorkSources::
                            time_discretization},
                    std::pair{
                        "kinetic_domain_transport",
                        &application::core::FreeSurfaceNumericalWorkSources::
                            kinetic_domain_transport},
                    std::pair{
                        "gravitational_transport_coupling",
                        &application::core::FreeSurfaceNumericalWorkSources::
                            gravitational_transport_coupling},
                    std::pair{"convection",
                              &application::core::
                                  FreeSurfaceNumericalWorkSources::convection},
                    std::pair{
                        "pressure_continuity",
                        &application::core::FreeSurfaceNumericalWorkSources::
                            pressure_continuity},
                    std::pair{
                        "surface_transport_coupling",
                        &application::core::FreeSurfaceNumericalWorkSources::
                            surface_transport_coupling},
                    std::pair{
                        "weak_boundary",
                        &application::core::FreeSurfaceNumericalWorkSources::
                            weak_boundary},
                    std::pair{"vms_pspg",
                              &application::core::
                                  FreeSurfaceNumericalWorkSources::vms_pspg},
                    std::pair{
                        "cut_stabilization",
                        &application::core::FreeSurfaceNumericalWorkSources::
                            cut_stabilization},
                    std::pair{
                        "ghost_penalty",
                        &application::core::FreeSurfaceNumericalWorkSources::
                            ghost_penalty},
                    std::pair{"aggregation",
                              &application::core::
                                  FreeSurfaceNumericalWorkSources::aggregation},
                    std::pair{"extension",
                              &application::core::
                                  FreeSurfaceNumericalWorkSources::extension},
                    std::pair{"pruning",
                              &application::core::
                                  FreeSurfaceNumericalWorkSources::pruning},
                    std::pair{"limiting",
                              &application::core::
                                  FreeSurfaceNumericalWorkSources::limiting},
                    std::pair{
                        "redistancing",
                        &application::core::FreeSurfaceNumericalWorkSources::
                            redistancing},
                    std::pair{
                        "local_reconciliation",
                        &application::core::FreeSurfaceNumericalWorkSources::
                            local_reconciliation},
                    std::pair{
                        "global_correction",
                        &application::core::FreeSurfaceNumericalWorkSources::
                            global_correction}})},
       })},
      {"derived_balance",
       jsonObject({
           {"stored_energy_before",
            jsonObservedReal(attempt.stored_energy_before, "unstaged")},
           {"stored_energy_physical_endpoint_before_maintenance",
            jsonObservedReal(
                attempt.stored_energy_physical_endpoint_before_maintenance,
                "unstaged")},
           {"stored_energy_after",
            jsonObservedReal(attempt.stored_energy_after, "unstaged")},
           {"physical_stored_energy_change",
            jsonObservedReal(attempt.physical_stored_energy_change,
                             "unstaged")},
           {"maintenance_stored_energy_change",
            jsonObservedReal(attempt.maintenance_stored_energy_change,
                             "unstaged")},
           {"stored_energy_change",
            jsonObservedReal(attempt.stored_energy_change, "unstaged")},
           {"integrated_physical_dissipation",
            jsonObservedReal(attempt.integrated_physical_dissipation,
                             "unstaged")},
           {"total_external_work",
            jsonObservedReal(attempt.total_external_work, "unstaged")},
           {"total_numerical_work",
            jsonObservedReal(attempt.total_numerical_work, "unstaged")},
           {"trial_balance_residual",
            jsonObservedReal(attempt.trial_balance_residual, "unstaged")},
       })},
      {"accepted_contribution",
       jsonObject({
           {"stored_energy_change",
            jsonReal(attempt.accepted_stored_energy_change)},
           {"physical_stored_energy_change",
            jsonReal(attempt.accepted_physical_stored_energy_change)},
           {"maintenance_stored_energy_change",
            jsonReal(attempt.accepted_maintenance_stored_energy_change)},
           {"integrated_physical_dissipation",
            jsonReal(attempt.accepted_integrated_physical_dissipation)},
           {"external_work", jsonReal(attempt.accepted_external_work)},
           {"numerical_work", jsonReal(attempt.accepted_numerical_work)},
           {"balance_residual", jsonReal(attempt.accepted_balance_residual)},
       })},
  });
}

} // namespace application_test::free_surface_lifecycle_capture
