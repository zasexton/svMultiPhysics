/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Options/BoundaryDataReaders.h"

#include "FE/Core/Logger.h"
#include "FE/Systems/FESystem.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Topology/CellTopology.h"

#include <fstream>
#include <stdexcept>
#include <unordered_set>

namespace svmp::Physics::options {

namespace {

std::unordered_set<svmp::gid_t> collect_temporal_spatial_target_vertex_gids(
    const svmp::MeshBase& mesh,
    int boundary_marker)
{
  std::unordered_set<svmp::gid_t> gids;
  const auto& vgids = mesh.vertex_gids();
  if (boundary_marker < 0) {
    gids.insert(vgids.begin(), vgids.end());
    return gids;
  }
  const auto add_vertex = [&](svmp::index_t v) {
    if (v < 0) {
      return;
    }
    const auto idx = static_cast<std::size_t>(v);
    if (idx >= vgids.size()) {
      return;
    }
    gids.insert(vgids[idx]);
  };
  const auto sorted_unique = [](std::vector<svmp::index_t> values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
  };

  const auto faces = mesh.faces_with_label(static_cast<svmp::label_t>(boundary_marker));
  for (const auto f : faces) {
    const auto stored_vertices = mesh.face_vertices(f);
    for (const auto v : stored_vertices) {
      add_vertex(v);
    }

    auto stored_key = sorted_unique(stored_vertices);
    if (stored_key.empty() || static_cast<std::size_t>(f) >= mesh.face2cell().size()) {
      continue;
    }

    const auto& adjacent = mesh.face2cell()[static_cast<std::size_t>(f)];
    for (const auto cell : adjacent) {
      if (cell == svmp::INVALID_INDEX || static_cast<std::size_t>(cell) >= mesh.n_cells()) {
        continue;
      }
      const auto& shape = mesh.cell_shape(cell);
      const auto face_view = svmp::CellTopology::get_oriented_boundary_faces_view(shape.family);
      if (!face_view.indices || !face_view.offsets || face_view.face_count <= 0) {
        continue;
      }

      for (int lf = 0; lf < face_view.face_count; ++lf) {
        std::vector<svmp::index_t> geometry_vertices;
        try {
          geometry_vertices = mesh.cell_face_geometry_dofs(cell, lf);
        } catch (const std::exception&) {
          geometry_vertices.clear();
        }
        if (geometry_vertices.empty()) {
          const auto [cell_vertices, n_cell_vertices] = mesh.cell_vertices_span(cell);
          const int begin = face_view.offsets[lf];
          const int end = face_view.offsets[lf + 1];
          for (int j = begin; j < end; ++j) {
            const auto local = face_view.indices[j];
            if (local >= 0 && static_cast<std::size_t>(local) < n_cell_vertices) {
              geometry_vertices.push_back(cell_vertices[local]);
            }
          }
        }

        auto geometry_key = sorted_unique(geometry_vertices);
        if (!std::includes(geometry_key.begin(), geometry_key.end(),
                           stored_key.begin(), stored_key.end())) {
          continue;
        }
        for (const auto v : geometry_vertices) {
          add_vertex(v);
        }
        break;
      }
    }
  }
  return gids;
}

} // namespace

std::shared_ptr<TemporalSpatialValues> read_temporal_and_spatial_values_file(const svmp::MeshBase& mesh,
                                                                            int boundary_marker,
                                                                            const std::string& file_path,
                                                                            MarkerCommunicator comm)
{
  std::ifstream in(file_path);
  if (!in.is_open()) {
    throw std::runtime_error("[svMultiPhysics::Physics] Failed to open temporal/spatial BC file '" + file_path + "'.");
  }

  int ndof = 0;
  int num_ts = 0;
  int num_nodes = 0;
  in >> ndof >> num_ts >> num_nodes;
  if (ndof <= 0 || num_ts <= 0 || num_nodes <= 0) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Invalid header in temporal/spatial BC file '" + file_path +
        "' (expected: <ndof> <num_ts> <num_nodes>).");
  }

  const int dim = mesh.dim();
  if (dim < 1 || dim > 3) {
    throw std::runtime_error("[svMultiPhysics::Physics] Invalid mesh dimension for temporal/spatial BC parsing.");
  }
  if (ndof > dim) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path + "' specifies ndof=" + std::to_string(ndof) +
        ", but mesh dimension is " + std::to_string(dim) + ".");
  }

  const auto target_gids =
      collect_temporal_spatial_target_vertex_gids(mesh, boundary_marker);
  if (mesh.vertex_gids().size() != mesh.n_vertices()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path +
        "' requires stable global vertex IDs on the boundary mesh.");
  }

  auto out = std::make_shared<TemporalSpatialValues>();
  out->dim = dim;
  out->dof = ndof;
  out->num_time_points = num_ts;
  out->boundary_marker = boundary_marker;
  out->file_path = file_path;
  out->t.resize(static_cast<std::size_t>(num_ts));

  // Time sequence (t0 must be 0 and increasing).
  for (int i = 0; i < num_ts; ++i) {
    double ti = 0.0;
    if (!(in >> ti) || !std::isfinite(ti)) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path +
          "' contains a missing or nonfinite time value at index " +
          std::to_string(i) + ".");
    }
    out->t[static_cast<std::size_t>(i)] = ti;
    if (i == 0) {
      if (std::abs(ti) > 1e-14) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path + "': first time value must be 0.");
      }
    } else {
      const double dt = ti - out->t[static_cast<std::size_t>(i - 1)];
      if (!(dt > 0.0)) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path + "': time values must be increasing.");
      }
    }
  }
  out->period = out->t.back();

  struct FileNodeRecord {
    long long file_node_id{0};
    std::vector<svmp::FE::Real> values{};
  };
  std::vector<FileNodeRecord> file_nodes;
  file_nodes.reserve(static_cast<std::size_t>(num_nodes));
  std::unordered_set<long long> seen_file_node_ids;
  for (int b = 0; b < num_nodes; ++b) {
    FileNodeRecord record;
    if (!(in >> record.file_node_id)) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path +
          "' ended before node record " + std::to_string(b + 1) + ".");
    }
    if (record.file_node_id <= 0) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path +
          "' contains a nonpositive node id: " + std::to_string(record.file_node_id) + ".");
    }
    if (!seen_file_node_ids.insert(record.file_node_id).second) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path +
          "' contains duplicate node id " + std::to_string(record.file_node_id) + ".");
    }

    record.values.reserve(static_cast<std::size_t>(num_ts) * static_cast<std::size_t>(ndof));
    for (int i = 0; i < num_ts; ++i) {
      for (int k = 0; k < ndof; ++k) {
        double value = 0.0;
        if (!(in >> value) || !std::isfinite(value)) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path +
              "' contains a missing or nonfinite value for node id " +
              std::to_string(record.file_node_id) + ".");
        }
        record.values.push_back(static_cast<svmp::FE::Real>(value));
      }
    }
    file_nodes.push_back(std::move(record));
  }
  std::string trailing_token;
  if (in >> trailing_token) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path +
        "' contains trailing data after " + std::to_string(num_nodes) +
        " node records (first unexpected token: '" + trailing_token + "').");
  }

  enum class NodeIdConvention {
    LegacyOneBasedOrdinal,
    ImportedGlobalId,
  };
  struct NodeIdCoverage {
    NodeIdConvention convention{NodeIdConvention::LegacyOneBasedOrdinal};
    std::vector<svmp::gid_t> mapped_gids{};
    std::size_t globally_matched_nodes{0};
    bool covers_global_boundary{false};
    bool exact{false};
  };

  const auto evaluate_convention = [&](NodeIdConvention convention) {
    NodeIdCoverage coverage;
    coverage.convention = convention;
    coverage.mapped_gids.reserve(file_nodes.size());
    std::unordered_set<svmp::gid_t> mapped_gid_set;
    mapped_gid_set.reserve(file_nodes.size());
    for (const auto& record : file_nodes) {
      const auto mapped =
          convention == NodeIdConvention::ImportedGlobalId
              ? static_cast<svmp::gid_t>(record.file_node_id)
              : static_cast<svmp::gid_t>(record.file_node_id - 1);
      coverage.mapped_gids.push_back(mapped);
      mapped_gid_set.insert(mapped);
    }

    std::vector<int> globally_matched(file_nodes.size(), 0);
    for (std::size_t i = 0; i < coverage.mapped_gids.size(); ++i) {
      globally_matched[i] = target_gids.count(coverage.mapped_gids[i]) != 0u ? 1 : 0;
    }
    int covers_global_boundary = std::all_of(
                                      target_gids.begin(),
                                      target_gids.end(),
                                      [&](svmp::gid_t gid) {
                                        return mapped_gid_set.count(gid) != 0u;
                                      })
                                      ? 1
                                      : 0;
#if FE_HAS_MPI
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
      MPI_Finalized(&mpi_finalized);
    }
    if (mpi_initialized && !mpi_finalized && comm != MPI_COMM_NULL) {
      MPI_Allreduce(
          MPI_IN_PLACE,
          globally_matched.data(),
          static_cast<int>(globally_matched.size()),
          MPI_INT,
          MPI_MAX,
          comm);
      MPI_Allreduce(
          MPI_IN_PLACE,
          &covers_global_boundary,
          1,
          MPI_INT,
          MPI_MIN,
          comm);
    }
#endif
    coverage.globally_matched_nodes = static_cast<std::size_t>(
        std::count(globally_matched.begin(), globally_matched.end(), 1));
    coverage.covers_global_boundary = covers_global_boundary != 0;
    coverage.exact =
        coverage.covers_global_boundary && coverage.globally_matched_nodes == file_nodes.size();
    return coverage;
  };

  const auto legacy_coverage = evaluate_convention(NodeIdConvention::LegacyOneBasedOrdinal);
  const auto imported_coverage = evaluate_convention(NodeIdConvention::ImportedGlobalId);
  if (legacy_coverage.exact == imported_coverage.exact) {
    std::ostringstream message;
    message << "[svMultiPhysics::Physics] Temporal/spatial BC file '" << file_path << "' node IDs ";
    if (legacy_coverage.exact) {
      message << "are ambiguous";
    } else {
      message << "do not exactly cover ";
      if (boundary_marker < 0) {
        message << "the mesh vertex set";
      } else {
        message << "boundary marker " << boundary_marker;
      }
    }
    message << " under either the legacy one-based ordinal convention or the imported global-ID convention"
            << " (file_nodes=" << file_nodes.size()
            << ", legacy_matched=" << legacy_coverage.globally_matched_nodes
            << ", legacy_covers_boundary=" << (legacy_coverage.covers_global_boundary ? 1 : 0)
            << ", imported_matched=" << imported_coverage.globally_matched_nodes
            << ", imported_covers_boundary=" << (imported_coverage.covers_global_boundary ? 1 : 0) << ").";
    throw std::runtime_error(message.str());
  }
  const auto& selected_coverage = imported_coverage.exact ? imported_coverage : legacy_coverage;

  out->node_ids.clear();
  out->coords.clear();
  out->d.clear();
  out->node_ids.reserve(static_cast<std::size_t>(num_nodes));
  out->coords.reserve(static_cast<std::size_t>(num_nodes));
  out->d.reserve(static_cast<std::size_t>(num_nodes) * static_cast<std::size_t>(num_ts) * static_cast<std::size_t>(ndof));

  int missing_local_vertex_count = 0;
  int non_boundary_file_node_count = 0;
  for (std::size_t b = 0; b < file_nodes.size(); ++b) {
    const auto node_gid = selected_coverage.mapped_gids[b];
    const auto node_idx = mesh.global_to_local_vertex(node_gid);

    const bool has_local_vertex = node_idx != svmp::INVALID_INDEX;
    const bool is_boundary_node = target_gids.count(node_gid) != 0u;
    if (!has_local_vertex) {
      ++missing_local_vertex_count;
    } else if (!is_boundary_node) {
      ++non_boundary_file_node_count;
    }
    const bool keep = has_local_vertex && is_boundary_node;
    if (keep) {
      out->node_ids.push_back(node_idx);

      const auto& X = mesh.X_ref();
      std::array<svmp::FE::Real, 3> p{0.0, 0.0, 0.0};
      const auto base = static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(dim);
      p[0] = static_cast<svmp::FE::Real>(X.at(base + 0));
      if (dim >= 2) {
        p[1] = static_cast<svmp::FE::Real>(X.at(base + 1));
      }
      if (dim >= 3) {
        p[2] = static_cast<svmp::FE::Real>(X.at(base + 2));
      }
      out->coords.push_back(p);

      const auto stored_idx = out->coords.size() - 1;
      out->node_index_by_key.emplace(TemporalSpatialValues::quantize(p, dim), stored_idx);
    }

    if (keep) {
      out->d.insert(out->d.end(), file_nodes[b].values.begin(), file_nodes[b].values.end());
    }
  }

  if (temporalSpatialBcTraceEnabled()) {
    int rank = 0;
#if FE_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    int finalized = 0;
    if (initialized) {
      MPI_Finalized(&finalized);
    }
    if (initialized && !finalized && comm != MPI_COMM_NULL) {
      MPI_Comm_rank(comm, &rank);
    }
#endif
    std::ostringstream oss;
    oss << "TemporalSpatialValues: rank=" << rank
        << " marker=" << boundary_marker
        << " file='" << file_path << "'"
        << " ndof=" << ndof
        << " time_points=" << num_ts
        << " file_nodes=" << num_nodes
        << " kept_nodes=" << out->coords.size()
        << " mapping_scope="
        << (boundary_marker < 0 ? "mesh_vertices" : "boundary_marker")
        << " target_nodes=" << target_gids.size()
        << " missing_local_vertex_nodes=" << missing_local_vertex_count
        << " non_boundary_file_nodes=" << non_boundary_file_node_count
        << " node_id_convention="
        << (selected_coverage.convention == NodeIdConvention::ImportedGlobalId ? "imported_global_id"
                                                                               : "legacy_one_based_ordinal")
        << " legacy_matched_nodes=" << legacy_coverage.globally_matched_nodes
        << " imported_matched_nodes=" << imported_coverage.globally_matched_nodes;
    navierStokesTraceLog(oss.str());
  }

  out->buildInterpolationMetadata();

  return out;
}

svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::NodePressureConstraintIdType
parse_node_pressure_constraint_id_type(std::string_view raw)
{
  using Options = svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions;

  const auto value = trim_copy(std::string(raw));
  if (value == "Global_vertex_gid") {
    return Options::NodePressureConstraintIdType::GlobalVertexGid;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] Unsupported <Node_pressure_constraints><Id_type> '" + value +
      "'. Supported value: Global_vertex_gid.");
}

std::vector<svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::NodePressureConstraint>
read_node_pressure_constraints_csv(const std::string& path)
{
  using Options = svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions;

  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("[svMultiPhysics::Physics] Failed to open node pressure constraints CSV file '" +
                             path + "'.");
  }

  std::vector<Options::NodePressureConstraint> out;
  std::unordered_map<svmp::FE::GlobalIndex, svmp::FE::Real> seen;

  bool header_seen = false;
  bool data_seen = false;
  int node_col = 0;
  int pressure_col = 1;

  std::string line;
  int line_number = 0;
  while (std::getline(file, line)) {
    ++line_number;
    const auto trimmed = trim_copy(line);
    if (trimmed.empty() || trimmed.front() == '#') {
      continue;
    }

    const auto fields = split_csv_line(trimmed);
    if (fields.size() < 2u) {
      throw std::runtime_error("[svMultiPhysics::Physics] Malformed node pressure CSV row in '" + path +
                               "' at line " + std::to_string(line_number) +
                               ": expected at least two comma-separated fields.");
    }

    if (!header_seen) {
      const auto c0 = lower_copy(fields[0]);
      const auto c1 = lower_copy(fields[1]);
      if (c0 == "node_id" || c0 == "pressure" || c1 == "node_id" || c1 == "pressure") {
        if (c0 == "node_id") {
          node_col = 0;
        } else if (c1 == "node_id") {
          node_col = 1;
        } else {
          throw std::runtime_error("[svMultiPhysics::Physics] Node pressure CSV header in '" + path +
                                   "' must contain a node_id column.");
        }

        if (c0 == "pressure") {
          pressure_col = 0;
        } else if (c1 == "pressure") {
          pressure_col = 1;
        } else {
          throw std::runtime_error("[svMultiPhysics::Physics] Node pressure CSV header in '" + path +
                                   "' must contain a pressure column.");
        }

        header_seen = true;
        continue;
      }
      header_seen = true;
    }

    const auto max_col = static_cast<std::size_t>(std::max(node_col, pressure_col));
    if (fields.size() <= max_col) {
      throw std::runtime_error("[svMultiPhysics::Physics] Malformed node pressure CSV row in '" + path +
                               "' at line " + std::to_string(line_number) +
                               ": missing node_id or pressure field.");
    }

    const auto context = std::string("node pressure CSV '") + path + "' line " + std::to_string(line_number);
    const auto node_id = parse_global_index(fields[static_cast<std::size_t>(node_col)], context + " node_id");
    const auto pressure_value = parse_double(fields[static_cast<std::size_t>(pressure_col)], context + " pressure");
    if (!std::isfinite(pressure_value)) {
      throw std::runtime_error("[svMultiPhysics::Physics] Non-finite pressure value in '" + path +
                               "' at line " + std::to_string(line_number) + ".");
    }
    const auto pressure = static_cast<svmp::FE::Real>(pressure_value);

    const auto it = seen.find(node_id);
    if (it != seen.end()) {
      if (!values_match(it->second, pressure)) {
        throw std::runtime_error("[svMultiPhysics::Physics] Conflicting duplicate node pressure value for node_id " +
                                 std::to_string(node_id) + " in '" + path + "'.");
      }
      continue;
    }

    seen.emplace(node_id, pressure);
    out.push_back(Options::NodePressureConstraint{node_id, pressure});
    data_seen = true;
  }

  if (!data_seen) {
    throw std::runtime_error("[svMultiPhysics::Physics] Node pressure constraints CSV file '" + path +
                             "' did not contain any node pressure values.");
  }

  return out;
}

MarkerGeometry local_marker_geometry(const svmp::MeshBase& mesh, int boundary_marker)
{
  MarkerGeometry out{};
  const auto faces = mesh.faces_with_label(static_cast<svmp::label_t>(boundary_marker));
  for (const auto f : faces) {
    const double a = static_cast<double>(mesh.face_area(f));
    const auto c = to_vec3(mesh.face_center(f));
    const auto n = to_vec3(mesh.face_normal(f));
    out.area += a;
    out.center_sum = out.center_sum + (a * c);
    out.normal_sum = out.normal_sum + (a * n);
  }
  return out;
}

[[nodiscard]] MarkerCommunicator markerCommunicator(
    const svmp::FE::systems::FESystem& system) noexcept
{
#if FE_HAS_MPI
  return system.activeMpiCommunicator();
#else
  (void)system;
  return 0;
#endif
}

namespace {

struct GidPairHash {
  std::size_t operator()(const std::pair<svmp::gid_t, svmp::gid_t>& p) const noexcept
  {
    const std::size_t h1 = std::hash<svmp::gid_t>{}(p.first);
    const std::size_t h2 = std::hash<svmp::gid_t>{}(p.second);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

struct GidSequenceHash {
  std::size_t operator()(const std::vector<svmp::gid_t>& values) const noexcept
  {
    std::uint64_t hash = 1469598103934665603ull;
    for (const auto value : values) {
      const auto mixed = static_cast<std::uint64_t>(std::hash<svmp::gid_t>{}(value));
      hash ^= mixed + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    }
    hash ^= static_cast<std::uint64_t>(values.size()) + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    return static_cast<std::size_t>(hash);
  }
};

[[nodiscard]] bool mpiMultiRankActive(MarkerCommunicator comm) noexcept
{
#if FE_HAS_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    return false;
  }

  int comm_size = 1;
  MPI_Comm_size(comm, &comm_size);
  return comm_size > 1;
#else
  (void)comm;
  return false;
#endif
}

template <std::size_t N>
struct MarkerFacePayload {
  std::vector<svmp::gid_t> ordered_vertex_gids{};
  std::array<double, N> values{};
};

template <std::size_t N, class PayloadBuilder>
std::vector<MarkerFacePayload<N>> gather_unique_marker_face_payloads(
    const svmp::MeshBase& mesh,
    int boundary_marker,
    MarkerCommunicator comm,
    PayloadBuilder&& payload_builder)
{
  const bool mpi_active = mpiMultiRankActive(comm);
  const auto& vgids = mesh.vertex_gids();
  if (mpi_active && vgids.size() != mesh.n_vertices()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] MPI boundary-face normalization for marker " +
        std::to_string(boundary_marker) + " requires stable vertex GIDs.");
  }

  std::vector<MarkerFacePayload<N>> local_payloads;
  const auto faces = mesh.faces_with_label(static_cast<svmp::label_t>(boundary_marker));
  local_payloads.reserve(faces.size());

  for (const auto f : faces) {
    MarkerFacePayload<N> payload;
    const auto verts = mesh.face_vertices(f);
    payload.ordered_vertex_gids.reserve(verts.size());
    for (const auto v : verts) {
      if (v == svmp::INVALID_INDEX) {
        continue;
      }
      const auto idx = static_cast<std::size_t>(v);
      if (idx >= mesh.n_vertices()) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Boundary face references an out-of-range vertex while processing marker " +
            std::to_string(boundary_marker) + ".");
      }
      const auto gid = (vgids.size() == mesh.n_vertices())
                           ? vgids[idx]
                           : static_cast<svmp::gid_t>(v);
      payload.ordered_vertex_gids.push_back(gid);
    }
    if (payload.ordered_vertex_gids.size() < 2u) {
      continue;
    }
    payload_builder(f, payload.values);
    local_payloads.push_back(std::move(payload));
  }

  auto deduplicate = [](std::vector<MarkerFacePayload<N>> payloads) {
    std::unordered_set<std::vector<svmp::gid_t>, GidSequenceHash> seen;
    std::vector<MarkerFacePayload<N>> unique;
    unique.reserve(payloads.size());
    for (auto& payload : payloads) {
      auto key = payload.ordered_vertex_gids;
      std::sort(key.begin(), key.end());
      key.erase(std::unique(key.begin(), key.end()), key.end());
      if (key.size() < 2u) {
        continue;
      }
      if (!seen.emplace(std::move(key)).second) {
        continue;
      }
      unique.push_back(std::move(payload));
    }
    return unique;
  };

  if (!mpi_active) {
    return deduplicate(std::move(local_payloads));
  }

#if FE_HAS_MPI
  int comm_size = 1;
  MPI_Comm_size(comm, &comm_size);

  const int local_face_count = static_cast<int>(local_payloads.size());
  std::vector<int> face_counts(static_cast<std::size_t>(comm_size), 0);
  MPI_Allgather(&local_face_count, 1, MPI_INT, face_counts.data(), 1, MPI_INT, comm);

  std::vector<int> face_displs(static_cast<std::size_t>(comm_size), 0);
  int total_faces = 0;
  for (int r = 0; r < comm_size; ++r) {
    face_displs[static_cast<std::size_t>(r)] = total_faces;
    total_faces += face_counts[static_cast<std::size_t>(r)];
  }

  std::vector<int> local_gid_sizes;
  local_gid_sizes.reserve(local_payloads.size());
  std::vector<svmp::gid_t> local_gid_data;
  for (const auto& payload : local_payloads) {
    local_gid_sizes.push_back(static_cast<int>(payload.ordered_vertex_gids.size()));
    local_gid_data.insert(local_gid_data.end(),
                          payload.ordered_vertex_gids.begin(),
                          payload.ordered_vertex_gids.end());
  }

  const int local_gid_count = static_cast<int>(local_gid_data.size());
  std::vector<int> gid_counts(static_cast<std::size_t>(comm_size), 0);
  MPI_Allgather(&local_gid_count, 1, MPI_INT, gid_counts.data(), 1, MPI_INT, comm);

  std::vector<int> gid_displs(static_cast<std::size_t>(comm_size), 0);
  int total_gid_count = 0;
  for (int r = 0; r < comm_size; ++r) {
    gid_displs[static_cast<std::size_t>(r)] = total_gid_count;
    total_gid_count += gid_counts[static_cast<std::size_t>(r)];
  }

  std::vector<int> all_gid_sizes(static_cast<std::size_t>(total_faces), 0);
  MPI_Allgatherv(local_gid_sizes.data(),
                 local_face_count,
                 MPI_INT,
                 all_gid_sizes.data(),
                 face_counts.data(),
                 face_displs.data(),
                 MPI_INT,
                 comm);

  std::vector<svmp::gid_t> all_gid_data(static_cast<std::size_t>(total_gid_count), svmp::gid_t{0});
  MPI_Allgatherv(local_gid_data.data(),
                 local_gid_count,
                 MPI_INT64_T,
                 all_gid_data.data(),
                 gid_counts.data(),
                 gid_displs.data(),
                 MPI_INT64_T,
                 comm);

  std::vector<double> local_values;
  local_values.reserve(local_payloads.size() * N);
  for (const auto& payload : local_payloads) {
    local_values.insert(local_values.end(), payload.values.begin(), payload.values.end());
  }

  std::vector<int> value_counts(static_cast<std::size_t>(comm_size), 0);
  std::vector<int> value_displs(static_cast<std::size_t>(comm_size), 0);
  int total_value_count = 0;
  for (int r = 0; r < comm_size; ++r) {
    value_displs[static_cast<std::size_t>(r)] = total_value_count;
    value_counts[static_cast<std::size_t>(r)] = face_counts[static_cast<std::size_t>(r)] * static_cast<int>(N);
    total_value_count += value_counts[static_cast<std::size_t>(r)];
  }

  std::vector<double> all_values(static_cast<std::size_t>(total_value_count), 0.0);
  MPI_Allgatherv(local_values.data(),
                 static_cast<int>(local_values.size()),
                 MPI_DOUBLE,
                 all_values.data(),
                 value_counts.data(),
                 value_displs.data(),
                 MPI_DOUBLE,
                 comm);

  std::vector<MarkerFacePayload<N>> gathered;
  gathered.reserve(static_cast<std::size_t>(total_faces));
  std::size_t gid_offset = 0u;
  for (int i = 0; i < total_faces; ++i) {
    const auto count = static_cast<std::size_t>(all_gid_sizes[static_cast<std::size_t>(i)]);
    if (gid_offset + count > all_gid_data.size()) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Corrupt MPI face-GID gather while processing marker " +
          std::to_string(boundary_marker) + ".");
    }
    MarkerFacePayload<N> payload;
    payload.ordered_vertex_gids.assign(all_gid_data.begin() + static_cast<std::ptrdiff_t>(gid_offset),
                                       all_gid_data.begin() + static_cast<std::ptrdiff_t>(gid_offset + count));
    gid_offset += count;

    const std::size_t value_offset = static_cast<std::size_t>(i) * N;
    if (value_offset + N > all_values.size()) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Corrupt MPI face-payload gather while processing marker " +
          std::to_string(boundary_marker) + ".");
    }
    std::copy_n(all_values.begin() + static_cast<std::ptrdiff_t>(value_offset), N, payload.values.begin());
    gathered.push_back(std::move(payload));
  }

  return deduplicate(std::move(gathered));
#else
  return deduplicate(std::move(local_payloads));
#endif
}

} // namespace

MarkerGeometry global_marker_geometry(const svmp::MeshBase& mesh,
                                      int boundary_marker,
                                      MarkerCommunicator comm)
{
  MarkerGeometry out{};
  const auto payloads = gather_unique_marker_face_payloads<7>(
      mesh, boundary_marker, comm,
      [&](svmp::index_t f, std::array<double, 7>& values) {
        const double a = static_cast<double>(mesh.face_area(f));
        const auto c = to_vec3(mesh.face_center(f));
        const auto n = to_vec3(mesh.face_normal(f));
        values[0] = a;
        values[1] = a * c.x;
        values[2] = a * c.y;
        values[3] = a * c.z;
        values[4] = a * n.x;
        values[5] = a * n.y;
        values[6] = a * n.z;
      });
  for (const auto& payload : payloads) {
    out.area += payload.values[0];
    out.center_sum = out.center_sum + Vec3d{payload.values[1], payload.values[2], payload.values[3]};
    out.normal_sum = out.normal_sum + Vec3d{payload.values[4], payload.values[5], payload.values[6]};
  }
  return out;
}

namespace {

std::vector<std::pair<svmp::gid_t, Vec3d>> gather_perimeter_vertex_coords(const svmp::MeshBase& mesh,
                                                                          int boundary_marker,
                                                                          MarkerCommunicator comm)
{
  const auto& vgids = mesh.vertex_gids();
  if (vgids.size() != mesh.n_vertices()) {
    return {};
  }

  // Determine the perimeter of the marker surface as the set of boundary edges of the
  // marker patch itself (edges that appear only once among unique marker faces).
  const auto marker_faces = gather_unique_marker_face_payloads<1>(
      mesh, boundary_marker, comm,
      [](svmp::index_t /*face*/, std::array<double, 1>& values) {
        values[0] = 0.0;
      });

  std::vector<svmp::gid_t> local_edge_gids;
  local_edge_gids.reserve(marker_faces.size() * 6u);

  for (const auto& face : marker_faces) {
    const auto& face_vgids = face.ordered_vertex_gids;
    for (std::size_t i = 0; i < face_vgids.size(); ++i) {
      svmp::gid_t a = face_vgids[i];
      svmp::gid_t b = face_vgids[(i + 1u) % face_vgids.size()];
      if (a == b) {
        continue;
      }
      if (a > b) {
        std::swap(a, b);
      }
      local_edge_gids.push_back(a);
      local_edge_gids.push_back(b);
    }
  }

  const auto& all_edge_gids = local_edge_gids;

  std::unordered_map<std::pair<svmp::gid_t, svmp::gid_t>, int, GidPairHash> edge_counts;
  edge_counts.reserve(all_edge_gids.size() / 2u);
  for (std::size_t i = 0; i + 1u < all_edge_gids.size(); i += 2u) {
    const auto key = std::make_pair(all_edge_gids[i], all_edge_gids[i + 1u]);
    ++edge_counts[key];
  }

  std::unordered_set<svmp::gid_t> perimeter_gids;
  perimeter_gids.reserve(edge_counts.size() * 2u);
  for (const auto& [e, count] : edge_counts) {
    if (count == 1) {
      perimeter_gids.insert(e.first);
      perimeter_gids.insert(e.second);
    }
  }

  std::vector<svmp::gid_t> local_gids;
  std::vector<double> local_xyz;
  local_gids.reserve(perimeter_gids.size());
  local_xyz.reserve(perimeter_gids.size() * 3u);

  for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh.n_vertices()); ++v) {
    const auto idx = static_cast<std::size_t>(v);
    if (idx >= vgids.size()) {
      continue;
    }
    const auto gid = vgids[idx];
    if (perimeter_gids.find(gid) == perimeter_gids.end()) {
      continue;
    }
    local_gids.push_back(gid);
    const auto p = to_vec3(mesh.get_vertex_coords(v));
    local_xyz.push_back(p.x);
    local_xyz.push_back(p.y);
    local_xyz.push_back(p.z);
  }

  std::vector<svmp::gid_t> all_gids = local_gids;
  std::vector<double> all_xyz = local_xyz;

#if FE_HAS_MPI
  if (mpiMultiRankActive(comm)) {
    int comm_size = 1;
    MPI_Comm_size(comm, &comm_size);

    const int local_gid_count = static_cast<int>(local_gids.size());
    std::vector<int> gid_counts(static_cast<std::size_t>(comm_size), 0);
    MPI_Allgather(&local_gid_count, 1, MPI_INT, gid_counts.data(), 1, MPI_INT, comm);

    std::vector<int> gid_displs(static_cast<std::size_t>(comm_size), 0);
    int total_gid_count = 0;
    for (int r = 0; r < comm_size; ++r) {
      gid_displs[static_cast<std::size_t>(r)] = total_gid_count;
      total_gid_count += gid_counts[static_cast<std::size_t>(r)];
    }

    all_gids.assign(static_cast<std::size_t>(total_gid_count), svmp::gid_t{0});
    MPI_Allgatherv(local_gids.data(),
                   local_gid_count,
                   MPI_INT64_T,
                   all_gids.data(),
                   gid_counts.data(),
                   gid_displs.data(),
                   MPI_INT64_T,
                   comm);

    std::vector<int> xyz_counts(static_cast<std::size_t>(comm_size), 0);
    std::vector<int> xyz_displs(static_cast<std::size_t>(comm_size), 0);
    int total_xyz_count = 0;
    for (int r = 0; r < comm_size; ++r) {
      xyz_displs[static_cast<std::size_t>(r)] = total_xyz_count;
      xyz_counts[static_cast<std::size_t>(r)] = gid_counts[static_cast<std::size_t>(r)] * 3;
      total_xyz_count += xyz_counts[static_cast<std::size_t>(r)];
    }

    all_xyz.assign(static_cast<std::size_t>(total_xyz_count), 0.0);
    MPI_Allgatherv(local_xyz.data(),
                   static_cast<int>(local_xyz.size()),
                   MPI_DOUBLE,
                   all_xyz.data(),
                   xyz_counts.data(),
                   xyz_displs.data(),
                   MPI_DOUBLE,
                   comm);
  }
#endif

  std::vector<std::pair<svmp::gid_t, Vec3d>> out;
  out.reserve(all_gids.size());
  for (std::size_t i = 0; i < all_gids.size(); ++i) {
    const std::size_t j = i * 3u;
    if (j + 2u >= all_xyz.size()) {
      break;
    }
    out.emplace_back(all_gids[i], Vec3d{all_xyz[j + 0], all_xyz[j + 1], all_xyz[j + 2]});
  }
  return out;
}

} // namespace

ParabolicProfileData build_parabolic_profile_data(const svmp::MeshBase& mesh,
                                                  int boundary_marker,
                                                  MarkerCommunicator comm)
{
  const auto g = global_marker_geometry(mesh, boundary_marker, comm);
  if (!(g.area > 0.0)) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Boundary marker " + std::to_string(boundary_marker) +
        " has zero area; cannot construct parabolic profile.");
  }
  ParabolicProfileData out{};
  out.center = (1.0 / g.area) * g.center_sum;

  const auto all_perim =
      gather_perimeter_vertex_coords(mesh, boundary_marker, comm);
  std::unordered_map<svmp::gid_t, Vec3d> unique;
  unique.reserve(all_perim.size());
  for (const auto& [gid, p] : all_perim) {
    if (unique.find(gid) == unique.end()) {
      unique.emplace(gid, p);
    }
  }

  out.perimeter_unit_dirs.clear();
  out.perimeter_r2.clear();
  out.perimeter_unit_dirs.reserve(unique.size());
  out.perimeter_r2.reserve(unique.size());

  for (const auto& [_, p] : unique) {
    const Vec3d v = p - out.center;
    const double r2 = norm2(v);
    if (!(r2 > 0.0)) {
      continue;
    }
    out.perimeter_r2.push_back(r2);
    out.perimeter_unit_dirs.push_back((1.0 / std::sqrt(r2)) * v);
  }

  if (out.perimeter_unit_dirs.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Parabolic inflow profile for boundary marker " + std::to_string(boundary_marker) +
        " requires a non-empty perimeter. Ensure this face shares an edge with another boundary face "
        "(e.g., a wall surface), or set <Profile>Flat</Profile>.");
  }

  return out;
}

double parabolic_weight(const ParabolicProfileData& data, const Vec3d& x)
{
  const Vec3d r = x - data.center;
  const double r2 = norm2(r);
  if (!(r2 > 0.0)) {
    return 1.0;
  }

  double best = -std::numeric_limits<double>::infinity();
  std::size_t best_i = 0;
  for (std::size_t i = 0; i < data.perimeter_unit_dirs.size(); ++i) {
    const double d = dot(r, data.perimeter_unit_dirs[i]);
    if (d > best) {
      best = d;
      best_i = i;
    }
  }

  const double R2 = data.perimeter_r2[best_i];
  if (!(R2 > 0.0)) {
    return 0.0;
  }

  const double w = 1.0 - (r2 / R2);
  return (w > 0.0) ? w : 0.0;
}

double integrate_parabolic_weight_over_marker(const svmp::MeshBase& mesh,
                                              int boundary_marker,
                                              const ParabolicProfileData& data,
                                              MarkerCommunicator comm)
{
  auto tri_area = [](const Vec3d& a, const Vec3d& b, const Vec3d& c) {
    const Vec3d ab = b - a;
    const Vec3d ac = c - a;
    return 0.5 * norm(cross(ab, ac));
  };

  auto tri_integral = [&](const Vec3d& p0, const Vec3d& p1, const Vec3d& p2) {
    const double a = tri_area(p0, p1, p2);
    if (!(a > 0.0)) {
      return 0.0;
    }
    // Degree-2 symmetric rule (3 points), weights sum to 1.
    const Vec3d x1 = (1.0 / 6.0) * p0 + (1.0 / 6.0) * p1 + (2.0 / 3.0) * p2;
    const Vec3d x2 = (1.0 / 6.0) * p0 + (2.0 / 3.0) * p1 + (1.0 / 6.0) * p2;
    const Vec3d x3 = (2.0 / 3.0) * p0 + (1.0 / 6.0) * p1 + (1.0 / 6.0) * p2;

    const double f = (parabolic_weight(data, x1) + parabolic_weight(data, x2) + parabolic_weight(data, x3)) / 3.0;
    return a * f;
  };

  auto face_integral = [&](const std::vector<svmp::index_t>& verts) {
    if (verts.size() < 3u) {
      return 0.0;
    }

    // Treat any face as a polygon and fan-triangulate it.
    // This is robust to meshes that store boundary faces as CellFamily::Polygon.
    std::vector<Vec3d> pts;
    pts.reserve(verts.size());
    for (const auto v : verts) {
      if (v == svmp::INVALID_INDEX) {
        continue;
      }
      pts.push_back(to_vec3(mesh.get_vertex_coords(v)));
    }
    if (pts.size() < 3u) {
      return 0.0;
    }

    double val = 0.0;
    const Vec3d p0 = pts[0];
    for (std::size_t i = 1; i + 1 < pts.size(); ++i) {
      val += tri_integral(p0, pts[i], pts[i + 1]);
    }
    return val;
  };

  double sum = 0.0;
  const auto payloads = gather_unique_marker_face_payloads<1>(
      mesh, boundary_marker, comm,
      [&](svmp::index_t f, std::array<double, 1>& values) {
        values[0] = face_integral(mesh.face_vertices(f));
      });
  for (const auto& payload : payloads) {
    sum += payload.values[0];
  }
  return sum;
}

} // namespace svmp::Physics::options
