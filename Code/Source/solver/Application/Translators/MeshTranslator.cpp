#include "Application/Translators/MeshTranslator.h"

#include "Application/Core/OopMpiLog.h"
#include "Parameters.h"

#include "Mesh/Mesh.h"
#include "Mesh/Labels/MeshLabels.h"
#include "Mesh/Search/CoordinateKey.h"
#include "Mesh/Topology/CellTopology.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

std::string lower_copy(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

struct FaceVertexKey {
  std::array<svmp::index_t, 8> inline_ids{};
  std::vector<svmp::index_t> overflow_ids;
  std::uint8_t size{0};

  bool operator==(const FaceVertexKey& other) const
  {
    if (size != other.size) {
      return false;
    }
    if (size <= inline_ids.size()) {
      for (std::uint8_t i = 0; i < size; ++i) {
        if (inline_ids[static_cast<std::size_t>(i)] != other.inline_ids[static_cast<std::size_t>(i)]) {
          return false;
        }
      }
      return true;
    }
    return overflow_ids == other.overflow_ids;
  }
};

struct FaceVertexKeyHash {
  std::size_t operator()(const FaceVertexKey& key) const noexcept
  {
    std::size_t h = static_cast<std::size_t>(key.size) + 0x9e3779b97f4a7c15ULL;
    const auto mix = [&h](svmp::index_t id) {
      auto v = static_cast<std::size_t>(id);
      h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6u) + (h >> 2u);
    };
    if (key.size <= key.inline_ids.size()) {
      for (std::uint8_t i = 0; i < key.size; ++i) {
        mix(key.inline_ids[static_cast<std::size_t>(i)]);
      }
    } else {
      for (auto id : key.overflow_ids) {
        mix(id);
      }
    }
    return h;
  }
};

FaceVertexKey make_face_vertex_key(const std::vector<svmp::index_t>& ids)
{
  FaceVertexKey key;
  key.size = static_cast<std::uint8_t>(std::min<std::size_t>(ids.size(), 255u));
  if (ids.size() <= key.inline_ids.size()) {
    for (std::size_t i = 0; i < ids.size(); ++i) {
      key.inline_ids[i] = ids[i];
    }
  } else {
    key.overflow_ids = ids;
  }
  return key;
}

bool has_nonlocal_vertex_gids(const svmp::MeshBase& mesh)
{
  const auto& gids = mesh.vertex_gids();
  if (gids.size() != mesh.n_vertices()) {
    return false;
  }
  for (std::size_t i = 0; i < gids.size(); ++i) {
    if (gids[i] != static_cast<svmp::gid_t>(i)) {
      return true;
    }
  }
  return false;
}

svmp::index_t match_face_vertex_to_volume(
    const svmp::Mesh& volume_mesh,
    const svmp::search::VertexCoordinateLocator& coordinate_locator,
    const svmp::MeshBase& face_mesh,
    svmp::index_t face_vertex,
    bool use_face_gids)
{
  const auto coordinate_match = coordinate_locator.find(face_mesh.X_ref(), face_mesh.dim(), face_vertex);
  if (coordinate_match != svmp::INVALID_INDEX) {
    return coordinate_match;
  }

  if (use_face_gids) {
    const auto& gids = face_mesh.vertex_gids();
    const auto i = static_cast<std::size_t>(face_vertex);
    if (face_vertex >= 0 && i < gids.size() && gids[i] != svmp::INVALID_GID) {
      const auto local = volume_mesh.base().global_to_local_vertex(gids[i]);
      if (local != svmp::INVALID_INDEX) {
        return local;
      }
    }
  }

  return svmp::INVALID_INDEX;
}

std::vector<svmp::index_t> cell_corner_vertices(const svmp::MeshBase& mesh, svmp::index_t c)
{
  auto [ptr, count] = mesh.cell_corner_vertices_span(c);
  return std::vector<svmp::index_t>(ptr, ptr + count);
}

std::vector<svmp::index_t> face_corner_vertices(const svmp::MeshBase& mesh, svmp::index_t f)
{
  auto [ptr, count] = mesh.face_vertices_span(f);
  std::size_t n_corners = count;
  const auto& shapes = mesh.face_shapes();
  if (static_cast<std::size_t>(f) < shapes.size() && shapes[static_cast<std::size_t>(f)].num_corners > 0) {
    n_corners = std::min(count, static_cast<std::size_t>(shapes[static_cast<std::size_t>(f)].num_corners));
  }
  return std::vector<svmp::index_t>(ptr, ptr + n_corners);
}

svmp::CellShape boundary_shape_for_corners(int dim, std::size_t n_corners, int order)
{
  svmp::CellShape shape{};
  if (dim == 2) {
    shape.family = svmp::CellFamily::Line;
    shape.num_corners = 2;
  } else if (n_corners == 3u) {
    shape.family = svmp::CellFamily::Triangle;
    shape.num_corners = 3;
  } else if (n_corners == 4u) {
    shape.family = svmp::CellFamily::Quad;
    shape.num_corners = 4;
  } else {
    shape.family = svmp::CellFamily::Polygon;
    shape.num_corners = static_cast<int>(n_corners);
  }
  shape.order = std::max(1, order);
  shape.is_mixed_order = false;
  return shape;
}

struct VertexCellAdjacency {
  std::vector<svmp::offset_t> offsets;
  std::vector<svmp::index_t> cells;
};

VertexCellAdjacency build_owned_vertex_cell_adjacency(const svmp::Mesh& mesh)
{
  VertexCellAdjacency adj;
  adj.offsets.assign(mesh.n_vertices() + 1u, 0);

  for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(mesh.n_cells()); ++c) {
    if (!mesh.is_owned_cell(c)) {
      continue;
    }
    auto [verts, nverts] = mesh.base().cell_vertices_span(c);
    for (std::size_t i = 0; i < nverts; ++i) {
      const auto v = verts[i];
      if (v >= 0 && static_cast<std::size_t>(v) < mesh.n_vertices()) {
        ++adj.offsets[static_cast<std::size_t>(v) + 1u];
      }
    }
  }

  for (std::size_t i = 0; i + 1u < adj.offsets.size(); ++i) {
    adj.offsets[i + 1u] += adj.offsets[i];
  }

  adj.cells.assign(static_cast<std::size_t>(adj.offsets.back()), svmp::INVALID_INDEX);
  auto cursor = adj.offsets;
  for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(mesh.n_cells()); ++c) {
    if (!mesh.is_owned_cell(c)) {
      continue;
    }
    auto [verts, nverts] = mesh.base().cell_vertices_span(c);
    for (std::size_t i = 0; i < nverts; ++i) {
      const auto v = verts[i];
      if (v < 0 || static_cast<std::size_t>(v) >= mesh.n_vertices()) {
        continue;
      }
      const auto pos = cursor[static_cast<std::size_t>(v)]++;
      adj.cells[static_cast<std::size_t>(pos)] = c;
    }
  }

  return adj;
}

struct MatchedBoundaryFace {
  svmp::index_t cell{svmp::INVALID_INDEX};
  std::vector<svmp::index_t> oriented_vertices;
  svmp::CellShape shape{};
};

std::optional<MatchedBoundaryFace> match_owned_cell_boundary_face(
    const svmp::Mesh& mesh,
    const std::vector<svmp::index_t>& sorted_face_vertices,
    const VertexCellAdjacency& vertex_cells)
{
  if (sorted_face_vertices.empty() || vertex_cells.offsets.empty()) {
    return std::nullopt;
  }

  svmp::index_t seed_vertex = svmp::INVALID_INDEX;
  std::size_t best_degree = std::numeric_limits<std::size_t>::max();
  for (const auto v : sorted_face_vertices) {
    if (v < 0 || static_cast<std::size_t>(v) + 1u >= vertex_cells.offsets.size()) {
      return std::nullopt;
    }
    const auto begin = static_cast<std::size_t>(vertex_cells.offsets[static_cast<std::size_t>(v)]);
    const auto end = static_cast<std::size_t>(vertex_cells.offsets[static_cast<std::size_t>(v) + 1u]);
    const auto degree = end - begin;
    if (degree < best_degree) {
      best_degree = degree;
      seed_vertex = v;
    }
  }
  if (seed_vertex == svmp::INVALID_INDEX || best_degree == 0u) {
    return std::nullopt;
  }

  const auto seed_begin =
      static_cast<std::size_t>(vertex_cells.offsets[static_cast<std::size_t>(seed_vertex)]);
  const auto seed_end =
      static_cast<std::size_t>(vertex_cells.offsets[static_cast<std::size_t>(seed_vertex) + 1u]);

  for (std::size_t pos = seed_begin; pos < seed_end; ++pos) {
    const auto cell = vertex_cells.cells[pos];
    if (cell == svmp::INVALID_INDEX || !mesh.is_owned_cell(cell)) {
      continue;
    }

    auto [cell_vertices, n_cell_vertices] = mesh.base().cell_vertices_span(cell);
    const auto& cshape = mesh.base().cell_shape(cell);
    const auto face_view = svmp::CellTopology::get_oriented_boundary_faces_view(cshape.family);
    if (!face_view.indices || !face_view.offsets || face_view.face_count <= 0) {
      continue;
    }

    for (int lf = 0; lf < face_view.face_count; ++lf) {
      const int begin = face_view.offsets[lf];
      const int end = face_view.offsets[lf + 1];
      if (end < begin || static_cast<std::size_t>(end - begin) != sorted_face_vertices.size()) {
        continue;
      }

      std::vector<svmp::index_t> candidate;
      candidate.reserve(sorted_face_vertices.size());
      bool valid = true;
      for (int j = begin; j < end; ++j) {
        const auto local = face_view.indices[j];
        if (local < 0 || static_cast<std::size_t>(local) >= n_cell_vertices) {
          valid = false;
          break;
        }
        candidate.push_back(cell_vertices[local]);
      }
      if (!valid) {
        continue;
      }

      auto sorted_candidate = candidate;
      std::sort(sorted_candidate.begin(), sorted_candidate.end());
      if (sorted_candidate != sorted_face_vertices) {
        continue;
      }

      std::vector<svmp::index_t> oriented_face_vertices;
      try {
        oriented_face_vertices = mesh.base().cell_face_geometry_dofs(cell, lf);
      } catch (const std::exception&) {
        oriented_face_vertices.clear();
      }
      if (oriented_face_vertices.empty()) {
        oriented_face_vertices = candidate;
      }

      MatchedBoundaryFace match;
      match.cell = cell;
      match.oriented_vertices = std::move(oriented_face_vertices);
      match.shape = boundary_shape_for_corners(mesh.dim(),
                                               sorted_face_vertices.size(),
                                               mesh.base().geometry_order(cell));
      return match;
    }
  }

  return std::nullopt;
}

} // namespace

namespace application {
namespace translators {

std::shared_ptr<svmp::Mesh> MeshTranslator::loadMesh(const MeshParameters& params)
{
  const std::string file_path = params.mesh_file_path.value();
  if (file_path.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] <Mesh_file_path> is required inside <Add_mesh name=\"" +
        params.name.value() + "\">.");
  }

  if (application::core::oopTraceEnabled()) {
    application::core::oopCout() << "[svMultiPhysics::Application] MeshTranslator: loading mesh file_path='" << file_path
                                 << "'" << std::endl;
  }

  svmp::MeshIOOptions io_opts{};
  io_opts.path = file_path;
  io_opts.format = detectFormat(file_path);
  if (io_opts.format.empty()) {
    throw std::runtime_error("[svMultiPhysics::Application] Unsupported mesh file extension: '" +
                             file_path + "'.");
  }
  io_opts.kv["codim1_topology"] = "none";
  io_opts.kv["edge_topology"] = "false";

  application::core::oopCout()
      << "[svMultiPhysics::Application] MeshTranslator: initial mesh storage codim1=none edge=false"
      << " (FE setup will materialize planned topology)." << std::endl;

  if (application::core::oopTraceEnabled()) {
    application::core::oopCout() << "[svMultiPhysics::Application] MeshTranslator: detected format='" << io_opts.format
                                 << "'" << std::endl;
  }

  auto mesh = std::make_shared<svmp::Mesh>(svmp::Mesh::load_parallel(io_opts, svmp::MeshComm::world()));
  if (application::core::oopTraceEnabled()) {
    application::core::oopCout() << "[svMultiPhysics::Application] MeshTranslator: mesh loaded dim=" << mesh->dim()
                                 << " vertices=" << mesh->n_vertices() << " cells=" << mesh->n_cells()
                                 << " faces=" << mesh->n_faces() << std::endl;
  }

  applyFaceLabels(*mesh, params.face_parameters);
  applyDomainLabels(*mesh, params);

  return mesh;
}

std::string MeshTranslator::detectFormat(const std::string& file_path)
{
  const auto ext = lower_copy(std::filesystem::path(file_path).extension().string());
  if (ext == ".vtu") return "vtu";
  if (ext == ".vtk") return "vtk";
  if (ext == ".vtp") return "vtp";
  if (ext == ".pvtu") return "pvtu";
  if (ext == ".pvtp") return "pvtp";
  if (ext == ".msh") return "msh";
  if (ext == ".mesh") return "mesh";
  return "";
}

void MeshTranslator::applyFaceLabels(svmp::Mesh& mesh,
                                     const std::vector<FaceParameters*>& face_params)
{
  if (face_params.empty()) {
    return;
  }

  application::core::oopCout() << "[svMultiPhysics::Application] MeshTranslator: applying face labels; count="
                               << static_cast<int>(face_params.size()) << std::endl;

  if (mesh.dim() <= 0) {
    throw std::runtime_error("[svMultiPhysics::Application] Loaded mesh has invalid dimension.");
  }

  const svmp::search::VertexCoordinateLocator coordinate_locator(mesh.base());
  const bool use_volume_gids = has_nonlocal_vertex_gids(mesh.base());

  const bool rebuild_boundary_faces_from_add_face_files =
      (mesh.n_faces() == 0u) || (mesh.world_size() > 1);

  if (rebuild_boundary_faces_from_add_face_files) {
    if (mesh.n_faces() != 0u && application::core::oopTraceEnabled()) {
      application::core::oopCout()
          << "[svMultiPhysics::Application] MeshTranslator: replacing pre-existing local face topology"
          << " with boundary-only topology from <Add_face> files for MPI-consistent boundary labels."
          << std::endl;
    }

    const auto vertex_cells = build_owned_vertex_cell_adjacency(mesh);

    std::vector<svmp::CellShape> boundary_shapes;
    std::vector<svmp::offset_t> boundary_offsets;
    std::vector<svmp::index_t> boundary_connectivity;
    std::vector<std::array<svmp::index_t, 2>> boundary_face2cell;
    std::vector<svmp::label_t> boundary_labels;
    std::vector<std::vector<std::string>> boundary_sets;
    std::unordered_map<FaceVertexKey, svmp::index_t, FaceVertexKeyHash> face_by_vertices;

    boundary_offsets.push_back(0);

    svmp::label_t next_label = 1;
    for (const auto* face : face_params) {
      if (!face) {
        continue;
      }

      const auto face_name = face->name.value();
      const auto face_path = face->face_file_path.value();
      if (face_name.empty()) {
        throw std::runtime_error("[svMultiPhysics::Application] <Add_face> is missing required name attribute.");
      }
      if (face_path.empty()) {
        throw std::runtime_error("[svMultiPhysics::Application] <Add_face name=\"" + face_name +
                                 "\"> is missing <Face_file_path>.");
      }

      application::core::oopCout() << "[svMultiPhysics::Application]   Face '" << face_name << "': file_path='"
                                   << face_path << "'" << std::endl;

      svmp::MeshIOOptions face_opts{};
      face_opts.path = face_path;
      face_opts.format = detectFormat(face_path);
      if (face_opts.format.empty()) {
        throw std::runtime_error("[svMultiPhysics::Application] Unsupported face mesh extension: '" +
                                 face_path + "'.");
      }
      face_opts.kv["force_min_dim"] = std::to_string(mesh.dim());
      face_opts.kv["codim1_topology"] = "none";
      face_opts.kv["edge_topology"] = "false";

      svmp::MeshBase face_mesh = svmp::MeshBase::load(face_opts);
      const bool use_face_gids = use_volume_gids && has_nonlocal_vertex_gids(face_mesh);

      const auto label = next_label++;
      application::core::oopCout() << "[svMultiPhysics::Application]   Face '" << face_name << "': format='"
                                   << face_opts.format << "' cells=" << face_mesh.n_cells() << " -> label=" << label
                                   << std::endl;
      mesh.register_label(face_name, label);

      svmp::index_t local_matched = 0;
      svmp::index_t local_skipped = 0;

      for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(face_mesh.n_cells()); ++c) {
        auto face_cell_vertices = cell_corner_vertices(face_mesh, c);

        std::vector<svmp::index_t> vol_vertex_ids;
        vol_vertex_ids.reserve(face_cell_vertices.size());

        bool have_all_vertices = true;
        for (auto fv : face_cell_vertices) {
          const auto volume_vertex =
              match_face_vertex_to_volume(mesh, coordinate_locator, face_mesh, fv, use_face_gids);
          if (volume_vertex == svmp::INVALID_INDEX) {
            have_all_vertices = false;
            break;
          }
          vol_vertex_ids.push_back(volume_vertex);
        }
        if (!have_all_vertices) {
          ++local_skipped;
          continue;
        }

        std::sort(vol_vertex_ids.begin(), vol_vertex_ids.end());
        const auto unique_end = std::unique(vol_vertex_ids.begin(), vol_vertex_ids.end());
        if (unique_end != vol_vertex_ids.end()) {
          ++local_skipped;
          continue;
        }

        const auto key = make_face_vertex_key(vol_vertex_ids);
        const auto existing = face_by_vertices.find(key);
        if (existing != face_by_vertices.end()) {
          const auto f = existing->second;
          boundary_labels[static_cast<std::size_t>(f)] = label;
          boundary_sets[static_cast<std::size_t>(f)].push_back(face_name);
          ++local_matched;
          continue;
        }

        auto match = match_owned_cell_boundary_face(mesh, vol_vertex_ids, vertex_cells);
        if (!match.has_value()) {
          ++local_skipped;
          continue;
        }

        const auto new_face = static_cast<svmp::index_t>(boundary_shapes.size());
        boundary_shapes.push_back(match->shape);
        boundary_connectivity.insert(boundary_connectivity.end(),
                                     match->oriented_vertices.begin(),
                                     match->oriented_vertices.end());
        boundary_offsets.push_back(static_cast<svmp::offset_t>(boundary_connectivity.size()));
        boundary_face2cell.push_back(
            std::array<svmp::index_t, 2>{{match->cell, svmp::INVALID_INDEX}});
        boundary_labels.push_back(label);
        boundary_sets.push_back(std::vector<std::string>{face_name});
        face_by_vertices.emplace(key, new_face);
        ++local_matched;
      }

      if (application::core::oopTraceEnabled() && local_matched == 0) {
        application::core::oopCout()
            << "[svMultiPhysics::Application]   Face '" << face_name
            << "': no local matches (this is expected on non-owning MPI ranks)." << std::endl;
      }
      if (application::core::oopTraceEnabled() && local_skipped > 0) {
        application::core::oopCout()
            << "[svMultiPhysics::Application]   Face '" << face_name
            << "': skipped local surface cells=" << local_skipped << std::endl;
      }
    }

    mesh.base().set_faces_from_arrays(std::move(boundary_shapes),
                                      std::move(boundary_offsets),
                                      std::move(boundary_connectivity),
                                      std::move(boundary_face2cell),
                                      svmp::MeshCodim1StorageMode::BoundaryOnly);
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mesh.n_faces()); ++f) {
      svmp::MeshLabels::set_boundary_label(mesh.base(), f, boundary_labels[static_cast<std::size_t>(f)]);
      for (const auto& set_name : boundary_sets[static_cast<std::size_t>(f)]) {
        mesh.add_to_set(svmp::EntityKind::Face, set_name, f);
      }
    }

    application::core::oopCout()
        << "[svMultiPhysics::Application] MeshTranslator: built boundary-only face topology faces="
        << mesh.n_faces() << std::endl;
    return;
  }

  std::unordered_map<FaceVertexKey, svmp::index_t, FaceVertexKeyHash> boundary_face_by_vertices;
  const auto boundary_faces = mesh.boundary_faces();
  boundary_face_by_vertices.reserve(boundary_faces.size());
  const auto& face2cell = mesh.face2cell();
  for (auto f : boundary_faces) {
    const auto fc = face2cell.at(static_cast<std::size_t>(f));
    const auto adj_cell =
        (fc[0] != svmp::INVALID_INDEX) ? fc[0] : (fc[1] != svmp::INVALID_INDEX) ? fc[1] : svmp::INVALID_INDEX;
    if (adj_cell == svmp::INVALID_INDEX || !mesh.is_owned_cell(adj_cell)) {
      continue;
    }
    auto verts = face_corner_vertices(mesh.base(), f);
    std::sort(verts.begin(), verts.end());
    boundary_face_by_vertices.emplace(make_face_vertex_key(verts), f);
  }

  svmp::label_t next_label = 1;
  for (const auto* face : face_params) {
    if (!face) {
      continue;
    }

    const auto face_name = face->name.value();
    const auto face_path = face->face_file_path.value();
    if (face_name.empty()) {
      throw std::runtime_error("[svMultiPhysics::Application] <Add_face> is missing required name attribute.");
    }
    if (face_path.empty()) {
      throw std::runtime_error("[svMultiPhysics::Application] <Add_face name=\"" + face_name +
                               "\"> is missing <Face_file_path>.");
    }

    application::core::oopCout() << "[svMultiPhysics::Application]   Face '" << face_name << "': file_path='"
                                 << face_path << "'" << std::endl;

    // Load face surface mesh (typically .vtp).
    // Force the face mesh to use at least the same spatial dimension as the volume mesh
    // so that coordinate keys match during vertex-based face matching.  Without this,
    // VTP surfaces that lie on a constant-Z plane would be loaded as dim=2 (dropping Z)
    // and all vertex lookups against the 3D volume mesh would fail.
    svmp::MeshIOOptions face_opts{};
    face_opts.path = face_path;
    face_opts.format = detectFormat(face_path);
    if (face_opts.format.empty()) {
      throw std::runtime_error("[svMultiPhysics::Application] Unsupported face mesh extension: '" +
                               face_path + "'.");
    }
    face_opts.kv["force_min_dim"] = std::to_string(mesh.dim());
    face_opts.kv["codim1_topology"] = "none";
    face_opts.kv["edge_topology"] = "false";

    svmp::MeshBase face_mesh = svmp::MeshBase::load(face_opts);
    const bool use_face_gids = has_nonlocal_vertex_gids(face_mesh);

    const auto label = next_label++;
    application::core::oopCout() << "[svMultiPhysics::Application]   Face '" << face_name << "': format='"
                                 << face_opts.format << "' cells=" << face_mesh.n_cells() << " -> label=" << label
                                 << std::endl;
    mesh.register_label(face_name, label);

    svmp::index_t local_matched = 0;
    svmp::index_t local_skipped = 0;

    for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(face_mesh.n_cells()); ++c) {
      auto face_cell_vertices = cell_corner_vertices(face_mesh, c);

      std::vector<svmp::index_t> vol_vertex_ids;
      vol_vertex_ids.reserve(face_cell_vertices.size());

      bool have_all_vertices = true;
      for (auto fv : face_cell_vertices) {
        const auto volume_vertex =
            match_face_vertex_to_volume(mesh, coordinate_locator, face_mesh, fv, use_face_gids);
        if (volume_vertex == svmp::INVALID_INDEX) {
          have_all_vertices = false;
          break;
        }
        vol_vertex_ids.push_back(volume_vertex);
      }
      if (!have_all_vertices) {
        ++local_skipped;
        continue;
      }

      std::sort(vol_vertex_ids.begin(), vol_vertex_ids.end());
      const auto it_face = boundary_face_by_vertices.find(make_face_vertex_key(vol_vertex_ids));
      if (it_face == boundary_face_by_vertices.end()) {
        ++local_skipped;
        continue;
      }

      svmp::MeshLabels::set_boundary_label(mesh.base(), it_face->second, label);
      mesh.add_to_set(svmp::EntityKind::Face, face_name, it_face->second);
      ++local_matched;
    }

    if (application::core::oopTraceEnabled() && local_matched == 0) {
      application::core::oopCout()
          << "[svMultiPhysics::Application]   Face '" << face_name
          << "': no local matches (this is expected on non-owning MPI ranks)." << std::endl;
    }
  }
}

void MeshTranslator::applyDomainLabels(svmp::Mesh& mesh, const MeshParameters& params)
{
  if (params.domain_file_path.defined() && !params.domain_file_path.value().empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] <Domain_file_path> is not supported in the new solver yet. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  if (!params.domain_id.defined()) {
    return;
  }

  const auto label = static_cast<svmp::label_t>(params.domain_id.value());
  application::core::oopCout() << "[svMultiPhysics::Application] MeshTranslator: setting domain_id=" << label
                               << " for " << mesh.n_cells() << " cells." << std::endl;
  for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(mesh.n_cells()); ++c) {
    svmp::MeshLabels::set_region_label(mesh.base(), c, label);
  }
}

} // namespace translators
} // namespace application
