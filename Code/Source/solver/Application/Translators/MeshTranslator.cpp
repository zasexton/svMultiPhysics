#include "Application/Translators/MeshTranslator.h"

#include "Parameters.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Labels/MeshLabels.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace {

std::string lower_copy(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string make_coord_key(const std::vector<svmp::real_t>& X, int dim, svmp::index_t v)
{
  std::ostringstream key;
  key << std::scientific << std::setprecision(16);
  const auto base = static_cast<size_t>(v) * static_cast<size_t>(dim);
  for (int d = 0; d < dim; ++d) {
    key << X.at(base + static_cast<size_t>(d)) << ",";
  }
  return key.str();
}

std::string make_id_key(const std::vector<svmp::index_t>& ids)
{
  std::string key;
  key.reserve(ids.size() * 12);
  for (auto id : ids) {
    key += std::to_string(id);
    key.push_back(',');
  }
  return key;
}

} // namespace

namespace application {
namespace translators {

std::shared_ptr<svmp::MeshBase> MeshTranslator::loadMesh(const MeshParameters& params)
{
  const std::string file_path = params.mesh_file_path.value();
  if (file_path.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] <Mesh_file_path> is required inside <Add_mesh name=\"" +
        params.name.value() + "\">.");
  }

  svmp::MeshIOOptions io_opts{};
  io_opts.path = file_path;
  io_opts.format = detectFormat(file_path);
  if (io_opts.format.empty()) {
    throw std::runtime_error("[svMultiPhysics::Application] Unsupported mesh file extension: '" +
                             file_path + "'.");
  }

  auto loaded = svmp::MeshBase::load(io_opts);
  auto mesh = std::make_shared<svmp::MeshBase>(std::move(loaded));

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

void MeshTranslator::applyFaceLabels(svmp::MeshBase& mesh,
                                     const std::vector<FaceParameters*>& face_params)
{
  if (face_params.empty()) {
    return;
  }

  if (mesh.dim() <= 0) {
    throw std::runtime_error("[svMultiPhysics::Application] Loaded mesh has invalid dimension.");
  }

  std::unordered_map<std::string, svmp::index_t> coord_to_vertex;
  coord_to_vertex.reserve(mesh.n_vertices());
  for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh.n_vertices()); ++v) {
    coord_to_vertex.emplace(make_coord_key(mesh.X_ref(), mesh.dim(), v), v);
  }

  std::unordered_map<std::string, svmp::index_t> boundary_face_by_vertices;
  const auto boundary_faces = mesh.boundary_faces();
  boundary_face_by_vertices.reserve(boundary_faces.size());
  for (auto f : boundary_faces) {
    auto verts = mesh.face_vertices(f);
    std::sort(verts.begin(), verts.end());
    boundary_face_by_vertices.emplace(make_id_key(verts), f);
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

    // Load face surface mesh (typically .vtp).
    svmp::MeshIOOptions face_opts{};
    face_opts.path = face_path;
    face_opts.format = detectFormat(face_path);
    if (face_opts.format.empty()) {
      throw std::runtime_error("[svMultiPhysics::Application] Unsupported face mesh extension: '" +
                               face_path + "'.");
    }

    svmp::MeshBase face_mesh = svmp::MeshBase::load(face_opts);

    const auto label = next_label++;
    mesh.register_label(face_name, label);

    for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(face_mesh.n_cells()); ++c) {
      auto face_cell_vertices = face_mesh.cell_vertices(c);

      std::vector<svmp::index_t> vol_vertex_ids;
      vol_vertex_ids.reserve(face_cell_vertices.size());

      for (auto fv : face_cell_vertices) {
        const auto key = make_coord_key(face_mesh.X_ref(), face_mesh.dim(), fv);
        const auto it = coord_to_vertex.find(key);
        if (it == coord_to_vertex.end()) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Face mesh '" + face_name +
              "' contains a vertex that cannot be matched to the volume mesh by coordinate.");
        }
        vol_vertex_ids.push_back(it->second);
      }

      std::sort(vol_vertex_ids.begin(), vol_vertex_ids.end());
      const auto it_face = boundary_face_by_vertices.find(make_id_key(vol_vertex_ids));
      if (it_face == boundary_face_by_vertices.end()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Face mesh '" + face_name +
            "' contains a cell that cannot be matched to a boundary face in the volume mesh.");
      }

      svmp::MeshLabels::set_boundary_label(mesh, it_face->second, label);
      mesh.add_to_set(svmp::EntityKind::Face, face_name, it_face->second);
    }
  }
}

void MeshTranslator::applyDomainLabels(svmp::MeshBase& mesh, const MeshParameters& params)
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
  for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(mesh.n_cells()); ++c) {
    svmp::MeshLabels::set_region_label(mesh, c, label);
  }
}

} // namespace translators
} // namespace application

