#include "Application/Translators/MeshTranslator.h"

#include "Application/Core/OopMpiLog.h"
#include "Parameters.h"

#include "Mesh/Mesh.h"
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

  std::unordered_map<std::string, svmp::index_t> coord_to_vertex;
  coord_to_vertex.reserve(mesh.n_vertices());
  for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh.n_vertices()); ++v) {
    coord_to_vertex.emplace(make_coord_key(mesh.X_ref(), mesh.dim(), v), v);
  }

  std::unordered_map<std::string, svmp::index_t> boundary_face_by_vertices;
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

    application::core::oopCout() << "[svMultiPhysics::Application]   Face '" << face_name << "': file_path='"
                                 << face_path << "'" << std::endl;

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
    application::core::oopCout() << "[svMultiPhysics::Application]   Face '" << face_name << "': format='"
                                 << face_opts.format << "' cells=" << face_mesh.n_cells() << " -> label=" << label
                                 << std::endl;
    mesh.register_label(face_name, label);

    svmp::index_t local_matched = 0;
    svmp::index_t local_skipped = 0;

    for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(face_mesh.n_cells()); ++c) {
      auto face_cell_vertices = face_mesh.cell_vertices(c);

      std::vector<svmp::index_t> vol_vertex_ids;
      vol_vertex_ids.reserve(face_cell_vertices.size());

      bool have_all_vertices = true;
      for (auto fv : face_cell_vertices) {
        const auto key = make_coord_key(face_mesh.X_ref(), face_mesh.dim(), fv);
        const auto it = coord_to_vertex.find(key);
        if (it == coord_to_vertex.end()) {
          have_all_vertices = false;
          break;
        }
        vol_vertex_ids.push_back(it->second);
      }
      if (!have_all_vertices) {
        ++local_skipped;
        continue;
      }

      std::sort(vol_vertex_ids.begin(), vol_vertex_ids.end());
      const auto it_face = boundary_face_by_vertices.find(make_id_key(vol_vertex_ids));
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
