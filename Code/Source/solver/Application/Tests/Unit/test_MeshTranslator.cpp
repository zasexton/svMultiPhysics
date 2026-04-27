#include <gtest/gtest.h>

#include "Application/Translators/MeshTranslator.h"
#include "Mesh/Mesh.h"
#include "Parameters.h"

#ifdef MESH_HAS_VTK
#include "Mesh/IO/VTKWriter.h"
#endif

#ifdef MESH_HAS_MPI
#include <mpi.h>
#endif

#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

#ifdef MESH_HAS_MPI
void finalize_mpi_if_needed()
{
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized) {
    MPI_Finalize();
  }
}
#endif

void ensure_mpi_initialized_for_mesh_translator()
{
#ifdef MESH_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    int argc = 0;
    char** argv = nullptr;
    MPI_Init(&argc, &argv);
    std::atexit(finalize_mpi_if_needed);
  }
#endif
}

svmp::CellShape make_shape(svmp::CellFamily family, int corners)
{
  svmp::CellShape shape{};
  shape.family = family;
  shape.num_corners = corners;
  shape.order = 1;
  return shape;
}

svmp::MeshBase make_volume_tetra_base()
{
  svmp::MeshBase base;
  std::vector<svmp::real_t> x = {
      0.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0,
  };
  std::vector<svmp::offset_t> offsets = {0, 4};
  std::vector<svmp::index_t> conn = {0, 1, 2, 3};
  base.build_from_arrays(3, x, offsets, conn, {make_shape(svmp::CellFamily::Tetra, 4)});
  base.set_vertex_gids({100, 101, 102, 103});
  return base;
}

#ifdef MESH_HAS_VTK
std::filesystem::path unique_temp_path(const std::string& stem)
{
  const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         (stem + "_" + std::to_string(ticks) + ".vtp");
}

void write_volume_mesh(const std::filesystem::path& path)
{
  auto volume = make_volume_tetra_base();

  svmp::MeshIOOptions opts{};
  opts.format = "vtu";
  opts.path = path.string();
  opts.kv["codim1_topology"] = "none";
  opts.kv["edge_topology"] = "false";
  svmp::VTKWriter::write(volume, opts);
}

void write_face_mesh(const std::filesystem::path& path,
                     std::vector<svmp::real_t> coords,
                     std::vector<svmp::gid_t> gids)
{
  svmp::MeshBase face;
  std::vector<svmp::offset_t> offsets = {0, 3};
  std::vector<svmp::index_t> conn = {0, 1, 2};
  face.build_from_arrays(3, std::move(coords), std::move(offsets), std::move(conn),
                         {make_shape(svmp::CellFamily::Triangle, 3)});
  face.set_vertex_gids(std::move(gids));

  svmp::MeshIOOptions opts{};
  opts.format = "vtp";
  opts.path = path.string();
  opts.kv["codim1_topology"] = "none";
  opts.kv["edge_topology"] = "false";
  svmp::VTKWriter::write(face, opts);
}

std::shared_ptr<svmp::Mesh> load_volume_with_face(const std::filesystem::path& volume_path,
                                                  const std::string& face_name,
                                                  const std::filesystem::path& face_path)
{
  MeshParameters mesh_params;
  mesh_params.name.set("mesh");
  mesh_params.mesh_file_path.set(volume_path.string());

  auto* face = new FaceParameters();
  face->name.set(face_name);
  face->face_file_path.set(face_path.string());
  mesh_params.face_parameters.push_back(face);

  return application::translators::MeshTranslator::loadMesh(mesh_params);
}
#endif

} // namespace

TEST(MeshTranslatorFaceLabels, MatchesFaceByGlobalVertexIdsBeforeCoordinates)
{
#ifndef MESH_HAS_VTK
  GTEST_SKIP() << "VTK support is required for face-file translator coverage.";
#else
  ensure_mpi_initialized_for_mesh_translator();
  auto volume_path = unique_temp_path("svmp_meshtranslator_gid_volume");
  const auto face_path = unique_temp_path("svmp_meshtranslator_gid_face");
  volume_path.replace_extension(".vtu");
  write_volume_mesh(volume_path);

  // Coordinates are deliberately offset. Matching should still succeed because
  // the face file carries nonlocal GlobalVertexID values from the volume mesh.
  write_face_mesh(face_path,
                  {
                      20.0, 20.0, 20.0,
                      21.0, 20.0, 20.0,
                      20.0, 21.0, 20.0,
                  },
                  {100, 101, 102});

  auto mesh = load_volume_with_face(volume_path, "marked", face_path);
  std::filesystem::remove(volume_path);
  std::filesystem::remove(face_path);

  const auto label = mesh->base().label_from_name("marked");
  ASSERT_NE(label, svmp::INVALID_LABEL);
  const auto marked_faces = mesh->base().faces_with_label(label);
  ASSERT_EQ(marked_faces.size(), 1u);
  EXPECT_TRUE(mesh->base().has_set(svmp::EntityKind::Face, "marked"));
  EXPECT_EQ(mesh->base().get_set(svmp::EntityKind::Face, "marked").size(), 1u);
#endif
}

TEST(MeshTranslatorFaceLabels, FallsBackToCompactCoordinateKeysForDefaultFaceGids)
{
#ifndef MESH_HAS_VTK
  GTEST_SKIP() << "VTK support is required for face-file translator coverage.";
#else
  ensure_mpi_initialized_for_mesh_translator();
  auto volume_path = unique_temp_path("svmp_meshtranslator_coordinate_volume");
  const auto face_path = unique_temp_path("svmp_meshtranslator_coordinate_face");
  volume_path.replace_extension(".vtu");
  write_volume_mesh(volume_path);

  // Default local-iota GIDs are intentionally not useful for matching this
  // surface to the volume mesh. The translator must use coordinate keys.
  write_face_mesh(face_path,
                  {
                      0.0, 1.0, 0.0,
                      1.0, 0.0, 0.0,
                      0.0, 0.0, 0.0,
                  },
                  {0, 1, 2});

  auto mesh = load_volume_with_face(volume_path, "coordinate_marked", face_path);
  std::filesystem::remove(volume_path);
  std::filesystem::remove(face_path);

  const auto label = mesh->base().label_from_name("coordinate_marked");
  ASSERT_NE(label, svmp::INVALID_LABEL);
  const auto marked_faces = mesh->base().faces_with_label(label);
  ASSERT_EQ(marked_faces.size(), 1u);
  EXPECT_EQ(mesh->base().get_set(svmp::EntityKind::Face, "coordinate_marked").size(), 1u);
#endif
}
