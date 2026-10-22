#include <gtest/gtest.h>

#include "Application/Core/LevelSetCutConfiguration.h"
#include "Application/Translators/MeshTranslator.h"
#include "Mesh/Mesh.h"
#include "Mesh/Core/StartupPartitionWeights.h"
#include "Parameters.h"
#include "tinyxml2.h"

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
#include <stdexcept>
#include <utility>
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

svmp::CellShape make_shape(svmp::CellFamily family, int corners, int order = 1)
{
  svmp::CellShape shape{};
  shape.family = family;
  shape.num_corners = corners;
  shape.order = order;
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

void write_quadratic_quad_volume_mesh(const std::filesystem::path& path)
{
  svmp::MeshBase volume;
  volume.build_from_arrays(2,
                           {
                               0.0, 0.0,
                               1.0, 0.0,
                               1.0, 1.0,
                               0.0, 1.0,
                               0.5, 0.0,
                               1.0, 0.5,
                               0.5, 1.0,
                               0.0, 0.5,
                               0.5, 0.5,
                           },
                           {0, 9},
                           {0, 1, 2, 3, 4, 5, 6, 7, 8},
                           {make_shape(svmp::CellFamily::Quad, 4, 2)});
  volume.set_vertex_gids({0, 1, 2, 3, 4, 5, 6, 7, 8});

  svmp::MeshIOOptions opts{};
  opts.format = "vtu";
  opts.path = path.string();
  opts.kv["codim1_topology"] = "none";
  opts.kv["edge_topology"] = "false";
  svmp::VTKWriter::write(volume, opts);
}

void write_subdivided_quadratic_edge_face_mesh(const std::filesystem::path& path)
{
  svmp::MeshBase face;
  face.build_from_arrays(2,
                         {
                             0.0, 0.0,
                             0.5, 0.0,
                             1.0, 0.0,
                         },
                         {0, 2, 4},
                         {0, 1, 1, 2},
                         {make_shape(svmp::CellFamily::Line, 2),
                          make_shape(svmp::CellFamily::Line, 2)});
  face.set_vertex_gids({0, 4, 1});

  svmp::MeshIOOptions opts{};
  opts.format = "vtp";
  opts.path = path.string();
  opts.kv["codim1_topology"] = "none";
  opts.kv["edge_topology"] = "false";
  svmp::VTKWriter::write(face, opts);
}

void write_quadratic_tetra_volume_mesh(const std::filesystem::path& path)
{
  svmp::MeshBase volume;
  volume.build_from_arrays(3,
                           {
                               0.0, 0.0, 0.0,
                               1.0, 0.0, 0.0,
                               0.0, 1.0, 0.0,
                               0.0, 0.0, 1.0,
                               0.5, 0.0, 0.0,
                               0.5, 0.5, 0.0,
                               0.0, 0.5, 0.0,
                               0.0, 0.0, 0.5,
                               0.5, 0.0, 0.5,
                               0.0, 0.5, 0.5,
                           },
                           {0, 10},
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                           {make_shape(svmp::CellFamily::Tetra, 4, 2)});
  volume.set_vertex_gids({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

  svmp::MeshIOOptions opts{};
  opts.format = "vtu";
  opts.path = path.string();
  opts.kv["codim1_topology"] = "none";
  opts.kv["edge_topology"] = "false";
  svmp::VTKWriter::write(volume, opts);
}

void write_quadratic_triangle_face_mesh(const std::filesystem::path& path)
{
  svmp::MeshBase face;
  face.build_from_arrays(3,
                         {
                             0.0, 0.0, 0.0,
                             1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.5, 0.0, 0.0,
                             0.5, 0.5, 0.0,
                             0.0, 0.5, 0.0,
                         },
                         {0, 6},
                         {0, 1, 2, 3, 4, 5},
                         {make_shape(svmp::CellFamily::Triangle, 3, 2)});
  face.set_vertex_gids({0, 1, 2, 4, 5, 6});

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

TEST(MeshTranslatorGhostLayers, DefaultsToZeroAndParsesExplicitValue)
{
  MeshParameters defaults;
  EXPECT_EQ(defaults.ghost_layers.value(), 0);

  tinyxml2::XMLDocument document;
  ASSERT_EQ(document.Parse(R"xml(
<Add_mesh name="tank">
  <Mesh_file_path>mesh.vtu</Mesh_file_path>
  <Ghost_layers>2</Ghost_layers>
</Add_mesh>
)xml"), tinyxml2::XML_SUCCESS)
      << document.ErrorStr();
  auto* element = document.FirstChildElement("Add_mesh");
  ASSERT_NE(element, nullptr);

  MeshParameters parsed;
  ASSERT_NO_THROW(parsed.set_values(element));
  EXPECT_TRUE(parsed.ghost_layers.defined());
  EXPECT_EQ(parsed.ghost_layers.value(), 2);
}

TEST(MeshTranslatorGhostLayers, RejectsNegativeValueBeforeMeshIO)
{
  MeshParameters parameters;
  parameters.name.set("tank");
  parameters.mesh_file_path.set("unused.vtu");
  parameters.ghost_layers.set("-1");

  EXPECT_THROW(
      (void)application::translators::MeshTranslator::loadMesh(parameters),
      std::invalid_argument);
}

TEST(MeshTranslatorGhostLayers, PassesAndLogsConfiguredChoice)
{
#ifndef MESH_HAS_VTK
  GTEST_SKIP() << "VTK support is required for mesh translator I/O coverage.";
#else
  ensure_mpi_initialized_for_mesh_translator();
  auto volume_path = unique_temp_path("svmp_meshtranslator_ghost_layers");
  volume_path.replace_extension(".vtu");
  write_volume_mesh(volume_path);

  MeshParameters parameters;
  parameters.name.set("tank");
  parameters.mesh_file_path.set(volume_path.string());
  parameters.ghost_layers.set("1");

  testing::internal::CaptureStdout();
  auto mesh = application::translators::MeshTranslator::loadMesh(parameters);
  const auto output = testing::internal::GetCapturedStdout();
  std::filesystem::remove(volume_path);

  ASSERT_NE(mesh, nullptr);
  EXPECT_NE(output.find("ghost_layers=1"), std::string::npos);
#endif
}

TEST(MeshTranslatorFaceLabels, FallsBackToGlobalVertexIdsWhenCoordinatesDiffer)
{
#ifndef MESH_HAS_VTK
  GTEST_SKIP() << "VTK support is required for face-file translator coverage.";
#else
  ensure_mpi_initialized_for_mesh_translator();
  auto volume_path = unique_temp_path("svmp_meshtranslator_gid_volume");
  const auto face_path = unique_temp_path("svmp_meshtranslator_gid_face");
  volume_path.replace_extension(".vtu");
  write_volume_mesh(volume_path);

  // Coordinates are deliberately offset. Coordinate matching fails, but matching
  // should still succeed because the face file carries nonlocal GlobalVertexID
  // values from the volume mesh.
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
  EXPECT_TRUE(mesh->is_owned_face(marked_faces.front()));
  EXPECT_EQ(mesh->owner_rank_face(marked_faces.front()), mesh->rank());
  EXPECT_EQ(mesh->n_owned_faces(), mesh->n_faces());
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

TEST(MeshTranslatorFaceLabels, FallsBackToCoordinatesWhenOnlyFaceHasNonlocalGids)
{
#ifndef MESH_HAS_VTK
  GTEST_SKIP() << "VTK support is required for face-file translator coverage.";
#else
  ensure_mpi_initialized_for_mesh_translator();
  auto volume_path = unique_temp_path("svmp_meshtranslator_local_gid_volume");
  const auto face_path = unique_temp_path("svmp_meshtranslator_one_based_face");
  volume_path.replace_extension(".vtu");

  svmp::MeshBase volume;
  volume.build_from_arrays(3,
                           {
                               0.0, 0.0, 0.0,
                               1.0, 0.0, 0.0,
                               0.0, 1.0, 0.0,
                               0.0, 0.0, 1.0,
                           },
                           {0, 4},
                           {0, 1, 2, 3},
                           {make_shape(svmp::CellFamily::Tetra, 4)});
  volume.set_vertex_gids({0, 1, 2, 3});

  svmp::MeshIOOptions volume_opts{};
  volume_opts.format = "vtu";
  volume_opts.path = volume_path.string();
  volume_opts.kv["codim1_topology"] = "none";
  volume_opts.kv["edge_topology"] = "false";
  svmp::VTKWriter::write(volume, volume_opts);

  // These one-based face GIDs would map to the wrong volume vertices if the
  // translator trusted them while the volume only has synthetic local-iota GIDs.
  write_face_mesh(face_path,
                  {
                      0.0, 0.0, 0.0,
                      1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                  },
                  {1, 2, 3});

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

TEST(MeshTranslatorFaceLabels, LabelsParentHighOrderFaceFromSubdividedLineSurface)
{
#ifndef MESH_HAS_VTK
  GTEST_SKIP() << "VTK support is required for face-file translator coverage.";
#else
  ensure_mpi_initialized_for_mesh_translator();
  auto volume_path = unique_temp_path("svmp_meshtranslator_quad9_volume");
  const auto face_path = unique_temp_path("svmp_meshtranslator_quad9_face");
  volume_path.replace_extension(".vtu");
  write_quadratic_quad_volume_mesh(volume_path);
  write_subdivided_quadratic_edge_face_mesh(face_path);

  auto mesh = load_volume_with_face(volume_path, "bottom", face_path);
  std::filesystem::remove(volume_path);
  std::filesystem::remove(face_path);

  const auto label = mesh->base().label_from_name("bottom");
  ASSERT_NE(label, svmp::INVALID_LABEL);
  const auto marked_faces = mesh->base().faces_with_label(label);
  ASSERT_EQ(marked_faces.size(), 1u);

  const auto face_id = marked_faces.front();
  auto [face_vertices, face_vertex_count] = mesh->base().face_vertices_span(face_id);
  ASSERT_NE(face_vertices, nullptr);
  EXPECT_EQ(face_vertex_count, 3u);
  ASSERT_LT(static_cast<std::size_t>(face_id), mesh->base().face_shapes().size());
  EXPECT_EQ(mesh->base().face_shapes()[static_cast<std::size_t>(face_id)].family,
            svmp::CellFamily::Line);
  EXPECT_EQ(mesh->base().face_shapes()[static_cast<std::size_t>(face_id)].order, 2);
  EXPECT_EQ(mesh->base().get_set(svmp::EntityKind::Face, "bottom").size(), 1u);
#endif
}

TEST(MeshTranslatorFaceLabels, LabelsFullTetra10FaceFromQuadraticTriangleSurface)
{
#ifndef MESH_HAS_VTK
  GTEST_SKIP() << "VTK support is required for face-file translator coverage.";
#else
  ensure_mpi_initialized_for_mesh_translator();
  auto volume_path = unique_temp_path("svmp_meshtranslator_tet10_volume");
  const auto face_path = unique_temp_path("svmp_meshtranslator_tet10_face");
  volume_path.replace_extension(".vtu");
  write_quadratic_tetra_volume_mesh(volume_path);
  write_quadratic_triangle_face_mesh(face_path);

  auto mesh = load_volume_with_face(volume_path, "bottom", face_path);
  std::filesystem::remove(volume_path);
  std::filesystem::remove(face_path);

  const auto label = mesh->base().label_from_name("bottom");
  ASSERT_NE(label, svmp::INVALID_LABEL);
  const auto marked_faces = mesh->base().faces_with_label(label);
  ASSERT_EQ(marked_faces.size(), 1u);

  const auto face_id = marked_faces.front();
  auto face_vertices = mesh->base().face_vertices(face_id);
  ASSERT_EQ(face_vertices.size(), 6u);
  std::sort(face_vertices.begin(), face_vertices.end());
  const std::vector<svmp::index_t> expected = {0, 1, 2, 4, 5, 6};
  EXPECT_EQ(face_vertices, expected);
  ASSERT_LT(static_cast<std::size_t>(face_id), mesh->base().face_shapes().size());
  EXPECT_EQ(mesh->base().face_shapes()[static_cast<std::size_t>(face_id)].family,
            svmp::CellFamily::Triangle);
  EXPECT_EQ(mesh->base().face_shapes()[static_cast<std::size_t>(face_id)].order, 2);
  EXPECT_EQ(mesh->base().get_set(svmp::EntityKind::Face, "bottom").size(), 1u);
#endif
}

#ifdef MESH_HAS_VTK
namespace {

struct OverlappingFaceFiles {
  std::filesystem::path volume;
  std::filesystem::path wall;
  std::filesystem::path surface;

  OverlappingFaceFiles()
      : volume(unique_temp_path("svmp_meshtranslator_overlap_volume")),
        wall(unique_temp_path("svmp_meshtranslator_overlap_wall")),
        surface(unique_temp_path("svmp_meshtranslator_overlap_surface"))
  {
    volume.replace_extension(".vtu");
    write_volume_mesh(volume);
    // The same boundary triangle in two face files, as the legacy fitted
    // SPHERIC Test 10 decks did for the wall faces of the top cell row.
    const std::vector<svmp::real_t> triangle = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    };
    write_face_mesh(wall, triangle, {100, 101, 102});
    write_face_mesh(surface, triangle, {100, 101, 102});
  }
  ~OverlappingFaceFiles()
  {
    std::filesystem::remove(volume);
    std::filesystem::remove(wall);
    std::filesystem::remove(surface);
  }

  void configure(MeshParameters& mesh_params) const
  {
    mesh_params.name.set("tank");
    mesh_params.mesh_file_path.set(volume.string());
    for (const auto& [name, path] :
         {std::pair<std::string, std::filesystem::path>{"wall", wall},
          std::pair<std::string, std::filesystem::path>{"free_surface", surface}}) {
      auto* face = new FaceParameters();
      face->name.set(name);
      face->face_file_path.set(path.string());
      mesh_params.face_parameters.push_back(face);
    }
  }
};

} // namespace
#endif

TEST(MeshTranslatorFaceLabels, OverlappingFaceFilesFailClosedNamingFacesAndFiles)
{
#ifndef MESH_HAS_VTK
  GTEST_SKIP() << "VTK support is required for face-file translator coverage.";
#else
  ensure_mpi_initialized_for_mesh_translator();
  const OverlappingFaceFiles files;
  MeshParameters mesh_params;
  EXPECT_FALSE(mesh_params.allow_overlapping_face_files.value());
  files.configure(mesh_params);

  try {
    (void)application::translators::MeshTranslator::loadMesh(mesh_params);
    FAIL() << "overlapping face files must be rejected";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("the face files of <Add_mesh name=\"tank\"> overlap"),
              std::string::npos)
        << message;
    EXPECT_NE(message.find("<Add_face name=\"wall\"> ('" + files.wall.string() +
                           "') and <Add_face name=\"free_surface\"> ('" +
                           files.surface.string() + "') share 1 boundary face(s)"),
              std::string::npos)
        << message;
    EXPECT_NE(message.find("<Allow_overlapping_face_files>true</Allow_overlapping_face_files>"),
              std::string::npos)
        << message;
  }
#endif
}

TEST(MeshTranslatorFaceLabels, LegacyOptOutKeepsTheLastLabelAndWarns)
{
#ifndef MESH_HAS_VTK
  GTEST_SKIP() << "VTK support is required for face-file translator coverage.";
#else
  ensure_mpi_initialized_for_mesh_translator();
  const OverlappingFaceFiles files;
  MeshParameters mesh_params;
  files.configure(mesh_params);
  mesh_params.allow_overlapping_face_files.set("true");

  testing::internal::CaptureStderr();
  auto mesh = application::translators::MeshTranslator::loadMesh(mesh_params);
  const auto warnings = testing::internal::GetCapturedStderr();

  ASSERT_NE(mesh, nullptr);
  EXPECT_NE(warnings.find("1 boundary face(s) of face file 'wall' are also listed in face file "
                          "'free_surface'"),
            std::string::npos)
      << warnings;
  const auto wall = mesh->base().label_from_name("wall");
  const auto surface = mesh->base().label_from_name("free_surface");
  ASSERT_NE(wall, svmp::INVALID_LABEL);
  ASSERT_NE(surface, svmp::INVALID_LABEL);
  EXPECT_TRUE(mesh->base().faces_with_label(wall).empty());
  EXPECT_EQ(mesh->base().faces_with_label(surface).size(), 1u);
  EXPECT_EQ(mesh->base().get_set(svmp::EntityKind::Face, "wall").size(), 1u);
  EXPECT_EQ(mesh->base().get_set(svmp::EntityKind::Face, "free_surface").size(), 1u);
#endif
}

TEST(MeshTranslatorFaceLabels, OverlapOptOutIsParsedFromAddMesh)
{
  tinyxml2::XMLDocument document;
  ASSERT_EQ(document.Parse(R"xml(
<Add_mesh name="tank">
  <Mesh_file_path>mesh.vtu</Mesh_file_path>
  <Allow_overlapping_face_files>true</Allow_overlapping_face_files>
</Add_mesh>
)xml"), tinyxml2::XML_SUCCESS)
      << document.ErrorStr();
  auto* element = document.FirstChildElement("Add_mesh");
  ASSERT_NE(element, nullptr);

  MeshParameters parsed;
  ASSERT_NO_THROW(parsed.set_values(element));
  EXPECT_TRUE(parsed.allow_overlapping_face_files.value());
}

TEST(MeshTranslatorFaceLabels, DisjointFaceFilesDoNotWarn)
{
#ifndef MESH_HAS_VTK
  GTEST_SKIP() << "VTK support is required for face-file translator coverage.";
#else
  ensure_mpi_initialized_for_mesh_translator();
  auto volume_path = unique_temp_path("svmp_meshtranslator_disjoint_volume");
  const auto face_path = unique_temp_path("svmp_meshtranslator_disjoint_face");
  volume_path.replace_extension(".vtu");
  write_volume_mesh(volume_path);
  write_face_mesh(face_path,
                  {
                      0.0, 0.0, 0.0,
                      1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                  },
                  {100, 101, 102});

  testing::internal::CaptureStderr();
  auto mesh = load_volume_with_face(volume_path, "bottom", face_path);
  const auto warnings = testing::internal::GetCapturedStderr();
  std::filesystem::remove(volume_path);
  std::filesystem::remove(face_path);

  ASSERT_NE(mesh, nullptr);
  EXPECT_EQ(warnings.find("are also listed in face file"), std::string::npos) << warnings;
#endif
}

// ---- Startup partition weighting (<Partition_weighting>free_surface) ----

namespace {

// Five separate tetrahedra (four vertices each) whose vertex values put
// cell 0 below the isovalue 0, cell 1 above it, cell 2 across it, and
// cells 3 and 4 on it at one vertex (from above and from below).
svmp::MeshBase make_sign_test_mesh(svmp::FieldScalarType type)
{
  svmp::MeshBase base;
  std::vector<svmp::real_t> x;
  std::vector<svmp::offset_t> offsets = {0};
  std::vector<svmp::index_t> conn;
  for (int c = 0; c < 5; ++c) {
    const svmp::real_t x0 = static_cast<svmp::real_t>(2 * c);
    const svmp::real_t corners[4][3] = {{x0, 0, 0}, {x0 + 1, 0, 0}, {x0, 1, 0}, {x0, 0, 1}};
    for (const auto& p : corners) {
      x.insert(x.end(), {p[0], p[1], p[2]});
    }
    for (int k = 0; k < 4; ++k) {
      conn.push_back(static_cast<svmp::index_t>(4 * c + k));
    }
    offsets.push_back(static_cast<svmp::offset_t>(conn.size()));
  }
  std::vector<svmp::CellShape> shapes(5, make_shape(svmp::CellFamily::Tetra, 4));
  base.build_from_arrays(3, x, offsets, conn, shapes);
  const double values[5][4] = {
      {-1.0, -2.0, -0.5, -1.0},  // below
      {1.0, 2.0, 0.5, 1.0},      // above
      {-1.0, 2.0, 0.5, 1.0},     // across
      {0.0, 2.0, 0.5, 1.0},      // on it, otherwise above
      {-1.0, -2.0, -0.5, 0.0},   // on it, otherwise below
  };
  const auto h = base.attach_field(svmp::EntityKind::Vertex, "phi", type, 1);
  for (int c = 0; c < 5; ++c) {
    for (int k = 0; k < 4; ++k) {
      if (type == svmp::FieldScalarType::Float64) {
        base.field_data_as<double>(h)[4 * c + k] = values[c][k];
      } else {
        base.field_data_as<float>(h)[4 * c + k] = static_cast<float>(values[c][k]);
      }
    }
  }
  return base;
}

// MeshParameters is neither copyable nor movable: parse into the caller's.
void parse_add_mesh(const char* xml, MeshParameters& parsed)
{
  tinyxml2::XMLDocument document;
  EXPECT_EQ(document.Parse(xml), tinyxml2::XML_SUCCESS) << document.ErrorStr();
  auto* element = document.FirstChildElement("Add_mesh");
  EXPECT_NE(element, nullptr);
  if (element != nullptr) {
    parsed.set_values(element);
  }
}

} // namespace

TEST(MeshPartitionWeighting, DefaultsAreAutomaticWithCutVolumeWorkWeights)
{
  using application::translators::MeshTranslator;
  using application::translators::PartitionWeightingMode;
  MeshParameters defaults;
  EXPECT_EQ(defaults.partition_weighting.value(), "auto");
  EXPECT_EQ(MeshTranslator::partitionWeightingMode(defaults), PartitionWeightingMode::Auto);
  // Relative cut-volume work of a full active, a cut and an inactive cell.
  EXPECT_EQ(defaults.partition_weight_active.value(), 1.0);
  EXPECT_EQ(defaults.partition_weight_cut.value(), 2.0);
  EXPECT_EQ(defaults.partition_weight_inactive.value(), 0.0);
}

TEST(MeshPartitionWeighting, ParsesTheModeAndRejectsUnknownValues)
{
  using application::translators::MeshTranslator;
  using application::translators::PartitionWeightingMode;
  MeshParameters absent;
  parse_add_mesh(R"xml(
<Add_mesh name="box">
  <Mesh_file_path>mesh.vtu</Mesh_file_path>
</Add_mesh>
)xml", absent);
  EXPECT_EQ(MeshTranslator::partitionWeightingMode(absent), PartitionWeightingMode::Auto);

  MeshParameters none;
  parse_add_mesh(R"xml(
<Add_mesh name="box">
  <Mesh_file_path>mesh.vtu</Mesh_file_path>
  <Partition_weighting>none</Partition_weighting>
</Add_mesh>
)xml", none);
  EXPECT_EQ(MeshTranslator::partitionWeightingMode(none), PartitionWeightingMode::None);

  MeshParameters automatic;
  parse_add_mesh(R"xml(
<Add_mesh name="box">
  <Mesh_file_path>mesh.vtu</Mesh_file_path>
  <Partition_weighting>AUTO</Partition_weighting>
</Add_mesh>
)xml", automatic);
  EXPECT_EQ(MeshTranslator::partitionWeightingMode(automatic), PartitionWeightingMode::Auto);

  MeshParameters on;
  parse_add_mesh(R"xml(
<Add_mesh name="box">
  <Mesh_file_path>mesh.vtu</Mesh_file_path>
  <Partition_weighting>Free_Surface</Partition_weighting>
  <Partition_weight_active>2</Partition_weight_active>
  <Partition_weight_inactive>0.25</Partition_weight_inactive>
  <Partition_weight_cut>3.5</Partition_weight_cut>
</Add_mesh>
)xml", on);
  EXPECT_EQ(MeshTranslator::partitionWeightingMode(on), PartitionWeightingMode::FreeSurface);
  EXPECT_EQ(on.partition_weight_active.value(), 2.0);
  EXPECT_EQ(on.partition_weight_inactive.value(), 0.25);
  EXPECT_EQ(on.partition_weight_cut.value(), 3.5);

  MeshParameters bogus;
  bogus.name.set("box");
  bogus.partition_weighting.set("cut_cells");
  EXPECT_THROW((void)MeshTranslator::partitionWeightingMode(bogus), std::runtime_error);
}

TEST(MeshPartitionWeighting, MapsClassWeightsOntoTheActiveSide)
{
  using application::translators::MeshTranslator;
  MeshParameters params;
  params.name.set("box");
  params.partition_weighting.set("free_surface");

  // Liquid on the negative side (LevelSetNegative): negative cells are active.
  const auto negative = MeshTranslator::freeSurfacePartitionWeights(params, "phi", 0.25, true);
  EXPECT_EQ(negative.vertex_field, "phi");
  EXPECT_EQ(negative.isovalue, 0.25);
  EXPECT_EQ(negative.negative, 1.0);
  EXPECT_EQ(negative.positive, 0.0);
  EXPECT_EQ(negative.cut, 2.0);
  EXPECT_FALSE(negative.fallback_to_unweighted);

  // Liquid on the positive side.
  const auto positive = MeshTranslator::freeSurfacePartitionWeights(params, "phi", 0.0, false);
  EXPECT_EQ(positive.negative, 0.0);
  EXPECT_EQ(positive.positive, 1.0);
  EXPECT_EQ(positive.cut, 2.0);

  EXPECT_THROW((void)MeshTranslator::freeSurfacePartitionWeights(params, "", 0.0, true), std::runtime_error);
  params.partition_weight_cut.set("-1");
  EXPECT_THROW((void)MeshTranslator::freeSurfacePartitionWeights(params, "phi", 0.0, true), std::runtime_error);
  params.partition_weight_cut.set("2e4");
  EXPECT_THROW((void)MeshTranslator::freeSurfacePartitionWeights(params, "phi", 0.0, true), std::runtime_error);
}

TEST(MeshPartitionWeighting, ResolvesTheStartupWeightsFromModeRunAndFreeSurface)
{
  using application::translators::FreeSurfaceLevelSet;
  using application::translators::MeshTranslator;
  FreeSurfaceLevelSet free_surface;
  free_surface.level_set_field = "phi";
  free_surface.isovalue = 0.0;
  free_surface.negative_side_active = true;
  const std::optional<FreeSurfaceLevelSet> none_found;

  MeshParameters automatic;
  automatic.name.set("box");
  // Default (auto), several ranks, unfitted free surface: weighted, with the
  // fallback to the unweighted partition when phi is not available.
  const auto weighted = MeshTranslator::startupPartitionWeights(automatic, free_surface, 4);
  EXPECT_EQ(weighted.vertex_field, "phi");
  EXPECT_EQ(weighted.negative, 1.0);
  EXPECT_EQ(weighted.positive, 0.0);
  EXPECT_EQ(weighted.cut, 2.0);
  EXPECT_TRUE(weighted.fallback_to_unweighted);
  // Serial runs are not partitioned; runs without an unfitted free surface
  // keep the cell-count partition.
  EXPECT_TRUE(MeshTranslator::startupPartitionWeights(automatic, free_surface, 1).vertex_field.empty());
  EXPECT_TRUE(MeshTranslator::startupPartitionWeights(automatic, none_found, 4).vertex_field.empty());

  MeshParameters none;
  none.name.set("box");
  none.partition_weighting.set("none");
  EXPECT_TRUE(MeshTranslator::startupPartitionWeights(none, free_surface, 4).vertex_field.empty());
  EXPECT_TRUE(MeshTranslator::startupPartitionWeights(none, none_found, 4).vertex_field.empty());

  MeshParameters explicit_on;
  explicit_on.name.set("box");
  explicit_on.partition_weighting.set("free_surface");
  const auto required = MeshTranslator::startupPartitionWeights(explicit_on, free_surface, 2);
  EXPECT_EQ(required.vertex_field, "phi");
  EXPECT_FALSE(required.fallback_to_unweighted);
  EXPECT_TRUE(MeshTranslator::startupPartitionWeights(explicit_on, none_found, 1).vertex_field.empty());
  EXPECT_THROW((void)MeshTranslator::startupPartitionWeights(explicit_on, none_found, 2), std::runtime_error);
}

TEST(MeshPartitionWeighting, OnlyUnfittedFreeSurfaceDecksResolveToWeights)
{
  using application::translators::MeshTranslator;
  const auto resolve = [](const char* xml) {
    tinyxml2::XMLDocument doc;
    EXPECT_EQ(doc.Parse(xml), tinyxml2::XML_SUCCESS) << doc.ErrorStr();
    auto* root = doc.FirstChildElement(Parameters::FSI_FILE.c_str());
    EXPECT_NE(root, nullptr);
    auto params = std::make_unique<Parameters>();
    if (root != nullptr) {
      params->set_equation_values(root);
    }
    const auto free_surface =
        MeshTranslator::unfittedFreeSurfaceLevelSet(application::core::activeCutVolumeRequests(*params));
    MeshParameters mesh;
    mesh.name.set("box");
    return std::make_pair(free_surface, MeshTranslator::startupPartitionWeights(mesh, free_surface, 4));
  };

  // Unfitted level-set free surface, liquid on the positive side.
  const auto [unfitted, unfitted_weights] = resolve(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>psi</Level_set_field_name>
      <Level_set_isovalue>0.25</Level_set_isovalue>
      <Generated_interface_domain_id>water_air</Generated_interface_domain_id>
      <Active_domain>LevelSetPositive</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
  ASSERT_TRUE(unfitted.has_value());
  EXPECT_EQ(unfitted->level_set_field, "psi");
  EXPECT_EQ(unfitted->isovalue, 0.25);
  EXPECT_FALSE(unfitted->negative_side_active);
  EXPECT_EQ(unfitted_weights.vertex_field, "psi");
  EXPECT_EQ(unfitted_weights.positive, 1.0);
  EXPECT_EQ(unfitted_weights.negative, 0.0);
  EXPECT_TRUE(unfitted_weights.fallback_to_unweighted);

  // Fitted (ALE) free surface: no unfitted free surface, cell-count partition.
  const auto [fitted, fitted_weights] = resolve(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>FittedALE</Implementation>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
  EXPECT_FALSE(fitted.has_value());
  EXPECT_TRUE(fitted_weights.vertex_field.empty());

  // Two-fluid material interface (both phases active): cell-count partition.
  const auto [two_fluid, two_fluid_weights] = resolve(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Free_surface_physical_model>IncompressibleTwoFluid</Free_surface_physical_model>
    <Level_set_field_name>phi</Level_set_field_name>
    <Generated_interface_domain_id>material_interface</Generated_interface_domain_id>
    <Material_interface_marker>71</Material_interface_marker>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
  EXPECT_FALSE(two_fluid.has_value());
  EXPECT_TRUE(two_fluid_weights.vertex_field.empty());

  // No free surface at all.
  const auto [plain, plain_weights] = resolve(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="wall">
      <Type>Dir</Type>
      <Value>0.0</Value>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
  EXPECT_FALSE(plain.has_value());
  EXPECT_TRUE(plain_weights.vertex_field.empty());
}

TEST(MeshPartitionWeighting, MeshOptionsRequestWeightsOnlyWithAVertexField)
{
  std::unordered_map<std::string, std::string> kv;
  EXPECT_FALSE(svmp::startup_partition_sign_weights(kv).has_value());
  kv["partition_weight_cut"] = "2";
  EXPECT_FALSE(svmp::startup_partition_sign_weights(kv).has_value());
  kv["partition_weight_vertex_field"] = "";
  EXPECT_FALSE(svmp::startup_partition_sign_weights(kv).has_value());

  kv["partition_weight_vertex_field"] = "phi";
  kv["partition_weight_isovalue"] = "-0.125";
  kv["partition_weight_negative"] = "1";
  kv["partition_weight_positive"] = "0";
  const auto parsed = svmp::startup_partition_sign_weights(kv);
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->vertex_field, "phi");
  EXPECT_EQ(parsed->isovalue, -0.125);
  EXPECT_EQ(parsed->negative, 1.0);
  EXPECT_EQ(parsed->positive, 0.0);
  EXPECT_EQ(parsed->cut, 2.0);

  for (const char* bad : {"-1", "nan", "1e5", "2x", "inf"}) {
    auto copy = kv;
    copy["partition_weight_cut"] = bad;
    EXPECT_THROW((void)svmp::startup_partition_sign_weights(copy), std::invalid_argument) << bad;
  }
  auto bad_iso = kv;
  bad_iso["partition_weight_isovalue"] = "zero";
  EXPECT_THROW((void)svmp::startup_partition_sign_weights(bad_iso), std::invalid_argument);

  EXPECT_FALSE(parsed->fallback_to_unweighted);
  auto fallback = kv;
  fallback["partition_weight_fallback"] = "unweighted";
  EXPECT_TRUE(svmp::startup_partition_sign_weights(fallback)->fallback_to_unweighted);
  fallback["partition_weight_fallback"] = "ignore";
  EXPECT_THROW((void)svmp::startup_partition_sign_weights(fallback), std::invalid_argument);
}

TEST(MeshPartitionWeighting, CellWeightsFollowTheVertexSigns)
{
  // Integer units: 100 per unit weight, at least one.
  EXPECT_EQ(svmp::startup_partition_weight_units(1.0), 100);
  EXPECT_EQ(svmp::startup_partition_weight_units(1.5), 150);
  EXPECT_EQ(svmp::startup_partition_weight_units(0.0), 1);
  EXPECT_EQ(svmp::startup_partition_weight_units(0.004), 1);
  EXPECT_EQ(svmp::startup_partition_weight_units(0.026), 3);

  svmp::StartupPartitionSignWeights weights;
  weights.vertex_field = "phi";
  weights.isovalue = 0.0;
  weights.negative = 1.0;
  weights.positive = 0.0;
  weights.cut = 2.0;
  for (const auto type : {svmp::FieldScalarType::Float64, svmp::FieldScalarType::Float32}) {
    const auto mesh = make_sign_test_mesh(type);
    const auto w = svmp::startup_partition_cell_weights(mesh, weights);
    ASSERT_EQ(w.size(), 5u);
    EXPECT_EQ(w[0], 100);  // below: active
    EXPECT_EQ(w[1], 1);    // above: inactive (zero weight, one unit)
    EXPECT_EQ(w[2], 200);  // across: cut
    EXPECT_EQ(w[3], 200);  // a vertex on the isovalue counts as cut
    EXPECT_EQ(w[4], 200);
  }

  // A shifted isovalue moves the classes: at -3 every cell lies above it.
  weights.isovalue = -3.0;
  const auto shifted = svmp::startup_partition_cell_weights(make_sign_test_mesh(svmp::FieldScalarType::Float64), weights);
  for (const auto w : shifted) {
    EXPECT_EQ(w, 1);
  }
}

TEST(MeshPartitionWeighting, CellWeightsRejectMissingOrNonScalarFields)
{
  svmp::StartupPartitionSignWeights weights;
  weights.vertex_field = "level_set";
  auto mesh = make_sign_test_mesh(svmp::FieldScalarType::Float64);
  EXPECT_THROW((void)svmp::startup_partition_cell_weights(mesh, weights), std::runtime_error);

  (void)mesh.attach_field(svmp::EntityKind::Vertex, "level_set", svmp::FieldScalarType::Float64, 3);
  EXPECT_THROW((void)svmp::startup_partition_cell_weights(mesh, weights), std::runtime_error);

  (void)mesh.attach_field(svmp::EntityKind::Vertex, "flag", svmp::FieldScalarType::Int32, 1);
  weights.vertex_field = "flag";
  EXPECT_THROW((void)svmp::startup_partition_cell_weights(mesh, weights), std::runtime_error);

  // A mesh without cells needs no field.
  svmp::MeshBase empty;
  EXPECT_TRUE(svmp::startup_partition_cell_weights(empty, weights).empty());
}
