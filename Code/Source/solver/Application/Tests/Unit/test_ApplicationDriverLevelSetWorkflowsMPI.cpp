#include <gtest/gtest.h>

// The workflow helper under test currently has internal linkage in the
// application driver.  Include the implementation, as the serial workflow
// tests do, so this exercises the production graph-extension implementation.
#include "../../Core/ApplicationDriver.cpp"

#include "Application/Translators/MeshTranslator.h"
#include "FE/Backends/FSILS/FsilsFactory.h"
#include "FE/Backends/FSILS/FsilsMatrix.h"
#include "FE/Backends/Utils/BackendOptions.h"
#include "FE/Forms/Forms.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Spaces/ProductSpace.h"
#include "FE/Systems/FormsInstaller.h"
#include "Mesh/Mesh.h"
#include "FE/Spaces/H1Space.h"
#include "Mesh/Fields/MeshFields.h"
#include "Parameters.h"
#include "tinyxml2.h"

#ifdef MESH_HAS_VTK
#include "Mesh/IO/VTKWriter.h"
#endif

#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

constexpr int kCellCount = 8;
constexpr std::size_t kComponents = 2u;
constexpr svmp::label_t kHorizontalWall = 4242;
constexpr svmp::label_t kLeftOnlyWall = 5101;
constexpr svmp::label_t kLeftOnlyExtraWall = 5102;
constexpr svmp::label_t kRightOnlyWall = 5201;

void labelHorizontalWalls(svmp::Mesh& mesh)
{
  auto& local_mesh = mesh.local_mesh();
  for (const auto face : local_mesh.boundary_faces()) {
    const auto normal = local_mesh.face_normal(face);
    if (std::abs(normal[1]) > 0.9 * std::abs(normal[0])) {
      mesh.set_boundary_label(face, kHorizontalWall);
    }
  }
}

struct QuadStripArrays {
  std::vector<svmp::real_t> coordinates;
  std::vector<svmp::offset_t> offsets;
  std::vector<svmp::index_t> connectivity;
  std::vector<svmp::CellShape> shapes;
};

[[nodiscard]] svmp::index_t stripVertex(int x_plane, int y)
{
  return static_cast<svmp::index_t>(2 * x_plane + y);
}

[[nodiscard]] QuadStripArrays makeQuadStripArrays()
{
  QuadStripArrays arrays;
  arrays.coordinates.reserve(
      static_cast<std::size_t>(2 * (kCellCount + 1)) * 2u);
  for (int x = 0; x <= kCellCount; ++x) {
    for (int y = 0; y <= 1; ++y) {
      arrays.coordinates.push_back(static_cast<svmp::real_t>(x));
      arrays.coordinates.push_back(static_cast<svmp::real_t>(y));
    }
  }

  arrays.offsets.assign(static_cast<std::size_t>(kCellCount) + 1u, 0);
  arrays.connectivity.reserve(static_cast<std::size_t>(kCellCount) * 4u);
  svmp::CellShape quad{};
  quad.family = svmp::CellFamily::Quad;
  quad.num_corners = 4;
  quad.order = 1;
  arrays.shapes.assign(static_cast<std::size_t>(kCellCount), quad);
  for (int cell = 0; cell < kCellCount; ++cell) {
    arrays.connectivity.push_back(stripVertex(cell, 0));
    arrays.connectivity.push_back(stripVertex(cell + 1, 0));
    arrays.connectivity.push_back(stripVertex(cell + 1, 1));
    arrays.connectivity.push_back(stripVertex(cell, 1));
    arrays.offsets[static_cast<std::size_t>(cell) + 1u] =
        static_cast<svmp::offset_t>(arrays.connectivity.size());
  }
  return arrays;
}

#ifdef MESH_HAS_VTK
constexpr svmp::gid_t kTranslatorVertexGidBase = 7000;
constexpr svmp::gid_t kTranslatorCellGidBase = 8000;

[[nodiscard]] double translatorPhiValue(svmp::gid_t gid)
{
  return -0.75 +
         0.125 * static_cast<double>(gid - kTranslatorVertexGidBase);
}

[[nodiscard]] double translatorPressureValue(svmp::gid_t gid)
{
  return 100.0 +
         2.5 * static_cast<double>(gid - kTranslatorVertexGidBase);
}

void writeMeshTranslatorGhostLayerFixture(
    const std::filesystem::path& volume_path,
    const std::filesystem::path& bottom_face_path)
{
  const auto arrays = makeQuadStripArrays();
  svmp::MeshBase volume;
  volume.build_from_arrays(/*spatial_dim=*/2,
                           arrays.coordinates,
                           arrays.offsets,
                           arrays.connectivity,
                           arrays.shapes);

  std::vector<svmp::gid_t> vertex_gids(volume.n_vertices());
  for (std::size_t vertex = 0; vertex < volume.n_vertices(); ++vertex) {
    vertex_gids[vertex] =
        kTranslatorVertexGidBase + static_cast<svmp::gid_t>(vertex);
  }
  std::vector<svmp::gid_t> cell_gids(volume.n_cells());
  for (std::size_t cell = 0; cell < volume.n_cells(); ++cell) {
    cell_gids[cell] =
        kTranslatorCellGidBase + static_cast<svmp::gid_t>(cell);
  }
  volume.set_vertex_gids(std::move(vertex_gids));
  volume.set_cell_gids(std::move(cell_gids));

  // These deliberately have no FieldDescriptor.  The SPHERIC Test 05 decks
  // carry the same descriptorless VTU point arrays, and the halo builder must
  // preserve their values on vertices introduced only by a ghost cell.
  const auto phi_handle = volume.attach_field(
      svmp::EntityKind::Vertex,
      "phi",
      svmp::FieldScalarType::Float64,
      1);
  const auto pressure_handle = volume.attach_field(
      svmp::EntityKind::Vertex,
      "Pressure",
      svmp::FieldScalarType::Float64,
      1);
  const auto node_id_handle = volume.attach_field(
      svmp::EntityKind::Vertex,
      "GlobalNodeID",
      svmp::FieldScalarType::Int32,
      1);
  auto* phi = volume.field_data_as<double>(phi_handle);
  auto* pressure = volume.field_data_as<double>(pressure_handle);
  auto* node_ids = volume.field_data_as<std::int32_t>(node_id_handle);
  const auto& gids = volume.vertex_gids();
  for (std::size_t vertex = 0; vertex < volume.n_vertices(); ++vertex) {
    phi[vertex] = translatorPhiValue(gids[vertex]);
    pressure[vertex] = translatorPressureValue(gids[vertex]);
    node_ids[vertex] = static_cast<std::int32_t>(gids[vertex]);
  }
  // Match the checked-in SPHERIC VTU convention: GlobalNodeID is a regular
  // point-data array rather than VTK's designated GlobalIds array.  The VTK
  // reader must use it for mesh numbering and still retain it as a field.
  volume.set_vertex_gids({});
  volume.finalize();

  svmp::MeshIOOptions volume_options{};
  volume_options.path = volume_path.string();
  volume_options.format = "vtu";
  volume_options.kv["codim1_topology"] = "none";
  volume_options.kv["edge_topology"] = "false";
  svmp::VTKWriter::write(volume, volume_options);

  svmp::MeshBase bottom_face;
  std::vector<svmp::real_t> face_coordinates;
  face_coordinates.reserve(static_cast<std::size_t>(kCellCount + 1) * 2u);
  std::vector<svmp::gid_t> face_vertex_gids;
  face_vertex_gids.reserve(static_cast<std::size_t>(kCellCount + 1));
  for (int x = 0; x <= kCellCount; ++x) {
    face_coordinates.push_back(static_cast<svmp::real_t>(x));
    face_coordinates.push_back(0.0);
    face_vertex_gids.push_back(
        kTranslatorVertexGidBase +
        static_cast<svmp::gid_t>(stripVertex(x, 0)));
  }

  std::vector<svmp::offset_t> face_offsets(
      static_cast<std::size_t>(kCellCount) + 1u,
      0);
  std::vector<svmp::index_t> face_connectivity;
  face_connectivity.reserve(static_cast<std::size_t>(kCellCount) * 2u);
  svmp::CellShape line{};
  line.family = svmp::CellFamily::Line;
  line.num_corners = 2;
  line.order = 1;
  std::vector<svmp::CellShape> face_shapes(
      static_cast<std::size_t>(kCellCount),
      line);
  for (int cell = 0; cell < kCellCount; ++cell) {
    face_connectivity.push_back(static_cast<svmp::index_t>(cell));
    face_connectivity.push_back(static_cast<svmp::index_t>(cell + 1));
    face_offsets[static_cast<std::size_t>(cell) + 1u] =
        static_cast<svmp::offset_t>(face_connectivity.size());
  }
  bottom_face.build_from_arrays(/*spatial_dim=*/2,
                                std::move(face_coordinates),
                                std::move(face_offsets),
                                std::move(face_connectivity),
                                std::move(face_shapes));
  bottom_face.set_vertex_gids(std::move(face_vertex_gids));
  const auto face_node_id_handle = bottom_face.attach_field(
      svmp::EntityKind::Vertex,
      "GlobalNodeID",
      svmp::FieldScalarType::Int32,
      1);
  auto* face_node_ids =
      bottom_face.field_data_as<std::int32_t>(face_node_id_handle);
  const auto& bottom_gids = bottom_face.vertex_gids();
  for (std::size_t vertex = 0; vertex < bottom_face.n_vertices(); ++vertex) {
    face_node_ids[vertex] =
        static_cast<std::int32_t>(bottom_gids[vertex]);
  }
  bottom_face.set_vertex_gids({});
  bottom_face.finalize();

  svmp::MeshIOOptions face_options{};
  face_options.path = bottom_face_path.string();
  face_options.format = "vtp";
  face_options.kv["codim1_topology"] = "none";
  face_options.kv["edge_topology"] = "false";
  svmp::VTKWriter::write(bottom_face, face_options);
}
#endif

[[nodiscard]] std::shared_ptr<svmp::Mesh> makeSerialQuadStrip()
{
  const auto arrays = makeQuadStripArrays();
  auto base = std::make_shared<svmp::MeshBase>();
  base->build_from_arrays(/*spatial_dim=*/2,
                          arrays.coordinates,
                          arrays.offsets,
                          arrays.connectivity,
                          arrays.shapes);
  std::vector<svmp::gid_t> vertex_gids(base->n_vertices(), 0);
  for (std::size_t vertex = 0; vertex < vertex_gids.size(); ++vertex) {
    vertex_gids[vertex] = static_cast<svmp::gid_t>(vertex);
  }
  std::vector<svmp::gid_t> cell_gids(base->n_cells(), 0);
  for (std::size_t cell = 0; cell < cell_gids.size(); ++cell) {
    cell_gids[cell] = static_cast<svmp::gid_t>(cell);
  }
  base->set_vertex_gids(std::move(vertex_gids));
  base->set_cell_gids(std::move(cell_gids));
  base->finalize();
  auto mesh =
      svmp::create_mesh(std::move(base), svmp::MeshComm(MPI_COMM_SELF));
  labelHorizontalWalls(*mesh);
  return mesh;
}

[[nodiscard]] std::shared_ptr<svmp::Mesh> makePartitionedQuadStrip()
{
  const auto arrays = makeQuadStripArrays();
  auto mesh = std::make_shared<svmp::Mesh>(svmp::MeshComm(MPI_COMM_WORLD));
  mesh->build_from_arrays_global_and_partition(
      /*spatial_dim=*/2,
      arrays.coordinates,
      arrays.offsets,
      arrays.connectivity,
      arrays.shapes,
      svmp::PartitionHint::Cells,
      /*ghost_layers=*/1,
      {{"partition_method", "block"}});
  labelHorizontalWalls(*mesh);
  return mesh;
}

[[nodiscard]] std::shared_ptr<svmp::Mesh>
makePartitionedQuadStripWithRankDisjointWallMarkers()
{
  const auto arrays = makeQuadStripArrays();
  auto mesh = std::make_shared<svmp::Mesh>(svmp::MeshComm(MPI_COMM_WORLD));
  mesh->build_from_arrays_global_and_partition(
      /*spatial_dim=*/2,
      arrays.coordinates,
      arrays.offsets,
      arrays.connectivity,
      arrays.shapes,
      svmp::PartitionHint::Cells,
      /*ghost_layers=*/1,
      {{"partition_method", "block"}});

  auto& local_mesh = mesh->local_mesh();
  for (const auto face : local_mesh.boundary_faces()) {
    const auto center = local_mesh.face_center(face);
    if (std::abs(center[0]) <= 1.0e-12) {
      mesh->set_boundary_label(face, kLeftOnlyWall);
    } else if (std::abs(center[0] - static_cast<double>(kCellCount)) <=
               1.0e-12) {
      mesh->set_boundary_label(face, kRightOnlyWall);
    } else if (std::abs(center[1]) <= 1.0e-12 && center[0] < 1.0) {
      // Rank 0 deliberately owns one more marker than rank 1.  Before the
      // communicator-wide union, this made the collective sequence in the
      // active-cut refresh differ across ranks.
      mesh->set_boundary_label(face, kLeftOnlyExtraWall);
    }
  }
  return mesh;
}

[[nodiscard]] std::unique_ptr<Parameters>
parseMpiWorkflowParametersXml(const char* xml)
{
  tinyxml2::XMLDocument doc;
  const auto status = doc.Parse(xml);
  if (status != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error(doc.ErrorStr());
  }
  auto* root = doc.FirstChildElement(Parameters::FSI_FILE.c_str());
  if (root == nullptr) {
    throw std::runtime_error("missing root solver element");
  }
  auto params = std::make_unique<Parameters>();
  params->set_equation_values(root);
  return params;
}

struct ExtensionInputs {
  std::vector<double> phi;
  std::vector<double> source;
  std::vector<std::uint8_t> active;
};

[[nodiscard]] ExtensionInputs makeExtensionInputs(const svmp::Mesh& mesh)
{
  ExtensionInputs input;
  input.phi.assign(mesh.n_vertices(), 0.0);
  input.source.assign(mesh.n_vertices() * kComponents, 0.0);
  input.active.assign(mesh.n_vertices(), 0u);
  const auto& coordinates = mesh.X_ref();
  for (std::size_t vertex = 0; vertex < mesh.n_vertices(); ++vertex) {
    const double x = static_cast<double>(coordinates[2u * vertex]);
    const double y = static_cast<double>(coordinates[2u * vertex + 1u]);
    input.phi[vertex] = x - 0.25;
    input.active[vertex] = input.phi[vertex] <= 0.0 ? 1u : 0u;
    // This field is constant along the level-set normal and affine along the
    // interface tangent.  The dry-wall projection should remove its second
    // component without changing the first component away from the end wall.
    input.source[kComponents * vertex] = 2.0 + y;
    input.source[kComponents * vertex + 1u] = 5.0 - 0.5 * y;
  }
  return input;
}

[[nodiscard]] ExtensionInputs makeTwoComponentExtensionInputs(
    const svmp::Mesh& mesh)
{
  ExtensionInputs input;
  input.phi.assign(mesh.n_vertices(), 0.0);
  input.source.assign(mesh.n_vertices() * kComponents, 0.0);
  input.active.assign(mesh.n_vertices(), 0u);
  const auto& coordinates = mesh.X_ref();
  for (std::size_t vertex = 0; vertex < mesh.n_vertices(); ++vertex) {
    const double x = static_cast<double>(coordinates[2u * vertex]);
    const double y = static_cast<double>(coordinates[2u * vertex + 1u]);
    input.phi[vertex] =
        std::min(x - 0.25,
                 static_cast<double>(kCellCount) - x - 0.35);
    input.active[vertex] = input.phi[vertex] <= 0.0 ? 1u : 0u;
    const bool left_branch = x <= 0.5 * static_cast<double>(kCellCount);
    input.source[kComponents * vertex] =
        left_branch ? 2.0 + 3.0 * y : -7.0 + 2.0 * y;
    input.source[kComponents * vertex + 1u] =
        left_branch ? -1.0 + 0.5 * y : 11.0 - 4.0 * y;
  }
  return input;
}

struct ExtensionRun {
  WallCompatibleVelocityExtensionResult report;
  std::vector<double> values;
  std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow> rows;
};

[[nodiscard]] ExtensionRun runExtension(const svmp::Mesh& mesh,
                                        const svmp::MeshComm& comm,
                                        std::array<bool, 3> component_mask =
                                            {false, true, false})
{
  const auto input = makeExtensionInputs(mesh);
  const std::vector<WallVelocityExtensionConstraint> constraints{{
      .boundary_label = kHorizontalWall,
      .constrained_components = component_mask}};
  ExtensionRun run;
  run.report = extendVelocityInLevelSetNormalBand(
      mesh,
      comm,
      input.phi,
      input.source,
      /*source_components=*/kComponents,
      input.active,
      /*target_components=*/kComponents,
      /*copy_components=*/kComponents,
      /*band_layers=*/kCellCount,
      /*enforce_wall_impermeability=*/true,
      std::span<const WallVelocityExtensionConstraint>(constraints),
      run.values,
      &run.rows);
  return run;
}

[[nodiscard]] ExtensionRun runTwoComponentExtension(
    const svmp::Mesh& mesh,
    const svmp::MeshComm& comm)
{
  const auto input = makeTwoComponentExtensionInputs(mesh);
  ExtensionRun run;
  run.report = extendVelocityInLevelSetNormalBand(
      mesh,
      comm,
      input.phi,
      input.source,
      /*source_components=*/kComponents,
      input.active,
      /*target_components=*/kComponents,
      /*copy_components=*/kComponents,
      /*band_layers=*/kCellCount / 2,
      /*enforce_wall_impermeability=*/false,
      std::span<const svmp::label_t>{},
      run.values);
  return run;
}

} // namespace

TEST(MeshTranslatorGhostLayersMPI,
     PreservesDescriptorlessFreeSurfaceFieldsAndBoundaryLabels)
{
#ifndef MESH_HAS_VTK
  GTEST_SKIP() << "VTK support is required for MeshTranslator MPI I/O coverage.";
#else
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  ASSERT_EQ(size, 2)
      << "This MeshTranslator ghost-layer test requires exactly two MPI ranks.";

  long long fixture_stamp = 0;
  if (rank == 0) {
    fixture_stamp = static_cast<long long>(
        std::chrono::steady_clock::now().time_since_epoch().count());
  }
  MPI_Bcast(&fixture_stamp,
            1,
            MPI_LONG_LONG,
            0,
            MPI_COMM_WORLD);
  const auto fixture_stem =
      std::filesystem::temp_directory_path() /
      ("svmp_mesh_translator_ghost_mpi_" +
       std::to_string(fixture_stamp));
  auto volume_path = fixture_stem;
  volume_path += ".vtu";
  auto bottom_face_path = fixture_stem;
  bottom_face_path += "_wall_bottom.vtp";

  int fixture_written = 1;
  std::string fixture_error;
  if (rank == 0) {
    try {
      writeMeshTranslatorGhostLayerFixture(volume_path, bottom_face_path);
    } catch (const std::exception& error) {
      fixture_written = 0;
      fixture_error = error.what();
    }
  }
  MPI_Bcast(&fixture_written, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (fixture_written == 0) {
    if (rank == 0) {
      ADD_FAILURE() << "Could not write the MeshTranslator MPI fixture: "
                    << fixture_error;
    }
    return;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  MeshParameters parameters;
  parameters.name.set("tank");
  parameters.mesh_file_path.set(volume_path.string());
  parameters.ghost_layers.set("1");
  auto* bottom_face = new FaceParameters();
  bottom_face->name.set("wall_bottom");
  bottom_face->face_file_path.set(bottom_face_path.string());
  parameters.face_parameters.push_back(bottom_face);

  std::shared_ptr<svmp::Mesh> mesh;
  int local_load_ok = 1;
  std::string load_error;
  try {
    mesh = application::translators::MeshTranslator::loadMesh(parameters);
  } catch (const std::exception& error) {
    local_load_ok = 0;
    load_error = error.what();
  }
  int load_ok = 0;
  MPI_Allreduce(&local_load_ok,
                &load_ok,
                1,
                MPI_INT,
                MPI_MIN,
                MPI_COMM_WORLD);
  if (load_ok == 0) {
    if (local_load_ok == 0) {
      ADD_FAILURE() << "MeshTranslator load failed on rank " << rank << ": "
                    << load_error;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      std::error_code error;
      std::filesystem::remove(volume_path, error);
      std::filesystem::remove(bottom_face_path, error);
    }
    return;
  }

  auto& local_mesh = mesh->local_mesh();
  const bool local_fields_present =
      local_mesh.has_field(svmp::EntityKind::Vertex, "phi") &&
      local_mesh.has_field(svmp::EntityKind::Vertex, "Pressure") &&
      local_mesh.has_field(svmp::EntityKind::Vertex, "GlobalNodeID");
  int local_fields_present_int = local_fields_present ? 1 : 0;
  int fields_present = 0;
  MPI_Allreduce(&local_fields_present_int,
                &fields_present,
                1,
                MPI_INT,
                MPI_MIN,
                MPI_COMM_WORLD);
  EXPECT_EQ(fields_present, 1)
      << "All D18/D38 free-surface input fields must survive VTU distribution.";
  if (fields_present == 0) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      std::error_code error;
      std::filesystem::remove(volume_path, error);
      std::filesystem::remove(bottom_face_path, error);
    }
    return;
  }

  const auto phi_handle =
      local_mesh.field_handle(svmp::EntityKind::Vertex, "phi");
  const auto pressure_handle =
      local_mesh.field_handle(svmp::EntityKind::Vertex, "Pressure");
  const auto node_id_handle =
      local_mesh.field_handle(svmp::EntityKind::Vertex, "GlobalNodeID");
  const auto* phi = local_mesh.field_data_as<const double>(phi_handle);
  const auto* pressure =
      local_mesh.field_data_as<const double>(pressure_handle);
  const auto* node_ids =
      local_mesh.field_data_as<const std::int32_t>(node_id_handle);
  const bool local_field_layout_ok =
      local_mesh.field_type(phi_handle) == svmp::FieldScalarType::Float64 &&
      local_mesh.field_components(phi_handle) == 1u &&
      local_mesh.field_type(pressure_handle) ==
          svmp::FieldScalarType::Float64 &&
      local_mesh.field_components(pressure_handle) == 1u &&
      local_mesh.field_type(node_id_handle) ==
          svmp::FieldScalarType::Int32 &&
      local_mesh.field_components(node_id_handle) == 1u && phi != nullptr &&
      pressure != nullptr && node_ids != nullptr;
  int local_field_layout_ok_int = local_field_layout_ok ? 1 : 0;
  int field_layout_ok = 0;
  MPI_Allreduce(&local_field_layout_ok_int,
                &field_layout_ok,
                1,
                MPI_INT,
                MPI_MIN,
                MPI_COMM_WORLD);
  EXPECT_EQ(field_layout_ok, 1)
      << "VTU field scalar types and component counts must remain unchanged.";
  if (field_layout_ok == 0) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      std::error_code error;
      std::filesystem::remove(volume_path, error);
      std::filesystem::remove(bottom_face_path, error);
    }
    return;
  }

  // VTK arrays do not carry the solver's optional FieldDescriptor.  This
  // confirms the regression really exercises the previously omitted
  // descriptorless-field path rather than a registered solver field.
  EXPECT_EQ(local_mesh.field_descriptor(phi_handle), nullptr);
  EXPECT_EQ(local_mesh.field_descriptor(pressure_handle), nullptr);
  EXPECT_EQ(local_mesh.field_descriptor(node_id_handle), nullptr);

  EXPECT_GT(mesh->n_ghost_cells(), 0u);
  EXPECT_GT(mesh->n_ghost_vertices(), 0u);
  const auto& vertex_gids = mesh->vertex_gids();
  EXPECT_EQ(vertex_gids.size(), mesh->n_vertices());
  int local_checked_ghost_vertices = 0;
  if (vertex_gids.size() == mesh->n_vertices()) {
    for (svmp::index_t vertex = 0;
         vertex < static_cast<svmp::index_t>(mesh->n_vertices());
         ++vertex) {
      if (!mesh->is_ghost_vertex(vertex)) {
        continue;
      }
      ++local_checked_ghost_vertices;
      const auto gid = vertex_gids[static_cast<std::size_t>(vertex)];
      EXPECT_NE(mesh->owner_rank_vertex(vertex), rank)
          << "rank=" << rank << " gid=" << gid;
      EXPECT_GE(mesh->owner_rank_vertex(vertex), 0)
          << "rank=" << rank << " gid=" << gid;
      EXPECT_NEAR(phi[static_cast<std::size_t>(vertex)],
                  translatorPhiValue(gid),
                  1.0e-12)
          << "rank=" << rank << " ghost gid=" << gid;
      EXPECT_NEAR(pressure[static_cast<std::size_t>(vertex)],
                  translatorPressureValue(gid),
                  1.0e-12)
          << "rank=" << rank << " ghost gid=" << gid;
      EXPECT_EQ(node_ids[static_cast<std::size_t>(vertex)],
                static_cast<std::int32_t>(gid))
          << "rank=" << rank << " ghost gid=" << gid;
    }
  }
  int checked_ghost_vertices = 0;
  MPI_Allreduce(&local_checked_ghost_vertices,
                &checked_ghost_vertices,
                1,
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD);
  EXPECT_GT(local_checked_ghost_vertices, 0)
      << "Each rank must validate fields on vertices introduced by its halo.";
  EXPECT_GT(checked_ghost_vertices, 0);

  const auto bottom_label = local_mesh.label_from_name("wall_bottom");
  EXPECT_NE(bottom_label, svmp::INVALID_LABEL);
  int local_bottom_faces = 0;
  bool local_bottom_geometry_ok = true;
  bool local_bottom_adjacency_owned = true;
  if (bottom_label != svmp::INVALID_LABEL) {
    const auto labeled_faces = local_mesh.faces_with_label(bottom_label);
    local_bottom_faces = static_cast<int>(labeled_faces.size());
    for (const auto face : labeled_faces) {
      auto [vertices, count] = local_mesh.face_vertices_span(face);
      if (vertices == nullptr || count != 2u) {
        local_bottom_geometry_ok = false;
        continue;
      }
      for (std::size_t i = 0; i < count; ++i) {
        const auto vertex = vertices[i];
        const auto y = local_mesh.X_ref()[
            2u * static_cast<std::size_t>(vertex) + 1u];
        if (std::abs(static_cast<double>(y)) > 1.0e-12) {
          local_bottom_geometry_ok = false;
        }
      }

      const auto adjacent_cells =
          local_mesh.face2cell().at(static_cast<std::size_t>(face));
      const auto cell = adjacent_cells[0] != svmp::INVALID_INDEX
                            ? adjacent_cells[0]
                            : adjacent_cells[1];
      if (cell == svmp::INVALID_INDEX || !mesh->is_owned_cell(cell)) {
        local_bottom_adjacency_owned = false;
      }
    }
  }
  int bottom_faces = 0;
  MPI_Allreduce(&local_bottom_faces,
                &bottom_faces,
                1,
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD);
  EXPECT_EQ(bottom_faces, kCellCount)
      << "Every physical bottom-wall segment must be labeled exactly once.";
  EXPECT_TRUE(local_bottom_geometry_ok);
  EXPECT_TRUE(local_bottom_adjacency_owned)
      << "MPI face reconstruction must attach labels to owned volume cells.";

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::error_code error;
    std::filesystem::remove(volume_path, error);
    std::filesystem::remove(bottom_face_path, error);
  }
#endif
}

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     ActiveSystemCommunicatorUsesFESystemCommunicator)
{
  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  ASSERT_EQ(world_size, 2);

  const auto mesh = makeSerialQuadStrip();
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/1);
  svmp::FE::systems::FESystem system(mesh);
  system.addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1});
  svmp::FE::systems::SetupOptions setup_options;
  setup_options.dof_options.my_rank = 0;
  setup_options.dof_options.world_size = 1;
  setup_options.dof_options.mpi_comm = MPI_COMM_SELF;
  ASSERT_NO_THROW(system.setup(setup_options));

  const auto active_comm = activeFESystemCommunicator(system);
  EXPECT_EQ(active_comm.size(), 1);
  EXPECT_EQ(active_comm.rank(), 0);
}

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     TraceSupportMaskPromotesRemotelyDiscoveredOwnerSourceRows)
{
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  ASSERT_EQ(size, 2)
      << "This owner-promotion test requires exactly two MPI ranks.";

  const auto mesh = makePartitionedQuadStrip();
  ASSERT_GT(mesh->n_ghost_vertices(), 0u);
  std::vector<std::uint8_t> trace_support(mesh->n_vertices(), 0u);
  long long selected_gid = -1;

  if (rank == 0) {
    const auto& local_mesh = mesh->local_mesh();
    svmp::index_t remote_owned_cell = svmp::INVALID_INDEX;
    for (svmp::index_t cell = 0; cell < local_mesh.n_cells(); ++cell) {
      if (mesh->owner_rank_cell(cell) == 1) {
        remote_owned_cell = cell;
        break;
      }
    }
    ASSERT_NE(remote_owned_cell, svmp::INVALID_INDEX)
        << "Rank 0 must retain a one-cell ghost owned by rank 1.";
    const std::array<svmp::FE::MeshIndex, 1> cells{{
        static_cast<svmp::FE::MeshIndex>(remote_owned_cell)}};
    ASSERT_GT(markVelocityExtensionTraceSupportCells(
                  *mesh,
                  std::span<const svmp::FE::MeshIndex>(cells),
                  trace_support),
              0u);

    auto [vertices, count] =
        local_mesh.cell_vertices_span(remote_owned_cell);
    ASSERT_NE(vertices, nullptr);
    const auto& gids = mesh->vertex_gids();
    for (std::size_t local = 0; local < count; ++local) {
      const auto vertex = static_cast<std::size_t>(vertices[local]);
      if (mesh->owner_rank_vertex(
              static_cast<svmp::index_t>(vertex)) == 1) {
        selected_gid = static_cast<long long>(gids[vertex]);
        break;
      }
    }
    ASSERT_GE(selected_gid, 0)
        << "The remote-owned ghost cell must contain a rank-1-owned vertex.";
  }
  MPI_Bcast(&selected_gid, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
  ASSERT_GE(selected_gid, 0);

  const auto find_selected_vertex = [&]() -> std::optional<std::size_t> {
    const auto& gids = mesh->vertex_gids();
    const auto found = std::find(
        gids.begin(),
        gids.end(),
        static_cast<svmp::gid_t>(selected_gid));
    if (found == gids.end()) {
      return std::nullopt;
    }
    return static_cast<std::size_t>(
        std::distance(gids.begin(), found));
  };
  const auto selected_vertex = find_selected_vertex();
  ASSERT_TRUE(selected_vertex.has_value());
  if (rank == 1) {
    ASSERT_EQ(trace_support[*selected_vertex], 0u)
        << "The owner must begin unmarked so the test exercises remote discovery.";
  }

  ASSERT_GT(synchronizeVelocityExtensionTraceSupportMask(
                *mesh, svmp::MeshComm(MPI_COMM_WORLD), trace_support),
            0u);
  ASSERT_EQ(trace_support[*selected_vertex], 1u);

  const auto input = makeExtensionInputs(*mesh);
  std::vector<double> extension;
  std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow> rows;
  const auto report = extendVelocityInLevelSetNormalBand(
      *mesh,
      svmp::MeshComm(MPI_COMM_WORLD),
      input.phi,
      input.source,
      /*source_components=*/kComponents,
      trace_support,
      /*target_components=*/kComponents,
      /*copy_components=*/kComponents,
      /*band_layers=*/kCellCount,
      /*enforce_wall_impermeability=*/false,
      std::span<const WallVelocityExtensionConstraint>{},
      extension,
      &rows);
  EXPECT_EQ(report.vertices_outside_band, 0u);
  for (std::size_t component = 0; component < kComponents; ++component) {
    EXPECT_DOUBLE_EQ(extension[kComponents * *selected_vertex + component],
                     input.source[kComponents * *selected_vertex + component]);
  }

  int local_source_rows = 0;
  for (const auto& row : rows) {
    if (mesh->vertex_gids()[static_cast<std::size_t>(row.vertex)] !=
        static_cast<svmp::gid_t>(selected_gid)) {
      continue;
    }
    EXPECT_EQ(rank, 1);
    ASSERT_EQ(row.dependencies.size(), 1u);
    EXPECT_EQ(row.dependencies.front().field,
              svmp::FE::level_set::
                  VelocityExtensionDependencyField::SourceVelocity);
    EXPECT_EQ(row.dependencies.front().vertex, row.vertex);
    EXPECT_EQ(row.dependencies.front().component, row.component);
    EXPECT_DOUBLE_EQ(row.dependencies.front().coefficient, 1.0);
    ++local_source_rows;
  }
  int global_source_rows = 0;
  MPI_Allreduce(&local_source_rows,
                &global_source_rows,
                1,
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD);
  EXPECT_EQ(global_source_rows, static_cast<int>(kComponents))
      << "The remotely discovered trace-support vertex must have exactly one "
         "owner source row per component.";
}

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     NormalBandExtensionMatchesSelfAcrossPartitionAndProjectsWalls)
{
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  ASSERT_EQ(size, 2)
      << "This partition-equivalence test requires exactly two MPI ranks.";

  const auto serial_mesh = makeSerialQuadStrip();
  const auto serial = runExtension(*serial_mesh,
                                   svmp::MeshComm(MPI_COMM_SELF));
  ASSERT_EQ(serial.report.vertices_outside_band, 0u);

  const auto partitioned_mesh = makePartitionedQuadStrip();
  ASSERT_GT(partitioned_mesh->n_ghost_vertices(), 0u);
  const auto partitioned_input = makeExtensionInputs(*partitioned_mesh);
  const auto local_active_count = static_cast<int>(std::count(
      partitioned_input.active.begin(), partitioned_input.active.end(), 1u));
  int rank_one_active_count = 0;
  if (rank == 1) {
    rank_one_active_count = local_active_count;
  }
  MPI_Bcast(&rank_one_active_count, 1, MPI_INT, 1, MPI_COMM_WORLD);
  ASSERT_EQ(rank_one_active_count, 0)
      << "Rank 1 must start beyond both the interface and its one-cell ghost "
         "halo so graph propagation has to cross the partition.";

  const auto partitioned = runExtension(*partitioned_mesh,
                                        svmp::MeshComm(MPI_COMM_WORLD));
  EXPECT_EQ(partitioned.report.vertices_outside_band, 0u);
  EXPECT_NEAR(partitioned.report.max_wall_normal_velocity, 0.0, 1.0e-12);

  // The algebraic map is owner-row/ghost-column: every global extension DOF
  // has exactly one row, but a dry row on rank 1 must be allowed to depend on
  // the rank-0-owned ghost layer that carried the extension across the
  // partition.  Reconstructing every emitted equation also verifies that the
  // saved coefficients are the exact frozen map used to produce the values.
  const auto local_row_count =
      static_cast<unsigned long long>(partitioned.rows.size());
  unsigned long long global_row_count = 0u;
  MPI_Allreduce(&local_row_count,
                &global_row_count,
                1,
                MPI_UNSIGNED_LONG_LONG,
                MPI_SUM,
                MPI_COMM_WORLD);
  EXPECT_EQ(global_row_count,
            static_cast<unsigned long long>(serial_mesh->n_vertices() *
                                            kComponents));
  bool has_off_rank_extension_dependency = false;
  for (const auto& row : partitioned.rows) {
    ASSERT_GE(row.vertex, 0);
    ASSERT_LT(static_cast<std::size_t>(row.vertex),
              partitioned_mesh->n_vertices());
    EXPECT_EQ(partitioned_mesh->owner_rank_vertex(
                  static_cast<svmp::index_t>(row.vertex)),
              rank)
        << "Only the owner may emit an algebraic extension row.";
    ASSERT_GE(row.component, 0);
    ASSERT_LT(row.component, static_cast<int>(kComponents));
    double reconstructed = 0.0;
    for (const auto& dependency : row.dependencies) {
      ASSERT_GE(dependency.vertex, 0);
      ASSERT_LT(static_cast<std::size_t>(dependency.vertex),
                partitioned_mesh->n_vertices());
      ASSERT_GE(dependency.component, 0);
      ASSERT_LT(dependency.component, static_cast<int>(kComponents));
      const auto dependency_vertex =
          static_cast<std::size_t>(dependency.vertex);
      const auto dependency_component =
          static_cast<std::size_t>(dependency.component);
      if (dependency.field ==
          svmp::FE::level_set::VelocityExtensionDependencyField::
              SourceVelocity) {
        reconstructed +=
            dependency.coefficient *
            partitioned_input.source[kComponents * dependency_vertex +
                                     dependency_component];
      } else {
        reconstructed +=
            dependency.coefficient *
            partitioned.values[kComponents * dependency_vertex +
                               dependency_component];
        EXPECT_LT(partitioned_input.phi[dependency_vertex],
                  partitioned_input.phi[static_cast<std::size_t>(row.vertex)])
            << "Frozen BFS dependencies must point to a strictly earlier "
               "normal-extension layer.";
        has_off_rank_extension_dependency =
            has_off_rank_extension_dependency ||
            partitioned_mesh->owner_rank_vertex(
                static_cast<svmp::index_t>(dependency_vertex)) != rank;
      }
    }
    EXPECT_NEAR(
        partitioned.values[kComponents * static_cast<std::size_t>(row.vertex) +
                           static_cast<std::size_t>(row.component)],
        reconstructed,
        1.0e-11)
        << "rank=" << rank << " local_vertex=" << row.vertex
        << " component=" << row.component;
  }
  int local_off_rank_dependency =
      has_off_rank_extension_dependency ? 1 : 0;
  int global_off_rank_dependencies = 0;
  MPI_Allreduce(&local_off_rank_dependency,
                &global_off_rank_dependencies,
                1,
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD);
  EXPECT_GT(global_off_rank_dependencies, 0)
      << "The fixture must exercise an owner row with an off-rank ghost "
         "extension dependency.";

  std::unordered_map<svmp::gid_t, std::size_t> serial_vertex_by_gid;
  const auto& serial_gids = serial_mesh->vertex_gids();
  for (std::size_t vertex = 0; vertex < serial_gids.size(); ++vertex) {
    serial_vertex_by_gid.emplace(serial_gids[vertex], vertex);
  }

  const auto& local_gids = partitioned_mesh->vertex_gids();
  const auto& local_coordinates = partitioned_mesh->X_ref();
  bool checked_far_owned_wall_vertex = false;
  for (std::size_t vertex = 0; vertex < local_gids.size(); ++vertex) {
    const auto serial_it = serial_vertex_by_gid.find(local_gids[vertex]);
    ASSERT_NE(serial_it, serial_vertex_by_gid.end());
    const auto serial_vertex = serial_it->second;
    for (std::size_t component = 0; component < kComponents; ++component) {
      EXPECT_NEAR(partitioned.values[kComponents * vertex + component],
                  serial.values[kComponents * serial_vertex + component],
                  1.0e-11)
          << "rank=" << rank << " gid=" << local_gids[vertex]
          << " component=" << component;
    }

    const double x = static_cast<double>(local_coordinates[2u * vertex]);
    if (partitioned_input.active[vertex] == 0u) {
      EXPECT_NEAR(partitioned.values[kComponents * vertex + 1u],
                  0.0,
                  1.0e-12)
          << "dry wall vertex gid=" << local_gids[vertex];
    }
    if (rank == 1 &&
        partitioned_mesh->owner_rank_vertex(
            static_cast<svmp::index_t>(vertex)) == rank &&
        std::abs(x - static_cast<double>(kCellCount - 1)) <= 1.0e-12) {
      EXPECT_GT(std::abs(partitioned.values[kComponents * vertex]), 1.0)
          << "The far owned dry-wall value must arrive through graph layers, "
             "not remain at its zero initialization.";
      checked_far_owned_wall_vertex = true;
    }
  }

  int local_far_check = checked_far_owned_wall_vertex ? 1 : 0;
  int global_far_checks = 0;
  MPI_Allreduce(&local_far_check,
                &global_far_checks,
                1,
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD);
  EXPECT_GT(global_far_checks, 0);

  // Repeat the partition-vs-MPI_COMM_SELF comparison with the actual
  // all-component mask of a strong homogeneous no-slip condition.  Every dry
  // horizontal-wall vertex must lose both its normal and tangential extension.
  const auto no_slip_mask = std::array<bool, 3>{true, true, true};
  const auto serial_no_slip = runExtension(
      *serial_mesh, svmp::MeshComm(MPI_COMM_SELF), no_slip_mask);
  const auto partitioned_no_slip = runExtension(
      *partitioned_mesh, svmp::MeshComm(MPI_COMM_WORLD), no_slip_mask);
  EXPECT_EQ(partitioned_no_slip.report.vertices_outside_band, 0u);
  EXPECT_NEAR(partitioned_no_slip.report.max_wall_normal_velocity,
              0.0,
              1.0e-12);
  for (std::size_t vertex = 0; vertex < local_gids.size(); ++vertex) {
    const auto serial_vertex = serial_vertex_by_gid.at(local_gids[vertex]);
    for (std::size_t component = 0; component < kComponents; ++component) {
      EXPECT_NEAR(
          partitioned_no_slip.values[kComponents * vertex + component],
          serial_no_slip.values[kComponents * serial_vertex + component],
          1.0e-11)
          << "no-slip partition mismatch rank=" << rank
          << " gid=" << local_gids[vertex]
          << " component=" << component;
      if (partitioned_input.active[vertex] == 0u) {
        EXPECT_NEAR(
            partitioned_no_slip.values[kComponents * vertex + component],
            0.0,
            1.0e-12)
            << "no-slip dry wall value rank=" << rank
            << " gid=" << local_gids[vertex]
            << " component=" << component;
      }
    }
  }
}

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     AlgebraicWetExtensionSolvesOffRankWallProjectedBandWithFsils)
{
#if !defined(FE_HAS_FSILS)
  GTEST_SKIP() << "FSILS is required for the production-backend regression.";
#else
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  ASSERT_EQ(size, 2)
      << "This distributed wet-extension solve requires exactly two ranks.";

  const auto mesh = makePartitionedQuadStrip();
  ASSERT_GT(mesh->n_ghost_vertices(), 0u);
  const auto extension_input = makeExtensionInputs(*mesh);
  auto frozen_extension = runExtension(*mesh, svmp::MeshComm(MPI_COMM_WORLD));
  EXPECT_EQ(frozen_extension.report.vertices_outside_band, 0u);
  EXPECT_NEAR(frozen_extension.report.max_wall_normal_velocity,
              0.0,
              1.0e-12);

  // Establish that this exact frozen map is not merely a one-layer local
  // stencil.  Starting from an owned row near x=7 on rank 1, follow its
  // extension-to-extension dependencies back to the rank-0-owned ghost
  // layer.  This is the same owner-row/ghost-column map installed below.
  using ConstraintRow =
      svmp::FE::level_set::VelocityExtensionConstraintRow;
  const auto row_key = [](svmp::FE::GlobalIndex vertex, int component) {
    return std::make_pair(vertex, component);
  };
  std::map<std::pair<svmp::FE::GlobalIndex, int>, const ConstraintRow*>
      owned_rows;
  for (const auto& row : frozen_extension.rows) {
    ASSERT_GE(row.vertex, 0);
    ASSERT_LT(static_cast<std::size_t>(row.vertex), mesh->n_vertices());
    EXPECT_EQ(mesh->owner_rank_vertex(
                  static_cast<svmp::index_t>(row.vertex)),
              rank);
    owned_rows.emplace(row_key(row.vertex, row.component), &row);
  }

  std::map<std::pair<svmp::FE::GlobalIndex, int>, int> depth_cache;
  std::set<std::pair<svmp::FE::GlobalIndex, int>> depth_stack;
  std::function<int(svmp::FE::GlobalIndex, int)> depth_to_off_rank;
  depth_to_off_rank = [&](svmp::FE::GlobalIndex vertex,
                          int component) -> int {
    const auto key = row_key(vertex, component);
    if (const auto found = depth_cache.find(key);
        found != depth_cache.end()) {
      return found->second;
    }
    if (!depth_stack.insert(key).second) {
      ADD_FAILURE() << "Frozen wet-extension dependency graph contains a cycle.";
      return -1;
    }
    int best = -1;
    const auto row = owned_rows.find(key);
    if (row != owned_rows.end()) {
      for (const auto& dependency : row->second->dependencies) {
        if (dependency.field !=
            svmp::FE::level_set::VelocityExtensionDependencyField::
                ExtensionVelocity) {
          continue;
        }
        const auto dependency_owner = mesh->owner_rank_vertex(
            static_cast<svmp::index_t>(dependency.vertex));
        if (dependency_owner != rank) {
          best = std::max(best, 1);
          continue;
        }
        const int child_depth =
            depth_to_off_rank(dependency.vertex, dependency.component);
        if (child_depth >= 0) {
          best = std::max(best, child_depth + 1);
        }
      }
    }
    depth_stack.erase(key);
    depth_cache.emplace(key, best);
    return best;
  };

  int local_dependency_depth = -1;
  const auto& coordinates = mesh->X_ref();
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    if (mesh->owner_rank_vertex(static_cast<svmp::index_t>(vertex)) != rank) {
      continue;
    }
    const double x = static_cast<double>(coordinates[2u * vertex]);
    if (std::abs(x - static_cast<double>(kCellCount - 1)) <= 1.0e-12) {
      local_dependency_depth = std::max(
          local_dependency_depth,
          depth_to_off_rank(static_cast<svmp::FE::GlobalIndex>(vertex), 0));
    }
  }
  int global_dependency_depth = -1;
  MPI_Allreduce(&local_dependency_depth,
                &global_dependency_depth,
                1,
                MPI_INT,
                MPI_MAX,
                MPI_COMM_WORLD);
  EXPECT_GE(global_dependency_depth, 3)
      << "The solve fixture must traverse at least three frozen graph edges "
         "before reaching its off-rank dependency.";

  const auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/1);
  const auto vector_space = svmp::FE::spaces::VectorSpace(
      svmp::FE::spaces::SpaceType::H1,
      svmp::FE::ElementType::Quad4,
      /*order=*/1,
      /*components=*/static_cast<int>(kComponents));
  ASSERT_TRUE(vector_space);

  svmp::FE::systems::FESystem system(mesh);
  const auto physical_velocity = system.addField(
      svmp::FE::systems::FieldSpec{
          .name = "Velocity",
          .space = vector_space,
          .components = static_cast<int>(kComponents)});
  const auto phi = system.addField(svmp::FE::systems::FieldSpec{
      .name = "phi", .space = scalar_space, .components = 1});
  system.addOperator("level_set");

  // A transient mass equation holds the physical velocity at its previous
  // value.  The remaining two blocks are the production E=P(phi)u global
  // kernel and the production implicit phi transport kernel.
  using namespace svmp::FE::forms;
  const auto source_state =
      StateField(physical_velocity, *vector_space, "Velocity");
  const auto source_test =
      TestField(physical_velocity, *vector_space, "Velocity_test");
  (void)svmp::FE::systems::installFormulation(
      system,
      "level_set",
      {physical_velocity},
      inner(source_state.dt(1), source_test).dx());

  svmp::FE::level_set::LevelSetTransportOptions transport{};
  transport.operator_tag = "level_set";
  transport.level_set.field_name = "phi";
  transport.level_set.auto_register_field = false;
  transport.velocity.field_name = "LevelSetAdvectionVelocity";
  transport.velocity.source =
      svmp::FE::level_set::LevelSetVelocitySource::CoupledField;
  transport.velocity.auto_register_field = true;
  transport.velocity.space = vector_space;
  transport.velocity.algebraic_extension_source_field_name = "Velocity";
  transport.supg.enabled = false;
  (void)svmp::FE::level_set::installLevelSetTransport(
      system, scalar_space, transport);

  const auto extension_velocity =
      system.findFieldByName("LevelSetAdvectionVelocity");
  ASSERT_NE(extension_velocity, svmp::FE::INVALID_FIELD_ID);
  const auto extension_kernel =
      svmp::FE::level_set::findLevelSetVelocityExtensionConstraintKernel(
          system, "level_set", extension_velocity);
  ASSERT_TRUE(extension_kernel);
  extension_kernel->setFrozenRows(frozen_extension.rows, 1u);

  svmp::FE::systems::SetupOptions setup_options;
  setup_options.assembler_name = "StandardAssembler";
  setup_options.assembly_options.ghost_policy =
      svmp::FE::assembly::GhostPolicy::ReverseScatter;
  setup_options.assembly_options.deterministic = true;
  setup_options.assembly_options.overlap_communication = false;
  setup_options.dof_options.global_numbering =
      svmp::FE::dofs::GlobalNumberingMode::OwnerContiguous;
  setup_options.dof_options.ownership =
      svmp::FE::dofs::OwnershipStrategy::LowestRank;
  setup_options.dof_options.my_rank = rank;
  setup_options.dof_options.world_size = size;
  setup_options.dof_options.mpi_comm = MPI_COMM_WORLD;
  setup_options.use_backend_row_ownership_for_assembly = true;
  setup_options.retain_serial_sparsity = false;
  ASSERT_NO_THROW(system.setup(setup_options));
  ASSERT_TRUE(system.isSetup());

  const auto* distributed_pattern =
      system.distributedSparsityIfAvailable("level_set");
  ASSERT_NE(distributed_pattern, nullptr)
      << "The regression must use owned-row distributed sparsity.";
  const auto n_dofs = system.dofHandler().getNumDofs();
  EXPECT_EQ(distributed_pattern->globalRows(), n_dofs);
  EXPECT_EQ(distributed_pattern->globalCols(), n_dofs);
  ASSERT_TRUE(system.dofPermutation());

  constexpr int dof_per_node =
      static_cast<int>(2u * kComponents + 1u);
  svmp::FE::backends::FsilsFactory factory(
      dof_per_node, system.dofPermutation(), MPI_COMM_WORLD);
  EXPECT_EQ(factory.backendKind(),
            svmp::FE::backends::BackendKind::FSILS);
  svmp::FE::backends::SolverOptions linear_options;
  linear_options.method = svmp::FE::backends::SolverMethod::GMRES;
  linear_options.preconditioner =
      svmp::FE::backends::PreconditionerType::Diagonal;
  linear_options.rel_tol = 1.0e-12;
  linear_options.abs_tol = 1.0e-13;
  linear_options.max_iter = 500;
  auto linear_solver = factory.createLinearSolver(linear_options);
  ASSERT_TRUE(linear_solver);

  constexpr svmp::FE::Real dt = 0.05;
  auto history = svmp::FE::timestepping::TimeHistory::allocate(
      factory,
      n_dofs,
      /*history_depth=*/2,
      /*allocate_second_order_state=*/false);
  history.setTime(0.0);
  history.setDt(dt);
  history.setPrevDt(dt);
  history.setStepIndex(0);

  auto integrator =
      std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
  svmp::FE::systems::TransientSystem transient(system, integrator);
  svmp::FE::timestepping::NewtonOptions newton_options;
  newton_options.residual_op = "level_set";
  newton_options.jacobian_op = "level_set";
  newton_options.max_iterations = 6;
  newton_options.abs_tolerance = 1.0e-10;
  newton_options.rel_tolerance = 1.0e-12;
  newton_options.step_tolerance = 0.0;
  newton_options.stagnation_tolerance = 0.0;
  newton_options.use_line_search = false;
  svmp::FE::timestepping::NewtonSolver newton(newton_options);
  svmp::FE::timestepping::NewtonWorkspace workspace;
  ASSERT_NO_THROW(newton.allocateWorkspace(system, factory, workspace));
  ASSERT_NE(dynamic_cast<svmp::FE::backends::FsilsMatrix*>(
                workspace.jacobian.get()),
            nullptr);
  history.repack(factory);

  const auto global_dof = [&](svmp::FE::FieldId field,
                              std::size_t vertex,
                              int component) {
    const auto* entity_map =
        system.fieldDofHandler(field).getEntityDofMap();
    if (entity_map == nullptr) {
      throw std::runtime_error("wet-extension solve field has no vertex map");
    }
    const auto dofs = entity_map->getVertexDofs(
        static_cast<svmp::FE::GlobalIndex>(vertex));
    if (component < 0 ||
        static_cast<std::size_t>(component) >= dofs.size()) {
      throw std::runtime_error("wet-extension solve component is out of range");
    }
    return system.fieldDofOffset(field) +
           dofs[static_cast<std::size_t>(component)];
  };

  // u=(2,5-y/2) has a deliberately nonzero wall-normal component.  The
  // frozen dry-side wall projection must remove E_y while retaining E_x=2.
  // phi_n=x-0.25 then has the exact implicit update phi_{n+1}=phi_n-2*dt.
  std::vector<svmp::FE::Real> previous(
      static_cast<std::size_t>(n_dofs), 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto x = static_cast<svmp::FE::Real>(coordinates[2u * vertex]);
    const auto y =
        static_cast<svmp::FE::Real>(coordinates[2u * vertex + 1u]);
    previous[static_cast<std::size_t>(
        global_dof(physical_velocity, vertex, 0))] = 2.0;
    previous[static_cast<std::size_t>(
        global_dof(physical_velocity, vertex, 1))] = 5.0 - 0.5 * y;
    previous[static_cast<std::size_t>(global_dof(phi, vertex, 0))] =
        x - 0.25;
  }
  scatterFeOrderedSolution(history.uPrev(), previous);
  scatterFeOrderedSolution(history.uPrev2(), previous);
  history.resetCurrentToPrevious();

  // Inspect the production FSILS matrix before the Krylov solver applies its
  // in-place scaling/preconditioner transformations.  This independently
  // proves that the implicit transport Jacobian contains the E-to-phi block,
  // rather than inferring that coupling only from the converged state.
  svmp::FE::systems::SystemStateView initial_state;
  initial_state.time = dt;
  initial_state.dt = dt;
  initial_state.dt_prev = dt;
  initial_state.u = history.uSpan();
  initial_state.u_prev = history.uPrevSpan();
  initial_state.u_prev2 = history.uPrev2Span();
  initial_state.u_vector = &history.u();
  initial_state.u_prev_vector = &history.uPrev();
  initial_state.u_prev2_vector = &history.uPrev2();
  const auto initial_time_context =
      integrator->buildContext(/*max_time_derivative_order=*/1,
                               initial_state);
  initial_state.time_integration = &initial_time_context;
  svmp::FE::systems::AssemblyRequest initial_request;
  initial_request.op = "level_set";
  initial_request.want_matrix = true;
  initial_request.want_vector = true;
  auto initial_matrix_view = workspace.jacobian->createAssemblyView();
  auto initial_residual_view = workspace.residual->createAssemblyView();
  ASSERT_TRUE(initial_matrix_view);
  ASSERT_TRUE(initial_residual_view);
  const auto initial_assembly = transient.assemble(
      initial_request,
      initial_state,
      initial_matrix_view.get(),
      initial_residual_view.get());
  ASSERT_TRUE(initial_assembly.success) << initial_assembly.error_message;

  double local_phi_extension_tangent = 0.0;
  const auto& phi_dofs = system.fieldDofHandler(phi).getDofMap();
  const auto* phi_entity_map =
      system.fieldDofHandler(phi).getEntityDofMap();
  ASSERT_NE(phi_entity_map, nullptr);
  for (std::size_t row_vertex = 0; row_vertex < mesh->n_vertices();
       ++row_vertex) {
    const auto local_row = phi_entity_map->getVertexDofs(
        static_cast<svmp::FE::GlobalIndex>(row_vertex));
    ASSERT_EQ(local_row.size(), 1u);
    if (!phi_dofs.isOwnedDof(local_row.front())) {
      continue;
    }
    const auto row_dof = global_dof(phi, row_vertex, 0);
    for (std::size_t column_vertex = 0;
         column_vertex < mesh->n_vertices();
         ++column_vertex) {
      const auto column_dof =
          global_dof(extension_velocity, column_vertex, 0);
      local_phi_extension_tangent = std::max(
          local_phi_extension_tangent,
          std::abs(static_cast<double>(
              workspace.jacobian->getEntry(row_dof, column_dof))));
    }
  }
  double global_phi_extension_tangent = 0.0;
  MPI_Allreduce(&local_phi_extension_tangent,
                &global_phi_extension_tangent,
                1,
                MPI_DOUBLE,
                MPI_MAX,
                MPI_COMM_WORLD);
  EXPECT_GT(global_phi_extension_tangent, 1.0e-6)
      << "The assembled transport Jacobian must carry E to phi.";

  svmp::FE::timestepping::NewtonReport solve_report{};
  ASSERT_NO_THROW(solve_report = newton.solveStep(
                      transient,
                      *linear_solver,
                      /*solve_time=*/dt,
                      history,
                      workspace));
  EXPECT_TRUE(solve_report.converged);
  EXPECT_GT(solve_report.iterations, 0);
  EXPECT_LE(solve_report.iterations, newton_options.max_iterations);
  EXPECT_TRUE(solve_report.linear.converged);
  EXPECT_GT(solve_report.linear.collective_calls, 0u);
  EXPECT_GT(solve_report.residual_norm0, 1.0e-6);
  EXPECT_LE(solve_report.residual_norm, newton_options.abs_tolerance);
  EXPECT_NEAR(workspace.residual->norm(),
              solve_report.residual_norm,
              1.0e-12);

  history.u().updateGhosts();
  const auto solved = gatherFeOrderedSolution(history.u());
  std::size_t local_dry_wall_vertices = 0u;
  bool local_far_band_value_checked = false;
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    if (mesh->owner_rank_vertex(static_cast<svmp::index_t>(vertex)) != rank) {
      continue;
    }
    const auto x = static_cast<svmp::FE::Real>(coordinates[2u * vertex]);
    const auto y =
        static_cast<svmp::FE::Real>(coordinates[2u * vertex + 1u]);
    EXPECT_NEAR(
        solved[static_cast<std::size_t>(
            global_dof(physical_velocity, vertex, 0))],
        2.0,
        2.0e-10);
    EXPECT_NEAR(
        solved[static_cast<std::size_t>(
            global_dof(physical_velocity, vertex, 1))],
        5.0 - 0.5 * y,
        2.0e-10);
    EXPECT_NEAR(
        solved[static_cast<std::size_t>(
            global_dof(extension_velocity, vertex, 0))],
        2.0,
        2.0e-10);
    EXPECT_NEAR(solved[static_cast<std::size_t>(global_dof(phi, vertex, 0))],
                x - 0.25 - 2.0 * dt,
                3.0e-10)
        << "rank=" << rank << " vertex=" << vertex;

    if (extension_input.active[vertex] == 0u) {
      ++local_dry_wall_vertices;
      EXPECT_GT(5.0 - 0.5 * y, 4.0);
      EXPECT_NEAR(
          solved[static_cast<std::size_t>(
              global_dof(extension_velocity, vertex, 1))],
          0.0,
          2.0e-10)
          << "The dry wall projection must remove normal extension velocity.";
    }
    if (rank == 1 &&
        std::abs(x - static_cast<svmp::FE::Real>(kCellCount - 1)) <=
            1.0e-12) {
      EXPECT_NEAR(
          solved[static_cast<std::size_t>(
              global_dof(extension_velocity, vertex, 0))],
          2.0,
          2.0e-10);
      local_far_band_value_checked = true;
    }
  }
  unsigned long long local_dry =
      static_cast<unsigned long long>(local_dry_wall_vertices);
  unsigned long long global_dry = 0u;
  MPI_Allreduce(&local_dry,
                &global_dry,
                1,
                MPI_UNSIGNED_LONG_LONG,
                MPI_SUM,
                MPI_COMM_WORLD);
  EXPECT_GT(global_dry, 0u);
  int local_far = local_far_band_value_checked ? 1 : 0;
  int global_far = 0;
  MPI_Allreduce(&local_far,
                &global_far,
                1,
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD);
  EXPECT_GT(global_far, 0);

  // Inspect the matrix left by the final Newton residual/Jacobian assembly.
  // At least one extension row must retain its off-rank ghost column with the
  // exact frozen coefficient, and the transport block must contain a genuine
  // E-to-phi tangent (not a prescribed-data substitution).
  bool local_assembled_off_rank_dependency = false;
  for (const auto& row : frozen_extension.rows) {
    const auto row_dof = global_dof(
        extension_velocity,
        static_cast<std::size_t>(row.vertex),
        row.component);
    for (const auto& dependency : row.dependencies) {
      if (dependency.field !=
              svmp::FE::level_set::VelocityExtensionDependencyField::
                  ExtensionVelocity ||
          mesh->owner_rank_vertex(
              static_cast<svmp::index_t>(dependency.vertex)) == rank) {
        continue;
      }
      const auto column_dof = global_dof(
          extension_velocity,
          static_cast<std::size_t>(dependency.vertex),
          dependency.component);
      EXPECT_NEAR(workspace.jacobian->getEntry(row_dof, column_dof),
                  -dependency.coefficient,
                  2.0e-12);
      local_assembled_off_rank_dependency = true;
    }
  }
  int local_assembled_off_rank =
      local_assembled_off_rank_dependency ? 1 : 0;
  int global_assembled_off_rank = 0;
  MPI_Allreduce(&local_assembled_off_rank,
                &global_assembled_off_rank,
                1,
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD);
  EXPECT_GT(global_assembled_off_rank, 0)
      << "FSILS must retain the extension owner's off-rank ghost column.";
#endif
}

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     CollidingComponentBandsMatchSelfAcrossPartitionWithoutBlending)
{
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  ASSERT_EQ(size, 2)
      << "This component-collision test requires exactly two MPI ranks.";

  const auto serial_mesh = makeSerialQuadStrip();
  const auto serial = runTwoComponentExtension(
      *serial_mesh, svmp::MeshComm(MPI_COMM_SELF));
  ASSERT_EQ(serial.report.vertices_outside_band, 0u);
  ASSERT_EQ(serial.report.component_collision_vertices, 2u);

  const auto partitioned_mesh = makePartitionedQuadStrip();
  ASSERT_GT(partitioned_mesh->n_ghost_vertices(), 0u);
  const auto partitioned = runTwoComponentExtension(
      *partitioned_mesh, svmp::MeshComm(MPI_COMM_WORLD));
  EXPECT_EQ(partitioned.report.vertices_outside_band, 0u);

  const auto local_collisions = static_cast<unsigned long long>(
      partitioned.report.component_collision_vertices);
  unsigned long long global_collisions = 0u;
  MPI_Allreduce(&local_collisions,
                &global_collisions,
                1,
                MPI_UNSIGNED_LONG_LONG,
                MPI_SUM,
                MPI_COMM_WORLD);
  EXPECT_EQ(global_collisions, 2u);

  std::unordered_map<svmp::gid_t, std::size_t> serial_vertex_by_gid;
  const auto& serial_gids = serial_mesh->vertex_gids();
  for (std::size_t vertex = 0; vertex < serial_gids.size(); ++vertex) {
    serial_vertex_by_gid.emplace(serial_gids[vertex], vertex);
  }

  const auto& local_gids = partitioned_mesh->vertex_gids();
  const auto& local_coordinates = partitioned_mesh->X_ref();
  for (std::size_t vertex = 0; vertex < local_gids.size(); ++vertex) {
    const auto serial_it = serial_vertex_by_gid.find(local_gids[vertex]);
    ASSERT_NE(serial_it, serial_vertex_by_gid.end());
    const auto serial_vertex = serial_it->second;
    for (std::size_t component = 0; component < kComponents; ++component) {
      EXPECT_NEAR(partitioned.values[kComponents * vertex + component],
                  serial.values[kComponents * serial_vertex + component],
                  1.0e-11)
          << "rank=" << rank << " gid=" << local_gids[vertex]
          << " component=" << component;
    }

    const double x = static_cast<double>(local_coordinates[2u * vertex]);
    const double y = static_cast<double>(local_coordinates[2u * vertex + 1u]);
    // Component selection is geometric: the reconstructed interfaces are at
    // x=0.25 and x=kCellCount-0.35, so the latter owns the x=4 collision
    // vertices.  Numeric component labels must not break this decision.
    constexpr double left_interface = 0.25;
    const double right_interface =
        static_cast<double>(kCellCount) - 0.35;
    const bool left_branch =
        x <= 0.5 * (left_interface + right_interface);
    EXPECT_NEAR(partitioned.values[kComponents * vertex],
                left_branch ? 2.0 + 3.0 * y : -7.0 + 2.0 * y,
                1.0e-11)
        << "rank=" << rank << " gid=" << local_gids[vertex];
    EXPECT_NEAR(partitioned.values[kComponents * vertex + 1u],
                left_branch ? -1.0 + 0.5 * y : 11.0 - 4.0 * y,
                1.0e-11)
        << "rank=" << rank << " gid=" << local_gids[vertex];
  }
}

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     AcceptedSnapshotPrescribedFrameIsPartitionInvariantAndConflictsFailClosed)
{
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  ASSERT_GE(size, 2)
      << "This partition-invariance test requires at least two MPI ranks.";

  constexpr int interface_marker = 1721;
  constexpr int wall_marker = 151;
  constexpr std::uint64_t revision = 73u;
  constexpr svmp::FE::GlobalIndex parent = 23;
  const auto pi = std::acos(svmp::FE::Real{-1.0});
  const auto target_angle = pi / svmp::FE::Real{3.0};

  svmp::FE::interfaces::FreeSurfaceGeometryRuleRecord record;
  record.role =
      svmp::FE::interfaces::FreeSurfaceGeometryRuleRole::Contact;
  record.retention =
      svmp::FE::interfaces::FreeSurfaceGeometryRetention::Retained;
  record.physical_boundary_marker = wall_marker;
  record.locally_owned = true;
  record.reference_rule.geometric_dimension = 0;
  record.reference_rule.provenance.parent_entity_global_id = parent;
  record.reference_rule.provenance.free_surface_snapshot_revision_key =
      revision;
  record.reference_rule.points.resize(1u);
  record.physical_rule.geometric_dimension = 0;
  record.physical_rule.free_surface_snapshot_revision_key = revision;
  record.physical_rule.physical_measure = 1.0;
  record.physical_rule.points.push_back(
      svmp::FE::geometry::MappedCutQuadraturePoint{
          .physical_point = {{0.5, 0.0, 0.0}},
          .physical_weight = 1.0,
          .normal = {{std::sin(target_angle),
                      std::cos(target_angle),
                      0.0}},
          .boundary_normal = {{0.0, -1.0, 0.0}},
      });
  svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalParameters parameters;
  parameters.liquid_side =
      svmp::FE::geometry::CutIntegrationSide::Negative;
  parameters.young_wall_coefficients.push_back(
      svmp::FE::interfaces::FreeSurfaceYoungWallCoefficient{
          .boundary_marker = wall_marker,
          .equilibrium_contact_angle_radians = target_angle,
      });
  const auto constraint = makeAcceptedSnapshotWallConstraint(
      record,
      svmp::FE::level_set::LevelSetWallContactConstraintKind::
          PrescribedAngle,
      interface_marker,
      revision,
      parameters,
      /*dimension=*/2);
  EXPECT_NEAR(constraint.target_angle_radians, target_angle, 1.0e-15);
  EXPECT_EQ(constraint.accepted_contact_line_tangent,
            (std::array<svmp::FE::Real, 3>{{0.0, 0.0, 1.0}}));

  const auto canonical = canonicalizeAcceptedWallConstraints(
      std::vector{constraint, constraint},
      svmp::MeshComm(MPI_COMM_WORLD),
      "MPI prescribed-frame duplicate test");
  ASSERT_EQ(canonical.size(), 1u);
  std::array<double, 8> payload{{
      canonical.front().target_angle_radians,
      canonical.front().physical_wall_normal[0],
      canonical.front().physical_wall_normal[1],
      canonical.front().physical_wall_normal[2],
      canonical.front().accepted_contact_point[0],
      canonical.front().accepted_contact_line_tangent[0],
      canonical.front().accepted_contact_line_tangent[1],
      canonical.front().accepted_contact_line_tangent[2],
  }};
  double local_max_difference = 0.0;
  for (const auto value : payload) {
    double minimum = 0.0;
    double maximum = 0.0;
    MPI_Allreduce(&value,
                  &minimum,
                  1,
                  MPI_DOUBLE,
                  MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&value,
                  &maximum,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  MPI_COMM_WORLD);
    local_max_difference =
        std::max(local_max_difference, std::abs(maximum - minimum));
  }
  EXPECT_DOUBLE_EQ(local_max_difference, 0.0);
  if (rank == 0) {
    ::testing::Test::RecordProperty(
        "application_prescribed_frame_mpi_max_difference",
        local_max_difference);
  }

  auto conflict = constraint;
  conflict.accepted_contact_point[0] += svmp::FE::Real{0.25};
  std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
      rank_local_constraints{constraint};
  if (rank == 0) {
    rank_local_constraints.push_back(conflict);
  }
  EXPECT_THROW(
      (void)canonicalizeAcceptedWallConstraints(
          std::move(rank_local_constraints),
          svmp::MeshComm(MPI_COMM_WORLD),
          "MPI prescribed-frame conflict test"),
      std::runtime_error);

  auto positive_parameters = parameters;
  positive_parameters.liquid_side =
      svmp::FE::geometry::CutIntegrationSide::Positive;
  const auto positive_constraint = makeAcceptedSnapshotWallConstraint(
      record,
      svmp::FE::level_set::LevelSetWallContactConstraintKind::
          PrescribedAngle,
      interface_marker,
      revision,
      positive_parameters,
      /*dimension=*/2);
  EXPECT_NEAR(positive_constraint.target_angle_radians,
              pi - target_angle,
              1.0e-15);
}

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     ActiveCutRefreshUsesCommunicatorWideSortedBoundaryMarkerUnion)
{
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  ASSERT_EQ(size, 2)
      << "This boundary-marker union test requires exactly two MPI ranks.";

  auto mesh = makePartitionedQuadStripWithRankDisjointWallMarkers();
  ASSERT_GT(mesh->n_ghost_vertices(), 0u);

  const auto mesh_field = svmp::MeshFields::attach_field(
      mesh->local_mesh(),
      svmp::EntityKind::Vertex,
      "phi",
      svmp::FieldScalarType::Float64,
      1);
  ASSERT_NE(svmp::MeshFields::field_data_as<svmp::real_t>(
                mesh->local_mesh(), mesh_field),
            nullptr);

  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/1);
  auto velocity_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, 2);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi", .space = scalar_space, .components = 1});
  const auto velocity = system->addField(svmp::FE::systems::FieldSpec{
      .name = "velocity", .space = velocity_space, .components = 2});
  svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalParameters
      functional_parameters;
  functional_parameters.liquid_side =
      svmp::FE::geometry::CutIntegrationSide::Negative;
  functional_parameters.surface_tension = svmp::FE::Real{0.75};
  const svmp::FE::Real equilibrium_angle =
      std::acos(svmp::FE::Real{2.0} / svmp::FE::Real{3.0});
  for (const int marker : {static_cast<int>(kLeftOnlyWall),
                           static_cast<int>(kRightOnlyWall)}) {
    functional_parameters.young_wall_coefficients.push_back(
        svmp::FE::interfaces::FreeSurfaceYoungWallCoefficient{
            .boundary_marker = marker,
            .equilibrium_contact_angle_radians = equilibrium_angle,
        });
    functional_parameters.dynamic_contact_coefficients.push_back(
        svmp::FE::interfaces::FreeSurfaceDynamicContactCoefficient{
            .boundary_marker = marker,
            .equilibrium_contact_angle_radians = equilibrium_angle,
            .mobility = svmp::FE::Real{0.5},
            .slip_length = svmp::FE::Real{0.2},
            .dynamic_viscosity = svmp::FE::Real{0.4},
        });
  }
  system->declareFreeSurfaceDiscreteFunctional(
      svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration{
          .interface_marker = 721,
          .level_set_field = phi,
          .velocity_field = velocity,
          .geometry_domain_id = "mpi_disjoint_wall_interface",
          .parameters = functional_parameters,
          .owner_component =
              "ApplicationDriverLevelSetWorkflowsMPI.FunctionalFixture",
      });

  svmp::FE::systems::SetupOptions setup_options;
  setup_options.dof_options.global_numbering =
      svmp::FE::dofs::GlobalNumberingMode::OwnerContiguous;
  setup_options.dof_options.ownership =
      svmp::FE::dofs::OwnershipStrategy::LowestRank;
  setup_options.dof_options.my_rank = rank;
  setup_options.dof_options.world_size = size;
  setup_options.dof_options.mpi_comm = MPI_COMM_WORLD;
  ASSERT_NO_THROW(system->setup(setup_options));

  const auto local_markers =
      svmp::FE::interfaces::boundaryMarkers(system->meshAccess());
  if (rank == 0) {
    EXPECT_EQ(local_markers,
              (std::vector<int>{kLeftOnlyWall, kLeftOnlyExtraWall}));
  } else {
    EXPECT_EQ(local_markers, (std::vector<int>{kRightOnlyWall}));
  }
  const auto global_markers = communicatorWideBoundaryMarkers(
      system->meshAccess(), activeFESystemCommunicator(*system));
  EXPECT_EQ(global_markers,
            (std::vector<int>{kLeftOnlyWall,
                              kLeftOnlyExtraWall,
                              kRightOnlyWall}));

  const auto& field_dofs = system->fieldDofHandler(phi);
  const auto* entity_map = field_dofs.getEntityDofMap();
  ASSERT_NE(entity_map, nullptr);
  const auto field_offset = system->fieldDofOffset(phi);
  ASSERT_GE(field_offset, 0);
  const auto solution_size =
      static_cast<std::size_t>(system->dofHandler().getNumDofs());
  std::vector<svmp::FE::Real> local_solution(solution_size, 0.0);
  std::vector<svmp::FE::Real> solution(solution_size, 0.0);
  const auto& coordinates = mesh->X_ref();
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto dofs = entity_map->getVertexDofs(
        static_cast<svmp::FE::GlobalIndex>(vertex));
    ASSERT_EQ(dofs.size(), 1u);
    const auto dof = dofs.front();
    ASSERT_GE(dof, 0);
    if (field_dofs.getDofMap().isOwnedDof(dof)) {
      const auto index = static_cast<std::size_t>(field_offset + dof);
      ASSERT_LT(index, local_solution.size());
      local_solution[index] =
          svmp::FE::Real{2.0} *
          (static_cast<svmp::FE::Real>(coordinates[2u * vertex + 1u]) -
           svmp::FE::Real{0.5});
    }
  }
  const auto& velocity_dofs = system->fieldDofHandler(velocity);
  const auto velocity_offset = system->fieldDofOffset(velocity);
  ASSERT_GE(velocity_offset, 0);
  for (svmp::FE::GlobalIndex cell = 0;
       cell < static_cast<svmp::FE::GlobalIndex>(mesh->n_cells());
       ++cell) {
    for (const auto dof : velocity_dofs.getCellDofs(cell)) {
      ASSERT_GE(dof, 0);
      if (!velocity_dofs.getDofMap().isOwnedDof(dof)) {
        continue;
      }
      const auto index = static_cast<std::size_t>(velocity_offset + dof);
      ASSERT_LT(index, local_solution.size());
      local_solution[index] = svmp::FE::Real{0.25};
    }
  }
  MPI_Allreduce(local_solution.data(),
                solution.data(),
                static_cast<int>(solution.size()),
                MPI_DOUBLE,
                MPI_SUM,
                MPI_COMM_WORLD);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);
  auto params = parseMpiWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>mpi_disjoint_wall_interface</Generated_interface_domain_id>
      <Interface_marker>721</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
  ActiveCutContextRefreshCache refresh_cache;
  ActiveCutContextRefreshReport report{};
  ASSERT_NO_THROW(
      report = refreshActiveCutIntegrationContextFromSolutionCached(
          sim,
          *params,
          std::span<const svmp::FE::Real>(solution.data(), solution.size()),
          lifecycle,
          refresh_cache,
          "application-driver-mpi-boundary-marker-union-test"));
  EXPECT_TRUE(report.refreshed);
  // The serial eight-cell strip has reference area 32 (the Quad4 parent area
  // is four) and physical area eight.  phi=y-0.5 bisects both exactly.  Ghost
  // cells must remain in each local cut context but must not enter these
  // communicator-global lifecycle/physical totals a second time.
  EXPECT_EQ(report.cell_count, static_cast<std::size_t>(kCellCount));
  EXPECT_EQ(report.interface_fragments,
            static_cast<std::size_t>(kCellCount));
  EXPECT_EQ(report.active_volume_regions,
            static_cast<std::size_t>(kCellCount));
  EXPECT_EQ(report.domain_interface_quadrature_point_count, 16u);
  EXPECT_EQ(report.domain_volume_quadrature_point_count, 96u);
  EXPECT_NEAR(report.negative_volume, 16.0, 1.0e-12);
  EXPECT_NEAR(report.positive_volume, 16.0, 1.0e-12);
  EXPECT_NEAR(report.negative_physical_volume, 4.0, 1.0e-12);
  EXPECT_NEAR(report.positive_physical_volume, 4.0, 1.0e-12);
  const auto current_functionals =
      evaluateCurrentFreeSurfaceDiscreteFunctionals(sim);
  const auto maintenance_functionals =
      levelSetMaintenanceFunctionalValues(sim, current_functionals);
  ASSERT_EQ(maintenance_functionals.size(), 1u);
  EXPECT_NE(
      maintenance_functionals.front().cut_topology_revision, 0u);
  const auto [minimum_cut_topology, maximum_cut_topology] =
      globalMinMaxUint64(
          maintenance_functionals.front().cut_topology_revision,
          svmp::MeshComm(MPI_COMM_WORLD));
  EXPECT_EQ(minimum_cut_topology, maximum_cut_topology);

  const auto active_cut_requests = activeCutVolumeRequests(*params);
  ASSERT_EQ(active_cut_requests.size(), 1u);
  const auto* original_context = sim.fe_system->cutIntegrationContext();
  ASSERT_NE(original_context, nullptr);
  const auto lifecycle_revision_before_transaction =
      lifecycle.valueRevision();
  const auto constraint_revision_before_transaction =
      sim.fe_system->constraintLayoutRevision();
  const auto sparsity_revision_before_transaction =
      sim.fe_system->sparsityPatternRevision();
  const auto constraint_count_before_transaction =
      sim.fe_system->constraints().numConstraints();
  const auto mesh_revisions_before_transaction =
      mesh->event_bus().revision_state();
  const auto refresh_cache_before_transaction = refresh_cache;
  const auto mesh_phi_count =
      mesh->field_components(mesh_field) *
      mesh->field_entity_count(mesh_field);
  const auto* mesh_phi_data_before_transaction =
      static_cast<const svmp::real_t*>(mesh->field_data(mesh_field));
  ASSERT_NE(mesh_phi_data_before_transaction, nullptr);
  const std::vector<svmp::real_t> mesh_phi_before_transaction(
      mesh_phi_data_before_transaction,
      mesh_phi_data_before_transaction + mesh_phi_count);

  auto candidate_solution = solution;
  for (std::size_t field_dof = 0;
       field_dof < static_cast<std::size_t>(field_dofs.getNumDofs());
       ++field_dof) {
    const auto index = static_cast<std::size_t>(field_offset) + field_dof;
    ASSERT_LT(index, candidate_solution.size());
    candidate_solution[index] += svmp::FE::Real{0.2};
  }
  LevelSetMaintenanceGeometryTransaction geometry_transaction(
      sim, lifecycle, refresh_cache, active_cut_requests);
  ActiveCutContextRefreshReport candidate_report{};
  ASSERT_NO_THROW(candidate_report = geometry_transaction.refresh(
                      *params,
                      std::span<const svmp::FE::Real>(
                          candidate_solution.data(),
                          candidate_solution.size())));
  EXPECT_TRUE(candidate_report.refreshed);
  EXPECT_NE(sim.fe_system->cutIntegrationContext(), original_context);
  EXPECT_TRUE(sim.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_TRUE(lifecycle.transactionActive());
  ASSERT_NO_THROW(geometry_transaction.rollback());

  EXPECT_EQ(sim.fe_system->cutIntegrationContext(), original_context);
  EXPECT_FALSE(sim.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle.transactionActive());
  EXPECT_EQ(lifecycle.valueRevision(),
            lifecycle_revision_before_transaction);
  EXPECT_EQ(sim.fe_system->constraintLayoutRevision(),
            constraint_revision_before_transaction);
  EXPECT_EQ(sim.fe_system->sparsityPatternRevision(),
            sparsity_revision_before_transaction);
  EXPECT_EQ(sim.fe_system->constraints().numConstraints(),
            constraint_count_before_transaction);
  ASSERT_EQ(refresh_cache.last_signature.has_value(),
            refresh_cache_before_transaction.last_signature.has_value());
  if (refresh_cache.last_signature.has_value()) {
    EXPECT_TRUE(*refresh_cache.last_signature ==
                *refresh_cache_before_transaction.last_signature);
  }
  ASSERT_EQ(
      refresh_cache.last_vector_signature.has_value(),
      refresh_cache_before_transaction.last_vector_signature.has_value());
  if (refresh_cache.last_vector_signature.has_value()) {
    EXPECT_TRUE(*refresh_cache.last_vector_signature ==
                *refresh_cache_before_transaction.last_vector_signature);
  }
  const auto mesh_revisions_after_transaction =
      mesh->event_bus().revision_state();
  EXPECT_EQ(mesh_revisions_after_transaction.geometry,
            mesh_revisions_before_transaction.geometry);
  EXPECT_EQ(mesh_revisions_after_transaction.reference_geometry,
            mesh_revisions_before_transaction.reference_geometry);
  EXPECT_EQ(mesh_revisions_after_transaction.current_geometry,
            mesh_revisions_before_transaction.current_geometry);
  EXPECT_EQ(mesh_revisions_after_transaction.reference_rebase,
            mesh_revisions_before_transaction.reference_rebase);
  EXPECT_EQ(mesh_revisions_after_transaction.topology,
            mesh_revisions_before_transaction.topology);
  EXPECT_EQ(mesh_revisions_after_transaction.ownership,
            mesh_revisions_before_transaction.ownership);
  EXPECT_EQ(mesh_revisions_after_transaction.numbering,
            mesh_revisions_before_transaction.numbering);
  EXPECT_EQ(mesh_revisions_after_transaction.field_layout,
            mesh_revisions_before_transaction.field_layout);
  EXPECT_EQ(mesh_revisions_after_transaction.labels,
            mesh_revisions_before_transaction.labels);
  EXPECT_EQ(mesh_revisions_after_transaction.active_configuration,
            mesh_revisions_before_transaction.active_configuration);
  const auto* mesh_phi_data_after_transaction =
      static_cast<const svmp::real_t*>(mesh->field_data(mesh_field));
  ASSERT_NE(mesh_phi_data_after_transaction, nullptr);
  EXPECT_EQ(std::vector<svmp::real_t>(
                mesh_phi_data_after_transaction,
                mesh_phi_data_after_transaction + mesh_phi_count),
            mesh_phi_before_transaction);

  const auto cached_restored_report =
      refreshActiveCutIntegrationContextFromSolutionCached(
          sim,
          *params,
          std::span<const svmp::FE::Real>(solution.data(), solution.size()),
          lifecycle,
          refresh_cache,
          "application-driver-mpi-maintenance-transaction-restored");
  EXPECT_FALSE(cached_restored_report.refreshed);
  EXPECT_EQ(sim.fe_system->cutIntegrationContext(), original_context);
  EXPECT_EQ(lifecycle.valueRevision(),
            lifecycle_revision_before_transaction);

  const int local_transaction_restored =
      sim.fe_system->cutIntegrationContext() == original_context &&
              lifecycle.valueRevision() ==
                  lifecycle_revision_before_transaction &&
              mesh_revisions_after_transaction.active_configuration ==
                  mesh_revisions_before_transaction.active_configuration
          ? 1
          : 0;
  int communicator_transaction_restored = 0;
  MPI_Allreduce(&local_transaction_restored,
                &communicator_transaction_restored,
                1,
                MPI_INT,
                MPI_MIN,
                MPI_COMM_WORLD);
  EXPECT_EQ(communicator_transaction_restored, 1);

  ASSERT_TRUE(sim.fe_system->dofPermutation());
  svmp::FE::backends::FsilsFactory maintenance_factory(
      /*dofs_per_node=*/3,
      sim.fe_system->dofPermutation(),
      MPI_COMM_WORLD);
  auto time_history = svmp::FE::timestepping::TimeHistory::allocate(
      maintenance_factory,
      sim.fe_system->dofHandler().getNumDofs(),
      /*history_depth=*/2,
      /*allocate_second_order_state=*/false);
  time_history.setTime(0.10);
  time_history.setDt(0.05);
  time_history.setPrevDt(0.05);
  time_history.setStepIndex(2);
  scatterFeOrderedSolution(time_history.u(), solution);
  scatterFeOrderedSolution(time_history.uPrev(), solution);
  scatterFeOrderedSolution(time_history.uPrev2(), solution);

  const auto contact_stages = evaluateAcceptedFreeSurfaceContactStages(
      sim,
      svmp::FE::Real{0.075},
      svmp::FE::Real{0.5},
      time_history.uPrev().valueRevision(),
      time_history.u().valueRevision(),
      std::span<const svmp::FE::Real>(solution.data(), solution.size()));
  ASSERT_EQ(contact_stages.size(), 1u);
  const auto contact_stage_constraints =
      captureAcceptedContactStageWallConstraints(sim, contact_stages);
  LevelSetMaintenanceRequest contact_protection_request{};
  contact_protection_request.conservative_phase.liquid_indicator.field_name =
      "phi";
  svmp::FE::level_set::LevelSetP1PhaseTransportGraph contact_graph{};
  contact_graph.nodes =
      static_cast<std::size_t>(field_dofs.getNumDofs());
  const auto contact_protected_nodes =
      conservativePhaseContactProtectedNodes(
          *sim.fe_system,
          contact_protection_request,
          contact_graph,
          contact_stage_constraints);
  ASSERT_EQ(contact_protected_nodes.size(), contact_graph.nodes);
  const auto protected_count = static_cast<unsigned long long>(std::count(
      contact_protected_nodes.begin(),
      contact_protected_nodes.end(),
      std::uint8_t{1u}));
  EXPECT_EQ(protected_count, 8u);
  unsigned long long minimum_protected_count = 0u;
  unsigned long long maximum_protected_count = 0u;
  MPI_Allreduce(&protected_count,
                &minimum_protected_count,
                1,
                MPI_UNSIGNED_LONG_LONG,
                MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(&protected_count,
                &maximum_protected_count,
                1,
                MPI_UNSIGNED_LONG_LONG,
                MPI_MAX,
                MPI_COMM_WORLD);
  EXPECT_EQ(minimum_protected_count, maximum_protected_count);
  time_history.updateGhosts();
  EXPECT_NE(contact_stages.front().endpoint_state_revision,
            time_history.u().valueRevision());
  ActiveCutContextRefreshReport endpoint_report{};
  ASSERT_NO_THROW(
      endpoint_report = refreshActiveCutIntegrationContextFromSolution(
          sim,
          *params,
          std::span<const svmp::FE::Real>(solution.data(), solution.size()),
          lifecycle,
          "application-driver-mpi-contact-endpoint-finalization-test"));
  EXPECT_TRUE(endpoint_report.refreshed);
  const auto endpoint_snapshot_revision =
      sim.fe_system->cutIntegrationContext()
          ->freeSurfaceGeometrySnapshotRevisionForMarker(721);
  EXPECT_NE(endpoint_snapshot_revision,
            contact_stages.front()
                .geometry_revision.snapshot_revision_key);

  LevelSetMaintenanceRequest maintenance_request{};
  maintenance_request.level_set_field_name = "phi";
  maintenance_request.reinitialization.enabled = true;
  maintenance_request.reinitialization.cadence_steps = 1;
  maintenance_request.reinitialization.max_iterations = 100;
  maintenance_request.reinitialization.signed_distance_tolerance = 1.0e-10;
  std::vector<LevelSetMaintenanceRequest> maintenance_requests{
      maintenance_request};
  ASSERT_TRUE(applyLevelSetMaintenance(
      sim,
      time_history,
      maintenance_requests,
      contact_stages,
      contact_stage_constraints,
      std::span<const svmp::FE::Real>(solution.data(), solution.size())));
  const auto maintained_solution =
      gatherFeOrderedSolution(time_history.u());
  ASSERT_EQ(maintained_solution.size(), solution.size());
  for (std::size_t i = 0;
       i < static_cast<std::size_t>(field_dofs.getNumDofs());
       ++i) {
    const auto index = static_cast<std::size_t>(field_offset) + i;
    ASSERT_LT(index, maintained_solution.size());
    EXPECT_NEAR(maintained_solution[index],
                svmp::FE::Real{0.5} * solution[index],
                1.0e-10)
        << "rank=" << rank << " field_dof=" << i;
  }
  ActiveCutContextRefreshReport maintained_report{};
  ASSERT_NO_THROW(
      maintained_report = refreshActiveCutIntegrationContextFromSolution(
          sim,
          *params,
          std::span<const svmp::FE::Real>(maintained_solution.data(),
                                          maintained_solution.size()),
          lifecycle,
          "application-driver-mpi-wall-aware-maintenance-test"));
  EXPECT_TRUE(maintained_report.refreshed);
  EXPECT_NE(
      sim.fe_system->cutIntegrationContext()
          ->freeSurfaceGeometrySnapshotRevisionForMarker(721),
      endpoint_snapshot_revision);
  ASSERT_NO_THROW(recordAcceptedFreeSurfaceDiscreteFunctionals(
      sim,
      /*accepted_step=*/2u,
      svmp::FE::Real{0.10},
      svmp::FE::Real{0.05},
      time_history.u().valueRevision(),
      contact_stages));
  const auto functional_history =
      sim.fe_system->freeSurfaceDiscreteFunctionalHistory();
  ASSERT_EQ(functional_history.size(), 1u);
  const auto& functional_record = functional_history.front();
  EXPECT_EQ(functional_record.accepted_step, 2u);
  EXPECT_EQ(functional_record.state_revision,
            time_history.u().valueRevision());
  EXPECT_NEAR(functional_record.state.owned_liquid_volume, 4.0, 1.0e-12);
  EXPECT_NEAR(functional_record.state.owned_liquid_gas_area, 8.0, 1.0e-12);
  EXPECT_NEAR(functional_record.state.liquid_gas_surface_energy,
              6.0,
              1.0e-12);
  EXPECT_NEAR(functional_record.state.young_wall_energy, -0.5, 1.0e-12);
  EXPECT_NEAR(functional_record.state.total_potential, 5.5, 1.0e-12);
  ASSERT_EQ(functional_record.state.walls.size(), 3u);
  for (const auto& wall : functional_record.state.walls) {
    const bool dynamic_wall =
        wall.boundary_marker == kLeftOnlyWall ||
        wall.boundary_marker == kRightOnlyWall;
    EXPECT_EQ(wall.equilibrium_contact_angle_radians.has_value(),
              dynamic_wall);
  }
  ASSERT_TRUE(functional_record.contact_stage.has_value());
  const auto& contact_stage = *functional_record.contact_stage;
  EXPECT_DOUBLE_EQ(contact_stage.stage_time, svmp::FE::Real{0.075});
  EXPECT_DOUBLE_EQ(contact_stage.stage_alpha_f, svmp::FE::Real{0.5});
  EXPECT_NEAR(contact_stage.state.owned_contact_measure, 2.0, 1.0e-12);
  EXPECT_NEAR(contact_stage.state.line_friction_dissipation,
              0.25,
              1.0e-12);
  EXPECT_NEAR(contact_stage.state.owned_wetted_wall_measure,
              1.0,
              1.0e-12);
  EXPECT_NEAR(contact_stage.state.wall_slip_dissipation,
              0.125,
              1.0e-12);
  EXPECT_NEAR(contact_stage.state.total_dissipation,
              0.375,
              1.0e-12);
  ASSERT_EQ(contact_stage.state.walls.size(), 2u);
  std::size_t contact_qpoints = 0u;
  for (const auto& wall : contact_stage.state.walls) {
    contact_qpoints += wall.owned_quadrature_point_count;
    EXPECT_EQ(wall.motion,
              svmp::FE::interfaces::FreeSurfaceContactMotion::Advancing);
    ASSERT_TRUE(wall.mean_contact_speed.has_value());
    ASSERT_TRUE(wall.mean_constitutive_residual.has_value());
    EXPECT_NEAR(*wall.mean_contact_speed, 0.25, 1.0e-12);
    EXPECT_NEAR(*wall.mean_constitutive_residual, 0.0, 1.0e-12);
    EXPECT_NEAR(wall.contact_speed_squared_integral, 0.0625, 1.0e-12);
    EXPECT_EQ(wall.owned_wetted_wall_quadrature_point_count, 2u);
    EXPECT_NEAR(wall.owned_wetted_wall_measure, 0.5, 1.0e-12);
    EXPECT_NEAR(wall.wall_slip_speed_squared_integral,
                0.03125,
                1.0e-12);
    EXPECT_NEAR(wall.wall_slip_dissipation, 0.0625, 1.0e-12);
    ASSERT_TRUE(wall.mean_wall_slip_speed.has_value());
    EXPECT_NEAR(*wall.mean_wall_slip_speed, 0.25, 1.0e-12);
  }
  EXPECT_EQ(contact_qpoints, 2u);

  const auto* cut_context = sim.fe_system->cutIntegrationContext();
  ASSERT_NE(cut_context, nullptr);
  const auto local_retained_negative_rules = static_cast<int>(
      cut_context
          ->generatedVolumeRulesForMarkerAndSide(
              721, svmp::FE::geometry::CutIntegrationSide::Negative)
          .size());
  int summed_rank_local_retained_negative_rules = 0;
  MPI_Allreduce(&local_retained_negative_rules,
                &summed_rank_local_retained_negative_rules,
                1,
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD);
  EXPECT_EQ(local_retained_negative_rules, 5);
  EXPECT_EQ(summed_rank_local_retained_negative_rules, 10)
      << "One ghost-cell rule per rank is intentionally retained for halo "
         "assembly; owned global metrics above must still equal the serial "
         "eight-cell result.";

  const auto contact_rule_count = [&](int boundary_marker) {
    svmp::FE::interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey key;
    key.source =
        svmp::FE::interfaces::LevelSetInterfaceSource::fromField(phi);
    key.domain_id = "mpi_disjoint_wall_interface";
    key.isovalue = 0.0;
    key.interface_marker = 721;
    key.boundary_marker = boundary_marker;
    const auto marker = svmp::FE::interfaces::
        stableGeneratedInterfaceBoundaryIntersectionMarker(key);
    const auto* context = sim.fe_system->cutIntegrationContext();
    return context == nullptr
               ? 0
               : static_cast<int>(
                     context->interfaceRulesForMarker(marker).size());
  };

  const int local_left_rules = contact_rule_count(kLeftOnlyWall);
  const int local_extra_rules = contact_rule_count(kLeftOnlyExtraWall);
  const int local_right_rules = contact_rule_count(kRightOnlyWall);
  int global_left_rules = 0;
  int global_extra_rules = 0;
  int global_right_rules = 0;
  MPI_Allreduce(&local_left_rules,
                &global_left_rules,
                1,
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&local_extra_rules,
                &global_extra_rules,
                1,
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&local_right_rules,
                &global_right_rules,
                1,
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD);
  EXPECT_EQ(global_left_rules, 1);
  EXPECT_EQ(global_extra_rules, 0);
  EXPECT_EQ(global_right_rules, 1);
}

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     VelocityExtensionArtifactsPublishEveryRankOrRollback)
{
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  ASSERT_EQ(size, 2);

  std::uint64_t unique = 0u;
  if (rank == 0) {
    unique = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
  }
  MPI_Bcast(&unique, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  const auto rollback_directory =
      std::filesystem::temp_directory_path() /
      ("svmp-velocity-extension-mpi-rollback-" +
       std::to_string(unique));
  const auto success_directory =
      std::filesystem::temp_directory_path() /
      ("svmp-velocity-extension-mpi-success-" +
       std::to_string(unique));

  const auto mesh = makePartitionedQuadStrip();
  const auto input = makeExtensionInputs(*mesh);
  const auto& revisions = mesh->event_bus().revision_state();
  const auto map_revision = application::core::velocityExtensionMapRevision(
      revisions.geometry,
      revisions.topology,
      revisions.ownership,
      revisions.numbering,
      71u,
      input.phi,
      input.active);
  const auto snapshot =
      application::core::buildVelocityExtensionMapSnapshot(
          *mesh,
          svmp::MeshComm::world(),
          map_revision,
          input.phi,
          input.source,
          kComponents,
          input.active,
          kComponents,
          kComponents,
          kCellCount,
          false,
          std::span<const WallVelocityExtensionConstraint>{});
  ASSERT_TRUE(snapshot);
  AcceptedVelocityExtensionMapRecord record{
      .level_set_field_name = "phi",
      .source_velocity_field_name = "Velocity",
      .target_velocity_field_name = "LevelSetAdvectionVelocity",
      .geometry_domain_id = "free_surface",
      .operator_tag = "level_set",
      .extension_method = "wall_compatible_normal",
      .isovalue = 0.0,
      .extension_band_layers = kCellCount,
      .enforce_wall_impermeability = false,
      .retained_side = LevelSetActiveSide::Negative,
      .snapshot = snapshot,
  };
  std::vector<AcceptedVelocityExtensionMapRecord> records{record};
  Parameters params;
  params.general_simulation_parameters.save_results_in_folder.set(
      rollback_directory.string());
  AcceptedVelocityExtensionMapRegistry accepted_maps;

  auto inconsistent_records = records;
  if (rank == 1) {
    inconsistent_records.front().target_velocity_field_name =
        "DifferentExtensionField";
  }
  EXPECT_THROW(
      writeAcceptedVelocityExtensionMapArtifacts(
          params,
          inconsistent_records,
          1u,
          0.1,
          0.1,
          41u,
          svmp::MeshComm::world(),
          accepted_maps),
      std::runtime_error);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    EXPECT_FALSE(std::filesystem::exists(rollback_directory));
  }
  MPI_Barrier(MPI_COMM_WORLD);

  application::core::VelocityExtensionMapArtifactResult preexisting;
  if (rank == 0) {
    preexisting = application::core::writeVelocityExtensionMapArtifact(
        rollback_directory / "velocity_extension_maps",
        application::core::VelocityExtensionMapArtifactContext{
            .level_set_field_name = record.level_set_field_name,
            .source_velocity_field_name =
                record.source_velocity_field_name,
            .target_velocity_field_name =
                record.target_velocity_field_name,
            .geometry_domain_id = record.geometry_domain_id,
            .operator_tag = record.operator_tag,
            .extension_method = record.extension_method,
            .retained_side = activeSideName(record.retained_side),
            .accepted_step = 1u,
            .accepted_time = 0.1,
            .time_step = 0.1,
            .state_revision = 41u,
            .isovalue = record.isovalue,
            .extension_band_layers = record.extension_band_layers,
            .enforce_wall_impermeability =
                record.enforce_wall_impermeability,
            .rank = rank,
            .ranks = size,
        },
        *snapshot);
    ASSERT_TRUE(preexisting.success) << preexisting.diagnostic;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  EXPECT_THROW(
      writeAcceptedVelocityExtensionMapArtifacts(
          params,
          records,
          1u,
          0.1,
          0.1,
          41u,
          svmp::MeshComm::world(),
          accepted_maps),
      std::runtime_error);
  EXPECT_TRUE(accepted_maps.empty());
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::size_t json_files = 0u;
    for (const auto& entry : std::filesystem::directory_iterator(
             rollback_directory / "velocity_extension_maps")) {
      json_files += entry.path().extension() == ".json" ? 1u : 0u;
    }
    EXPECT_EQ(json_files, 1u)
        << "The successful remote shard must be removed after any-rank publication failure.";
    std::error_code cleanup_error;
    std::filesystem::remove_all(rollback_directory, cleanup_error);
    EXPECT_FALSE(cleanup_error);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  params.general_simulation_parameters.save_results_in_folder.set(
      success_directory.string());
  ASSERT_NO_THROW(writeAcceptedVelocityExtensionMapArtifacts(
      params,
      records,
      2u,
      0.2,
      0.1,
      42u,
      svmp::MeshComm::world(),
      accepted_maps));
  EXPECT_EQ(accepted_maps.size(), 1u);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::size_t json_files = 0u;
    for (const auto& entry : std::filesystem::directory_iterator(
             success_directory / "velocity_extension_maps")) {
      if (entry.path().extension() != ".json") {
        continue;
      }
      ++json_files;
      std::ifstream input_stream(entry.path());
      ASSERT_TRUE(input_stream.is_open());
      const std::string contents{
          std::istreambuf_iterator<char>{input_stream},
          std::istreambuf_iterator<char>{}};
      EXPECT_NE(contents.find(
                    "\"schema\":\"svmp.velocity_extension_map.v1\""),
                std::string::npos);
      EXPECT_NE(contents.find("\"ranks\":2"), std::string::npos);
    }
    EXPECT_EQ(json_files, 2u);
    std::error_code cleanup_error;
    std::filesystem::remove_all(success_directory, cleanup_error);
    EXPECT_FALSE(cleanup_error);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     ConservativePhaseArtifactPublishesOnceAfterCollectivePreflight)
{
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  ASSERT_EQ(size, 2);

  std::uint64_t unique = 0u;
  if (rank == 0) {
    unique = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
  }
  MPI_Bcast(&unique, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  const auto output_directory =
      std::filesystem::temp_directory_path() /
      ("svmp-conservative-phase-mpi-artifact-" +
       std::to_string(unique));

  Parameters params;
  params.general_simulation_parameters.save_results_in_folder.set(
      output_directory.string());
  std::vector<LevelSetMaintenanceRequest> requests(1u);
  auto& request = requests.front();
  request.level_set_field_name = "phi";
  request.conservative_phase.enabled = true;
  request.conservative_phase.write_flux_artifacts = true;
  request.conservative_phase.flux_artifact_cadence_steps = 1;
  request.conservative_phase.liquid_indicator.field_name = "phase";
  request.volume_cut_request = ActiveCutVolumeRequest{};
  request.volume_cut_request->domain_id = "mpi_phase_interface";
  svmp::FE::level_set::LevelSetP1PhaseTransportGraph graph;
  graph.success = true;
  graph.geometry_revision = 2u;
  graph.topology_revision = 3u;
  graph.ownership_revision = 4u;
  graph.numbering_revision = 5u;
  graph.dof_layout_revision = 6u;
  request.conservative_phase_graph = std::move(graph);

  const std::array<svmp::FE::Real, 2> volumes{1.0, 1.0};
  const std::array<svmp::FE::Real, 2> previous{0.75, 0.25};
  const std::array<svmp::FE::Real, 2> lower{0.0, 0.0};
  const std::array<svmp::FE::Real, 2> upper{1.0, 1.0};
  const std::array<svmp::FE::level_set::LevelSetPhaseFluxEdge, 1>
      flux_edges{
          svmp::FE::level_set::LevelSetPhaseFluxEdge{
              0, 1, 0.05, 0.20},
      };
  auto correction = svmp::FE::level_set::
      applyLevelSetConservativePhaseFluxCorrection(
          svmp::FE::level_set::LevelSetPhaseFluxStageView{
              .lumped_control_volume = volumes,
              .previous_liquid_indicator = previous,
              .lower_liquid_indicator = lower,
              .upper_liquid_indicator = upper,
              .interior_edges = flux_edges,
          });
  ASSERT_TRUE(correction.success) << correction.diagnostic;

  ConservativePhaseCandidateResult candidate;
  candidate.maintenance_ledgers.resize(1u);
  auto& stage = candidate.maintenance_ledgers.front().transport_stage;
  stage.success = true;
  stage.courant_satisfied = true;
  stage.low_order_coefficients_nonnegative = true;
  stage.strong_form_decomposition_satisfied = true;
  stage.maximum_courant = 0.25;
  stage.nodal_courant = {0.25, 0.25};
  stage.physical_boundary_mass_transfer = {0.0, 0.0};
  stage.discrete_divergence_mass_source = {0.0, 0.0};
  stage.flux_edges.assign(flux_edges.begin(), flux_edges.end());
  stage.correction = std::move(correction);
  candidate.maintenance_ledgers.front().region_ledger =
      svmp::FE::level_set::buildLevelSetPhaseRegionLedgers(
          stage.correction,
          std::span<const
              svmp::FE::level_set::LevelSetPhaseRegionDefinition>{});
  ASSERT_TRUE(candidate.maintenance_ledgers.front().region_ledger.success)
      << candidate.maintenance_ledgers.front().region_ledger.diagnostic;

  auto inconsistent_requests = requests;
  if (rank == 1) {
    inconsistent_requests.front().conservative_phase
        .write_flux_artifacts = false;
  }
  EXPECT_THROW(
      writeAcceptedConservativePhaseArtifacts(
          params,
          inconsistent_requests,
          candidate,
          12u,
          svmp::FE::Real{0.6},
          svmp::FE::Real{0.05},
          17u,
          svmp::MeshComm::world()),
      std::runtime_error);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    EXPECT_FALSE(std::filesystem::exists(output_directory));
  }
  MPI_Barrier(MPI_COMM_WORLD);

  auto inconsistent_region_requests = requests;
  if (rank == 1) {
    inconsistent_region_requests.front().conservative_phase
        .fixed_flux_regions =
        svmp::FE::level_set::parseLevelSetPhaseRegionBoxes(
            "rank_one_only|observer|*|*|*|*|*|*");
  }
  EXPECT_THROW(
      writeAcceptedConservativePhaseArtifacts(
          params,
          inconsistent_region_requests,
          candidate,
          12u,
          svmp::FE::Real{0.6},
          svmp::FE::Real{0.05},
          17u,
          svmp::MeshComm::world()),
      std::runtime_error);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    EXPECT_FALSE(std::filesystem::exists(output_directory));
  }
  MPI_Barrier(MPI_COMM_WORLD);

  ConservativePhaseCandidateResult locally_invalid_candidate;
  locally_invalid_candidate.maintenance_ledgers =
      candidate.maintenance_ledgers;
  if (rank == 1) {
    locally_invalid_candidate.maintenance_ledgers.clear();
  }
  EXPECT_THROW(
      writeAcceptedConservativePhaseArtifacts(
          params,
          requests,
          locally_invalid_candidate,
          12u,
          svmp::FE::Real{0.6},
          svmp::FE::Real{0.05},
          17u,
          svmp::MeshComm::world()),
      std::runtime_error);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    EXPECT_FALSE(std::filesystem::exists(output_directory));
  }
  MPI_Barrier(MPI_COMM_WORLD);

  ASSERT_NO_THROW(writeAcceptedConservativePhaseArtifacts(
      params,
      requests,
      candidate,
      12u,
      svmp::FE::Real{0.6},
      svmp::FE::Real{0.05},
      17u,
      svmp::MeshComm::world()));
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    const auto artifact_path =
        output_directory / "conservative_phase_flux" /
        "conservative_phase_flux_phase_step_00000012.json";
    ASSERT_TRUE(std::filesystem::is_regular_file(artifact_path));
    EXPECT_FALSE(std::filesystem::exists(
        std::filesystem::path(artifact_path.string() + ".tmp")));
    std::ifstream input(artifact_path);
    ASSERT_TRUE(input.is_open());
    const std::string contents{
        std::istreambuf_iterator<char>{input},
        std::istreambuf_iterator<char>{}};
    EXPECT_NE(contents.find("\"accepted_step\":12"),
              std::string::npos);
    EXPECT_NE(contents.find("\"nodes\":[{"),
              std::string::npos);
    std::error_code cleanup_error;
    std::filesystem::remove_all(output_directory, cleanup_error);
    EXPECT_FALSE(cleanup_error);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     MaintenanceWorkRowsAreIdenticalAcrossTwoRanks)
{
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    GTEST_SKIP() << "This equality fixture requires two ranks.";
  }

  const auto value = [](double total, std::uint64_t snapshot) {
    return application::core::LevelSetAuthoritativeFunctionalValue{
        .interface_marker = 902,
        .snapshot_revision = snapshot,
        .mesh_topology_revision = 44u,
        .cut_topology_revision = 45u,
        .liquid_volume = 1.5,
        .liquid_gas_area = 2.25,
        .wetted_wall_area = 0.625,
        .contact_measure = 0.25,
        .surface_energy = 3.5,
        .young_wall_energy = -0.375,
        .volume_constraint_potential = total - 3.125,
        .total_potential = total,
    };
  };

  application::core::LevelSetMaintenanceWorkLedger ledger;
  ledger.beginTransaction(
      application::core::LevelSetMaintenanceWorkTransaction{
          .transaction_id = 601u,
          .step = 12u,
          .attempt = 3u,
          .time = 0.6,
          .dt = 0.05,
          .declared_stage =
              application::core::LevelSetMaintenanceDeclaredStage::
                  ProspectiveAcceptedEndpoint,
          .extension_map_revision = 77u,
      });
  ledger.stageRow(
      application::core::LevelSetMaintenanceWorkSubstage::
          Reinitialization,
      7001u,
      7002u,
      {value(4.0, 810u)},
      {value(4.2, 811u)});
  ledger.stageRow(
      application::core::LevelSetMaintenanceWorkSubstage::
          GlobalCorrection,
      7002u,
      7003u,
      {value(4.2, 811u)},
      {value(4.1, 812u)});
  ledger.commitTransaction();
  ASSERT_EQ(ledger.acceptedRows().size(), 2u);
  ASSERT_EQ(ledger.acceptedAttempts().size(), 1u);

  const auto& attempt = ledger.acceptedAttempts().front();
  const std::array<std::uint64_t, 8> local_attempt_metadata{
      attempt.transaction_id,
      static_cast<std::uint64_t>(attempt.status),
      attempt.step,
      attempt.attempt,
      static_cast<std::uint64_t>(attempt.declared_stage),
      attempt.extension_map_revision.value_or(0u),
      static_cast<std::uint64_t>(attempt.row_count),
      static_cast<std::uint64_t>(
          attempt.accepted_numerical_work != 0.0)};
  std::array<std::uint64_t, 8> minimum_attempt_metadata{};
  std::array<std::uint64_t, 8> maximum_attempt_metadata{};
  MPI_Allreduce(
      local_attempt_metadata.data(),
      minimum_attempt_metadata.data(),
      static_cast<int>(local_attempt_metadata.size()),
      MPI_UINT64_T,
      MPI_MIN,
      MPI_COMM_WORLD);
  MPI_Allreduce(
      local_attempt_metadata.data(),
      maximum_attempt_metadata.data(),
      static_cast<int>(local_attempt_metadata.size()),
      MPI_UINT64_T,
      MPI_MAX,
      MPI_COMM_WORLD);
  EXPECT_EQ(minimum_attempt_metadata, maximum_attempt_metadata);

  for (const auto& row : ledger.acceptedRows()) {
    const std::array<std::uint64_t, 16> local_metadata{
        row.transaction_id,
        static_cast<std::uint64_t>(row.status),
        static_cast<std::uint64_t>(row.substage),
        row.step,
        row.attempt,
        row.algebraic_state_revision_before,
        row.algebraic_state_revision_after,
        row.snapshot_set_revision_before,
        row.snapshot_set_revision_after,
        row.mesh_topology_set_revision_before,
        row.mesh_topology_set_revision_after,
        row.cut_topology_set_revision_before,
        row.cut_topology_set_revision_after,
        row.extension_map_revision_before.value_or(0u),
        row.extension_map_revision_after.value_or(0u),
        static_cast<std::uint64_t>(row.declared_stage),
    };
    std::array<std::uint64_t, 16> minimum_metadata{};
    std::array<std::uint64_t, 16> maximum_metadata{};
    MPI_Allreduce(
        local_metadata.data(),
        minimum_metadata.data(),
        static_cast<int>(local_metadata.size()),
        MPI_UINT64_T,
        MPI_MIN,
        MPI_COMM_WORLD);
    MPI_Allreduce(
        local_metadata.data(),
        maximum_metadata.data(),
        static_cast<int>(local_metadata.size()),
        MPI_UINT64_T,
        MPI_MAX,
        MPI_COMM_WORLD);
    EXPECT_EQ(minimum_metadata, maximum_metadata);

    ASSERT_EQ(row.before.size(), 1u);
    ASSERT_EQ(row.after.size(), 1u);
    const std::array<double, 6> local_values{
        row.time,
        row.dt,
        row.before.front().total_potential,
        row.after.front().total_potential,
        row.numerical_work,
        row.accepted_numerical_work,
    };
    std::array<double, 6> minimum_values{};
    std::array<double, 6> maximum_values{};
    MPI_Allreduce(
        local_values.data(),
        minimum_values.data(),
        static_cast<int>(local_values.size()),
        MPI_DOUBLE,
        MPI_MIN,
        MPI_COMM_WORLD);
    MPI_Allreduce(
        local_values.data(),
        maximum_values.data(),
        static_cast<int>(local_values.size()),
        MPI_DOUBLE,
        MPI_MAX,
        MPI_COMM_WORLD);
    EXPECT_EQ(minimum_values, maximum_values);
  }
}

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     MaintenanceAlgebraicRevisionRejectsRankLocalSlices)
{
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    GTEST_SKIP() << "This collective provenance fixture requires two ranks.";
  }

  const std::vector<svmp::FE::Real> gathered_fe_state{
      1.0, -2.0, 3.5, 4.25};
  const auto revision =
      collectiveLevelSetMaintenanceAlgebraicRevision(
          gathered_fe_state, svmp::MeshComm(MPI_COMM_WORLD));
  const auto [minimum_revision, maximum_revision] =
      globalMinMaxUint64(
          revision, svmp::MeshComm(MPI_COMM_WORLD));
  EXPECT_EQ(minimum_revision, maximum_revision);

  const std::vector<svmp::FE::Real> rank_local_slice{
      static_cast<svmp::FE::Real>(rank + 1)};
  EXPECT_THROW(
      (void)collectiveLevelSetMaintenanceAlgebraicRevision(
          rank_local_slice, svmp::MeshComm(MPI_COMM_WORLD)),
      std::runtime_error);
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
