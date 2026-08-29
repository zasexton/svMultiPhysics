#include <gtest/gtest.h>

// The workflow helper under test currently has internal linkage in the
// application driver.  Include the implementation, as the serial workflow
// tests do, so this exercises the production graph-extension implementation.
#include "../../Core/ApplicationDriver.cpp"

#include "Application/Translators/MeshTranslator.h"
#include "FE/Assembly/AssemblyContext.h"
#include "FE/Assembly/AssemblyKernel.h"
#include "FE/Backends/FSILS/FsilsFactory.h"
#include "FE/Backends/FSILS/FsilsMatrix.h"
#include "FE/Backends/FSILS/FsilsVector.h"
#include "FE/Backends/Utils/BackendOptions.h"
#include "FE/Forms/Forms.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Sparsity/DistributedSparsityPattern.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Spaces/ProductSpace.h"
#include "FE/Systems/FormsInstaller.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/DistributedTopology.h"
#include "FE/Spaces/H1Space.h"
#include "Mesh/Fields/MeshFields.h"
#include "NativeManufacturedChannelMPIHarness.h"
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
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

namespace channel_ns =
    svmp::Physics::formulations::navier_stokes;

class MpiWorkflowScopedEnvVar {
public:
  MpiWorkflowScopedEnvVar(const char* key,
                          std::optional<std::string> value)
      : key_(key)
  {
    if (const char* old = std::getenv(key)) {
      original_ = std::string(old);
    }
    set(std::move(value));
  }

  ~MpiWorkflowScopedEnvVar() { set(original_); }

private:
  void set(const std::optional<std::string>& value) const
  {
    if (value.has_value()) {
      ::setenv(key_, value->c_str(), 1);
    } else {
      ::unsetenv(key_);
    }
  }

  const char* key_;
  std::optional<std::string> original_{};
};

constexpr int kCellCount = 8;
constexpr std::size_t kComponents = 2u;
constexpr svmp::label_t kHorizontalWall = 4242;
constexpr svmp::label_t kLeftOnlyWall = 5101;
constexpr svmp::label_t kLeftOnlyExtraWall = 5102;
constexpr svmp::label_t kRightOnlyWall = 5201;

class MpiWorkflowScaledMassKernel final
    : public svmp::FE::assembly::AssemblyKernel {
public:
  MpiWorkflowScaledMassKernel(svmp::FE::Real matrix_scale,
                              svmp::FE::Real vector_scale)
      : matrix_scale_(matrix_scale),
        vector_scale_(vector_scale)
  {
  }

  [[nodiscard]] svmp::FE::assembly::RequiredData getRequiredData()
      const override
  {
    using svmp::FE::assembly::RequiredData;
    return RequiredData::BasisValues |
           RequiredData::IntegrationWeights;
  }

  [[nodiscard]] bool hasStateIndependentMatrix()
      const noexcept override
  {
    return true;
  }

  void computeCell(
      const svmp::FE::assembly::AssemblyContext& context,
      svmp::FE::assembly::KernelOutput& output) override
  {
    const auto test_dofs = context.numTestDofs();
    const auto trial_dofs = context.numTrialDofs();
    const bool want_matrix = output.has_matrix;
    const bool want_vector = output.has_vector;
    output.reserve(
        test_dofs,
        trial_dofs,
        want_matrix,
        want_vector);

    for (svmp::FE::LocalIndex q = 0;
         q < context.numQuadraturePoints();
         ++q) {
      const auto weight = context.integrationWeight(q);
      for (svmp::FE::LocalIndex i = 0; i < test_dofs; ++i) {
        const auto test_value = context.basisValue(i, q);
        if (want_vector) {
          output.vectorEntry(i) +=
              vector_scale_ * weight * test_value;
        }
        if (!want_matrix) {
          continue;
        }
        for (svmp::FE::LocalIndex j = 0;
             j < trial_dofs;
             ++j) {
          output.matrixEntry(i, j) +=
              matrix_scale_ * weight * test_value *
              context.trialBasisValue(j, q);
        }
      }
    }
  }

  [[nodiscard]] std::string name() const override
  {
    return "MpiWorkflowScaledMassKernel";
  }

private:
  svmp::FE::Real matrix_scale_{0.0};
  svmp::FE::Real vector_scale_{0.0};
};

void installMpiWorkflowExactConstantPressureCertificate(
    svmp::FE::systems::FESystem& system,
    svmp::FE::FieldId velocity,
    svmp::FE::FieldId pressure)
{
  constexpr std::array<const char*, 6> diagnostic_operators{
      "equations_diagnostic_ns_free_surface_pressure_virtual_work",
      "equations_diagnostic_ns_free_surface_surface_energy_virtual_work",
      "equations_diagnostic_ns_free_surface_gravitational_potential_virtual_work",
      "equations_diagnostic_ns_free_surface_physical_potential_virtual_work",
      "equations_diagnostic_ns_free_surface_pressure_representability_load_virtual_work",
      "equations_diagnostic_ns_free_surface_conservative_balance",
  };
  constexpr std::array<svmp::FE::Real, 6> vector_scales{
      -2.0, 2.0, 0.0, 2.0, 2.0, 0.0};
  for (std::size_t i = 0u;
       i < diagnostic_operators.size();
       ++i) {
    system.addOperator(diagnostic_operators[i]);
    system.addCellKernel(
        diagnostic_operators[i],
        velocity,
        velocity,
        std::make_shared<MpiWorkflowScaledMassKernel>(
            /*matrix_scale=*/0.0,
            vector_scales[i]));
  }

  constexpr const char* pair_operator =
      "equations_diagnostic_ns_free_surface_pressure_representability_pair";
  system.addOperator(pair_operator);
  system.addCellKernel(
      pair_operator,
      velocity,
      pressure,
      std::make_shared<MpiWorkflowScaledMassKernel>(
          /*matrix_scale=*/1.0,
          /*vector_scale=*/0.0));
  system.addCellKernel(
      pair_operator,
      pressure,
      velocity,
      std::make_shared<MpiWorkflowScaledMassKernel>(
          /*matrix_scale=*/1.0,
          /*vector_scale=*/0.0));
}

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
makePartitionedFlatCapillaryFanMesh(int normal_axis)
{
  if (normal_axis != 0 && normal_axis != 1) {
    throw std::invalid_argument(
        "flat capillary fan normal axis must be zero or one");
  }

  std::vector<svmp::real_t> coordinates{
      0.0, 0.0,
      3.0, 0.0,
      3.0, 1.0,
      0.0, 1.0,
      1.5, 0.25,
  };
  if (normal_axis == 0) {
    for (std::size_t vertex = 0u;
         vertex < coordinates.size() / 2u;
         ++vertex) {
      const auto x = coordinates[2u * vertex];
      const auto y = coordinates[2u * vertex + 1u];
      coordinates[2u * vertex] = y;
      coordinates[2u * vertex + 1u] = 3.0 - x;
    }
  }
  const std::vector<svmp::offset_t> offsets{0, 3, 6, 9, 12};
  const std::vector<svmp::index_t> connectivity{
      0, 1, 4,
      1, 2, 4,
      2, 3, 4,
      3, 0, 4,
  };
  svmp::CellShape triangle{};
  triangle.family = svmp::CellFamily::Triangle;
  triangle.num_corners = 3;
  triangle.order = 1;
  const std::vector<svmp::CellShape> shapes(4u, triangle);

  auto mesh = std::make_shared<svmp::Mesh>(
      svmp::MeshComm(MPI_COMM_WORLD));
  mesh->build_from_arrays_global_and_partition(
      /*spatial_dim=*/2,
      coordinates,
      offsets,
      connectivity,
      shapes,
      svmp::PartitionHint::Cells,
      /*ghost_layers=*/1,
      {{"partition_method", "block"}});
  return mesh;
}

[[nodiscard]] std::shared_ptr<svmp::Mesh>
makePartitionedHydrostaticPressureMesh(
    int normal_axis,
    bool column_major_cells,
    bool reverse_vertex_numbering)
{
  if (normal_axis != 0 && normal_axis != 1) {
    throw std::invalid_argument(
        "hydrostatic pressure mesh normal axis must be zero or one");
  }

  constexpr std::size_t columns = 5u;
  const std::array<std::array<svmp::real_t, columns>, 5> x_rows{{
      {{0.0, 0.75, 1.50, 2.25, 3.0}},
      {{0.0, 0.68, 1.47, 2.29, 3.0}},
      {{0.0, 0.81, 1.55, 2.18, 3.0}},
      {{0.0, 0.72, 1.42, 2.33, 3.0}},
      {{0.0, 0.75, 1.50, 2.25, 3.0}},
  }};
  constexpr std::array<svmp::real_t, 5> y_rows{{
      0.0, 0.2, 0.4, 0.7, 1.0}};

  std::vector<svmp::real_t> coordinates;
  coordinates.reserve(2u * columns * y_rows.size());
  for (std::size_t row = 0u; row < y_rows.size(); ++row) {
    for (std::size_t column = 0u; column < columns; ++column) {
      coordinates.push_back(x_rows[row][column]);
      coordinates.push_back(y_rows[row]);
    }
  }
  if (normal_axis == 0) {
    for (std::size_t vertex = 0u;
         vertex < coordinates.size() / 2u;
         ++vertex) {
      const auto tangent_coordinate = coordinates[2u * vertex];
      const auto normal_coordinate = coordinates[2u * vertex + 1u];
      coordinates[2u * vertex] = normal_coordinate;
      coordinates[2u * vertex + 1u] = 3.0 - tangent_coordinate;
    }
  }

  const auto vertex_count = coordinates.size() / 2u;
  if (reverse_vertex_numbering) {
    std::vector<svmp::real_t> reversed_coordinates(coordinates.size());
    for (std::size_t old_vertex = 0u;
         old_vertex < vertex_count;
         ++old_vertex) {
      const auto new_vertex = vertex_count - 1u - old_vertex;
      for (std::size_t component = 0u; component < 2u; ++component) {
        reversed_coordinates[2u * new_vertex + component] =
            coordinates[2u * old_vertex + component];
      }
    }
    coordinates = std::move(reversed_coordinates);
  }

  std::vector<svmp::offset_t> offsets{0};
  std::vector<svmp::index_t> connectivity;
  std::vector<svmp::CellShape> shapes;
  svmp::CellShape triangle{};
  triangle.family = svmp::CellFamily::Triangle;
  triangle.num_corners = 3;
  triangle.order = 1;
  const auto vertex_index = [&](std::size_t row, std::size_t column) {
    auto vertex = row * columns + column;
    if (reverse_vertex_numbering) {
      vertex = vertex_count - 1u - vertex;
    }
    return static_cast<svmp::index_t>(vertex);
  };
  const auto append_triangle = [&](svmp::index_t first,
                                   svmp::index_t second,
                                   svmp::index_t third) {
    connectivity.insert(connectivity.end(), {first, second, third});
    offsets.push_back(static_cast<svmp::offset_t>(connectivity.size()));
    shapes.push_back(triangle);
  };
  const auto append_quad = [&](std::size_t row, std::size_t column) {
    const auto lower_left = vertex_index(row, column);
    const auto lower_right = vertex_index(row, column + 1u);
    const auto upper_left = vertex_index(row + 1u, column);
    const auto upper_right = vertex_index(row + 1u, column + 1u);
    if ((row + column) % 2u == 0u) {
      append_triangle(lower_left, lower_right, upper_right);
      append_triangle(lower_left, upper_right, upper_left);
    } else {
      append_triangle(lower_left, lower_right, upper_left);
      append_triangle(lower_right, upper_right, upper_left);
    }
  };
  if (column_major_cells) {
    for (std::size_t column = 0u; column + 1u < columns; ++column) {
      for (std::size_t row = 0u; row + 1u < y_rows.size(); ++row) {
        append_quad(row, column);
      }
    }
  } else {
    for (std::size_t row = 0u; row + 1u < y_rows.size(); ++row) {
      for (std::size_t column = 0u; column + 1u < columns; ++column) {
        append_quad(row, column);
      }
    }
  }

  auto mesh = std::make_shared<svmp::Mesh>(
      svmp::MeshComm(MPI_COMM_WORLD));
  mesh->build_from_arrays_global_and_partition(
      /*spatial_dim=*/2,
      coordinates,
      offsets,
      connectivity,
      shapes,
      svmp::PartitionHint::Cells,
      /*ghost_layers=*/1,
      {{"partition_method", "block"}});
  return mesh;
}

[[nodiscard]] std::shared_ptr<svmp::Mesh>
makePartitionedHydrostaticPressureMesh3D(
    int normal_axis,
    bool tangent_major_cells,
    bool reverse_vertex_numbering)
{
  if (normal_axis < 0 || normal_axis >= 3) {
    throw std::invalid_argument(
        "three-dimensional hydrostatic pressure mesh normal axis must be "
        "zero, one, or two");
  }

  constexpr std::array<svmp::real_t, 3> first_tangent_coordinates{{
      0.0, 1.5, 3.0}};
  constexpr std::array<svmp::real_t, 3> second_tangent_coordinates{{
      0.0, 1.0, 2.0}};
  constexpr std::array<svmp::real_t, 5> normal_coordinates{{
      0.0, 0.2, 0.4, 0.7, 1.0}};
  constexpr std::array<std::array<std::size_t, 4>, 6> tetrahedra{{
      {{0, 1, 2, 6}},
      {{0, 2, 3, 6}},
      {{0, 3, 7, 6}},
      {{0, 7, 4, 6}},
      {{0, 4, 5, 6}},
      {{0, 5, 1, 6}},
  }};

  const int first_tangent_axis = (normal_axis + 1) % 3;
  const int second_tangent_axis = (normal_axis + 2) % 3;
  const auto unnumbered_vertex_index =
      [&](std::size_t first_tangent,
          std::size_t second_tangent,
          std::size_t normal) {
        return first_tangent +
               first_tangent_coordinates.size() *
                   (second_tangent +
                    second_tangent_coordinates.size() * normal);
      };
  const auto vertex_count =
      first_tangent_coordinates.size() *
      second_tangent_coordinates.size() * normal_coordinates.size();
  const auto vertex_index = [&](std::size_t first_tangent,
                                std::size_t second_tangent,
                                std::size_t normal) {
    auto vertex = unnumbered_vertex_index(
        first_tangent, second_tangent, normal);
    if (reverse_vertex_numbering) {
      vertex = vertex_count - 1u - vertex;
    }
    return static_cast<svmp::index_t>(vertex);
  };

  std::vector<svmp::real_t> coordinates(3u * vertex_count, 0.0);
  for (std::size_t normal = 0u;
       normal < normal_coordinates.size();
       ++normal) {
    for (std::size_t second_tangent = 0u;
         second_tangent < second_tangent_coordinates.size();
         ++second_tangent) {
      for (std::size_t first_tangent = 0u;
           first_tangent < first_tangent_coordinates.size();
           ++first_tangent) {
        auto first_tangent_coordinate =
            first_tangent_coordinates[first_tangent];
        auto second_tangent_coordinate =
            second_tangent_coordinates[second_tangent];
        if (first_tangent > 0u &&
            first_tangent + 1u < first_tangent_coordinates.size()) {
          const auto phase = static_cast<int>(
              (3u * normal + 2u * second_tangent + first_tangent) % 5u) -
                             2;
          first_tangent_coordinate +=
              svmp::real_t{0.035} * static_cast<svmp::real_t>(phase);
        }
        if (second_tangent > 0u &&
            second_tangent + 1u < second_tangent_coordinates.size()) {
          const auto phase = static_cast<int>(
              (2u * normal + first_tangent + second_tangent) % 5u) -
                             2;
          second_tangent_coordinate +=
              svmp::real_t{0.04} * static_cast<svmp::real_t>(phase);
        }
        std::array<svmp::real_t, 3> point{};
        point[static_cast<std::size_t>(normal_axis)] =
            normal_coordinates[normal];
        point[static_cast<std::size_t>(first_tangent_axis)] =
            first_tangent_coordinate;
        point[static_cast<std::size_t>(second_tangent_axis)] =
            second_tangent_coordinate;
        const auto vertex = static_cast<std::size_t>(
            vertex_index(first_tangent, second_tangent, normal));
        for (std::size_t component = 0u; component < 3u; ++component) {
          coordinates[3u * vertex + component] = point[component];
        }
      }
    }
  }

  std::vector<svmp::offset_t> offsets{0};
  std::vector<svmp::index_t> connectivity;
  std::vector<svmp::CellShape> shapes;
  const auto append_hexahedron =
      [&](std::size_t first_tangent,
          std::size_t second_tangent,
          std::size_t normal) {
        const std::array<svmp::index_t, 8> nodes{{
            vertex_index(first_tangent, second_tangent, normal),
            vertex_index(first_tangent + 1u, second_tangent, normal),
            vertex_index(
                first_tangent + 1u, second_tangent + 1u, normal),
            vertex_index(first_tangent, second_tangent + 1u, normal),
            vertex_index(first_tangent, second_tangent, normal + 1u),
            vertex_index(
                first_tangent + 1u, second_tangent, normal + 1u),
            vertex_index(first_tangent + 1u,
                         second_tangent + 1u,
                         normal + 1u),
            vertex_index(
                first_tangent, second_tangent + 1u, normal + 1u),
        }};
        for (const auto& tetrahedron : tetrahedra) {
          for (const auto local_vertex : tetrahedron) {
            connectivity.push_back(nodes[local_vertex]);
          }
          offsets.push_back(
              static_cast<svmp::offset_t>(connectivity.size()));
          shapes.push_back(
              svmp::CellShape{svmp::CellFamily::Tetra, 4, 1});
        }
      };
  if (tangent_major_cells) {
    for (std::size_t first_tangent = 0u;
         first_tangent + 1u < first_tangent_coordinates.size();
         ++first_tangent) {
      for (std::size_t second_tangent = 0u;
           second_tangent + 1u < second_tangent_coordinates.size();
           ++second_tangent) {
        for (std::size_t normal = 0u;
             normal + 1u < normal_coordinates.size();
             ++normal) {
          append_hexahedron(first_tangent, second_tangent, normal);
        }
      }
    }
  } else {
    for (std::size_t normal = 0u;
         normal + 1u < normal_coordinates.size();
         ++normal) {
      for (std::size_t second_tangent = 0u;
           second_tangent + 1u < second_tangent_coordinates.size();
           ++second_tangent) {
        for (std::size_t first_tangent = 0u;
             first_tangent + 1u < first_tangent_coordinates.size();
             ++first_tangent) {
          append_hexahedron(first_tangent, second_tangent, normal);
        }
      }
    }
  }

  auto mesh = std::make_shared<svmp::Mesh>(
      svmp::MeshComm(MPI_COMM_WORLD));
  mesh->build_from_arrays_global_and_partition(
      /*spatial_dim=*/3,
      coordinates,
      offsets,
      connectivity,
      shapes,
      svmp::PartitionHint::Cells,
      /*ghost_layers=*/1,
      {{"partition_method", "block"}});
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

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     ActiveCutTopologyFingerprintIsInvariantToOwnedRuleRedistribution)
{
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    GTEST_SKIP()
        << "This topology-fingerprint fixture requires exactly two ranks.";
  }

  constexpr std::uint64_t request_a = 0x101u;
  constexpr std::uint64_t request_b = 0x202u;
  const std::array<std::uint64_t, 2> request_order =
      rank == 0
          ? std::array<std::uint64_t, 2>{request_a, request_b}
          : std::array<std::uint64_t, 2>{request_b, request_a};
  constexpr std::array<std::uint64_t, 5> global_rules{
      0x1101u, 0x2202u, 0x3303u, 0x4404u, 0x5505u};

  std::vector<std::uint64_t> first_partition;
  if (rank == 0) {
    first_partition = {global_rules[0], global_rules[2], global_rules[4]};
  } else {
    first_partition = {global_rules[1], global_rules[3]};
  }
  const auto comm = svmp::MeshComm(MPI_COMM_WORLD);
  const auto first_fingerprint =
      collectivePartitionIndependentCutTopologyFingerprint(
          request_order, first_partition, comm);
  EXPECT_NE(first_fingerprint, 0u);
  EXPECT_EQ(
      collectivePartitionIndependentCutTopologyFingerprint(
          request_order, global_rules, svmp::MeshComm::self()),
      first_fingerprint);

  std::vector<std::uint64_t> redistributed_partition;
  if (rank == 0) {
    redistributed_partition = {global_rules[3], global_rules[0]};
  } else {
    redistributed_partition = {
        global_rules[4], global_rules[2], global_rules[1]};
  }
  EXPECT_EQ(
      collectivePartitionIndependentCutTopologyFingerprint(
          request_order, redistributed_partition, comm),
      first_fingerprint);

  // Concentrating every owned identity on one rank emulates another valid
  // ownership partition without changing the global semantic topology.
  std::vector<std::uint64_t> concentrated_partition;
  if (rank == 0) {
    concentrated_partition.assign(global_rules.rbegin(), global_rules.rend());
  }
  EXPECT_EQ(
      collectivePartitionIndependentCutTopologyFingerprint(
          request_order, concentrated_partition, comm),
      first_fingerprint);

  auto changed_partition = redistributed_partition;
  if (rank == 0) {
    changed_partition.front() ^= 0x8000000000000000ull;
  }
  EXPECT_NE(
      collectivePartitionIndependentCutTopologyFingerprint(
          request_order, changed_partition, comm),
      first_fingerprint);

  const auto [minimum_fingerprint, maximum_fingerprint] =
      globalMinMaxUint64(first_fingerprint, comm);
  EXPECT_EQ(minimum_fingerprint, maximum_fingerprint);
}

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     ActiveCutTopologyFingerprintCoordinatesRankLocalPreparationFailure)
{
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    GTEST_SKIP()
        << "This topology-fingerprint fixture requires exactly two ranks.";
  }

  std::vector<ActiveCutTopologySnapshotBinding> bindings;
  if (rank == 0) {
    bindings.push_back(ActiveCutTopologySnapshotBinding{});
  }
  EXPECT_THROW(
      activeCutContextTopologyFingerprint(
          bindings, svmp::MeshComm(MPI_COMM_WORLD)),
      std::runtime_error);
}

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
     StaticCapillaryActiveSupportUnionIsCollectiveAndBounded)
{
  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  ASSERT_EQ(world_size, 2);

  const svmp::MeshComm comm(MPI_COMM_WORLD);
  const auto gathered = communicatorWideIndexUnion(
      {std::size_t{2}, static_cast<std::size_t>(rank), std::size_t{2}},
      /*upper_bound=*/4u,
      comm,
      "static-capillary MPI test support");
  EXPECT_EQ(
      gathered,
      (std::vector<std::size_t>{0u, 1u, 2u}));

  const std::vector<std::size_t> invalid =
      rank == 1 ? std::vector<std::size_t>{4u}
                : std::vector<std::size_t>{0u};
  EXPECT_THROW(
      (void)communicatorWideIndexUnion(
          invalid,
          /*upper_bound=*/4u,
          comm,
          "static-capillary MPI invalid support"),
      std::runtime_error);
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
  const auto solved = gatherFeOrderedSolution(
      history.u(), svmp::MeshComm(MPI_COMM_WORLD));
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
     StaticCapillaryPublicationIsCollectiveWithExactSyntheticCertificate)
{
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    GTEST_SKIP()
        << "This static-capillary publication fixture requires two ranks.";
  }

  constexpr int interface_marker = 723;
  auto mesh = makePartitionedQuadStrip();
  ASSERT_GT(mesh->n_ghost_vertices(), 0u);
  const auto mesh_field = svmp::MeshFields::attach_field(
      mesh->local_mesh(),
      svmp::EntityKind::Vertex,
      "phi",
      svmp::FieldScalarType::Float64,
      1);
  ASSERT_NE(
      svmp::MeshFields::field_data_as<svmp::real_t>(
          mesh->local_mesh(), mesh_field),
      nullptr);

  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/1);
  auto velocity_space =
      svmp::FE::spaces::SpaceFactory::create_vector_h1(
          svmp::FE::ElementType::Quad4,
          /*order=*/1,
          /*components=*/2);
  auto system =
      std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(
      svmp::FE::systems::FieldSpec{
          .name = "phi",
          .space = scalar_space,
          .components = 1});
  const auto velocity = system->addField(
      svmp::FE::systems::FieldSpec{
          .name = "synthetic_velocity",
          .space = velocity_space,
          .components = 2});
  const auto pressure = system->addField(
      svmp::FE::systems::FieldSpec{
          .name = "synthetic_pressure",
          .space = scalar_space,
          .components = 1});

  svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalParameters
      functional_parameters;
  functional_parameters.liquid_side =
      svmp::FE::geometry::CutIntegrationSide::Negative;
  functional_parameters.surface_tension = 1.0;
  system->declareFreeSurfaceDiscreteFunctional(
      svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration{
          .interface_marker = interface_marker,
          .level_set_field = phi,
          .velocity_field = velocity,
          .geometry_domain_id = "mpi_static_capillary_publication",
          .parameters = functional_parameters,
          .active_volume_energy_parameters =
              svmp::FE::interfaces::
                  FreeSurfaceActiveVolumeEnergyParameters{
                      .liquid_side =
                          svmp::FE::geometry::
                              CutIntegrationSide::Negative,
                      .density = 1.0,
                      .gravitational_acceleration =
                          {{0.0, 0.0, 0.0}},
                      .gravitational_reference_point =
                          {{0.0, 0.0, 0.0}},
                  },
          .static_conservative_body_force_complete = true,
          .capillary_balance_method =
              svmp::FE::systems::
                  FreeSurfaceCapillaryBalanceMethod::
                      DiscreteEnergyVolumeStationarity,
          .capillary_balance_qualification =
              svmp::FE::systems::
                  FreeSurfaceCapillaryBalanceQualification::
                      PrerequisiteOnly,
          .owner_component =
              "ApplicationDriverLevelSetWorkflowsMPI.StaticCapillaryPublication",
      });

  system->addOperator("equations");
  for (const auto field : {phi, velocity, pressure}) {
    system->addCellKernel(
        "equations",
        field,
        field,
        std::make_shared<MpiWorkflowScaledMassKernel>(
            /*matrix_scale=*/1.0,
            /*vector_scale=*/0.0));
  }
  // The full-domain mass pair isolates collective publication mechanics. It
  // is not a physical static-cap or cut-pressure qualification result.
  installMpiWorkflowExactConstantPressureCertificate(
      *system, velocity, pressure);

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
  ASSERT_NO_THROW(system->setup(setup_options));
  ASSERT_TRUE(system->dofPermutation());

  const auto solution_size = static_cast<std::size_t>(
      system->dofHandler().getNumDofs());
  std::vector<svmp::FE::Real> local_solution(
      solution_size, 0.0);
  std::vector<svmp::FE::Real> solution(
      solution_size, 0.0);
  const auto& phi_dofs = system->fieldDofHandler(phi);
  const auto* phi_entity_map = phi_dofs.getEntityDofMap();
  ASSERT_NE(phi_entity_map, nullptr);
  const auto phi_offset = system->fieldDofOffset(phi);
  ASSERT_GE(phi_offset, 0);
  const auto& coordinates = mesh->X_ref();
  for (std::size_t vertex = 0u;
       vertex < mesh->n_vertices();
       ++vertex) {
    const auto dofs = phi_entity_map->getVertexDofs(
        static_cast<svmp::FE::GlobalIndex>(vertex));
    ASSERT_EQ(dofs.size(), 1u);
    const auto dof = dofs.front();
    ASSERT_GE(dof, 0);
    if (!phi_dofs.getDofMap().isOwnedDof(dof)) {
      continue;
    }
    const auto index = static_cast<std::size_t>(
        phi_offset + dof);
    ASSERT_LT(index, local_solution.size());
    local_solution[index] =
        static_cast<svmp::FE::Real>(
            coordinates[2u * vertex + 1u]) -
        svmp::FE::Real{0.5};
  }
  MPI_Allreduce(
      local_solution.data(),
      solution.data(),
      static_cast<int>(solution.size()),
      MPI_DOUBLE,
      MPI_SUM,
      MPI_COMM_WORLD);

  auto previous = solution;
  auto older = solution;
  const auto velocity_offset =
      system->fieldDofOffset(velocity);
  const auto velocity_count =
      system->fieldDofHandler(velocity).getNumDofs();
  const auto pressure_offset =
      system->fieldDofOffset(pressure);
  const auto pressure_count =
      system->fieldDofHandler(pressure).getNumDofs();
  for (svmp::FE::GlobalIndex i = 0;
       i < phi_dofs.getNumDofs();
       ++i) {
    previous[static_cast<std::size_t>(phi_offset + i)] +=
        svmp::FE::Real{1.0};
    older[static_cast<std::size_t>(phi_offset + i)] +=
        svmp::FE::Real{2.0};
  }
  for (svmp::FE::GlobalIndex i = 0;
       i < velocity_count;
       ++i) {
    previous[
        static_cast<std::size_t>(velocity_offset + i)] =
        svmp::FE::Real{3.0};
    older[
        static_cast<std::size_t>(velocity_offset + i)] =
        svmp::FE::Real{4.0};
  }
  for (svmp::FE::GlobalIndex i = 0;
       i < pressure_count;
       ++i) {
    previous[
        static_cast<std::size_t>(pressure_offset + i)] =
        svmp::FE::Real{5.0};
    older[
        static_cast<std::size_t>(pressure_offset + i)] =
        svmp::FE::Real{6.0};
  }
  const auto previous_velocity_revision =
      collectiveFreeSurfaceFieldRevision(
          *system,
          velocity,
          previous,
          activeFESystemCommunicator(*system),
          "MPI previous velocity revision");
  auto unrelated_fields_changed = previous;
  for (svmp::FE::GlobalIndex i = 0;
       i < phi_dofs.getNumDofs();
       ++i) {
    unrelated_fields_changed[
        static_cast<std::size_t>(phi_offset + i)] +=
        svmp::FE::Real{7.0};
  }
  for (svmp::FE::GlobalIndex i = 0;
       i < pressure_count;
       ++i) {
    unrelated_fields_changed[
        static_cast<std::size_t>(pressure_offset + i)] +=
        svmp::FE::Real{8.0};
  }
  EXPECT_EQ(
      collectiveFreeSurfaceFieldRevision(
          *system,
          velocity,
          unrelated_fields_changed,
          activeFESystemCommunicator(*system),
          "MPI unchanged velocity revision"),
      previous_velocity_revision);
  EXPECT_NE(
      collectiveFreeSurfaceFieldRevision(
          *system,
          velocity,
          older,
          activeFESystemCommunicator(*system),
          "MPI changed velocity revision"),
      previous_velocity_revision);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);
  sim.backend =
      std::make_unique<svmp::FE::backends::FsilsFactory>(
          /*dofs_per_node=*/4,
          sim.fe_system->dofPermutation(),
          MPI_COMM_WORLD);
  ASSERT_NE(sim.backend, nullptr);
  const auto* distributed_equations =
      sim.fe_system->distributedSparsityIfAvailable("equations");
  ASSERT_NE(distributed_equations, nullptr);
  auto backend_layout_matrix =
      sim.backend->createMatrix(*distributed_equations);
  ASSERT_NE(backend_layout_matrix, nullptr);
  svmp::FE::backends::SolverOptions linear_options;
  linear_options.method =
      svmp::FE::backends::SolverMethod::GMRES;
  linear_options.preconditioner =
      svmp::FE::backends::PreconditionerType::Diagonal;
  linear_options.rel_tol = 1.0e-12;
  linear_options.abs_tol = 1.0e-13;
  linear_options.max_iter = 500;
  sim.linear_solver =
      sim.backend->createLinearSolver(linear_options);
  ASSERT_NE(sim.linear_solver, nullptr);

  auto allocated_history =
      svmp::FE::timestepping::TimeHistory::allocate(
          *sim.backend,
          sim.fe_system->dofHandler().getNumDofs(),
          /*history_depth=*/2,
          /*allocate_second_order_state=*/false);
  sim.time_history =
      std::make_unique<
          svmp::FE::timestepping::TimeHistory>(
          std::move(allocated_history));
  sim.time_history->setTime(0.2);
  sim.time_history->setDt(0.1);
  sim.time_history->setPrevDt(0.1);
  scatterFeOrderedSolution(
      sim.time_history->u(), solution);
  scatterFeOrderedSolution(
      sim.time_history->uPrev(), previous);
  scatterFeOrderedSolution(
      sim.time_history->uPrev2(), older);
  const auto owned_rows =
      sim.time_history->u().ownedGlobalRows();
  ASSERT_FALSE(owned_rows.empty());
  EXPECT_LT(owned_rows.size(), solution_size);
  const auto local_owned_row_count =
      static_cast<unsigned long long>(owned_rows.size());
  unsigned long long global_owned_row_count = 0u;
  ASSERT_EQ(
      MPI_Allreduce(&local_owned_row_count,
                    &global_owned_row_count,
                    1,
                    MPI_UNSIGNED_LONG_LONG,
                    MPI_SUM,
                    MPI_COMM_WORLD),
      MPI_SUCCESS);
  EXPECT_EQ(global_owned_row_count,
            static_cast<unsigned long long>(solution_size));

  auto params = parseMpiWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Level_set_field_name>phi</Level_set_field_name>
    <Enable_static_capillary_equilibrium_initialization>true</Enable_static_capillary_equilibrium_initialization>
    <Static_capillary_projected_gradient_tolerance>1.0e100</Static_capillary_projected_gradient_tolerance>
  </Add_equation>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>mpi_static_capillary_publication</Generated_interface_domain_id>
      <Interface_marker>723</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
  auto requests = levelSetMaintenanceRequests(*params);
  ASSERT_EQ(requests.size(), 1u);

  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle
      lifecycle;
  ActiveCutContextRefreshCache refresh_cache;
  const auto initial_report =
      refreshActiveCutIntegrationContextCached(
          sim,
          *params,
          sim.time_history->u(),
          lifecycle,
          refresh_cache,
          "application-driver-mpi-static-capillary-publication-initial");
  ASSERT_TRUE(initial_report.refreshed);
  ASSERT_NE(initial_report.topology_key, 0u);

  const auto expected_pressure_certificate =
      evaluateStaticCapillaryPressureCertificate(
          sim,
          solution,
          requests.front().static_capillary_equilibrium,
          /*initialize_compatible_pressure=*/true);
  ASSERT_TRUE(
      expected_pressure_certificate.report
          .static_compatible_pressure_initializer_passed)
      << expected_pressure_certificate.report
             .static_compatible_pressure_initializer_reason;
  const auto& expected_solution =
      expected_pressure_certificate.certified_solution;
  ASSERT_EQ(expected_solution.size(), solution.size());
  svmp::FE::Real maximum_pressure_update = 0.0;
  for (svmp::FE::GlobalIndex i = 0; i < pressure_count; ++i) {
    const auto index = static_cast<std::size_t>(pressure_offset + i);
    maximum_pressure_update =
        std::max(maximum_pressure_update,
                 std::abs(expected_solution[index] - solution[index]));
  }
  EXPECT_GT(maximum_pressure_update, svmp::FE::Real{0.0});

  bool initialized = false;
  ASSERT_NO_THROW(
      initialized =
          initializeDiscreteStaticCapillaryEquilibrium(
              sim,
              *params,
              requests,
              lifecycle,
              refresh_cache));
  ASSERT_TRUE(initialized);
  ASSERT_TRUE(
      requests.front().static_capillary_equilibrium_initialized);
  EXPECT_NEAR(
      requests.front()
          .static_capillary_equilibrium
          .target_liquid_volume,
      4.0,
      1.0e-12);
  EXPECT_FALSE(
      sim.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle.transactionActive());

  auto expected_previous = previous;
  auto expected_older = older;
  std::copy(
      expected_solution.begin() +
          static_cast<std::ptrdiff_t>(phi_offset),
      expected_solution.begin() +
          static_cast<std::ptrdiff_t>(
              phi_offset + phi_dofs.getNumDofs()),
      expected_previous.begin() +
          static_cast<std::ptrdiff_t>(phi_offset));
  std::copy(
      expected_solution.begin() +
          static_cast<std::ptrdiff_t>(phi_offset),
      expected_solution.begin() +
          static_cast<std::ptrdiff_t>(
              phi_offset + phi_dofs.getNumDofs()),
      expected_older.begin() +
          static_cast<std::ptrdiff_t>(phi_offset));
  const auto final_solution =
      captureFeOrderedVectorCollectively(
          sim.time_history->u(),
          activeFESystemCommunicator(*sim.fe_system));
  EXPECT_EQ(final_solution, expected_solution);
  EXPECT_EQ(
      captureFeOrderedVectorCollectively(
          sim.time_history->uPrev(),
          activeFESystemCommunicator(*sim.fe_system)),
      expected_previous);
  EXPECT_EQ(
      captureFeOrderedVectorCollectively(
          sim.time_history->uPrev2(),
          activeFESystemCommunicator(*sim.fe_system)),
      expected_older);

  const auto final_revision =
      collectiveLevelSetMaintenanceAlgebraicRevision(
          final_solution,
          svmp::MeshComm(MPI_COMM_WORLD));
  const auto [minimum_revision, maximum_revision] =
      globalMinMaxUint64(
          final_revision,
          svmp::MeshComm(MPI_COMM_WORLD));
  EXPECT_EQ(minimum_revision, maximum_revision);
  ASSERT_TRUE(refresh_cache.topology_key.has_value());
  const auto [minimum_topology, maximum_topology] =
      globalMinMaxUint64(
          *refresh_cache.topology_key,
          svmp::MeshComm(MPI_COMM_WORLD));
  EXPECT_EQ(minimum_topology, maximum_topology);
}

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     PhysicalFlatCapillaryEquilibriumMatchesAcrossTwoRankPartition)
{
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    GTEST_SKIP()
        << "This physical flat-capillary fixture requires two ranks.";
  }

  constexpr int interface_marker = 724;
  constexpr int first_wall_marker = 7241;
  constexpr int second_wall_marker = 7242;
  constexpr int lower_anchor_marker = 7243;
  constexpr int upper_anchor_marker = 7244;
  constexpr svmp::FE::Real pi =
      svmp::FE::Real{3.141592653589793238462643383279502884};
  constexpr svmp::FE::Real contact_angle =
      pi / svmp::FE::Real{2.0};
  MpiWorkflowScopedEnvVar conservative_balance_diagnostic(
      "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC",
      std::string("1"));

  constexpr std::array<svmp::FE::Real, 3> normal_offsets{
      svmp::FE::Real{0.35},
      svmp::FE::Real{0.5},
      svmp::FE::Real{0.65},
  };
  std::size_t case_count = 0u;
  std::size_t area_gradient_case_count = 0u;
  svmp::FE::Real maximum_kkt_residual = 0.0;
  svmp::FE::Real maximum_kkt_relative_distance = 0.0;
  svmp::FE::Real maximum_pressure_jump_error = 0.0;
  svmp::FE::Real maximum_volume_error = 0.0;
  svmp::FE::Real maximum_surface_energy_error = 0.0;
  svmp::FE::Real maximum_phi_update_across_cases = 0.0;

  for (int normal_axis = 0; normal_axis < 2; ++normal_axis) {
    const int tangent_axis = 1 - normal_axis;
    for (const bool positive_side : {false, true}) {
      for (const auto normal_offset : normal_offsets) {
        SCOPED_TRACE(::testing::Message()
                     << "rank=" << rank
                     << " normal_axis=" << normal_axis
                     << " active_side="
                     << (positive_side ? "positive" : "negative")
                     << " normal_offset=" << normal_offset);
        ++case_count;

        const bool run_area_gradient_case =
            normal_axis == 0 && !positive_side &&
            normal_offset == svmp::FE::Real{0.5};
        const std::array<
            channel_ns::FreeSurfaceSurfaceTensionForm, 2>
            surface_tension_forms{{
                channel_ns::FreeSurfaceSurfaceTensionForm::SurfaceStress,
                channel_ns::FreeSurfaceSurfaceTensionForm::
                    KinematicAreaGradientTraction,
            }};
        const std::size_t surface_tension_form_count =
            run_area_gradient_case ? 2u : 1u;
        for (std::size_t surface_tension_form_index = 0u;
             surface_tension_form_index < surface_tension_form_count;
             ++surface_tension_form_index) {
          const auto surface_tension_form =
              surface_tension_forms[surface_tension_form_index];
          const bool uses_area_gradient_traction =
              surface_tension_form ==
              channel_ns::FreeSurfaceSurfaceTensionForm::
                  KinematicAreaGradientTraction;
          area_gradient_case_count +=
              uses_area_gradient_traction ? 1u : 0u;
          SCOPED_TRACE(::testing::Message()
                       << "surface_tension_form="
                       << (uses_area_gradient_traction
                               ? "KinematicAreaGradientTraction"
                               : "SurfaceStress"));

        auto mesh =
            makePartitionedFlatCapillaryFanMesh(normal_axis);
        ASSERT_GT(mesh->n_ghost_vertices(), 0u);
        auto& local_mesh = mesh->local_mesh();
        std::array<int, 4> local_marker_present{};
        constexpr svmp::FE::Real coordinate_tolerance = 1.0e-12;
        for (const auto face : local_mesh.boundary_faces()) {
          const auto vertices = local_mesh.face_vertices(face);
          ASSERT_EQ(vertices.size(), 2u);
          bool on_first_wall = true;
          bool on_second_wall = true;
          bool on_lower_anchor = true;
          bool on_upper_anchor = true;
          for (const auto vertex : vertices) {
            const auto point = local_mesh.get_vertex_coords(vertex);
            on_first_wall =
                on_first_wall &&
                std::abs(point[tangent_axis]) <=
                    coordinate_tolerance;
            on_second_wall =
                on_second_wall &&
                std::abs(point[tangent_axis] -
                         svmp::FE::Real{3.0}) <=
                    coordinate_tolerance;
            on_lower_anchor =
                on_lower_anchor &&
                std::abs(point[normal_axis]) <=
                    coordinate_tolerance;
            on_upper_anchor =
                on_upper_anchor &&
                std::abs(point[normal_axis] -
                         svmp::FE::Real{1.0}) <=
                    coordinate_tolerance;
          }
          if (on_first_wall) {
            mesh->set_boundary_label(face, first_wall_marker);
            local_marker_present[0] = 1;
          } else if (on_second_wall) {
            mesh->set_boundary_label(face, second_wall_marker);
            local_marker_present[1] = 1;
          } else if (on_lower_anchor) {
            mesh->set_boundary_label(face, lower_anchor_marker);
            local_marker_present[2] = 1;
          } else if (on_upper_anchor) {
            mesh->set_boundary_label(face, upper_anchor_marker);
            local_marker_present[3] = 1;
          } else {
            FAIL()
                << "Distributed flat-capillary fixture found an unclassified face.";
          }
        }
        std::array<int, 4> global_marker_present{};
        ASSERT_EQ(
            MPI_Allreduce(local_marker_present.data(),
                          global_marker_present.data(),
                          static_cast<int>(local_marker_present.size()),
                          MPI_INT,
                          MPI_MAX,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);
        for (const auto present : global_marker_present) {
          EXPECT_EQ(present, 1);
        }

        const auto mesh_field =
            svmp::MeshFields::attach_field(
                local_mesh,
                svmp::EntityKind::Vertex,
                "phi_physical_flat_mpi",
                svmp::FieldScalarType::Float64,
                1);
        ASSERT_NE(
            svmp::MeshFields::field_data_as<svmp::real_t>(
                local_mesh, mesh_field),
            nullptr);

        auto scalar_space =
            svmp::FE::spaces::SpaceFactory::create_h1(
                svmp::FE::ElementType::Triangle3,
                /*order=*/1);
        auto velocity_space =
            svmp::FE::spaces::SpaceFactory::create_vector_h1(
                svmp::FE::ElementType::Triangle3,
                /*order=*/1,
                /*components=*/2);
        auto system =
            std::make_unique<svmp::FE::systems::FESystem>(mesh);
        const auto phi = system->addField(
            svmp::FE::systems::FieldSpec{
                .name = "phi_physical_flat_mpi",
                .space = scalar_space,
                .components = 1});
        svmp::FE::FieldId kappa = svmp::FE::INVALID_FIELD_ID;
        if (uses_area_gradient_traction) {
          kappa = system->addField(
              svmp::FE::systems::FieldSpec{
                  .name = "kappa_physical_flat_mpi",
                  .space = scalar_space,
                  .components = 1,
                  .source_kind = svmp::FE::systems::
                      FieldSourceKind::PrescribedData,
              });
        }

        channel_ns::IncompressibleNavierStokesVMSOptions options;
        options.velocity_field_name = "u_physical_flat_mpi";
        options.pressure_field_name = "p_physical_flat_mpi";
        options.density = 1.0;
        options.viscosity = 0.01;
        options.enable_convection = false;
        options.enable_vms = false;
        options.jit_policy.enable = false;
        options.velocity_dirichlet.push_back(
            channel_ns::IncompressibleNavierStokesVMSOptions::
                VelocityDirichletBC{
                    .boundary_marker = lower_anchor_marker,
                    .value = {0.0, 0.0, 0.0},
                });
        options.velocity_dirichlet.push_back(
            channel_ns::IncompressibleNavierStokesVMSOptions::
                VelocityDirichletBC{
                    .boundary_marker = upper_anchor_marker,
                    .value = {0.0, 0.0, 0.0},
                });
        options.velocity_dirichlet.push_back(
            channel_ns::IncompressibleNavierStokesVMSOptions::
                VelocityDirichletBC{
                    .boundary_marker = first_wall_marker,
                    .value = {0.0, 0.0, 0.0},
                    .active_components = {tangent_axis == 0,
                                          tangent_axis == 1,
                                          false},
                });
        options.velocity_dirichlet.push_back(
            channel_ns::IncompressibleNavierStokesVMSOptions::
                VelocityDirichletBC{
                    .boundary_marker = second_wall_marker,
                    .value = {0.0, 0.0, 0.0},
                    .active_components = {tangent_axis == 0,
                                          tangent_axis == 1,
                                          false},
                });

        using ContactLine =
            channel_ns::IncompressibleNavierStokesVMSOptions::
                FreeSurfaceContactLine;
        auto free_surface =
            channel_ns::IncompressibleNavierStokesVMSOptions::
                FreeSurfaceBoundary{
                    .implementation =
                        channel_ns::FreeSurfaceImplementation::
                            UnfittedLevelSet,
                    .interface_marker = interface_marker,
                    .level_set_field_name =
                        "phi_physical_flat_mpi",
                    .generated_interface_domain_id =
                        "physical_flat_capillary_mpi",
                    .generated_interface_geometry = "LinearCorner",
                    .active_domain =
                        positive_side
                            ? channel_ns::FreeSurfaceActiveDomain::
                                  LevelSetPositive
                            : channel_ns::FreeSurfaceActiveDomain::
                                  LevelSetNegative,
                    .active_domain_method =
                        channel_ns::FreeSurfaceActiveDomainMethod::
                            CutVolume,
                    .external_pressure = 0.0,
                    .surface_tension = 1.0,
                    .surface_tension_form = surface_tension_form,
                    .curvature = 0.0,
                    .curvature_field_name =
                        uses_area_gradient_traction
                            ? "kappa_physical_flat_mpi"
                            : "",
                    .use_level_set_curvature = false,
                    .small_cut_aggregation = false,
                };
        free_surface.contact_lines.push_back(
            ContactLine{
                .configuration = ContactLine::DynamicRenE{
                    .wall_boundary_marker = first_wall_marker,
                    .contact_line_marker = -1,
                    .equilibrium_contact_angle_radians =
                        contact_angle,
                    .wall_normal = {
                        tangent_axis == 0 ? -1.0 : 0.0,
                        tangent_axis == 1 ? -1.0 : 0.0,
                        0.0},
                    .mobility = 1.0,
                    .slip_length = 1.0,
                }});
        free_surface.contact_lines.push_back(
            ContactLine{
                .configuration = ContactLine::DynamicRenE{
                    .wall_boundary_marker = second_wall_marker,
                    .contact_line_marker = -1,
                    .equilibrium_contact_angle_radians =
                        contact_angle,
                    .wall_normal = {
                        tangent_axis == 0 ? 1.0 : 0.0,
                        tangent_axis == 1 ? 1.0 : 0.0,
                        0.0},
                    .mobility = 1.0,
                    .slip_length = 1.0,
                }});
        options.free_surface.push_back(std::move(free_surface));

        channel_ns::IncompressibleNavierStokesVMSModule module(
            velocity_space, scalar_space, std::move(options));
        module.registerOn(*system);
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
        ASSERT_NO_THROW(system->setup(setup_options));
        ASSERT_TRUE(system->dofPermutation());
        std::vector<svmp::FE::Real> curvature_sentinel;
        std::uint64_t curvature_revision_before = 0u;
        if (uses_area_gradient_traction) {
          const auto curvature_count = static_cast<std::size_t>(
              system->fieldDofHandler(kappa).getNumDofs());
          curvature_sentinel.assign(
              curvature_count, svmp::FE::Real{8.0});
          system->setPrescribedFieldCoefficients(
              kappa, curvature_sentinel);
          curvature_revision_before =
              system->prescribedFieldRevision(kappa);
        }

        const auto velocity =
            system->findFieldByName("u_physical_flat_mpi");
        const auto pressure =
            system->findFieldByName("p_physical_flat_mpi");
        ASSERT_NE(velocity, svmp::FE::INVALID_FIELD_ID);
        ASSERT_NE(pressure, svmp::FE::INVALID_FIELD_ID);

        const auto solution_size = static_cast<std::size_t>(
            system->dofHandler().getNumDofs());
        ASSERT_LE(
            solution_size,
            static_cast<std::size_t>(
                std::numeric_limits<int>::max()));
        std::vector<svmp::FE::Real> local_solution(
            solution_size, 0.0);
        std::vector<svmp::FE::Real> current(
            solution_size, 0.0);
        const auto& phi_dofs = system->fieldDofHandler(phi);
        const auto* phi_entity_map = phi_dofs.getEntityDofMap();
        ASSERT_NE(phi_entity_map, nullptr);
        const auto phi_offset = system->fieldDofOffset(phi);
        ASSERT_GE(phi_offset, 0);
        const auto& coordinates = mesh->X_ref();
        for (std::size_t vertex = 0u;
             vertex < mesh->n_vertices();
             ++vertex) {
          const auto dofs = phi_entity_map->getVertexDofs(
              static_cast<svmp::FE::GlobalIndex>(vertex));
          ASSERT_EQ(dofs.size(), 1u);
          const auto dof = dofs.front();
          ASSERT_GE(dof, 0);
          if (!phi_dofs.getDofMap().isOwnedDof(dof)) {
            continue;
          }
          const auto index = static_cast<std::size_t>(
              phi_offset + dof);
          ASSERT_LT(index, local_solution.size());
          const auto signed_coordinate =
              static_cast<svmp::FE::Real>(
                  coordinates[2u * vertex +
                              static_cast<std::size_t>(normal_axis)]) -
              normal_offset;
          local_solution[index] =
              positive_side ? -signed_coordinate
                            : signed_coordinate;
        }
        ASSERT_EQ(
            MPI_Allreduce(local_solution.data(),
                          current.data(),
                          static_cast<int>(current.size()),
                          MPI_DOUBLE,
                          MPI_SUM,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);

        application::core::SimulationComponents sim;
        sim.primary_mesh = mesh;
        sim.fe_system = std::move(system);
        sim.backend =
            std::make_unique<svmp::FE::backends::FsilsFactory>(
                /*dofs_per_node=*/4,
                sim.fe_system->dofPermutation(),
                MPI_COMM_WORLD);
        ASSERT_NE(sim.backend, nullptr);
        const auto* distributed_equations =
            sim.fe_system->distributedSparsityIfAvailable("equations");
        ASSERT_NE(distributed_equations, nullptr);
        auto backend_layout_matrix =
            sim.backend->createMatrix(*distributed_equations);
        ASSERT_NE(backend_layout_matrix, nullptr);
        svmp::FE::backends::SolverOptions linear_options;
        linear_options.method =
            svmp::FE::backends::SolverMethod::GMRES;
        linear_options.preconditioner =
            svmp::FE::backends::PreconditionerType::Diagonal;
        linear_options.rel_tol = 1.0e-12;
        linear_options.abs_tol = 1.0e-13;
        linear_options.max_iter = 500;
        sim.linear_solver =
            sim.backend->createLinearSolver(linear_options);
        ASSERT_NE(sim.linear_solver, nullptr);

        auto allocated_history =
            svmp::FE::timestepping::TimeHistory::allocate(
                *sim.backend,
                sim.fe_system->dofHandler().getNumDofs(),
                /*history_depth=*/2,
                /*allocate_second_order_state=*/true);
        sim.time_history =
            std::make_unique<
                svmp::FE::timestepping::TimeHistory>(
                std::move(allocated_history));
        sim.time_history->setTime(0.0);
        sim.time_history->setDt(0.1);
        sim.time_history->setPrevDt(0.1);
        scatterFeOrderedSolution(sim.time_history->u(), current);
        scatterFeOrderedSolution(sim.time_history->uPrev(), current);
        scatterFeOrderedSolution(sim.time_history->uPrev2(), current);
        sim.time_history->uDot().zero();
        sim.time_history->uDDot().zero();
        sim.time_history->updateGhosts();
        const auto owned_rows =
            sim.time_history->u().ownedGlobalRows();
        ASSERT_FALSE(owned_rows.empty());
        EXPECT_LT(owned_rows.size(), solution_size);
        const auto local_owned_row_count =
            static_cast<unsigned long long>(owned_rows.size());
        unsigned long long global_owned_row_count = 0u;
        ASSERT_EQ(
            MPI_Allreduce(&local_owned_row_count,
                          &global_owned_row_count,
                          1,
                          MPI_UNSIGNED_LONG_LONG,
                          MPI_SUM,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);
        EXPECT_EQ(global_owned_row_count,
                  static_cast<unsigned long long>(solution_size));

        const char* active_domain_name =
            positive_side ? "LevelSetPositive"
                          : "LevelSetNegative";
        const char* contact_wall_normals =
            tangent_axis == 0
                ? "-1.0 0.0 0.0; 1.0 0.0 0.0"
                : "0.0 -1.0 0.0; 0.0 1.0 0.0";
        const std::string level_set_curvature_parameters =
            uses_area_gradient_traction
                ? R"xml(
    <Enable_curvature_projection>true</Enable_curvature_projection>
    <Curvature_field_name>kappa_physical_flat_mpi</Curvature_field_name>
    <Curvature_projection_recovery_mode>KinematicAreaGradient</Curvature_projection_recovery_mode>
    <Curvature_projection_kinematic_area_gradient_filter_coefficient>0.0</Curvature_projection_kinematic_area_gradient_filter_coefficient>)xml"
                : std::string{};
        const char* surface_tension_form_name =
            uses_area_gradient_traction
                ? "KinematicAreaGradientTraction"
                : "SurfaceStress";
        const std::string traction_curvature_parameters =
            uses_area_gradient_traction
                ? R"xml(
      <Curvature_field_name>kappa_physical_flat_mpi</Curvature_field_name>
      <Use_level_set_curvature>false</Use_level_set_curvature>)xml"
                : std::string{};
        const std::string parameter_xml =
            std::string(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Level_set_field_name>phi_physical_flat_mpi</Level_set_field_name>
    <Enable_static_capillary_equilibrium_initialization>true</Enable_static_capillary_equilibrium_initialization>
    <Static_capillary_volume_tolerance>1.0e-11</Static_capillary_volume_tolerance>
    <Static_capillary_projected_gradient_tolerance>2.0e-6</Static_capillary_projected_gradient_tolerance>
    <Static_capillary_constant_pressure_kkt_max_residual_norm>2.0e-10</Static_capillary_constant_pressure_kkt_max_residual_norm>
    <Static_capillary_constant_pressure_kkt_max_relative_distance>2.0e-10</Static_capillary_constant_pressure_kkt_max_relative_distance>)xml") +
            level_set_curvature_parameters + R"xml(
  </Add_equation>
  <Add_equation type="fluid">
    <Add_BC name="physical_flat_capillary_mpi">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi_physical_flat_mpi</Level_set_field_name>
      <Generated_interface_domain_id>physical_flat_capillary_mpi</Generated_interface_domain_id>
      <Interface_marker>724</Interface_marker>
      <Generated_interface_geometry>LinearCorner</Generated_interface_geometry>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>)xml" + active_domain_name +
            R"xml(</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
      <Small_cut_aggregation>false</Small_cut_aggregation>
      <Surface_tension>1.0</Surface_tension>
      <Surface_tension_form>)xml" + surface_tension_form_name +
            R"xml(</Surface_tension_form>)xml" +
            traction_curvature_parameters + R"xml(
      <Contact_line_model>DynamicContactAngle</Contact_line_model>
      <Contact_angle_degrees>90.0</Contact_angle_degrees>
      <Contact_line_wall_markers>7241;7242</Contact_line_wall_markers>
      <Contact_line_wall_normals>)xml" + contact_wall_normals +
            R"xml(</Contact_line_wall_normals>
      <Contact_line_mobility>1.0</Contact_line_mobility>
      <Wall_slip_model>Navier</Wall_slip_model>
      <Wall_slip_length>1.0</Wall_slip_length>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml";
        auto params =
            parseMpiWorkflowParametersXml(parameter_xml.c_str());
        auto requests = levelSetMaintenanceRequests(*params);
        ASSERT_EQ(requests.size(), 1u);
        ASSERT_TRUE(
            requests.front().static_capillary_equilibrium_enabled);
        EXPECT_EQ(
            requests.front().curvature_projection_enabled,
            uses_area_gradient_traction);
        if (uses_area_gradient_traction) {
          EXPECT_EQ(
              requests.front().curvature_projection.recovery_mode,
              svmp::FE::level_set::LevelSetCurvatureRecoveryMode::
                  KinematicAreaGradient);
        }

        svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle
            lifecycle;
        ActiveCutContextRefreshCache refresh_cache;
        const auto initial_report =
            refreshActiveCutIntegrationContextCached(
                sim,
                *params,
                sim.time_history->u(),
                lifecycle,
                refresh_cache,
                "application-driver-mpi-physical-flat-initial");
        ASSERT_TRUE(initial_report.refreshed);
        ASSERT_NE(initial_report.topology_key, 0u);
        const auto initial_functionals =
            evaluateCurrentFreeSurfaceDiscreteFunctionals(sim);
        ASSERT_EQ(initial_functionals.size(), 1u);
        const auto expected_volume =
            svmp::FE::Real{3.0} * normal_offset;
        EXPECT_NEAR(
            initial_functionals.front().state.owned_liquid_volume,
            expected_volume,
            1.0e-13);
        EXPECT_NEAR(
            initial_functionals.front().state
                .liquid_gas_surface_energy,
            svmp::FE::Real{3.0},
            1.0e-13);
        EXPECT_NEAR(
            initial_functionals.front().state.young_wall_energy,
            svmp::FE::Real{0.0},
            1.0e-13);

        bool initialized = false;
        ASSERT_NO_THROW(
            initialized =
                initializeDiscreteStaticCapillaryEquilibrium(
                    sim,
                    *params,
                    requests,
                    lifecycle,
                    refresh_cache));
        ASSERT_TRUE(initialized);
        ASSERT_TRUE(
            requests.front()
                .static_capillary_equilibrium_initialized);
        if (uses_area_gradient_traction) {
          const auto projected_curvature =
              sim.fe_system->prescribedFieldCoefficients(kappa);
          ASSERT_EQ(
              projected_curvature.size(),
              curvature_sentinel.size());
          EXPECT_TRUE(std::all_of(
              projected_curvature.begin(),
              projected_curvature.end(),
              [](svmp::FE::Real value) {
                return std::isfinite(value);
              }));
          EXPECT_TRUE(std::all_of(
              projected_curvature.begin(),
              projected_curvature.end(),
              [](svmp::FE::Real value) {
                return std::abs(value) <=
                       svmp::FE::Real{1.0e-12};
              }));
          EXPECT_NE(
              std::vector<svmp::FE::Real>(
                  projected_curvature.begin(),
                  projected_curvature.end()),
              curvature_sentinel);
          EXPECT_GT(
              sim.fe_system->prescribedFieldRevision(kappa),
              curvature_revision_before);
        }

        const auto certified_solution =
            capturePostacceptMaintenanceVectorCollectively(
                sim.time_history->u(),
                activeFESystemCommunicator(*sim.fe_system));
        const auto pressure_certificate =
            evaluateStaticCapillaryPressureCertificate(
                sim,
                certified_solution,
                requests.front().static_capillary_equilibrium,
                /*initialize_compatible_pressure=*/false);
        const auto& certificate = pressure_certificate.report;
        ASSERT_TRUE(
            certificate
                .pressure_representability_diagnostic_sampled);
        ASSERT_TRUE(certificate.constant_pressure_kkt_available)
            << certificate.constant_pressure_kkt_reason;
        EXPECT_LE(
            certificate.constant_pressure_kkt_residual_norm,
            2.0e-10);
        EXPECT_LE(
            certificate.constant_pressure_kkt_relative_distance,
            2.0e-10);
        EXPECT_NEAR(
            certificate.constant_pressure_kkt_pressure_jump,
            0.0,
            2.0e-10);

        const auto final_functionals =
            evaluateCurrentFreeSurfaceDiscreteFunctionals(sim);
        ASSERT_EQ(final_functionals.size(), 1u);
        const auto volume_error = std::abs(
            final_functionals.front().state.owned_liquid_volume -
            expected_volume);
        const auto surface_energy_error = std::abs(
            final_functionals.front().state
                .liquid_gas_surface_energy -
            svmp::FE::Real{3.0});
        EXPECT_LE(volume_error, 1.0e-11);
        EXPECT_LE(surface_energy_error, 2.0e-10);
        EXPECT_NEAR(
            final_functionals.front().state.young_wall_energy,
            svmp::FE::Real{0.0},
            1.0e-13);

        const auto field_count = static_cast<std::size_t>(
            sim.fe_system->fieldDofHandler(phi).getNumDofs());
        svmp::FE::Real maximum_phi_update = 0.0;
        for (std::size_t i = 0u; i < field_count; ++i) {
          maximum_phi_update = std::max(
              maximum_phi_update,
              std::abs(
                  certified_solution[
                      static_cast<std::size_t>(phi_offset) + i] -
                  current[
                      static_cast<std::size_t>(phi_offset) + i]));
        }
        EXPECT_LE(maximum_phi_update, 2.0e-7);

        const auto communicator =
            activeFESystemCommunicator(*sim.fe_system);
        for (const auto scalar : {
                 certificate.constant_pressure_kkt_residual_norm,
                 certificate.constant_pressure_kkt_relative_distance,
                 certificate.constant_pressure_kkt_pressure_jump,
                 static_cast<double>(volume_error),
                 static_cast<double>(surface_energy_error),
                 static_cast<double>(maximum_phi_update)}) {
          EXPECT_EQ(globalMinDouble(scalar, communicator),
                    globalMaxDouble(scalar, communicator));
        }
        const auto final_revision =
            collectiveLevelSetMaintenanceAlgebraicRevision(
                certified_solution, communicator);
        const auto [minimum_revision, maximum_revision] =
            globalMinMaxUint64(final_revision, communicator);
        EXPECT_EQ(minimum_revision, maximum_revision);

        maximum_kkt_residual = std::max(
            maximum_kkt_residual,
            static_cast<svmp::FE::Real>(
                certificate.constant_pressure_kkt_residual_norm));
        maximum_kkt_relative_distance = std::max(
            maximum_kkt_relative_distance,
            static_cast<svmp::FE::Real>(
                certificate
                    .constant_pressure_kkt_relative_distance));
        maximum_pressure_jump_error = std::max(
            maximum_pressure_jump_error,
            static_cast<svmp::FE::Real>(std::abs(
                certificate.constant_pressure_kkt_pressure_jump)));
        maximum_volume_error =
            std::max(maximum_volume_error, volume_error);
        maximum_surface_energy_error = std::max(
            maximum_surface_energy_error, surface_energy_error);
        maximum_phi_update_across_cases = std::max(
            maximum_phi_update_across_cases,
            maximum_phi_update);
        }
      }
    }
  }

  EXPECT_EQ(case_count, 12u);
  EXPECT_EQ(area_gradient_case_count, 1u);
  RecordProperty("wp4_physical_flat_mpi_rank_count", size);
  RecordProperty("wp4_physical_flat_mpi_partition_layout_count", 1);
  RecordProperty(
      "wp4_physical_flat_mpi_coordinate_direction_count", 2);
  RecordProperty(
      "wp4_physical_flat_mpi_wall_orientation_count", 2);
  RecordProperty(
      "wp4_physical_flat_mpi_active_side_count", 2);
  RecordProperty("wp4_physical_flat_mpi_cut_offset_count", 3);
  RecordProperty("wp4_physical_flat_mpi_matrix_case_count",
                 case_count);
  RecordProperty("wp4_area_gradient_static_mpi_case_count",
                 area_gradient_case_count);
  RecordProperty(
      "wp4_physical_flat_mpi_constant_pressure_kkt_residual_norm",
      maximum_kkt_residual);
  RecordProperty(
      "wp4_physical_flat_mpi_constant_pressure_kkt_relative_distance",
      maximum_kkt_relative_distance);
  RecordProperty(
      "wp4_physical_flat_mpi_pressure_jump_absolute_error",
      maximum_pressure_jump_error);
  RecordProperty("wp4_physical_flat_mpi_volume_error",
                 maximum_volume_error);
  RecordProperty("wp4_physical_flat_mpi_surface_energy_error",
                 maximum_surface_energy_error);
  RecordProperty("wp4_physical_flat_mpi_maximum_phi_update",
                 maximum_phi_update_across_cases);
}

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     HydrostaticGravityWithFixedPressureGaugeMatchesAcrossTwoRankPartition)
{
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    GTEST_SKIP()
        << "This hydrostatic fixed-gauge fixture requires two ranks.";
  }

  constexpr int interface_marker = 725;
  constexpr int first_wall_marker = 7251;
  constexpr int second_wall_marker = 7252;
  constexpr int lower_anchor_marker = 7253;
  constexpr int upper_anchor_marker = 7254;
  constexpr int third_wall_marker = 7255;
  constexpr int fourth_wall_marker = 7256;
  constexpr svmp::FE::Real density = 1.25;
  constexpr svmp::FE::Real gravity_magnitude = 0.4;
  constexpr svmp::FE::Real pi =
      svmp::FE::Real{3.141592653589793238462643383279502884};
  constexpr svmp::FE::Real contact_angle =
      pi / svmp::FE::Real{2.0};
  MpiWorkflowScopedEnvVar conservative_balance_diagnostic(
      "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC",
      std::string("1"));

  constexpr std::array<svmp::FE::Real, 3> normal_offsets{
      svmp::FE::Real{0.35},
      svmp::FE::Real{0.5},
      svmp::FE::Real{0.65},
  };
  constexpr std::size_t partition_layout_count = 2u;
  constexpr std::size_t vertex_numbering_count = 2u;
  constexpr std::size_t normal_axis_count = 2u;
  constexpr std::size_t three_dimensional_normal_axis_count = 3u;
  constexpr std::size_t dof_ownership_strategy_count = 2u;
  constexpr std::size_t fe_global_numbering_mode_count = 2u;
  std::size_t case_count = 0u;
  std::size_t two_dimensional_case_count = 0u;
  std::size_t three_dimensional_case_count = 0u;
  std::size_t owner_contiguous_nonidentity_case_count = 0u;
  std::size_t
      three_dimensional_owner_contiguous_nonidentity_case_count = 0u;
  std::size_t three_dimensional_shared_vertex_case_count = 0u;
  svmp::FE::Real maximum_pressure_residual = 0.0;
  svmp::FE::Real maximum_pressure_relative_distance = 0.0;
  svmp::FE::Real maximum_exact_field_production_residual = 0.0;
  svmp::FE::Real maximum_production_residual = 0.0;
  svmp::FE::Real maximum_initializer_pressure_representative_distance = 0.0;
  svmp::FE::Real maximum_exact_initializer_pressure_update = 0.0;
  svmp::FE::Real maximum_gravitational_energy_error = 0.0;
  svmp::FE::Real maximum_volume_error = 0.0;
  svmp::FE::Real maximum_surface_energy_error = 0.0;
  svmp::FE::Real maximum_phi_update = 0.0;

  struct HydrostaticMeshVariant {
    int spatial_dimension = 2;
    int normal_axis = 0;
    bool reverse_vertex_numbering = false;
    bool alternate_cell_order = false;
    bool highest_rank_dof_ownership = false;
    bool dense_global_dof_numbering = false;
  };
  std::vector<HydrostaticMeshVariant> mesh_variants;
  const auto two_dimensional_mesh_variant_count =
      partition_layout_count * vertex_numbering_count * normal_axis_count *
      dof_ownership_strategy_count * fe_global_numbering_mode_count;
  const auto three_dimensional_mesh_variant_count =
      partition_layout_count * vertex_numbering_count *
      three_dimensional_normal_axis_count * dof_ownership_strategy_count *
      fe_global_numbering_mode_count;
  mesh_variants.reserve(two_dimensional_mesh_variant_count +
                        three_dimensional_mesh_variant_count);
  for (std::size_t mesh_variant = 0u;
       mesh_variant < two_dimensional_mesh_variant_count;
       ++mesh_variant) {
    mesh_variants.push_back(HydrostaticMeshVariant{
        .spatial_dimension = 2,
        .normal_axis = static_cast<int>(mesh_variant % normal_axis_count),
        .reverse_vertex_numbering =
            ((mesh_variant / normal_axis_count) % vertex_numbering_count) !=
            0u,
        .alternate_cell_order =
            ((mesh_variant /
              (normal_axis_count * vertex_numbering_count)) %
             partition_layout_count) != 0u,
        .highest_rank_dof_ownership =
            ((mesh_variant /
              (normal_axis_count * vertex_numbering_count *
               partition_layout_count)) %
             dof_ownership_strategy_count) != 0u,
        .dense_global_dof_numbering =
            (mesh_variant /
             (normal_axis_count * vertex_numbering_count *
              partition_layout_count * dof_ownership_strategy_count)) != 0u,
    });
  }
  for (std::size_t mesh_variant = 0u;
       mesh_variant < three_dimensional_mesh_variant_count;
       ++mesh_variant) {
    mesh_variants.push_back(HydrostaticMeshVariant{
        .spatial_dimension = 3,
        .normal_axis = static_cast<int>(
            mesh_variant % three_dimensional_normal_axis_count),
        .reverse_vertex_numbering =
            ((mesh_variant / three_dimensional_normal_axis_count) %
             vertex_numbering_count) != 0u,
        .alternate_cell_order =
            ((mesh_variant /
              (three_dimensional_normal_axis_count *
               vertex_numbering_count)) %
             partition_layout_count) != 0u,
        .highest_rank_dof_ownership =
            ((mesh_variant /
              (three_dimensional_normal_axis_count *
               vertex_numbering_count * partition_layout_count)) %
             dof_ownership_strategy_count) != 0u,
        .dense_global_dof_numbering =
            (mesh_variant /
             (three_dimensional_normal_axis_count *
              vertex_numbering_count * partition_layout_count *
              dof_ownership_strategy_count)) != 0u,
    });
  }

  for (const auto& mesh_variant : mesh_variants) {
    const int spatial_dimension = mesh_variant.spatial_dimension;
    const int normal_axis = mesh_variant.normal_axis;
    const bool reverse_vertex_numbering =
        mesh_variant.reverse_vertex_numbering;
    const bool column_major_cells = mesh_variant.alternate_cell_order;
    const bool highest_rank_dof_ownership =
        mesh_variant.highest_rank_dof_ownership;
    const bool dense_global_dof_numbering =
        mesh_variant.dense_global_dof_numbering;
    std::array<int, 2> tangent_axes{};
    std::size_t tangent_axis_count = 0u;
    for (int axis = 0; axis < spatial_dimension; ++axis) {
      if (axis != normal_axis) {
        tangent_axes[tangent_axis_count++] = axis;
      }
    }
    ASSERT_EQ(tangent_axis_count,
              static_cast<std::size_t>(spatial_dimension - 1));
    const int tangent_axis = tangent_axes.front();
    for (const bool positive_side : {false, true}) {
      const auto gauge_normal_coordinate =
          positive_side ? svmp::FE::Real{1.0} : svmp::FE::Real{0.0};
      for (const auto normal_offset : normal_offsets) {
        for (const svmp::FE::Real gravity_direction :
             {svmp::FE::Real{-1.0}, svmp::FE::Real{1.0}}) {
          const auto gravity = gravity_direction * gravity_magnitude;
          const auto external_pressure =
              density * gravity *
              (normal_offset - gauge_normal_coordinate);
          SCOPED_TRACE(::testing::Message()
                       << "rank=" << rank
                       << " spatial_dimension=" << spatial_dimension
                       << " cell_order="
                       << (column_major_cells ? "column-major"
                                              : "row-major")
                       << " vertex_numbering="
                       << (reverse_vertex_numbering ? "reversed"
                                                    : "forward")
                       << " dof_ownership="
                       << (highest_rank_dof_ownership ? "highest-rank"
                                                      : "lowest-rank")
                       << " fe_global_numbering="
                       << (dense_global_dof_numbering ? "dense-global-ids"
                                                      : "owner-contiguous")
                       << " normal_axis=" << normal_axis
                       << " active_side="
                       << (positive_side ? "positive" : "negative")
                       << " normal_offset=" << normal_offset
                       << " gravity=" << gravity
                       << " external_pressure=" << external_pressure);
          ++case_count;
          if (spatial_dimension == 2) {
            ++two_dimensional_case_count;
          } else {
            ++three_dimensional_case_count;
          }

          auto mesh = spatial_dimension == 2
                          ? makePartitionedHydrostaticPressureMesh(
                                normal_axis,
                                column_major_cells,
                                reverse_vertex_numbering)
                          : makePartitionedHydrostaticPressureMesh3D(
                                normal_axis,
                                column_major_cells,
                                reverse_vertex_numbering);
          ASSERT_GT(mesh->n_ghost_vertices(), 0u);
          auto& local_mesh = mesh->local_mesh();
          unsigned long long local_owned_cell_count = 0u;
          for (std::size_t cell = 0u;
               cell < mesh->n_cells();
               ++cell) {
            const auto local_cell = static_cast<svmp::index_t>(cell);
            if (mesh->owner_rank_cell(local_cell) != rank) {
              continue;
            }
            ++local_owned_cell_count;
          }
          std::array<unsigned long long, 2> owned_cell_counts{};
          ASSERT_EQ(MPI_Allgather(&local_owned_cell_count,
                                  1,
                                  MPI_UNSIGNED_LONG_LONG,
                                  owned_cell_counts.data(),
                                  1,
                                  MPI_UNSIGNED_LONG_LONG,
                                  MPI_COMM_WORLD),
                    MPI_SUCCESS);
          const auto expected_owned_cell_count =
              spatial_dimension == 2 ? 16u : 48u;
          EXPECT_EQ(owned_cell_counts[0], expected_owned_cell_count);
          EXPECT_EQ(owned_cell_counts[1], expected_owned_cell_count);

          if (spatial_dimension == 2) {
            std::array<int, 2> local_probe_counts{};
            std::array<int, 2> local_probe_owner_plus_one{};
            for (std::size_t cell = 0u;
                 cell < mesh->n_cells();
                 ++cell) {
              const auto local_cell = static_cast<svmp::index_t>(cell);
              if (mesh->owner_rank_cell(local_cell) != rank) {
                continue;
              }
              const auto center = local_mesh.cell_center(local_cell);
              const auto logical_tangent_coordinate =
                  normal_axis == 0
                      ? svmp::FE::Real{3.0} - center[tangent_axis]
                      : center[tangent_axis];
              const bool lower_right =
                  logical_tangent_coordinate > svmp::FE::Real{2.25} &&
                  center[normal_axis] < svmp::FE::Real{0.2};
              const bool upper_left =
                  logical_tangent_coordinate < svmp::FE::Real{0.81} &&
                  center[normal_axis] > svmp::FE::Real{0.7};
              if (lower_right) {
                ++local_probe_counts[0];
                local_probe_owner_plus_one[0] = rank + 1;
              }
              if (upper_left) {
                ++local_probe_counts[1];
                local_probe_owner_plus_one[1] = rank + 1;
              }
            }
            std::array<int, 2> global_probe_counts{};
            std::array<int, 2> global_probe_owner_plus_one{};
            ASSERT_EQ(
                MPI_Allreduce(local_probe_counts.data(),
                              global_probe_counts.data(),
                              static_cast<int>(local_probe_counts.size()),
                              MPI_INT,
                              MPI_SUM,
                              MPI_COMM_WORLD),
                MPI_SUCCESS);
            ASSERT_EQ(
                MPI_Allreduce(
                    local_probe_owner_plus_one.data(),
                    global_probe_owner_plus_one.data(),
                    static_cast<int>(local_probe_owner_plus_one.size()),
                    MPI_INT,
                    MPI_MAX,
                    MPI_COMM_WORLD),
                MPI_SUCCESS);
            EXPECT_EQ(global_probe_counts,
                      (std::array<int, 2>{2, 2}));
            EXPECT_EQ(
                global_probe_owner_plus_one,
                (column_major_cells ? std::array<int, 2>{2, 1}
                                    : std::array<int, 2>{1, 2}));
          }

          struct ContactWall {
            int marker = -1;
            int axis = 0;
            svmp::FE::Real coordinate = 0.0;
            svmp::FE::Real outward_normal = 0.0;
          };
          const std::array<int, 4> wall_markers{{
              first_wall_marker,
              second_wall_marker,
              third_wall_marker,
              fourth_wall_marker,
          }};
          std::vector<ContactWall> contact_walls;
          contact_walls.reserve(2u * tangent_axis_count);
          for (std::size_t tangent = 0u;
               tangent < tangent_axis_count;
               ++tangent) {
            const auto axis = tangent_axes[tangent];
            const auto maximum_coordinate =
                spatial_dimension == 2 || axis == (normal_axis + 1) % 3
                    ? svmp::FE::Real{3.0}
                    : svmp::FE::Real{2.0};
            contact_walls.push_back(ContactWall{
                .marker = wall_markers[2u * tangent],
                .axis = axis,
                .coordinate = 0.0,
                .outward_normal = -1.0,
            });
            contact_walls.push_back(ContactWall{
                .marker = wall_markers[2u * tangent + 1u],
                .axis = axis,
                .coordinate = maximum_coordinate,
                .outward_normal = 1.0,
            });
          }
          ASSERT_EQ(contact_walls.size(),
                    static_cast<std::size_t>(
                        2 * (spatial_dimension - 1)));

          std::array<int, 6> local_marker_present{};
          constexpr svmp::FE::Real coordinate_tolerance = 1.0e-12;
          const auto physical_boundary_faces =
              svmp::DistributedTopology::global_boundary_faces(
                  *mesh, /*owned_only=*/false);
          for (const auto face : physical_boundary_faces) {
            const auto vertices = local_mesh.face_vertices(face);
            ASSERT_EQ(vertices.size(),
                      static_cast<std::size_t>(spatial_dimension));
            std::array<bool, 4> on_contact_wall{{true, true, true, true}};
            bool on_lower_anchor = true;
            bool on_upper_anchor = true;
            for (const auto vertex : vertices) {
              const auto point = local_mesh.get_vertex_coords(vertex);
              for (std::size_t wall = 0u;
                   wall < contact_walls.size();
                   ++wall) {
                on_contact_wall[wall] =
                    on_contact_wall[wall] &&
                    std::abs(point[contact_walls[wall].axis] -
                             contact_walls[wall].coordinate) <=
                        coordinate_tolerance;
              }
              on_lower_anchor =
                  on_lower_anchor &&
                  std::abs(point[normal_axis]) <= coordinate_tolerance;
              on_upper_anchor =
                  on_upper_anchor &&
                  std::abs(point[normal_axis] - svmp::FE::Real{1.0}) <=
                      coordinate_tolerance;
            }
            bool classified = false;
            for (std::size_t wall = 0u;
                 wall < contact_walls.size();
                 ++wall) {
              if (on_contact_wall[wall]) {
                mesh->set_boundary_label(face, contact_walls[wall].marker);
                local_marker_present[wall] = 1;
                classified = true;
                break;
              }
            }
            if (classified) {
              continue;
            }
            if (on_lower_anchor) {
              mesh->set_boundary_label(face, lower_anchor_marker);
              local_marker_present[4] = 1;
            } else if (on_upper_anchor) {
              mesh->set_boundary_label(face, upper_anchor_marker);
              local_marker_present[5] = 1;
            } else {
              FAIL() << "Distributed hydrostatic fixture found an "
                        "unclassified physical boundary face.";
            }
          }
          std::array<int, 6> global_marker_present{};
          ASSERT_EQ(
              MPI_Allreduce(local_marker_present.data(),
                            global_marker_present.data(),
                            static_cast<int>(local_marker_present.size()),
                            MPI_INT,
                            MPI_MAX,
                            MPI_COMM_WORLD),
              MPI_SUCCESS);
          for (std::size_t wall = 0u;
               wall < contact_walls.size();
               ++wall) {
            EXPECT_EQ(global_marker_present[wall], 1);
          }
          EXPECT_EQ(global_marker_present[4], 1);
          EXPECT_EQ(global_marker_present[5], 1);

          std::array<unsigned long long, 6> local_marker_face_counts{};
          const auto owned_physical_boundary_faces =
              svmp::DistributedTopology::global_boundary_faces(
                  *mesh, /*owned_only=*/true);
          for (const auto face : owned_physical_boundary_faces) {
            const auto marker = mesh->boundary_label(face);
            for (std::size_t wall = 0u;
                 wall < contact_walls.size();
                 ++wall) {
              if (marker == contact_walls[wall].marker) {
                ++local_marker_face_counts[wall];
              }
            }
            if (marker == lower_anchor_marker) {
              ++local_marker_face_counts[4];
            } else if (marker == upper_anchor_marker) {
              ++local_marker_face_counts[5];
            }
          }
          std::array<unsigned long long, 6> global_marker_face_counts{};
          ASSERT_EQ(
              MPI_Allreduce(local_marker_face_counts.data(),
                            global_marker_face_counts.data(),
                            static_cast<int>(
                                local_marker_face_counts.size()),
                            MPI_UNSIGNED_LONG_LONG,
                            MPI_SUM,
                            MPI_COMM_WORLD),
              MPI_SUCCESS);
          const auto expected_contact_wall_face_count =
              spatial_dimension == 2 ? 4u : 16u;
          const auto expected_anchor_face_count =
              spatial_dimension == 2 ? 4u : 8u;
          for (std::size_t wall = 0u;
               wall < contact_walls.size();
               ++wall) {
            EXPECT_EQ(global_marker_face_counts[wall],
                      expected_contact_wall_face_count);
          }
          EXPECT_EQ(global_marker_face_counts[4],
                    expected_anchor_face_count);
          EXPECT_EQ(global_marker_face_counts[5],
                    expected_anchor_face_count);

          const auto& vertex_gids = local_mesh.vertex_gids();
          ASSERT_EQ(vertex_gids.size(), mesh->n_vertices());
          auto local_max_vertex_gid = svmp::gid_t{-1};
          for (const auto gid : vertex_gids) {
            local_max_vertex_gid = std::max(local_max_vertex_gid, gid);
          }
          svmp::gid_t global_max_vertex_gid = svmp::gid_t{-1};
          ASSERT_EQ(MPI_Allreduce(&local_max_vertex_gid,
                                  &global_max_vertex_gid,
                                  1,
                                  MPI_INT64_T,
                                  MPI_MAX,
                                  MPI_COMM_WORLD),
                    MPI_SUCCESS);
          ASSERT_GE(global_max_vertex_gid, svmp::gid_t{0});
          const auto global_vertex_count =
              static_cast<std::size_t>(global_max_vertex_gid + 1);
          std::vector<int> local_owned_cell_vertex_adjacency(
              global_vertex_count, 0);
          for (std::size_t cell = 0u; cell < mesh->n_cells(); ++cell) {
            const auto local_cell = static_cast<svmp::index_t>(cell);
            if (mesh->owner_rank_cell(local_cell) != rank) {
              continue;
            }
            for (const auto vertex : local_mesh.cell_vertices(local_cell)) {
              ASSERT_GE(vertex, svmp::index_t{0});
              ASSERT_LT(static_cast<std::size_t>(vertex),
                        vertex_gids.size());
              const auto gid =
                  vertex_gids[static_cast<std::size_t>(vertex)];
              ASSERT_GE(gid, svmp::gid_t{0});
              ASSERT_LT(static_cast<std::size_t>(gid),
                        global_vertex_count);
              local_owned_cell_vertex_adjacency[
                  static_cast<std::size_t>(gid)] = 1;
            }
          }
          std::vector<int> global_owned_cell_vertex_adjacency(
              global_vertex_count, 0);
          ASSERT_LE(global_vertex_count,
                    static_cast<std::size_t>(
                        std::numeric_limits<int>::max()));
          ASSERT_EQ(MPI_Allreduce(
                        local_owned_cell_vertex_adjacency.data(),
                        global_owned_cell_vertex_adjacency.data(),
                        static_cast<int>(global_vertex_count),
                        MPI_INT,
                        MPI_SUM,
                        MPI_COMM_WORLD),
                    MPI_SUCCESS);
          auto shared_vertex_gid = svmp::gid_t{-1};
          for (std::size_t gid = 0u;
               gid < global_owned_cell_vertex_adjacency.size();
               ++gid) {
            if (global_owned_cell_vertex_adjacency[gid] == size) {
              shared_vertex_gid = static_cast<svmp::gid_t>(gid);
              break;
            }
          }
          ASSERT_GE(shared_vertex_gid, svmp::gid_t{0});
          auto local_gauge_gid = std::numeric_limits<svmp::gid_t>::max();
          for (std::size_t vertex = 0u;
               vertex < mesh->n_vertices();
               ++vertex) {
            const auto point = local_mesh.get_vertex_coords(
                static_cast<svmp::index_t>(vertex));
            if (std::abs(point[normal_axis] - gauge_normal_coordinate) <=
                coordinate_tolerance) {
              local_gauge_gid =
                  std::min(local_gauge_gid, vertex_gids[vertex]);
            }
          }
          svmp::gid_t gauge_gid =
              std::numeric_limits<svmp::gid_t>::max();
          ASSERT_EQ(MPI_Allreduce(&local_gauge_gid,
                                  &gauge_gid,
                                  1,
                                  MPI_INT64_T,
                                  MPI_MIN,
                                  MPI_COMM_WORLD),
                    MPI_SUCCESS);
          ASSERT_NE(gauge_gid, std::numeric_limits<svmp::gid_t>::max());
          ASSERT_GE(gauge_gid, svmp::gid_t{0});
          const auto upper_gauge_layer_first_gid =
              spatial_dimension == 2 ? 20 : 36;
          const auto expected_gauge_gid = static_cast<svmp::gid_t>(
              reverse_vertex_numbering
                  ? (positive_side ? 0 : upper_gauge_layer_first_gid)
                  : (positive_side ? upper_gauge_layer_first_gid : 0));
          EXPECT_EQ(gauge_gid, expected_gauge_gid);

          const auto mesh_field = svmp::MeshFields::attach_field(
              local_mesh,
              svmp::EntityKind::Vertex,
              "phi_physical_hydrostatic_fixed_gauge_mpi",
              svmp::FieldScalarType::Float64,
              1);
          auto* mesh_phi =
              svmp::MeshFields::field_data_as<svmp::real_t>(
                  local_mesh, mesh_field);
          ASSERT_NE(mesh_phi, nullptr);
          const auto& coordinates = mesh->X_ref();
          ASSERT_EQ(coordinates.size(),
                    static_cast<std::size_t>(spatial_dimension) *
                        mesh->n_vertices());
          for (std::size_t vertex = 0u;
               vertex < mesh->n_vertices();
               ++vertex) {
            mesh_phi[vertex] =
                coordinates[static_cast<std::size_t>(spatial_dimension) *
                                vertex +
                            static_cast<std::size_t>(normal_axis)] -
                normal_offset;
          }

          const auto element_type =
              spatial_dimension == 2
                  ? svmp::FE::ElementType::Triangle3
                  : svmp::FE::ElementType::Tetra4;
          auto scalar_space =
              svmp::FE::spaces::SpaceFactory::create_h1(
                  element_type,
                  /*order=*/1);
          auto velocity_space =
              svmp::FE::spaces::SpaceFactory::create_vector_h1(
                  element_type,
                  /*order=*/1,
                  /*components=*/spatial_dimension);
          auto system =
              std::make_unique<svmp::FE::systems::FESystem>(mesh);
          const auto phi = system->addField(
              svmp::FE::systems::FieldSpec{
                  .name = "phi_physical_hydrostatic_fixed_gauge_mpi",
                  .space = scalar_space,
                  .components = 1});

          channel_ns::IncompressibleNavierStokesVMSOptions options;
          options.velocity_field_name =
              "u_physical_hydrostatic_fixed_gauge_mpi";
          options.pressure_field_name =
              "p_physical_hydrostatic_fixed_gauge_mpi";
          options.density = density;
          options.viscosity = 0.01;
          options.body_force[normal_axis] = gravity;
          options.enable_convection = false;
          options.enable_vms = false;
          options.jit_policy.enable = false;
          options.velocity_dirichlet.push_back(
              channel_ns::IncompressibleNavierStokesVMSOptions::
                  VelocityDirichletBC{
                      .boundary_marker =
                          positive_side ? upper_anchor_marker
                                        : lower_anchor_marker,
                      .value = {0.0, 0.0, 0.0},
                  });
          for (const auto& wall : contact_walls) {
            options.velocity_dirichlet.push_back(
                channel_ns::IncompressibleNavierStokesVMSOptions::
                    VelocityDirichletBC{
                        .boundary_marker = wall.marker,
                        .value = {0.0, 0.0, 0.0},
                        .active_components = {wall.axis == 0,
                                              wall.axis == 1,
                                              wall.axis == 2},
                    });
          }
          options.node_pressure_constraints.id_type =
              channel_ns::IncompressibleNavierStokesVMSOptions::
                  NodePressureConstraintIdType::GlobalVertexGid;
          options.node_pressure_constraints.values.push_back(
              channel_ns::IncompressibleNavierStokesVMSOptions::
                  NodePressureConstraint{
                      .node_id =
                          static_cast<svmp::FE::GlobalIndex>(gauge_gid),
                      .pressure = 0.0,
                  });

          using ContactLine =
              channel_ns::IncompressibleNavierStokesVMSOptions::
                  FreeSurfaceContactLine;
          auto free_surface =
              channel_ns::IncompressibleNavierStokesVMSOptions::
                  FreeSurfaceBoundary{
                      .implementation =
                          channel_ns::FreeSurfaceImplementation::
                              UnfittedLevelSet,
                      .interface_marker = interface_marker,
                      .level_set_field_name =
                          "phi_physical_hydrostatic_fixed_gauge_mpi",
                      .generated_interface_domain_id =
                          "physical_hydrostatic_fixed_gauge_mpi",
                      .generated_interface_geometry = "LinearCorner",
                      .active_domain =
                          positive_side
                              ? channel_ns::FreeSurfaceActiveDomain::
                                    LevelSetPositive
                              : channel_ns::FreeSurfaceActiveDomain::
                                    LevelSetNegative,
                      .active_domain_method =
                          channel_ns::FreeSurfaceActiveDomainMethod::
                              CutVolume,
                      .external_pressure = external_pressure,
                      .surface_tension = 1.0,
                      .surface_tension_form =
                          channel_ns::FreeSurfaceSurfaceTensionForm::
                              SurfaceStress,
                      .curvature = 0.0,
                      .use_level_set_curvature = false,
                      .small_cut_aggregation = false,
          };
          for (const auto& wall : contact_walls) {
            free_surface.contact_lines.push_back(
                ContactLine{
                    .configuration = ContactLine::DynamicRenE{
                        .wall_boundary_marker = wall.marker,
                        .contact_line_marker = -1,
                        .equilibrium_contact_angle_radians = contact_angle,
                        .wall_normal = {
                            wall.axis == 0 ? wall.outward_normal : 0.0,
                            wall.axis == 1 ? wall.outward_normal : 0.0,
                            wall.axis == 2 ? wall.outward_normal : 0.0},
                        .mobility = 1.0,
                        .slip_length = 1.0,
                    }});
          }
          options.free_surface.push_back(std::move(free_surface));

          channel_ns::IncompressibleNavierStokesVMSModule module(
              velocity_space, scalar_space, std::move(options));
          module.registerOn(*system);
          const auto velocity = system->findFieldByName(
              "u_physical_hydrostatic_fixed_gauge_mpi");
          const auto pressure = system->findFieldByName(
              "p_physical_hydrostatic_fixed_gauge_mpi");
          ASSERT_NE(velocity, svmp::FE::INVALID_FIELD_ID);
          ASSERT_NE(pressure, svmp::FE::INVALID_FIELD_ID);

          svmp::FE::systems::SetupOptions setup_options;
          setup_options.assembler_name = "StandardAssembler";
          setup_options.assembly_options.ghost_policy =
              svmp::FE::assembly::GhostPolicy::ReverseScatter;
          setup_options.assembly_options.deterministic = true;
          setup_options.assembly_options.overlap_communication = false;
          setup_options.dof_options.global_numbering =
              dense_global_dof_numbering
                  ? svmp::FE::dofs::GlobalNumberingMode::DenseGlobalIds
                  : svmp::FE::dofs::GlobalNumberingMode::OwnerContiguous;
          setup_options.dof_options.ownership =
              highest_rank_dof_ownership
                  ? svmp::FE::dofs::OwnershipStrategy::HighestRank
                  : svmp::FE::dofs::OwnershipStrategy::LowestRank;
          setup_options.dof_options.my_rank = rank;
          setup_options.dof_options.world_size = size;
          setup_options.dof_options.mpi_comm = MPI_COMM_WORLD;
          setup_options.use_backend_row_ownership_for_assembly = true;
          setup_options.retain_serial_sparsity = false;
          ASSERT_NO_THROW(system->setup(setup_options));
          ASSERT_TRUE(system->dofPermutation());

          const auto solution_size = static_cast<std::size_t>(
              system->dofHandler().getNumDofs());
          ASSERT_LE(solution_size,
                    static_cast<std::size_t>(
                        std::numeric_limits<int>::max()));
          std::vector<svmp::FE::Real> local_current(solution_size, 0.0);
          std::vector<svmp::FE::Real> local_exact(solution_size, 0.0);
          std::vector<svmp::FE::Real> current(solution_size, 0.0);
          std::vector<svmp::FE::Real> exact_solution(solution_size, 0.0);
          const auto& phi_dofs = system->fieldDofHandler(phi);
          const auto* phi_entity_map = phi_dofs.getEntityDofMap();
          ASSERT_NE(phi_entity_map, nullptr);
          const auto& pressure_dofs = system->fieldDofHandler(pressure);
          const auto* pressure_entity_map =
              pressure_dofs.getEntityDofMap();
          ASSERT_NE(pressure_entity_map, nullptr);
          const auto phi_offset = system->fieldDofOffset(phi);
          const auto pressure_offset = system->fieldDofOffset(pressure);
          ASSERT_GE(phi_offset, 0);
          ASSERT_GE(pressure_offset, 0);
          std::size_t shared_ownership_probe_vertex_count = 0u;
          unsigned long long local_pressure_numbering_mismatch_count = 0u;
          for (std::size_t vertex = 0u;
               vertex < mesh->n_vertices();
               ++vertex) {
            const auto normal_coordinate =
                static_cast<svmp::FE::Real>(
                    coordinates[static_cast<std::size_t>(spatial_dimension) *
                                    vertex +
                                static_cast<std::size_t>(normal_axis)]);
            const auto signed_coordinate =
                normal_coordinate - normal_offset;
            const auto phi_vertex_dofs =
                phi_entity_map->getVertexDofs(
                    static_cast<svmp::FE::GlobalIndex>(vertex));
            ASSERT_EQ(phi_vertex_dofs.size(), 1u);
            const auto phi_dof = phi_vertex_dofs.front();
            ASSERT_GE(phi_dof, 0);
            if (phi_dofs.getDofMap().isOwnedDof(phi_dof)) {
              const auto index =
                  static_cast<std::size_t>(phi_offset + phi_dof);
              ASSERT_LT(index, solution_size);
              local_current[index] = signed_coordinate;
              local_exact[index] = signed_coordinate;
            }

            const auto pressure_vertex_dofs =
                pressure_entity_map->getVertexDofs(
                    static_cast<svmp::FE::GlobalIndex>(vertex));
            ASSERT_EQ(pressure_vertex_dofs.size(), 1u);
            const auto pressure_dof = pressure_vertex_dofs.front();
            ASSERT_GE(pressure_dof, 0);
            const auto pressure_vertex_gid = static_cast<svmp::FE::GlobalIndex>(
                vertex_gids[vertex]);
            if (dense_global_dof_numbering) {
              EXPECT_EQ(pressure_dof, pressure_vertex_gid);
            }
            if (pressure_dofs.getDofMap().isOwnedDof(pressure_dof) &&
                pressure_dof != pressure_vertex_gid) {
              ++local_pressure_numbering_mismatch_count;
            }
            bool is_ownership_probe_vertex =
                pressure_vertex_gid == shared_vertex_gid;
            if (spatial_dimension == 2) {
              const auto logical_tangent_coordinate =
                  normal_axis == 0
                      ? svmp::FE::Real{3.0} -
                            static_cast<svmp::FE::Real>(
                                coordinates[
                                    static_cast<std::size_t>(
                                        spatial_dimension) *
                                        vertex +
                                    static_cast<std::size_t>(tangent_axis)])
                      : static_cast<svmp::FE::Real>(
                            coordinates[
                                static_cast<std::size_t>(spatial_dimension) *
                                    vertex +
                                static_cast<std::size_t>(tangent_axis)]);
              is_ownership_probe_vertex =
                  std::abs(normal_coordinate - svmp::FE::Real{0.4}) <=
                      coordinate_tolerance &&
                  std::abs(logical_tangent_coordinate -
                           svmp::FE::Real{1.55}) <= coordinate_tolerance;
            }
            if (is_ownership_probe_vertex) {
              ++shared_ownership_probe_vertex_count;
              EXPECT_EQ(
                  pressure_dofs.getDofMap().getDofOwner(pressure_dof),
                  highest_rank_dof_ownership ? 1 : 0);
            }
            if (pressure_dofs.getDofMap().isOwnedDof(pressure_dof)) {
              const auto index =
                  static_cast<std::size_t>(pressure_offset + pressure_dof);
              ASSERT_LT(index, solution_size);
              local_exact[index] =
                  density * gravity *
                  (normal_coordinate - gauge_normal_coordinate);
            }
          }
          EXPECT_EQ(shared_ownership_probe_vertex_count, 1u);
          if (spatial_dimension == 3) {
            ++three_dimensional_shared_vertex_case_count;
          }
          unsigned long long pressure_numbering_mismatch_count = 0u;
          ASSERT_EQ(MPI_Allreduce(
                        &local_pressure_numbering_mismatch_count,
                        &pressure_numbering_mismatch_count,
                        1,
                        MPI_UNSIGNED_LONG_LONG,
                        MPI_SUM,
                        MPI_COMM_WORLD),
                    MPI_SUCCESS);
          if (dense_global_dof_numbering) {
            EXPECT_EQ(pressure_numbering_mismatch_count, 0u);
          } else if (pressure_numbering_mismatch_count > 0u) {
            ++owner_contiguous_nonidentity_case_count;
            if (spatial_dimension == 3) {
              ++three_dimensional_owner_contiguous_nonidentity_case_count;
            }
          }
          ASSERT_EQ(MPI_Allreduce(local_current.data(),
                                  current.data(),
                                  static_cast<int>(solution_size),
                                  MPI_DOUBLE,
                                  MPI_SUM,
                                  MPI_COMM_WORLD),
                    MPI_SUCCESS);
          ASSERT_EQ(MPI_Allreduce(local_exact.data(),
                                  exact_solution.data(),
                                  static_cast<int>(solution_size),
                                  MPI_DOUBLE,
                                  MPI_SUM,
                                  MPI_COMM_WORLD),
                    MPI_SUCCESS);
          system->updateConstraints(/*time=*/0.0, /*dt=*/0.1);
          system->constraints().distribute(current);
          system->constraints().distribute(exact_solution);
          std::fill(local_current.begin(), local_current.end(), 0.0);
          std::fill(local_exact.begin(), local_exact.end(), 0.0);
          for (const auto field : {phi, velocity, pressure}) {
            const auto field_offset = system->fieldDofOffset(field);
            const auto& field_dofs = system->fieldDofHandler(field);
            ASSERT_GE(field_offset, 0);
            ASSERT_GE(field_dofs.getNumDofs(), 0);
            for (svmp::FE::GlobalIndex dof = 0;
                 dof < field_dofs.getNumDofs();
                 ++dof) {
              if (!field_dofs.getDofMap().isOwnedDof(dof)) {
                continue;
              }
              const auto index =
                  static_cast<std::size_t>(field_offset + dof);
              ASSERT_LT(index, solution_size);
              local_current[index] = current[index];
              local_exact[index] = exact_solution[index];
            }
          }
          std::vector<svmp::FE::Real> constrained_current(
              solution_size, 0.0);
          std::vector<svmp::FE::Real> constrained_exact(
              solution_size, 0.0);
          ASSERT_EQ(MPI_Allreduce(local_current.data(),
                                  constrained_current.data(),
                                  static_cast<int>(solution_size),
                                  MPI_DOUBLE,
                                  MPI_SUM,
                                  MPI_COMM_WORLD),
                    MPI_SUCCESS);
          ASSERT_EQ(MPI_Allreduce(local_exact.data(),
                                  constrained_exact.data(),
                                  static_cast<int>(solution_size),
                                  MPI_DOUBLE,
                                  MPI_SUM,
                                  MPI_COMM_WORLD),
                    MPI_SUCCESS);
          current = std::move(constrained_current);
          exact_solution = std::move(constrained_exact);

          application::core::SimulationComponents sim;
          sim.primary_mesh = mesh;
          sim.fe_system = std::move(system);
          sim.backend =
              std::make_unique<svmp::FE::backends::FsilsFactory>(
                  /*dofs_per_node=*/spatial_dimension + 2,
                  sim.fe_system->dofPermutation(),
                  MPI_COMM_WORLD);
          ASSERT_NE(sim.backend, nullptr);
          const auto* distributed_equations =
              sim.fe_system->distributedSparsityIfAvailable("equations");
          ASSERT_NE(distributed_equations, nullptr);
          auto backend_layout_matrix =
              sim.backend->createMatrix(*distributed_equations);
          ASSERT_NE(backend_layout_matrix, nullptr);
          svmp::FE::backends::SolverOptions linear_options;
          linear_options.method =
              svmp::FE::backends::SolverMethod::GMRES;
          linear_options.preconditioner =
              svmp::FE::backends::PreconditionerType::Diagonal;
          linear_options.rel_tol = 1.0e-12;
          linear_options.abs_tol = 1.0e-13;
          linear_options.max_iter = 500;
          sim.linear_solver =
              sim.backend->createLinearSolver(linear_options);
          ASSERT_NE(sim.linear_solver, nullptr);

          auto allocated_history =
              svmp::FE::timestepping::TimeHistory::allocate(
                  *sim.backend,
                  sim.fe_system->dofHandler().getNumDofs(),
                  /*history_depth=*/2,
                  /*allocate_second_order_state=*/true);
          sim.time_history =
              std::make_unique<svmp::FE::timestepping::TimeHistory>(
                  std::move(allocated_history));
          sim.time_history->setTime(0.0);
          sim.time_history->setDt(0.1);
          sim.time_history->setPrevDt(0.1);
          scatterFeOrderedSolution(sim.time_history->u(), current);
          scatterFeOrderedSolution(sim.time_history->uPrev(), current);
          scatterFeOrderedSolution(sim.time_history->uPrev2(), current);
          sim.time_history->uDot().zero();
          sim.time_history->uDDot().zero();
          sim.time_history->updateGhosts();
          const auto owned_rows =
              sim.time_history->u().ownedGlobalRows();
          ASSERT_FALSE(owned_rows.empty());
          EXPECT_LT(owned_rows.size(), solution_size);
          const auto local_owned_row_count =
              static_cast<unsigned long long>(owned_rows.size());
          unsigned long long global_owned_row_count = 0u;
          ASSERT_EQ(MPI_Allreduce(&local_owned_row_count,
                                  &global_owned_row_count,
                                  1,
                                  MPI_UNSIGNED_LONG_LONG,
                                  MPI_SUM,
                                  MPI_COMM_WORLD),
                    MPI_SUCCESS);
          EXPECT_EQ(global_owned_row_count,
                    static_cast<unsigned long long>(solution_size));

          std::ostringstream contact_wall_markers_xml;
          std::ostringstream contact_wall_normals_xml;
          for (std::size_t wall_index = 0u;
               wall_index < contact_walls.size();
               ++wall_index) {
            if (wall_index > 0u) {
              contact_wall_markers_xml << ';';
              contact_wall_normals_xml << "; ";
            }
            const auto& wall = contact_walls[wall_index];
            contact_wall_markers_xml << wall.marker;
            for (int component = 0; component < 3; ++component) {
              if (component > 0) {
                contact_wall_normals_xml << ' ';
              }
              contact_wall_normals_xml
                  << (component == wall.axis ? wall.outward_normal : 0.0);
            }
          }
          std::ostringstream parameter_xml;
          parameter_xml << std::setprecision(17) << R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Level_set_field_name>phi_physical_hydrostatic_fixed_gauge_mpi</Level_set_field_name>
    <Enable_static_capillary_equilibrium_initialization>true</Enable_static_capillary_equilibrium_initialization>
    <Static_capillary_volume_tolerance>1.0e-11</Static_capillary_volume_tolerance>
    <Static_capillary_projected_gradient_tolerance>2.0e-6</Static_capillary_projected_gradient_tolerance>
    <Static_capillary_pressure_representability_max_residual_norm>2.0e-10</Static_capillary_pressure_representability_max_residual_norm>
    <Static_capillary_pressure_representability_max_relative_distance>2.0e-10</Static_capillary_pressure_representability_max_relative_distance>
    <Static_capillary_physical_equilibrium_max_residual_norm>2.0e-10</Static_capillary_physical_equilibrium_max_residual_norm>
    <Static_capillary_constant_pressure_kkt_max_residual_norm>2.0e-10</Static_capillary_constant_pressure_kkt_max_residual_norm>
    <Static_capillary_constant_pressure_kkt_max_relative_distance>2.0e-10</Static_capillary_constant_pressure_kkt_max_relative_distance>
  </Add_equation>
  <Add_equation type="fluid">
    <Add_BC name="physical_hydrostatic_fixed_gauge_mpi">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi_physical_hydrostatic_fixed_gauge_mpi</Level_set_field_name>
      <Generated_interface_domain_id>physical_hydrostatic_fixed_gauge_mpi</Generated_interface_domain_id>
      <Interface_marker>725</Interface_marker>
      <Generated_interface_geometry>LinearCorner</Generated_interface_geometry>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>)xml"
                        << (positive_side ? "LevelSetPositive"
                                          : "LevelSetNegative")
                        << R"xml(</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
      <Small_cut_aggregation>false</Small_cut_aggregation>
      <External_pressure>)xml"
                        << external_pressure
                        << R"xml(</External_pressure>
      <Surface_tension>1.0</Surface_tension>
      <Surface_tension_form>SurfaceStress</Surface_tension_form>
      <Contact_line_model>DynamicContactAngle</Contact_line_model>
      <Contact_angle_degrees>90.0</Contact_angle_degrees>
      <Contact_line_wall_markers>)xml"
                        << contact_wall_markers_xml.str()
                        << R"xml(</Contact_line_wall_markers>
      <Contact_line_wall_normals>)xml"
                        << contact_wall_normals_xml.str()
                        << R"xml(</Contact_line_wall_normals>
      <Contact_line_mobility>1.0</Contact_line_mobility>
      <Wall_slip_model>Navier</Wall_slip_model>
      <Wall_slip_length>1.0</Wall_slip_length>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml";
          const auto parameter_text = parameter_xml.str();
          auto params =
              parseMpiWorkflowParametersXml(parameter_text.c_str());
          auto requests = levelSetMaintenanceRequests(*params);
          ASSERT_EQ(requests.size(), 1u);
          ASSERT_TRUE(
              requests.front().static_capillary_equilibrium_enabled);

          svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle
              lifecycle;
          ActiveCutContextRefreshCache refresh_cache;
          const auto initial_report =
              refreshActiveCutIntegrationContextCached(
                  sim,
                  *params,
                  sim.time_history->u(),
                  lifecycle,
                  refresh_cache,
                  "application-driver-mpi-hydrostatic-fixed-gauge-initial");
          ASSERT_TRUE(initial_report.refreshed);
          ASSERT_NE(initial_report.topology_key, 0u);
          auto initial_functionals =
              evaluateCurrentFreeSurfaceDiscreteFunctionals(sim);
          ASSERT_EQ(initial_functionals.size(), 1u);
          attachAcceptedFreeSurfaceActiveVolumeEnergies(
              sim, current, initial_functionals);
          ASSERT_TRUE(
              initial_functionals.front().active_volume_energy.has_value());
          const auto tangent_measure =
              spatial_dimension == 2 ? svmp::FE::Real{3.0}
                                     : svmp::FE::Real{6.0};
          const auto expected_volume =
              tangent_measure *
              (positive_side
                   ? svmp::FE::Real{1.0} - normal_offset
                   : normal_offset);
          const auto active_first_moment =
              svmp::FE::Real{0.5} * tangent_measure *
              (positive_side
                   ? svmp::FE::Real{1.0} -
                         normal_offset * normal_offset
                   : normal_offset * normal_offset);
          const auto expected_gravitational_energy =
              -density * gravity * active_first_moment;
          EXPECT_NEAR(
              initial_functionals.front().state.owned_liquid_volume,
              expected_volume,
              1.0e-13);
          EXPECT_NEAR(
              initial_functionals.front().state.liquid_gas_surface_energy,
              tangent_measure,
              1.0e-13);
          EXPECT_NEAR(initial_functionals.front().state.young_wall_energy,
                      svmp::FE::Real{0.0},
                      1.0e-13);
          EXPECT_NEAR(
              initial_functionals.front()
                  .active_volume_energy->gravitational_energy,
              expected_gravitational_energy,
              2.0e-13);

          const auto pressure_offset_index =
              static_cast<std::size_t>(pressure_offset);
          const auto pressure_count = static_cast<std::size_t>(
              sim.fe_system->fieldDofHandler(pressure).getNumDofs());
          const std::vector<svmp::FE::Real>
              expected_pressure_coefficients(
                  exact_solution.begin() +
                      static_cast<std::ptrdiff_t>(pressure_offset_index),
                  exact_solution.begin() +
                      static_cast<std::ptrdiff_t>(pressure_offset_index +
                                                  pressure_count));
          const auto exact_pressure_certificate =
              evaluateStaticCapillaryPressureCertificate(
                  sim,
                  exact_solution,
                  requests.front().static_capillary_equilibrium,
                  /*initialize_compatible_pressure=*/false);
          const auto& exact_certificate =
              exact_pressure_certificate.report;
          ASSERT_TRUE(
              exact_certificate.pressure_representability_diagnostic_sampled);
          EXPECT_LE(exact_certificate.residual_norm, 2.0e-12);

          const auto exact_initialized_pressure_certificate =
              evaluateStaticCapillaryPressureCertificate(
                  sim,
                  exact_solution,
                  requests.front().static_capillary_equilibrium,
                  /*initialize_compatible_pressure=*/true);
          const auto& exact_initialized_report =
              exact_initialized_pressure_certificate.report;
          ASSERT_TRUE(
              exact_initialized_report
                  .static_compatible_pressure_initializer_applied);
          ASSERT_TRUE(
              exact_initialized_report
                  .static_compatible_pressure_initializer_passed);
          EXPECT_LE(exact_initialized_report.residual_norm, 2.0e-12);
          ASSERT_EQ(
              exact_initialized_pressure_certificate.certified_solution.size(),
              exact_solution.size());
          svmp::FE::Real exact_initializer_pressure_update = 0.0;
          for (std::size_t i = 0u; i < pressure_count; ++i) {
            exact_initializer_pressure_update =
                std::max(
                    exact_initializer_pressure_update,
                    std::abs(
                        exact_initialized_pressure_certificate
                            .certified_solution[pressure_offset_index + i] -
                        exact_solution[pressure_offset_index + i]));
          }
          EXPECT_LE(exact_initializer_pressure_update, 2.0e-12);

          bool initialized = false;
          ASSERT_NO_THROW(
              initialized = initializeDiscreteStaticCapillaryEquilibrium(
                  sim,
                  *params,
                  requests,
                  lifecycle,
                  refresh_cache));
          ASSERT_TRUE(initialized);
          ASSERT_TRUE(
              requests.front().static_capillary_equilibrium_initialized);

          const auto communicator =
              activeFESystemCommunicator(*sim.fe_system);
          const auto certified_solution =
              capturePostacceptMaintenanceVectorCollectively(
                  sim.time_history->u(), communicator);
          const auto pressure_certificate =
              evaluateStaticCapillaryPressureCertificate(
                  sim,
                  certified_solution,
                  requests.front().static_capillary_equilibrium,
                  /*initialize_compatible_pressure=*/false);
          const auto& certificate = pressure_certificate.report;
          ASSERT_TRUE(
              certificate.pressure_representability_diagnostic_sampled);
          ASSERT_TRUE(certificate.pressure_representability_available)
              << certificate.pressure_representability_reason;
          EXPECT_TRUE(certificate.pressure_representability_converged);
          EXPECT_FALSE(certificate.pressure_representability_breakdown);
          EXPECT_LE(certificate.pressure_representability_residual_norm,
                    2.0e-10);
          EXPECT_LE(certificate.pressure_representability_relative_distance,
                    2.0e-10);
          EXPECT_LE(certificate.residual_norm, 2.0e-10);
          EXPECT_FALSE(
              certificate.constant_pressure_constraints_preserve_constants);
          EXPECT_FALSE(certificate.constant_pressure_kkt_available);

          svmp::FE::Real initializer_pressure_representative_distance =
              0.0;
          for (std::size_t i = 0u;
               i < expected_pressure_coefficients.size();
               ++i) {
            initializer_pressure_representative_distance =
                std::max(
                    initializer_pressure_representative_distance,
                    std::abs(certified_solution[
                                 pressure_offset_index + i] -
                             expected_pressure_coefficients[i]));
          }
          EXPECT_TRUE(std::isfinite(
              initializer_pressure_representative_distance));

          const auto phi_offset_index =
              static_cast<std::size_t>(phi_offset);
          const auto phi_count = static_cast<std::size_t>(
              sim.fe_system->fieldDofHandler(phi).getNumDofs());
          svmp::FE::Real phi_update = 0.0;
          for (std::size_t i = 0u; i < phi_count; ++i) {
            phi_update =
                std::max(phi_update,
                         std::abs(certified_solution[
                                      phi_offset_index + i] -
                                  current[phi_offset_index + i]));
          }
          EXPECT_LE(phi_update, 2.0e-7);

          auto final_functionals =
              evaluateCurrentFreeSurfaceDiscreteFunctionals(sim);
          ASSERT_EQ(final_functionals.size(), 1u);
          attachAcceptedFreeSurfaceActiveVolumeEnergies(
              sim, certified_solution, final_functionals);
          ASSERT_TRUE(
              final_functionals.front().active_volume_energy.has_value());
          const auto gravitational_energy_error = std::abs(
              final_functionals.front()
                  .active_volume_energy->gravitational_energy -
              expected_gravitational_energy);
          const auto volume_error = std::abs(
              final_functionals.front().state.owned_liquid_volume -
              expected_volume);
          const auto surface_energy_error = std::abs(
              final_functionals.front().state.liquid_gas_surface_energy -
              tangent_measure);
          EXPECT_LE(gravitational_energy_error, 2.0e-10);
          EXPECT_LE(volume_error, 1.0e-11);
          EXPECT_LE(surface_energy_error, 2.0e-10);
          EXPECT_NEAR(final_functionals.front().state.young_wall_energy,
                      svmp::FE::Real{0.0},
                      1.0e-13);

          for (const auto scalar : {
                   static_cast<double>(exact_certificate.residual_norm),
                   static_cast<double>(
                       exact_initializer_pressure_update),
                   static_cast<double>(
                       certificate.pressure_representability_residual_norm),
                   static_cast<double>(
                       certificate
                           .pressure_representability_relative_distance),
                   static_cast<double>(certificate.residual_norm),
                   static_cast<double>(
                       initializer_pressure_representative_distance),
                   static_cast<double>(gravitational_energy_error),
                   static_cast<double>(volume_error),
                   static_cast<double>(surface_energy_error),
                   static_cast<double>(phi_update)}) {
            EXPECT_EQ(globalMinDouble(scalar, communicator),
                      globalMaxDouble(scalar, communicator));
          }
          const auto final_revision =
              collectiveLevelSetMaintenanceAlgebraicRevision(
                  certified_solution, communicator);
          const auto [minimum_revision, maximum_revision] =
              globalMinMaxUint64(final_revision, communicator);
          EXPECT_EQ(minimum_revision, maximum_revision);

          maximum_pressure_residual =
              std::max(maximum_pressure_residual,
                       static_cast<svmp::FE::Real>(
                           certificate
                               .pressure_representability_residual_norm));
          maximum_pressure_relative_distance =
              std::max(maximum_pressure_relative_distance,
                       static_cast<svmp::FE::Real>(
                           certificate
                               .pressure_representability_relative_distance));
          maximum_exact_field_production_residual =
              std::max(maximum_exact_field_production_residual,
                       static_cast<svmp::FE::Real>(
                           exact_certificate.residual_norm));
          maximum_production_residual =
              std::max(maximum_production_residual,
                       static_cast<svmp::FE::Real>(
                           certificate.residual_norm));
          maximum_initializer_pressure_representative_distance =
              std::max(
                  maximum_initializer_pressure_representative_distance,
                  initializer_pressure_representative_distance);
          maximum_exact_initializer_pressure_update =
              std::max(maximum_exact_initializer_pressure_update,
                       exact_initializer_pressure_update);
          maximum_gravitational_energy_error =
              std::max(maximum_gravitational_energy_error,
                       gravitational_energy_error);
          maximum_volume_error =
              std::max(maximum_volume_error, volume_error);
          maximum_surface_energy_error =
              std::max(maximum_surface_energy_error,
                       surface_energy_error);
          maximum_phi_update =
              std::max(maximum_phi_update, phi_update);
        }
      }
    }
  }

  EXPECT_EQ(two_dimensional_case_count, 384u);
  EXPECT_EQ(three_dimensional_case_count, 576u);
  EXPECT_EQ(case_count, 960u);
  EXPECT_EQ(owner_contiguous_nonidentity_case_count,
            case_count / fe_global_numbering_mode_count);
  EXPECT_EQ(three_dimensional_owner_contiguous_nonidentity_case_count,
            three_dimensional_case_count /
                fe_global_numbering_mode_count);
  EXPECT_EQ(three_dimensional_shared_vertex_case_count,
            three_dimensional_case_count);
  RecordProperty("wp4_hydrostatic_mpi_rank_count", size);
  RecordProperty("wp4_hydrostatic_mpi_partition_layout_count",
                 partition_layout_count);
  RecordProperty("wp4_hydrostatic_mpi_global_vertex_numbering_count",
                 vertex_numbering_count);
  RecordProperty("wp4_hydrostatic_mpi_dof_ownership_strategy_count",
                 dof_ownership_strategy_count);
  RecordProperty("wp4_hydrostatic_mpi_fe_global_numbering_mode_count",
                 fe_global_numbering_mode_count);
  RecordProperty(
      "wp4_hydrostatic_mpi_owner_contiguous_nonidentity_case_count",
      owner_contiguous_nonidentity_case_count);
  RecordProperty(
      "wp4_hydrostatic_mpi_three_dimensional_partition_layout_count",
      partition_layout_count);
  RecordProperty(
      "wp4_hydrostatic_mpi_three_dimensional_global_vertex_numbering_count",
      vertex_numbering_count);
  RecordProperty(
      "wp4_hydrostatic_mpi_three_dimensional_dof_ownership_strategy_count",
      dof_ownership_strategy_count);
  RecordProperty(
      "wp4_hydrostatic_mpi_three_dimensional_fe_global_numbering_mode_count",
      fe_global_numbering_mode_count);
  RecordProperty(
      "wp4_hydrostatic_mpi_three_dimensional_owner_contiguous_nonidentity_"
      "case_count",
      three_dimensional_owner_contiguous_nonidentity_case_count);
  RecordProperty(
      "wp4_hydrostatic_mpi_three_dimensional_shared_vertex_case_count",
      three_dimensional_shared_vertex_case_count);
  RecordProperty("wp4_hydrostatic_mpi_spatial_dimension", 3);
  RecordProperty("wp4_hydrostatic_mpi_spatial_dimension_count", 2);
  RecordProperty("wp4_hydrostatic_mpi_coordinate_direction_count", 3);
  RecordProperty("wp4_hydrostatic_mpi_dimension_coordinate_pair_count", 5);
  RecordProperty("wp4_hydrostatic_mpi_wall_orientation_count", 3);
  RecordProperty("wp4_hydrostatic_mpi_active_side_count", 2);
  RecordProperty("wp4_hydrostatic_mpi_cut_offset_count",
                 normal_offsets.size());
  RecordProperty("wp4_hydrostatic_mpi_gravity_direction_count", 2);
  RecordProperty("wp4_hydrostatic_mpi_two_dimensional_case_count",
                 two_dimensional_case_count);
  RecordProperty("wp4_hydrostatic_mpi_three_dimensional_case_count",
                 three_dimensional_case_count);
  RecordProperty("wp4_hydrostatic_mpi_fixed_zero_pressure_gauge_case_count",
                 case_count);
  RecordProperty("wp4_hydrostatic_mpi_matrix_case_count", case_count);
  RecordProperty(
      "wp4_hydrostatic_mpi_pressure_representability_residual_norm",
      maximum_pressure_residual);
  RecordProperty("wp4_hydrostatic_mpi_pressure_relative_distance",
                 maximum_pressure_relative_distance);
  RecordProperty(
      "wp4_hydrostatic_mpi_exact_field_production_residual_norm",
      maximum_exact_field_production_residual);
  RecordProperty("wp4_hydrostatic_mpi_production_residual_norm",
                 maximum_production_residual);
  RecordProperty(
      "wp4_hydrostatic_mpi_initializer_pressure_representative_distance",
      maximum_initializer_pressure_representative_distance);
  RecordProperty(
      "wp4_hydrostatic_mpi_exact_initializer_pressure_update",
      maximum_exact_initializer_pressure_update);
  RecordProperty("wp4_hydrostatic_mpi_gravitational_energy_error",
                 maximum_gravitational_energy_error);
  RecordProperty("wp4_hydrostatic_mpi_volume_error",
                 maximum_volume_error);
  RecordProperty("wp4_hydrostatic_mpi_surface_energy_error",
                 maximum_surface_energy_error);
  RecordProperty("wp4_hydrostatic_mpi_maximum_phi_update",
                 maximum_phi_update);
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
  const auto phase = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phase", .space = scalar_space, .components = 1});
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

  system->addOperator("equations");
  for (const auto field : {phi, phase, velocity}) {
    system->addCellKernel(
        "equations",
        field,
        field,
        std::make_shared<MpiWorkflowScaledMassKernel>(
            /*matrix_scale=*/1.0,
            /*vector_scale=*/0.0));
  }

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
  sim.backend =
      std::make_unique<svmp::FE::backends::FsilsFactory>(
          /*dofs_per_node=*/4,
          sim.fe_system->dofPermutation(),
          MPI_COMM_WORLD);
  ASSERT_NE(sim.backend, nullptr);
  const auto* distributed_equations =
      sim.fe_system->distributedSparsityIfAvailable("equations");
  ASSERT_NE(distributed_equations, nullptr);
  auto backend_layout_matrix =
      sim.backend->createMatrix(*distributed_equations);
  ASSERT_NE(backend_layout_matrix, nullptr);
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
  EXPECT_NE(report.topology_key, 0u);
  const auto [minimum_refresh_topology, maximum_refresh_topology] =
      globalMinMaxUint64(
          report.topology_key,
          svmp::MeshComm(MPI_COMM_WORLD));
  EXPECT_EQ(minimum_refresh_topology, maximum_refresh_topology);
  const auto current_functionals =
      evaluateCurrentFreeSurfaceDiscreteFunctionals(sim);
  const auto maintenance_functionals =
      levelSetMaintenanceFunctionalValues(
          sim, current_functionals, solution);
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
  LevelSetMaintenanceRequest geometry_request;
  geometry_request.level_set_field_name = "phi";
  geometry_request.volume_cut_request =
      active_cut_requests.front();
  geometry_request.conservative_phase.enabled = true;
  geometry_request.conservative_phase.liquid_indicator.field_name =
      "phase";
  geometry_request.conservative_phase_initialized = true;
  geometry_request.conservative_phase_graph =
      svmp::FE::level_set::buildLevelSetP1PhaseTransportGraph(
          *sim.fe_system, phase);
  ASSERT_TRUE(geometry_request.conservative_phase_graph->success)
      << geometry_request.conservative_phase_graph->diagnostic;
  ASSERT_TRUE(
      geometry_request.conservative_phase_graph->distributed);
  ASSERT_TRUE(
      geometry_request.conservative_phase_graph
          ->replicated_sparse_graph);
  const auto& distributed_graph =
      *geometry_request.conservative_phase_graph;
  EXPECT_EQ(
      distributed_graph.geometry_revision,
      sim.fe_system->meshAccess().geometryRevision());
  EXPECT_EQ(
      distributed_graph.topology_revision,
      sim.fe_system->meshAccess().topologyRevision());
  EXPECT_EQ(
      distributed_graph.ownership_revision,
      sim.fe_system->meshAccess().ownershipRevision());
  EXPECT_EQ(
      distributed_graph.numbering_revision,
      sim.fe_system->meshAccess().numberingRevision());
  const auto [minimum_graph_geometry, maximum_graph_geometry] =
      globalMinMaxUint64(
          distributed_graph.geometry_revision,
          svmp::MeshComm(MPI_COMM_WORLD));
  const auto [minimum_graph_topology, maximum_graph_topology] =
      globalMinMaxUint64(
          distributed_graph.topology_revision,
          svmp::MeshComm(MPI_COMM_WORLD));
  const auto [minimum_graph_ownership, maximum_graph_ownership] =
      globalMinMaxUint64(
          distributed_graph.ownership_revision,
          svmp::MeshComm(MPI_COMM_WORLD));
  EXPECT_TRUE(
      minimum_graph_geometry != maximum_graph_geometry ||
      minimum_graph_topology != maximum_graph_topology ||
      minimum_graph_ownership != maximum_graph_ownership)
      << "The fixture must retain unequal partition-local mesh revision "
         "stamps to exercise cut-context publication.";
  const std::vector<LevelSetMaintenanceRequest>
      geometry_requests{geometry_request};
  int stage_callback_sentinel = 0;
  ASSERT_NO_THROW({
    requireCollectiveLevelSetMaintenanceRequestSchedule(
        geometry_requests,
        LevelSetMaintenanceScheduleStage::
            ProspectiveAcceptedEndpoint,
        /*completed_step=*/3,
        svmp::MeshComm(MPI_COMM_WORLD));
    ++stage_callback_sentinel;
  });
  int every_rank_reached_stage_callback = 0;
  MPI_Allreduce(
      &stage_callback_sentinel,
      &every_rank_reached_stage_callback,
      1,
      MPI_INT,
      MPI_MIN,
      MPI_COMM_WORLD);
  EXPECT_EQ(every_rank_reached_stage_callback, 1);
  const auto geometry_state =
      canonicalLevelSetMaintenanceGeometryState(
          sim, geometry_requests);
  ASSERT_TRUE(geometry_state.supported);
  ASSERT_FALSE(geometry_state.words.empty());
  EXPECT_TRUE(
      application::core::
          collectiveLevelSetMaintenanceCanonicalWordsAgree(
              geometry_state.words,
              svmp::MeshComm(MPI_COMM_WORLD)));
  auto drifted_geometry_words = geometry_state.words;
  if (rank + 1 == size) {
    // The canonical stream ends in the final authoritative snapshot
    // revision key.  Alter it on one rank to model live-geometry revision
    // drift without mutating the shared production fixture.
    ++drifted_geometry_words.back();
  }
  EXPECT_FALSE(
      application::core::
          collectiveLevelSetMaintenanceCanonicalWordsAgree(
              drifted_geometry_words,
              svmp::MeshComm(MPI_COMM_WORLD)));
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
  auto time_history = svmp::FE::timestepping::TimeHistory::allocate(
      *sim.backend,
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

  const auto accepted_endpoint =
      capturePostacceptMaintenanceVectorCollectively(
          time_history.u(),
          activeFESystemCommunicator(*sim.fe_system));
  EXPECT_EQ(accepted_endpoint, solution);
  const auto previous_state_revision =
      collectiveLevelSetMaintenanceAlgebraicRevision(
          std::span<const svmp::FE::Real>(
              solution.data(), solution.size()),
          activeFESystemCommunicator(*sim.fe_system));
  const auto endpoint_state_revision =
      collectiveLevelSetMaintenanceAlgebraicRevision(
          std::span<const svmp::FE::Real>(
              accepted_endpoint.data(),
              accepted_endpoint.size()),
          activeFESystemCommunicator(*sim.fe_system));
  auto contact_stages = evaluateAcceptedFreeSurfaceContactStages(
      sim,
      svmp::FE::Real{0.10},
      svmp::FE::Real{1.0},
      previous_state_revision,
      endpoint_state_revision,
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
  const auto pre_maintenance_endpoint_state_revision =
      collectiveLevelSetMaintenanceAlgebraicRevision(
          std::span<const svmp::FE::Real>(
              solution.data(), solution.size()),
          activeFESystemCommunicator(*sim.fe_system));
  EXPECT_EQ(contact_stages.front().endpoint_state_revision,
            pre_maintenance_endpoint_state_revision);
  const auto local_pre_maintenance_revision =
      static_cast<unsigned long long>(
          pre_maintenance_endpoint_state_revision);
  unsigned long long minimum_pre_maintenance_revision = 0u;
  unsigned long long maximum_pre_maintenance_revision = 0u;
  MPI_Allreduce(&local_pre_maintenance_revision,
                &minimum_pre_maintenance_revision,
                1,
                MPI_UNSIGNED_LONG_LONG,
                MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(&local_pre_maintenance_revision,
                &maximum_pre_maintenance_revision,
                1,
                MPI_UNSIGNED_LONG_LONG,
                MPI_MAX,
                MPI_COMM_WORLD);
  EXPECT_EQ(minimum_pre_maintenance_revision,
            maximum_pre_maintenance_revision);
  ASSERT_NO_THROW(
      bindAcceptedFreeSurfaceContactStagesToEndpointRevision(
          contact_stages,
          pre_maintenance_endpoint_state_revision,
          std::span<const svmp::FE::Real>(
              solution.data(), solution.size()),
          activeFESystemCommunicator(*sim.fe_system)));
  EXPECT_EQ(
      contact_stages.front().endpoint_state_revision,
      pre_maintenance_endpoint_state_revision);
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
      capturePostacceptMaintenanceVectorCollectively(
          time_history.u(),
          activeFESystemCommunicator(*sim.fe_system));
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
  const auto accepted_state_revision =
      collectiveLevelSetMaintenanceAlgebraicRevision(
          maintained_solution,
          activeFESystemCommunicator(*sim.fe_system));
  ASSERT_NO_THROW(recordAcceptedFreeSurfaceDiscreteFunctionals(
      sim,
      /*accepted_step=*/2u,
      svmp::FE::Real{0.10},
      svmp::FE::Real{0.05},
      pre_maintenance_endpoint_state_revision,
      accepted_state_revision,
      contact_stages));
  const auto functional_history =
      sim.fe_system->freeSurfaceDiscreteFunctionalHistory();
  ASSERT_EQ(functional_history.size(), 1u);
  const auto& functional_record = functional_history.front();
  EXPECT_EQ(functional_record.accepted_step, 2u);
  EXPECT_EQ(
      functional_record.pre_maintenance_endpoint_state_revision,
      pre_maintenance_endpoint_state_revision);
  EXPECT_NE(
      functional_record.pre_maintenance_endpoint_state_revision,
      functional_record.state_revision);
  EXPECT_EQ(functional_record.state_revision,
            accepted_state_revision);
  EXPECT_NE(functional_record.cut_topology_revision, 0u);
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
  EXPECT_EQ(
      contact_stage.endpoint_state_revision,
      functional_record.pre_maintenance_endpoint_state_revision);
  EXPECT_DOUBLE_EQ(contact_stage.stage_time, svmp::FE::Real{0.10});
  EXPECT_DOUBLE_EQ(contact_stage.stage_alpha_f, svmp::FE::Real{1.0});
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

  auto& graph_before_rank_local_invalidation =
      requireCurrentConservativePhaseGraph(
          *sim.fe_system, geometry_request);
  ASSERT_EQ(graph_before_rank_local_invalidation.diagnostic, "ok");
  const auto collective_state_before_rank_local_invalidation =
      graph_before_rank_local_invalidation.collective_state;
  ASSERT_TRUE(collective_state_before_rank_local_invalidation);
  if (rank == 0) {
    mesh->event_bus().notify(svmp::MeshEvent::GeometryChanged);
  } else {
    graph_before_rank_local_invalidation.diagnostic =
        "cached graph sentinel";
  }
  const bool local_graph_is_stale =
      graph_before_rank_local_invalidation.geometry_revision !=
      sim.fe_system->meshAccess().geometryRevision();
  int local_stale_count = local_graph_is_stale ? 1 : 0;
  int global_stale_count = 0;
  MPI_Allreduce(&local_stale_count,
                &global_stale_count,
                1,
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD);
  ASSERT_EQ(global_stale_count, 1)
      << "Exactly one rank must invalidate its local graph cache stamp.";

  auto& graph_after_collective_rebuild =
      requireCurrentConservativePhaseGraph(
          *sim.fe_system, geometry_request);
  EXPECT_NE(
      graph_after_collective_rebuild.collective_state,
      collective_state_before_rank_local_invalidation);
  EXPECT_EQ(graph_after_collective_rebuild.diagnostic, "ok");
  EXPECT_EQ(
      graph_after_collective_rebuild.geometry_revision,
      sim.fe_system->meshAccess().geometryRevision());
  EXPECT_EQ(
      graph_after_collective_rebuild.topology_revision,
      sim.fe_system->meshAccess().topologyRevision());
  EXPECT_EQ(
      graph_after_collective_rebuild.ownership_revision,
      sim.fe_system->meshAccess().ownershipRevision());
  EXPECT_EQ(
      graph_after_collective_rebuild.numbering_revision,
      sim.fe_system->meshAccess().numberingRevision());
  EXPECT_EQ(
      graph_after_collective_rebuild.dof_layout_revision,
      sim.fe_system->fieldDofHandler(phase).dofLayoutRevision());
  EXPECT_EQ(
      graph_after_collective_rebuild.nodes,
      static_cast<std::size_t>(
          sim.fe_system->fieldDofHandler(phase).getNumDofs()));
  const bool local_rebuild_completed =
      graph_after_collective_rebuild.success &&
      graph_after_collective_rebuild.diagnostic == "ok" &&
      graph_after_collective_rebuild.collective_state !=
          collective_state_before_rank_local_invalidation &&
      graph_after_collective_rebuild.geometry_revision ==
          sim.fe_system->meshAccess().geometryRevision() &&
      graph_after_collective_rebuild.topology_revision ==
          sim.fe_system->meshAccess().topologyRevision() &&
      graph_after_collective_rebuild.ownership_revision ==
          sim.fe_system->meshAccess().ownershipRevision() &&
      graph_after_collective_rebuild.numbering_revision ==
          sim.fe_system->meshAccess().numberingRevision() &&
      graph_after_collective_rebuild.dof_layout_revision ==
          sim.fe_system->fieldDofHandler(phase).dofLayoutRevision() &&
      graph_after_collective_rebuild.nodes ==
          static_cast<std::size_t>(
              sim.fe_system->fieldDofHandler(phase).getNumDofs());
  const int local_rebuild_completed_value =
      local_rebuild_completed ? 1 : 0;
  int every_rank_rebuilt = 0;
  MPI_Allreduce(&local_rebuild_completed_value,
                &every_rank_rebuilt,
                1,
                MPI_INT,
                MPI_MIN,
                MPI_COMM_WORLD);
  EXPECT_EQ(every_rank_rebuilt, 1);

  ASSERT_NE(sim.fe_system->cutIntegrationContext(), nullptr);
  const auto* current_context =
      sim.fe_system->cutIntegrationContext();
  const auto level_set_markers =
      current_context->generatedLevelSetInterfaceMarkers();
  ASSERT_EQ(level_set_markers.size(), 1u);
  const auto* level_set_provenance =
      current_context
          ->findGeneratedLevelSetInterfacePublicationProvenance(
              level_set_markers.front());
  ASSERT_NE(level_set_provenance, nullptr);
  auto level_set_policy_request = level_set_provenance->request;
  if (rank == 0) {
    level_set_policy_request.tolerance *= 2.0;
  }
  auto level_set_policy_candidate =
      std::make_shared<svmp::FE::assembly::CutIntegrationContext>();
  level_set_policy_candidate->addGeneratedInterfaceDomain(
      svmp::FE::interfaces::LevelSetInterfaceDomain(
          std::move(level_set_policy_request)),
      level_set_provenance->volume_side_filter,
      level_set_provenance->publication_domain_id);
  ASSERT_THROW(
      sim.fe_system->setCutIntegrationContext(
          level_set_policy_candidate),
      svmp::FE::InvalidArgumentException);

  const svmp::FE::assembly::
      GeneratedInterfaceBoundaryPublicationProvenance*
          boundary_provenance = nullptr;
  for (const int marker :
       current_context->generatedInterfaceMarkers()) {
    const auto* candidate =
        current_context
            ->findGeneratedInterfaceBoundaryPublicationProvenance(
                marker);
    if (candidate != nullptr &&
        (boundary_provenance == nullptr ||
         candidate->stable_owner_key <
             boundary_provenance->stable_owner_key)) {
      boundary_provenance = candidate;
    }
  }
  ASSERT_NE(boundary_provenance, nullptr);
  auto boundary_policy_request = boundary_provenance->request;
  if (rank == 0) {
    ++boundary_policy_request.quadrature_order;
  }
  auto boundary_policy_candidate =
      std::make_shared<svmp::FE::assembly::CutIntegrationContext>();
  boundary_policy_candidate
      ->addGeneratedInterfaceBoundaryIntersectionDomain(
          svmp::FE::interfaces::
              GeneratedInterfaceBoundaryIntersectionDomain(
                  std::move(boundary_policy_request)));
  ASSERT_THROW(
      sim.fe_system->setCutIntegrationContext(
          boundary_policy_candidate),
      svmp::FE::InvalidArgumentException);

  if (rank == 0) {
    sim.fe_system->addCutIntegrationContextUpdateCallback(
        svmp::FE::systems::CutIntegrationContextUpdateCallback{
            .name = "rank_specific_publication_callback",
            .callback = [](const auto*) {},
        });
  }
  const auto publication_candidate =
      std::make_shared<svmp::FE::assembly::CutIntegrationContext>(
          *sim.fe_system->cutIntegrationContext());
  EXPECT_THROW(
      sim.fe_system->setCutIntegrationContext(publication_candidate),
      svmp::FE::InvalidArgumentException);
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
  graph.dimension = 2;
  graph.nodes = 2u;
  graph.edges = {
      svmp::FE::level_set::LevelSetP1PhaseGradientEdge{
          .first_node = 0,
          .second_node = 1,
          .owner_rank = 0,
      },
  };
  request.conservative_phase_graph = std::move(graph);

  const std::array<svmp::FE::Real, 2> volumes{1.0, 1.0};
  const std::array<svmp::FE::Real, 2> previous{0.75, 0.25};
  // The production split certifies the exact clipped q^n one-ring bounds.
  // Both nodes share the single canonical edge in this fixture.
  const std::array<svmp::FE::Real, 2> lower{0.25, 0.25};
  const std::array<svmp::FE::Real, 2> upper{0.75, 0.75};
  const std::array<svmp::FE::level_set::LevelSetPhaseFluxEdge, 1>
      flux_edges{
          svmp::FE::level_set::LevelSetPhaseFluxEdge{
              0, 1, -0.05, 0.20},
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
  stage.replicated_stage_inputs_satisfied = true;
  stage.maximum_courant = 0.25;
  constexpr svmp::FE::Real time_step = 0.05;
  constexpr svmp::FE::Real step_start_time = 0.55;
  const svmp::FE::Real accepted_time = step_start_time + time_step;
  stage.time_step = time_step;
  stage.sampled_nodal_velocity = {
      std::array<svmp::FE::Real, 3>{1.0, 0.0, 0.0},
      std::array<svmp::FE::Real, 3>{0.5, 0.25, 0.0},
  };
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
  auto& maintenance_ledger = candidate.maintenance_ledgers.front();
  maintenance_ledger.maximum_nodal_boundary_mass_transfer = 0.0;
  maintenance_ledger.boundary_mass_tolerance = 1.0e-12;
  const auto graph_identity =
      svmp::FE::level_set::levelSetP1PhaseGraphIdentity(
          *request.conservative_phase_graph);
  maintenance_ledger.split_stage_provenance =
      svmp::FE::level_set::LevelSetP1PhaseSplitStageProvenance{
          .scheme = svmp::FE::level_set::LevelSetP1PhaseSplitScheme::
              BackwardEulerExplicitIndicatorEndpointVelocity,
          .transport_mesh_policy =
              svmp::FE::level_set::LevelSetP1PhaseTransportMeshPolicy::
                  FixedBackground,
          .temporal_order = 1,
          .prospective_step = 12u,
          .attempt = 1u,
          .step_start_time = step_start_time,
          .step_end_time = accepted_time,
          .q_input_time = step_start_time,
          .velocity_state_time = accepted_time,
          .time_step = time_step,
          .operator_state_revision = 0x71u,
          .previous_q_revision =
              svmp::FE::level_set::levelSetP1PhaseScalarContentRevision(
                  previous),
          .nodal_velocity_revision =
              svmp::FE::level_set::levelSetP1PhaseVelocityContentRevision(
                  stage.sampled_nodal_velocity),
          .previous_graph_identity = graph_identity,
          .operator_graph_identity = graph_identity,
          .final_flux_ledger_digest =
              svmp::FE::level_set::levelSetP1PhaseFluxLedgerDigest(stage),
          .stage_options = stage.executed_options,
      };

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
          accepted_time,
          time_step,
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
          accepted_time,
          time_step,
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
          accepted_time,
          time_step,
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
      accepted_time,
      time_step,
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
        .kinetic_energy = 1.0,
        .gravitational_energy = 2.0,
        .gravitational_potential_power = -0.5,
        .surface_wall_potential_power = 0.25,
        .volume_constraint_potential_power = -0.125,
        .bulk_viscous_dissipation_rate = 0.75,
        .external_pressure_power = -0.4,
        .modeled_stored_energy = 6.125,
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
  const std::array<std::uint64_t, 10> local_attempt_metadata{
      attempt.transaction_id,
      static_cast<std::uint64_t>(attempt.status),
      attempt.step,
      attempt.attempt,
      static_cast<std::uint64_t>(attempt.declared_stage),
      attempt.extension_map_revision.value_or(0u),
      static_cast<std::uint64_t>(attempt.row_count),
      static_cast<std::uint64_t>(
          attempt.accepted_numerical_work != 0.0),
      static_cast<std::uint64_t>(
          attempt.modeled_energy_numerical_work.has_value()),
      static_cast<std::uint64_t>(
          attempt.accepted_modeled_energy_numerical_work
              .has_value())};
  std::array<std::uint64_t, 10> minimum_attempt_metadata{};
  std::array<std::uint64_t, 10> maximum_attempt_metadata{};
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
  ASSERT_EQ(minimum_attempt_metadata[8], 1u);
  ASSERT_EQ(minimum_attempt_metadata[9], 1u);
  const std::array<double, 4> local_attempt_values{
      attempt.numerical_work,
      attempt.accepted_numerical_work,
      *attempt.modeled_energy_numerical_work,
      *attempt.accepted_modeled_energy_numerical_work};
  std::array<double, 4> minimum_attempt_values{};
  std::array<double, 4> maximum_attempt_values{};
  MPI_Allreduce(
      local_attempt_values.data(),
      minimum_attempt_values.data(),
      static_cast<int>(local_attempt_values.size()),
      MPI_DOUBLE,
      MPI_MIN,
      MPI_COMM_WORLD);
  MPI_Allreduce(
      local_attempt_values.data(),
      maximum_attempt_values.data(),
      static_cast<int>(local_attempt_values.size()),
      MPI_DOUBLE,
      MPI_MAX,
      MPI_COMM_WORLD);
  EXPECT_EQ(minimum_attempt_values, maximum_attempt_values);

  const auto& breakdown = attempt.modeled_energy_breakdown;
  const auto substage_metadata =
      [](const application::core::
             LevelSetMaintenanceModeledEnergySubstage& substage) {
        return std::array<std::uint64_t, 3>{
            static_cast<std::uint64_t>(substage.row_count),
            static_cast<std::uint64_t>(
                substage.modeled_energy_change.has_value()),
            static_cast<std::uint64_t>(
                substage.accepted_modeled_energy_change.has_value())};
      };
  const std::array<
      application::core::LevelSetMaintenanceModeledEnergySubstage,
      6>
      breakdown_substages{
          breakdown.transport,
          breakdown.limiting,
          breakdown.reinitialization,
          breakdown.geometry_reconciliation,
          breakdown.global_correction,
          breakdown.numerical_maintenance_total};
  std::array<std::uint64_t, 18> local_breakdown_metadata{};
  for (std::size_t index = 0u;
       index < breakdown_substages.size();
       ++index) {
    const auto metadata =
        substage_metadata(breakdown_substages[index]);
    std::copy(
        metadata.begin(),
        metadata.end(),
        local_breakdown_metadata.begin() +
            static_cast<std::ptrdiff_t>(3u * index));
  }
  std::array<std::uint64_t, 18> minimum_breakdown_metadata{};
  std::array<std::uint64_t, 18> maximum_breakdown_metadata{};
  MPI_Allreduce(
      local_breakdown_metadata.data(),
      minimum_breakdown_metadata.data(),
      static_cast<int>(local_breakdown_metadata.size()),
      MPI_UINT64_T,
      MPI_MIN,
      MPI_COMM_WORLD);
  MPI_Allreduce(
      local_breakdown_metadata.data(),
      maximum_breakdown_metadata.data(),
      static_cast<int>(local_breakdown_metadata.size()),
      MPI_UINT64_T,
      MPI_MAX,
      MPI_COMM_WORLD);
  EXPECT_EQ(
      minimum_breakdown_metadata, maximum_breakdown_metadata);
  EXPECT_EQ(breakdown.transport.row_count, 0u);
  EXPECT_EQ(breakdown.limiting.row_count, 0u);
  EXPECT_EQ(breakdown.reinitialization.row_count, 1u);
  EXPECT_EQ(breakdown.geometry_reconciliation.row_count, 0u);
  EXPECT_EQ(breakdown.global_correction.row_count, 1u);
  EXPECT_EQ(
      breakdown.numerical_maintenance_total.row_count, 2u);
  ASSERT_TRUE(
      breakdown.reinitialization.modeled_energy_change.has_value());
  ASSERT_TRUE(
      breakdown.reinitialization.accepted_modeled_energy_change
          .has_value());
  ASSERT_TRUE(
      breakdown.global_correction.modeled_energy_change
          .has_value());
  ASSERT_TRUE(
      breakdown.global_correction.accepted_modeled_energy_change
          .has_value());
  ASSERT_TRUE(
      breakdown.numerical_maintenance_total.modeled_energy_change
          .has_value());
  ASSERT_TRUE(
      breakdown.numerical_maintenance_total
          .accepted_modeled_energy_change.has_value());
  const std::array<double, 6> local_breakdown_values{
      *breakdown.reinitialization.modeled_energy_change,
      *breakdown.reinitialization.accepted_modeled_energy_change,
      *breakdown.global_correction.modeled_energy_change,
      *breakdown.global_correction.accepted_modeled_energy_change,
      *breakdown.numerical_maintenance_total.modeled_energy_change,
      *breakdown.numerical_maintenance_total
           .accepted_modeled_energy_change};
  std::array<double, 6> minimum_breakdown_values{};
  std::array<double, 6> maximum_breakdown_values{};
  MPI_Allreduce(
      local_breakdown_values.data(),
      minimum_breakdown_values.data(),
      static_cast<int>(local_breakdown_values.size()),
      MPI_DOUBLE,
      MPI_MIN,
      MPI_COMM_WORLD);
  MPI_Allreduce(
      local_breakdown_values.data(),
      maximum_breakdown_values.data(),
      static_cast<int>(local_breakdown_values.size()),
      MPI_DOUBLE,
      MPI_MAX,
      MPI_COMM_WORLD);
  EXPECT_EQ(minimum_breakdown_values, maximum_breakdown_values);
  EXPECT_NEAR(
      *breakdown.numerical_maintenance_total
           .modeled_energy_change,
      *breakdown.reinitialization.modeled_energy_change +
          *breakdown.global_correction.modeled_energy_change,
      1.0e-15);

  auto after_limiting = value(4.05, 813u);
  after_limiting.kinetic_energy = 0.95;
  after_limiting.modeled_stored_energy = 6.075;
  ledger.beginTransaction(
      application::core::LevelSetMaintenanceWorkTransaction{
          .transaction_id = 602u,
          .step = 12u,
          .attempt = 3u,
          .time = 0.6,
          .dt = 0.05,
          .declared_stage =
              application::core::LevelSetMaintenanceDeclaredStage::
                  AcceptedEndpointPostStep,
          .extension_map_revision = 77u,
      });
  ledger.stageRow(
      application::core::LevelSetMaintenanceWorkSubstage::Limiting,
      7003u,
      7004u,
      {value(4.1, 812u)},
      {after_limiting});
  ledger.commitTransaction();
  ASSERT_EQ(ledger.acceptedRows().size(), 3u);
  ASSERT_EQ(ledger.acceptedAttempts().size(), 2u);

  const auto step_account =
      application::core::
          aggregateLevelSetMaintenanceAcceptedStepEnergy(
              ledger.acceptedAttempts(), ledger.acceptedRows());
  ASSERT_TRUE(step_account.has_value());
  const auto& step_breakdown =
      step_account->modeled_energy_breakdown;
  ASSERT_TRUE(step_account->maintenance_start.has_value());
  ASSERT_TRUE(step_account->post_transport.has_value());
  ASSERT_TRUE(step_account->maintenance_end.has_value());
  ASSERT_TRUE(
      step_account->maintenance_start->modeled_stored_energy
          .has_value());
  ASSERT_TRUE(
      step_account->post_transport->modeled_stored_energy
          .has_value());
  ASSERT_TRUE(
      step_account->maintenance_end->modeled_stored_energy
          .has_value());
  ASSERT_TRUE(
      step_account->maintenance_end
          ->gravitational_potential_power.has_value());
  ASSERT_TRUE(
      step_account->maintenance_end
          ->surface_wall_potential_power.has_value());
  ASSERT_TRUE(
      step_account->maintenance_end
          ->volume_constraint_potential_power.has_value());
  ASSERT_TRUE(
      step_account->maintenance_end
          ->bulk_viscous_dissipation_rate.has_value());
  ASSERT_TRUE(
      step_account->maintenance_end
          ->external_pressure_power.has_value());
  EXPECT_FALSE(
      step_account->physical_transport_endpoint_residual
          .has_value());
  ASSERT_TRUE(
      step_account->numerical_maintenance_endpoint_residual
          .has_value());
  ASSERT_TRUE(
      step_breakdown.limiting.accepted_modeled_energy_change
          .has_value());
  ASSERT_TRUE(
      step_breakdown.reinitialization
          .accepted_modeled_energy_change.has_value());
  ASSERT_TRUE(
      step_breakdown.global_correction
          .accepted_modeled_energy_change.has_value());
  ASSERT_TRUE(
      step_breakdown.numerical_maintenance_total
          .accepted_modeled_energy_change.has_value());
  const auto physical_channels =
      application::core::
          evaluateLevelSetMaintenancePhysicalEndpointChannels(
              *step_account,
              /*preceding_gravitational_energy=*/2.1,
              /*preceding_surface_wall_energy=*/3.0);
  ASSERT_TRUE(
      physical_channels.surface_wall_energy_change.has_value());
  ASSERT_TRUE(
      physical_channels.surface_transport_coupling_work
          .has_value());
  ASSERT_TRUE(
      physical_channels.gravitational_energy_change.has_value());
  ASSERT_TRUE(
      physical_channels.gravitational_transport_coupling_work
          .has_value());
  ASSERT_TRUE(
      physical_channels.bulk_viscous_dissipation_rate
          .has_value());
  ASSERT_TRUE(
      physical_channels.external_pressure_work.has_value());
  const std::array<std::uint64_t, 11> local_step_metadata{
      step_account->step,
      step_account->attempt,
      static_cast<std::uint64_t>(
          step_account->transaction_count),
      static_cast<std::uint64_t>(step_account->row_count),
      static_cast<std::uint64_t>(
          step_breakdown.transport.row_count),
      static_cast<std::uint64_t>(
          step_breakdown.limiting.row_count),
      static_cast<std::uint64_t>(
          step_breakdown.reinitialization.row_count),
      static_cast<std::uint64_t>(
          step_breakdown.global_correction.row_count),
      step_account->maintenance_start
          ->algebraic_state_revision,
      step_account->post_transport
          ->algebraic_state_revision,
      step_account->maintenance_end
          ->algebraic_state_revision,
  };
  std::array<std::uint64_t, 11> minimum_step_metadata{};
  std::array<std::uint64_t, 11> maximum_step_metadata{};
  MPI_Allreduce(
      local_step_metadata.data(),
      minimum_step_metadata.data(),
      static_cast<int>(local_step_metadata.size()),
      MPI_UINT64_T,
      MPI_MIN,
      MPI_COMM_WORLD);
  MPI_Allreduce(
      local_step_metadata.data(),
      maximum_step_metadata.data(),
      static_cast<int>(local_step_metadata.size()),
      MPI_UINT64_T,
      MPI_MAX,
      MPI_COMM_WORLD);
  EXPECT_EQ(minimum_step_metadata, maximum_step_metadata);
  EXPECT_EQ(step_account->transaction_count, 2u);
  EXPECT_EQ(step_account->row_count, 3u);
  const std::array<double, 21> local_step_values{
      step_account->time,
      step_account->dt,
      *step_breakdown.limiting.accepted_modeled_energy_change,
      *step_breakdown.reinitialization
           .accepted_modeled_energy_change,
      *step_breakdown.global_correction
           .accepted_modeled_energy_change,
      *step_breakdown.numerical_maintenance_total
           .accepted_modeled_energy_change,
      *step_account->maintenance_start->modeled_stored_energy,
      *step_account->post_transport->modeled_stored_energy,
      *step_account->maintenance_end->modeled_stored_energy,
      *step_account->numerical_maintenance_endpoint_residual,
      *step_account->maintenance_end
           ->gravitational_potential_power,
      *step_account->maintenance_end
           ->surface_wall_potential_power,
      *step_account->maintenance_end
           ->volume_constraint_potential_power,
      *step_account->maintenance_end
           ->bulk_viscous_dissipation_rate,
      *step_account->maintenance_end
           ->external_pressure_power,
      *physical_channels.surface_wall_energy_change,
      *physical_channels.surface_transport_coupling_work,
      *physical_channels.gravitational_energy_change,
      *physical_channels.gravitational_transport_coupling_work,
      *physical_channels.bulk_viscous_dissipation_rate,
      *physical_channels.external_pressure_work,
  };
  std::array<double, 21> minimum_step_values{};
  std::array<double, 21> maximum_step_values{};
  MPI_Allreduce(
      local_step_values.data(),
      minimum_step_values.data(),
      static_cast<int>(local_step_values.size()),
      MPI_DOUBLE,
      MPI_MIN,
      MPI_COMM_WORLD);
  MPI_Allreduce(
      local_step_values.data(),
      maximum_step_values.data(),
      static_cast<int>(local_step_values.size()),
      MPI_DOUBLE,
      MPI_MAX,
      MPI_COMM_WORLD);
  EXPECT_EQ(minimum_step_values, maximum_step_values);
  EXPECT_NEAR(
      *step_breakdown.numerical_maintenance_total
           .accepted_modeled_energy_change,
      -0.05,
      1.0e-15);
  EXPECT_NEAR(
      *physical_channels.surface_transport_coupling_work,
      0.1125,
      1.0e-15);
  EXPECT_NEAR(
      *physical_channels.gravitational_transport_coupling_work,
      -0.075,
      1.0e-15);
  EXPECT_NEAR(
      *physical_channels.external_pressure_work,
      -0.02,
      1.0e-15);

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
    const auto& before = row.before.front();
    const auto& after = row.after.front();
    const std::array<std::uint64_t, 18> local_channel_presence{
        before.kinetic_energy.has_value(),
        before.gravitational_energy.has_value(),
        before.gravitational_potential_power.has_value(),
        before.surface_wall_potential_power.has_value(),
        before.volume_constraint_potential_power.has_value(),
        before.bulk_viscous_dissipation_rate.has_value(),
        before.external_pressure_power.has_value(),
        before.modeled_stored_energy.has_value(),
        after.kinetic_energy.has_value(),
        after.gravitational_energy.has_value(),
        after.gravitational_potential_power.has_value(),
        after.surface_wall_potential_power.has_value(),
        after.volume_constraint_potential_power.has_value(),
        after.bulk_viscous_dissipation_rate.has_value(),
        after.external_pressure_power.has_value(),
        after.modeled_stored_energy.has_value(),
        row.modeled_energy_numerical_work.has_value(),
        row.accepted_modeled_energy_numerical_work.has_value(),
    };
    std::array<std::uint64_t, 18> minimum_channel_presence{};
    std::array<std::uint64_t, 18> maximum_channel_presence{};
    MPI_Allreduce(
        local_channel_presence.data(),
        minimum_channel_presence.data(),
        static_cast<int>(local_channel_presence.size()),
        MPI_UINT64_T,
        MPI_MIN,
        MPI_COMM_WORLD);
    MPI_Allreduce(
        local_channel_presence.data(),
        maximum_channel_presence.data(),
        static_cast<int>(local_channel_presence.size()),
        MPI_UINT64_T,
        MPI_MAX,
        MPI_COMM_WORLD);
    EXPECT_EQ(minimum_channel_presence, maximum_channel_presence);
    ASSERT_TRUE(std::all_of(
        minimum_channel_presence.begin(),
        minimum_channel_presence.end(),
        [](std::uint64_t present) { return present == 1u; }));
    const std::array<double, 24> local_values{
        row.time,
        row.dt,
        before.total_potential,
        after.total_potential,
        *before.kinetic_energy,
        *after.kinetic_energy,
        *before.gravitational_energy,
        *after.gravitational_energy,
        *before.gravitational_potential_power,
        *after.gravitational_potential_power,
        *before.surface_wall_potential_power,
        *after.surface_wall_potential_power,
        *before.volume_constraint_potential_power,
        *after.volume_constraint_potential_power,
        *before.bulk_viscous_dissipation_rate,
        *after.bulk_viscous_dissipation_rate,
        *before.external_pressure_power,
        *after.external_pressure_power,
        *before.modeled_stored_energy,
        *after.modeled_stored_energy,
        row.numerical_work,
        row.accepted_numerical_work,
        *row.modeled_energy_numerical_work,
        *row.accepted_modeled_energy_numerical_work,
    };
    std::array<double, 24> minimum_values{};
    std::array<double, 24> maximum_values{};
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
     ContactStageBindingUsesCollectiveContentAndConsensusCoverage)
{
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    GTEST_SKIP() << "This collective provenance fixture requires two ranks.";
  }

  svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration
      inconsistent_declaration;
  if (rank == 0) {
    inconsistent_declaration.parameters.dynamic_contact_coefficients
        .push_back(
            svmp::FE::interfaces::
                FreeSurfaceDynamicContactCoefficient{});
  }
  const std::array<
      svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration,
      1u>
      inconsistent_declarations{{inconsistent_declaration}};
  bool local_declaration_mismatch_rejected = false;
  try {
    (void)requireCommunicatorConsistentFreeSurfaceAcceptanceCoverage(
        inconsistent_declarations, svmp::MeshComm::world());
  } catch (const std::runtime_error&) {
    local_declaration_mismatch_rejected = true;
  }
  const int local_declaration_rejection =
      local_declaration_mismatch_rejected ? 1 : 0;
  int communicator_declaration_rejection = 0;
  MPI_Allreduce(&local_declaration_rejection,
                &communicator_declaration_rejection,
                1,
                MPI_INT,
                MPI_MIN,
                MPI_COMM_WORLD);
  EXPECT_EQ(communicator_declaration_rejection, 1);

  svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration
      inconsistent_method_declaration;
  inconsistent_method_declaration.interface_marker = 701;
  if (rank == 0) {
    inconsistent_method_declaration.capillary_balance_method =
        svmp::FE::systems::FreeSurfaceCapillaryBalanceMethod::
            DiscreteEnergyVolumeStationarity;
    inconsistent_method_declaration.capillary_balance_qualification =
        svmp::FE::systems::FreeSurfaceCapillaryBalanceQualification::
            PrerequisiteOnly;
  }
  const std::array<
      svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration,
      1u>
      inconsistent_method_declarations{{
          inconsistent_method_declaration}};
  bool local_method_mismatch_rejected = false;
  try {
    (void)requireCommunicatorConsistentFreeSurfaceAcceptanceCoverage(
        inconsistent_method_declarations, svmp::MeshComm::world());
  } catch (const std::runtime_error&) {
    local_method_mismatch_rejected = true;
  }
  const int local_method_rejection =
      local_method_mismatch_rejected ? 1 : 0;
  int communicator_method_rejection = 0;
  MPI_Allreduce(&local_method_rejection,
                &communicator_method_rejection,
                1,
                MPI_INT,
                MPI_MIN,
                MPI_COMM_WORLD);
  EXPECT_EQ(communicator_method_rejection, 1);

  svmp::FE::backends::FsilsVector local_full_state(4);
  {
    const std::array<svmp::FE::Real, 4> values{
        1.0, -2.0, 3.5, 4.25};
    auto state = local_full_state.localSpan();
    std::copy(values.begin(), values.end(), state.begin());
  }
  const auto raw_revision_before_rank_local_access =
      local_full_state.valueRevision();
  if (rank == 0) {
    (void)local_full_state.localSpan();
    EXPECT_NE(local_full_state.valueRevision(),
              raw_revision_before_rank_local_access);
  } else {
    EXPECT_EQ(local_full_state.valueRevision(),
              raw_revision_before_rank_local_access);
  }
  const auto [minimum_raw_revision, maximum_raw_revision] =
      globalMinMaxUint64(
          local_full_state.valueRevision(), svmp::MeshComm::world());
  EXPECT_NE(minimum_raw_revision, maximum_raw_revision);

  const auto& const_local_full_state = local_full_state;
  const auto full_ordered_state = const_local_full_state.localSpan();
  const auto content_revision =
      collectiveLevelSetMaintenanceAlgebraicRevision(
          full_ordered_state, svmp::MeshComm::world());
  const auto [minimum_content_revision, maximum_content_revision] =
      globalMinMaxUint64(content_revision, svmp::MeshComm::world());
  EXPECT_EQ(minimum_content_revision, maximum_content_revision);

  auto rank_inconsistent_state = std::vector<svmp::FE::Real>(
      full_ordered_state.begin(), full_ordered_state.end());
  if (rank == 0) {
    rank_inconsistent_state.back() += svmp::FE::Real{0.125};
  }
  bool local_content_mismatch_rejected = false;
  try {
    (void)collectiveLevelSetMaintenanceAlgebraicRevision(
        rank_inconsistent_state, svmp::MeshComm::world());
  } catch (const std::runtime_error&) {
    local_content_mismatch_rejected = true;
  }
  const int local_content_rejection =
      local_content_mismatch_rejected ? 1 : 0;
  int communicator_content_rejection = 0;
  MPI_Allreduce(&local_content_rejection,
                &communicator_content_rejection,
                1,
                MPI_INT,
                MPI_MIN,
                MPI_COMM_WORLD);
  EXPECT_EQ(communicator_content_rejection, 1);

  DynamicContactFirstOrderGeneralizedAlphaObservation
      rank_inconsistent_operator_stage{
          .step_index = 2,
          .attempt_index = 0,
          .step_start_time = svmp::FE::Real{0.05},
          .step_end_time = svmp::FE::Real{0.10},
          .state_time = svmp::FE::Real{0.075},
          .rate_time = svmp::FE::Real{0.075},
          .provenance = {
              .alpha_m = svmp::FE::Real{0.5},
              .alpha_f = svmp::FE::Real{0.5},
              .gamma = svmp::FE::Real{0.5},
              .dt = svmp::FE::Real{0.05},
          },
          .operator_stage_state = {
              .full_state_size = 4,
              .fields = {{
                  .field = 0,
                  .offset = 0,
                  .count = 4,
                  .values = std::vector<svmp::FE::Real>(
                      full_ordered_state.begin(), full_ordered_state.end()),
              }},
          },
      };
  if (rank == 0) {
    rank_inconsistent_operator_stage.operator_stage_state.fields.front()
        .values.back() += svmp::FE::Real{0.125};
  }
  bool local_operator_stage_mismatch_rejected = false;
  try {
    requireDynamicContactOperatorStageObservationConsensus(
        rank_inconsistent_operator_stage, svmp::MeshComm::world());
  } catch (const std::runtime_error&) {
    local_operator_stage_mismatch_rejected = true;
  }
  const int local_operator_stage_rejection =
      local_operator_stage_mismatch_rejected ? 1 : 0;
  int communicator_operator_stage_rejection = 0;
  MPI_Allreduce(&local_operator_stage_rejection,
                &communicator_operator_stage_rejection,
                1,
                MPI_INT,
                MPI_MIN,
                MPI_COMM_WORLD);
  EXPECT_EQ(communicator_operator_stage_rejection, 1);

  rank_inconsistent_operator_stage.operator_stage_state.fields.front()
      .values.assign(full_ordered_state.begin(), full_ordered_state.end());
  rank_inconsistent_operator_stage.operator_stage_state.fields.front()
      .values.front() = rank == 0 ? svmp::FE::Real{-0.0}
                                  : svmp::FE::Real{0.0};
  bool local_signed_zero_mismatch_rejected = false;
  try {
    requireDynamicContactOperatorStageObservationConsensus(
        rank_inconsistent_operator_stage, svmp::MeshComm::world());
  } catch (const std::runtime_error&) {
    local_signed_zero_mismatch_rejected = true;
  }
  const int local_signed_zero_rejection =
      local_signed_zero_mismatch_rejected ? 1 : 0;
  int communicator_signed_zero_rejection = 0;
  MPI_Allreduce(&local_signed_zero_rejection,
                &communicator_signed_zero_rejection,
                1,
                MPI_INT,
                MPI_MIN,
                MPI_COMM_WORLD);
  EXPECT_EQ(communicator_signed_zero_rejection, 1);

  svmp::FE::systems::FreeSurfaceAcceptedContactStageState stage;
  stage.stage_time = svmp::FE::Real{0.075};
  stage.stage_alpha_f = svmp::FE::Real{0.5};
  stage.first_order_generalized_alpha =
      svmp::FE::systems::
          FreeSurfaceFirstOrderGeneralizedAlphaProvenance{
              .alpha_m = svmp::FE::Real{0.5},
              .alpha_f = svmp::FE::Real{0.5},
              .gamma = svmp::FE::Real{0.5},
              .dt = svmp::FE::Real{0.05},
          };
  stage.previous_state_revision = content_revision;
  stage.endpoint_state_revision = 17u;
  stage.stage_state_revision = 19u;
  stage.geometry_revision.snapshot_revision_key = 23u;
  std::vector<svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
      stages{stage};
  const auto endpoint_revision_before_rejected_bind =
      stages.front().endpoint_state_revision;
  const auto stage_revision_before_rejected_bind =
      stages.front().stage_state_revision;

  auto rank_inconsistent_stage_solution =
      std::vector<svmp::FE::Real>(
          full_ordered_state.begin(), full_ordered_state.end());
  if (rank == 0) {
    rank_inconsistent_stage_solution.front() +=
        svmp::FE::Real{0.25};
  }
  bool local_stage_bind_rejected = false;
  try {
    bindAcceptedFreeSurfaceContactStagesToEndpointRevision(
        stages,
        content_revision,
        rank_inconsistent_stage_solution,
        svmp::MeshComm::world());
  } catch (const std::runtime_error&) {
    local_stage_bind_rejected = true;
  }
  const int local_stage_bind_rejection =
      local_stage_bind_rejected ? 1 : 0;
  int communicator_stage_bind_rejection = 0;
  MPI_Allreduce(&local_stage_bind_rejection,
                &communicator_stage_bind_rejection,
                1,
                MPI_INT,
                MPI_MIN,
                MPI_COMM_WORLD);
  EXPECT_EQ(communicator_stage_bind_rejection, 1);
  EXPECT_EQ(stages.front().endpoint_state_revision,
            endpoint_revision_before_rejected_bind);
  EXPECT_EQ(stages.front().stage_state_revision,
            stage_revision_before_rejected_bind);

  const auto consistent_stage_solution =
      std::vector<svmp::FE::Real>(
          full_ordered_state.begin(), full_ordered_state.end());
  auto rank_inconsistent_provenance_stages = stages;
  if (rank == 0) {
    const auto rank_zero_parameters =
        svmp::FE::timestepping::utils::
            generalizedAlphaFirstOrderFromRhoInf(0.2);
    rank_inconsistent_provenance_stages.front()
        .first_order_generalized_alpha =
        svmp::FE::systems::
            FreeSurfaceFirstOrderGeneralizedAlphaProvenance{
                .alpha_m = rank_zero_parameters.alpha_m,
                .alpha_f = rank_zero_parameters.alpha_f,
                .gamma = rank_zero_parameters.gamma,
                .dt = svmp::FE::Real{0.05},
            };
  }
  bool local_provenance_bind_rejected = false;
  try {
    bindAcceptedFreeSurfaceContactStagesToEndpointRevision(
        rank_inconsistent_provenance_stages,
        content_revision,
        consistent_stage_solution,
        svmp::MeshComm::world());
  } catch (const std::runtime_error&) {
    local_provenance_bind_rejected = true;
  }
  const int local_provenance_rejection =
      local_provenance_bind_rejected ? 1 : 0;
  int communicator_provenance_rejection = 0;
  MPI_Allreduce(&local_provenance_rejection,
                &communicator_provenance_rejection,
                1,
                MPI_INT,
                MPI_MIN,
                MPI_COMM_WORLD);
  EXPECT_EQ(communicator_provenance_rejection, 1);
  EXPECT_EQ(
      rank_inconsistent_provenance_stages.front().endpoint_state_revision,
      endpoint_revision_before_rejected_bind);
  EXPECT_EQ(
      rank_inconsistent_provenance_stages.front().stage_state_revision,
      stage_revision_before_rejected_bind);

  ASSERT_NO_THROW(
      bindAcceptedFreeSurfaceContactStagesToEndpointRevision(
          stages,
          content_revision,
          consistent_stage_solution,
          svmp::MeshComm::world()));
  EXPECT_EQ(stages.front().endpoint_state_revision, content_revision);
  EXPECT_EQ(
      stages.front().stage_state_revision,
      acceptedContactStageRevision(
          stages.front().previous_state_revision,
          content_revision,
          stages.front().geometry_revision.snapshot_revision_key,
          stages.front().stage_time,
          stages.front().stage_alpha_f,
          consistent_stage_solution,
          stages.front().first_order_generalized_alpha));
  const auto [minimum_bound_stage_revision,
              maximum_bound_stage_revision] =
      globalMinMaxUint64(
          stages.front().stage_state_revision,
          svmp::MeshComm::world());
  EXPECT_EQ(minimum_bound_stage_revision,
            maximum_bound_stage_revision);
}

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     ContactStageFailureRestoresEveryRankBeforeCollectiveRejection)
{
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    GTEST_SKIP() << "This contact-stage failure fixture requires two ranks.";
  }

  std::exception_ptr local_stage_failure;
  if (rank == 0) {
    local_stage_failure = std::make_exception_ptr(
        std::runtime_error("rank-zero contact-stage failure"));
  }
  int endpoint_restore_count = 0;
  bool rejected = false;
  std::string diagnostic;
  try {
    restoreAcceptedContactStageEndpointAndRequireCollectiveSuccess(
        local_stage_failure,
        [&] { ++endpoint_restore_count; },
        svmp::MeshComm(MPI_COMM_WORLD));
  } catch (const std::runtime_error& error) {
    rejected = true;
    diagnostic = error.what();
  }

  const int local_rejected = rejected ? 1 : 0;
  int every_rank_rejected = 0;
  int minimum_restore_count = 0;
  int maximum_restore_count = 0;
  MPI_Allreduce(
      &local_rejected,
      &every_rank_rejected,
      1,
      MPI_INT,
      MPI_MIN,
      MPI_COMM_WORLD);
  MPI_Allreduce(
      &endpoint_restore_count,
      &minimum_restore_count,
      1,
      MPI_INT,
      MPI_MIN,
      MPI_COMM_WORLD);
  MPI_Allreduce(
      &endpoint_restore_count,
      &maximum_restore_count,
      1,
      MPI_INT,
      MPI_MAX,
      MPI_COMM_WORLD);

  EXPECT_EQ(every_rank_rejected, 1);
  EXPECT_EQ(minimum_restore_count, 1);
  EXPECT_EQ(maximum_restore_count, 1);
  if (rank == 0) {
    EXPECT_EQ(diagnostic, "rank-zero contact-stage failure");
  } else {
    EXPECT_NE(
        diagnostic.find(
            "accepted_contact_stage_evaluation_and_endpoint_restore"),
        std::string::npos);
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

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     KinematicAreaGradientTractionDispatchesCollectiveCurvatureRecovery)
{
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    GTEST_SKIP() << "This collective curvature fixture requires two ranks.";
  }

  constexpr int interface_marker = 733;
  auto mesh = makePartitionedHydrostaticPressureMesh(
      /*normal_axis=*/1,
      /*column_major_cells=*/false,
      /*reverse_vertex_numbering=*/false);
  auto scalar_space = svmp::FE::spaces::SpaceFactory::create_h1(
      svmp::FE::ElementType::Triangle3, /*order=*/1);
  auto velocity_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, 2);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi_total_energy_collective",
      .space = scalar_space,
      .components = 1,
      .source_kind =
          svmp::FE::systems::FieldSourceKind::PrescribedData,
  });
  const auto kappa = system->addField(svmp::FE::systems::FieldSpec{
      .name = "kappa_total_energy_collective",
      .space = scalar_space,
      .components = 1,
      .source_kind =
          svmp::FE::systems::FieldSourceKind::PrescribedData,
  });
  const auto velocity = system->addField(svmp::FE::systems::FieldSpec{
      .name = "velocity_total_energy_collective",
      .space = velocity_space,
      .components = 2,
  });

  svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalParameters
      functional_parameters;
  functional_parameters.liquid_side =
      svmp::FE::geometry::CutIntegrationSide::Negative;
  functional_parameters.surface_tension = 0.5;
  system->declareFreeSurfaceDiscreteFunctional(
      svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration{
          .interface_marker = interface_marker,
          .level_set_field = phi,
          .curvature_field = kappa,
          .velocity_field = velocity,
          .geometry_domain_id = "total_energy_collective",
          .parameters = functional_parameters,
          .endpoint_functional_power_enabled = true,
          .capillary_balance_method = svmp::FE::systems::
              FreeSurfaceCapillaryBalanceMethod::
                  KinematicAreaGradientEnergyTraction,
          .capillary_balance_qualification = svmp::FE::systems::
              FreeSurfaceCapillaryBalanceQualification::PrerequisiteOnly,
          .owner_component =
              "ApplicationDriverLevelSetWorkflowsMPI.TotalEnergyTraction",
      });
  ASSERT_NO_THROW(system->setup({}));

  constexpr svmp::FE::Real center_x = 1.5;
  constexpr svmp::FE::Real center_y = 0.5;
  constexpr svmp::FE::Real radius = 0.42;
  std::vector<svmp::FE::Real> phi_vertex_values(
      mesh->n_vertices(), svmp::FE::Real{0.0});
  const auto& local_mesh = mesh->local_mesh();
  for (std::size_t vertex = 0u; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = local_mesh.get_vertex_coords(
        static_cast<svmp::index_t>(vertex));
    phi_vertex_values[vertex] =
        std::hypot(point[0] - center_x, point[1] - center_y) - radius;
  }
  ASSERT_NO_THROW(setScalarPrescribedVertexFieldFromValues(
      *system,
      *mesh,
      phi,
      std::span<const svmp::FE::Real>(phi_vertex_values.data(),
                                      phi_vertex_values.size()),
      "Collective total-energy curvature fixture"));

  LevelSetMaintenanceRequest request;
  request.level_set_field_name = "phi_total_energy_collective";
  request.curvature_projection_enabled = true;
  request.curvature_field_name = "kappa_total_energy_collective";
  request.curvature_projection.recovery_mode =
      svmp::FE::level_set::LevelSetCurvatureRecoveryMode::
          KinematicAreaGradient;
  request.curvature_projection
      .kinematic_area_gradient_filter_coefficient = 0.0;
  request.volume_cut_request = application::core::ActiveCutVolumeRequest{
      .level_set_field_name = "phi_total_energy_collective",
      .domain_id = "total_energy_collective",
      .requested_interface_marker = interface_marker,
      .active_side = application::core::LevelSetActiveSide::Negative,
  };
  std::vector<LevelSetMaintenanceRequest> requests{request};
  ASSERT_NO_THROW(bindKinematicAreaGradientTractionMaintenance(
      *system, requests));
  ASSERT_TRUE(requests.front()
                  .curvature_projection
                  .kinematic_area_gradient_negative_liquid_side);
  EXPECT_TRUE(requests.front()
                  .curvature_projection
                  .kinematic_area_gradient_young_walls.empty());

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);
  svmp::FE::systems::SystemStateView state{};
  CurvatureProjectionCache curvature_cache;
  ASSERT_EQ(projectLevelSetCurvatureFieldsFromState(
                sim,
                state,
                {},
                requests,
                /*step=*/0,
                "collective_total_energy_traction",
                /*honor_cadence=*/false,
                &curvature_cache),
            1u);
  ASSERT_EQ(curvature_cache.entries.size(), 1u);
  const auto& result =
      curvature_cache.entries.begin()->second.last_result;
  ASSERT_TRUE(result.success) << result.diagnostic;
  EXPECT_TRUE(result.kinematic_area_gradient_collective_replication);
  EXPECT_EQ(result.kinematic_area_gradient_parallel_size, 2);
  EXPECT_EQ(result.kinematic_area_gradient_gathered_owned_cells, 32u);
  EXPECT_GT(result.kinematic_area_gradient_cut_cells, 0u);
  EXPECT_GT(result.kinematic_area_gradient_operator_vertices, 0u);
  EXPECT_TRUE(std::isfinite(
      result.kinematic_area_gradient_mass_weighted_mean_curvature));
  EXPECT_TRUE(std::isfinite(
      result.kinematic_area_gradient_mass_weighted_rms_deviation));

  const auto projected = evaluateVertexField(
      *sim.fe_system,
      *mesh,
      kappa,
      state,
      1u,
      "checking collective total-energy curvature");
  ASSERT_EQ(projected.size(), mesh->n_vertices());
  ASSERT_EQ(projected,
            curvature_cache.entries.begin()
                ->second.last_curvature_vertex_values);
  const auto& vertex_gids = local_mesh.vertex_gids();
  ASSERT_EQ(vertex_gids.size(), mesh->n_vertices());
  auto local_max_gid = svmp::gid_t{-1};
  for (const auto gid : vertex_gids) {
    local_max_gid = std::max(local_max_gid, gid);
  }
  svmp::gid_t global_max_gid = svmp::gid_t{-1};
  ASSERT_EQ(MPI_Allreduce(&local_max_gid,
                          &global_max_gid,
                          1,
                          MPI_INT64_T,
                          MPI_MAX,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);
  ASSERT_GE(global_max_gid, svmp::gid_t{0});
  const auto global_vertex_count =
      static_cast<std::size_t>(global_max_gid + 1);
  std::vector<double> owned_curvature(global_vertex_count, 0.0);
  std::vector<int> owned_counts(global_vertex_count, 0);
  for (std::size_t vertex = 0u; vertex < mesh->n_vertices(); ++vertex) {
    if (mesh->owner_rank_vertex(static_cast<svmp::index_t>(vertex)) !=
        rank) {
      continue;
    }
    const auto gid = vertex_gids[vertex];
    ASSERT_GE(gid, svmp::gid_t{0});
    ASSERT_LT(static_cast<std::size_t>(gid), global_vertex_count);
    owned_curvature[static_cast<std::size_t>(gid)] = projected[vertex];
    owned_counts[static_cast<std::size_t>(gid)] = 1;
  }
  std::vector<double> global_curvature(global_vertex_count, 0.0);
  std::vector<int> global_counts(global_vertex_count, 0);
  ASSERT_EQ(MPI_Allreduce(owned_curvature.data(),
                          global_curvature.data(),
                          static_cast<int>(global_vertex_count),
                          MPI_DOUBLE,
                          MPI_SUM,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);
  ASSERT_EQ(MPI_Allreduce(owned_counts.data(),
                          global_counts.data(),
                          static_cast<int>(global_vertex_count),
                          MPI_INT,
                          MPI_SUM,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);
  svmp::FE::Real maximum_absolute_curvature = 0.0;
  svmp::FE::Real curvature_checksum = 0.0;
  for (std::size_t vertex = 0u; vertex < global_vertex_count; ++vertex) {
    EXPECT_EQ(global_counts[vertex], 1);
    EXPECT_TRUE(std::isfinite(global_curvature[vertex]));
    maximum_absolute_curvature = std::max(
        maximum_absolute_curvature,
        std::abs(global_curvature[vertex]));
    curvature_checksum += static_cast<svmp::FE::Real>(vertex + 1u) *
                          global_curvature[vertex];
  }
  EXPECT_GT(maximum_absolute_curvature, svmp::FE::Real{0.0});

  svmp::FE::Real minimum_checksum = 0.0;
  svmp::FE::Real maximum_checksum = 0.0;
  ASSERT_EQ(MPI_Allreduce(&curvature_checksum,
                          &minimum_checksum,
                          1,
                          MPI_DOUBLE,
                          MPI_MIN,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);
  ASSERT_EQ(MPI_Allreduce(&curvature_checksum,
                          &maximum_checksum,
                          1,
                          MPI_DOUBLE,
                          MPI_MAX,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);
  EXPECT_DOUBLE_EQ(minimum_checksum, maximum_checksum);

  const auto prescribed_revision =
      sim.fe_system->prescribedFieldRevision(kappa);
  std::uint64_t minimum_revision = 0u;
  std::uint64_t maximum_revision = 0u;
  ASSERT_EQ(MPI_Allreduce(&prescribed_revision,
                          &minimum_revision,
                          1,
                          MPI_UINT64_T,
                          MPI_MIN,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);
  ASSERT_EQ(MPI_Allreduce(&prescribed_revision,
                          &maximum_revision,
                          1,
                          MPI_UINT64_T,
                          MPI_MAX,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);
  EXPECT_GT(minimum_revision, 0u);
  EXPECT_EQ(minimum_revision, maximum_revision);

  RecordProperty("wp4_total_energy_collective_rank_count", size);
  RecordProperty("wp4_total_energy_collective_owned_cell_count",
                 result.kinematic_area_gradient_gathered_owned_cells);
  RecordProperty("wp4_total_energy_collective_cut_cell_count",
                 result.kinematic_area_gradient_cut_cells);
  RecordProperty("wp4_total_energy_collective_max_abs_curvature",
                 maximum_absolute_curvature);
  RecordProperty("wp4_total_energy_collective_curvature_checksum",
                 curvature_checksum);
}

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     CollectiveFeOrderedGatherReconstructsDisjointOwnedRows)
{
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    GTEST_SKIP() << "This collective gather fixture requires two ranks.";
  }

  using svmp::FE::GlobalIndex;
  using svmp::FE::Real;
  using svmp::FE::sparsity::DistributedSparsityPattern;
  using svmp::FE::sparsity::IndexRange;

  constexpr GlobalIndex global_size = 4;
  const IndexRange owned =
      rank == 0 ? IndexRange{0, 2} : IndexRange{2, 4};
  DistributedSparsityPattern pattern(
      owned, owned, global_size, global_size);
  pattern.ensureDiagonal();
  if (rank == 0) {
    pattern.addEntry(1, 2);
  } else {
    pattern.addEntry(2, 1);
  }
  pattern.finalize();
  if (rank == 0) {
    pattern.setGhostRows(
        std::vector<GlobalIndex>{2},
        std::vector<GlobalIndex>{0, 2},
        std::vector<GlobalIndex>{1, 2});
  } else {
    pattern.setGhostRows(
        std::vector<GlobalIndex>{1},
        std::vector<GlobalIndex>{0, 2},
        std::vector<GlobalIndex>{1, 2});
  }

  svmp::FE::backends::FsilsFactory factory(
      /*dof_per_node=*/1, {}, MPI_COMM_WORLD);
  auto layout_matrix = factory.createMatrix(pattern);
  auto vector = factory.createVector(global_size);
  ASSERT_NE(layout_matrix, nullptr);
  ASSERT_NE(vector, nullptr);

  const auto owned_rows = vector->ownedGlobalRows();
  ASSERT_EQ(owned_rows.size(), 2u);
  std::vector<Real> owned_values;
  owned_values.reserve(owned_rows.size());
  for (const auto row : owned_rows) {
    owned_values.push_back(Real{10.0} + static_cast<Real>(row));
  }
  auto view = vector->createAssemblyView();
  ASSERT_NE(view, nullptr);
  view->beginAssemblyPhase();
  view->setVectorEntries(owned_rows, owned_values);
  view->finalizeAssembly();
  vector->updateGhosts();

  const auto gathered = gatherFeOrderedSolution(
      *vector, svmp::MeshComm(MPI_COMM_WORLD));
  ASSERT_EQ(gathered.size(), 4u);
  EXPECT_DOUBLE_EQ(gathered[0], 10.0);
  EXPECT_DOUBLE_EQ(gathered[1], 11.0);
  EXPECT_DOUBLE_EQ(gathered[2], 12.0);
  EXPECT_DOUBLE_EQ(gathered[3], 13.0);

  const auto revision =
      collectiveLevelSetMaintenanceAlgebraicRevision(
          gathered, svmp::MeshComm(MPI_COMM_WORLD));
  const auto [minimum_revision, maximum_revision] =
      globalMinMaxUint64(
          revision, svmp::MeshComm(MPI_COMM_WORLD));
  EXPECT_EQ(minimum_revision, maximum_revision);
}

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     MaintenanceRequestScheduleRejectsRankSpecificDriftBeforeStageCallbacks)
{
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    GTEST_SKIP() << "This request-schedule fixture requires two ranks.";
  }

  LevelSetMaintenanceRequest first;
  first.level_set_field_name = "phi_first";
  first.velocity.source =
      svmp::FE::level_set::LevelSetVelocitySource::ConstantVector;
  first.reinitialization.enabled = true;
  first.reinitialization.cadence_steps = 2;
  first.bound_preserving.enabled = true;
  first.static_capillary_equilibrium_enabled = true;

  LevelSetMaintenanceRequest second;
  second.level_set_field_name = "phi_second";
  second.velocity.source =
      svmp::FE::level_set::LevelSetVelocitySource::ConstantVector;
  second.reinitialization.enabled = true;
  second.reinitialization.cadence_steps = 3;
  second.volume_correction.enabled = true;
  second.volume_correction.cadence_steps = 4;
  const std::vector<LevelSetMaintenanceRequest> baseline{
      first, second};

  using Mutation =
      std::function<void(std::vector<LevelSetMaintenanceRequest>&)>;
  const std::array<std::pair<const char*, Mutation>, 5> cases{{
      {"request_count",
       [](auto& requests) { requests.pop_back(); }},
      {"request_order",
       [](auto& requests) {
         std::swap(requests[0], requests[1]);
       }},
      {"reinitialization_cadence",
       [](auto& requests) {
         ++requests[0].reinitialization.cadence_steps;
       }},
      {"velocity_source",
       [](auto& requests) {
         requests[0].velocity.source =
             svmp::FE::level_set::LevelSetVelocitySource::
                 CoupledField;
       }},
      {"static_capillary_kkt_tolerance",
       [](auto& requests) {
         requests[0]
             .static_capillary_equilibrium
             .constant_pressure_kkt_max_relative_distance *= 2.0;
       }},
  }};

  for (const auto& [name, mutate] : cases) {
    auto requests = baseline;
    if (rank + 1 == size) {
      mutate(requests);
    }
    bool rejected = false;
    int stage_callback_sentinel = 0;
    try {
      requireCollectiveLevelSetMaintenanceRequestSchedule(
          requests,
          LevelSetMaintenanceScheduleStage::
              ProspectiveAcceptedEndpoint,
          /*completed_step=*/6,
          svmp::MeshComm(MPI_COMM_WORLD));
      ++stage_callback_sentinel;
    } catch (const std::runtime_error&) {
      rejected = true;
    }

    const int local_rejected = rejected ? 1 : 0;
    int every_rank_rejected = 0;
    int maximum_stage_callback_sentinel = 0;
    MPI_Allreduce(
        &local_rejected,
        &every_rank_rejected,
        1,
        MPI_INT,
        MPI_MIN,
        MPI_COMM_WORLD);
    MPI_Allreduce(
        &stage_callback_sentinel,
        &maximum_stage_callback_sentinel,
        1,
        MPI_INT,
        MPI_MAX,
        MPI_COMM_WORLD);
    EXPECT_EQ(every_rank_rejected, 1) << name;
    EXPECT_EQ(maximum_stage_callback_sentinel, 0) << name;
  }
}

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     ConservativePhaseCandidateStageRejectsRankOnlyDriftWithoutMutation)
{
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    GTEST_SKIP() << "This candidate-stage fixture requires two ranks.";
  }

  ConservativePhaseCandidateStageSnapshot baseline;
  baseline.temporal_order = 1;
  baseline.prospective_step = 7u;
  baseline.attempt = 3u;
  baseline.step_start_time = 0.5;
  baseline.step_end_time = 0.6;
  baseline.q_input_time = 0.5;
  baseline.velocity_state_time = 0.6;
  baseline.time_step = 0.1;
  baseline.operator_state_revision = 0x8412u;
  baseline.request_schedule_words = {0x11u, 0x22u};
  baseline.requests.resize(1u);
  auto& baseline_request = baseline.requests.front();
  baseline_request.enabled = true;
  baseline_request.phase_field_name = "phase";
  baseline_request.velocity_field_name = "velocity";
  baseline_request.velocity_source =
      svmp::FE::level_set::LevelSetVelocitySource::ConstantVector;
  baseline_request.graph_identity.dimension = 2;
  baseline_request.graph_identity.nodes = 2u;
  baseline_request.graph_identity.edges = 1u;
  baseline_request.graph_identity.dof_layout_revision = 0x31u;
  baseline_request.graph_identity.content_revision = 0x42u;
  baseline_request.sampled_nodal_velocity = {
      std::array<svmp::FE::Real, 3>{1.0, 0.0, 0.0},
      std::array<svmp::FE::Real, 3>{0.5, -0.25, 0.0},
  };

  using Mutation =
      std::function<void(ConservativePhaseCandidateStageSnapshot&)>;
  const std::array<std::pair<const char*, Mutation>, 3> cases{{
      {"request",
       [](auto& stage) {
         stage.requests.front().phase_field_name = "rank_only_phase";
       }},
      {"velocity",
       [](auto& stage) {
         stage.requests.front().sampled_nodal_velocity.front()[0] =
             svmp::FE::Real{1.25};
       }},
      {"attempt",
       [](auto& stage) { ++stage.attempt; }},
  }};
  const std::vector<svmp::FE::Real> retained_candidate_state{
      0.2, 0.4, 0.6};
  const auto comm = svmp::MeshComm(MPI_COMM_WORLD);

  for (const auto& [name, mutate] : cases) {
    auto stage = baseline;
    auto candidate_state = retained_candidate_state;
    if (rank + 1 == size) {
      mutate(stage);
    }
    const bool stage_agrees =
        collectiveConservativePhaseCandidateStageSnapshotAgrees(
            stage, comm);
    if (stage_agrees) {
      candidate_state.front() = svmp::FE::Real{9.0};
    }
    EXPECT_FALSE(stage_agrees) << name;
    EXPECT_EQ(candidate_state, retained_candidate_state) << name;

    int reached_sentinel = 1;
    int every_rank_reached_sentinel = 0;
    MPI_Allreduce(
        &reached_sentinel,
        &every_rank_reached_sentinel,
        1,
        MPI_INT,
        MPI_MIN,
        MPI_COMM_WORLD);
    EXPECT_EQ(every_rank_reached_sentinel, 1) << name;
  }

  auto rank_asymmetric_velocity =
      baseline.requests.front().sampled_nodal_velocity;
  if (rank + 1 == size) {
    rank_asymmetric_velocity.push_back(
        std::array<svmp::FE::Real, 3>{0.0, 0.0, 0.0});
  }
  EXPECT_FALSE(collectiveConservativePhaseVelocityBitsAgree(
      rank_asymmetric_velocity, comm));
  int reached_size_mismatch_sentinel = 1;
  int every_rank_reached_size_mismatch_sentinel = 0;
  MPI_Allreduce(
      &reached_size_mismatch_sentinel,
      &every_rank_reached_size_mismatch_sentinel,
      1,
      MPI_INT,
      MPI_MIN,
      MPI_COMM_WORLD);
  EXPECT_EQ(every_rank_reached_size_mismatch_sentinel, 1);
}

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     PostacceptMaintenanceTopologyRequiresCommunicatorConsistentEvidence)
{
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    GTEST_SKIP() << "This topology-evidence fixture requires two ranks.";
  }

  application::core::LevelSetMaintenanceWorkLedger ledger;
  ledger.beginTransaction(
      application::core::LevelSetMaintenanceWorkTransaction{
          .transaction_id = 901u,
          .step = 17u,
          .attempt = 2u,
          .time = 0.85,
          .dt = 0.05,
          .declared_stage =
              application::core::LevelSetMaintenanceDeclaredStage::
                  AcceptedEndpointPostStep,
          .extension_map_revision = 71u,
      });
  const auto comm = svmp::MeshComm(MPI_COMM_WORLD);
  const auto collective_decision =
      [&](const PostacceptMaintenanceTopologyEvidence& evidence,
          PostacceptMaintenanceTopologyDecision local_decision) {
        const std::array<std::uint64_t, 7> evidence_words{{
            /*tracking_required=*/1u,
            evidence.baseline_present ? 1u : 0u,
            evidence.baseline_key,
            evidence.final_present ? 1u : 0u,
            evidence.final_key,
            evidence.keys_equal ? 1u : 0u,
            static_cast<std::uint64_t>(local_decision),
        }};
        return application::core::
            collectiveLevelSetMaintenanceTransactionDecision(
                ledger,
                local_decision !=
                    PostacceptMaintenanceTopologyDecision::
                        InvariantFailure,
                comm,
                evidence_words);
      };

  constexpr std::uint64_t baseline_key = 0x51a7u;
  constexpr std::uint64_t changed_key = 0x62b8u;

  // A communicator-consistent complete mismatch first passes the
  // non-topology transaction consensus.  The driver then maps that exact
  // evidence to its typed maintenance-only topology rollback.
  const auto consistent_mismatch =
      postacceptMaintenanceTopologyEvidence(
          std::optional<std::uint64_t>{baseline_key},
          std::optional<std::uint64_t>{changed_key});
  const auto consistent_mismatch_classification =
      classifyPostacceptMaintenanceTopologyEvidence(
          consistent_mismatch);
  EXPECT_EQ(
      consistent_mismatch_classification,
      PostacceptMaintenanceTopologyDecision::RejectMaintenance);
  EXPECT_EQ(
      collective_decision(
          consistent_mismatch,
          consistent_mismatch_classification),
      application::core::
          LevelSetMaintenanceTransactionDecision::Commit);

  // One rank seeing A/A while the other sees A/B is evidence divergence,
  // not a communicator-wide typed topology mismatch.
  const auto asymmetric_final =
      postacceptMaintenanceTopologyEvidence(
          std::optional<std::uint64_t>{baseline_key},
          std::optional<std::uint64_t>{
              rank == 0 ? baseline_key : changed_key});
  const auto asymmetric_final_classification =
      classifyPostacceptMaintenanceTopologyEvidence(
          asymmetric_final);
  const auto local_asymmetric_classification =
      static_cast<std::uint64_t>(asymmetric_final_classification);
  std::uint64_t minimum_asymmetric_classification = 0u;
  std::uint64_t maximum_asymmetric_classification = 0u;
  MPI_Allreduce(
      &local_asymmetric_classification,
      &minimum_asymmetric_classification,
      1,
      MPI_UINT64_T,
      MPI_MIN,
      MPI_COMM_WORLD);
  MPI_Allreduce(
      &local_asymmetric_classification,
      &maximum_asymmetric_classification,
      1,
      MPI_UINT64_T,
      MPI_MAX,
      MPI_COMM_WORLD);
  EXPECT_NE(
      minimum_asymmetric_classification,
      maximum_asymmetric_classification);
  EXPECT_EQ(
      collective_decision(
          asymmetric_final,
          asymmetric_final_classification),
      application::core::
          LevelSetMaintenanceTransactionDecision::Reject);

  // Even locally stable A/A versus B/B evidence must reject through exact
  // canonical-word consensus rather than becoming a topology event.
  const auto rank_local_stable_key =
      rank == 0 ? baseline_key : changed_key;
  const auto asymmetric_stable =
      postacceptMaintenanceTopologyEvidence(
          std::optional<std::uint64_t>{rank_local_stable_key},
          std::optional<std::uint64_t>{rank_local_stable_key});
  const auto asymmetric_stable_classification =
      classifyPostacceptMaintenanceTopologyEvidence(
          asymmetric_stable);
  EXPECT_EQ(
      asymmetric_stable_classification,
      PostacceptMaintenanceTopologyDecision::Commit);
  EXPECT_EQ(
      collective_decision(
          asymmetric_stable,
          asymmetric_stable_classification),
      application::core::
          LevelSetMaintenanceTransactionDecision::Reject);

  int reached_after_topology_consensus = 1;
  int every_rank_reached_after_topology_consensus = 0;
  MPI_Allreduce(
      &reached_after_topology_consensus,
      &every_rank_reached_after_topology_consensus,
      1,
      MPI_INT,
      MPI_MIN,
      MPI_COMM_WORLD);
  EXPECT_EQ(every_rank_reached_after_topology_consensus, 1);
}

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     PostacceptMaintenanceRecoveryFailureSuppressesRejectedEvidence)
{
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    GTEST_SKIP() << "This recovery-readiness fixture requires two ranks.";
  }

  const bool local_geometry_recovery_ready = rank == 0;
  const bool geometry_recovery_ready =
      collectivePostacceptMaintenanceRecoveryPhaseReady(
          local_geometry_recovery_ready,
          svmp::MeshComm(MPI_COMM_WORLD));
  EXPECT_FALSE(geometry_recovery_ready);

  const bool terminal_rejected_evidence_may_publish =
      postacceptMaintenanceRejectionEvidenceMayPublish(
          geometry_recovery_ready,
          /*checkpoint_restored=*/true,
          /*restored_topology_verified=*/true,
          /*ledger_rejection_preflight_ready=*/true);
  EXPECT_FALSE(terminal_rejected_evidence_may_publish);

  int reached_after_failed_recovery_phase = 1;
  int every_rank_reached_after_failed_recovery_phase = 0;
  MPI_Allreduce(
      &reached_after_failed_recovery_phase,
      &every_rank_reached_after_failed_recovery_phase,
      1,
      MPI_INT,
      MPI_MIN,
      MPI_COMM_WORLD);
  EXPECT_EQ(every_rank_reached_after_failed_recovery_phase, 1);
}

TEST(ApplicationDriverLevelSetWorkflowsMPI,
     NativeCertifiedManufacturedChannelIsRepartitionIndependent)
{
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    GTEST_SKIP() << "This manufactured-channel fixture requires two ranks.";
  }

  namespace fixture = native_manufactured_channel_mpi;
  using ActiveSide = svmp::FE::geometry::CutIntegrationSide;
  constexpr std::array<const char*, 2> partition_methods{{
      "block",
      "metis",
  }};
  constexpr std::array<ActiveSide, 2> active_sides{{
      ActiveSide::Negative,
      ActiveSide::Positive,
  }};
  constexpr std::array<svmp::FE::Real, 11> wet_fractions{{
      0.0,
      1.0e-8,
      1.0e-6,
      1.0e-4,
      1.0e-2,
      0.1,
      0.25,
      0.49,
      0.5,
      0.51,
      1.0,
  }};
  constexpr std::array<svmp::FE::Real, 3> parent_measures{{
      fixture::window_height * fixture::channel_depth,
      fixture::window_height * fixture::channel_depth,
      fixture::channel_length * fixture::window_height,
  }};
  constexpr std::array<svmp::FE::Real, 3> full_work{{
      fixture::Harness::expectedFullForceWork(),
      fixture::Harness::expectedFullFluxWork(),
      fixture::Harness::expectedFullPenaltyWork(),
  }};

  using SideSamples = std::array<std::vector<fixture::Sample>, 2>;
  std::array<SideSamples, 2> samples_by_partition;
  std::array<std::vector<int>, 2> partition_owners;
  svmp::FE::Real maximum_analytic_measure_error = 0.0;
  svmp::FE::Real maximum_analytic_work_error = 0.0;
  svmp::FE::Real maximum_partition_measure_difference = 0.0;
  svmp::FE::Real maximum_partition_work_difference = 0.0;
  svmp::FE::Real maximum_side_reversal_work_difference = 0.0;
  svmp::FE::Real maximum_vertex_limit_mismatch = 0.0;
  svmp::FE::Real maximum_trace_bound = 0.0;
  svmp::FE::Real maximum_trace_ratio = 0.0;
  std::size_t maximum_factorized_input_dimension = 0u;
  std::size_t maximum_trace_patch_count = 0u;
  std::size_t maximum_localized_trace_patch_count = 0u;
  std::size_t maximum_trace_support_overlap = 0u;

  for (std::size_t partition_index = 0u;
       partition_index < partition_methods.size();
       ++partition_index) {
    for (std::size_t side_index = 0u;
         side_index < active_sides.size();
         ++side_index) {
      fixture::Harness harness(
          active_sides[side_index],
          /*upper_subdivisions=*/2,
          partition_methods[partition_index]);
      if (side_index == 0u) {
        partition_owners[partition_index] = harness.partitionOwners();
      } else {
        EXPECT_EQ(harness.partitionOwners(),
                  partition_owners[partition_index]);
      }
      auto& samples =
          samples_by_partition[partition_index][side_index];
      samples.reserve(wet_fractions.size());
      for (const auto fraction : wet_fractions) {
        SCOPED_TRACE(::testing::Message()
                     << "partition_method="
                     << partition_methods[partition_index]
                     << " active_side_index=" << side_index
                     << " wet_fraction=" << fraction);
        samples.push_back(harness.sample(fraction));
        const auto& sample = samples.back();
        EXPECT_EQ(sample.target_wet_fraction, fraction);
        EXPECT_NE(sample.trace_certificate_digest, 0u);
        EXPECT_TRUE(sample.trace_revision_match);
        EXPECT_TRUE(sample.trace_factorized_proof_valid);
        EXPECT_LE(sample.trace_maximum_factorized_input_dimension,
                  svmp::FE::math::dense_exact_dyadic_maximum_dimension);
        EXPECT_LE(sample.trace_grouped_symmetric_ratio,
                  (1.0 - 0.25) * (1.0 - 0.25));
        EXPECT_EQ(sample.physical_role_boundary_term_count, 0u);
        maximum_trace_ratio = std::max(
            maximum_trace_ratio,
            sample.trace_grouped_symmetric_ratio);
        maximum_trace_bound = std::max(
            maximum_trace_bound,
            sample.trace_global_conservative_upper_bound);
        maximum_factorized_input_dimension = std::max(
            maximum_factorized_input_dimension,
            sample.trace_maximum_factorized_input_dimension);
        maximum_trace_patch_count = std::max(
            maximum_trace_patch_count,
            sample.trace_patch_count);
        maximum_localized_trace_patch_count = std::max(
            maximum_localized_trace_patch_count,
            sample.trace_localized_support_patch_count);
        maximum_trace_support_overlap = std::max(
            maximum_trace_support_overlap,
            sample.trace_maximum_support_overlap);
        for (std::size_t role = 0u; role < parent_measures.size(); ++role) {
          SCOPED_TRACE(::testing::Message() << "boundary_role=" << role);
          EXPECT_GT(sample.generated_route_term_counts[role], 0u);
          EXPECT_NEAR(sample.parent_measures[role],
                      parent_measures[role],
                      5.0e-11);
          EXPECT_NEAR(sample.active_measures[role],
                      fraction * parent_measures[role],
                      5.0e-11);
          EXPECT_NEAR(sample.operator_work[role],
                      fraction * full_work[role],
                      5.0e-10);
          maximum_analytic_measure_error = std::max(
              maximum_analytic_measure_error,
              std::abs(sample.parent_measures[role] -
                       parent_measures[role]));
          maximum_analytic_measure_error = std::max(
              maximum_analytic_measure_error,
              std::abs(sample.active_measures[role] -
                       fraction * parent_measures[role]));
          maximum_analytic_work_error = std::max(
              maximum_analytic_work_error,
              std::abs(sample.operator_work[role] -
                       fraction * full_work[role]));
        }
        if (fraction == 0.0) {
          EXPECT_EQ(sample.active_rule_counts,
                    (std::array<std::size_t, 3>{{0u, 0u, 0u}}));
          EXPECT_EQ(sample.retained_active_rule_counts,
                    (std::array<std::size_t, 3>{{0u, 0u, 0u}}));
          EXPECT_EQ(sample.operator_work,
                    (std::array<svmp::FE::Real, 3>{{0.0, 0.0, 0.0}}));
          EXPECT_EQ(sample.trace_patch_count, 0u);
          EXPECT_EQ(sample.trace_boundary_rule_count, 0u);
        } else {
          for (std::size_t role = 0u;
               role < sample.active_rule_counts.size();
               ++role) {
            EXPECT_GT(sample.active_rule_counts[role], 0u);
            EXPECT_LE(sample.retained_active_rule_counts[role],
                      sample.active_rule_counts[role]);
          }
          EXPECT_GT(sample.retained_active_rule_counts[2], 0u);
          EXPECT_GT(sample.trace_patch_count, 0u);
          EXPECT_EQ(sample.trace_boundary_rule_count,
                    sample.retained_active_rule_counts[2]);
        }

        unsigned long long local_digest =
            static_cast<unsigned long long>(
                sample.trace_certificate_digest);
        unsigned long long minimum_digest = 0u;
        unsigned long long maximum_digest = 0u;
        MPI_Allreduce(&local_digest,
                      &minimum_digest,
                      1,
                      MPI_UNSIGNED_LONG_LONG,
                      MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(&local_digest,
                      &maximum_digest,
                      1,
                      MPI_UNSIGNED_LONG_LONG,
                      MPI_MAX,
                      MPI_COMM_WORLD);
        EXPECT_EQ(minimum_digest, maximum_digest);
      }

      ASSERT_EQ(samples.size(), wet_fractions.size());
      constexpr std::size_t left_index = 7u;
      constexpr std::size_t crossing_index = 8u;
      constexpr std::size_t right_index = 9u;
      for (std::size_t role = 0u; role < full_work.size(); ++role) {
        const auto left_limit =
            samples[left_index].operator_work[role] +
            0.01 * full_work[role];
        const auto right_limit =
            samples[right_index].operator_work[role] -
            0.01 * full_work[role];
        const auto crossing =
            samples[crossing_index].operator_work[role];
        maximum_vertex_limit_mismatch = std::max(
            maximum_vertex_limit_mismatch,
            std::max(std::abs(left_limit - crossing),
                     std::abs(right_limit - crossing)));
      }
    }
  }

  ASSERT_EQ(partition_owners[0].size(), partition_owners[1].size());
  EXPECT_NE(partition_owners[0], partition_owners[1]);
  for (std::size_t partition_index = 0u;
       partition_index < partition_methods.size();
       ++partition_index) {
    const auto& negative = samples_by_partition[partition_index][0];
    const auto& positive = samples_by_partition[partition_index][1];
    ASSERT_EQ(negative.size(), positive.size());
    for (std::size_t sample_index = 0u;
         sample_index < negative.size();
         ++sample_index) {
      for (std::size_t role = 0u; role < full_work.size(); ++role) {
        maximum_side_reversal_work_difference = std::max(
            maximum_side_reversal_work_difference,
            std::abs(negative[sample_index].operator_work[role] -
                     positive[sample_index].operator_work[role]));
      }
    }
  }

  for (std::size_t side_index = 0u;
       side_index < active_sides.size();
       ++side_index) {
    const auto& block = samples_by_partition[0][side_index];
    const auto& graph = samples_by_partition[1][side_index];
    ASSERT_EQ(block.size(), graph.size());
    for (std::size_t sample_index = 0u;
         sample_index < block.size();
         ++sample_index) {
      SCOPED_TRACE(::testing::Message()
                   << "partition_comparison_side=" << side_index
                   << " wet_fraction=" << wet_fractions[sample_index]);
      EXPECT_EQ(block[sample_index].active_rule_counts,
                graph[sample_index].active_rule_counts);
      EXPECT_EQ(block[sample_index].retained_active_rule_counts,
                graph[sample_index].retained_active_rule_counts);
      EXPECT_EQ(block[sample_index].active_markers,
                graph[sample_index].active_markers);
      EXPECT_EQ(block[sample_index].trace_patch_count,
                graph[sample_index].trace_patch_count);
      EXPECT_EQ(block[sample_index].trace_localized_support_patch_count,
                graph[sample_index].trace_localized_support_patch_count);
      EXPECT_EQ(block[sample_index].trace_boundary_rule_count,
                graph[sample_index].trace_boundary_rule_count);
      EXPECT_EQ(block[sample_index].trace_maximum_support_overlap,
                graph[sample_index].trace_maximum_support_overlap);
      EXPECT_EQ(block[sample_index].trace_global_conservative_upper_bound,
                graph[sample_index].trace_global_conservative_upper_bound);
      EXPECT_EQ(block[sample_index].trace_grouped_symmetric_ratio,
                graph[sample_index].trace_grouped_symmetric_ratio);
      const auto& block_patches =
          block[sample_index].trace_partition_invariant_patches;
      const auto& graph_patches =
          graph[sample_index].trace_partition_invariant_patches;
      ASSERT_EQ(block_patches.size(), graph_patches.size());
      for (std::size_t patch_index = 0u;
           patch_index < block_patches.size();
           ++patch_index) {
        SCOPED_TRACE(
            ::testing::Message() << "trace_patch_index=" << patch_index);
        const auto& block_patch = block_patches[patch_index];
        const auto& graph_patch = graph_patches[patch_index];
        EXPECT_EQ(block_patch.localized_support_patch,
                  graph_patch.localized_support_patch);
        EXPECT_EQ(block_patch.root_cell_gid, graph_patch.root_cell_gid);
        EXPECT_EQ(block_patch.support_cell_gids,
                  graph_patch.support_cell_gids);
        EXPECT_EQ(block_patch.boundary_rule_physical_keys,
                  graph_patch.boundary_rule_physical_keys);
        EXPECT_EQ(block_patch.boundary_rule_count,
                  graph_patch.boundary_rule_count);
        EXPECT_EQ(block_patch.raw_support_dof_count,
                  graph_patch.raw_support_dof_count);
        EXPECT_EQ(block_patch.terminal_tangent_dof_count,
                  graph_patch.terminal_tangent_dof_count);
        EXPECT_EQ(block_patch.rigid_mode_candidate_count,
                  graph_patch.rigid_mode_candidate_count);
        EXPECT_EQ(block_patch.structural_rigid_mode_count,
                  graph_patch.structural_rigid_mode_count);
        EXPECT_EQ(block_patch.rigid_mode_constraint_rank,
                  graph_patch.rigid_mode_constraint_rank);
        EXPECT_EQ(block_patch.maximum_cell_support_overlap,
                  graph_patch.maximum_cell_support_overlap);
        EXPECT_EQ(block_patch.retained_support_physical_volume,
                  graph_patch.retained_support_physical_volume);
        EXPECT_EQ(block_patch.generated_boundary_physical_measure,
                  graph_patch.generated_boundary_physical_measure);
        EXPECT_EQ(block_patch.directly_proven_upper_bound,
                  graph_patch.directly_proven_upper_bound);
        EXPECT_EQ(block_patch.rigid_mode_quotient_status,
                  graph_patch.rigid_mode_quotient_status);
        EXPECT_EQ(block_patch.proof_input, graph_patch.proof_input);
        EXPECT_EQ(block_patch.exact_rigid_factor_action_proven,
                  graph_patch.exact_rigid_factor_action_proven);
        EXPECT_EQ(block_patch.denominator_positive_definite_proven,
                  graph_patch.denominator_positive_definite_proven);
        EXPECT_EQ(block_patch.numerator_positive_semidefinite_proven,
                  graph_patch.numerator_positive_semidefinite_proven);
        EXPECT_EQ(block_patch.upper_inequality_proven,
                  graph_patch.upper_inequality_proven);
        EXPECT_EQ(block_patch.exact_factorized_materialization_proven,
                  graph_patch.exact_factorized_materialization_proven);
        EXPECT_EQ(block_patch.exact_sparse_map_applied,
                  graph_patch.exact_sparse_map_applied);
        EXPECT_EQ(block_patch.exact_common_kernel_proven,
                  graph_patch.exact_common_kernel_proven);
        EXPECT_EQ(block_patch.exact_dimension,
                  graph_patch.exact_dimension);
        EXPECT_EQ(block_patch.denominator_rank,
                  graph_patch.denominator_rank);
        EXPECT_EQ(block_patch.numerator_rank,
                  graph_patch.numerator_rank);
        EXPECT_EQ(block_patch.numerator_gram_block_count,
                  graph_patch.numerator_gram_block_count);
        EXPECT_EQ(block_patch.denominator_gram_block_count,
                  graph_patch.denominator_gram_block_count);
        EXPECT_EQ(block_patch.numerator_gram_row_count,
                  graph_patch.numerator_gram_row_count);
        EXPECT_EQ(block_patch.denominator_gram_row_count,
                  graph_patch.denominator_gram_row_count);
        EXPECT_EQ(block_patch.numerator_weight_term_count,
                  graph_patch.numerator_weight_term_count);
        EXPECT_EQ(block_patch.denominator_weight_term_count,
                  graph_patch.denominator_weight_term_count);
        EXPECT_EQ(block_patch.factorized_input_dimension,
                  graph_patch.factorized_input_dimension);
        EXPECT_EQ(block_patch.exact_common_kernel_nullity,
                  graph_patch.exact_common_kernel_nullity);
      }
      for (std::size_t role = 0u; role < full_work.size(); ++role) {
        maximum_partition_measure_difference = std::max(
            maximum_partition_measure_difference,
            std::abs(block[sample_index].active_measures[role] -
                     graph[sample_index].active_measures[role]));
        maximum_partition_work_difference = std::max(
            maximum_partition_work_difference,
            std::abs(block[sample_index].operator_work[role] -
                     graph[sample_index].operator_work[role]));
      }
    }
  }

  EXPECT_LE(maximum_analytic_measure_error, 5.0e-11);
  EXPECT_LE(maximum_analytic_work_error, 5.0e-10);
  EXPECT_LE(maximum_partition_measure_difference, 5.0e-11);
  EXPECT_LE(maximum_partition_work_difference, 5.0e-10);
  EXPECT_LE(maximum_side_reversal_work_difference, 5.0e-10);
  EXPECT_LE(maximum_vertex_limit_mismatch, 5.0e-10);
  EXPECT_LE(maximum_factorized_input_dimension,
            svmp::FE::math::dense_exact_dyadic_maximum_dimension);

  if (rank == 0) {
    const auto record_real = [](const char* name, svmp::FE::Real value) {
      std::ostringstream text;
      text << std::setprecision(
                  std::numeric_limits<svmp::FE::Real>::max_digits10)
           << value;
      ::testing::Test::RecordProperty(name, text.str());
    };
    ::testing::Test::RecordProperty(
        "native_channel_mpi_partition_count",
        static_cast<int>(partition_methods.size()));
    ::testing::Test::RecordProperty(
        "native_channel_mpi_active_side_count",
        static_cast<int>(active_sides.size()));
    ::testing::Test::RecordProperty(
        "native_channel_mpi_wet_fraction_count",
        static_cast<int>(wet_fractions.size()));
    ::testing::Test::RecordProperty(
        "native_channel_mpi_overlap_layers",
        fixture::aggregation_overlap_layers);
    ::testing::Test::RecordProperty(
        "native_channel_mpi_maximum_factorized_input_dimension",
        static_cast<int>(maximum_factorized_input_dimension));
    record_real(
        "native_channel_mpi_maximum_trace_bound",
        maximum_trace_bound);
    record_real(
        "native_channel_mpi_maximum_trace_ratio",
        maximum_trace_ratio);
    ::testing::Test::RecordProperty(
        "native_channel_mpi_maximum_trace_patch_count",
        static_cast<int>(maximum_trace_patch_count));
    ::testing::Test::RecordProperty(
        "native_channel_mpi_maximum_localized_trace_patch_count",
        static_cast<int>(maximum_localized_trace_patch_count));
    ::testing::Test::RecordProperty(
        "native_channel_mpi_maximum_trace_support_overlap",
        static_cast<int>(maximum_trace_support_overlap));
    record_real(
        "native_channel_mpi_maximum_partition_measure_difference",
        maximum_partition_measure_difference);
    record_real(
        "native_channel_mpi_maximum_partition_work_difference",
        maximum_partition_work_difference);
    record_real(
        "native_channel_mpi_maximum_side_reversal_work_difference",
        maximum_side_reversal_work_difference);
    record_real(
        "native_channel_mpi_maximum_vertex_limit_mismatch",
        maximum_vertex_limit_mismatch);
    std::cout
        << std::setprecision(
               std::numeric_limits<svmp::FE::Real>::max_digits10)
        << "native_channel_mpi_summary partitions="
        << partition_methods.size()
        << " active_sides=" << active_sides.size()
        << " wet_fractions=" << wet_fractions.size()
        << " overlap_layers=" << fixture::aggregation_overlap_layers
        << " maximum_trace_bound=" << maximum_trace_bound
        << " maximum_trace_ratio=" << maximum_trace_ratio
        << " maximum_factorized_input_dimension="
        << maximum_factorized_input_dimension
        << " maximum_trace_patches=" << maximum_trace_patch_count
        << " maximum_localized_trace_patches="
        << maximum_localized_trace_patch_count
        << " maximum_trace_support_overlap="
        << maximum_trace_support_overlap
        << " maximum_partition_measure_difference="
        << maximum_partition_measure_difference
        << " maximum_partition_work_difference="
        << maximum_partition_work_difference
        << " maximum_side_reversal_work_difference="
        << maximum_side_reversal_work_difference
        << " maximum_vertex_limit_mismatch="
        << maximum_vertex_limit_mismatch << '\n';
  }
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
