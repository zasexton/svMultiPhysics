#include <gtest/gtest.h>

// The workflow helpers exercised here currently live in ApplicationDriver.cpp's
// anonymous namespace; include the implementation to test them without
// widening the production API.
#include "../../Core/ApplicationDriver.cpp"

#include "FE/Assembly/AssemblyContext.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Assembly/AssemblyKernel.h"
#include "FE/Backends/Interfaces/BackendFactory.h"
#include "FE/Backends/Interfaces/BackendKind.h"
#include "FE/Interfaces/LevelSetInterfaceBuilder.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/ProductSpace.h"
#include "FE/Spaces/SpaceFactory.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Fields/MeshFields.h"
#include "Mesh/Mesh.h"
#include "Parameters.h"
#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "tinyxml2.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace channel_ns =
    svmp::Physics::formulations::navier_stokes;

std::shared_ptr<svmp::Mesh> makeWorkflowTriangleMesh()
{
  auto base = std::make_shared<svmp::MeshBase>();

  const std::vector<svmp::real_t> x_ref = {
      0.0, 0.0,
      1.0, 0.0,
      0.0, 1.0,
  };
  const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 3};
  const std::vector<svmp::index_t> cell2vertex = {0, 1, 2};

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Triangle;
  shape.num_corners = 3;
  shape.order = 1;
  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      {shape});
  base->finalize();

  return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::Mesh> makeWorkflowSkewedExtensionTriangleMesh(
    double upper_y = 2.0)
{
  auto base = std::make_shared<svmp::MeshBase>();

  // The dry vertex is outside the tangential span of both wet neighbors.
  // An affine tangential regression therefore requires a negative weight,
  // while the bounded fallback remains a positive partition of unity.
  const std::vector<svmp::real_t> x_ref = {
      0.0, 1.0,
      1.0, 0.0,
      0.0, upper_y,
  };
  const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 3};
  const std::vector<svmp::index_t> cell2vertex = {0, 1, 2};

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Triangle;
  shape.num_corners = 3;
  shape.order = 1;
  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      {shape});
  base->finalize();

  return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::Mesh> makeWorkflowBiquadraticQuadMesh()
{
  auto base = std::make_shared<svmp::MeshBase>();

  const std::vector<svmp::real_t> x_ref = {
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,
      0.5, 0.0,
      1.0, 0.5,
      0.5, 1.0,
      0.0, 0.5,
      0.5, 0.5,
  };
  const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 9};
  const std::vector<svmp::index_t> cell2vertex = {
      0, 1, 2, 3, 4, 5, 6, 7, 8};

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Quad;
  shape.num_corners = 4;
  shape.order = 2;
  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      {shape});
  base->finalize();

  return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::Mesh> makeDisconnectedWorkflowQuadPairMesh()
{
  auto base = std::make_shared<svmp::MeshBase>();

  // Keep the two components geometrically close while retaining distinct
  // topology.  A nearest-point extension can then accidentally copy data
  // between components, whereas a cell-graph normal-band extension cannot.
  const std::vector<svmp::real_t> x_ref = {
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,
      0.0, 1.05,
      1.0, 1.05,
      1.0, 2.05,
      0.0, 2.05,
  };
  const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4, 8};
  const std::vector<svmp::index_t> cell2vertex = {
      0, 1, 2, 3,
      4, 5, 6, 7,
  };

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Quad;
  shape.num_corners = 4;
  shape.order = 1;
  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      {shape, shape});
  base->finalize();

  return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::Mesh> makeWorkflowThreeQuadStripMesh()
{
  auto base = std::make_shared<svmp::MeshBase>();

  const std::vector<svmp::real_t> x_ref = {
      0.0, 0.0,
      1.0, 0.0,
      2.0, 0.0,
      3.0, 0.0,
      0.0, 1.0,
      1.0, 1.0,
      2.0, 1.0,
      3.0, 1.0,
  };
  const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4, 8, 12};
  const std::vector<svmp::index_t> cell2vertex = {
      0, 1, 5, 4,
      1, 2, 6, 5,
      2, 3, 7, 6,
  };

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Quad;
  shape.num_corners = 4;
  shape.order = 1;
  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      {shape, shape, shape});
  base->finalize();

  return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::Mesh> makeWorkflowFlatCapillaryFanMesh(int normal_axis)
{
  if (normal_axis != 0 && normal_axis != 1) {
    throw std::invalid_argument(
        "flat capillary fan normal axis must be zero or one");
  }
  auto base = std::make_shared<svmp::MeshBase>();

  // The off-interface interior vertex gives the order-one velocity space an
  // unconstrained volume-variation direction.  The optional proper rotation
  // exchanges the interface-normal coordinate without reversing cell
  // orientation.  The matrix selects the level-set sign so every triangle
  // retains liquid support and the active pressure space contains a constant.
  std::vector<svmp::real_t> x_ref = {
      0.0, 0.0,
      3.0, 0.0,
      3.0, 1.0,
      0.0, 1.0,
      1.5, 0.25,
  };
  if (normal_axis == 0) {
    for (std::size_t vertex = 0u; vertex < x_ref.size() / 2u; ++vertex) {
      const auto x = x_ref[2u * vertex];
      const auto y = x_ref[2u * vertex + 1u];
      x_ref[2u * vertex] = y;
      x_ref[2u * vertex + 1u] = 3.0 - x;
    }
  }
  const std::vector<svmp::offset_t> cell2vertex_offsets = {
      0, 3, 6, 9, 12};
  const std::vector<svmp::index_t> cell2vertex = {
      0, 1, 4,
      1, 2, 4,
      2, 3, 4,
      3, 0, 4,
  };

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Triangle;
  shape.num_corners = 3;
  shape.order = 1;
  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      {shape, shape, shape, shape});
  base->finalize();

  return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::Mesh> makeWorkflowHydrostaticPressureMesh(
    int normal_axis)
{
  if (normal_axis != 0 && normal_axis != 1) {
    throw std::invalid_argument(
        "hydrostatic pressure mesh normal axis must be zero or one");
  }
  auto base = std::make_shared<svmp::MeshBase>();

  // Use several active vertex layers and mildly irregular interior tangent
  // coordinates so the fixed-gauge P1 pressure field has more admissible
  // velocity-test rows than pressure unknowns and no fan-mesh checkerboard
  // mode.  The coordinate layers remain aligned with the selected normal so
  // the hydrostatic pressure field is represented exactly.
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

  std::vector<svmp::real_t> x_ref;
  x_ref.reserve(2u * columns * y_rows.size());
  for (std::size_t row = 0u; row < y_rows.size(); ++row) {
    for (std::size_t column = 0u; column < columns; ++column) {
      x_ref.push_back(x_rows[row][column]);
      x_ref.push_back(y_rows[row]);
    }
  }
  if (normal_axis == 0) {
    for (std::size_t vertex = 0u; vertex < x_ref.size() / 2u; ++vertex) {
      const auto tangent_coordinate = x_ref[2u * vertex];
      const auto normal_coordinate = x_ref[2u * vertex + 1u];
      x_ref[2u * vertex] = normal_coordinate;
      x_ref[2u * vertex + 1u] = 3.0 - tangent_coordinate;
    }
  }

  std::vector<svmp::offset_t> cell2vertex_offsets{0};
  std::vector<svmp::index_t> cell2vertex;
  std::vector<svmp::CellShape> cell_shapes;
  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Triangle;
  shape.num_corners = 3;
  shape.order = 1;
  for (std::size_t row = 0u; row + 1u < y_rows.size(); ++row) {
    for (std::size_t column = 0u; column + 1u < columns; ++column) {
      const auto lower_left =
          static_cast<svmp::index_t>(row * columns + column);
      const auto lower_right = lower_left + 1;
      const auto upper_left = lower_left +
                              static_cast<svmp::index_t>(columns);
      const auto upper_right = upper_left + 1;
      if ((row + column) % 2u == 0u) {
        cell2vertex.insert(cell2vertex.end(),
                           {lower_left, lower_right, upper_right,
                            lower_left, upper_right, upper_left});
      } else {
        cell2vertex.insert(cell2vertex.end(),
                           {lower_left, lower_right, upper_left,
                            lower_right, upper_right, upper_left});
      }
      cell2vertex_offsets.push_back(
          static_cast<svmp::offset_t>(cell2vertex.size() - 3u));
      cell2vertex_offsets.push_back(
          static_cast<svmp::offset_t>(cell2vertex.size()));
      cell_shapes.push_back(shape);
      cell_shapes.push_back(shape);
    }
  }

  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      cell_shapes);
  base->finalize();
  return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::Mesh> makeWorkflowHydrostaticPressureMesh3D(
    int normal_axis)
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
  const auto vertex_index = [&](std::size_t first_tangent,
                                std::size_t second_tangent,
                                std::size_t normal) {
    return static_cast<svmp::index_t>(
        first_tangent +
        first_tangent_coordinates.size() *
            (second_tangent +
             second_tangent_coordinates.size() * normal));
  };

  std::vector<svmp::real_t> coordinates;
  coordinates.reserve(
      3u * first_tangent_coordinates.size() *
      second_tangent_coordinates.size() * normal_coordinates.size());
  // Keep every exterior wall planar while perturbing the interior tangent
  // line across normal layers to avoid a tensor-product pressure mode.
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
        coordinates.insert(coordinates.end(), point.begin(), point.end());
      }
    }
  }

  std::vector<svmp::offset_t> cell_offsets{0};
  std::vector<svmp::index_t> cell_vertices;
  std::vector<svmp::CellShape> cell_shapes;
  const auto cell_count =
      (first_tangent_coordinates.size() - 1u) *
      (second_tangent_coordinates.size() - 1u) *
      (normal_coordinates.size() - 1u) * tetrahedra.size();
  cell_offsets.reserve(cell_count + 1u);
  cell_vertices.reserve(4u * cell_count);
  cell_shapes.reserve(cell_count);
  for (std::size_t normal = 0u;
       normal + 1u < normal_coordinates.size();
       ++normal) {
    for (std::size_t second_tangent = 0u;
         second_tangent + 1u < second_tangent_coordinates.size();
         ++second_tangent) {
      for (std::size_t first_tangent = 0u;
           first_tangent + 1u < first_tangent_coordinates.size();
           ++first_tangent) {
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
            cell_vertices.push_back(nodes[local_vertex]);
          }
          cell_offsets.push_back(
              static_cast<svmp::offset_t>(cell_vertices.size()));
          cell_shapes.push_back(
              svmp::CellShape{svmp::CellFamily::Tetra, 4, 1});
        }
      }
    }
  }

  auto base = std::make_shared<svmp::MeshBase>();
  base->build_from_arrays(
      /*spatial_dim=*/3,
      coordinates,
      cell_offsets,
      cell_vertices,
      cell_shapes);
  base->finalize();
  return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::Mesh> makeWorkflowFourQuadStripMesh(
    bool reverse_vertex_numbering = false)
{
  auto base = std::make_shared<svmp::MeshBase>();

  std::vector<svmp::real_t> x_ref = {
      0.0, 0.0,
      1.0, 0.0,
      2.0, 0.0,
      3.0, 0.0,
      4.0, 0.0,
      0.0, 1.0,
      1.0, 1.0,
      2.0, 1.0,
      3.0, 1.0,
      4.0, 1.0,
  };
  const std::vector<svmp::offset_t> cell2vertex_offsets = {
      0, 4, 8, 12, 16};
  std::vector<svmp::index_t> cell2vertex = {
      0, 1, 6, 5,
      1, 2, 7, 6,
      2, 3, 8, 7,
      3, 4, 9, 8,
  };
  if (reverse_vertex_numbering) {
    constexpr std::size_t vertex_count = 10u;
    std::vector<svmp::real_t> reversed(x_ref.size(), 0.0);
    for (std::size_t old_vertex = 0u;
         old_vertex < vertex_count;
         ++old_vertex) {
      const std::size_t new_vertex = vertex_count - 1u - old_vertex;
      reversed[2u * new_vertex] = x_ref[2u * old_vertex];
      reversed[2u * new_vertex + 1u] = x_ref[2u * old_vertex + 1u];
    }
    x_ref = std::move(reversed);
    for (auto& vertex : cell2vertex) {
      vertex = static_cast<svmp::index_t>(
          vertex_count - 1u - static_cast<std::size_t>(vertex));
    }
  }

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Quad;
  shape.num_corners = 4;
  shape.order = 1;
  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      {shape, shape, shape, shape});
  base->finalize();

  return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::Mesh> makeWorkflowStructuredQuadMesh(int subdivisions)
{
  if (subdivisions <= 0) {
    throw std::invalid_argument(
        "structured workflow mesh requires positive subdivisions");
  }

  auto base = std::make_shared<svmp::MeshBase>();
  const auto vertex_extent = static_cast<std::size_t>(subdivisions + 1);
  std::vector<svmp::real_t> x_ref;
  x_ref.reserve(2u * vertex_extent * vertex_extent);
  for (int row = 0; row <= subdivisions; ++row) {
    for (int column = 0; column <= subdivisions; ++column) {
      x_ref.push_back(static_cast<svmp::real_t>(column) / subdivisions);
      x_ref.push_back(static_cast<svmp::real_t>(row) / subdivisions);
    }
  }

  std::vector<svmp::offset_t> cell2vertex_offsets;
  std::vector<svmp::index_t> cell2vertex;
  cell2vertex_offsets.reserve(
      static_cast<std::size_t>(subdivisions * subdivisions) + 1u);
  cell2vertex.reserve(
      4u * static_cast<std::size_t>(subdivisions * subdivisions));
  cell2vertex_offsets.push_back(0);
  for (int row = 0; row < subdivisions; ++row) {
    for (int column = 0; column < subdivisions; ++column) {
      const auto lower_left = static_cast<svmp::index_t>(
          static_cast<std::size_t>(row) * vertex_extent +
          static_cast<std::size_t>(column));
      const auto lower_right = lower_left + 1;
      const auto upper_left =
          lower_left + static_cast<svmp::index_t>(vertex_extent);
      const auto upper_right = upper_left + 1;
      cell2vertex.insert(cell2vertex.end(),
                         {lower_left,
                          lower_right,
                          upper_right,
                          upper_left});
      cell2vertex_offsets.push_back(
          static_cast<svmp::offset_t>(cell2vertex.size()));
    }
  }

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Quad;
  shape.num_corners = 4;
  shape.order = 1;
  std::vector<svmp::CellShape> shapes(
      static_cast<std::size_t>(subdivisions * subdivisions), shape);
  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      shapes);
  base->finalize();

  return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::Mesh> makeWorkflowQuadPatch2x2Mesh()
{
  auto base = std::make_shared<svmp::MeshBase>();

  const std::vector<svmp::real_t> x_ref = {
      0.0, 0.0,
      1.0, 0.0,
      2.0, 0.0,
      0.0, 1.0,
      1.0, 1.0,
      2.0, 1.0,
      0.0, 2.0,
      1.0, 2.0,
      2.0, 2.0,
  };
  const std::vector<svmp::offset_t> cell2vertex_offsets = {
      0, 4, 8, 12, 16};
  const std::vector<svmp::index_t> cell2vertex = {
      0, 1, 4, 3,
      1, 2, 5, 4,
      3, 4, 7, 6,
      4, 5, 8, 7,
  };

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Quad;
  shape.num_corners = 4;
  shape.order = 1;
  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      {shape, shape, shape, shape});
  base->finalize();

  return svmp::create_mesh(std::move(base));
}

constexpr int kNativeChannelInterfaceMarker = 27410;
constexpr int kNativeChannelInletMarker = 27411;
constexpr int kNativeChannelOutletMarker = 27412;
constexpr int kNativeChannelSideWallMarker = 27413;
constexpr int kNativeChannelAnchorMarker = 27414;
constexpr int kNativeChannelOtherMarker = 27415;
constexpr svmp::FE::Real kNativeChannelLength = 2.0;
constexpr svmp::FE::Real kNativeChannelWindowHeight = 1.0;
constexpr svmp::FE::Real kNativeChannelDepth = 1.0;

std::shared_ptr<svmp::Mesh> makeNativeManufacturedChannelMesh(
    int upper_subdivisions)
{
  if (upper_subdivisions < 2) {
    throw std::invalid_argument(
        "native manufactured channel requires at least two upper layers");
  }

  constexpr int cells_x = 2;
  constexpr int cells_z = 1;
  std::vector<svmp::real_t> y_coordinates;
  y_coordinates.reserve(static_cast<std::size_t>(upper_subdivisions + 4));
  y_coordinates.push_back(-2.0);
  y_coordinates.push_back(-1.0);
  for (int layer = 0; layer <= upper_subdivisions; ++layer) {
    y_coordinates.push_back(
        static_cast<svmp::real_t>(layer) /
        static_cast<svmp::real_t>(upper_subdivisions));
  }
  y_coordinates.push_back(2.0);

  const int nodes_x = cells_x + 1;
  const int nodes_y = static_cast<int>(y_coordinates.size());
  const int nodes_z = cells_z + 1;
  const auto vertex_index =
      [nodes_x, nodes_y](int i, int j, int k) {
        return static_cast<svmp::index_t>(
            i + nodes_x * (j + nodes_y * k));
      };

  std::vector<svmp::real_t> coordinates;
  coordinates.reserve(
      static_cast<std::size_t>(nodes_x * nodes_y * nodes_z * 3));
  for (int k = 0; k < nodes_z; ++k) {
    for (int j = 0; j < nodes_y; ++j) {
      for (int i = 0; i < nodes_x; ++i) {
        coordinates.push_back(
            kNativeChannelLength *
            static_cast<svmp::real_t>(i) /
            static_cast<svmp::real_t>(cells_x));
        coordinates.push_back(y_coordinates[static_cast<std::size_t>(j)]);
        coordinates.push_back(
            kNativeChannelDepth *
            static_cast<svmp::real_t>(k) /
            static_cast<svmp::real_t>(cells_z));
      }
    }
  }

  constexpr std::array<std::array<std::size_t, 4>, 6> tetrahedra{{
      {{0, 1, 2, 6}},
      {{0, 2, 3, 6}},
      {{0, 3, 7, 6}},
      {{0, 7, 4, 6}},
      {{0, 4, 5, 6}},
      {{0, 5, 1, 6}},
  }};
  constexpr std::array<std::array<std::size_t, 4>, 6>
      inlet_tetrahedra{{
          {{1, 0, 3, 7}},
          {{1, 0, 4, 7}},
          {{1, 2, 3, 7}},
          {{1, 2, 6, 7}},
          {{1, 5, 4, 7}},
          {{1, 5, 6, 7}},
      }};
  std::vector<svmp::offset_t> cell_offsets{0};
  std::vector<svmp::index_t> cell_vertices;
  std::vector<svmp::CellShape> cell_shapes;
  const int cells_y = nodes_y - 1;
  const auto cell_count = static_cast<std::size_t>(
      cells_x * cells_y * cells_z *
      static_cast<int>(tetrahedra.size()));
  cell_offsets.reserve(cell_count + 1u);
  cell_vertices.reserve(4u * cell_count);
  cell_shapes.reserve(cell_count);
  for (int k = 0; k < cells_z; ++k) {
    for (int j = 0; j < cells_y; ++j) {
      for (int i = 0; i < cells_x; ++i) {
        const std::array<svmp::index_t, 8> nodes{{
            vertex_index(i, j, k),
            vertex_index(i + 1, j, k),
            vertex_index(i + 1, j + 1, k),
            vertex_index(i, j + 1, k),
            vertex_index(i, j, k + 1),
            vertex_index(i + 1, j, k + 1),
            vertex_index(i + 1, j + 1, k + 1),
            vertex_index(i, j + 1, k + 1),
        }};
        // Mirror the left-half body diagonal so the leading inlet wet strip
        // is backed by a cut volume whose fraction is linear in strip width.
        // The right half uses the opposite orientation for the outlet.  Their
        // shared-face diagonal remains identical, and the side-wall strip is
        // linear on both halves.  This lets the production pruning policy
        // remove only higher-order corner remnants at the smallest samples.
        const auto& cell_tetrahedra =
            i == 0 ? inlet_tetrahedra : tetrahedra;
        for (const auto& tetrahedron : cell_tetrahedra) {
          for (const auto local_vertex : tetrahedron) {
            cell_vertices.push_back(nodes[local_vertex]);
          }
          cell_offsets.push_back(
              static_cast<svmp::offset_t>(cell_vertices.size()));
          cell_shapes.push_back(
              svmp::CellShape{svmp::CellFamily::Tetra, 4, 1});
        }
      }
    }
  }

  auto base = std::make_shared<svmp::MeshBase>();
  base->build_from_arrays(
      /*spatial_dim=*/3,
      coordinates,
      cell_offsets,
      cell_vertices,
      cell_shapes);
  base->finalize();
  base->register_label(
      "native_channel_inlet",
      static_cast<svmp::label_t>(kNativeChannelInletMarker));
  base->register_label(
      "native_channel_outlet",
      static_cast<svmp::label_t>(kNativeChannelOutletMarker));
  base->register_label(
      "native_channel_side_wall",
      static_cast<svmp::label_t>(kNativeChannelSideWallMarker));
  base->register_label(
      "native_channel_anchor",
      static_cast<svmp::label_t>(kNativeChannelAnchorMarker));
  base->register_label(
      "native_channel_other",
      static_cast<svmp::label_t>(kNativeChannelOtherMarker));

  constexpr svmp::FE::Real tolerance = 1.0e-12;
  const auto on_plane = [tolerance](svmp::FE::Real value,
                                    svmp::FE::Real target) {
    return std::abs(value - target) <= tolerance;
  };
  std::array<std::size_t, 5> marker_counts{};
  for (const auto face : base->boundary_faces()) {
    const auto vertices = base->face_vertices(face);
    if (vertices.size() != 3u) {
      throw std::runtime_error(
          "native manufactured channel has a nontriangular boundary face");
    }
    bool on_inlet = true;
    bool on_outlet = true;
    bool on_side_wall = true;
    bool on_anchor = true;
    svmp::FE::Real minimum_y =
        std::numeric_limits<svmp::FE::Real>::infinity();
    svmp::FE::Real maximum_y =
        -std::numeric_limits<svmp::FE::Real>::infinity();
    for (const auto vertex : vertices) {
      const auto point = base->get_vertex_coords(vertex);
      on_inlet = on_inlet && on_plane(point[0], 0.0);
      on_outlet =
          on_outlet && on_plane(point[0], kNativeChannelLength);
      on_side_wall =
          on_side_wall && on_plane(point[2], kNativeChannelDepth);
      on_anchor = on_anchor && on_plane(point[1], -2.0);
      minimum_y = std::min(
          minimum_y, static_cast<svmp::FE::Real>(point[1]));
      maximum_y = std::max(
          maximum_y, static_cast<svmp::FE::Real>(point[1]));
    }
    const bool in_measured_window =
        minimum_y >= -tolerance &&
        maximum_y <= kNativeChannelWindowHeight + tolerance;

    int marker = kNativeChannelOtherMarker;
    std::size_t marker_index = 4u;
    if (on_inlet && in_measured_window) {
      marker = kNativeChannelInletMarker;
      marker_index = 0u;
    } else if (on_outlet && in_measured_window) {
      marker = kNativeChannelOutletMarker;
      marker_index = 1u;
    } else if (on_side_wall && in_measured_window) {
      marker = kNativeChannelSideWallMarker;
      marker_index = 2u;
    } else if (on_anchor) {
      marker = kNativeChannelAnchorMarker;
      marker_index = 3u;
    }
    base->set_boundary_label(
        face, static_cast<svmp::label_t>(marker));
    ++marker_counts[marker_index];
  }

  const auto expected_end_faces =
      static_cast<std::size_t>(2 * upper_subdivisions);
  const auto expected_side_faces =
      static_cast<std::size_t>(
          2 * cells_x * upper_subdivisions);
  const auto expected_anchor_faces =
      static_cast<std::size_t>(2 * cells_x * cells_z);
  if (marker_counts[0] != expected_end_faces ||
      marker_counts[1] != expected_end_faces ||
      marker_counts[2] != expected_side_faces ||
      marker_counts[3] != expected_anchor_faces ||
      marker_counts[4] == 0u) {
    throw std::runtime_error(
        "native manufactured channel boundary labeling is incomplete");
  }

  const auto phi_handle = svmp::MeshFields::attach_field(
      *base,
      svmp::EntityKind::Vertex,
      "phi_native_channel",
      svmp::FieldScalarType::Float64,
      1);
  auto* phi_values =
      svmp::MeshFields::field_data_as<svmp::real_t>(
          *base, phi_handle);
  if (phi_values == nullptr) {
    throw std::runtime_error(
        "native manufactured channel level-set field allocation failed");
  }
  for (svmp::index_t vertex = 0;
       vertex < static_cast<svmp::index_t>(base->n_vertices());
       ++vertex) {
    phi_values[static_cast<std::size_t>(vertex)] =
        base->get_vertex_coords(vertex)[1] - 0.5;
  }

  return svmp::create_mesh(std::move(base));
}

std::array<svmp::FE::Real, 3> workflowVertexPoint(const svmp::Mesh& mesh,
                                                  std::size_t vertex)
{
  const auto& coords = mesh.X_ref();
  const int dim = mesh.dim();
  std::array<svmp::FE::Real, 3> point{0.0, 0.0, 0.0};
  for (int d = 0; d < dim; ++d) {
    point[static_cast<std::size_t>(d)] =
        static_cast<svmp::FE::Real>(
            coords[vertex * static_cast<std::size_t>(dim) +
                   static_cast<std::size_t>(d)]);
  }
  return point;
}

svmp::FE::Real workflowPhi(const svmp::Mesh& mesh, std::size_t vertex)
{
  const auto point = workflowVertexPoint(mesh, vertex);
  return point[0] - svmp::FE::Real{0.25};
}

svmp::FE::Real workflowVerticalPhi(const svmp::Mesh& mesh, std::size_t vertex)
{
  const auto point = workflowVertexPoint(mesh, vertex);
  return point[1] - svmp::FE::Real{0.75};
}

std::array<svmp::FE::Real, 2> workflowVelocity(const svmp::Mesh& mesh,
                                               std::size_t vertex)
{
  const auto point = workflowVertexPoint(mesh, vertex);
  const auto x = point[0];
  const auto y = point[1];
  return {svmp::FE::Real{2.0} + svmp::FE::Real{3.0} * x - y +
              svmp::FE::Real{0.25} * x * y,
          svmp::FE::Real{-1.0} + svmp::FE::Real{0.5} * x +
              svmp::FE::Real{2.0} * y};
}

std::vector<svmp::FE::Real> projectWorkflowVertexValues(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::FieldId field,
    std::span<const svmp::FE::Real> vertex_values,
    std::size_t components,
    std::string_view context)
{
  const auto n_dofs =
      static_cast<std::size_t>(system.fieldDofHandler(field).getNumDofs());
  std::vector<svmp::FE::Real> coefficients(n_dofs, 0.0);
  std::vector<std::uint8_t> assigned(n_dofs, 0u);
  const auto projection = system.projectMeshVertexValuesToFieldCoefficients(
      field,
      vertex_values,
      components,
      std::span<svmp::FE::Real>(coefficients.data(), coefficients.size()),
      std::span<std::uint8_t>(assigned.data(), assigned.size()),
      context);
  if (projection.unassigned_dofs != 0u ||
      projection.values_written != n_dofs) {
    throw std::runtime_error(
        std::string(context) + ": incomplete workflow projection");
  }
  return coefficients;
}

void writeWorkflowFieldSlice(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::FieldId field,
    std::span<const svmp::FE::Real> coefficients,
    std::vector<svmp::FE::Real>& solution)
{
  const auto offset = system.fieldDofOffset(field);
  if (offset < 0 ||
      static_cast<std::size_t>(offset) + coefficients.size() >
          solution.size()) {
    throw std::runtime_error("workflow test field slice is outside solution");
  }
  for (std::size_t i = 0; i < coefficients.size(); ++i) {
    solution[static_cast<std::size_t>(offset) + i] = coefficients[i];
  }
}

std::unique_ptr<Parameters> parseWorkflowParametersXml(const char* xml)
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

class WorkflowScopedEnvVar {
public:
  WorkflowScopedEnvVar(const char* key, std::optional<std::string> value)
      : key_(key)
  {
    if (const char* old = std::getenv(key)) {
      original_ = std::string(old);
    }
    set(std::move(value));
  }

  ~WorkflowScopedEnvVar() { set(original_); }

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

struct NativeManufacturedChannelSample {
  svmp::FE::Real target_wet_fraction{0.0};
  std::array<svmp::FE::Real, 3> active_measures{};
  std::array<svmp::FE::Real, 3> parent_measures{};
  std::array<std::size_t, 3> active_rule_counts{};
  std::array<std::size_t, 3> retained_active_rule_counts{};
  std::array<int, 3> active_markers{{-1, -1, -1}};
  std::array<svmp::FE::Real, 3> operator_work{};
  std::array<std::size_t, 3> generated_route_term_counts{};
  std::size_t physical_role_boundary_term_count{0u};
  std::size_t trace_certificate_count{0u};
  std::size_t trace_patch_count{0u};
  std::size_t trace_localized_support_patch_count{0u};
  std::size_t trace_localized_root_patch_count{0u};
  std::size_t trace_maximum_factorized_input_dimension{0u};
  std::size_t trace_boundary_rule_count{0u};
  std::uint64_t trace_certificate_digest{0u};
  svmp::FE::Real trace_global_conservative_upper_bound{0.0};
  svmp::FE::Real trace_maximum_patch_conservative_upper_bound{0.0};
  svmp::FE::Real trace_to_penalty_ratio{0.0};
  svmp::FE::Real trace_grouped_symmetric_ratio{0.0};
  svmp::FE::Real trace_symmetric_energy_floor{0.0};
  std::size_t trace_maximum_support_overlap{0u};
  bool trace_revision_match{false};
  bool trace_factorized_proof_valid{false};
};

class NativeManufacturedChannelHarness {
public:
  using ActiveSide = svmp::FE::geometry::CutIntegrationSide;

  inline static constexpr svmp::FE::Real inlet_traction = 1.25;
  inline static constexpr svmp::FE::Real outlet_pressure = 1.2;
  inline static constexpr svmp::FE::Real prescribed_side_velocity = 0.4;
  inline static constexpr svmp::FE::Real viscosity = 0.02;
  inline static constexpr svmp::FE::Real nitsche_gamma = 16.0;
  inline static constexpr svmp::FE::Real side_facet_normal_scale = 2.0 / 3.0;

  NativeManufacturedChannelHarness(ActiveSide active_side,
                                   int upper_subdivisions)
      : active_side_(active_side),
        upper_subdivisions_(upper_subdivisions),
        mesh_(makeNativeManufacturedChannelMesh(upper_subdivisions)),
        system_(std::make_unique<svmp::FE::systems::FESystem>(mesh_))
  {
    if (active_side_ != ActiveSide::Negative &&
        active_side_ != ActiveSide::Positive) {
      throw std::invalid_argument(
          "native manufactured channel requires a volume active side");
    }

    auto pressure_space = svmp::FE::spaces::SpaceFactory::create_h1(
        svmp::FE::ElementType::Tetra4,
        /*order=*/1);
    auto velocity_space =
        svmp::FE::spaces::SpaceFactory::create_vector_h1(
            svmp::FE::ElementType::Tetra4,
            /*order=*/1,
            /*components=*/3);
    level_set_ = system_->addField(svmp::FE::systems::FieldSpec{
        .name = "phi_native_channel",
        .space = pressure_space,
        .components = 1,
    });

    channel_ns::IncompressibleNavierStokesVMSOptions options;
    options.symmetric_nitsche_energy_qualification_scope =
        channel_ns::SymmetricNitscheEnergyQualificationScope::
            JointLowLevelPrerequisite;
    options.velocity_field_name = "u_native_channel";
    options.pressure_field_name = "p_native_channel";
    options.density = 1.0;
    options.viscosity = viscosity;
    options.enable_convection = false;
    options.enable_vms = false;
    options.jit_policy.enable = false;
    options.velocity_dirichlet.push_back(
        channel_ns::IncompressibleNavierStokesVMSOptions::
            VelocityDirichletBC{
                .boundary_marker = kNativeChannelAnchorMarker,
                .value = {0.0, 0.0, 0.0},
            });
    options.velocity_dirichlet_weak.push_back(
        channel_ns::IncompressibleNavierStokesVMSOptions::
            VelocityDirichletBC{
                .boundary_marker = kNativeChannelSideWallMarker,
                .value = {0.0, 0.0, prescribed_side_velocity},
            });
    options.traction_neumann.push_back(
        channel_ns::IncompressibleNavierStokesVMSOptions::
            TractionNeumannBC{
                .boundary_marker = kNativeChannelInletMarker,
                .traction = {0.0, inlet_traction, 0.0},
            });
    options.pressure_outflow.push_back(
        channel_ns::IncompressibleNavierStokesVMSOptions::
            PressureOutflowBC{
                .boundary_marker = kNativeChannelOutletMarker,
                .pressure = outlet_pressure,
                .backflow_beta = 0.0,
            });
    options.free_surface.push_back(
        channel_ns::IncompressibleNavierStokesVMSOptions::
            FreeSurfaceBoundary{
                .implementation =
                    channel_ns::FreeSurfaceImplementation::UnfittedLevelSet,
                .interface_marker = kNativeChannelInterfaceMarker,
                .level_set_field_name = "phi_native_channel",
                .generated_interface_domain_id =
                    "native_manufactured_channel",
                .generated_interface_geometry = "LinearCorner",
                .level_set_isovalue = 0.0,
                .active_domain =
                    active_side_ == ActiveSide::Negative
                        ? channel_ns::FreeSurfaceActiveDomain::LevelSetNegative
                        : channel_ns::FreeSurfaceActiveDomain::LevelSetPositive,
                .active_domain_method =
                    channel_ns::FreeSurfaceActiveDomainMethod::CutVolume,
                .external_pressure = 0.0,
                .surface_tension = 0.0,
                .use_level_set_curvature = false,
                .cut_cell_stabilization = {
                    .enabled = false,
                },
                .small_cut_aggregation = true,
            });
    options.nitsche_gamma = nitsche_gamma;
    options.nitsche_symmetric = true;
    options.nitsche_scale_with_p = false;

    channel_ns::IncompressibleNavierStokesVMSModule module(
        velocity_space, pressure_space, std::move(options));
    module.registerOn(*system_);
    system_->setup({});
    velocity_ = system_->findFieldByName("u_native_channel");
    pressure_ = system_->findFieldByName("p_native_channel");
    if (velocity_ == svmp::FE::INVALID_FIELD_ID ||
        pressure_ == svmp::FE::INVALID_FIELD_ID) {
      throw std::runtime_error(
          "native manufactured channel fluid fields are unavailable");
    }

    solution_.assign(
        static_cast<std::size_t>(system_->dofHandler().getNumDofs()),
        0.0);
    previous_ = solution_;
    probes_[0] = constantVelocityProbe({0.0, 1.0, 0.0});
    probes_[1] = constantVelocityProbe({1.0, 0.0, 0.0});
    probes_[2] = constantVelocityProbe({0.0, 0.0, 1.0});

    const char* active_side_token =
        active_side_ == ActiveSide::Negative
            ? "LevelSetNegative"
            : "LevelSetPositive";
    const std::string xml =
        std::string(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="native_channel_free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi_native_channel</Level_set_field_name>
      <Generated_interface_domain_id>native_manufactured_channel</Generated_interface_domain_id>
      <Interface_marker>27410</Interface_marker>
      <Generated_interface_geometry>LinearCorner</Generated_interface_geometry>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>)xml") +
        active_side_token +
        R"xml(</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
      <Small_cut_aggregation>true</Small_cut_aggregation>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml";
    params_ = parseWorkflowParametersXml(xml.c_str());

    sim_.primary_mesh = mesh_;
    sim_.fe_system = std::move(system_);
  }

  [[nodiscard]] NativeManufacturedChannelSample sample(
      svmp::FE::Real wet_fraction)
  {
    if (!(wet_fraction >= 0.0) || !(wet_fraction <= 1.0) ||
        !std::isfinite(wet_fraction)) {
      throw std::invalid_argument(
          "native manufactured channel wet fraction is outside [0,1]");
    }
    const svmp::FE::Real exterior_offset =
        0.05 / static_cast<svmp::FE::Real>(upper_subdivisions_);
    const svmp::FE::Real interface_height =
        wet_fraction == 0.0
            ? -exterior_offset
            : wet_fraction == 1.0
                  ? kNativeChannelWindowHeight + exterior_offset
                  : wet_fraction;
    std::vector<svmp::FE::Real> level_set_vertex_values(
        mesh_->n_vertices(), 0.0);
    for (std::size_t vertex = 0; vertex < mesh_->n_vertices(); ++vertex) {
      const auto y = workflowVertexPoint(*mesh_, vertex)[1];
      level_set_vertex_values[vertex] =
          active_side_ == ActiveSide::Negative
              ? y - interface_height
              : interface_height - y;
    }
    const auto level_set_coefficients = projectWorkflowVertexValues(
        *sim_.fe_system,
        level_set_,
        level_set_vertex_values,
        1u,
        "native manufactured channel level set");
    writeWorkflowFieldSlice(
        *sim_.fe_system,
        level_set_,
        level_set_coefficients,
        solution_);
    previous_ = solution_;

    const auto refresh_report =
        refreshActiveCutIntegrationContextFromSolution(
            sim_,
            *params_,
            solution_,
            lifecycle_,
            "native-manufactured-channel-test");
    if (!refresh_report.refreshed ||
        refresh_report.value_revision == 0u) {
      throw std::runtime_error(
          "native manufactured channel refresh produced no revision");
    }
    const auto* context = sim_.fe_system->cutIntegrationContext();
    if (context == nullptr ||
        context->freeSurfaceGeometrySnapshots().size() != 1u) {
      throw std::runtime_error(
          "native manufactured channel has no unique geometry snapshot");
    }
    const auto snapshot =
        context->freeSurfaceGeometrySnapshots().front();
    if (!snapshot) {
      throw std::runtime_error(
          "native manufactured channel geometry snapshot is null");
    }
    if (snapshot->interfaceDomain()
            .request()
            .aligned_zero_interface_parent_side != active_side_) {
      throw std::runtime_error(
          "native manufactured channel aligned interface parent side is stale");
    }

    NativeManufacturedChannelSample result;
    result.target_wet_fraction = wet_fraction;
    constexpr std::array<int, 3> physical_markers{{
        kNativeChannelInletMarker,
        kNativeChannelOutletMarker,
        kNativeChannelSideWallMarker,
    }};
    for (std::size_t role = 0u; role < physical_markers.size(); ++role) {
      const svmp::FE::interfaces::GeneratedActiveBoundaryDomain*
          negative = nullptr;
      const svmp::FE::interfaces::GeneratedActiveBoundaryDomain*
          positive = nullptr;
      for (const auto& active : snapshot->activeBoundaryDomains()) {
        if (active.request().boundary_marker != physical_markers[role]) {
          continue;
        }
        if (active.request().side == ActiveSide::Negative) {
          negative = &active;
        } else if (active.request().side == ActiveSide::Positive) {
          positive = &active;
        }
      }
      if (negative == nullptr || positive == nullptr) {
        throw std::runtime_error(
            "native manufactured channel boundary partition is incomplete");
      }
      const auto& selected =
          active_side_ == ActiveSide::Negative ? *negative : *positive;
      const auto selected_rule_role =
          active_side_ == ActiveSide::Negative
              ? svmp::FE::interfaces::
                    FreeSurfaceGeometryRuleRole::NegativeExteriorBoundary
              : svmp::FE::interfaces::
                    FreeSurfaceGeometryRuleRole::PositiveExteriorBoundary;
      for (const auto& record : snapshot->rules()) {
        if (record.physical_boundary_marker != physical_markers[role] ||
            (record.role != svmp::FE::interfaces::
                                FreeSurfaceGeometryRuleRole::
                                    NegativeExteriorBoundary &&
             record.role != svmp::FE::interfaces::
                                FreeSurfaceGeometryRuleRole::
                                    PositiveExteriorBoundary)) {
          continue;
        }
        result.parent_measures[role] +=
            record.physical_rule.physical_measure;
        if (record.role == selected_rule_role) {
          result.active_measures[role] +=
              record.physical_rule.physical_measure;
        }
      }
      result.active_rule_counts[role] =
          selected.boundaryQuadratureRules().size();
      result.active_markers[role] = selected.marker();
      result.retained_active_rule_counts[role] =
          context->interfaceRulesForMarker(
              result.active_markers[role]).size();
    }

    const auto& definition =
        sim_.fe_system->operatorDefinition("equations");
    for (const auto& term : definition.boundary) {
      result.physical_role_boundary_term_count +=
          static_cast<std::size_t>(
              std::find(
                  physical_markers.begin(),
                  physical_markers.end(),
                  term.marker) != physical_markers.end());
    }
    for (const auto& term : definition.interface_faces) {
      for (std::size_t role = 0u; role < physical_markers.size(); ++role) {
        result.generated_route_term_counts[role] +=
            static_cast<std::size_t>(
                term.marker == result.active_markers[role]);
      }
    }

    const auto trace_records =
        sim_.fe_system->generatedBoundaryNitscheTraceCertificates();
    result.trace_certificate_count = trace_records.size();
    if (trace_records.size() != 1u) {
      throw std::runtime_error(
          "native manufactured channel requires one trace certificate");
    }
    const auto& trace = trace_records.front();
    result.trace_patch_count =
        trace.certificate.certified_patch_count;
    result.trace_localized_support_patch_count =
        trace.certificate.localized_support_patch_count;
    for (const auto& patch : trace.certificate.patches) {
      result.trace_localized_root_patch_count +=
          static_cast<std::size_t>(
              patch.localized_support_patch &&
              patch.support_cell_gids.size() > 1u);
      result.trace_maximum_factorized_input_dimension =
          std::max(
              result.trace_maximum_factorized_input_dimension,
              patch.generalized_bound.exact_dyadic
                  .factorized_input_dimension);
    }
    result.trace_boundary_rule_count =
        trace.certificate.generated_boundary_rule_count;
    result.trace_certificate_digest =
        trace.certificate.canonical_certificate_digest;
    result.trace_global_conservative_upper_bound =
        trace.certificate.global_conservative_upper_bound;
    result.trace_maximum_patch_conservative_upper_bound =
        trace.certificate.maximum_patch_conservative_upper_bound;
    result.trace_to_penalty_ratio =
        trace.trace_to_penalty_ratio;
    result.trace_grouped_symmetric_ratio =
        trace.grouped_symmetric_trace_to_penalty_ratio;
    result.trace_symmetric_energy_floor =
        trace.symmetric_energy_ratio_lower_bound.value_or(0.0);
    result.trace_maximum_support_overlap =
        trace.certificate.maximum_support_overlap;
    result.trace_revision_match =
        trace.policy.physical_boundary_marker ==
            kNativeChannelSideWallMarker &&
        trace.policy.generated_active_boundary_marker ==
            result.active_markers[2] &&
        trace.certificate.cut_context_content_revision ==
            context->contentRevision() &&
        trace.certificate.free_surface_snapshot_revision ==
            snapshot->revision().snapshot_revision_key &&
        trace.certificate.source_value_revision ==
            refresh_report.value_revision &&
        trace.aggregation_report != nullptr;
    result.trace_factorized_proof_valid =
        trace.certificate.patches.size() ==
            trace.certificate.certified_patch_count &&
        result.trace_localized_support_patch_count ==
            static_cast<std::size_t>(std::count_if(
                trace.certificate.patches.begin(),
                trace.certificate.patches.end(),
                [](const auto& patch) {
                  return patch.localized_support_patch;
                })) &&
        std::all_of(
            trace.certificate.patches.begin(),
            trace.certificate.patches.end(),
            [](const auto& patch) {
              const auto& exact = patch.generalized_bound.exact_dyadic;
              return exact.applied &&
                     exact.denominator_positive_definite_proven &&
                     exact.numerator_positive_semidefinite_proven &&
                     exact.upper_inequality_proven &&
                     exact.proof_input ==
                         svmp::FE::math::DenseExactDyadicProofInput::
                             FactorizedBinary64PositiveForm &&
                     exact.exact_factorized_materialization_proven &&
                     exact.exact_sparse_map_applied &&
                     exact.factorized_input_digest != 0u &&
                     exact.exact_common_kernel_proven;
            });

    const auto dof_count = sim_.fe_system->dofHandler().getNumDofs();
    svmp::FE::assembly::DenseVectorView residual(dof_count);
    residual.zero();
    svmp::FE::systems::SystemStateView state;
    state.dt = 1.0;
    state.u = solution_;
    state.u_prev = previous_;
    const svmp::FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context = integrator.buildContext(1, state);
    state.time_integration = &time_context;
    svmp::FE::systems::AssemblyRequest request;
    request.op = "equations";
    request.want_matrix = false;
    request.want_vector = true;
    const auto assembly =
        sim_.fe_system->assemble(request, state, nullptr, &residual);
    if (!assembly.success) {
      throw std::runtime_error(
          "native manufactured channel assembly failed: " +
          assembly.error_message);
    }
    for (std::size_t role = 0u; role < probes_.size(); ++role) {
      if (probes_[role].size() !=
          static_cast<std::size_t>(dof_count)) {
        throw std::runtime_error(
            "native manufactured channel probe size is stale");
      }
      for (svmp::FE::GlobalIndex row = 0; row < dof_count; ++row) {
        result.operator_work[role] +=
            residual[row] *
            probes_[role][static_cast<std::size_t>(row)];
      }
    }
    return result;
  }

  [[nodiscard]] static constexpr svmp::FE::Real
  expectedFullForceWork() noexcept
  {
    return -inlet_traction * kNativeChannelWindowHeight *
           kNativeChannelDepth;
  }

  [[nodiscard]] static constexpr svmp::FE::Real
  expectedFullFluxWork() noexcept
  {
    return outlet_pressure * kNativeChannelWindowHeight *
           kNativeChannelDepth;
  }

  [[nodiscard]] static constexpr svmp::FE::Real
  expectedFullPenaltyWork() noexcept
  {
    return -nitsche_gamma * viscosity / side_facet_normal_scale *
           prescribed_side_velocity * kNativeChannelLength *
           kNativeChannelWindowHeight;
  }

private:
  [[nodiscard]] std::vector<svmp::FE::Real> constantVelocityProbe(
      const std::array<svmp::FE::Real, 3>& value) const
  {
    std::vector<svmp::FE::Real> vertex_values(
        3u * mesh_->n_vertices(), 0.0);
    for (std::size_t vertex = 0; vertex < mesh_->n_vertices(); ++vertex) {
      for (std::size_t component = 0u; component < value.size(); ++component) {
        vertex_values[3u * vertex + component] = value[component];
      }
    }
    const auto coefficients = projectWorkflowVertexValues(
        *system_,
        velocity_,
        vertex_values,
        3u,
        "native manufactured channel velocity probe");
    std::vector<svmp::FE::Real> probe(
        static_cast<std::size_t>(system_->dofHandler().getNumDofs()),
        0.0);
    writeWorkflowFieldSlice(*system_, velocity_, coefficients, probe);
    return probe;
  }

  ActiveSide active_side_{ActiveSide::Negative};
  int upper_subdivisions_{2};
  std::shared_ptr<svmp::Mesh> mesh_{};
  std::unique_ptr<svmp::FE::systems::FESystem> system_{};
  svmp::FE::FieldId level_set_{svmp::FE::INVALID_FIELD_ID};
  svmp::FE::FieldId velocity_{svmp::FE::INVALID_FIELD_ID};
  svmp::FE::FieldId pressure_{svmp::FE::INVALID_FIELD_ID};
  std::vector<svmp::FE::Real> solution_{};
  std::vector<svmp::FE::Real> previous_{};
  std::array<std::vector<svmp::FE::Real>, 3> probes_{};
  application::core::SimulationComponents sim_{};
  std::unique_ptr<Parameters> params_{};
  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle_{};
};

class WorkflowNoOpCellKernel final : public svmp::FE::assembly::AssemblyKernel {
public:
  [[nodiscard]] svmp::FE::assembly::RequiredData getRequiredData()
      const override
  {
    return svmp::FE::assembly::RequiredData::None;
  }

  void computeCell(const svmp::FE::assembly::AssemblyContext&,
                   svmp::FE::assembly::KernelOutput&) override
  {
  }

  [[nodiscard]] std::string name() const override
  {
    return "WorkflowNoOpCellKernel";
  }
};

class WorkflowScaledMassKernel final
    : public svmp::FE::assembly::AssemblyKernel {
public:
  WorkflowScaledMassKernel(svmp::FE::Real matrix_scale,
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
    bool want_matrix =
        output.has_matrix || !output.local_matrix.empty();
    bool want_vector =
        output.has_vector || !output.local_vector.empty();
    if (!want_matrix && !want_vector) {
      want_matrix = true;
      want_vector = true;
    }
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
    return "WorkflowScaledMassKernel";
  }

private:
  svmp::FE::Real matrix_scale_{0.0};
  svmp::FE::Real vector_scale_{0.0};
};

void installWorkflowExactConstantPressureCertificate(
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
        std::make_shared<WorkflowScaledMassKernel>(
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
      std::make_shared<WorkflowScaledMassKernel>(
          /*matrix_scale=*/1.0,
          /*vector_scale=*/0.0));
  system.addCellKernel(
      pair_operator,
      pressure,
      velocity,
      std::make_shared<WorkflowScaledMassKernel>(
          /*matrix_scale=*/1.0,
          /*vector_scale=*/0.0));
}

class WorkflowEffectiveConfigurationModule final
    : public svmp::Physics::PhysicsModule {
public:
  WorkflowEffectiveConfigurationModule(std::string component,
                                       std::string json)
      : artifact_{.component = std::move(component),
                  .json = std::move(json)}
  {
  }

  void registerOn(svmp::FE::systems::FESystem&) const override {}

  [[nodiscard]] std::optional<svmp::Physics::EffectiveConfigurationArtifact>
  effectiveConfigurationArtifact() const override
  {
    return artifact_;
  }

private:
  svmp::Physics::EffectiveConfigurationArtifact artifact_{};
};

} // namespace

TEST(ApplicationDriverLevelSetWorkflows,
     WritesOneDeterministicallyOrderedEffectiveConfigurationArtifact)
{
  const auto unique = std::chrono::steady_clock::now()
                          .time_since_epoch()
                          .count();
  const auto output_directory =
      std::filesystem::temp_directory_path() /
      ("svmp-effective-configuration-" + std::to_string(unique));
  std::filesystem::create_directories(output_directory);

  Parameters params;
  params.general_simulation_parameters.save_results_in_folder.set(
      output_directory.string());
  application::core::SimulationComponents sim;
  sim.physics_modules.push_back(
      std::make_unique<WorkflowEffectiveConfigurationModule>(
          "z_component",
          R"({"artifact_schema_version":1,"component":"z_component"})"));
  sim.physics_modules.push_back(
      std::make_unique<WorkflowEffectiveConfigurationModule>(
          "a_component",
          R"({"artifact_schema_version":1,"component":"a_component"})"));

  writeEffectiveConfigurationArtifact(
      sim, params, svmp::MeshComm::world());

  const auto artifact_path =
      output_directory / "effective_configuration.json";
  std::ifstream input(artifact_path);
  ASSERT_TRUE(input.is_open());
  const std::string contents{
      std::istreambuf_iterator<char>{input},
      std::istreambuf_iterator<char>{}};
  EXPECT_EQ(
      contents,
      "{\"artifact_schema_version\":1,\"modules\":["
      "{\"artifact_schema_version\":1,\"component\":\"a_component\"},"
      "{\"artifact_schema_version\":1,\"component\":\"z_component\"}]}\n");
  EXPECT_FALSE(std::filesystem::exists(
      output_directory / "effective_configuration.json.tmp"));

  std::error_code cleanup_error;
  std::filesystem::remove_all(output_directory, cleanup_error);
  EXPECT_FALSE(cleanup_error);
}

TEST(ApplicationDriverLevelSetWorkflows,
     MonolithicNewtonControlsHonorEveryCoupledEquation)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Coupled>true</Coupled>
    <Min_iterations>2</Min_iterations>
    <Max_iterations>4</Max_iterations>
    <Tolerance>1.0e-4</Tolerance>
  </Add_equation>
  <Add_equation type="fluid">
    <Coupled>true</Coupled>
    <Min_iterations>1</Min_iterations>
    <Max_iterations>12</Max_iterations>
    <Tolerance>2.0e-2</Tolerance>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  svmp::FE::timestepping::NewtonOptions options{};
  applyMonolithicEquationNewtonControls(*params, options);

  EXPECT_EQ(options.min_iterations, 2);
  EXPECT_EQ(options.max_iterations, 12);
  EXPECT_DOUBLE_EQ(options.rel_tolerance, 2.0e-2);
  EXPECT_DOUBLE_EQ(options.abs_tolerance, 1.0e-10);
}

TEST(ApplicationDriverLevelSetWorkflows,
     GeneralizedAlphaContactStageProvenanceIsAuthenticAndAttemptBound)
{
  std::unique_ptr<svmp::FE::backends::BackendFactory> factory;
  try {
    factory = svmp::FE::backends::BackendFactory::create(
        svmp::FE::backends::BackendKind::FSILS);
  } catch (const std::exception&) {
    GTEST_SKIP() << "Requires an available FE vector backend.";
  }
  auto history = svmp::FE::timestepping::TimeHistory::allocate(
      *factory,
      /*size=*/4,
      /*history_depth=*/2,
      /*allocate_second_order_state=*/true);
  history.setTime(0.3);
  history.setDt(0.1);
  history.setPrevDt(0.1);
  history.setStepIndex(3);

  constexpr double rho_inf = 0.2;
  const auto parameters =
      svmp::FE::timestepping::utils::
          generalizedAlphaFirstOrderFromRhoInf(rho_inf);
  ASSERT_GT(parameters.alpha_m, 1.0);
  const svmp::FE::timestepping::CandidateStageObservation observation{
      .scheme = svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
      .temporal_order = 1,
      .step_index = 4,
      .attempt_index = 2,
      .step_start_time = 0.3,
      .step_end_time = 0.4,
      .state_time = 0.3 + parameters.alpha_f * 0.1,
      .rate_time = 0.3 + parameters.alpha_m * 0.1,
      .dt = 0.1,
      .generalized_alpha =
          svmp::FE::timestepping::GeneralizedAlphaStageMetadata{
              .alpha_f = parameters.alpha_f,
              .alpha_m = parameters.alpha_m,
              .gamma = parameters.gamma,
          },
      .mesh_revision =
          svmp::FE::timestepping::CandidateStageMeshRevision{},
      .state_vector = &history.u(),
      .rate_vector = &history.uDot(),
  };
  const auto captured =
      makeDynamicContactFirstOrderGeneralizedAlphaObservation(observation);
  EXPECT_EQ(captured.attempt_index, 2);
  EXPECT_DOUBLE_EQ(captured.provenance.alpha_m, parameters.alpha_m);
  EXPECT_DOUBLE_EQ(captured.provenance.alpha_f, parameters.alpha_f);
  EXPECT_DOUBLE_EQ(captured.provenance.gamma, parameters.gamma);
  EXPECT_DOUBLE_EQ(captured.provenance.dt, 0.1);

  const auto resolved = resolveFreeSurfaceContactStageTemporalCoordinates(
      svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
      history,
      /*expected_attempt_index=*/2,
      captured);
  EXPECT_DOUBLE_EQ(resolved.stage_time, observation.state_time);
  EXPECT_DOUBLE_EQ(resolved.stage_alpha_f, parameters.alpha_f);
  ASSERT_TRUE(resolved.first_order_generalized_alpha.has_value());
  EXPECT_EQ(*resolved.first_order_generalized_alpha, captured.provenance);
  EXPECT_THROW(
      (void)resolveFreeSurfaceContactStageTemporalCoordinates(
          svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
          history,
          /*expected_attempt_index=*/3,
          captured),
      std::logic_error);

  auto invalid_observation = observation;
  invalid_observation.generalized_alpha->gamma += 0.125;
  EXPECT_THROW(
      (void)makeDynamicContactFirstOrderGeneralizedAlphaObservation(
          invalid_observation),
      std::invalid_argument);
  const svmp::FE::systems::FreeSurfaceFirstOrderGeneralizedAlphaProvenance
      unsupported_custom_parameters{
          .alpha_m = 0.75,
          .alpha_f = 0.75,
          .gamma = 0.5,
          .dt = 0.1,
      };
  EXPECT_FALSE(firstOrderGeneralizedAlphaContactProvenanceValid(
      unsupported_custom_parameters));
  for (const double supported_rho : {0.0, 0.2, 1.0}) {
    const auto supported =
        svmp::FE::timestepping::utils::
            generalizedAlphaFirstOrderFromRhoInf(supported_rho);
    EXPECT_TRUE(firstOrderGeneralizedAlphaContactProvenanceValid(
        svmp::FE::systems::
            FreeSurfaceFirstOrderGeneralizedAlphaProvenance{
                .alpha_m = supported.alpha_m,
                .alpha_f = supported.alpha_f,
                .gamma = supported.gamma,
                .dt = 0.1,
            }));
  }

  for (const auto endpoint_scheme : {
           svmp::FE::timestepping::SchemeKind::BackwardEuler,
           svmp::FE::timestepping::SchemeKind::BDF2,
           svmp::FE::timestepping::SchemeKind::VSVO_BDF}) {
    const auto endpoint = resolveFreeSurfaceContactStageTemporalCoordinates(
        endpoint_scheme,
        history,
        /*expected_attempt_index=*/2,
        std::nullopt);
    EXPECT_DOUBLE_EQ(endpoint.stage_time, 0.4);
    EXPECT_DOUBLE_EQ(endpoint.stage_alpha_f, 1.0);
    EXPECT_FALSE(endpoint.first_order_generalized_alpha.has_value());
  }
}

TEST(ApplicationDriverLevelSetWorkflows,
     GeneralizedAlphaContactStageMeshProvenanceSeparatesGeneratedState)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr int contact_wall_marker = 1701;
  constexpr int unrelated_marker = 2718;
  auto mesh = makeWorkflowTriangleMesh();
  ASSERT_GE(mesh->local_mesh().boundary_faces().size(), 2u);
  const auto contact_face =
      mesh->local_mesh().boundary_faces().front();
  const auto unrelated_face =
      mesh->local_mesh().boundary_faces()[1u];
  mesh->local_mesh().set_boundary_label(
      contact_face, contact_wall_marker);
  svmp::FE::systems::FESystem system(mesh);
  const auto& live_mesh = system.meshAccess();
  const svmp::FE::timestepping::CandidateStageMeshRevision captured{
      .geometry_revision = live_mesh.geometryRevision(),
      .topology_revision = live_mesh.topologyRevision(),
      .ownership_revision = live_mesh.ownershipRevision(),
      .numbering_revision = live_mesh.numberingRevision(),
      .field_layout_revision = live_mesh.fieldLayoutRevision(),
      .label_revision = live_mesh.labelRevision(),
      .active_configuration_epoch = live_mesh.activeConfigurationEpoch(),
      .coordinate_configuration_key =
          live_mesh.coordinateConfigurationKey(),
  };
  svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration declaration;
  declaration.parameters.dynamic_contact_coefficients.push_back({
      .boundary_marker = contact_wall_marker,
      .equilibrium_contact_angle_radians = svmp::FE::Real{1.0},
      .mobility = svmp::FE::Real{1.0},
      .slip_length = svmp::FE::Real{1.0},
      .dynamic_viscosity = svmp::FE::Real{1.0},
  });
  const std::vector<
      svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration>
      declarations{declaration};
  DynamicContactFirstOrderGeneralizedAlphaObservation observation{
      .mesh_revision = captured,
      .contact_wall_boundary_fingerprint =
          dynamicContactWallBoundaryFingerprint(
              live_mesh, declarations),
  };
  EXPECT_TRUE(dynamicContactStageMeshRevisionMatches(captured, live_mesh));
  EXPECT_TRUE(dynamicContactStageProvenanceMatches(
      observation, live_mesh, declarations));

  // Endpoint cut-domain publication owns generated mesh fields and labels.
  // Those epochs may advance between exact stage capture and reconstruction;
  // retained FE slices validate their own layout separately.
  const auto field_layout_revision = live_mesh.fieldLayoutRevision();
  (void)svmp::MeshFields::attach_field(
      mesh->local_mesh(),
      svmp::EntityKind::Vertex,
      "dynamic_contact_generated_state_probe",
      svmp::FieldScalarType::Float64,
      1);
  EXPECT_GT(live_mesh.fieldLayoutRevision(), field_layout_revision);
  EXPECT_TRUE(dynamicContactStageMeshRevisionMatches(captured, live_mesh));
  EXPECT_TRUE(dynamicContactStageProvenanceMatches(
      observation, live_mesh, declarations));

  const auto label_revision = live_mesh.labelRevision();
  mesh->local_mesh().set_boundary_label(
      unrelated_face, unrelated_marker);
  EXPECT_GT(live_mesh.labelRevision(), label_revision);
  EXPECT_TRUE(dynamicContactStageMeshRevisionMatches(captured, live_mesh));
  EXPECT_TRUE(dynamicContactStageProvenanceMatches(
      observation, live_mesh, declarations));

  // A global label epoch is too coarse to distinguish generated-domain
  // publication from a physical contact-wall change. The targeted boundary
  // fingerprint keeps the latter fail-closed.
  mesh->local_mesh().set_boundary_label(
      contact_face, unrelated_marker);
  EXPECT_TRUE(dynamicContactStageMeshRevisionMatches(captured, live_mesh));
  EXPECT_FALSE(dynamicContactStageProvenanceMatches(
      observation, live_mesh, declarations));
  mesh->local_mesh().set_boundary_label(
      contact_face, contact_wall_marker);
  EXPECT_TRUE(dynamicContactStageProvenanceMatches(
      observation, live_mesh, declarations));

  // This system is bound to reference coordinates. A mutation confined to
  // the unused current frame must not invalidate its stage provenance.
  mesh->local_mesh().mark_current_geometry_changed();
  EXPECT_TRUE(dynamicContactStageMeshRevisionMatches(captured, live_mesh));
  EXPECT_TRUE(dynamicContactStageProvenanceMatches(
      observation, live_mesh, declarations));

  // Mutating the coordinate frame selected by MeshAccess must invalidate the
  // captured stage immediately.
  mesh->local_mesh().mark_reference_geometry_changed();
  EXPECT_FALSE(dynamicContactStageMeshRevisionMatches(captured, live_mesh));
  EXPECT_FALSE(dynamicContactStageProvenanceMatches(
      observation, live_mesh, declarations));
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     GeneralizedAlphaMaintenanceCommonDeltaMapsEveryPostAcceptState)
{
  const std::vector<svmp::FE::Real> repair_delta{
      svmp::FE::Real{0.25},
      svmp::FE::Real{-0.5},
      svmp::FE::Real{0.0}};
  const auto original_delta = repair_delta;

  for (const double rho_inf : {0.0, 0.2, 0.5, 0.75, 1.0}) {
    const auto parameters =
        svmp::FE::timestepping::utils::
            generalizedAlphaFirstOrderFromRhoInf(rho_inf);
    const auto scheme =
        makeFirstOrderGeneralizedAlphaMaintenanceScheme(
            static_cast<svmp::FE::Real>(parameters.alpha_m),
            static_cast<svmp::FE::Real>(parameters.alpha_f),
            static_cast<svmp::FE::Real>(parameters.gamma),
            svmp::FE::Real{0.125});
    const auto plan =
        planFirstOrderGeneralizedAlphaMaintenancePublication(
            scheme,
            FirstOrderGeneralizedAlphaMaintenanceClosure::
                SameRepresentationDelta,
            repair_delta,
            repair_delta);

    EXPECT_EQ(
        plan.status,
        FirstOrderGeneralizedAlphaMaintenancePlanStatus::
            AlgebraicallyComplete);
    ASSERT_TRUE(plan.post_accept.has_value());
    ASSERT_TRUE(plan.implied_prior_state_delta.has_value());
    EXPECT_EQ(plan.requested_stage_state_delta, repair_delta);
    EXPECT_EQ(plan.requested_endpoint_state_delta, repair_delta);
    EXPECT_EQ(*plan.implied_prior_state_delta, repair_delta);
    EXPECT_EQ(plan.post_accept->u_delta, repair_delta);
    EXPECT_EQ(plan.post_accept->u_prev_delta, repair_delta);
    EXPECT_EQ(
        plan.post_accept->u_prev2_and_deeper_delta, repair_delta);
    EXPECT_EQ(
        plan.post_accept->prior_rate_delta,
        std::vector<svmp::FE::Real>(repair_delta.size(), 0.0));
    EXPECT_EQ(
        plan.post_accept->u_dot_delta,
        std::vector<svmp::FE::Real>(repair_delta.size(), 0.0));
    EXPECT_EQ(
        plan.post_accept->accepted_stage_rate_delta,
        std::vector<svmp::FE::Real>(repair_delta.size(), 0.0));
    EXPECT_TRUE(
        plan.post_accept->maintained_first_order_u_ddot_unchanged);
    EXPECT_FALSE(plan.requires_separate_geometric_motion_account);
    EXPECT_LE(
        plan.max_stage_state_identity_residual,
        plan.identity_tolerance);
    EXPECT_LE(
        plan.max_endpoint_update_identity_residual,
        plan.identity_tolerance);
    EXPECT_LE(
        plan.max_stage_rate_identity_residual,
        plan.identity_tolerance);
    ASSERT_TRUE(scheme.alpha_m.has_value());
    ASSERT_TRUE(scheme.gamma.has_value());
    EXPECT_NEAR(*scheme.alpha_m, parameters.alpha_m, 1.0e-15);
    EXPECT_NEAR(*scheme.gamma, parameters.gamma, 1.0e-15);
  }

  const FirstOrderGeneralizedAlphaMaintenanceScheme alpha_f_only_scheme{
      .alpha_f = svmp::FE::Real{0.75},
  };
  const auto alpha_f_only_plan =
      planFirstOrderGeneralizedAlphaMaintenancePublication(
          alpha_f_only_scheme,
          FirstOrderGeneralizedAlphaMaintenanceClosure::
              SameRepresentationDelta,
          repair_delta,
          repair_delta);
  EXPECT_EQ(
      alpha_f_only_plan.status,
      FirstOrderGeneralizedAlphaMaintenancePlanStatus::
          AlgebraicallyComplete);
  ASSERT_TRUE(alpha_f_only_plan.post_accept.has_value());
  EXPECT_EQ(alpha_f_only_plan.post_accept->u_delta, repair_delta);
  EXPECT_EQ(
      alpha_f_only_plan.post_accept->u_prev2_and_deeper_delta,
      repair_delta);
  EXPECT_EQ(
      alpha_f_only_plan.post_accept->prior_rate_delta,
      std::vector<svmp::FE::Real>(repair_delta.size(), 0.0));
  EXPECT_EQ(
      alpha_f_only_plan.post_accept->u_dot_delta,
      std::vector<svmp::FE::Real>(repair_delta.size(), 0.0));
  EXPECT_EQ(
      alpha_f_only_plan.post_accept->accepted_stage_rate_delta,
      std::vector<svmp::FE::Real>(repair_delta.size(), 0.0));

  EXPECT_EQ(repair_delta, original_delta);
}

TEST(ApplicationDriverLevelSetWorkflows,
     GeneralizedAlphaMaintenanceFixedPriorPolicyMapsEndpointRate)
{
  const std::vector<svmp::FE::Real> endpoint_delta{
      svmp::FE::Real{0.3},
      svmp::FE::Real{-0.6},
      svmp::FE::Real{0.0}};
  const auto original_endpoint_delta = endpoint_delta;

  for (const double rho_inf : {0.2, 1.0 / 3.0, 0.5, 1.0}) {
    SCOPED_TRACE(::testing::Message() << "rho_inf=" << rho_inf);
    const auto parameters =
        svmp::FE::timestepping::utils::
            generalizedAlphaFirstOrderFromRhoInf(rho_inf);
    const auto scheme =
        makeFirstOrderGeneralizedAlphaMaintenanceScheme(
            static_cast<svmp::FE::Real>(parameters.alpha_m),
            static_cast<svmp::FE::Real>(parameters.alpha_f),
            static_cast<svmp::FE::Real>(parameters.gamma),
            svmp::FE::Real{0.25});
    std::vector<svmp::FE::Real> stage_delta(endpoint_delta.size());
    std::transform(
        endpoint_delta.begin(),
        endpoint_delta.end(),
        stage_delta.begin(),
        [&](svmp::FE::Real delta) {
          return scheme.alpha_f * delta;
        });
    // Exercise scaled rather than bit-exact closure validation.
    stage_delta.back() = svmp::FE::Real{1.0e-14};
    const auto original_stage_delta = stage_delta;

    const auto plan =
        planFirstOrderGeneralizedAlphaMaintenancePublication(
            scheme,
            FirstOrderGeneralizedAlphaMaintenanceClosure::
                PreservePriorStateAndRate,
            stage_delta,
            endpoint_delta);

    EXPECT_EQ(
        plan.status,
        FirstOrderGeneralizedAlphaMaintenancePlanStatus::
            AlgebraicallyComplete);
    ASSERT_TRUE(plan.post_accept.has_value());
    ASSERT_TRUE(plan.implied_prior_state_delta.has_value());
    const std::vector<svmp::FE::Real> zero_delta(
        endpoint_delta.size(), 0.0);
    EXPECT_EQ(*plan.implied_prior_state_delta, zero_delta);
    EXPECT_EQ(plan.post_accept->u_delta, endpoint_delta);
    EXPECT_EQ(plan.post_accept->u_prev_delta, endpoint_delta);
    EXPECT_EQ(
        plan.post_accept->u_prev2_and_deeper_delta, zero_delta);
    EXPECT_EQ(plan.post_accept->prior_rate_delta, zero_delta);
    ASSERT_EQ(
        plan.post_accept->u_dot_delta.size(), endpoint_delta.size());
    ASSERT_EQ(
        plan.post_accept->accepted_stage_rate_delta.size(),
        endpoint_delta.size());
    for (std::size_t i = 0u; i < endpoint_delta.size(); ++i) {
      const auto expected_endpoint_rate =
          endpoint_delta[i] / (*scheme.gamma * *scheme.dt);
      EXPECT_NEAR(
          plan.post_accept->u_dot_delta[i],
          expected_endpoint_rate,
          1.0e-14);
      EXPECT_NEAR(
          plan.post_accept->accepted_stage_rate_delta[i],
          *scheme.alpha_m * expected_endpoint_rate,
          1.0e-14);
    }
    EXPECT_TRUE(plan.requires_separate_geometric_motion_account);
    EXPECT_TRUE(
        plan.post_accept->maintained_first_order_u_ddot_unchanged);
    EXPECT_GT(plan.max_stage_state_identity_residual, 0.0);
    EXPECT_LE(
        plan.max_stage_state_identity_residual,
        plan.identity_tolerance);
    EXPECT_LE(
        plan.max_endpoint_update_identity_residual,
        plan.identity_tolerance);
    EXPECT_LE(
        plan.max_stage_rate_identity_residual,
        plan.identity_tolerance);
    EXPECT_EQ(stage_delta, original_stage_delta);
    if (rho_inf == 0.2) {
      EXPECT_GT(*scheme.alpha_m, svmp::FE::Real{1.0});
    }
    if (rho_inf == 1.0) {
      EXPECT_DOUBLE_EQ(*scheme.alpha_m, *scheme.gamma);
    }
  }
  EXPECT_EQ(endpoint_delta, original_endpoint_delta);
}

TEST(ApplicationDriverLevelSetWorkflows,
     GeneralizedAlphaMaintenanceArbitraryRepairPairRemainsFailClosed)
{
  const auto parameters =
      svmp::FE::timestepping::utils::
          generalizedAlphaFirstOrderFromRhoInf(0.5);
  const auto scheme =
      makeFirstOrderGeneralizedAlphaMaintenanceScheme(
          static_cast<svmp::FE::Real>(parameters.alpha_m),
          static_cast<svmp::FE::Real>(parameters.alpha_f),
          static_cast<svmp::FE::Real>(parameters.gamma),
          svmp::FE::Real{0.25});
  const std::vector<svmp::FE::Real> stage_delta{0.4, -0.2};
  const std::vector<svmp::FE::Real> endpoint_delta{0.1, 0.3};

  const auto plan =
      planFirstOrderGeneralizedAlphaMaintenancePublication(
          scheme,
          FirstOrderGeneralizedAlphaMaintenanceClosure::
              PreservePriorStateAndRate,
          stage_delta,
          endpoint_delta);

  EXPECT_EQ(
      plan.status,
      FirstOrderGeneralizedAlphaMaintenancePlanStatus::ClosureMismatch);
  EXPECT_FALSE(plan.post_accept.has_value());
  EXPECT_TRUE(plan.requires_separate_geometric_motion_account);
  ASSERT_TRUE(plan.implied_prior_state_delta.has_value());
  ASSERT_EQ(plan.implied_prior_state_delta->size(), stage_delta.size());
  for (std::size_t i = 0u; i < stage_delta.size(); ++i) {
    const auto expected_prior_delta =
        (stage_delta[i] - scheme.alpha_f * endpoint_delta[i]) /
        (svmp::FE::Real{1.0} - scheme.alpha_f);
    EXPECT_NEAR(
        (*plan.implied_prior_state_delta)[i],
        expected_prior_delta,
        1.0e-14);
  }
  EXPECT_NE(
      plan.diagnostic.find("no selected prior-rate"),
      std::string::npos);

  const auto near_endpoint_alpha = std::nextafter(
      svmp::FE::Real{1.0}, svmp::FE::Real{0.0});
  const auto near_singular_scheme =
      FirstOrderGeneralizedAlphaMaintenanceScheme{
          .alpha_f = near_endpoint_alpha,
      };
  const auto near_singular_plan =
      planFirstOrderGeneralizedAlphaMaintenancePublication(
          near_singular_scheme,
          FirstOrderGeneralizedAlphaMaintenanceClosure::
              SameRepresentationDelta,
          stage_delta,
          endpoint_delta);
  EXPECT_EQ(
      near_singular_plan.status,
      FirstOrderGeneralizedAlphaMaintenancePlanStatus::
          NearSingularStageInversion);
  EXPECT_FALSE(near_singular_plan.post_accept.has_value());
  EXPECT_FALSE(near_singular_plan.implied_prior_state_delta.has_value());
  EXPECT_TRUE(
      near_singular_plan.requires_separate_geometric_motion_account);

  const auto endpoint_scheme =
      FirstOrderGeneralizedAlphaMaintenanceScheme{
          .alpha_f = svmp::FE::Real{1.0},
      };
  const auto endpoint_policy_plan =
      planFirstOrderGeneralizedAlphaMaintenancePublication(
          endpoint_scheme,
          FirstOrderGeneralizedAlphaMaintenanceClosure::
              PreservePriorStateAndRate,
          endpoint_delta,
          endpoint_delta);
  EXPECT_EQ(
      endpoint_policy_plan.status,
      FirstOrderGeneralizedAlphaMaintenancePlanStatus::
          EndpointPolicyMismatch);
  EXPECT_FALSE(endpoint_policy_plan.post_accept.has_value());
  const auto endpoint_delta_mismatch_plan =
      planFirstOrderGeneralizedAlphaMaintenancePublication(
          endpoint_scheme,
          FirstOrderGeneralizedAlphaMaintenanceClosure::
              SameRepresentationDelta,
          stage_delta,
          endpoint_delta);
  EXPECT_EQ(
      endpoint_delta_mismatch_plan.status,
      FirstOrderGeneralizedAlphaMaintenancePlanStatus::
          EndpointPolicyMismatch);
  EXPECT_FALSE(endpoint_delta_mismatch_plan.post_accept.has_value());

  const FirstOrderGeneralizedAlphaMaintenanceScheme alpha_f_only_scheme{
      .alpha_f = svmp::FE::Real{0.75},
  };
  const std::vector<svmp::FE::Real> compatible_endpoint_delta{0.4};
  const std::vector<svmp::FE::Real> compatible_stage_delta{0.3};
  const auto missing_rate_metadata_plan =
      planFirstOrderGeneralizedAlphaMaintenancePublication(
          alpha_f_only_scheme,
          FirstOrderGeneralizedAlphaMaintenanceClosure::
              PreservePriorStateAndRate,
          compatible_stage_delta,
          compatible_endpoint_delta);
  EXPECT_EQ(
      missing_rate_metadata_plan.status,
      FirstOrderGeneralizedAlphaMaintenancePlanStatus::
          MissingRateParameters);
  EXPECT_FALSE(missing_rate_metadata_plan.post_accept.has_value());
  EXPECT_NE(
      missing_rate_metadata_plan.diagnostic.find("authentic alpha_m"),
      std::string::npos);
}

TEST(ApplicationDriverLevelSetWorkflows,
     GeneralizedAlphaMaintenancePlannerRejectsInvalidInputs)
{
  EXPECT_THROW(
      (void)makeFirstOrderGeneralizedAlphaMaintenanceScheme(
          0.75,
          std::numeric_limits<svmp::FE::Real>::quiet_NaN(),
          0.5,
          0.1),
      std::invalid_argument);
  EXPECT_THROW(
      (void)makeFirstOrderGeneralizedAlphaMaintenanceScheme(
          0.75, -0.1, 1.35, 0.1),
      std::invalid_argument);
  EXPECT_THROW(
      (void)makeFirstOrderGeneralizedAlphaMaintenanceScheme(
          0.75, 1.01, 0.24, 0.1),
      std::invalid_argument);
  EXPECT_THROW(
      (void)makeFirstOrderGeneralizedAlphaMaintenanceScheme(
          0.75, 0.75, 0.5, 0.0),
      std::invalid_argument);

  auto scheme =
      makeFirstOrderGeneralizedAlphaMaintenanceScheme(
          0.75, 0.75, 0.5, 0.1);
  const std::vector<svmp::FE::Real> one_delta{0.1};
  const std::vector<svmp::FE::Real> two_deltas{0.1, 0.2};
  const std::vector<svmp::FE::Real> empty;
  EXPECT_THROW(
      (void)planFirstOrderGeneralizedAlphaMaintenancePublication(
          scheme,
          FirstOrderGeneralizedAlphaMaintenanceClosure::
              SameRepresentationDelta,
          one_delta,
          two_deltas),
      std::invalid_argument);
  EXPECT_THROW(
      (void)planFirstOrderGeneralizedAlphaMaintenancePublication(
          scheme,
          FirstOrderGeneralizedAlphaMaintenanceClosure::
              SameRepresentationDelta,
          empty,
          empty),
      std::invalid_argument);
  const std::vector<svmp::FE::Real> nonfinite_delta{
      std::numeric_limits<svmp::FE::Real>::infinity()};
  EXPECT_THROW(
      (void)planFirstOrderGeneralizedAlphaMaintenancePublication(
          scheme,
          FirstOrderGeneralizedAlphaMaintenanceClosure::
              SameRepresentationDelta,
          nonfinite_delta,
          nonfinite_delta),
      std::invalid_argument);
  const FirstOrderGeneralizedAlphaMaintenanceScheme partial_scheme{
      .alpha_m = svmp::FE::Real{0.75},
      .alpha_f = svmp::FE::Real{0.75},
  };
  EXPECT_THROW(
      (void)planFirstOrderGeneralizedAlphaMaintenancePublication(
          partial_scheme,
          FirstOrderGeneralizedAlphaMaintenanceClosure::
              SameRepresentationDelta,
          one_delta,
          one_delta),
      std::invalid_argument);
  *scheme.gamma += 0.1;
  EXPECT_THROW(
      (void)planFirstOrderGeneralizedAlphaMaintenancePublication(
          scheme,
          FirstOrderGeneralizedAlphaMaintenanceClosure::
              SameRepresentationDelta,
          one_delta,
          one_delta),
      std::invalid_argument);
}

TEST(ApplicationDriverLevelSetWorkflows,
     ActiveCutTopologyFingerprintUsesOnlySemanticGlobalRuleIdentity)
{
  ActiveCutVolumeRequest request;
  request.level_set_field_name = "phi";
  request.domain_id = "semantic-topology-a";
  request.requested_interface_marker = 713;
  request.active_side = LevelSetActiveSide::Negative;

  svmp::FE::interfaces::FreeSurfaceGeometryRevision revision;
  revision.source_id = "field:phi";
  revision.domain_id = request.domain_id;
  revision.interface_marker = request.requested_interface_marker;
  revision.isovalue = request.isovalue;
  revision.source_layout_revision = 3u;
  revision.source_value_revision = 5u;
  revision.mesh_geometry_revision = 7u;
  revision.mesh_topology_revision = 11u;
  revision.ownership_revision = 13u;
  revision.numbering_revision = 17u;
  revision.snapshot_revision_key = 19u;
  const auto request_identity =
      activeCutTopologyRequestIdentity(request, revision);

  auto changed_epochs = revision;
  changed_epochs.source_layout_revision += 101u;
  changed_epochs.source_value_revision += 103u;
  changed_epochs.mesh_geometry_revision += 107u;
  changed_epochs.mesh_topology_revision += 109u;
  changed_epochs.ownership_revision += 113u;
  changed_epochs.numbering_revision += 127u;
  changed_epochs.snapshot_revision_key += 131u;
  EXPECT_EQ(
      activeCutTopologyRequestIdentity(request, changed_epochs),
      request_identity);

  auto other_request = request;
  other_request.domain_id = "semantic-topology-b";
  auto other_revision = revision;
  other_revision.domain_id = other_request.domain_id;
  const auto other_request_identity =
      activeCutTopologyRequestIdentity(other_request, other_revision);
  EXPECT_NE(other_request_identity, request_identity);
  auto positive_request = request;
  positive_request.active_side = LevelSetActiveSide::Positive;
  EXPECT_NE(
      activeCutTopologyRequestIdentity(positive_request, revision),
      request_identity);

  svmp::FE::interfaces::FreeSurfaceGeometryRuleRecord rule;
  rule.role = svmp::FE::interfaces::FreeSurfaceGeometryRuleRole::Interface;
  rule.retention =
      svmp::FE::interfaces::FreeSurfaceGeometryRetention::Retained;
  rule.physical_boundary_marker = -1;
  rule.locally_owned = true;
  rule.topology_id = "cell-101-segment-0";
  rule.source_topology_key = 29u;
  rule.component_id = 31;
  rule.source_fragment_stable_ids = {37u, 41u};
  rule.reference_rule.kind =
      svmp::FE::geometry::CutQuadratureKind::Interface;
  rule.reference_rule.side =
      svmp::FE::geometry::CutIntegrationSide::Interface;
  rule.reference_rule.geometric_dimension = 1;
  rule.reference_rule.full_cell_equivalent = false;
  rule.reference_rule.measure = 2.0;
  rule.reference_rule.provenance.parent_entity = 2;
  rule.reference_rule.provenance.parent_boundary_entity = -1;
  rule.reference_rule.provenance.parent_entity_global_id = 101;
  rule.reference_rule.provenance.parent_boundary_entity_global_id =
      svmp::FE::INVALID_GLOBAL_INDEX;
  rule.reference_rule.provenance.owner_rank = 0;
  rule.reference_rule.provenance.marker = 713;
  rule.reference_rule.provenance.cut_topology_id = rule.topology_id;
  rule.reference_rule.provenance.cut_topology_revision = 43u;
  rule.reference_rule.provenance.source_value_revision = 47u;
  rule.reference_rule.provenance.source_stable_id = 53u;
  rule.reference_rule.provenance.free_surface_snapshot_revision_key = 59u;
  rule.physical_rule.cut_topology_revision = 61u;
  rule.physical_rule.source_value_revision = 67u;
  rule.physical_rule.free_surface_snapshot_revision_key = 71u;
  rule.physical_rule.physical_measure = 3.0;

  const auto rule_fingerprint =
      freeSurfaceRuleSemanticTopologyFingerprint(
          rule, request_identity);
  auto changed_content = rule;
  changed_content.component_id += 1;
  std::reverse(
      changed_content.source_fragment_stable_ids.begin(),
      changed_content.source_fragment_stable_ids.end());
  changed_content.reference_rule.provenance.parent_entity = 99;
  changed_content.reference_rule.provenance.owner_rank = 7;
  changed_content.reference_rule.provenance.cut_topology_revision += 1u;
  changed_content.reference_rule.provenance.source_value_revision += 1u;
  changed_content.reference_rule.provenance.source_stable_id += 1u;
  changed_content.reference_rule.provenance
      .free_surface_snapshot_revision_key += 1u;
  changed_content.reference_rule.measure += 1.0;
  changed_content.physical_rule.cut_topology_revision += 1u;
  changed_content.physical_rule.source_value_revision += 1u;
  changed_content.physical_rule.free_surface_snapshot_revision_key += 1u;
  changed_content.physical_rule.physical_measure += 1.0;
  EXPECT_EQ(
      freeSurfaceRuleSemanticTopologyFingerprint(
          changed_content, request_identity),
      rule_fingerprint);

  auto changed_parent = rule;
  changed_parent.reference_rule.provenance.parent_entity_global_id = 102;
  EXPECT_NE(
      freeSurfaceRuleSemanticTopologyFingerprint(
          changed_parent, request_identity),
      rule_fingerprint);
  auto changed_boundary_parent = rule;
  changed_boundary_parent.reference_rule.provenance
      .parent_boundary_entity_global_id = 211;
  EXPECT_NE(
      freeSurfaceRuleSemanticTopologyFingerprint(
          changed_boundary_parent, request_identity),
      rule_fingerprint);
  auto changed_retention = rule;
  changed_retention.retention =
      svmp::FE::interfaces::FreeSurfaceGeometryRetention::PrunedSmallVolume;
  EXPECT_NE(
      freeSurfaceRuleSemanticTopologyFingerprint(
          changed_retention, request_identity),
      rule_fingerprint);
  auto changed_topology = rule;
  changed_topology.topology_id = "cell-101-segment-1";
  changed_topology.reference_rule.provenance.cut_topology_id =
      changed_topology.topology_id;
  const auto changed_topology_fingerprint =
      freeSurfaceRuleSemanticTopologyFingerprint(
          changed_topology, request_identity);
  EXPECT_EQ(changed_topology_fingerprint, rule_fingerprint);

  auto changed_source_topology = rule;
  changed_source_topology.source_topology_key += 1u;
  EXPECT_NE(
      freeSurfaceRuleSemanticTopologyFingerprint(
          changed_source_topology, request_identity),
      rule_fingerprint);

  const std::array<std::uint64_t, 2> request_order_a{
      request_identity, other_request_identity};
  const std::array<std::uint64_t, 2> request_order_b{
      other_request_identity, request_identity};
  const auto other_rule_fingerprint =
      freeSurfaceRuleSemanticTopologyFingerprint(
          rule, other_request_identity);
  const std::array<std::uint64_t, 3> rule_order_a{
      rule_fingerprint,
      changed_topology_fingerprint,
      other_rule_fingerprint};
  const std::array<std::uint64_t, 3> rule_order_b{
      other_rule_fingerprint,
      rule_fingerprint,
      changed_topology_fingerprint};
  const auto comm = svmp::MeshComm::self();
  const auto fingerprint_a =
      collectivePartitionIndependentCutTopologyFingerprint(
          request_order_a, rule_order_a, comm);
  EXPECT_NE(fingerprint_a, 0u);
  EXPECT_EQ(
      collectivePartitionIndependentCutTopologyFingerprint(
          request_order_b, rule_order_b, comm),
      fingerprint_a);
  const std::array<std::uint64_t, 4> duplicated_rule{
      rule_fingerprint,
      changed_topology_fingerprint,
      other_rule_fingerprint,
      rule_fingerprint};
  EXPECT_NE(
      collectivePartitionIndependentCutTopologyFingerprint(
          request_order_a, duplicated_rule, comm),
      fingerprint_a);
}

TEST(ApplicationDriverLevelSetWorkflows,
     TransientCutTopologyFingerprintIsQualifiedOnlyForLinearCornerGeometry)
{
  using Mode =
      svmp::FE::level_set::GeneratedInterfaceGeometryMode;
  ActiveCutVolumeRequest linear_request;
  linear_request.geometry_mode = Mode::LinearCorner;
  ActiveCutVolumeRequest high_order_request = linear_request;
  high_order_request.geometry_mode = Mode::HighOrderImplicit;

  EXPECT_TRUE(transientCutTopologyFingerprintSupportsRequests(
      std::span<const ActiveCutVolumeRequest>{}));
  EXPECT_TRUE(transientCutTopologyFingerprintSupportsRequests(
      std::span<const ActiveCutVolumeRequest>{&linear_request, 1u}));
  const std::array<ActiveCutVolumeRequest, 2> mixed_requests{
      linear_request, high_order_request};
  EXPECT_FALSE(transientCutTopologyFingerprintSupportsRequests(
      mixed_requests));
}

TEST(ApplicationDriverLevelSetWorkflows,
     ActiveCutTopologyFingerprintFailsClosedOnIncompleteLocalPreparation)
{
  const std::array<ActiveCutTopologySnapshotBinding, 1> bindings{{}};
  EXPECT_THROW(
      activeCutContextTopologyFingerprint(
          bindings, svmp::MeshComm::self()),
      std::runtime_error);
}

TEST(ApplicationDriverLevelSetWorkflows,
     ProductionCutSourceTopologyDetectsReusedIdTransitions)
{
  svmp::FE::interfaces::CutInterfaceDomainRequest request;
  request.source =
      svmp::FE::interfaces::LevelSetInterfaceSource::fromEvaluator(
          "topology-source-test", 1u, 1u);
  request.interface_marker = 713;
  request.isovalue = 0.0;
  request.tolerance = 1.0e-12;
  request.quadrature_order = 1;

  const auto make_record =
      [&](svmp::FE::interfaces::CutInterfaceFragment fragment,
          svmp::FE::ElementType type) {
        fragment.parent_cell_global_id = 101;
        fragment.owner_rank = 0;
        svmp::FE::interfaces::FreeSurfaceGeometryRuleRecord record;
        record.role =
            svmp::FE::interfaces::FreeSurfaceGeometryRuleRole::Interface;
        record.retention =
            svmp::FE::interfaces::FreeSurfaceGeometryRetention::Retained;
        record.locally_owned = true;
        record.topology_id = fragment.topology_id;
        record.source_topology_key =
            svmp::FE::interfaces::freeSurfaceGeometrySourceTopologyKey(
                fragment, type, request.tolerance);
        record.reference_rule = fragment.toCutQuadratureRule(request);
        return record;
      };

  svmp::FE::interfaces::LevelSetCellCutInput quad_input;
  quad_input.parent_cell = 2;
  quad_input.element_type = svmp::FE::ElementType::Quad4;
  quad_input.node_coordinates = {
      {{-1.0, -1.0, 0.0}},
      {{1.0, -1.0, 0.0}},
      {{1.0, 1.0, 0.0}},
      {{-1.0, 1.0, 0.0}}};
  quad_input.level_set_values = {-1.0, 1.0, 1.0, -1.0};
  const auto vertical =
      svmp::FE::interfaces::cutLinearLevelSetCell2D(request, quad_input);
  // Move the interface while preserving the same canonical corner signs and
  // parent-edge incidences.  An epoch-free topology descriptor must not turn
  // this ordinary value/coordinate change into a topology event.
  quad_input.level_set_values = {-2.0, 0.5, 0.5, -2.0};
  const auto moved_vertical =
      svmp::FE::interfaces::cutLinearLevelSetCell2D(request, quad_input);
  quad_input.level_set_values = {-1.0, -1.0, 1.0, 1.0};
  const auto horizontal =
      svmp::FE::interfaces::cutLinearLevelSetCell2D(request, quad_input);
  quad_input.level_set_values = {0.0, 1.0, 1.0, -1.0};
  const auto vertex_touch =
      svmp::FE::interfaces::cutLinearLevelSetCell2D(request, quad_input);
  ASSERT_EQ(vertical.fragments.size(), 1u);
  ASSERT_EQ(moved_vertical.fragments.size(), 1u);
  ASSERT_EQ(horizontal.fragments.size(), 1u);
  ASSERT_EQ(vertex_touch.fragments.size(), 1u);
  EXPECT_EQ(vertical.fragments.front().topology_id,
            moved_vertical.fragments.front().topology_id);
  EXPECT_EQ(vertical.fragments.front().topology_id,
            horizontal.fragments.front().topology_id);
  EXPECT_EQ(vertical.fragments.front().topology_id,
            vertex_touch.fragments.front().topology_id);
  EXPECT_EQ(vertical.fragments.front().degeneracy,
            svmp::FE::interfaces::CutInterfaceDegeneracy::None);
  EXPECT_EQ(horizontal.fragments.front().degeneracy,
            svmp::FE::interfaces::CutInterfaceDegeneracy::None);
  EXPECT_EQ(vertex_touch.fragments.front().degeneracy,
            svmp::FE::interfaces::CutInterfaceDegeneracy::VertexTouch);

  const auto vertical_record = make_record(
      vertical.fragments.front(), svmp::FE::ElementType::Quad4);
  const auto moved_vertical_record = make_record(
      moved_vertical.fragments.front(), svmp::FE::ElementType::Quad4);
  const auto horizontal_record = make_record(
      horizontal.fragments.front(), svmp::FE::ElementType::Quad4);
  const auto vertex_touch_record = make_record(
      vertex_touch.fragments.front(), svmp::FE::ElementType::Quad4);
  constexpr std::uint64_t request_identity = 0x12345678u;
  const auto vertical_fingerprint =
      freeSurfaceRuleSemanticTopologyFingerprint(
          vertical_record, request_identity);
  EXPECT_EQ(moved_vertical_record.source_topology_key,
            vertical_record.source_topology_key);
  EXPECT_EQ(
      freeSurfaceRuleSemanticTopologyFingerprint(
          moved_vertical_record, request_identity),
      vertical_fingerprint);
  EXPECT_NE(
      freeSurfaceRuleSemanticTopologyFingerprint(
          horizontal_record, request_identity),
      vertical_fingerprint);
  EXPECT_NE(
      freeSurfaceRuleSemanticTopologyFingerprint(
          vertex_touch_record, request_identity),
      vertical_fingerprint);

  svmp::FE::interfaces::LevelSetCellCutInput tetra_input;
  tetra_input.parent_cell = 2;
  tetra_input.element_type = svmp::FE::ElementType::Tetra4;
  tetra_input.node_coordinates = {
      {{0.0, 0.0, 0.0}},
      {{1.0, 0.0, 0.0}},
      {{0.0, 1.0, 0.0}},
      {{0.0, 0.0, 1.0}}};
  tetra_input.level_set_values = {-1.0, 1.0, 1.0, 1.0};
  const auto triangle =
      svmp::FE::interfaces::cutLinearLevelSetCell3D(request, tetra_input);
  tetra_input.level_set_values = {-1.0, -1.0, 1.0, 1.0};
  const auto quadrilateral =
      svmp::FE::interfaces::cutLinearLevelSetCell3D(request, tetra_input);
  ASSERT_EQ(triangle.fragments.size(), 1u);
  ASSERT_EQ(quadrilateral.fragments.size(), 1u);
  EXPECT_EQ(triangle.fragments.front().topology_id,
            quadrilateral.fragments.front().topology_id);
  EXPECT_EQ(triangle.fragments.front().vertices.size(), 3u);
  EXPECT_EQ(quadrilateral.fragments.front().vertices.size(), 4u);
  const auto triangle_record = make_record(
      triangle.fragments.front(), svmp::FE::ElementType::Tetra4);
  const auto quadrilateral_record = make_record(
      quadrilateral.fragments.front(), svmp::FE::ElementType::Tetra4);
  EXPECT_NE(
      freeSurfaceRuleSemanticTopologyFingerprint(
          triangle_record, request_identity),
      freeSurfaceRuleSemanticTopologyFingerprint(
          quadrilateral_record, request_identity));
}

TEST(ApplicationDriverLevelSetWorkflows,
     TransientCutTopologyAttemptIsMonotoneAndResetsOnlyPerAttempt)
{
  constexpr std::uint64_t accepted_key = 0x1234u;
  constexpr std::uint64_t trial_key = 0x5678u;
  const auto report = [](std::uint64_t topology_key) {
    ActiveCutContextRefreshReport result;
    result.topology_key = topology_key;
    return result;
  };

  TransientCutTopologyAttemptTracker tracker;
  tracker.observe(report(accepted_key), "before_physics_solve");
  EXPECT_FALSE(tracker.attemptActive());
  EXPECT_FALSE(tracker.acceptedTopologyKey().has_value());

  tracker.beginAttempt();
  tracker.observe(report(accepted_key), "before_physics_solve");
  ASSERT_NO_THROW(tracker.requireAcceptedBaseline());
  ASSERT_TRUE(tracker.acceptedTopologyKey().has_value());
  EXPECT_EQ(*tracker.acceptedTopologyKey(), accepted_key);
  EXPECT_FALSE(tracker.attemptTainted());

  tracker.observe(report(trial_key), "line_search_trial_residual");
  tracker.observe(
      report(accepted_key), "final_candidate_topology_gate");
  EXPECT_TRUE(tracker.attemptTainted());
  ASSERT_TRUE(tracker.firstMismatchedTopologyKey().has_value());
  EXPECT_EQ(*tracker.firstMismatchedTopologyKey(), trial_key);
  EXPECT_EQ(
      tracker.firstMismatchProvenance(),
      "line_search_trial_residual");
  EXPECT_TRUE(tracker.candidateMustReject(accepted_key));

  tracker.discardAttempt();
  EXPECT_FALSE(tracker.attemptActive());
  tracker.observe(report(trial_key), "accepted_step");
  EXPECT_EQ(*tracker.acceptedTopologyKey(), accepted_key);

  tracker.beginAttempt();
  EXPECT_FALSE(tracker.attemptTainted());
  EXPECT_FALSE(tracker.firstMismatchedTopologyKey().has_value());
  tracker.observe(report(accepted_key), "before_physics_solve");
  tracker.observe(
      report(accepted_key), "final_candidate_topology_gate");
  EXPECT_FALSE(tracker.candidateMustReject(accepted_key));
  ASSERT_NO_THROW(tracker.completeAttempt(accepted_key));
  EXPECT_FALSE(tracker.attemptActive());
  ASSERT_TRUE(tracker.acceptedTopologyKey().has_value());
  EXPECT_EQ(*tracker.acceptedTopologyKey(), accepted_key);

  TransientCutTopologyAttemptTracker fixed_storage_tracker;
  fixed_storage_tracker.beginAttempt();
  fixed_storage_tracker.observe(
      report(accepted_key), "before_physics_solve");
  std::string caller_owned_provenance(512u, 'p');
  fixed_storage_tracker.observe(
      report(trial_key), caller_owned_provenance);
  const std::string retained_provenance{
      fixed_storage_tracker.firstMismatchProvenance()};
  caller_owned_provenance.assign(512u, 'q');
  EXPECT_EQ(
      fixed_storage_tracker.firstMismatchProvenance(),
      std::string_view{retained_provenance});
  EXPECT_LE(
      fixed_storage_tracker.firstMismatchProvenance().size(), 128u);
}

TEST(ApplicationDriverLevelSetWorkflows,
     TransientCutTopologyBaselineSeedsOnlyAtBeforeSolveAndFailsClosed)
{
  const auto report = [](std::uint64_t topology_key) {
    ActiveCutContextRefreshReport result;
    result.topology_key = topology_key;
    return result;
  };

  TransientCutTopologyAttemptTracker wrong_provenance;
  wrong_provenance.beginAttempt();
  wrong_provenance.observe(report(0x91u), "initial");
  EXPECT_FALSE(wrong_provenance.acceptedTopologyKey().has_value());
  EXPECT_TRUE(wrong_provenance.attemptTainted());
  EXPECT_THROW(
      wrong_provenance.requireAcceptedBaseline(), std::runtime_error);
  wrong_provenance.observe(
      report(0x91u), "final_candidate_topology_gate");
  EXPECT_TRUE(wrong_provenance.candidateMustReject(0x91u));

  TransientCutTopologyAttemptTracker missing_key;
  missing_key.beginAttempt();
  missing_key.observe(report(0u), "before_physics_solve");
  EXPECT_FALSE(missing_key.acceptedTopologyKey().has_value());
  EXPECT_TRUE(missing_key.attemptTainted());
  EXPECT_THROW(missing_key.requireAcceptedBaseline(), std::runtime_error);
}

TEST(ApplicationDriverLevelSetWorkflows,
     PostacceptMaintenanceTopologyEvidenceMakesEqualNonzeroKeysCommitEligible)
{
  constexpr std::uint64_t accepted_topology_key = 0x51a7u;

  const auto evidence = postacceptMaintenanceTopologyEvidence(
      std::optional<std::uint64_t>{accepted_topology_key},
      std::optional<std::uint64_t>{accepted_topology_key});

  EXPECT_TRUE(evidence.baseline_present);
  EXPECT_EQ(evidence.baseline_key, accepted_topology_key);
  EXPECT_TRUE(evidence.final_present);
  EXPECT_EQ(evidence.final_key, accepted_topology_key);
  EXPECT_TRUE(evidence.keys_equal);
  EXPECT_EQ(
      classifyPostacceptMaintenanceTopologyEvidence(evidence),
      PostacceptMaintenanceTopologyDecision::Commit);
}

TEST(ApplicationDriverLevelSetWorkflows,
     PostacceptMaintenanceTopologyEvidenceRejectsOnlyACompleteMismatch)
{
  constexpr std::uint64_t baseline_key = 0x51a7u;
  constexpr std::uint64_t changed_key = 0x62b8u;
  const auto mismatch = postacceptMaintenanceTopologyEvidence(
      std::optional<std::uint64_t>{baseline_key},
      std::optional<std::uint64_t>{changed_key});
  EXPECT_TRUE(mismatch.baseline_present);
  EXPECT_TRUE(mismatch.final_present);
  EXPECT_FALSE(mismatch.keys_equal);
  EXPECT_EQ(
      classifyPostacceptMaintenanceTopologyEvidence(mismatch),
      PostacceptMaintenanceTopologyDecision::RejectMaintenance);

  const std::array<PostacceptMaintenanceTopologyEvidence, 7>
      invalid_evidence{{
          postacceptMaintenanceTopologyEvidence(
              std::nullopt,
              std::optional<std::uint64_t>{baseline_key}),
          postacceptMaintenanceTopologyEvidence(
              std::optional<std::uint64_t>{baseline_key},
              std::nullopt),
          postacceptMaintenanceTopologyEvidence(
              std::optional<std::uint64_t>{0u},
              std::optional<std::uint64_t>{baseline_key}),
          postacceptMaintenanceTopologyEvidence(
              std::optional<std::uint64_t>{baseline_key},
              std::optional<std::uint64_t>{0u}),
          PostacceptMaintenanceTopologyEvidence{
              .baseline_present = true,
              .baseline_key = baseline_key,
              .final_present = true,
              .final_key = baseline_key,
              .keys_equal = false,
          },
          PostacceptMaintenanceTopologyEvidence{
              .baseline_present = true,
              .baseline_key = baseline_key,
              .final_present = true,
              .final_key = changed_key,
              .keys_equal = true,
          },
          PostacceptMaintenanceTopologyEvidence{
              .baseline_present = false,
              .baseline_key = baseline_key,
              .final_present = true,
              .final_key = baseline_key,
              .keys_equal = true,
          },
      }};
  for (const auto& evidence : invalid_evidence) {
    EXPECT_EQ(
        classifyPostacceptMaintenanceTopologyEvidence(evidence),
        PostacceptMaintenanceTopologyDecision::InvariantFailure);
  }
}

TEST(ApplicationDriverLevelSetWorkflows,
     PostacceptMaintenanceRejectionEvidenceWaitsForCompleteRecovery)
{
  for (unsigned int readiness_mask = 0u;
       readiness_mask < 16u;
       ++readiness_mask) {
    const bool geometry_restored = (readiness_mask & 1u) != 0u;
    const bool checkpoint_restored = (readiness_mask & 2u) != 0u;
    const bool restored_topology_verified =
        (readiness_mask & 4u) != 0u;
    const bool ledger_rejection_preflight_ready =
        (readiness_mask & 8u) != 0u;
    EXPECT_EQ(
        postacceptMaintenanceRejectionEvidenceMayPublish(
            geometry_restored,
            checkpoint_restored,
            restored_topology_verified,
            ledger_rejection_preflight_ready),
        readiness_mask == 15u)
        << "readiness_mask=" << readiness_mask;
  }
}

TEST(ApplicationDriverLevelSetWorkflows,
     ContactStageEndpointRestorePreservesTheOriginalStageFailure)
{
  const auto stage_failure = std::make_exception_ptr(
      std::runtime_error("original contact-stage failure"));
  bool endpoint_restore_attempted = false;

  try {
    restoreAcceptedContactStageEndpointAndRequireCollectiveSuccess(
        stage_failure,
        [&] {
          endpoint_restore_attempted = true;
          throw std::runtime_error("secondary endpoint-restore failure");
        },
        svmp::MeshComm::self());
    FAIL() << "Expected the saved contact-stage failure to be rethrown";
  } catch (const std::runtime_error& error) {
    EXPECT_EQ(std::string(error.what()),
              "original contact-stage failure");
  }
  EXPECT_TRUE(endpoint_restore_attempted);
}

TEST(ApplicationDriverLevelSetWorkflows,
     ActiveCutRefreshInvalidationPreservesItsObserver)
{
  ActiveCutContextRefreshCache cache;
  cache.last_signature = ActiveCutContextRefreshSignature{};
  cache.last_vector_signature = ActiveCutContextRefreshSignature{};
  cache.evaluated_state_source_revisions.emplace(3, 7u);
  cache.topology_key = 11u;
  std::size_t observations = 0u;
  cache.observer =
      [&](const ActiveCutContextRefreshReport&, std::string_view provenance) {
        EXPECT_EQ(provenance, "after_invalidation");
        ++observations;
      };

  cache.invalidateGeneratedState();
  EXPECT_FALSE(cache.last_signature.has_value());
  EXPECT_FALSE(cache.last_vector_signature.has_value());
  EXPECT_TRUE(cache.evaluated_state_source_revisions.empty());
  EXPECT_FALSE(cache.topology_key.has_value());
  ASSERT_TRUE(static_cast<bool>(cache.observer));

  observeActiveCutContextRefresh(
      cache, ActiveCutContextRefreshReport{}, "after_invalidation");
  EXPECT_EQ(observations, 1u);
}

TEST(ApplicationDriverLevelSetWorkflows,
     StaticCapillaryActiveSupportUnionDeduplicatesAndRejectsInvalidIndices)
{
  const auto gathered = communicatorWideIndexUnion(
      {3u, 1u, 3u, 2u},
      /*upper_bound=*/4u,
      svmp::MeshComm::self(),
      "static-capillary serial test support");
  EXPECT_EQ(
      gathered,
      (std::vector<std::size_t>{1u, 2u, 3u}));
  EXPECT_THROW(
      (void)communicatorWideIndexUnion(
          {4u},
          /*upper_bound=*/4u,
          svmp::MeshComm::self(),
          "static-capillary serial invalid support"),
      std::runtime_error);
}

TEST(ApplicationDriverLevelSetWorkflows,
     ParsesAndCanonicalizesStaticCapillaryInitializationControls)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Level_set_field_name>phi</Level_set_field_name>
    <Enable_static_capillary_equilibrium_initialization>true</Enable_static_capillary_equilibrium_initialization>
    <Static_capillary_volume_tolerance>2.0e-9</Static_capillary_volume_tolerance>
    <Static_capillary_projected_gradient_tolerance>3.0e-8</Static_capillary_projected_gradient_tolerance>
    <Static_capillary_pressure_representability_max_residual_norm>3.5e-10</Static_capillary_pressure_representability_max_residual_norm>
    <Static_capillary_pressure_representability_max_relative_distance>3.75e-9</Static_capillary_pressure_representability_max_relative_distance>
    <Static_capillary_physical_equilibrium_max_residual_norm>3.875e-10</Static_capillary_physical_equilibrium_max_residual_norm>
    <Static_capillary_constant_pressure_kkt_max_residual_norm>4.0e-10</Static_capillary_constant_pressure_kkt_max_residual_norm>
    <Static_capillary_constant_pressure_kkt_max_relative_distance>5.0e-9</Static_capillary_constant_pressure_kkt_max_relative_distance>
    <Static_capillary_finite_difference_reference_coefficient_scale>0.25</Static_capillary_finite_difference_reference_coefficient_scale>
    <Static_capillary_finite_difference_relative_step>6.0e-6</Static_capillary_finite_difference_relative_step>
    <Static_capillary_minimum_finite_difference_step>7.0e-12</Static_capillary_minimum_finite_difference_step>
    <Static_capillary_finite_difference_max_shrinks>8</Static_capillary_finite_difference_max_shrinks>
    <Static_capillary_max_iterations>9</Static_capillary_max_iterations>
    <Static_capillary_max_line_search_iterations>10</Static_capillary_max_line_search_iterations>
    <Static_capillary_projected_gradient_inverse_stiffness>0.75</Static_capillary_projected_gradient_inverse_stiffness>
    <Static_capillary_tangent_trust_radius>0.125</Static_capillary_tangent_trust_radius>
    <Static_capillary_maximum_coefficient_update_linf>0.375</Static_capillary_maximum_coefficient_update_linf>
    <Static_capillary_line_search_shrink>0.4</Static_capillary_line_search_shrink>
    <Static_capillary_armijo_fraction>2.0e-4</Static_capillary_armijo_fraction>
    <Static_capillary_minimum_volume_merit_penalty>2.5</Static_capillary_minimum_volume_merit_penalty>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  const auto requests = levelSetMaintenanceRequests(*params);
  ASSERT_EQ(requests.size(), 1u);
  const auto& request = requests.front();
  EXPECT_TRUE(request.static_capillary_equilibrium_enabled);
  EXPECT_FALSE(request.static_capillary_equilibrium_initialized);
  const auto& options = request.static_capillary_equilibrium;
  EXPECT_DOUBLE_EQ(options.volume_tolerance, 2.0e-9);
  EXPECT_DOUBLE_EQ(options.projected_gradient_tolerance, 3.0e-8);
  EXPECT_DOUBLE_EQ(
      options.pressure_representability_max_residual_norm, 3.5e-10);
  EXPECT_DOUBLE_EQ(
      options.pressure_representability_max_relative_distance, 3.75e-9);
  EXPECT_DOUBLE_EQ(
      options.physical_equilibrium_max_residual_norm, 3.875e-10);
  EXPECT_DOUBLE_EQ(
      options.constant_pressure_kkt_max_residual_norm, 4.0e-10);
  EXPECT_DOUBLE_EQ(
      options.constant_pressure_kkt_max_relative_distance, 5.0e-9);
  EXPECT_DOUBLE_EQ(
      options.finite_difference_reference_coefficient_scale, 0.25);
  EXPECT_DOUBLE_EQ(options.finite_difference_relative_step, 6.0e-6);
  EXPECT_DOUBLE_EQ(options.minimum_finite_difference_step, 7.0e-12);
  EXPECT_EQ(options.finite_difference_max_shrinks, 8);
  EXPECT_EQ(options.max_iterations, 9);
  EXPECT_EQ(options.max_line_search_iterations, 10);
  EXPECT_DOUBLE_EQ(
      options.projected_gradient_inverse_stiffness, 0.75);
  EXPECT_DOUBLE_EQ(options.tangent_trust_radius, 0.125);
  EXPECT_DOUBLE_EQ(options.maximum_coefficient_update_linf, 0.375);
  EXPECT_DOUBLE_EQ(options.line_search_shrink, 0.4);
  EXPECT_DOUBLE_EQ(options.armijo_fraction, 2.0e-4);
  EXPECT_DOUBLE_EQ(options.minimum_volume_merit_penalty, 2.5);

  const auto canonical =
      canonicalLevelSetMaintenanceRequestSchedule(
          requests,
          LevelSetMaintenanceScheduleStage::
              TransientInitialization,
          /*completed_step=*/0);
  ASSERT_TRUE(canonical.supported);
  ASSERT_FALSE(canonical.words.empty());
  EXPECT_EQ(canonical.words.front(), 3u);
  auto changed_requests = requests;
  changed_requests.front()
      .static_capillary_equilibrium
      .physical_equilibrium_max_residual_norm *= 2.0;
  const auto changed =
      canonicalLevelSetMaintenanceRequestSchedule(
          changed_requests,
          LevelSetMaintenanceScheduleStage::
              TransientInitialization,
          /*completed_step=*/0);
  EXPECT_NE(canonical.words, changed.words);
}

TEST(ApplicationDriverLevelSetWorkflows,
     ParsesAndCanonicalizesCurvatureRecoveryControls)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Level_set_field_name>phi</Level_set_field_name>
    <Enable_curvature_projection>true</Enable_curvature_projection>
    <Curvature_field_name>kappa_projected</Curvature_field_name>
    <Curvature_projection_supplemental_sample_weight>0.125</Curvature_projection_supplemental_sample_weight>
    <Curvature_projection_recovery_mode>generated_interface_patch</Curvature_projection_recovery_mode>
    <Curvature_projection_kinematic_area_gradient_filter_coefficient>0.75</Curvature_projection_kinematic_area_gradient_filter_coefficient>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  const auto requests = levelSetMaintenanceRequests(*params);
  ASSERT_EQ(requests.size(), 1u);
  const auto& request = requests.front();
  EXPECT_TRUE(request.curvature_projection_enabled);
  EXPECT_EQ(request.curvature_field_name, "kappa_projected");
  EXPECT_DOUBLE_EQ(
      request.curvature_projection.supplemental_sample_weight, 0.125);
  EXPECT_EQ(
      request.curvature_projection.recovery_mode,
      svmp::FE::level_set::LevelSetCurvatureRecoveryMode::
          GeneratedInterfacePatch);
  EXPECT_DOUBLE_EQ(
      request.curvature_projection
          .kinematic_area_gradient_filter_coefficient,
      0.75);

  const auto canonical = canonicalLevelSetMaintenanceRequestSchedule(
      requests,
      LevelSetMaintenanceScheduleStage::TransientInitialization,
      /*completed_step=*/0);
  ASSERT_TRUE(canonical.supported);
  auto changed_requests = requests;
  changed_requests.front().curvature_projection.recovery_mode =
      svmp::FE::level_set::LevelSetCurvatureRecoveryMode::LevelSetQuadratic;
  const auto changed = canonicalLevelSetMaintenanceRequestSchedule(
      changed_requests,
      LevelSetMaintenanceScheduleStage::TransientInitialization,
      /*completed_step=*/0);
  EXPECT_NE(canonical.words, changed.words);
  changed_requests = requests;
  changed_requests.front()
      .curvature_projection
      .kinematic_area_gradient_filter_coefficient = 0.5;
  const auto changed_filter = canonicalLevelSetMaintenanceRequestSchedule(
      changed_requests,
      LevelSetMaintenanceScheduleStage::TransientInitialization,
      /*completed_step=*/0);
  EXPECT_NE(canonical.words, changed_filter.words);
}

TEST(ApplicationDriverLevelSetWorkflows,
     KinematicAreaGradientMaintenanceBindsTotalEnergyDeclaration)
{
  constexpr int interface_marker = 731;
  auto mesh = makeWorkflowTriangleMesh();
  auto scalar_space = svmp::FE::spaces::SpaceFactory::create_h1(
      svmp::FE::ElementType::Triangle3, /*order=*/1);
  auto velocity_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, 2);
  svmp::FE::systems::FESystem system(mesh);
  const auto phi = system.addField(svmp::FE::systems::FieldSpec{
      .name = "phi_total_energy_binding",
      .space = scalar_space,
      .components = 1,
      .source_kind =
          svmp::FE::systems::FieldSourceKind::PrescribedData,
  });
  const auto kappa = system.addField(svmp::FE::systems::FieldSpec{
      .name = "kappa_total_energy_binding",
      .space = scalar_space,
      .components = 1,
      .source_kind =
          svmp::FE::systems::FieldSourceKind::PrescribedData,
  });
  const auto velocity = system.addField(svmp::FE::systems::FieldSpec{
      .name = "velocity_total_energy_binding",
      .space = velocity_space,
      .components = 2,
  });
  svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalParameters
      functional_parameters;
  functional_parameters.liquid_side =
      svmp::FE::geometry::CutIntegrationSide::Positive;
  functional_parameters.surface_tension = 0.5;
  functional_parameters.young_wall_coefficients = {
      {.boundary_marker = 19,
       .equilibrium_contact_angle_radians = 2.1},
      {.boundary_marker = 7,
       .equilibrium_contact_angle_radians = 0.9},
  };
  system.declareFreeSurfaceDiscreteFunctional(
      svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration{
          .interface_marker = interface_marker,
          .level_set_field = phi,
          .curvature_field = kappa,
          .velocity_field = velocity,
          .geometry_domain_id = "total_energy_binding",
          .parameters = functional_parameters,
          .endpoint_functional_power_enabled = true,
          .capillary_balance_method = svmp::FE::systems::
              FreeSurfaceCapillaryBalanceMethod::
                  KinematicAreaGradientEnergyTraction,
          .capillary_balance_qualification = svmp::FE::systems::
              FreeSurfaceCapillaryBalanceQualification::PrerequisiteOnly,
          .owner_component = "total_energy_binding_test",
      });

  LevelSetMaintenanceRequest request;
  request.level_set_field_name = "phi_total_energy_binding";
  request.curvature_projection_enabled = true;
  request.curvature_field_name = "kappa_total_energy_binding";
  request.curvature_projection.recovery_mode =
      svmp::FE::level_set::LevelSetCurvatureRecoveryMode::
          KinematicAreaGradient;
  request.curvature_projection
      .kinematic_area_gradient_filter_coefficient = 0.0;
  request.volume_cut_request = application::core::ActiveCutVolumeRequest{
      .level_set_field_name = "phi_total_energy_binding",
      .domain_id = "total_energy_binding",
      .requested_interface_marker = interface_marker,
      .active_side = application::core::LevelSetActiveSide::Positive,
  };
  std::vector<LevelSetMaintenanceRequest> requests{request};
  ASSERT_NO_THROW(bindKinematicAreaGradientTractionMaintenance(
      system, requests));
  ASSERT_EQ(requests.size(), 1u);
  const auto& options = requests.front().curvature_projection;
  EXPECT_FALSE(options.kinematic_area_gradient_negative_liquid_side);
  ASSERT_EQ(options.kinematic_area_gradient_young_walls.size(), 2u);
  EXPECT_EQ(
      options.kinematic_area_gradient_young_walls[0].boundary_marker,
      7);
  EXPECT_DOUBLE_EQ(
      options.kinematic_area_gradient_young_walls[0]
          .equilibrium_contact_angle_radians,
      0.9);
  EXPECT_EQ(
      options.kinematic_area_gradient_young_walls[1].boundary_marker,
      19);
  EXPECT_DOUBLE_EQ(
      options.kinematic_area_gradient_young_walls[1]
          .equilibrium_contact_angle_radians,
      2.1);

  const auto canonical = canonicalLevelSetMaintenanceRequestSchedule(
      requests,
      LevelSetMaintenanceScheduleStage::TransientInitialization,
      /*completed_step=*/0);
  ASSERT_TRUE(canonical.supported);
  auto changed = requests;
  changed.front()
      .curvature_projection
      .kinematic_area_gradient_young_walls[0]
      .equilibrium_contact_angle_radians += 0.1;
  const auto changed_canonical =
      canonicalLevelSetMaintenanceRequestSchedule(
          changed,
          LevelSetMaintenanceScheduleStage::TransientInitialization,
          /*completed_step=*/0);
  EXPECT_NE(canonical.words, changed_canonical.words);

  auto conflicting = requests;
  conflicting.front()
      .curvature_projection
      .kinematic_area_gradient_young_walls[0]
      .equilibrium_contact_angle_radians += 0.25;
  EXPECT_THROW(
      bindKinematicAreaGradientTractionMaintenance(system, conflicting),
      std::runtime_error);
  auto mismatched_isovalue = requests;
  mismatched_isovalue.front().isovalue = 0.125;
  EXPECT_THROW(
      bindKinematicAreaGradientTractionMaintenance(
          system, mismatched_isovalue),
      std::runtime_error);
  auto mismatched_geometry_policy = requests;
  mismatched_geometry_policy.front()
      .volume_cut_request->geometry_tangent_policy =
      svmp::FE::level_set::GeometryTangentPolicy::
          DifferentiatedQuadrature;
  EXPECT_THROW(
      bindKinematicAreaGradientTractionMaintenance(
          system, mismatched_geometry_policy),
      std::runtime_error);
  auto filtered_recovery = requests;
  filtered_recovery.front()
      .curvature_projection
      .kinematic_area_gradient_filter_coefficient = 0.25;
  EXPECT_THROW(
      bindKinematicAreaGradientTractionMaintenance(
          system, filtered_recovery),
      std::runtime_error);
  auto wrong_recovery = requests;
  wrong_recovery.front().curvature_projection.recovery_mode =
      svmp::FE::level_set::LevelSetCurvatureRecoveryMode::
          GeneratedInterfacePatch;
  EXPECT_THROW(
      bindKinematicAreaGradientTractionMaintenance(
          system, wrong_recovery),
      std::runtime_error);
}

TEST(ApplicationDriverLevelSetWorkflows,
     StandaloneKinematicAreaGradientDiagnosticRetainsFilterConfiguration)
{
  auto mesh = makeWorkflowTriangleMesh();
  auto scalar_space = svmp::FE::spaces::SpaceFactory::create_h1(
      svmp::FE::ElementType::Triangle3, /*order=*/1);
  svmp::FE::systems::FESystem system(mesh);
  system.addField(svmp::FE::systems::FieldSpec{
      .name = "phi_standalone_area_gradient",
      .space = scalar_space,
      .components = 1,
      .source_kind =
          svmp::FE::systems::FieldSourceKind::PrescribedData,
  });
  system.addField(svmp::FE::systems::FieldSpec{
      .name = "kappa_standalone_area_gradient",
      .space = scalar_space,
      .components = 1,
      .source_kind =
          svmp::FE::systems::FieldSourceKind::PrescribedData,
  });

  LevelSetMaintenanceRequest request;
  request.level_set_field_name = "phi_standalone_area_gradient";
  request.curvature_projection_enabled = true;
  request.curvature_field_name = "kappa_standalone_area_gradient";
  request.curvature_projection.recovery_mode =
      svmp::FE::level_set::LevelSetCurvatureRecoveryMode::
          KinematicAreaGradient;
  request.curvature_projection
      .kinematic_area_gradient_filter_coefficient = 0.5;
  request.curvature_projection
      .kinematic_area_gradient_negative_liquid_side = false;
  request.curvature_projection.kinematic_area_gradient_young_walls = {
      {.boundary_marker = 17,
       .equilibrium_contact_angle_radians = 1.2},
  };
  std::vector<LevelSetMaintenanceRequest> requests{request};

  ASSERT_NO_THROW(bindKinematicAreaGradientTractionMaintenance(
      system, requests));
  ASSERT_EQ(requests.size(), 1u);
  const auto& options = requests.front().curvature_projection;
  EXPECT_DOUBLE_EQ(
      options.kinematic_area_gradient_filter_coefficient, 0.5);
  EXPECT_FALSE(options.kinematic_area_gradient_negative_liquid_side);
  ASSERT_EQ(options.kinematic_area_gradient_young_walls.size(), 1u);
  EXPECT_EQ(
      options.kinematic_area_gradient_young_walls.front().boundary_marker,
      17);
  EXPECT_DOUBLE_EQ(
      options.kinematic_area_gradient_young_walls.front()
          .equilibrium_contact_angle_radians,
      1.2);
  EXPECT_FALSE(requests.front().volume_cut_request.has_value());
}

TEST(ApplicationDriverLevelSetWorkflows,
     UncoupledNewtonControlsRetainPrimaryEquationCompatibility)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Min_iterations>2</Min_iterations>
    <Max_iterations>4</Max_iterations>
    <Tolerance>1.0e-4</Tolerance>
  </Add_equation>
  <Add_equation type="fluid">
    <Min_iterations>1</Min_iterations>
    <Max_iterations>12</Max_iterations>
    <Tolerance>2.0e-2</Tolerance>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  svmp::FE::timestepping::NewtonOptions options{};
  applyMonolithicEquationNewtonControls(*params, options);

  EXPECT_EQ(options.min_iterations, 1);
  EXPECT_EQ(options.max_iterations, 12);
  EXPECT_DOUBLE_EQ(options.rel_tolerance, 2.0e-2);
  EXPECT_DOUBLE_EQ(options.abs_tolerance, 1.0e-10);
}

TEST(ApplicationDriverLevelSetWorkflows,
     CoupledLevelSetAddsNamedFieldAbsoluteAndRelativeResidualCriterion)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Coupled>true</Coupled>
    <Tolerance>1.0e-4</Tolerance>
    <Level_set_field_name>phi</Level_set_field_name>
    <LS type="GMRES">
      <Absolute_tolerance>2.0e-12</Absolute_tolerance>
    </LS>
  </Add_equation>
  <Add_equation type="fluid">
    <Coupled>true</Coupled>
    <Tolerance>2.0e-2</Tolerance>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  auto mesh = makeWorkflowTriangleMesh();
  auto space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Triangle3, 1);
  svmp::FE::systems::FESystem system(mesh);
  const auto phi = system.addField(svmp::FE::systems::FieldSpec{
      .name = "phi", .space = space, .components = 1});
  ASSERT_NO_THROW(system.setup({}));

  svmp::FE::timestepping::NewtonOptions options{};
  ASSERT_NO_THROW(applyCoupledLevelSetFieldResidualCriteria(
      system, *params, options));

  ASSERT_EQ(options.field_residual_criteria.size(), 1u);
  EXPECT_EQ(options.field_residual_criteria.front().field, phi);
  EXPECT_DOUBLE_EQ(options.field_residual_criteria.front().abs_tolerance,
                   2.0e-12);
  EXPECT_DOUBLE_EQ(options.field_residual_criteria.front().rel_tolerance,
                   1.0e-4);
}

TEST(ApplicationDriverLevelSetWorkflows,
     CoupledLevelSetFieldResidualUsesLinearAbsoluteToleranceDefault)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Coupled>true</Coupled>
    <Tolerance>1.0e-4</Tolerance>
    <Level_set_field_name>phi</Level_set_field_name>
  </Add_equation>
  <Add_equation type="fluid">
    <Coupled>true</Coupled>
    <Tolerance>2.0e-2</Tolerance>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  auto mesh = makeWorkflowTriangleMesh();
  auto space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Triangle3, 1);
  svmp::FE::systems::FESystem system(mesh);
  (void)system.addField(svmp::FE::systems::FieldSpec{
      .name = "phi", .space = space, .components = 1});
  ASSERT_NO_THROW(system.setup({}));

  svmp::FE::timestepping::NewtonOptions options{};
  ASSERT_NO_THROW(applyCoupledLevelSetFieldResidualCriteria(
      system, *params, options));

  ASSERT_EQ(options.field_residual_criteria.size(), 1u);
  EXPECT_DOUBLE_EQ(options.field_residual_criteria.front().abs_tolerance,
                   1.0e-10);
  EXPECT_DOUBLE_EQ(options.field_residual_criteria.front().rel_tolerance,
                   1.0e-4);
}

TEST(ApplicationDriverLevelSetWorkflows,
     CoupledLevelSetFieldResidualMergesStrictestEffectiveTolerances)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Coupled>true</Coupled>
    <Tolerance>1.0e-3</Tolerance>
    <Level_set_field_name>phi</Level_set_field_name>
    <LS type="GMRES">
      <Absolute_tolerance>1.0e-8</Absolute_tolerance>
    </LS>
  </Add_equation>
  <Add_equation type="level_set_transport">
    <Coupled>true</Coupled>
    <Tolerance>1.0e-5</Tolerance>
    <Level_set_field_name>phi</Level_set_field_name>
    <LS type="GMRES">
      <Absolute_tolerance>1.0e-9</Absolute_tolerance>
    </LS>
  </Add_equation>
  <Add_equation type="fluid">
    <Coupled>true</Coupled>
    <Tolerance>2.0e-2</Tolerance>
    <LS type="GMRES">
      <Absolute_tolerance>2.0e-12</Absolute_tolerance>
    </LS>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  auto mesh = makeWorkflowTriangleMesh();
  auto space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Triangle3, 1);
  svmp::FE::systems::FESystem system(mesh);
  (void)system.addField(svmp::FE::systems::FieldSpec{
      .name = "phi", .space = space, .components = 1});
  ASSERT_NO_THROW(system.setup({}));

  svmp::FE::timestepping::NewtonOptions options{};
  ASSERT_NO_THROW(applyCoupledLevelSetFieldResidualCriteria(
      system, *params, options));

  ASSERT_EQ(options.field_residual_criteria.size(), 1u);
  EXPECT_DOUBLE_EQ(options.field_residual_criteria.front().abs_tolerance,
                   1.0e-9);
  EXPECT_DOUBLE_EQ(options.field_residual_criteria.front().rel_tolerance,
                   1.0e-5);
}

TEST(ApplicationDriverLevelSetWorkflows,
     UncoupledLevelSetDoesNotAddFieldResidualCriterion)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Tolerance>1.0e-4</Tolerance>
    <Level_set_field_name>phi</Level_set_field_name>
  </Add_equation>
  <Add_equation type="fluid">
    <Tolerance>2.0e-2</Tolerance>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  auto mesh = makeWorkflowTriangleMesh();
  auto space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Triangle3, 1);
  svmp::FE::systems::FESystem system(mesh);
  (void)system.addField(svmp::FE::systems::FieldSpec{
      .name = "phi", .space = space, .components = 1});
  ASSERT_NO_THROW(system.setup({}));

  svmp::FE::timestepping::NewtonOptions options{};
  ASSERT_NO_THROW(applyCoupledLevelSetFieldResidualCriteria(
      system, *params, options));
  EXPECT_TRUE(options.field_residual_criteria.empty());
}

TEST(ApplicationDriverLevelSetWorkflows,
     TransientLineSearchTrialSynchronizationFollowsResidualContract)
{
  {
    WorkflowScopedEnvVar unset("SVMP_SYNC_LINE_SEARCH_TRIALS", std::nullopt);
    EXPECT_FALSE(synchronizeTransientLineSearchTrials(
        /*residual_defining_state_changes=*/false));
    EXPECT_TRUE(synchronizeTransientLineSearchTrials(
        /*residual_defining_state_changes=*/true));
  }
  {
    WorkflowScopedEnvVar disabled("SVMP_SYNC_LINE_SEARCH_TRIALS",
                                  std::string("0"));
    EXPECT_FALSE(synchronizeTransientLineSearchTrials(
        /*residual_defining_state_changes=*/true));
  }
  {
    WorkflowScopedEnvVar enabled("SVMP_SYNC_LINE_SEARCH_TRIALS",
                                 std::string("1"));
    EXPECT_TRUE(synchronizeTransientLineSearchTrials(
        /*residual_defining_state_changes=*/false));
  }
}

TEST(ApplicationDriverLevelSetWorkflows,
     RejectedTimeStepRestorationBypassesGeneratedStateCadence)
{
  using Point =
      svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;

  EXPECT_TRUE(restoresAcceptedTimeStepGeneratedState(
      Point::RestoredTimeStepState));
  EXPECT_TRUE(restoresAcceptedTimeStepGeneratedState(
      Point::RestoredProjectedTimeStepState));
  EXPECT_FALSE(restoresAcceptedTimeStepGeneratedState(
      Point::RestoredNonlinearState));
  EXPECT_FALSE(restoresAcceptedTimeStepGeneratedState(
      Point::RestoredOuterFixedPointState));

  EXPECT_FALSE(refreshesFrozenLevelSetExtensionAtStateSync(
      Point::RestoredTimeStepState,
      /*use_external_state_fixed_point=*/false));
  EXPECT_TRUE(refreshesFrozenLevelSetExtensionAtStateSync(
      Point::RestoredProjectedTimeStepState,
      /*use_external_state_fixed_point=*/false));
  EXPECT_TRUE(refreshesFrozenLevelSetExtensionAtStateSync(
      Point::AcceptedNonlinearState,
      /*use_external_state_fixed_point=*/true));
}

TEST(ApplicationDriverLevelSetWorkflows,
     ActiveCutGeneralizedAlphaPdeRateInitializationDefaultsOnAndAllowsOptOut)
{
  {
    WorkflowScopedEnvVar unset(
        "SVMP_GENERALIZED_ALPHA_PDE_UDOT_INIT", std::nullopt);
    EXPECT_TRUE(generalizedAlphaPdeRateInitializationRequested(
        /*active_cut_domain_present=*/true));
  }
  {
    WorkflowScopedEnvVar disabled(
        "SVMP_GENERALIZED_ALPHA_PDE_UDOT_INIT", std::string("0"));
    EXPECT_FALSE(generalizedAlphaPdeRateInitializationRequested(
        /*active_cut_domain_present=*/true));
  }
  {
    WorkflowScopedEnvVar enabled(
        "SVMP_GENERALIZED_ALPHA_PDE_UDOT_INIT", std::string("1"));
    EXPECT_TRUE(generalizedAlphaPdeRateInitializationRequested(
        /*active_cut_domain_present=*/true));
  }
}

TEST(ApplicationDriverLevelSetWorkflows,
     SelectsBackwardEulerAndRejectsUnsupportedTransientScheme)
{
  GeneralSimulationParameters parameters;

  const auto default_selection =
      resolveTransientTimeIntegrationSelection(parameters);
  EXPECT_EQ(
      default_selection.scheme,
      svmp::FE::timestepping::SchemeKind::GeneralizedAlpha);
  EXPECT_EQ(default_selection.canonical_name, "GeneralizedAlpha");
  ASSERT_TRUE(
      default_selection.generalized_alpha_rho_inf.has_value());
  EXPECT_DOUBLE_EQ(
      *default_selection.generalized_alpha_rho_inf, 0.5);
  EXPECT_NEAR(
      static_cast<double>(default_selection.stage_alpha_f),
      2.0 / 3.0,
      1.0e-15);

  parameters.transient_time_integration_scheme.set_raw_value(
      "BackwardEuler");
  parameters.spectral_radius_of_infinite_time_step.set_raw_value(
      std::numeric_limits<double>::quiet_NaN());
  const auto backward_euler_selection =
      resolveTransientTimeIntegrationSelection(parameters);
  EXPECT_EQ(
      backward_euler_selection.scheme,
      svmp::FE::timestepping::SchemeKind::BackwardEuler);
  EXPECT_EQ(
      backward_euler_selection.canonical_name, "BackwardEuler");
  EXPECT_FALSE(
      backward_euler_selection.generalized_alpha_rho_inf.has_value());
  EXPECT_DOUBLE_EQ(
      static_cast<double>(backward_euler_selection.stage_alpha_f),
      1.0);

  parameters.transient_time_integration_scheme.set_raw_value(
      "GeneralizedAlpha");
  EXPECT_THROW(
      (void)resolveTransientTimeIntegrationSelection(parameters),
      svmp::FE::InvalidArgumentException);

  parameters.transient_time_integration_scheme.set_raw_value(
      "backwardeuler");
  EXPECT_THROW(
      (void)resolveTransientTimeIntegrationSelection(parameters),
      std::runtime_error);
  parameters.transient_time_integration_scheme.set_raw_value("BDF2");
  EXPECT_THROW(
      (void)resolveTransientTimeIntegrationSelection(parameters),
      std::runtime_error);
}

TEST(ApplicationDriverLevelSetWorkflows,
     LinearCornerRefreshReportsRefreshedJacobianCheckGeometry)
{
  application::core::ActiveCutVolumeRequest request{};
  request.geometry_mode =
      svmp::FE::level_set::GeneratedInterfaceGeometryMode::LinearCorner;
  request.geometry_tangent_policy =
      svmp::FE::level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature;

  svmp::FE::timestepping::NewtonOptions options{};
  applyJacobianCheckGeometryProvenance(
      options,
      std::vector<application::core::ActiveCutVolumeRequest>{request},
      /*refresh_generated_geometry_within_solve=*/true,
      /*has_frozen_algebraic_level_set_extension=*/true);

  EXPECT_EQ(options.jacobian_check_geometry_mode,
            svmp::FE::timestepping::JacobianCheckGeometryMode::
                RefreshedGeometry);
  EXPECT_EQ(options.jacobian_check_geometry_tangent_policy,
            "RefreshedFrozenQuadrature");
}

TEST(ApplicationDriverLevelSetWorkflows,
     FrozenExtensionWithoutCutRefreshReportsFixedJacobianCheckGeometry)
{
  application::core::ActiveCutVolumeRequest request{};
  svmp::FE::timestepping::NewtonOptions options{};
  applyJacobianCheckGeometryProvenance(
      options,
      std::vector<application::core::ActiveCutVolumeRequest>{request},
      /*refresh_generated_geometry_within_solve=*/false,
      /*has_frozen_algebraic_level_set_extension=*/true);

  EXPECT_EQ(options.jacobian_check_geometry_mode,
            svmp::FE::timestepping::JacobianCheckGeometryMode::FixedGeometry);
  EXPECT_EQ(options.jacobian_check_geometry_tangent_policy,
            "fixed-topology algebraic wet-extension solve");
}

TEST(ApplicationDriverLevelSetWorkflows,
     OuterFixedPointReportsFrozenInnerJacobianGeometry)
{
  application::core::ActiveCutVolumeRequest request{};
  request.geometry_mode =
      svmp::FE::level_set::GeneratedInterfaceGeometryMode::LinearCorner;
  request.geometry_tangent_policy =
      svmp::FE::level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature;

  svmp::FE::timestepping::NewtonOptions options{};
  applyJacobianCheckGeometryProvenance(
      options,
      std::vector<application::core::ActiveCutVolumeRequest>{request},
      /*refresh_generated_geometry_within_solve=*/false,
      /*has_frozen_algebraic_level_set_extension=*/false,
      /*use_external_state_fixed_point=*/true);

  EXPECT_EQ(options.jacobian_check_geometry_mode,
            svmp::FE::timestepping::JacobianCheckGeometryMode::FixedGeometry);
  EXPECT_EQ(options.jacobian_check_geometry_tangent_policy,
            "outer-fixed-point frozen geometry (RefreshedFrozenQuadrature)");
}

TEST(ApplicationDriverLevelSetWorkflows,
     RegisteredFittedALEOperatorStageHistoryRejectsUnsupportedScheme)
{
  EXPECT_NO_THROW(requireFittedALEOperatorStageSchemeCoverage(
      /*declaration_count=*/0u,
      /*scheme_supported=*/false,
      "BDF2",
      /*temporal_order=*/1));
  EXPECT_NO_THROW(requireFittedALEOperatorStageSchemeCoverage(
      /*declaration_count=*/1u,
      /*scheme_supported=*/true,
      "BackwardEuler",
      /*temporal_order=*/1));

  try {
    requireFittedALEOperatorStageSchemeCoverage(
        /*declaration_count=*/2u,
        /*scheme_supported=*/false,
        "BDF2",
        /*temporal_order=*/1);
    FAIL() << "Expected registered unsupported fitted-ALE measurements to "
              "fail closed";
  } catch (const std::runtime_error& error) {
    const std::string message(error.what());
    EXPECT_NE(message.find("require a supported temporal-order-one scheme"),
              std::string::npos);
    EXPECT_NE(message.find("scheme='BDF2'"), std::string::npos);
    EXPECT_NE(message.find("declaration_count=2"), std::string::npos);
  }
}

TEST(ApplicationDriverLevelSetWorkflows,
     CutTopologyChangeTraceIdentifiesNonsmoothNewtonEvent)
{
  WorkflowScopedEnvVar trace("SVMP_OOP_SOLVER_TRACE", std::string("1"));

  ActiveCutContextRefreshReport report{};
  report.refreshed = true;
  report.topology_key = 0x2222u;
  report.request_policy_key = 0x3333u;
  report.value_revision = 7u;
  report.cell_count = 2u;
  report.interface_fragments = 1u;
  report.active_volume_regions = 2u;
  report.active_cut_cells = 1u;
  report.active_quadrature_points = 4u;
  report.domain_total_quadrature_point_count = 6u;
  report.backend_volume_quadrature_point_count = 4u;
  report.backend_interface_quadrature_point_count = 2u;

  std::optional<std::uint64_t> previous_topology_key{0x1111u};

  testing::internal::CaptureStdout();
  logCutTopologyChange(
      report,
      svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint::
          LineSearchTrialResidual,
      previous_topology_key,
      "steady");
  const auto output = testing::internal::GetCapturedStdout();

  ASSERT_TRUE(previous_topology_key.has_value());
  EXPECT_EQ(*previous_topology_key, report.topology_key);
  EXPECT_NE(output.find("diagnostic=cut_topology_change_nonsmooth_event"),
            std::string::npos);
  EXPECT_NE(output.find("event_class=nonsmooth_cut_topology_change"),
            std::string::npos);
  EXPECT_NE(output.find("newton_consistency=not_expected"),
            std::string::npos);
  EXPECT_NE(output.find("jacobian_validity=piecewise_smooth_topology_only"),
            std::string::npos);
  EXPECT_NE(output.find("sync_point=line_search_trial"), std::string::npos);
}

TEST(ApplicationDriverLevelSetWorkflows,
     ActiveSupportRefreshEvaluatesHierarchicalLevelSet)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowBiquadraticQuadMesh();
  const auto mesh_field = svmp::MeshFields::attach_field(
      mesh->local_mesh(),
      svmp::EntityKind::Vertex,
      "phi",
      svmp::FieldScalarType::Float64,
      1);
  auto* mesh_phi = svmp::MeshFields::field_data_as<svmp::real_t>(
      mesh->local_mesh(), mesh_field);
  ASSERT_NE(mesh_phi, nullptr);
  std::fill(mesh_phi, mesh_phi + mesh->n_vertices(), 99.0);

  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/2,
      svmp::FE::BasisType::Hierarchical);

  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1});
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_vertex_values(mesh->n_vertices(), 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    phi_vertex_values[vertex] = workflowPhi(*mesh, vertex);
  }
  const auto phi_coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      std::span<const svmp::FE::Real>(phi_vertex_values.data(),
                                      phi_vertex_values.size()),
      1u,
      "ApplicationDriver active refresh hierarchical phi");

  std::vector<svmp::FE::Real> solution(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, phi_coefficients, solution);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);

  application::core::ActiveCutVolumeRequest request{};
  request.level_set_field_name = "phi";
  request.domain_id = "workflow-active-refresh";
  request.active_side = application::core::LevelSetActiveSide::Negative;

  const auto changed = syncActiveLevelSetVertexFieldsFromSolution(
      sim,
      std::vector<application::core::ActiveCutVolumeRequest>{request},
      std::span<const svmp::FE::Real>(solution.data(), solution.size()));
  EXPECT_EQ(changed, 1u);

  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    EXPECT_NEAR(mesh_phi[vertex], phi_vertex_values[vertex], 1.0e-10)
        << "vertex " << vertex;
  }
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     RefreshesMultipleGeneratedCutDomainsIntoOneContext)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowBiquadraticQuadMesh();
  const auto phi_a_mesh_field = svmp::MeshFields::attach_field(
      mesh->local_mesh(),
      svmp::EntityKind::Vertex,
      "phi_a",
      svmp::FieldScalarType::Float64,
      1);
  const auto phi_b_mesh_field = svmp::MeshFields::attach_field(
      mesh->local_mesh(),
      svmp::EntityKind::Vertex,
      "phi_b",
      svmp::FieldScalarType::Float64,
      1);
  ASSERT_NE(svmp::MeshFields::field_data_as<svmp::real_t>(
                mesh->local_mesh(), phi_a_mesh_field),
            nullptr);
  ASSERT_NE(svmp::MeshFields::field_data_as<svmp::real_t>(
                mesh->local_mesh(), phi_b_mesh_field),
            nullptr);

  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/2);

  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi_a = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi_a",
      .space = scalar_space,
      .components = 1});
  const auto phi_b = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi_b",
      .space = scalar_space,
      .components = 1});
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_a_vertex_values(mesh->n_vertices(), 0.0);
  std::vector<svmp::FE::Real> phi_b_vertex_values(mesh->n_vertices(), 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    phi_a_vertex_values[vertex] = workflowPhi(*mesh, vertex);
    phi_b_vertex_values[vertex] = workflowVerticalPhi(*mesh, vertex);
  }
  const auto phi_a_coefficients = projectWorkflowVertexValues(
      *system,
      phi_a,
      std::span<const svmp::FE::Real>(phi_a_vertex_values.data(),
                                      phi_a_vertex_values.size()),
      1u,
      "ApplicationDriver multiple cut-domain phi_a");
  const auto phi_b_coefficients = projectWorkflowVertexValues(
      *system,
      phi_b,
      std::span<const svmp::FE::Real>(phi_b_vertex_values.data(),
                                      phi_b_vertex_values.size()),
      1u,
      "ApplicationDriver multiple cut-domain phi_b");

  std::vector<svmp::FE::Real> solution(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi_a, phi_a_coefficients, solution);
  writeWorkflowFieldSlice(*system, phi_b, phi_b_coefficients, solution);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);

  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="left_free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi_a</Level_set_field_name>
      <Generated_interface_domain_id>left_interface</Generated_interface_domain_id>
      <Interface_marker>701</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
    <Add_BC name="top_free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi_b</Level_set_field_name>
      <Generated_interface_domain_id>top_interface</Generated_interface_domain_id>
      <Interface_marker>702</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetPositive</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
  const auto requests = application::core::activeCutVolumeRequests(*params);
  ASSERT_EQ(requests.size(), 2u);

  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
  const auto report = refreshActiveCutIntegrationContextFromSolution(
      sim,
      *params,
      std::span<const svmp::FE::Real>(solution.data(), solution.size()),
      lifecycle,
      "application-driver-multiple-cut-domain-test");
  EXPECT_TRUE(report.refreshed);
  EXPECT_GE(report.interface_fragments, 2u);
  EXPECT_GT(report.active_volume_regions, 0u);

  const auto* context = sim.fe_system->cutIntegrationContext();
  ASSERT_NE(context, nullptr);
  EXPECT_TRUE(context->hasGeneratedInterfaceMarker(701));
  EXPECT_TRUE(context->hasGeneratedInterfaceMarker(702));
  EXPECT_TRUE(context->hasGeneratedVolumeMarker(701));
  EXPECT_TRUE(context->hasGeneratedVolumeMarker(702));
  EXPECT_FALSE(context->interfaceRulesForMarker(701).empty());
  EXPECT_FALSE(context->interfaceRulesForMarker(702).empty());
  EXPECT_FALSE(context
                   ->generatedVolumeRulesForMarkerAndSide(
                       701,
                       svmp::FE::geometry::CutIntegrationSide::Negative)
                   .empty());
  EXPECT_FALSE(context
                   ->generatedVolumeRulesForMarkerAndSide(
                       702,
                       svmp::FE::geometry::CutIntegrationSide::Positive)
                   .empty());
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     PositiveDryEndpointBuildsCompleteAuthoritativeGeometry)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  using ActiveSide = svmp::FE::geometry::CutIntegrationSide;
  NativeManufacturedChannelHarness harness(
      ActiveSide::Positive,
      /*upper_subdivisions=*/2);
  const auto sample = harness.sample(0.0);
  EXPECT_EQ(sample.target_wet_fraction, 0.0);
  EXPECT_EQ(sample.active_measures,
            (std::array<svmp::FE::Real, 3>{{0.0, 0.0, 0.0}}));
  EXPECT_EQ(sample.active_rule_counts,
            (std::array<std::size_t, 3>{{0u, 0u, 0u}}));
  EXPECT_EQ(sample.retained_active_rule_counts,
            (std::array<std::size_t, 3>{{0u, 0u, 0u}}));
  EXPECT_EQ(sample.operator_work,
            (std::array<svmp::FE::Real, 3>{{0.0, 0.0, 0.0}}));
  EXPECT_EQ(sample.trace_certificate_count, 1u);
  EXPECT_EQ(sample.trace_patch_count, 0u);
  EXPECT_EQ(sample.trace_boundary_rule_count, 0u);
  EXPECT_NE(sample.trace_certificate_digest, 0u);
  EXPECT_TRUE(sample.trace_revision_match);
  EXPECT_TRUE(sample.trace_factorized_proof_valid);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     NativeCertifiedManufacturedChannelTracksSharpBoundaryWork)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  using ActiveSide = svmp::FE::geometry::CutIntegrationSide;
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
      kNativeChannelWindowHeight * kNativeChannelDepth,
      kNativeChannelWindowHeight * kNativeChannelDepth,
      kNativeChannelLength * kNativeChannelWindowHeight,
  }};
  constexpr std::array<svmp::FE::Real, 3> full_work{{
      NativeManufacturedChannelHarness::expectedFullForceWork(),
      NativeManufacturedChannelHarness::expectedFullFluxWork(),
      NativeManufacturedChannelHarness::expectedFullPenaltyWork(),
  }};

  std::array<std::vector<NativeManufacturedChannelSample>, 2>
      samples_by_side;
  svmp::FE::Real maximum_measure_error = 0.0;
  svmp::FE::Real maximum_work_error = 0.0;
  svmp::FE::Real maximum_dry_work_magnitude = 0.0;
  svmp::FE::Real maximum_vertex_limit_mismatch = 0.0;
  std::size_t minimum_positive_trace_patch_count =
      std::numeric_limits<std::size_t>::max();
  std::size_t maximum_localized_root_patch_count = 0u;
  std::size_t maximum_factorized_input_dimension = 0u;
  std::size_t maximum_trace_support_overlap = 0u;
  svmp::FE::Real maximum_trace_upper_bound = 0.0;
  svmp::FE::Real maximum_trace_ratio = 0.0;

  for (std::size_t side_index = 0u;
       side_index < active_sides.size();
       ++side_index) {
    NativeManufacturedChannelHarness harness(
        active_sides[side_index],
        /*upper_subdivisions=*/2);
    auto& samples = samples_by_side[side_index];
    samples.reserve(wet_fractions.size());
    for (const auto fraction : wet_fractions) {
      SCOPED_TRACE(::testing::Message()
                   << "active_side_index=" << side_index
                   << " wet_fraction=" << fraction);
      samples.push_back(harness.sample(fraction));
      const auto& sample = samples.back();
      EXPECT_EQ(sample.target_wet_fraction, fraction);
      EXPECT_EQ(sample.trace_certificate_count, 1u);
      EXPECT_NE(sample.trace_certificate_digest, 0u);
      EXPECT_TRUE(sample.trace_revision_match);
      EXPECT_TRUE(sample.trace_factorized_proof_valid);
      EXPECT_LE(
          sample.trace_maximum_factorized_input_dimension,
          svmp::FE::math::dense_exact_dyadic_maximum_dimension);
      maximum_localized_root_patch_count = std::max(
          maximum_localized_root_patch_count,
          sample.trace_localized_root_patch_count);
      maximum_factorized_input_dimension = std::max(
          maximum_factorized_input_dimension,
          sample.trace_maximum_factorized_input_dimension);
      maximum_trace_support_overlap = std::max(
          maximum_trace_support_overlap,
          sample.trace_maximum_support_overlap);
      maximum_trace_upper_bound = std::max(
          maximum_trace_upper_bound,
          sample.trace_global_conservative_upper_bound);
      maximum_trace_ratio = std::max(
          maximum_trace_ratio,
          sample.trace_grouped_symmetric_ratio);
      EXPECT_GE(sample.trace_global_conservative_upper_bound,
                sample.trace_maximum_patch_conservative_upper_bound);
      EXPECT_EQ(sample.trace_to_penalty_ratio,
                sample.trace_grouped_symmetric_ratio);
      EXPECT_EQ(sample.trace_symmetric_energy_floor, 0.25);
      EXPECT_LE(sample.trace_grouped_symmetric_ratio,
                (1.0 - sample.trace_symmetric_energy_floor) *
                    (1.0 - sample.trace_symmetric_energy_floor));
      EXPECT_EQ(sample.physical_role_boundary_term_count, 0u);
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
        maximum_measure_error = std::max(
            maximum_measure_error,
            std::abs(
                sample.parent_measures[role] -
                parent_measures[role]));
        maximum_measure_error = std::max(
            maximum_measure_error,
            std::abs(
                sample.active_measures[role] -
                fraction * parent_measures[role]));
        maximum_work_error = std::max(
            maximum_work_error,
            std::abs(
                sample.operator_work[role] -
                fraction * full_work[role]));
      }

      if (fraction == 0.0) {
        EXPECT_EQ(sample.active_rule_counts,
                  (std::array<std::size_t, 3>{{0u, 0u, 0u}}));
        EXPECT_EQ(sample.trace_patch_count, 0u);
        EXPECT_EQ(sample.trace_boundary_rule_count, 0u);
        for (const auto value : sample.operator_work) {
          maximum_dry_work_magnitude = std::max(
              maximum_dry_work_magnitude, std::abs(value));
        }
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
        minimum_positive_trace_patch_count = std::min(
            minimum_positive_trace_patch_count,
            sample.trace_patch_count);
      }
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
      const auto crossing_value =
          samples[crossing_index].operator_work[role];
      maximum_vertex_limit_mismatch = std::max(
          maximum_vertex_limit_mismatch,
          std::max(std::abs(left_limit - crossing_value),
                   std::abs(right_limit - crossing_value)));
    }
  }

  svmp::FE::Real maximum_active_side_work_difference = 0.0;
  ASSERT_EQ(samples_by_side[0].size(), samples_by_side[1].size());
  for (std::size_t sample_index = 0u;
       sample_index < samples_by_side[0].size();
       ++sample_index) {
    for (std::size_t role = 0u; role < full_work.size(); ++role) {
      maximum_active_side_work_difference = std::max(
          maximum_active_side_work_difference,
          std::abs(
              samples_by_side[0][sample_index].operator_work[role] -
              samples_by_side[1][sample_index].operator_work[role]));
    }
  }

  EXPECT_EQ(maximum_dry_work_magnitude, 0.0);
  EXPECT_LE(maximum_measure_error, 5.0e-11);
  EXPECT_LE(maximum_work_error, 5.0e-10);
  EXPECT_LE(maximum_vertex_limit_mismatch, 5.0e-10);
  EXPECT_LE(maximum_active_side_work_difference, 5.0e-10);
  EXPECT_GT(minimum_positive_trace_patch_count, 0u);
  EXPECT_GT(maximum_localized_root_patch_count, 0u);
  EXPECT_LE(
      maximum_factorized_input_dimension,
      svmp::FE::math::dense_exact_dyadic_maximum_dimension);
  const auto record_real = [](const char* name, svmp::FE::Real value) {
    std::ostringstream text;
    text << std::setprecision(
                std::numeric_limits<svmp::FE::Real>::max_digits10)
         << value;
    ::testing::Test::RecordProperty(name, text.str());
  };
  ::testing::Test::RecordProperty(
      "native_channel_active_side_count",
      static_cast<int>(active_sides.size()));
  ::testing::Test::RecordProperty(
      "native_channel_wet_fraction_count",
      static_cast<int>(wet_fractions.size()));
  ::testing::Test::RecordProperty(
      "native_channel_boundary_role_count", 3);
  ::testing::Test::RecordProperty(
      "native_channel_minimum_positive_trace_patch_count",
      static_cast<int>(minimum_positive_trace_patch_count));
  ::testing::Test::RecordProperty(
      "native_channel_maximum_localized_root_patch_count",
      static_cast<int>(maximum_localized_root_patch_count));
  ::testing::Test::RecordProperty(
      "native_channel_maximum_factorized_input_dimension",
      static_cast<int>(maximum_factorized_input_dimension));
  ::testing::Test::RecordProperty(
      "native_channel_maximum_trace_support_overlap",
      static_cast<int>(maximum_trace_support_overlap));
  record_real(
      "native_channel_maximum_trace_upper_bound",
      maximum_trace_upper_bound);
  record_real(
      "native_channel_maximum_trace_ratio",
      maximum_trace_ratio);
  record_real(
      "native_channel_maximum_measure_error",
      maximum_measure_error);
  record_real(
      "native_channel_maximum_work_error",
      maximum_work_error);
  record_real(
      "native_channel_maximum_vertex_limit_mismatch",
      maximum_vertex_limit_mismatch);
  record_real(
      "native_channel_maximum_active_side_work_difference",
      maximum_active_side_work_difference);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     DefaultSmallCutAggregationRetainsInactiveCutVolumeRules)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowBiquadraticQuadMesh();
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
      /*order=*/2);

  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1});
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_vertex_values(mesh->n_vertices(), 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    phi_vertex_values[vertex] = workflowPhi(*mesh, vertex);
  }
  const auto phi_coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      std::span<const svmp::FE::Real>(phi_vertex_values.data(),
                                      phi_vertex_values.size()),
      1u,
      "ApplicationDriver aggregation cut-retention phi");

  std::vector<svmp::FE::Real> solution(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, phi_coefficients, solution);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);

  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>extension_interface</Generated_interface_domain_id>
      <Interface_marker>703</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
  const auto requests = application::core::activeCutVolumeRequests(*params);
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests.front().volume_retention,
            application::core::ActiveCutVolumeRetention::ActiveAndInactive);

  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
  const auto report = refreshActiveCutIntegrationContextFromSolution(
      sim,
      *params,
      std::span<const svmp::FE::Real>(solution.data(), solution.size()),
      lifecycle,
      "application-driver-aggregation-retention-test");
  EXPECT_TRUE(report.refreshed);

  const auto* context = sim.fe_system->cutIntegrationContext();
  ASSERT_NE(context, nullptr);
  EXPECT_TRUE(context->hasGeneratedInterfaceMarker(703));
  EXPECT_TRUE(context->hasGeneratedVolumeMarker(703));
  EXPECT_FALSE(context
                   ->generatedVolumeRulesForMarkerAndSide(
                       703,
                       svmp::FE::geometry::CutIntegrationSide::Negative)
                   .empty());
  EXPECT_FALSE(context
                   ->generatedVolumeRulesForMarkerAndSide(
                       703,
                       svmp::FE::geometry::CutIntegrationSide::Positive)
                   .empty());
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     AcceptedFunctionalUsesAuthoritativeSnapshotAndRecordsGlobalState)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr int interface_marker = 707;
  constexpr svmp::FE::Real gamma = svmp::FE::Real{0.65};
  auto mesh = makeWorkflowBiquadraticQuadMesh();
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
      /*order=*/2);
  auto velocity_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(
          scalar_space, 2);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1});
  const auto velocity =
      system->addField(svmp::FE::systems::FieldSpec{
          .name = "Velocity",
          .space = velocity_space,
          .components = 2});
  svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalParameters parameters;
  parameters.liquid_side =
      svmp::FE::geometry::CutIntegrationSide::Negative;
  parameters.surface_tension = gamma;
  system->declareFreeSurfaceDiscreteFunctional(
      svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration{
          .interface_marker = interface_marker,
          .level_set_field = phi,
          .velocity_field = velocity,
          .geometry_domain_id = "functional_interface",
          .parameters = parameters,
          .active_volume_energy_parameters =
              svmp::FE::interfaces::
                  FreeSurfaceActiveVolumeEnergyParameters{
                      .liquid_side =
                          svmp::FE::geometry::CutIntegrationSide::Negative,
                      .density = svmp::FE::Real{2.0},
                      .gravitational_acceleration =
                          {{svmp::FE::Real{0.0},
                            svmp::FE::Real{-1.0},
                            svmp::FE::Real{0.0}}},
                      .gravitational_reference_point =
                          {{svmp::FE::Real{0.0},
                            svmp::FE::Real{0.0},
                            svmp::FE::Real{0.0}}},
                  },
          .active_volume_dissipation_parameters =
              svmp::FE::interfaces::
                  FreeSurfaceActiveVolumeDissipationParameters{
                      .liquid_side =
                          svmp::FE::geometry::CutIntegrationSide::Negative,
                      .dynamic_viscosity = svmp::FE::Real{0.5},
                  },
          .external_pressure_power_parameters =
              svmp::FE::interfaces::
                  FreeSurfaceExternalPressurePowerParameters{
                      .liquid_side =
                          svmp::FE::geometry::CutIntegrationSide::Negative,
                      .external_pressure = svmp::FE::Real{2.5},
                  },
          .endpoint_functional_power_enabled = true,
          .capillary_balance_method =
              svmp::FE::systems::
                  FreeSurfaceCapillaryBalanceMethod::
                      DiscreteEnergyVolumeStationarity,
          .capillary_balance_qualification =
              svmp::FE::systems::
                  FreeSurfaceCapillaryBalanceQualification::
                      PrerequisiteOnly,
          .owner_component =
              "ApplicationDriverLevelSetWorkflows.FunctionalFixture",
      });
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_vertex_values(mesh->n_vertices(), 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    phi_vertex_values[vertex] = workflowPhi(*mesh, vertex);
  }
  const auto phi_coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      std::span<const svmp::FE::Real>(phi_vertex_values.data(),
                                      phi_vertex_values.size()),
      1u,
      "ApplicationDriver accepted functional phi");
  std::vector<svmp::FE::Real> velocity_vertex_values(
      mesh->n_vertices() * 2u, svmp::FE::Real{0.0});
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    velocity_vertex_values[2u * vertex] = svmp::FE::Real{2.0};
    velocity_vertex_values[2u * vertex + 1u] =
        svmp::FE::Real{-1.0};
  }
  const auto velocity_coefficients = projectWorkflowVertexValues(
      *system,
      velocity,
      velocity_vertex_values,
      2u,
      "ApplicationDriver accepted functional velocity");
  std::vector<svmp::FE::Real> solution(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, phi_coefficients, solution);
  writeWorkflowFieldSlice(
      *system, velocity, velocity_coefficients, solution);
  std::vector<svmp::FE::Real> previous_velocity_vertex_values(
      mesh->n_vertices() * 2u, svmp::FE::Real{0.0});
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    previous_velocity_vertex_values[2u * vertex] =
        svmp::FE::Real{1.0};
  }
  const auto previous_velocity_coefficients =
      projectWorkflowVertexValues(
          *system,
          velocity,
          previous_velocity_vertex_values,
          2u,
          "ApplicationDriver previous accepted functional velocity");
  auto previous_solution = solution;
  writeWorkflowFieldSlice(
      *system,
      velocity,
      previous_velocity_coefficients,
      previous_solution);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>functional_interface</Generated_interface_domain_id>
      <Interface_marker>707</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
  const auto report = refreshActiveCutIntegrationContextFromSolution(
      sim,
      *params,
      std::span<const svmp::FE::Real>(solution.data(), solution.size()),
      lifecycle,
      "application-driver-accepted-functional-test");
  ASSERT_TRUE(report.refreshed);
  const auto current_functionals =
      evaluateCurrentFreeSurfaceDiscreteFunctionals(sim);
  const auto maintenance_functionals =
      levelSetMaintenanceFunctionalValues(
          sim, current_functionals, solution);
  ASSERT_EQ(maintenance_functionals.size(), 1u);
  EXPECT_EQ(
      maintenance_functionals.front().snapshot_revision,
      current_functionals.front()
          .geometry_revision.snapshot_revision_key);
  EXPECT_EQ(
      maintenance_functionals.front().mesh_topology_revision,
      current_functionals.front()
          .geometry_revision.mesh_topology_revision);
  EXPECT_NE(
      maintenance_functionals.front().cut_topology_revision, 0u);
  ASSERT_TRUE(
      maintenance_functionals.front().kinetic_energy.has_value());
  ASSERT_TRUE(
      maintenance_functionals.front()
          .gravitational_energy.has_value());
  ASSERT_TRUE(
      maintenance_functionals.front()
          .gravitational_potential_power.has_value());
  ASSERT_TRUE(
      maintenance_functionals.front()
          .modeled_stored_energy.has_value());
  ASSERT_TRUE(
      maintenance_functionals.front()
          .surface_wall_potential_power.has_value());
  ASSERT_TRUE(
      maintenance_functionals.front()
          .bulk_viscous_dissipation_rate.has_value());
  ASSERT_TRUE(
      maintenance_functionals.front()
          .external_pressure_power.has_value());
  EXPECT_NEAR(
      *maintenance_functionals.front().kinetic_energy,
      svmp::FE::Real{5.0} *
          maintenance_functionals.front().liquid_volume,
      1.0e-13);
  EXPECT_NEAR(
      *maintenance_functionals.front()
           .gravitational_potential_power,
      svmp::FE::Real{-2.0} *
          maintenance_functionals.front().liquid_volume,
      1.0e-13);
  EXPECT_NEAR(
      *maintenance_functionals.front().modeled_stored_energy,
      *maintenance_functionals.front().kinetic_energy +
          *maintenance_functionals.front().gravitational_energy +
          maintenance_functionals.front().surface_energy +
          maintenance_functionals.front().young_wall_energy,
      1.0e-13);
  EXPECT_NEAR(
      *maintenance_functionals.front()
           .surface_wall_potential_power,
      svmp::FE::Real{0.0},
      1.0e-13);
  EXPECT_NEAR(
      *maintenance_functionals.front()
           .bulk_viscous_dissipation_rate,
      svmp::FE::Real{0.0},
      1.0e-13);
  EXPECT_TRUE(std::isfinite(
      *maintenance_functionals.front().external_pressure_power));
  const auto accepted_state_revision =
      collectiveLevelSetMaintenanceAlgebraicRevision(
          solution,
          activeFESystemCommunicator(*sim.fe_system));
  const auto previous_state_revision =
      collectiveLevelSetMaintenanceAlgebraicRevision(
          previous_solution,
          activeFESystemCommunicator(*sim.fe_system));
  const auto previous_velocity_revision =
      collectiveFreeSurfaceFieldRevision(
          *sim.fe_system,
          velocity,
          previous_solution,
          activeFESystemCommunicator(*sim.fe_system),
          "ApplicationDriver previous velocity revision");
  const auto endpoint_velocity_revision =
      collectiveFreeSurfaceFieldRevision(
          *sim.fe_system,
          velocity,
          solution,
          activeFESystemCommunicator(*sim.fe_system),
          "ApplicationDriver endpoint velocity revision");
  ASSERT_NO_THROW(recordInitialFreeSurfaceDiscreteFunctionalBaseline(
      sim,
      /*initial_step=*/0u,
      svmp::FE::Real{0.0},
      std::span<const svmp::FE::Real>(
          previous_solution.data(), previous_solution.size()),
      /*record_backward_euler_kinetic_baseline=*/true));
  const auto baseline_history =
      sim.fe_system->freeSurfaceDiscreteFunctionalHistory();
  ASSERT_EQ(baseline_history.size(), 1u);
  const auto& baseline = baseline_history.front();
  EXPECT_EQ(baseline.accepted_step, 0u);
  EXPECT_DOUBLE_EQ(baseline.accepted_time, svmp::FE::Real{0.0});
  EXPECT_DOUBLE_EQ(baseline.dt, svmp::FE::Real{0.0});
  EXPECT_EQ(
      baseline.pre_maintenance_endpoint_state_revision,
      previous_state_revision);
  EXPECT_EQ(baseline.state_revision, previous_state_revision);
  EXPECT_NE(baseline.cut_topology_revision, 0u);
  ASSERT_TRUE(baseline.endpoint_functional_power.has_value());
  EXPECT_NEAR(
      baseline.endpoint_functional_power
          ->total_potential_variation,
      svmp::FE::Real{0.0},
      1.0e-13);
  ASSERT_TRUE(baseline.active_volume_energy.has_value());
  EXPECT_NEAR(
      baseline.active_volume_energy
          ->gravitational_potential_power,
      svmp::FE::Real{0.0},
      1.0e-13);
  ASSERT_TRUE(baseline.active_volume_dissipation.has_value());
  EXPECT_NEAR(
      baseline.active_volume_dissipation
          ->bulk_viscous_dissipation_rate,
      svmp::FE::Real{0.0},
      1.0e-13);
  ASSERT_TRUE(baseline.external_pressure_power.has_value());
  EXPECT_NEAR(
      baseline.external_pressure_power->owned_liquid_gas_area,
      baseline.state.owned_liquid_gas_area,
      1.0e-13);
  EXPECT_NEAR(
      baseline.external_pressure_power->external_pressure_power,
      -svmp::FE::Real{2.5} *
          baseline.external_pressure_power
              ->outward_liquid_volume_flux_rate,
      1.0e-13);
  ASSERT_TRUE(baseline.backward_euler_kinetic_work.has_value());
  EXPECT_EQ(
      baseline.backward_euler_kinetic_work
          ->previous_velocity_revision,
      previous_velocity_revision);
  EXPECT_EQ(
      baseline.backward_euler_kinetic_work
          ->endpoint_velocity_revision,
      previous_velocity_revision);
  EXPECT_NEAR(
      baseline.backward_euler_kinetic_work
          ->kinetic_energy_before_on_endpoint_domain,
      baseline.active_volume_energy->kinetic_energy,
      1.0e-13);
  EXPECT_NEAR(
      baseline.backward_euler_kinetic_work
          ->kinetic_energy_after,
      baseline.active_volume_energy->kinetic_energy,
      1.0e-13);
  EXPECT_DOUBLE_EQ(
      baseline.backward_euler_kinetic_work
          ->step_integrated_inertia_work,
      svmp::FE::Real{0.0});
  EXPECT_DOUBLE_EQ(
      baseline.backward_euler_kinetic_work
          ->time_discretization_loss,
      svmp::FE::Real{0.0});
  EXPECT_DOUBLE_EQ(
      baseline.backward_euler_kinetic_work->identity_residual,
      svmp::FE::Real{0.0});
  EXPECT_THROW(
      recordAcceptedFreeSurfaceDiscreteFunctionals(
          sim,
          /*accepted_step=*/3u,
          svmp::FE::Real{0.15},
          svmp::FE::Real{0.05},
          accepted_state_revision,
          accepted_state_revision,
          {},
          std::span<const svmp::FE::Real>(
              solution.data(), solution.size()),
          std::span<const svmp::FE::Real>(
              solution.data(), solution.size())),
      std::runtime_error);
  EXPECT_EQ(
      sim.fe_system->freeSurfaceDiscreteFunctionalHistory().size(),
      1u);
  ASSERT_NO_THROW(recordAcceptedFreeSurfaceDiscreteFunctionals(
      sim,
      /*accepted_step=*/3u,
      svmp::FE::Real{0.15},
      svmp::FE::Real{0.05},
      accepted_state_revision,
      accepted_state_revision,
      {},
      std::span<const svmp::FE::Real>(
          solution.data(), solution.size()),
      std::span<const svmp::FE::Real>(
          previous_solution.data(), previous_solution.size())));

  const auto history = sim.fe_system->freeSurfaceDiscreteFunctionalHistory();
  ASSERT_EQ(history.size(), 2u);
  const auto& record = history.back();
  EXPECT_EQ(record.accepted_step, 3u);
  EXPECT_EQ(
      record.pre_maintenance_endpoint_state_revision,
      accepted_state_revision);
  EXPECT_EQ(record.state_revision, accepted_state_revision);
  EXPECT_NE(record.cut_topology_revision, 0u);
  EXPECT_TRUE(record.geometry_revision.complete());
  EXPECT_EQ(record.geometry_revision.interface_marker, interface_marker);
  EXPECT_EQ(record.geometry_revision.domain_id, "functional_interface");
  EXPECT_GT(record.state.owned_liquid_volume, 0.0);
  EXPECT_GT(record.state.owned_liquid_gas_area, 0.0);
  EXPECT_NEAR(record.state.liquid_gas_surface_energy,
              gamma * record.state.owned_liquid_gas_area,
              1.0e-13);
  EXPECT_NEAR(record.state.total_potential,
              record.state.liquid_gas_surface_energy,
              1.0e-13);
  ASSERT_TRUE(record.endpoint_functional_power.has_value());
  EXPECT_NEAR(
      record.endpoint_functional_power
          ->total_potential_variation,
      svmp::FE::Real{0.0},
      1.0e-13);
  ASSERT_TRUE(record.active_volume_energy.has_value());
  ASSERT_TRUE(record.active_volume_dissipation.has_value());
  EXPECT_NEAR(
      record.active_volume_dissipation->owned_liquid_volume,
      record.state.owned_liquid_volume,
      1.0e-13);
  EXPECT_NEAR(
      record.active_volume_dissipation
          ->bulk_viscous_dissipation_rate,
      svmp::FE::Real{0.0},
      1.0e-13);
  ASSERT_TRUE(record.external_pressure_power.has_value());
  EXPECT_NEAR(
      record.external_pressure_power->owned_liquid_gas_area,
      record.state.owned_liquid_gas_area,
      1.0e-13);
  EXPECT_NEAR(
      record.external_pressure_power->external_pressure_power,
      -svmp::FE::Real{2.5} *
          record.external_pressure_power
              ->outward_liquid_volume_flux_rate,
      1.0e-13);
  EXPECT_NEAR(
      record.active_volume_energy->owned_liquid_volume,
      record.state.owned_liquid_volume,
      1.0e-13);
  EXPECT_NEAR(
      record.active_volume_energy->kinetic_energy,
      svmp::FE::Real{5.0} * record.state.owned_liquid_volume,
      1.0e-13);
  EXPECT_TRUE(
      std::isfinite(
          record.active_volume_energy->gravitational_energy));
  EXPECT_NEAR(
      record.active_volume_energy
          ->gravitational_potential_power,
      svmp::FE::Real{-2.0} *
          record.state.owned_liquid_volume,
      1.0e-13);
  EXPECT_NEAR(
      record.active_volume_energy->total_energy,
      record.active_volume_energy->kinetic_energy +
          record.active_volume_energy->gravitational_energy,
      1.0e-13);
  ASSERT_TRUE(record.backward_euler_kinetic_work.has_value());
  EXPECT_EQ(
      record.backward_euler_kinetic_work
          ->previous_velocity_revision,
      previous_velocity_revision);
  EXPECT_EQ(
      record.backward_euler_kinetic_work
          ->endpoint_velocity_revision,
      endpoint_velocity_revision);
  EXPECT_NEAR(
      record.backward_euler_kinetic_work
          ->kinetic_energy_before_on_endpoint_domain,
      record.state.owned_liquid_volume,
      1.0e-13);
  EXPECT_NEAR(
      record.backward_euler_kinetic_work->kinetic_energy_after,
      svmp::FE::Real{5.0} * record.state.owned_liquid_volume,
      1.0e-13);
  EXPECT_NEAR(
      record.backward_euler_kinetic_work
          ->kinetic_energy_change_on_endpoint_domain,
      svmp::FE::Real{4.0} * record.state.owned_liquid_volume,
      1.0e-13);
  EXPECT_NEAR(
      record.backward_euler_kinetic_work
          ->step_integrated_inertia_work,
      svmp::FE::Real{6.0} * record.state.owned_liquid_volume,
      1.0e-13);
  EXPECT_NEAR(
      record.backward_euler_kinetic_work
          ->time_discretization_loss,
      svmp::FE::Real{2.0} * record.state.owned_liquid_volume,
      1.0e-13);
  EXPECT_NEAR(
      record.backward_euler_kinetic_work->identity_residual,
      svmp::FE::Real{0.0},
      1.0e-13);
  ASSERT_NO_THROW(recordAcceptedFreeSurfaceDiscreteFunctionals(
      sim,
      /*accepted_step=*/3u,
      svmp::FE::Real{0.15},
      svmp::FE::Real{0.05},
      accepted_state_revision,
      accepted_state_revision,
      {},
      std::span<const svmp::FE::Real>(
          solution.data(), solution.size()),
      std::span<const svmp::FE::Real>(
          previous_solution.data(), previous_solution.size())));
  EXPECT_EQ(sim.fe_system->freeSurfaceDiscreteFunctionalHistory().size(), 2u);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     StaticCapillaryHistoryStagingPreservesUnrelatedHistoryAndRates)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowTriangleMesh();
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Triangle3,
      /*order=*/1);
  svmp::FE::systems::FESystem system(mesh);
  const auto phi = system.addField(
      svmp::FE::systems::FieldSpec{
          .name = "phi",
          .space = scalar_space,
          .components = 1});
  const auto passive = system.addField(
      svmp::FE::systems::FieldSpec{
          .name = "passive",
          .space = scalar_space,
          .components = 1});
  ASSERT_NO_THROW(system.setup({}));

  std::unique_ptr<svmp::FE::backends::BackendFactory> factory;
  try {
    factory =
        svmp::FE::backends::BackendFactory::create(
            svmp::FE::backends::BackendKind::Eigen);
  } catch (const std::exception&) {
    GTEST_SKIP() << "Requires the Eigen FE backend.";
  }
  ASSERT_NE(factory, nullptr);
  auto history =
      svmp::FE::timestepping::TimeHistory::allocate(
          *factory,
          system.dofHandler().getNumDofs(),
          /*history_depth=*/2,
          /*allocate_second_order_state=*/true);

  const auto solution_size = static_cast<std::size_t>(
      system.dofHandler().getNumDofs());
  const auto phi_offset = static_cast<std::size_t>(
      system.fieldDofOffset(phi));
  const auto phi_count = static_cast<std::size_t>(
      system.fieldDofHandler(phi).getNumDofs());
  const auto passive_offset = static_cast<std::size_t>(
      system.fieldDofOffset(passive));
  const auto passive_count = static_cast<std::size_t>(
      system.fieldDofHandler(passive).getNumDofs());
  const auto make_state =
      [&](svmp::FE::Real phi_base,
          svmp::FE::Real passive_base) {
        std::vector<svmp::FE::Real> values(
            solution_size, 0.0);
        for (std::size_t i = 0u; i < phi_count; ++i) {
          values[phi_offset + i] =
              phi_base + static_cast<svmp::FE::Real>(i);
        }
        for (std::size_t i = 0u;
             i < passive_count;
             ++i) {
          values[passive_offset + i] =
              passive_base +
              static_cast<svmp::FE::Real>(i);
        }
        return values;
      };

  const auto current = make_state(1.0, 10.0);
  const auto previous = make_state(4.0, 20.0);
  const auto older = make_state(7.0, 30.0);
  const auto certified = make_state(40.0, 50.0);
  const auto rate = make_state(3.0, 60.0);
  const auto acceleration = make_state(6.0, 70.0);
  scatterFeOrderedSolution(history.u(), current);
  scatterFeOrderedSolution(history.uPrev(), previous);
  scatterFeOrderedSolution(history.uPrev2(), older);
  scatterFeOrderedSolution(history.uDot(), rate);
  scatterFeOrderedSolution(history.uDDot(), acceleration);
  const std::array<std::vector<svmp::FE::Real>, 2>
      preserved_history{previous, older};

  auto invalid_preserved_history = preserved_history;
  invalid_preserved_history.back().pop_back();
  EXPECT_THROW(
      stageStaticCapillaryHistoryForPublication(
          system,
          phi,
          certified,
          invalid_preserved_history,
          history),
      std::runtime_error);
  EXPECT_EQ(
      gatherFeOrderedSolution(history.u()),
      current);
  EXPECT_EQ(
      gatherFeOrderedSolution(history.uPrev()),
      previous);
  EXPECT_EQ(
      gatherFeOrderedSolution(history.uPrev2()),
      older);
  EXPECT_EQ(
      gatherFeOrderedSolution(history.uDot()),
      rate);
  EXPECT_EQ(
      gatherFeOrderedSolution(history.uDDot()),
      acceleration);

  ASSERT_NO_THROW(
      stageStaticCapillaryHistoryForPublication(
          system,
          phi,
          certified,
          preserved_history,
          history));

  auto expected_previous = previous;
  auto expected_older = older;
  std::copy(
      certified.begin() +
          static_cast<std::ptrdiff_t>(phi_offset),
      certified.begin() +
          static_cast<std::ptrdiff_t>(
              phi_offset + phi_count),
      expected_previous.begin() +
          static_cast<std::ptrdiff_t>(phi_offset));
  std::copy(
      certified.begin() +
          static_cast<std::ptrdiff_t>(phi_offset),
      certified.begin() +
          static_cast<std::ptrdiff_t>(
              phi_offset + phi_count),
      expected_older.begin() +
          static_cast<std::ptrdiff_t>(phi_offset));
  auto expected_rate = rate;
  auto expected_acceleration = acceleration;
  std::fill(
      expected_rate.begin() +
          static_cast<std::ptrdiff_t>(phi_offset),
      expected_rate.begin() +
          static_cast<std::ptrdiff_t>(
              phi_offset + phi_count),
      0.0);
  std::fill(
      expected_acceleration.begin() +
          static_cast<std::ptrdiff_t>(phi_offset),
      expected_acceleration.begin() +
          static_cast<std::ptrdiff_t>(
              phi_offset + phi_count),
      0.0);

  EXPECT_EQ(
      gatherFeOrderedSolution(history.u()),
      certified);
  EXPECT_EQ(
      gatherFeOrderedSolution(history.uPrev()),
      expected_previous);
  EXPECT_EQ(
      gatherFeOrderedSolution(history.uPrev2()),
      expected_older);
  EXPECT_EQ(
      gatherFeOrderedSolution(history.uDot()),
      expected_rate);
  EXPECT_EQ(
      gatherFeOrderedSolution(history.uDDot()),
      expected_acceleration);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     StaticCapillaryInitializationRollsBackWhenPressureCertificateIsUnavailable)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr int interface_marker = 708;
  auto mesh = makeWorkflowTriangleMesh();
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
      svmp::FE::ElementType::Triangle3,
      /*order=*/1);
  auto velocity_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(
          scalar_space, 2);
  auto system =
      std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(
      svmp::FE::systems::FieldSpec{
          .name = "phi",
          .space = scalar_space,
          .components = 1});
  const auto passive = system->addField(
      svmp::FE::systems::FieldSpec{
          .name = "passive",
          .space = scalar_space,
          .components = 1});
  const auto velocity = system->addField(
      svmp::FE::systems::FieldSpec{
          .name = "synthetic_velocity",
          .space = velocity_space,
          .components = 2});
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
          .geometry_domain_id = "static_capillary_rollback",
          .parameters = functional_parameters,
          .active_volume_energy_parameters =
              svmp::FE::interfaces::
                  FreeSurfaceActiveVolumeEnergyParameters{
                      .liquid_side =
                          svmp::FE::geometry::CutIntegrationSide::Negative,
                      .density = svmp::FE::Real{1.0},
                      .gravitational_acceleration =
                          {{svmp::FE::Real{0.0},
                            svmp::FE::Real{0.0},
                            svmp::FE::Real{0.0}}},
                      .gravitational_reference_point =
                          {{svmp::FE::Real{0.0},
                            svmp::FE::Real{0.0},
                            svmp::FE::Real{0.0}}},
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
              "ApplicationDriverLevelSetWorkflows.StaticCapillaryRollback",
      });
  // The production residual is intentionally empty. The acceptance probe can
  // assemble it, but the physical pressure/surface diagnostic operators are
  // absent and must make the static initializer fail closed.
  system->addOperator("equations");
  ASSERT_NO_THROW(system->setup({}));

  const std::vector<svmp::FE::Real> phi_vertex_values{
      -0.25, 0.75, -0.25};
  const auto phi_coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      phi_vertex_values,
      /*components=*/1u,
      "ApplicationDriver static capillary rollback phi");
  std::vector<svmp::FE::Real> passive_coefficients(
      static_cast<std::size_t>(
          system->fieldDofHandler(passive).getNumDofs()),
      0.0);
  for (std::size_t i = 0u;
       i < passive_coefficients.size();
       ++i) {
    passive_coefficients[i] =
        10.0 + static_cast<svmp::FE::Real>(i);
  }
  std::vector<svmp::FE::Real> current(
      static_cast<std::size_t>(
          system->dofHandler().getNumDofs()),
      0.0);
  writeWorkflowFieldSlice(
      *system, phi, phi_coefficients, current);
  writeWorkflowFieldSlice(
      *system, passive, passive_coefficients, current);
  auto previous = current;
  auto older = current;
  const auto passive_offset = static_cast<std::size_t>(
      system->fieldDofOffset(passive));
  for (std::size_t i = 0u;
       i < passive_coefficients.size();
       ++i) {
    previous[passive_offset + i] += 20.0;
    older[passive_offset + i] += 40.0;
  }

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);
  try {
    sim.backend =
        svmp::FE::backends::BackendFactory::create(
            svmp::FE::backends::BackendKind::Eigen);
  } catch (const std::exception&) {
    GTEST_SKIP() << "Requires the Eigen FE backend.";
  }
  ASSERT_NE(sim.backend, nullptr);
  svmp::FE::backends::SolverOptions linear_options;
  linear_options.method =
      svmp::FE::backends::SolverMethod::GMRES;
  linear_options.preconditioner =
      svmp::FE::backends::PreconditionerType::Diagonal;
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
  sim.time_history->setTime(0.2);
  sim.time_history->setDt(0.1);
  sim.time_history->setPrevDt(0.1);
  scatterFeOrderedSolution(sim.time_history->u(), current);
  scatterFeOrderedSolution(
      sim.time_history->uPrev(), previous);
  scatterFeOrderedSolution(
      sim.time_history->uPrev2(), older);
  std::vector<svmp::FE::Real> rate(
      current.size(), svmp::FE::Real{3.0});
  std::vector<svmp::FE::Real> acceleration(
      current.size(), svmp::FE::Real{4.0});
  scatterFeOrderedSolution(
      sim.time_history->uDot(), rate);
  scatterFeOrderedSolution(
      sim.time_history->uDDot(), acceleration);

  auto params = parseWorkflowParametersXml(R"xml(
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
      <Generated_interface_domain_id>static_capillary_rollback</Generated_interface_domain_id>
      <Interface_marker>708</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
  auto requests = levelSetMaintenanceRequests(*params);
  ASSERT_EQ(requests.size(), 1u);
  ASSERT_TRUE(
      requests.front().static_capillary_equilibrium_enabled);
  ASSERT_TRUE(requests.front().volume_cut_request.has_value());

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
          "application-driver-static-capillary-rollback-initial");
  ASSERT_TRUE(initial_report.refreshed);
  ASSERT_NE(initial_report.topology_key, 0u);
  const auto* initial_context =
      sim.fe_system->cutIntegrationContext();
  ASSERT_NE(initial_context, nullptr);
  const auto lifecycle_revision = lifecycle.valueRevision();
  const auto refresh_cache_before = refresh_cache;
  const auto* mesh_phi_data_before =
      static_cast<const double*>(
          mesh->field_data(mesh_field));
  ASSERT_NE(mesh_phi_data_before, nullptr);
  const auto mesh_phi_count =
      mesh->field_components(mesh_field) *
      mesh->field_entity_count(mesh_field);
  const std::vector<double> mesh_phi_before(
      mesh_phi_data_before,
      mesh_phi_data_before + mesh_phi_count);

  // Exercise the incomplete body-force preflight on the complete application
  // transaction, then restore this synthetic declaration so the original
  // diagnostic rollback case below retains its exact purpose.
  auto declarations =
      sim.fe_system->freeSurfaceDiscreteFunctionalDeclarations();
  ASSERT_EQ(declarations.size(), 1u);
  auto& mutable_declaration = const_cast<
      svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration&>(
      declarations.front());
  const bool saved_body_force_contract =
      mutable_declaration.static_conservative_body_force_complete;
  ASSERT_TRUE(saved_body_force_contract);
  mutable_declaration.static_conservative_body_force_complete = false;
  const auto target_volume_before_body_force_rejection =
      requests.front().static_capillary_equilibrium.target_liquid_volume;
  const auto functional_history_size_before_body_force_rejection =
      sim.fe_system->freeSurfaceDiscreteFunctionalHistory().size();
  std::string body_force_failure;
  try {
    (void)initializeDiscreteStaticCapillaryEquilibrium(
        sim,
        *params,
        requests,
        lifecycle,
        refresh_cache);
  } catch (const std::runtime_error& error) {
    body_force_failure = error.what();
  }
  EXPECT_NE(
      body_force_failure.find(
          "requires an active-volume energy declaration whose gravitational acceleration is the complete velocity-independent body acceleration"),
      std::string::npos)
      << body_force_failure;
  EXPECT_FALSE(
      requests.front().static_capillary_equilibrium_initialized);
  EXPECT_DOUBLE_EQ(
      requests.front().static_capillary_equilibrium.target_liquid_volume,
      target_volume_before_body_force_rejection);
  EXPECT_EQ(
      sim.fe_system->freeSurfaceDiscreteFunctionalHistory().size(),
      functional_history_size_before_body_force_rejection);
  EXPECT_EQ(gatherFeOrderedSolution(sim.time_history->u()), current);
  EXPECT_EQ(gatherFeOrderedSolution(sim.time_history->uPrev()), previous);
  EXPECT_EQ(gatherFeOrderedSolution(sim.time_history->uPrev2()), older);
  EXPECT_EQ(gatherFeOrderedSolution(sim.time_history->uDot()), rate);
  EXPECT_EQ(gatherFeOrderedSolution(sim.time_history->uDDot()), acceleration);
  EXPECT_EQ(sim.fe_system->cutIntegrationContext(), initial_context);
  EXPECT_EQ(lifecycle.valueRevision(), lifecycle_revision);
  EXPECT_EQ(refresh_cache.topology_key, refresh_cache_before.topology_key);
  const auto* mesh_phi_data_after_body_force_rejection =
      static_cast<const double*>(mesh->field_data(mesh_field));
  ASSERT_NE(mesh_phi_data_after_body_force_rejection, nullptr);
  EXPECT_EQ(
      std::vector<double>(
          mesh_phi_data_after_body_force_rejection,
          mesh_phi_data_after_body_force_rejection + mesh_phi_count),
      mesh_phi_before);
  mutable_declaration.static_conservative_body_force_complete =
      saved_body_force_contract;

  const auto saved_velocity_field = mutable_declaration.velocity_field;
  const auto expect_velocity_binding_rejection =
      [&](svmp::FE::FieldId candidate,
          std::string_view expected_diagnostic) {
        mutable_declaration.velocity_field = candidate;
        std::string binding_failure;
        try {
          (void)initializeDiscreteStaticCapillaryEquilibrium(
              sim,
              *params,
              requests,
              lifecycle,
              refresh_cache);
        } catch (const std::runtime_error& error) {
          binding_failure = error.what();
        }
        EXPECT_NE(
            binding_failure.find(expected_diagnostic),
            std::string::npos)
            << binding_failure;
        EXPECT_FALSE(
            requests.front().static_capillary_equilibrium_initialized);
        EXPECT_DOUBLE_EQ(
            requests.front()
                .static_capillary_equilibrium
                .target_liquid_volume,
            target_volume_before_body_force_rejection);
        EXPECT_EQ(
            sim.fe_system->freeSurfaceDiscreteFunctionalHistory().size(),
            functional_history_size_before_body_force_rejection);
        EXPECT_EQ(gatherFeOrderedSolution(sim.time_history->u()), current);
        EXPECT_EQ(
            gatherFeOrderedSolution(sim.time_history->uPrev()), previous);
        EXPECT_EQ(
            gatherFeOrderedSolution(sim.time_history->uPrev2()), older);
        EXPECT_EQ(gatherFeOrderedSolution(sim.time_history->uDot()), rate);
        EXPECT_EQ(
            gatherFeOrderedSolution(sim.time_history->uDDot()),
            acceleration);
        EXPECT_EQ(sim.fe_system->cutIntegrationContext(), initial_context);
        EXPECT_EQ(lifecycle.valueRevision(), lifecycle_revision);
        EXPECT_EQ(
            refresh_cache.topology_key,
            refresh_cache_before.topology_key);
        const auto* mesh_phi_data_after =
            static_cast<const double*>(mesh->field_data(mesh_field));
        EXPECT_NE(mesh_phi_data_after, nullptr);
        if (mesh_phi_data_after != nullptr) {
          EXPECT_EQ(
              std::vector<double>(
                  mesh_phi_data_after,
                  mesh_phi_data_after + mesh_phi_count),
              mesh_phi_before);
        }
      };
  expect_velocity_binding_rejection(
      svmp::FE::INVALID_FIELD_ID,
      "requires a declared velocity field");
  expect_velocity_binding_rejection(
      phi,
      "registered, finalized, dimension-compatible unknown vector volume field");
  mutable_declaration.velocity_field = saved_velocity_field;

  std::string failure;
  try {
    (void)initializeDiscreteStaticCapillaryEquilibrium(
        sim,
        *params,
        requests,
        lifecycle,
        refresh_cache);
  } catch (const std::runtime_error& error) {
    failure = error.what();
  }
  EXPECT_NE(
      failure.find(
          "pressure_representability_unavailable_at_parameter_stationary_geometry"),
      std::string::npos)
      << failure;
  EXPECT_FALSE(
      requests.front().static_capillary_equilibrium_initialized);
  EXPECT_EQ(
      gatherFeOrderedSolution(sim.time_history->u()),
      current);
  EXPECT_EQ(
      gatherFeOrderedSolution(sim.time_history->uPrev()),
      previous);
  EXPECT_EQ(
      gatherFeOrderedSolution(sim.time_history->uPrev2()),
      older);
  EXPECT_EQ(
      gatherFeOrderedSolution(sim.time_history->uDot()),
      rate);
  EXPECT_EQ(
      gatherFeOrderedSolution(sim.time_history->uDDot()),
      acceleration);
  EXPECT_EQ(
      sim.fe_system->cutIntegrationContext(),
      initial_context);
  EXPECT_EQ(lifecycle.valueRevision(), lifecycle_revision);
  EXPECT_FALSE(
      sim.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle.transactionActive());
  ASSERT_EQ(
      refresh_cache.last_signature.has_value(),
      refresh_cache_before.last_signature.has_value());
  if (refresh_cache.last_signature.has_value()) {
    EXPECT_TRUE(
        *refresh_cache.last_signature ==
        *refresh_cache_before.last_signature);
  }
  ASSERT_EQ(
      refresh_cache.last_vector_signature.has_value(),
      refresh_cache_before.last_vector_signature.has_value());
  if (refresh_cache.last_vector_signature.has_value()) {
    EXPECT_TRUE(
        *refresh_cache.last_vector_signature ==
        *refresh_cache_before.last_vector_signature);
  }
  EXPECT_EQ(
      refresh_cache.topology_key,
      refresh_cache_before.topology_key);
  const auto* mesh_phi_data_after =
      static_cast<const double*>(
          mesh->field_data(mesh_field));
  ASSERT_NE(mesh_phi_data_after, nullptr);
  EXPECT_EQ(
      std::vector<double>(
          mesh_phi_data_after,
          mesh_phi_data_after + mesh_phi_count),
      mesh_phi_before);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     StaticCapillaryInitializationPublishesAfterExactSyntheticCertificate)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr int interface_marker = 709;
  auto mesh = makeWorkflowTriangleMesh();
  const auto mesh_field = svmp::MeshFields::attach_field(
      mesh->local_mesh(),
      svmp::EntityKind::Vertex,
      "phi",
      svmp::FieldScalarType::Float64,
      1);

  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Triangle3,
      /*order=*/1);
  auto velocity_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(
          scalar_space, 2);
  auto system =
      std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(
      svmp::FE::systems::FieldSpec{
          .name = "phi",
          .space = scalar_space,
          .components = 1});
  const auto passive = system->addField(
      svmp::FE::systems::FieldSpec{
          .name = "passive",
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
          .geometry_domain_id = "static_capillary_publication",
          .parameters = functional_parameters,
          .active_volume_energy_parameters =
              svmp::FE::interfaces::
                  FreeSurfaceActiveVolumeEnergyParameters{
                      .liquid_side =
                          svmp::FE::geometry::CutIntegrationSide::Negative,
                      .density = svmp::FE::Real{1.0},
                      .gravitational_acceleration =
                          {{svmp::FE::Real{0.0},
                            svmp::FE::Real{0.0},
                            svmp::FE::Real{0.0}}},
                      .gravitational_reference_point =
                          {{svmp::FE::Real{0.0},
                            svmp::FE::Real{0.0},
                            svmp::FE::Real{0.0}}},
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
              "ApplicationDriverLevelSetWorkflows.StaticCapillaryPublication",
      });
  system->addOperator("equations");
  // This compact mass-pair fixture certifies application transaction
  // mechanics only. It is not physical static-cap qualification evidence.
  installWorkflowExactConstantPressureCertificate(
      *system, velocity, pressure);
  ASSERT_NO_THROW(system->setup({}));

  const std::vector<svmp::FE::Real> phi_vertex_values{
      -0.25, 0.75, -0.25};
  const auto phi_coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      phi_vertex_values,
      /*components=*/1u,
      "ApplicationDriver static capillary publication phi");
  std::vector<svmp::FE::Real> current(
      static_cast<std::size_t>(
          system->dofHandler().getNumDofs()),
      0.0);
  writeWorkflowFieldSlice(
      *system, phi, phi_coefficients, current);
  const auto fill_field =
      [&](svmp::FE::FieldId field,
          svmp::FE::Real base,
          std::vector<svmp::FE::Real>& values) {
        const auto offset = static_cast<std::size_t>(
            system->fieldDofOffset(field));
        const auto count = static_cast<std::size_t>(
            system->fieldDofHandler(field).getNumDofs());
        for (std::size_t i = 0u; i < count; ++i) {
          values[offset + i] =
              base + static_cast<svmp::FE::Real>(i);
        }
      };
  fill_field(passive, 10.0, current);
  fill_field(velocity, 20.0, current);

  auto previous = current;
  auto older = current;
  const auto add_to_field =
      [&](svmp::FE::FieldId field,
          svmp::FE::Real previous_increment,
          svmp::FE::Real older_increment) {
        const auto offset = static_cast<std::size_t>(
            system->fieldDofOffset(field));
        const auto count = static_cast<std::size_t>(
            system->fieldDofHandler(field).getNumDofs());
        for (std::size_t i = 0u; i < count; ++i) {
          previous[offset + i] += previous_increment;
          older[offset + i] += older_increment;
        }
      };
  add_to_field(phi, 1.0, 2.0);
  add_to_field(passive, 3.0, 4.0);
  add_to_field(velocity, 5.0, 6.0);
  add_to_field(pressure, 7.0, 8.0);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);
  try {
    sim.backend =
        svmp::FE::backends::BackendFactory::create(
            svmp::FE::backends::BackendKind::Eigen);
  } catch (const std::exception&) {
    GTEST_SKIP() << "Requires the Eigen FE backend.";
  }
  ASSERT_NE(sim.backend, nullptr);
  svmp::FE::backends::SolverOptions linear_options;
  linear_options.method =
      svmp::FE::backends::SolverMethod::GMRES;
  linear_options.preconditioner =
      svmp::FE::backends::PreconditionerType::Diagonal;
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
  sim.time_history->setTime(0.2);
  sim.time_history->setDt(0.1);
  sim.time_history->setPrevDt(0.1);
  scatterFeOrderedSolution(sim.time_history->u(), current);
  scatterFeOrderedSolution(
      sim.time_history->uPrev(), previous);
  scatterFeOrderedSolution(
      sim.time_history->uPrev2(), older);
  std::vector<svmp::FE::Real> rate(current.size(), 0.0);
  std::vector<svmp::FE::Real> acceleration(current.size(), 0.0);
  for (std::size_t i = 0u; i < current.size(); ++i) {
    rate[i] = 40.0 + static_cast<svmp::FE::Real>(i);
    acceleration[i] =
        50.0 + static_cast<svmp::FE::Real>(i);
  }
  scatterFeOrderedSolution(
      sim.time_history->uDot(), rate);
  scatterFeOrderedSolution(
      sim.time_history->uDDot(), acceleration);

  auto params = parseWorkflowParametersXml(R"xml(
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
      <Generated_interface_domain_id>static_capillary_publication</Generated_interface_domain_id>
      <Interface_marker>709</Interface_marker>
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
          "application-driver-static-capillary-publication-initial");
  ASSERT_TRUE(initial_report.refreshed);
  ASSERT_NE(initial_report.topology_key, 0u);
  const auto* mesh_phi_data_before =
      static_cast<const double*>(
          mesh->field_data(mesh_field));
  ASSERT_NE(mesh_phi_data_before, nullptr);
  const auto mesh_phi_count =
      mesh->field_components(mesh_field) *
      mesh->field_entity_count(mesh_field);
  const std::vector<double> mesh_phi_before(
      mesh_phi_data_before,
      mesh_phi_data_before + mesh_phi_count);

  const auto* context_before_velocity_rejection =
      sim.fe_system->cutIntegrationContext();
  ASSERT_NE(context_before_velocity_rejection, nullptr);
  const auto lifecycle_revision_before_velocity_rejection =
      lifecycle.valueRevision();
  const auto topology_key_before_velocity_rejection =
      refresh_cache.topology_key;
  const auto target_volume_before_velocity_rejection =
      requests.front().static_capillary_equilibrium.target_liquid_volume;
  const auto functional_history_size_before_velocity_rejection =
      sim.fe_system->freeSurfaceDiscreteFunctionalHistory().size();
  std::string velocity_failure;
  try {
    (void)initializeDiscreteStaticCapillaryEquilibrium(
        sim,
        *params,
        requests,
        lifecycle,
        refresh_cache);
  } catch (const std::runtime_error& error) {
    velocity_failure = error.what();
  }
  EXPECT_NE(
      velocity_failure.find(
          "requires an exactly zero finite current velocity field"),
      std::string::npos)
      << velocity_failure;
  EXPECT_FALSE(
      requests.front().static_capillary_equilibrium_initialized);
  EXPECT_DOUBLE_EQ(
      requests.front().static_capillary_equilibrium.target_liquid_volume,
      target_volume_before_velocity_rejection);
  EXPECT_EQ(
      sim.fe_system->freeSurfaceDiscreteFunctionalHistory().size(),
      functional_history_size_before_velocity_rejection);
  EXPECT_EQ(gatherFeOrderedSolution(sim.time_history->u()), current);
  EXPECT_EQ(gatherFeOrderedSolution(sim.time_history->uPrev()), previous);
  EXPECT_EQ(gatherFeOrderedSolution(sim.time_history->uPrev2()), older);
  EXPECT_EQ(gatherFeOrderedSolution(sim.time_history->uDot()), rate);
  EXPECT_EQ(gatherFeOrderedSolution(sim.time_history->uDDot()), acceleration);
  EXPECT_EQ(
      sim.fe_system->cutIntegrationContext(),
      context_before_velocity_rejection);
  EXPECT_EQ(
      lifecycle.valueRevision(),
      lifecycle_revision_before_velocity_rejection);
  EXPECT_EQ(
      refresh_cache.topology_key,
      topology_key_before_velocity_rejection);
  const auto* mesh_phi_data_after_velocity_rejection =
      static_cast<const double*>(mesh->field_data(mesh_field));
  ASSERT_NE(mesh_phi_data_after_velocity_rejection, nullptr);
  EXPECT_EQ(
      std::vector<double>(
          mesh_phi_data_after_velocity_rejection,
          mesh_phi_data_after_velocity_rejection + mesh_phi_count),
      mesh_phi_before);

  // The publication fixture now enters its successful path from an exactly
  // static current velocity state, matching the production preflight contract.
  const auto velocity_offset = static_cast<std::size_t>(
      sim.fe_system->fieldDofOffset(velocity));
  const auto velocity_count = static_cast<std::size_t>(
      sim.fe_system->fieldDofHandler(velocity).getNumDofs());
  std::fill(
      current.begin() + static_cast<std::ptrdiff_t>(velocity_offset),
      current.begin() + static_cast<std::ptrdiff_t>(
                            velocity_offset + velocity_count),
      svmp::FE::Real{0.0});
  scatterFeOrderedSolution(sim.time_history->u(), current);
  sim.time_history->updateGhosts();

  const auto expected_pressure_certificate =
      evaluateStaticCapillaryPressureCertificate(
          sim,
          current,
          requests.front().static_capillary_equilibrium,
          /*initialize_compatible_pressure=*/true);
  ASSERT_TRUE(
      expected_pressure_certificate.report
          .static_compatible_pressure_initializer_passed)
      << expected_pressure_certificate.report
             .static_compatible_pressure_initializer_reason;
  const auto& expected_current =
      expected_pressure_certificate.certified_solution;
  ASSERT_EQ(expected_current.size(), current.size());
  const auto pressure_offset = static_cast<std::size_t>(
      sim.fe_system->fieldDofOffset(pressure));
  const auto pressure_count = static_cast<std::size_t>(
      sim.fe_system->fieldDofHandler(pressure).getNumDofs());
  svmp::FE::Real maximum_pressure_update = 0.0;
  for (std::size_t i = 0u; i < pressure_count; ++i) {
    maximum_pressure_update =
        std::max(maximum_pressure_update,
                 std::abs(expected_current[pressure_offset + i] -
                          current[pressure_offset + i]));
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
  EXPECT_GT(
      requests.front()
          .static_capillary_equilibrium
          .target_liquid_volume,
      0.0);
  EXPECT_FALSE(
      sim.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle.transactionActive());
  EXPECT_NE(lifecycle.valueRevision(), 0u);
  ASSERT_NE(sim.fe_system->cutIntegrationContext(), nullptr);
  ASSERT_TRUE(refresh_cache.topology_key.has_value());
  EXPECT_NE(*refresh_cache.topology_key, 0u);

  auto expected_previous = previous;
  auto expected_older = older;
  const auto phi_offset = static_cast<std::size_t>(
      sim.fe_system->fieldDofOffset(phi));
  const auto phi_count = static_cast<std::size_t>(
      sim.fe_system->fieldDofHandler(phi).getNumDofs());
  std::copy(
      expected_current.begin() +
          static_cast<std::ptrdiff_t>(phi_offset),
      expected_current.begin() +
          static_cast<std::ptrdiff_t>(
              phi_offset + phi_count),
      expected_previous.begin() +
          static_cast<std::ptrdiff_t>(phi_offset));
  std::copy(
      expected_current.begin() +
          static_cast<std::ptrdiff_t>(phi_offset),
      expected_current.begin() +
          static_cast<std::ptrdiff_t>(
              phi_offset + phi_count),
      expected_older.begin() +
          static_cast<std::ptrdiff_t>(phi_offset));
  auto expected_rate = rate;
  auto expected_acceleration = acceleration;
  std::fill(
      expected_rate.begin() +
          static_cast<std::ptrdiff_t>(phi_offset),
      expected_rate.begin() +
          static_cast<std::ptrdiff_t>(
              phi_offset + phi_count),
      0.0);
  std::fill(
      expected_acceleration.begin() +
          static_cast<std::ptrdiff_t>(phi_offset),
      expected_acceleration.begin() +
          static_cast<std::ptrdiff_t>(
              phi_offset + phi_count),
      0.0);

  EXPECT_EQ(
      gatherFeOrderedSolution(sim.time_history->u()),
      expected_current);
  EXPECT_EQ(
      gatherFeOrderedSolution(sim.time_history->uPrev()),
      expected_previous);
  EXPECT_EQ(
      gatherFeOrderedSolution(sim.time_history->uPrev2()),
      expected_older);
  EXPECT_EQ(
      gatherFeOrderedSolution(sim.time_history->uDot()),
      expected_rate);
  EXPECT_EQ(
      gatherFeOrderedSolution(sim.time_history->uDDot()),
      expected_acceleration);
  const auto* mesh_phi_data_after =
      static_cast<const double*>(
          mesh->field_data(mesh_field));
  ASSERT_NE(mesh_phi_data_after, nullptr);
  EXPECT_EQ(
      std::vector<double>(
          mesh_phi_data_after,
          mesh_phi_data_after + mesh_phi_count),
      mesh_phi_before);

  EXPECT_FALSE(
      initializeDiscreteStaticCapillaryEquilibrium(
          sim,
          *params,
          requests,
          lifecycle,
          refresh_cache));
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     StaticCapillaryInitializationAcceptsPhysicalFlatSurfaceStressEquilibrium)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr int interface_marker = 710;
  constexpr int left_wall_marker = 7101;
  constexpr int right_wall_marker = 7102;
  constexpr int lower_anchor_marker = 7103;
  constexpr int upper_anchor_marker = 7104;
  constexpr svmp::FE::Real pi =
      svmp::FE::Real{3.141592653589793238462643383279502884};
  constexpr svmp::FE::Real contact_angle = pi / svmp::FE::Real{2.0};
  WorkflowScopedEnvVar conservative_balance_diagnostic(
      "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", std::string("1"));

  constexpr std::array<svmp::FE::Real, 3> normal_offsets{
      svmp::FE::Real{0.35},
      svmp::FE::Real{0.5},
      svmp::FE::Real{0.65},
  };
  std::size_t case_count = 0u;
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
                     << "normal_axis=" << normal_axis << " active_side="
                     << (positive_side ? "positive" : "negative")
                     << " normal_offset=" << normal_offset);
        ++case_count;

        auto mesh = makeWorkflowFlatCapillaryFanMesh(normal_axis);
        auto& local_mesh = mesh->local_mesh();
        std::array<std::size_t, 4> marker_counts{};
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
            on_first_wall = on_first_wall && std::abs(point[tangent_axis]) <=
                                                 coordinate_tolerance;
            on_second_wall = on_second_wall && std::abs(point[tangent_axis] -
                                                        svmp::FE::Real{3.0}) <=
                                                   coordinate_tolerance;
            on_lower_anchor = on_lower_anchor && std::abs(point[normal_axis]) <=
                                                     coordinate_tolerance;
            on_upper_anchor =
                on_upper_anchor &&
                std::abs(point[normal_axis] - svmp::FE::Real{1.0}) <=
                    coordinate_tolerance;
          }
          if (on_first_wall) {
            mesh->set_boundary_label(face, left_wall_marker);
            ++marker_counts[0];
          } else if (on_second_wall) {
            mesh->set_boundary_label(face, right_wall_marker);
            ++marker_counts[1];
          } else if (on_lower_anchor) {
            mesh->set_boundary_label(face, lower_anchor_marker);
            ++marker_counts[2];
          } else if (on_upper_anchor) {
            mesh->set_boundary_label(face, upper_anchor_marker);
            ++marker_counts[3];
          } else {
            FAIL()
                << "Flat static-capillary fixture found an unclassified face.";
          }
        }
        EXPECT_EQ(marker_counts[0], 1u);
        EXPECT_EQ(marker_counts[1], 1u);
        EXPECT_EQ(marker_counts[2], 1u);
        EXPECT_EQ(marker_counts[3], 1u);

        const auto mesh_field =
            svmp::MeshFields::attach_field(local_mesh,
                                           svmp::EntityKind::Vertex,
                                           "phi_physical_flat_static",
                                           svmp::FieldScalarType::Float64,
                                           1);
        ASSERT_NE(svmp::MeshFields::field_data_as<svmp::real_t>(local_mesh,
                                                                mesh_field),
                  nullptr);

        auto scalar_space = svmp::FE::spaces::SpaceFactory::create_h1(
            svmp::FE::ElementType::Triangle3,
            /*order=*/1);
        auto velocity_space = svmp::FE::spaces::SpaceFactory::create_vector_h1(
            svmp::FE::ElementType::Triangle3,
            /*order=*/1,
            /*components=*/2);
        auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
        const auto phi = system->addField(
            svmp::FE::systems::FieldSpec{.name = "phi_physical_flat_static",
                                         .space = scalar_space,
                                         .components = 1});

        channel_ns::IncompressibleNavierStokesVMSOptions options;
        options.velocity_field_name = "u_physical_flat_static";
        options.pressure_field_name = "p_physical_flat_static";
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
                    .boundary_marker = left_wall_marker,
                    .value = {0.0, 0.0, 0.0},
                    .active_components = {tangent_axis == 0,
                                          tangent_axis == 1,
                                          false},
                });
        options.velocity_dirichlet.push_back(
            channel_ns::IncompressibleNavierStokesVMSOptions::
                VelocityDirichletBC{
                    .boundary_marker = right_wall_marker,
                    .value = {0.0, 0.0, 0.0},
                    .active_components = {tangent_axis == 0,
                                          tangent_axis == 1,
                                          false},
                });

        using ContactLine = channel_ns::IncompressibleNavierStokesVMSOptions::
            FreeSurfaceContactLine;
        auto free_surface = channel_ns::IncompressibleNavierStokesVMSOptions::
            FreeSurfaceBoundary{
                .implementation =
                    channel_ns::FreeSurfaceImplementation::UnfittedLevelSet,
                .interface_marker = interface_marker,
                .level_set_field_name = "phi_physical_flat_static",
                .generated_interface_domain_id =
                    "physical_flat_static_capillary",
                .generated_interface_geometry = "LinearCorner",
                .active_domain =
                    positive_side
                        ? channel_ns::FreeSurfaceActiveDomain::LevelSetPositive
                        : channel_ns::FreeSurfaceActiveDomain::LevelSetNegative,
                .active_domain_method =
                    channel_ns::FreeSurfaceActiveDomainMethod::CutVolume,
                .external_pressure = 0.0,
                .surface_tension = 1.0,
                .surface_tension_form =
                    channel_ns::FreeSurfaceSurfaceTensionForm::SurfaceStress,
                .curvature = 0.0,
                .use_level_set_curvature = false,
                .small_cut_aggregation = false,
            };
        free_surface.contact_lines.push_back(
            ContactLine{.configuration = ContactLine::DynamicRenE{
                            .wall_boundary_marker = left_wall_marker,
                            .contact_line_marker = -1,
                            .equilibrium_contact_angle_radians = contact_angle,
                            .wall_normal = {tangent_axis == 0 ? -1.0 : 0.0,
                                            tangent_axis == 1 ? -1.0 : 0.0,
                                            0.0},
                            .mobility = 1.0,
                            .slip_length = 1.0,
                        }});
        free_surface.contact_lines.push_back(
            ContactLine{.configuration = ContactLine::DynamicRenE{
                            .wall_boundary_marker = right_wall_marker,
                            .contact_line_marker = -1,
                            .equilibrium_contact_angle_radians = contact_angle,
                            .wall_normal = {tangent_axis == 0 ? 1.0 : 0.0,
                                            tangent_axis == 1 ? 1.0 : 0.0,
                                            0.0},
                            .mobility = 1.0,
                            .slip_length = 1.0,
                        }});
        options.free_surface.push_back(std::move(free_surface));

        channel_ns::IncompressibleNavierStokesVMSModule module(
            velocity_space, scalar_space, std::move(options));
        module.registerOn(*system);
        ASSERT_NO_THROW(system->setup({}));
        const auto velocity = system->findFieldByName("u_physical_flat_static");
        const auto pressure = system->findFieldByName("p_physical_flat_static");
        ASSERT_NE(velocity, svmp::FE::INVALID_FIELD_ID);
        ASSERT_NE(pressure, svmp::FE::INVALID_FIELD_ID);

        std::vector<svmp::FE::Real> phi_vertex_values(mesh->n_vertices(), 0.0);
        for (std::size_t vertex = 0u; vertex < mesh->n_vertices(); ++vertex) {
          const auto signed_coordinate =
              workflowVertexPoint(*mesh, vertex)[normal_axis] - normal_offset;
          phi_vertex_values[vertex] =
              positive_side ? -signed_coordinate : signed_coordinate;
        }
        const auto phi_coefficients = projectWorkflowVertexValues(
            *system,
            phi,
            phi_vertex_values,
            /*components=*/1u,
            "ApplicationDriver physical flat static-capillary phi");
        std::vector<svmp::FE::Real> current(
            static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
        writeWorkflowFieldSlice(*system, phi, phi_coefficients, current);

        application::core::SimulationComponents sim;
        sim.primary_mesh = mesh;
        sim.fe_system = std::move(system);
        try {
          sim.backend = svmp::FE::backends::BackendFactory::create(
              svmp::FE::backends::BackendKind::FSILS);
        } catch (const std::exception&) {
          GTEST_SKIP() << "Requires an available FE vector backend.";
        }
        ASSERT_NE(sim.backend, nullptr);
        svmp::FE::backends::SolverOptions linear_options;
        linear_options.method = svmp::FE::backends::SolverMethod::GMRES;
        linear_options.preconditioner =
            svmp::FE::backends::PreconditionerType::Diagonal;
        sim.linear_solver = sim.backend->createLinearSolver(linear_options);
        ASSERT_NE(sim.linear_solver, nullptr);
        auto allocated_history = svmp::FE::timestepping::TimeHistory::allocate(
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

        const char* active_domain_name =
            positive_side ? "LevelSetPositive" : "LevelSetNegative";
        const char* contact_wall_normals = tangent_axis == 0
                                               ? "-1.0 0.0 0.0; 1.0 0.0 0.0"
                                               : "0.0 -1.0 0.0; 0.0 1.0 0.0";
        const std::string parameter_xml = std::string(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Level_set_field_name>phi_physical_flat_static</Level_set_field_name>
    <Enable_static_capillary_equilibrium_initialization>true</Enable_static_capillary_equilibrium_initialization>
    <Static_capillary_volume_tolerance>1.0e-11</Static_capillary_volume_tolerance>
    <Static_capillary_projected_gradient_tolerance>2.0e-6</Static_capillary_projected_gradient_tolerance>
    <Static_capillary_constant_pressure_kkt_max_residual_norm>2.0e-10</Static_capillary_constant_pressure_kkt_max_residual_norm>
    <Static_capillary_constant_pressure_kkt_max_relative_distance>2.0e-10</Static_capillary_constant_pressure_kkt_max_relative_distance>
  </Add_equation>
  <Add_equation type="fluid">
    <Add_BC name="physical_flat_static_capillary">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi_physical_flat_static</Level_set_field_name>
      <Generated_interface_domain_id>physical_flat_static_capillary</Generated_interface_domain_id>
      <Interface_marker>710</Interface_marker>
      <Generated_interface_geometry>LinearCorner</Generated_interface_geometry>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>)xml") + active_domain_name +
                                          R"xml(</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
      <Small_cut_aggregation>false</Small_cut_aggregation>
      <Surface_tension>1.0</Surface_tension>
      <Surface_tension_form>SurfaceStress</Surface_tension_form>
      <Contact_line_model>DynamicContactAngle</Contact_line_model>
      <Contact_angle_degrees>90.0</Contact_angle_degrees>
      <Contact_line_wall_markers>7101;7102</Contact_line_wall_markers>
      <Contact_line_wall_normals>)xml" + contact_wall_normals +
                                          R"xml(</Contact_line_wall_normals>
      <Contact_line_mobility>1.0</Contact_line_mobility>
      <Wall_slip_model>Navier</Wall_slip_model>
      <Wall_slip_length>1.0</Wall_slip_length>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml";
        auto params = parseWorkflowParametersXml(parameter_xml.c_str());
        auto requests = levelSetMaintenanceRequests(*params);
        ASSERT_EQ(requests.size(), 1u);
        ASSERT_TRUE(requests.front().static_capillary_equilibrium_enabled);

        svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
        ActiveCutContextRefreshCache refresh_cache;
        const auto initial_report = refreshActiveCutIntegrationContextCached(
            sim,
            *params,
            sim.time_history->u(),
            lifecycle,
            refresh_cache,
            "application-driver-physical-flat-static-initial");
        ASSERT_TRUE(initial_report.refreshed);
        const auto initial_functionals =
            evaluateCurrentFreeSurfaceDiscreteFunctionals(sim);
        ASSERT_EQ(initial_functionals.size(), 1u);
        const auto expected_volume = svmp::FE::Real{3.0} * normal_offset;
        EXPECT_NEAR(initial_functionals.front().state.owned_liquid_volume,
                    expected_volume,
                    1.0e-13);
        EXPECT_NEAR(initial_functionals.front().state.liquid_gas_surface_energy,
                    svmp::FE::Real{3.0},
                    1.0e-13);
        EXPECT_NEAR(initial_functionals.front().state.young_wall_energy,
                    svmp::FE::Real{0.0},
                    1.0e-13);

        bool initialized = false;
        ASSERT_NO_THROW(
            initialized = initializeDiscreteStaticCapillaryEquilibrium(
                sim, *params, requests, lifecycle, refresh_cache));
        ASSERT_TRUE(initialized);
        ASSERT_TRUE(requests.front().static_capillary_equilibrium_initialized);

        const auto certified_solution =
            gatherFeOrderedSolution(sim.time_history->u());
        const auto pressure_certificate =
            evaluateStaticCapillaryPressureCertificate(
                sim,
                certified_solution,
                requests.front().static_capillary_equilibrium,
                /*initialize_compatible_pressure=*/false);
        const auto& certificate = pressure_certificate.report;
        ASSERT_TRUE(certificate.pressure_representability_diagnostic_sampled);
        ASSERT_TRUE(certificate.constant_pressure_kkt_available)
            << certificate.constant_pressure_kkt_reason;
        EXPECT_LE(certificate.constant_pressure_kkt_residual_norm, 2.0e-10);
        EXPECT_LE(certificate.constant_pressure_kkt_relative_distance, 2.0e-10);
        EXPECT_NEAR(
            certificate.constant_pressure_kkt_pressure_jump, 0.0, 2.0e-10);

        const auto final_functionals =
            evaluateCurrentFreeSurfaceDiscreteFunctionals(sim);
        ASSERT_EQ(final_functionals.size(), 1u);
        const auto volume_error =
            std::abs(final_functionals.front().state.owned_liquid_volume -
                     expected_volume);
        const auto surface_energy_error =
            std::abs(final_functionals.front().state.liquid_gas_surface_energy -
                     svmp::FE::Real{3.0});
        EXPECT_LE(volume_error, 1.0e-11);
        EXPECT_LE(surface_energy_error, 2.0e-10);
        EXPECT_NEAR(final_functionals.front().state.young_wall_energy,
                    svmp::FE::Real{0.0},
                    1.0e-13);

        const auto phi_offset =
            static_cast<std::size_t>(sim.fe_system->fieldDofOffset(phi));
        const auto phi_count = static_cast<std::size_t>(
            sim.fe_system->fieldDofHandler(phi).getNumDofs());
        svmp::FE::Real maximum_phi_update = 0.0;
        for (std::size_t i = 0u; i < phi_count; ++i) {
          maximum_phi_update =
              std::max(maximum_phi_update,
                       std::abs(certified_solution[phi_offset + i] -
                                current[phi_offset + i]));
        }
        EXPECT_LE(maximum_phi_update, 2.0e-7);

        maximum_kkt_residual =
            std::max(maximum_kkt_residual,
                     static_cast<svmp::FE::Real>(
                         certificate.constant_pressure_kkt_residual_norm));
        maximum_kkt_relative_distance =
            std::max(maximum_kkt_relative_distance,
                     static_cast<svmp::FE::Real>(
                         certificate.constant_pressure_kkt_relative_distance));
        maximum_pressure_jump_error =
            std::max(maximum_pressure_jump_error,
                     static_cast<svmp::FE::Real>(std::abs(
                         certificate.constant_pressure_kkt_pressure_jump)));
        maximum_volume_error = std::max(maximum_volume_error, volume_error);
        maximum_surface_energy_error =
            std::max(maximum_surface_energy_error, surface_energy_error);
        maximum_phi_update_across_cases =
            std::max(maximum_phi_update_across_cases, maximum_phi_update);
      }
    }
  }

  EXPECT_EQ(case_count, 12u);
  RecordProperty("wp4_physical_flat_spatial_dimension", 2);
  RecordProperty("wp4_physical_flat_contact_wall_count", 2);
  RecordProperty("wp4_physical_flat_coordinate_direction_count", 2);
  RecordProperty("wp4_physical_flat_wall_orientation_count", 2);
  RecordProperty("wp4_physical_flat_active_side_count", 2);
  RecordProperty("wp4_physical_flat_cut_offset_count", 3);
  RecordProperty("wp4_physical_flat_matrix_case_count", case_count);
  RecordProperty("wp4_physical_flat_zero_gravity_case_count", case_count);
  RecordProperty("wp4_physical_flat_free_pressure_gauge_case_count",
                 case_count);
  RecordProperty("wp4_physical_flat_mpi_rank_count", 1);
  RecordProperty("wp4_physical_flat_constant_pressure_kkt_residual_norm",
                 maximum_kkt_residual);
  RecordProperty("wp4_physical_flat_constant_pressure_kkt_relative_distance",
                 maximum_kkt_relative_distance);
  RecordProperty("wp4_physical_flat_pressure_jump_absolute_error",
                 maximum_pressure_jump_error);
  RecordProperty("wp4_physical_flat_volume_error", maximum_volume_error);
  RecordProperty("wp4_physical_flat_surface_energy_error",
                 maximum_surface_energy_error);
  RecordProperty("wp4_physical_flat_maximum_phi_update",
                 maximum_phi_update_across_cases);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     StaticCapillaryInitializationBalancesHydrostaticGravityWithFixedPressureGauge)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr int interface_marker = 720;
  constexpr int left_wall_marker = 7201;
  constexpr int right_wall_marker = 7202;
  constexpr int lower_anchor_marker = 7203;
  constexpr int upper_anchor_marker = 7204;
  constexpr int front_wall_marker = 7205;
  constexpr int back_wall_marker = 7206;
  constexpr svmp::FE::Real density = 1.25;
  constexpr svmp::FE::Real gravity_magnitude = 0.4;
  constexpr svmp::FE::Real pi =
      svmp::FE::Real{3.141592653589793238462643383279502884};
  constexpr svmp::FE::Real contact_angle = pi / svmp::FE::Real{2.0};
  WorkflowScopedEnvVar conservative_balance_diagnostic(
      "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", std::string("1"));

  struct HydrostaticCase {
    int spatial_dimension = 2;
    int normal_axis = 0;
    bool positive_side = false;
    svmp::FE::Real normal_offset = 0.0;
    svmp::FE::Real gravity_direction = 0.0;
  };
  constexpr std::array<svmp::FE::Real, 3> normal_offsets{
      svmp::FE::Real{0.35},
      svmp::FE::Real{0.5},
      svmp::FE::Real{0.65},
  };
  std::vector<HydrostaticCase> hydrostatic_cases;
  hydrostatic_cases.reserve(60u);
  for (const int spatial_dimension : {2, 3}) {
    for (int normal_axis = 0;
         normal_axis < spatial_dimension;
         ++normal_axis) {
      for (const bool positive_side : {false, true}) {
        for (const auto normal_offset : normal_offsets) {
          for (const svmp::FE::Real gravity_direction :
               {svmp::FE::Real{-1.0}, svmp::FE::Real{1.0}}) {
            hydrostatic_cases.push_back(HydrostaticCase{
                spatial_dimension,
                normal_axis,
                positive_side,
                normal_offset,
                gravity_direction});
          }
        }
      }
    }
  }

  std::size_t case_count = 0u;
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

  for (const auto& hydrostatic_case : hydrostatic_cases) {
    const auto spatial_dimension = hydrostatic_case.spatial_dimension;
    const auto normal_axis = hydrostatic_case.normal_axis;
    std::vector<int> tangent_axes;
    tangent_axes.reserve(static_cast<std::size_t>(spatial_dimension - 1));
    if (spatial_dimension == 2) {
      tangent_axes.push_back(1 - normal_axis);
    } else {
      tangent_axes.push_back((normal_axis + 1) % 3);
      tangent_axes.push_back((normal_axis + 2) % 3);
    }
    ASSERT_EQ(tangent_axes.size(),
              static_cast<std::size_t>(spatial_dimension - 1));
    const auto positive_side = hydrostatic_case.positive_side;
    const auto normal_offset = hydrostatic_case.normal_offset;
    const auto gravity_direction = hydrostatic_case.gravity_direction;
    const auto gravity = gravity_direction * gravity_magnitude;
    const auto gauge_normal_coordinate =
        positive_side ? svmp::FE::Real{1.0} : svmp::FE::Real{0.0};
    const auto external_pressure =
        density * gravity * (normal_offset - gauge_normal_coordinate);
    SCOPED_TRACE(::testing::Message()
                 << "spatial_dimension=" << spatial_dimension
                 << " normal_axis=" << normal_axis
                 << " active_side="
                 << (positive_side ? "positive" : "negative")
                 << " normal_offset=" << normal_offset
                 << " gravity=" << gravity
                 << " external_pressure=" << external_pressure);
    ++case_count;

    auto mesh = spatial_dimension == 2
                    ? makeWorkflowHydrostaticPressureMesh(normal_axis)
                    : makeWorkflowHydrostaticPressureMesh3D(normal_axis);
    auto& local_mesh = mesh->local_mesh();
    struct ContactWall {
      int marker = -1;
      int axis = 0;
      svmp::FE::Real coordinate = 0.0;
      svmp::FE::Real outward_normal = 0.0;
    };
    std::vector<ContactWall> contact_walls{
        ContactWall{left_wall_marker,
                    tangent_axes[0],
                    svmp::FE::Real{0.0},
                    svmp::FE::Real{-1.0}},
        ContactWall{right_wall_marker,
                    tangent_axes[0],
                    svmp::FE::Real{3.0},
                    svmp::FE::Real{1.0}},
    };
    if (spatial_dimension == 3) {
      contact_walls.push_back(
          ContactWall{front_wall_marker,
                      tangent_axes[1],
                      svmp::FE::Real{0.0},
                      svmp::FE::Real{-1.0}});
      contact_walls.push_back(
          ContactWall{back_wall_marker,
                      tangent_axes[1],
                      svmp::FE::Real{2.0},
                      svmp::FE::Real{1.0}});
    }
    ASSERT_EQ(contact_walls.size(),
              static_cast<std::size_t>(2 * (spatial_dimension - 1)));
    const auto expected_contact_wall_face_count =
        spatial_dimension == 2 ? 4u : 16u;
    const auto expected_anchor_face_count =
        spatial_dimension == 2 ? 4u : 8u;
    std::array<std::size_t, 6> marker_counts{};
    constexpr svmp::FE::Real coordinate_tolerance = 1.0e-12;
    for (const auto face : local_mesh.boundary_faces()) {
      const auto vertices = local_mesh.face_vertices(face);
      ASSERT_EQ(vertices.size(), static_cast<std::size_t>(spatial_dimension));
      std::vector<bool> on_contact_wall(contact_walls.size(), true);
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
          ++marker_counts[wall];
          classified = true;
          break;
        }
      }
      if (classified) {
        continue;
      }
      if (on_lower_anchor) {
        mesh->set_boundary_label(face, lower_anchor_marker);
        ++marker_counts[4];
      } else if (on_upper_anchor) {
        mesh->set_boundary_label(face, upper_anchor_marker);
        ++marker_counts[5];
      } else {
        FAIL() << "Hydrostatic static-capillary fixture found an unclassified face.";
      }
    }
    for (std::size_t wall = 0u;
         wall < contact_walls.size();
         ++wall) {
      EXPECT_EQ(marker_counts[wall], expected_contact_wall_face_count);
    }
    EXPECT_EQ(marker_counts[4], expected_anchor_face_count);
    EXPECT_EQ(marker_counts[5], expected_anchor_face_count);

    std::optional<svmp::FE::GlobalIndex> gauge_vertex_id;
    for (std::size_t vertex = 0u; vertex < mesh->n_vertices(); ++vertex) {
      const auto point = workflowVertexPoint(*mesh, vertex);
      if (std::abs(point[normal_axis] - gauge_normal_coordinate) <=
          coordinate_tolerance) {
        gauge_vertex_id = static_cast<svmp::FE::GlobalIndex>(vertex);
        break;
      }
    }
    ASSERT_TRUE(gauge_vertex_id.has_value());

    const auto mesh_field =
        svmp::MeshFields::attach_field(local_mesh,
                                       svmp::EntityKind::Vertex,
                                       "phi_physical_hydrostatic_fixed_gauge",
                                       svmp::FieldScalarType::Float64,
                                       1);
    auto* mesh_phi = svmp::MeshFields::field_data_as<svmp::real_t>(
        local_mesh, mesh_field);
    ASSERT_NE(mesh_phi, nullptr);
    std::vector<svmp::FE::Real> phi_vertex_values(mesh->n_vertices(), 0.0);
    for (std::size_t vertex = 0u; vertex < mesh->n_vertices(); ++vertex) {
      phi_vertex_values[vertex] =
          workflowVertexPoint(*mesh, vertex)[normal_axis] - normal_offset;
      mesh_phi[vertex] = phi_vertex_values[vertex];
    }

    const auto element_type =
        spatial_dimension == 2 ? svmp::FE::ElementType::Triangle3
                               : svmp::FE::ElementType::Tetra4;
    auto scalar_space = svmp::FE::spaces::SpaceFactory::create_h1(
        element_type, /*order=*/1);
    auto velocity_space = svmp::FE::spaces::SpaceFactory::create_vector_h1(
        element_type,
        /*order=*/1,
        /*components=*/spatial_dimension);
    auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
    const auto phi = system->addField(
        svmp::FE::systems::FieldSpec{
            .name = "phi_physical_hydrostatic_fixed_gauge",
            .space = scalar_space,
            .components = 1});

    channel_ns::IncompressibleNavierStokesVMSOptions options;
    options.velocity_field_name = "u_physical_hydrostatic_fixed_gauge";
    options.pressure_field_name = "p_physical_hydrostatic_fixed_gauge";
    options.density = density;
    options.viscosity = 0.01;
    options.body_force[normal_axis] = gravity;
    options.enable_convection = false;
    options.enable_vms = false;
    options.jit_policy.enable = false;
    options.velocity_dirichlet.push_back(
        channel_ns::IncompressibleNavierStokesVMSOptions::
            VelocityDirichletBC{
                .boundary_marker = positive_side ? upper_anchor_marker
                                                 : lower_anchor_marker,
                .value = {0.0, 0.0, 0.0},
            });
    for (const auto& wall : contact_walls) {
      std::array<bool, 3> active_components{};
      active_components[static_cast<std::size_t>(wall.axis)] = true;
      options.velocity_dirichlet.push_back(
          channel_ns::IncompressibleNavierStokesVMSOptions::
              VelocityDirichletBC{
                  .boundary_marker = wall.marker,
                  .value = {0.0, 0.0, 0.0},
                  .active_components = active_components,
              });
    }
    options.node_pressure_constraints.id_type =
        channel_ns::IncompressibleNavierStokesVMSOptions::
            NodePressureConstraintIdType::LocalVertexId;
    options.node_pressure_constraints.values.push_back(
        channel_ns::IncompressibleNavierStokesVMSOptions::
            NodePressureConstraint{
                .node_id = *gauge_vertex_id,
                .pressure = 0.0,
            });

    using ContactLine = channel_ns::IncompressibleNavierStokesVMSOptions::
        FreeSurfaceContactLine;
    auto free_surface =
        channel_ns::IncompressibleNavierStokesVMSOptions::
            FreeSurfaceBoundary{
                .implementation =
                    channel_ns::FreeSurfaceImplementation::UnfittedLevelSet,
                .interface_marker = interface_marker,
                .level_set_field_name =
                    "phi_physical_hydrostatic_fixed_gauge",
                .generated_interface_domain_id =
                    "physical_hydrostatic_fixed_gauge",
                .generated_interface_geometry = "LinearCorner",
                .active_domain =
                    positive_side
                        ? channel_ns::FreeSurfaceActiveDomain::LevelSetPositive
                        : channel_ns::FreeSurfaceActiveDomain::LevelSetNegative,
                .active_domain_method =
                    channel_ns::FreeSurfaceActiveDomainMethod::CutVolume,
                .external_pressure = external_pressure,
                .surface_tension = 1.0,
                .surface_tension_form =
                    channel_ns::FreeSurfaceSurfaceTensionForm::SurfaceStress,
                .curvature = 0.0,
                .use_level_set_curvature = false,
                .small_cut_aggregation = false,
            };
    for (const auto& wall : contact_walls) {
      ContactLine::DynamicRenE dynamic_contact;
      dynamic_contact.wall_boundary_marker = wall.marker;
      dynamic_contact.contact_line_marker = -1;
      dynamic_contact.equilibrium_contact_angle_radians = contact_angle;
      dynamic_contact.wall_normal[static_cast<std::size_t>(wall.axis)] =
          wall.outward_normal;
      dynamic_contact.mobility = 1.0;
      dynamic_contact.slip_length = 1.0;
      free_surface.contact_lines.push_back(
          ContactLine{.configuration = std::move(dynamic_contact)});
    }
    options.free_surface.push_back(std::move(free_surface));

    channel_ns::IncompressibleNavierStokesVMSModule module(
        velocity_space, scalar_space, std::move(options));
    module.registerOn(*system);
    const auto velocity = system->findFieldByName(
        "u_physical_hydrostatic_fixed_gauge");
    const auto pressure = system->findFieldByName(
        "p_physical_hydrostatic_fixed_gauge");
    ASSERT_NE(velocity, svmp::FE::INVALID_FIELD_ID);
    ASSERT_NE(pressure, svmp::FE::INVALID_FIELD_ID);
    ASSERT_NO_THROW(system->setup({}));

    const auto phi_coefficients = projectWorkflowVertexValues(
        *system,
        phi,
        phi_vertex_values,
        /*components=*/1u,
        "ApplicationDriver hydrostatic fixed-gauge phi");
    std::vector<svmp::FE::Real> current(
        static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
    writeWorkflowFieldSlice(*system, phi, phi_coefficients, current);

    application::core::SimulationComponents sim;
    sim.primary_mesh = mesh;
    sim.fe_system = std::move(system);
    try {
      sim.backend = svmp::FE::backends::BackendFactory::create(
          svmp::FE::backends::BackendKind::FSILS);
    } catch (const std::exception&) {
      GTEST_SKIP() << "Requires an available FE vector backend.";
    }
    ASSERT_NE(sim.backend, nullptr);
    svmp::FE::backends::SolverOptions linear_options;
    linear_options.method = svmp::FE::backends::SolverMethod::GMRES;
    linear_options.preconditioner =
        svmp::FE::backends::PreconditionerType::Diagonal;
    sim.linear_solver = sim.backend->createLinearSolver(linear_options);
    ASSERT_NE(sim.linear_solver, nullptr);
    auto allocated_history = svmp::FE::timestepping::TimeHistory::allocate(
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

    std::ostringstream contact_wall_markers;
    std::ostringstream contact_wall_normals;
    for (std::size_t wall_index = 0u;
         wall_index < contact_walls.size();
         ++wall_index) {
      if (wall_index != 0u) {
        contact_wall_markers << ';';
        contact_wall_normals << "; ";
      }
      contact_wall_markers << contact_walls[wall_index].marker;
      std::array<svmp::FE::Real, 3> wall_normal{};
      wall_normal[static_cast<std::size_t>(contact_walls[wall_index].axis)] =
          contact_walls[wall_index].outward_normal;
      contact_wall_normals << wall_normal[0] << ' ' << wall_normal[1]
                           << ' ' << wall_normal[2];
    }

    std::ostringstream parameter_xml;
    parameter_xml << std::setprecision(17) << R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Level_set_field_name>phi_physical_hydrostatic_fixed_gauge</Level_set_field_name>
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
    <Add_BC name="physical_hydrostatic_fixed_gauge">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi_physical_hydrostatic_fixed_gauge</Level_set_field_name>
      <Generated_interface_domain_id>physical_hydrostatic_fixed_gauge</Generated_interface_domain_id>
      <Interface_marker>720</Interface_marker>
      <Generated_interface_geometry>LinearCorner</Generated_interface_geometry>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>)xml"
                  << (positive_side ? "LevelSetPositive" : "LevelSetNegative")
                  << R"xml(</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
      <Small_cut_aggregation>false</Small_cut_aggregation>
      <External_pressure>)xml"
                  << external_pressure << R"xml(</External_pressure>
      <Surface_tension>1.0</Surface_tension>
      <Surface_tension_form>SurfaceStress</Surface_tension_form>
      <Contact_line_model>DynamicContactAngle</Contact_line_model>
      <Contact_angle_degrees>90.0</Contact_angle_degrees>
      <Contact_line_wall_markers>)xml"
                  << contact_wall_markers.str()
                  << R"xml(</Contact_line_wall_markers>
      <Contact_line_wall_normals>)xml"
                  << contact_wall_normals.str()
                  << R"xml(</Contact_line_wall_normals>
      <Contact_line_mobility>1.0</Contact_line_mobility>
      <Wall_slip_model>Navier</Wall_slip_model>
      <Wall_slip_length>1.0</Wall_slip_length>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml";
    const auto parameter_text = parameter_xml.str();
    auto params = parseWorkflowParametersXml(parameter_text.c_str());
    auto requests = levelSetMaintenanceRequests(*params);
    ASSERT_EQ(requests.size(), 1u);
    ASSERT_TRUE(requests.front().static_capillary_equilibrium_enabled);

    svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    ActiveCutContextRefreshCache refresh_cache;
    const auto initial_report = refreshActiveCutIntegrationContextCached(
        sim,
        *params,
        sim.time_history->u(),
        lifecycle,
        refresh_cache,
        "application-driver-hydrostatic-fixed-gauge-initial");
    ASSERT_TRUE(initial_report.refreshed);
    auto initial_functionals =
        evaluateCurrentFreeSurfaceDiscreteFunctionals(sim);
    ASSERT_EQ(initial_functionals.size(), 1u);
    attachAcceptedFreeSurfaceActiveVolumeEnergies(
        sim, current, initial_functionals);
    ASSERT_TRUE(initial_functionals.front().active_volume_energy.has_value());
    const auto interface_measure = spatial_dimension == 2
                                       ? svmp::FE::Real{3.0}
                                       : svmp::FE::Real{6.0};
    const auto expected_volume =
        interface_measure *
        (positive_side ? svmp::FE::Real{1.0} - normal_offset
                       : normal_offset);
    const auto active_first_moment =
        svmp::FE::Real{0.5} * interface_measure *
        (positive_side
             ? svmp::FE::Real{1.0} - normal_offset * normal_offset
             : normal_offset * normal_offset);
    const auto expected_gravitational_energy =
        -density * gravity * active_first_moment;
    EXPECT_NEAR(initial_functionals.front().state.owned_liquid_volume,
                expected_volume,
                1.0e-13);
    EXPECT_NEAR(initial_functionals.front().state.liquid_gas_surface_energy,
                interface_measure,
                1.0e-13);
    EXPECT_NEAR(initial_functionals.front().state.young_wall_energy,
                svmp::FE::Real{0.0},
                1.0e-13);
    EXPECT_NEAR(
        initial_functionals.front().active_volume_energy->gravitational_energy,
        expected_gravitational_energy,
        2.0e-13);

    std::vector<svmp::FE::Real> expected_pressure_vertex_values(
        mesh->n_vertices(), 0.0);
    for (std::size_t vertex = 0u; vertex < mesh->n_vertices(); ++vertex) {
      const auto normal_coordinate =
          workflowVertexPoint(*mesh, vertex)[normal_axis];
      expected_pressure_vertex_values[vertex] =
          density * gravity *
          (normal_coordinate - gauge_normal_coordinate);
    }
    const auto analytic_pressure_coefficients = projectWorkflowVertexValues(
        *sim.fe_system,
        pressure,
        expected_pressure_vertex_values,
        /*components=*/1u,
        "ApplicationDriver hydrostatic fixed-gauge pressure");
    auto exact_solution = current;
    writeWorkflowFieldSlice(
        *sim.fe_system,
        pressure,
        analytic_pressure_coefficients,
        exact_solution);
    sim.fe_system->updateConstraints(
        sim.time_history->time(), sim.time_history->dt());
    sim.fe_system->constraints().distribute(exact_solution);
    const auto pressure_offset = static_cast<std::size_t>(
        sim.fe_system->fieldDofOffset(pressure));
    const auto pressure_count = static_cast<std::size_t>(
        sim.fe_system->fieldDofHandler(pressure).getNumDofs());
    const std::vector<svmp::FE::Real> expected_pressure_coefficients(
        exact_solution.begin() +
            static_cast<std::ptrdiff_t>(pressure_offset),
        exact_solution.begin() +
            static_cast<std::ptrdiff_t>(pressure_offset + pressure_count));
    const auto exact_pressure_certificate =
        evaluateStaticCapillaryPressureCertificate(
            sim,
            exact_solution,
            requests.front().static_capillary_equilibrium,
            /*initialize_compatible_pressure=*/false);
    const auto& exact_certificate = exact_pressure_certificate.report;
    ASSERT_TRUE(exact_certificate.pressure_representability_diagnostic_sampled);
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
        exact_initialized_report.static_compatible_pressure_initializer_applied);
    ASSERT_TRUE(
        exact_initialized_report.static_compatible_pressure_initializer_passed);
    EXPECT_LE(exact_initialized_report.residual_norm, 2.0e-12);
    ASSERT_EQ(exact_initialized_pressure_certificate.certified_solution.size(),
              exact_solution.size());
    svmp::FE::Real exact_initializer_pressure_update = 0.0;
    for (std::size_t i = 0u; i < pressure_count; ++i) {
      exact_initializer_pressure_update =
          std::max(
              exact_initializer_pressure_update,
              std::abs(
                  exact_initialized_pressure_certificate.certified_solution[
                      pressure_offset + i] -
                  exact_solution[pressure_offset + i]));
    }
    EXPECT_LE(exact_initializer_pressure_update, 2.0e-12);
    maximum_exact_initializer_pressure_update =
        std::max(maximum_exact_initializer_pressure_update,
                 exact_initializer_pressure_update);

    bool initialized = false;
    ASSERT_NO_THROW(
        initialized = initializeDiscreteStaticCapillaryEquilibrium(
            sim, *params, requests, lifecycle, refresh_cache));
    ASSERT_TRUE(initialized);
    ASSERT_TRUE(requests.front().static_capillary_equilibrium_initialized);

    const auto certified_solution =
        gatherFeOrderedSolution(sim.time_history->u());
    const auto pressure_certificate =
        evaluateStaticCapillaryPressureCertificate(
            sim,
            certified_solution,
            requests.front().static_capillary_equilibrium,
            /*initialize_compatible_pressure=*/false);
    const auto& certificate = pressure_certificate.report;
    ASSERT_TRUE(certificate.pressure_representability_diagnostic_sampled);
    ASSERT_TRUE(certificate.pressure_representability_available)
        << certificate.pressure_representability_reason;
    EXPECT_TRUE(certificate.pressure_representability_converged);
    EXPECT_FALSE(certificate.pressure_representability_breakdown);
    EXPECT_LE(certificate.pressure_representability_residual_norm, 2.0e-10);
    EXPECT_LE(certificate.pressure_representability_relative_distance,
              2.0e-10);
    EXPECT_LE(certificate.residual_norm, 2.0e-10);
    EXPECT_FALSE(certificate.constant_pressure_constraints_preserve_constants);
    EXPECT_FALSE(certificate.constant_pressure_kkt_available);

    svmp::FE::Real initializer_pressure_representative_distance = 0.0;
    for (std::size_t i = 0u; i < expected_pressure_coefficients.size(); ++i) {
      initializer_pressure_representative_distance =
          std::max(initializer_pressure_representative_distance,
                   std::abs(certified_solution[pressure_offset + i] -
                            expected_pressure_coefficients[i]));
    }
    EXPECT_TRUE(std::isfinite(initializer_pressure_representative_distance));

    const auto phi_offset = static_cast<std::size_t>(
        sim.fe_system->fieldDofOffset(phi));
    svmp::FE::Real phi_update = 0.0;
    for (std::size_t i = 0u; i < phi_coefficients.size(); ++i) {
      phi_update =
          std::max(phi_update,
                   std::abs(certified_solution[phi_offset + i] -
                            current[phi_offset + i]));
    }
    EXPECT_LE(phi_update, 2.0e-7);

    auto final_functionals =
        evaluateCurrentFreeSurfaceDiscreteFunctionals(sim);
    ASSERT_EQ(final_functionals.size(), 1u);
    attachAcceptedFreeSurfaceActiveVolumeEnergies(
        sim, certified_solution, final_functionals);
    ASSERT_TRUE(final_functionals.front().active_volume_energy.has_value());
    const auto gravitational_energy_error = std::abs(
        final_functionals.front().active_volume_energy->gravitational_energy -
        expected_gravitational_energy);
    const auto volume_error = std::abs(
        final_functionals.front().state.owned_liquid_volume -
        expected_volume);
    const auto surface_energy_error = std::abs(
        final_functionals.front().state.liquid_gas_surface_energy -
        interface_measure);
    EXPECT_LE(gravitational_energy_error, 2.0e-10);
    EXPECT_LE(volume_error, 1.0e-11);
    EXPECT_LE(surface_energy_error, 2.0e-10);
    EXPECT_NEAR(final_functionals.front().state.young_wall_energy,
                svmp::FE::Real{0.0},
                1.0e-13);

    maximum_pressure_residual =
        std::max(maximum_pressure_residual,
                 static_cast<svmp::FE::Real>(
                     certificate.pressure_representability_residual_norm));
    maximum_pressure_relative_distance =
        std::max(maximum_pressure_relative_distance,
                 static_cast<svmp::FE::Real>(
                     certificate.pressure_representability_relative_distance));
    maximum_exact_field_production_residual =
        std::max(maximum_exact_field_production_residual,
                 static_cast<svmp::FE::Real>(exact_certificate.residual_norm));
    maximum_production_residual =
        std::max(maximum_production_residual,
                 static_cast<svmp::FE::Real>(certificate.residual_norm));
    maximum_initializer_pressure_representative_distance =
        std::max(maximum_initializer_pressure_representative_distance,
                 initializer_pressure_representative_distance);
    maximum_gravitational_energy_error =
        std::max(maximum_gravitational_energy_error,
                 gravitational_energy_error);
    maximum_volume_error =
        std::max(maximum_volume_error, volume_error);
    maximum_surface_energy_error =
        std::max(maximum_surface_energy_error, surface_energy_error);
    maximum_phi_update = std::max(maximum_phi_update, phi_update);
  }

  EXPECT_EQ(case_count, 60u);
  RecordProperty("wp4_hydrostatic_spatial_dimension", 3);
  RecordProperty("wp4_hydrostatic_spatial_dimension_count", 2);
  RecordProperty("wp4_hydrostatic_coordinate_direction_count", 3);
  RecordProperty("wp4_hydrostatic_dimension_coordinate_pair_count", 5);
  RecordProperty("wp4_hydrostatic_wall_orientation_count", 3);
  RecordProperty("wp4_hydrostatic_active_side_count", 2);
  RecordProperty("wp4_hydrostatic_cut_offset_count", normal_offsets.size());
  RecordProperty("wp4_hydrostatic_gravity_direction_count", 2);
  RecordProperty("wp4_hydrostatic_two_dimensional_case_count", 24);
  RecordProperty("wp4_hydrostatic_three_dimensional_case_count", 36);
  RecordProperty("wp4_hydrostatic_fixed_zero_pressure_gauge_case_count",
                 case_count);
  RecordProperty("wp4_hydrostatic_matrix_case_count", case_count);
  RecordProperty("wp4_hydrostatic_pressure_representability_residual_norm",
                 maximum_pressure_residual);
  RecordProperty("wp4_hydrostatic_pressure_relative_distance",
                 maximum_pressure_relative_distance);
  RecordProperty("wp4_hydrostatic_exact_field_production_residual_norm",
                 maximum_exact_field_production_residual);
  RecordProperty("wp4_hydrostatic_production_residual_norm",
                 maximum_production_residual);
  RecordProperty(
      "wp4_hydrostatic_initializer_pressure_representative_distance",
      maximum_initializer_pressure_representative_distance);
  RecordProperty("wp4_hydrostatic_exact_initializer_pressure_update",
                 maximum_exact_initializer_pressure_update);
  RecordProperty("wp4_hydrostatic_gravitational_energy_error",
                 maximum_gravitational_energy_error);
  RecordProperty("wp4_hydrostatic_volume_error", maximum_volume_error);
  RecordProperty("wp4_hydrostatic_surface_energy_error",
                 maximum_surface_energy_error);
  RecordProperty("wp4_hydrostatic_maximum_phi_update", maximum_phi_update);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     ConsumedContactMarkerAcceptsCurvedHighOrderFragments)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr int wall_marker = 17;
  constexpr int interface_marker = 705;
  auto mesh = makeWorkflowBiquadraticQuadMesh();
  for (const auto face : mesh->local_mesh().boundary_faces())
  {
    mesh->local_mesh().set_boundary_label(face, wall_marker);
  }
  const auto mesh_field =
      svmp::MeshFields::attach_field(mesh->local_mesh(),
                                     svmp::EntityKind::Vertex,
                                     "phi",
                                     svmp::FieldScalarType::Float64,
                                     1);
  ASSERT_NE(svmp::MeshFields::field_data_as<svmp::real_t>(mesh->local_mesh(),
                                                          mesh_field),
            nullptr);

  auto scalar_space =
      std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Quad4,
                                                  /*order=*/2);
  auto velocity_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, 2);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi", .space = scalar_space, .components = 1 });
  const auto velocity = system->addField(svmp::FE::systems::FieldSpec{
      .name = "velocity", .space = velocity_space, .components = 2 });

  svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalParameters
      functional_parameters;
  functional_parameters.liquid_side =
      svmp::FE::geometry::CutIntegrationSide::Negative;
  functional_parameters.surface_tension = svmp::FE::Real{ 0.8 };
  functional_parameters.young_wall_coefficients.push_back(
      svmp::FE::interfaces::FreeSurfaceYoungWallCoefficient{
          .boundary_marker = wall_marker,
          .equilibrium_contact_angle_radians =
              svmp::FE::Real{ 1.04719755119659774615421446109316763 },
      });
  functional_parameters.dynamic_contact_coefficients.push_back(
      svmp::FE::interfaces::FreeSurfaceDynamicContactCoefficient{
          .boundary_marker = wall_marker,
          .equilibrium_contact_angle_radians =
              svmp::FE::Real{ 1.04719755119659774615421446109316763 },
          .mobility = svmp::FE::Real{ 0.5 },
          .slip_length = svmp::FE::Real{ 0.2 },
          .dynamic_viscosity = svmp::FE::Real{ 0.4 },
      });
  system->declareFreeSurfaceDiscreteFunctional(
      svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration{
          .interface_marker = interface_marker,
          .level_set_field = phi,
          .velocity_field = velocity,
          .geometry_domain_id = "degenerate_contact",
          .parameters = functional_parameters,
          .owner_component =
              "ApplicationDriverLevelSetWorkflows.ContactStageFixture",
      });

  svmp::FE::interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey key{};
  key.source = svmp::FE::interfaces::LevelSetInterfaceSource::fromField(phi);
  key.domain_id = "degenerate_contact";
  key.isovalue = 0.0;
  key.interface_marker = interface_marker;
  key.boundary_marker = wall_marker;
  const int contact_marker =
      svmp::FE::interfaces::stableGeneratedInterfaceBoundaryIntersectionMarker(
          key);
  system->registerGeneratedEmbeddedInterfaceMarker(contact_marker);
  ASSERT_TRUE(
      system->isGeneratedEmbeddedInterfaceMarkerRegistered(contact_marker));
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_vertex_values(mesh->n_vertices(), 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex)
  {
    const auto point = workflowVertexPoint(*mesh, vertex);
    // The interface crosses curved parent geometry.  The generated contact
    // trace must remain available to a registered consumer without falling
    // back to a skipped fragment.
    phi_vertex_values[vertex] =
        svmp::FE::Real{ 2.0 } * (point[0] - svmp::FE::Real{ 0.45 });
  }
  const auto coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      std::span<const svmp::FE::Real>(phi_vertex_values.data(),
                                      phi_vertex_values.size()),
      1u,
      "ApplicationDriver degenerate-contact phi");
  std::vector<svmp::FE::Real> solution(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, coefficients, solution);
  const auto velocity_dofs = static_cast<std::size_t>(
      system->fieldDofHandler(velocity).getNumDofs());
  std::vector<svmp::FE::Real> velocity_coefficients(
      velocity_dofs, svmp::FE::Real{0.2});
  writeWorkflowFieldSlice(
      *system, velocity, velocity_coefficients, solution);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>degenerate_contact</Generated_interface_domain_id>
      <Interface_marker>705</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
  ASSERT_NO_THROW((void)refreshActiveCutIntegrationContextFromSolution(
      sim,
      *params,
      std::span<const svmp::FE::Real>(solution.data(), solution.size()),
      lifecycle,
      "application-driver-curved-contact-test"));

  const auto* context = sim.fe_system->cutIntegrationContext();
  ASSERT_NE(context, nullptr);
  ASSERT_TRUE(context->hasFreeSurfaceGeometrySnapshotForMarker(contact_marker));
  ASSERT_EQ(context->freeSurfaceGeometrySnapshots().size(), 1u);
  const auto snapshot = context->freeSurfaceGeometrySnapshots().front();
  ASSERT_NE(snapshot, nullptr);
  ASSERT_EQ(snapshot->contactDomains().size(), 1u);
  EXPECT_EQ(snapshot->contactDomains().front().marker(), contact_marker);
  const auto summary = snapshot->contactDomains().front().summary();
  EXPECT_EQ(summary.fragment_count, 2u);
  EXPECT_EQ(summary.active_fragment_count, 2u);
  EXPECT_EQ(summary.skipped_fragment_count, 0u);
  EXPECT_EQ(snapshot->ledger().orphan_contact_fragment_count, 0u);
  EXPECT_EQ(snapshot->ledger().stale_revision_count, 0u);

  auto endpoint_solution = solution;
  auto previous_solution = solution;
  const auto phi_offset =
      static_cast<std::size_t>(sim.fe_system->fieldDofOffset(phi));
  const auto phi_dof_count = static_cast<std::size_t>(
      sim.fe_system->fieldDofHandler(phi).getNumDofs());
  ASSERT_LE(phi_offset + phi_dof_count, solution.size());
  std::vector<svmp::FE::Real> endpoint_phi_values(
      mesh->n_vertices(), svmp::FE::Real{0.0});
  std::vector<svmp::FE::Real> previous_phi_values(
      mesh->n_vertices(), svmp::FE::Real{0.0});
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    endpoint_phi_values[vertex] =
        svmp::FE::Real{2.0} * (point[0] - svmp::FE::Real{0.60});
    previous_phi_values[vertex] =
        svmp::FE::Real{2.0} * (point[0] - svmp::FE::Real{0.30});
  }
  const auto endpoint_coefficients = projectWorkflowVertexValues(
      *sim.fe_system,
      phi,
      endpoint_phi_values,
      1u,
      "ApplicationDriver endpoint contact-stage phi");
  const auto previous_coefficients = projectWorkflowVertexValues(
      *sim.fe_system,
      phi,
      previous_phi_values,
      1u,
      "ApplicationDriver previous contact-stage phi");
  writeWorkflowFieldSlice(
      *sim.fe_system, phi, endpoint_coefficients, endpoint_solution);
  writeWorkflowFieldSlice(
      *sim.fe_system, phi, previous_coefficients, previous_solution);
  for (std::size_t i = 0; i < phi_dof_count; ++i) {
    EXPECT_NEAR(
        svmp::FE::Real{0.5} *
            (endpoint_solution[phi_offset + i] +
             previous_solution[phi_offset + i]),
        solution[phi_offset + i],
        1.0e-13);
  }
  auto factory = svmp::FE::backends::BackendFactory::create(
      svmp::FE::backends::BackendKind::FSILS);
  ASSERT_NE(factory, nullptr);
  auto time_history = svmp::FE::timestepping::TimeHistory::allocate(
      *factory,
      sim.fe_system->dofHandler().getNumDofs(),
      /*history_depth=*/2,
      /*allocate_second_order_state=*/true);
  time_history.setTime(0.10);
  time_history.setDt(0.05);
  time_history.setPrevDt(0.05);
  time_history.setStepIndex(2);
  scatterFeOrderedSolution(time_history.u(), endpoint_solution);
  scatterFeOrderedSolution(time_history.uPrev(), previous_solution);
  scatterFeOrderedSolution(time_history.uPrev2(), previous_solution);
  std::vector<svmp::FE::Real> contact_rate(endpoint_solution.size());
  std::vector<svmp::FE::Real> contact_acceleration(
      endpoint_solution.size());
  for (std::size_t i = 0u; i < endpoint_solution.size(); ++i) {
    contact_rate[i] = svmp::FE::Real{2.0} +
                      static_cast<svmp::FE::Real>(i);
    contact_acceleration[i] = svmp::FE::Real{3.0} +
                              static_cast<svmp::FE::Real>(i);
  }
  scatterFeOrderedSolution(time_history.uDot(), contact_rate);
  scatterFeOrderedSolution(time_history.uDDot(), contact_acceleration);

  // Emulate the four TimeLoop acceptance transitions without requiring a
  // second nonlinear solve fixture: finalized generalized-alpha endpoint,
  // final-candidate stage rebuild, TimeHistory::acceptStep(), then the first
  // accepted-callback provenance capture/bind.
  auto acceptance_order_history =
      svmp::FE::timestepping::TimeHistory::allocate(
          *factory,
          sim.fe_system->dofHandler().getNumDofs(),
          /*history_depth=*/2,
          /*allocate_second_order_state=*/false);
  acceptance_order_history.setTime(0.05);
  acceptance_order_history.setDt(0.05);
  acceptance_order_history.setPrevDt(0.05);
  acceptance_order_history.setStepIndex(1);
  scatterFeOrderedSolution(
      acceptance_order_history.u(), previous_solution);
  scatterFeOrderedSolution(
      acceptance_order_history.uPrev(), previous_solution);
  scatterFeOrderedSolution(
      acceptance_order_history.uPrev2(), previous_solution);
  // The solve/finalization transition publishes the authoritative endpoint
  // that on_step_candidate_ready must consume.
  scatterFeOrderedSolution(
      acceptance_order_history.u(), endpoint_solution);
  const auto finalized_endpoint_solution =
      gatherFeOrderedSolution(acceptance_order_history.u());
  const auto candidate_ready_previous_solution =
      gatherFeOrderedSolution(acceptance_order_history.uPrev());
  EXPECT_EQ(finalized_endpoint_solution, endpoint_solution);
  EXPECT_EQ(candidate_ready_previous_solution, previous_solution);

  ASSERT_NO_THROW((void)refreshActiveCutIntegrationContextFromSolution(
      sim,
      *params,
      finalized_endpoint_solution,
      lifecycle,
      "application-driver-contact-endpoint-before-stage-test"));
  ActiveCutContextRefreshCache contact_stage_refresh_cache;
  const auto active_cut_requests = activeCutVolumeRequests(*params);
  const auto previous_state_revision =
      collectiveLevelSetMaintenanceAlgebraicRevision(
          candidate_ready_previous_solution,
          activeFESystemCommunicator(*sim.fe_system));
  const auto endpoint_state_revision =
      collectiveLevelSetMaintenanceAlgebraicRevision(
          finalized_endpoint_solution,
          activeFESystemCommunicator(*sim.fe_system));
  const svmp::FE::systems::
      FreeSurfaceFirstOrderGeneralizedAlphaProvenance
          generalized_alpha_provenance{
              .alpha_m = svmp::FE::Real{0.5},
              .alpha_f = svmp::FE::Real{0.5},
              .gamma = svmp::FE::Real{0.5},
              .dt = svmp::FE::Real{0.05},
          };
  const auto& contact_stage_mesh = sim.fe_system->meshAccess();
  std::vector<svmp::FE::Real> exact_stage_phi_vertex_values(
      mesh->n_vertices(), svmp::FE::Real{0.0});
  for (std::size_t vertex = 0u; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    exact_stage_phi_vertex_values[vertex] =
        svmp::FE::Real{2.0} * (point[0] - svmp::FE::Real{0.40});
  }
  const auto exact_stage_phi_coefficients = projectWorkflowVertexValues(
      *sim.fe_system,
      phi,
      exact_stage_phi_vertex_values,
      1u,
      "ApplicationDriver exact non-affine contact-stage phi");
  auto exact_operator_stage_solution = solution;
  writeWorkflowFieldSlice(
      *sim.fe_system,
      phi,
      exact_stage_phi_coefficients,
      exact_operator_stage_solution);
  const auto velocity_offset = static_cast<std::size_t>(
      sim.fe_system->fieldDofOffset(velocity));
  ASSERT_LT(velocity_offset + 1u, exact_operator_stage_solution.size());
  exact_operator_stage_solution[velocity_offset] = svmp::FE::Real{0.35};
  exact_operator_stage_solution[velocity_offset + 1u] =
      svmp::FE::Real{-0.0};
  const auto contact_stage_declarations =
      sim.fe_system->freeSurfaceDiscreteFunctionalDeclarations();
  DynamicContactFirstOrderGeneralizedAlphaObservation
      exact_operator_stage_observation{
          .step_index = 2,
          .attempt_index = 0,
          .step_start_time = svmp::FE::Real{0.05},
          .step_end_time = svmp::FE::Real{0.10},
          .state_time = svmp::FE::Real{0.075},
          .rate_time = svmp::FE::Real{0.075},
          .mesh_revision = {
              .geometry_revision = contact_stage_mesh.geometryRevision(),
              .topology_revision = contact_stage_mesh.topologyRevision(),
              .ownership_revision = contact_stage_mesh.ownershipRevision(),
              .numbering_revision = contact_stage_mesh.numberingRevision(),
              .field_layout_revision =
                  contact_stage_mesh.fieldLayoutRevision(),
              .label_revision = contact_stage_mesh.labelRevision(),
              .active_configuration_epoch =
                  contact_stage_mesh.activeConfigurationEpoch(),
              .coordinate_configuration_key =
                  contact_stage_mesh.coordinateConfigurationKey(),
          },
          .contact_wall_boundary_fingerprint =
              dynamicContactWallBoundaryFingerprint(
                  contact_stage_mesh, contact_stage_declarations),
          .provenance = generalized_alpha_provenance,
      };
  auto exact_operator_stage_history =
      svmp::FE::timestepping::TimeHistory::allocate(
          *factory,
          sim.fe_system->dofHandler().getNumDofs(),
          /*history_depth=*/2,
          /*allocate_second_order_state=*/false);
  scatterFeOrderedSolution(
      exact_operator_stage_history.u(), exact_operator_stage_solution);
  exact_operator_stage_observation.operator_stage_state =
      captureDynamicContactOperatorStageState(
          *sim.fe_system,
          contact_stage_declarations,
          exact_operator_stage_history.u(),
          activeFESystemCommunicator(*sim.fe_system));
  std::optional<std::size_t> retained_phi_entry;
  std::optional<std::size_t> retained_velocity_entry;
  for (auto& field :
       exact_operator_stage_observation.operator_stage_state.fields) {
    if (field.field == phi) {
      ASSERT_FALSE(field.values.empty());
      retained_phi_entry = static_cast<std::size_t>(field.offset);
    }
    if (field.field == velocity) {
      ASSERT_GT(field.values.size(), 1u);
      EXPECT_EQ(
          std::bit_cast<std::uint64_t>(field.values[1]),
          std::bit_cast<std::uint64_t>(svmp::FE::Real{-0.0}));
      retained_velocity_entry = static_cast<std::size_t>(field.offset);
    }
  }
  ASSERT_TRUE(retained_phi_entry.has_value());
  ASSERT_TRUE(retained_velocity_entry.has_value());
  // This is the production on_step_candidate_ready rebuild, after endpoint
  // finalization rather than from an earlier conservative-phase candidate.
  auto contact_stage_candidate =
      buildAcceptedFreeSurfaceContactStageCandidate(
          sim,
          *params,
          lifecycle,
          contact_stage_refresh_cache,
          active_cut_requests,
          svmp::FE::Real{0.075},
          svmp::FE::Real{0.5},
          previous_state_revision,
          endpoint_state_revision,
          candidate_ready_previous_solution,
          finalized_endpoint_solution,
          nullptr,
          generalized_alpha_provenance,
          &exact_operator_stage_observation);
  auto& contact_stages = contact_stage_candidate.stages;
  const auto& contact_stage_constraints =
      contact_stage_candidate.constraints;
  ASSERT_EQ(contact_stages.size(), 1u);
  ASSERT_TRUE(
      contact_stages.front().first_order_generalized_alpha.has_value());
  EXPECT_EQ(
      *contact_stages.front().first_order_generalized_alpha,
      generalized_alpha_provenance);
  EXPECT_DOUBLE_EQ(
      contact_stage_candidate.stage_solution[*retained_velocity_entry],
      svmp::FE::Real{0.35});
  EXPECT_NE(
      contact_stage_candidate.stage_solution[*retained_velocity_entry],
      svmp::FE::Real{0.5} *
          (candidate_ready_previous_solution[*retained_velocity_entry] +
           finalized_endpoint_solution[*retained_velocity_entry]));
  EXPECT_DOUBLE_EQ(
      contact_stage_candidate.stage_solution[*retained_phi_entry],
      exact_operator_stage_solution[*retained_phi_entry]);
  EXPECT_NE(
      contact_stage_candidate.stage_solution[*retained_phi_entry],
      svmp::FE::Real{0.5} *
          (candidate_ready_previous_solution[*retained_phi_entry] +
           finalized_endpoint_solution[*retained_phi_entry]));
  ASSERT_EQ(contact_stages.front().state.walls.size(), 1u);
  EXPECT_GT(contact_stages.front().state.owned_contact_measure, 0.0);
  EXPECT_GT(contact_stages.front().state.line_friction_dissipation, 0.0);
  EXPECT_NEAR(
      contact_stages.front().state.walls.front().mean_contact_position[0],
      svmp::FE::Real{0.40},
      1.0e-12);
  for (const auto component :
       contact_stages.front()
           .state.walls.front()
           .mean_contact_line_tangent) {
    EXPECT_TRUE(std::isfinite(component));
  }
  EXPECT_EQ(
      contact_stages.front().stage_state_revision,
      acceptedContactStageRevision(
          previous_state_revision,
          endpoint_state_revision,
          contact_stages.front().geometry_revision.snapshot_revision_key,
          svmp::FE::Real{0.075},
          svmp::FE::Real{0.5},
          contact_stage_candidate.stage_solution,
          generalized_alpha_provenance));
  auto changed_stage_solution = contact_stage_candidate.stage_solution;
  ASSERT_FALSE(changed_stage_solution.empty());
  changed_stage_solution.back() += svmp::FE::Real{0.125};
  EXPECT_NE(
      contact_stages.front().stage_state_revision,
      acceptedContactStageRevision(
          previous_state_revision,
          endpoint_state_revision,
          contact_stages.front().geometry_revision.snapshot_revision_key,
          svmp::FE::Real{0.075},
          svmp::FE::Real{0.5},
          changed_stage_solution,
          generalized_alpha_provenance));
  auto changed_generalized_alpha_provenance =
      generalized_alpha_provenance;
  changed_generalized_alpha_provenance.dt = svmp::FE::Real{0.10};
  EXPECT_NE(
      contact_stages.front().stage_state_revision,
      acceptedContactStageRevision(
          previous_state_revision,
          endpoint_state_revision,
          contact_stages.front().geometry_revision.snapshot_revision_key,
          svmp::FE::Real{0.075},
          svmp::FE::Real{0.5},
          contact_stage_candidate.stage_solution,
          changed_generalized_alpha_provenance));
  const auto raw_endpoint_revision_before_accept =
      acceptance_order_history.u().valueRevision();
  acceptance_order_history.acceptStep(0.05);
  EXPECT_EQ(acceptance_order_history.stepIndex(), 2);
  EXPECT_DOUBLE_EQ(acceptance_order_history.time(), 0.10);
  EXPECT_NE(acceptance_order_history.u().valueRevision(),
            raw_endpoint_revision_before_accept);
  const auto accepted_callback_endpoint =
      gatherFeOrderedSolution(acceptance_order_history.u());
  EXPECT_EQ(accepted_callback_endpoint, finalized_endpoint_solution);
  const auto pre_maintenance_endpoint_state_revision =
      collectiveLevelSetMaintenanceAlgebraicRevision(
          accepted_callback_endpoint,
          activeFESystemCommunicator(*sim.fe_system));
  EXPECT_EQ(contact_stages.front().endpoint_state_revision,
            pre_maintenance_endpoint_state_revision);
  // These are the first accepted-callback operations: reject content drift,
  // then bind the preserved endpoint content and recompute the composite
  // stage hash after acceptStep has changed backend mutation counters.
  ASSERT_NO_THROW(
      assertAcceptedFreeSurfaceContactStageEndpointUnchanged(
          endpoint_state_revision,
          accepted_callback_endpoint,
          activeFESystemCommunicator(*sim.fe_system)));
  ASSERT_NO_THROW(
      bindAcceptedFreeSurfaceContactStagesToEndpointRevision(
          contact_stages,
          pre_maintenance_endpoint_state_revision,
          contact_stage_candidate.stage_solution,
          activeFESystemCommunicator(*sim.fe_system)));
  EXPECT_EQ(
      contact_stages.front().endpoint_state_revision,
      pre_maintenance_endpoint_state_revision);
  EXPECT_EQ(
      contact_stages.front().stage_state_revision,
      acceptedContactStageRevision(
          contact_stages.front().previous_state_revision,
          pre_maintenance_endpoint_state_revision,
          contact_stages.front()
              .geometry_revision.snapshot_revision_key,
          contact_stages.front().stage_time,
          contact_stages.front().stage_alpha_f,
          contact_stage_candidate.stage_solution,
          generalized_alpha_provenance));
  ASSERT_NO_THROW(
      assertAcceptedFreeSurfaceContactStageEndpointUnchanged(
          pre_maintenance_endpoint_state_revision,
          accepted_callback_endpoint,
          activeFESystemCommunicator(*sim.fe_system)));
  auto stale_endpoint_solution = endpoint_solution;
  ASSERT_FALSE(stale_endpoint_solution.empty());
  stale_endpoint_solution.back() += svmp::FE::Real{0.125};
  EXPECT_THROW(
      assertAcceptedFreeSurfaceContactStageEndpointUnchanged(
          pre_maintenance_endpoint_state_revision,
          stale_endpoint_solution,
          activeFESystemCommunicator(*sim.fe_system)),
      std::runtime_error);
  const auto endpoint_contact_stages =
      evaluateAcceptedFreeSurfaceContactStages(
          sim,
          svmp::FE::Real{0.10},
          svmp::FE::Real{1.0},
          previous_state_revision,
          pre_maintenance_endpoint_state_revision,
          endpoint_solution);
  ASSERT_EQ(endpoint_contact_stages.size(), 1u);
  ASSERT_EQ(endpoint_contact_stages.front().state.walls.size(), 1u);
  EXPECT_NEAR(
      endpoint_contact_stages.front()
          .state.walls.front()
          .mean_contact_position[0],
      svmp::FE::Real{0.60},
      1.0e-12);
  const auto endpoint_snapshot_revision =
      sim.fe_system->cutIntegrationContext()
          ->freeSurfaceGeometrySnapshotRevisionForMarker(interface_marker);
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

  const auto* context_before_nonendpoint_rejection =
      sim.fe_system->cutIntegrationContext();
  ASSERT_NE(context_before_nonendpoint_rejection, nullptr);
  const auto context_revision_before_nonendpoint_rejection =
      context_before_nonendpoint_rejection
          ->freeSurfaceGeometrySnapshotRevisionForMarker(interface_marker);
  const auto lifecycle_revision_before_nonendpoint_rejection =
      lifecycle.valueRevision();
  const auto endpoint_vector_revision_before_nonendpoint_rejection =
      time_history.u().valueRevision();
  const auto previous_vector_revision_before_nonendpoint_rejection =
      time_history.uPrev().valueRevision();
  const auto older_vector_revision_before_nonendpoint_rejection =
      time_history.uPrev2().valueRevision();
  const auto rate_before_nonendpoint_rejection =
      gatherFeOrderedSolution(time_history.uDot());
  const auto rate_vector_revision_before_nonendpoint_rejection =
      time_history.uDot().valueRevision();
  const auto acceleration_before_nonendpoint_rejection =
      gatherFeOrderedSolution(time_history.uDDot());
  const auto acceleration_vector_revision_before_nonendpoint_rejection =
      time_history.uDDot().valueRevision();
  const auto request_schedule_before_nonendpoint_rejection =
      canonicalLevelSetMaintenanceRequestSchedule(
          maintenance_requests,
          LevelSetMaintenanceScheduleStage::AcceptedEndpointPostStep,
          time_history.stepIndex());
  ASSERT_TRUE(request_schedule_before_nonendpoint_rejection.supported);
  const auto* mesh_phi_before_nonendpoint_rejection =
      static_cast<const double*>(mesh->field_data(mesh_field));
  ASSERT_NE(mesh_phi_before_nonendpoint_rejection, nullptr);
  const auto rejection_mesh_phi_count =
      mesh->field_components(mesh_field) *
      mesh->field_entity_count(mesh_field);
  const std::vector<double> mesh_phi_values_before_nonendpoint_rejection(
      mesh_phi_before_nonendpoint_rejection,
      mesh_phi_before_nonendpoint_rejection + rejection_mesh_phi_count);

  auto rejected_candidate_solution = endpoint_solution;
  std::string direct_nonendpoint_failure;
  try {
    (void)stageLevelSetProjectionReinitialization(
        sim,
        time_history,
        maintenance_request,
        phi,
        svmp::FE::Real{0.10},
        rejected_candidate_solution,
        contact_stages,
        contact_stage_constraints,
        contact_stage_candidate.stage_solution);
  } catch (const std::runtime_error& error) {
    direct_nonendpoint_failure = error.what();
  }
  EXPECT_NE(
      direct_nonendpoint_failure.find(
          "cannot apply a non-endpoint generalized-alpha stage repair"),
      std::string::npos)
      << direct_nonendpoint_failure;
  EXPECT_NE(
      direct_nonendpoint_failure.find("history/rate publication"),
      std::string::npos)
      << direct_nonendpoint_failure;
  EXPECT_EQ(rejected_candidate_solution, endpoint_solution);

  auto missing_stage_requests = maintenance_requests;
  EXPECT_THROW(
      (void)applyLevelSetMaintenance(
          sim, time_history, missing_stage_requests),
      std::runtime_error);
  EXPECT_EQ(gatherFeOrderedSolution(time_history.u()), endpoint_solution);
  EXPECT_EQ(gatherFeOrderedSolution(time_history.uPrev()), previous_solution);

  auto nonendpoint_stage_requests = maintenance_requests;
  std::string nonendpoint_stage_failure;
  try {
    (void)applyLevelSetMaintenance(
        sim,
        time_history,
        nonendpoint_stage_requests,
        contact_stages,
        contact_stage_constraints,
        contact_stage_candidate.stage_solution);
  } catch (const std::runtime_error& error) {
    nonendpoint_stage_failure = error.what();
  }
  EXPECT_NE(
      nonendpoint_stage_failure.find(
          "cannot apply a non-endpoint generalized-alpha stage repair"),
      std::string::npos)
      << nonendpoint_stage_failure;
  EXPECT_EQ(gatherFeOrderedSolution(time_history.u()), endpoint_solution);
  EXPECT_EQ(gatherFeOrderedSolution(time_history.uPrev()), previous_solution);
  EXPECT_EQ(gatherFeOrderedSolution(time_history.uPrev2()), previous_solution);
  EXPECT_EQ(
      gatherFeOrderedSolution(time_history.uDot()),
      rate_before_nonendpoint_rejection);
  EXPECT_EQ(
      gatherFeOrderedSolution(time_history.uDDot()),
      acceleration_before_nonendpoint_rejection);
  EXPECT_EQ(
      time_history.u().valueRevision(),
      endpoint_vector_revision_before_nonendpoint_rejection);
  EXPECT_EQ(
      time_history.uPrev().valueRevision(),
      previous_vector_revision_before_nonendpoint_rejection);
  EXPECT_EQ(
      time_history.uPrev2().valueRevision(),
      older_vector_revision_before_nonendpoint_rejection);
  EXPECT_EQ(
      time_history.uDot().valueRevision(),
      rate_vector_revision_before_nonendpoint_rejection);
  EXPECT_EQ(
      time_history.uDDot().valueRevision(),
      acceleration_vector_revision_before_nonendpoint_rejection);
  const auto request_schedule_after_nonendpoint_rejection =
      canonicalLevelSetMaintenanceRequestSchedule(
          nonendpoint_stage_requests,
          LevelSetMaintenanceScheduleStage::AcceptedEndpointPostStep,
          time_history.stepIndex());
  EXPECT_EQ(
      request_schedule_after_nonendpoint_rejection.words,
      request_schedule_before_nonendpoint_rejection.words);
  EXPECT_EQ(
      sim.fe_system->cutIntegrationContext(),
      context_before_nonendpoint_rejection);
  EXPECT_EQ(
      sim.fe_system->cutIntegrationContext()
          ->freeSurfaceGeometrySnapshotRevisionForMarker(interface_marker),
      context_revision_before_nonendpoint_rejection);
  EXPECT_EQ(
      lifecycle.valueRevision(),
      lifecycle_revision_before_nonendpoint_rejection);
  const auto* mesh_phi_after_nonendpoint_rejection =
      static_cast<const double*>(mesh->field_data(mesh_field));
  ASSERT_NE(mesh_phi_after_nonendpoint_rejection, nullptr);
  EXPECT_EQ(
      std::vector<double>(
          mesh_phi_after_nonendpoint_rejection,
          mesh_phi_after_nonendpoint_rejection + rejection_mesh_phi_count),
      mesh_phi_values_before_nonendpoint_rejection);

  auto endpoint_maintenance_stage_candidate =
      buildAcceptedFreeSurfaceContactStageCandidate(
          sim,
          *params,
          lifecycle,
          contact_stage_refresh_cache,
          active_cut_requests,
          svmp::FE::Real{0.10},
          svmp::FE::Real{1.0},
          previous_state_revision,
          pre_maintenance_endpoint_state_revision,
          previous_solution,
          endpoint_solution,
          nullptr);
  auto& endpoint_maintenance_stages =
      endpoint_maintenance_stage_candidate.stages;
  const auto& endpoint_maintenance_constraints =
      endpoint_maintenance_stage_candidate.constraints;
  ASSERT_EQ(endpoint_maintenance_stages.size(), 1u);
  ASSERT_EQ(endpoint_maintenance_stages.front().state.walls.size(), 1u);
  EXPECT_EQ(
      endpoint_maintenance_stage_candidate.stage_solution,
      endpoint_solution);
  EXPECT_DOUBLE_EQ(
      endpoint_maintenance_stages.front().stage_time,
      svmp::FE::Real{0.10});
  EXPECT_DOUBLE_EQ(
      endpoint_maintenance_stages.front().stage_alpha_f,
      svmp::FE::Real{1.0});
  EXPECT_NEAR(
      endpoint_maintenance_stages.front()
          .state.walls.front()
          .mean_contact_position[0],
      svmp::FE::Real{0.60},
      1.0e-12);

  auto candidate_only_solution = endpoint_solution;
  const auto candidate_only_reinitialization =
      stageLevelSetProjectionReinitialization(
          sim,
          time_history,
          maintenance_request,
          phi,
          svmp::FE::Real{0.10},
          candidate_only_solution,
          endpoint_maintenance_stages,
          endpoint_maintenance_constraints,
          endpoint_maintenance_stage_candidate.stage_solution);
  EXPECT_TRUE(candidate_only_reinitialization.applied);
  EXPECT_TRUE(candidate_only_reinitialization.repair.converged);
  EXPECT_EQ(candidate_only_reinitialization.repair.wall_contact_constraints,
            1u);
  EXPECT_DOUBLE_EQ(
      candidate_only_reinitialization.repair.max_contact_line_displacement,
      svmp::FE::Real{0.0});
  EXPECT_DOUBLE_EQ(
      candidate_only_reinitialization.repair
          .max_contact_angle_change_radians,
      svmp::FE::Real{0.0});

  testing::internal::CaptureStdout();
  const bool maintenance_changed = applyLevelSetMaintenance(
      sim,
      time_history,
      maintenance_requests,
      endpoint_maintenance_stages,
      endpoint_maintenance_constraints,
      endpoint_maintenance_stage_candidate.stage_solution);
  const auto maintenance_output = testing::internal::GetCapturedStdout();
  ASSERT_TRUE(maintenance_changed);
  EXPECT_NE(maintenance_output.find(
                "wall_contact_model=accepted_dynamic_stage"),
            std::string::npos);
  EXPECT_NE(maintenance_output.find("wall_contact_constraints=1"),
            std::string::npos);
  EXPECT_NE(maintenance_output.find("max_contact_line_displacement=0"),
            std::string::npos);
  EXPECT_NE(maintenance_output.find("max_contact_angle_change_radians=0"),
            std::string::npos);
  EXPECT_NE(maintenance_output.find("accepted_contact_stage_alpha_f=1"),
            std::string::npos);

  const auto endpoint_after = gatherFeOrderedSolution(time_history.u());
  const auto previous_after = gatherFeOrderedSolution(time_history.uPrev());
  ASSERT_EQ(endpoint_after.size(), endpoint_solution.size());
  EXPECT_EQ(endpoint_after, candidate_only_solution);
  EXPECT_EQ(
      gatherFeOrderedSolution(time_history.uDot()),
      rate_before_nonendpoint_rejection);
  EXPECT_EQ(
      time_history.uDot().valueRevision(),
      rate_vector_revision_before_nonendpoint_rejection);
  EXPECT_EQ(
      gatherFeOrderedSolution(time_history.uDDot()),
      acceleration_before_nonendpoint_rejection);
  EXPECT_EQ(
      time_history.uDDot().valueRevision(),
      acceleration_vector_revision_before_nonendpoint_rejection);
  for (std::size_t i = 0; i < phi_dof_count; ++i) {
    const auto index = phi_offset + i;
    EXPECT_NEAR(endpoint_after[index] - endpoint_solution[index],
                previous_after[index] - previous_solution[index],
                1.0e-12);
  }
  ASSERT_NO_THROW((void)refreshActiveCutIntegrationContextFromSolution(
      sim,
      *params,
      std::span<const svmp::FE::Real>(endpoint_after.data(),
                                      endpoint_after.size()),
      lifecycle,
      "application-driver-wall-aware-maintenance-test"));
  const auto* maintained_context = sim.fe_system->cutIntegrationContext();
  ASSERT_NE(maintained_context, nullptr);
  const auto maintained_snapshot_revision =
      maintained_context->freeSurfaceGeometrySnapshotRevisionForMarker(
          interface_marker);
  EXPECT_NE(maintained_snapshot_revision,
            snapshot->revision().snapshot_revision_key);
  EXPECT_NE(maintained_snapshot_revision, endpoint_snapshot_revision);

  const auto accepted_state_revision =
      collectiveLevelSetMaintenanceAlgebraicRevision(
          endpoint_after,
          activeFESystemCommunicator(*sim.fe_system));
  const auto maintained_endpoint_stages =
      evaluateAcceptedFreeSurfaceContactStages(
          sim,
          svmp::FE::Real{0.10},
          svmp::FE::Real{1.0},
          previous_state_revision,
          accepted_state_revision,
          endpoint_after);
  ASSERT_EQ(maintained_endpoint_stages.size(), 1u);
  ASSERT_EQ(maintained_endpoint_stages.front().state.walls.size(), 1u);
  EXPECT_NEAR(
      maintained_endpoint_stages.front()
          .state.walls.front()
          .mean_contact_position[0],
      svmp::FE::Real{0.60},
      1.0e-12);
  ASSERT_TRUE(
      endpoint_contact_stages.front()
          .state.walls.front()
          .mean_dynamic_angle_radians.has_value());
  ASSERT_TRUE(
      maintained_endpoint_stages.front()
          .state.walls.front()
          .mean_dynamic_angle_radians.has_value());
  EXPECT_NEAR(
      *maintained_endpoint_stages.front()
           .state.walls.front()
           .mean_dynamic_angle_radians,
      *endpoint_contact_stages.front()
           .state.walls.front()
           .mean_dynamic_angle_radians,
      1.0e-12);
  ASSERT_NO_THROW(recordAcceptedFreeSurfaceDiscreteFunctionals(
      sim,
      /*accepted_step=*/2u,
      svmp::FE::Real{0.10},
      svmp::FE::Real{0.05},
      pre_maintenance_endpoint_state_revision,
      accepted_state_revision,
      endpoint_maintenance_stages));
  const auto history =
      sim.fe_system->freeSurfaceDiscreteFunctionalHistory();
  ASSERT_EQ(history.size(), 1u);
  EXPECT_EQ(
      history.front().pre_maintenance_endpoint_state_revision,
      pre_maintenance_endpoint_state_revision);
  EXPECT_NE(history.front().pre_maintenance_endpoint_state_revision,
            history.front().state_revision);
  EXPECT_EQ(history.front().state_revision, accepted_state_revision);
  ASSERT_TRUE(history.front().contact_stage.has_value());
  EXPECT_EQ(
      history.front().contact_stage->endpoint_state_revision,
      history.front().pre_maintenance_endpoint_state_revision);
  EXPECT_DOUBLE_EQ(history.front().contact_stage->stage_time,
                   svmp::FE::Real{0.10});
  EXPECT_DOUBLE_EQ(history.front().contact_stage->stage_alpha_f,
                   svmp::FE::Real{1.0});
  EXPECT_EQ(history.front().contact_stage->geometry_revision
                .snapshot_revision_key,
            endpoint_maintenance_stages.front()
                .geometry_revision.snapshot_revision_key);
  EXPECT_TRUE(history.front().contact_line_kinematics.empty());

  auto next_solution = endpoint_after;
  std::vector<svmp::FE::Real> next_phi_values(
      mesh->n_vertices(), svmp::FE::Real{0.0});
  for (std::size_t vertex = 0u; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    next_phi_values[vertex] =
        svmp::FE::Real{2.0} *
        (point[0] - svmp::FE::Real{0.65});
  }
  const auto next_phi_coefficients = projectWorkflowVertexValues(
      *sim.fe_system,
      phi,
      next_phi_values,
      1u,
      "ApplicationDriver next accepted contact-stage phi");
  writeWorkflowFieldSlice(
      *sim.fe_system, phi, next_phi_coefficients, next_solution);
  ASSERT_NO_THROW((void)refreshActiveCutIntegrationContextFromSolution(
      sim,
      *params,
      std::span<const svmp::FE::Real>(next_solution.data(),
                                      next_solution.size()),
      lifecycle,
      "application-driver-next-contact-endpoint-test"));
  const auto next_state_revision =
      collectiveLevelSetMaintenanceAlgebraicRevision(
          next_solution,
          activeFESystemCommunicator(*sim.fe_system));
  const auto next_contact_stages =
      evaluateAcceptedFreeSurfaceContactStages(
          sim,
          svmp::FE::Real{0.15},
          svmp::FE::Real{1.0},
          accepted_state_revision,
          next_state_revision,
          next_solution);
  ASSERT_EQ(next_contact_stages.size(), 1u);
  ASSERT_EQ(next_contact_stages.front().state.walls.size(), 1u);
  EXPECT_NEAR(
      next_contact_stages.front()
          .state.walls.front()
          .mean_contact_position[0],
      svmp::FE::Real{0.65},
      1.0e-12);

  testing::internal::CaptureStdout();
  ASSERT_NO_THROW(recordAcceptedFreeSurfaceDiscreteFunctionals(
      sim,
      /*accepted_step=*/3u,
      svmp::FE::Real{0.15},
      svmp::FE::Real{0.05},
      next_state_revision,
      next_state_revision,
      next_contact_stages));
  const auto contact_kinematics_output =
      testing::internal::GetCapturedStdout();
  EXPECT_NE(
      contact_kinematics_output.find(
          "contact_geometric_kinematics_available=true"),
      std::string::npos);
  EXPECT_NE(
      contact_kinematics_output.find(
          "contact_projected_centroid_speed="),
      std::string::npos);
  EXPECT_NE(
      contact_kinematics_output.find(
          "contact_fluid_minus_geometric_speed="),
      std::string::npos);

  const auto kinematics_history =
      sim.fe_system->freeSurfaceDiscreteFunctionalHistory();
  ASSERT_EQ(kinematics_history.size(), 2u);
  EXPECT_TRUE(
      kinematics_history.front().contact_line_kinematics.empty());
  ASSERT_TRUE(kinematics_history.back().contact_stage.has_value());
  ASSERT_EQ(
      kinematics_history.back().contact_line_kinematics.size(), 1u);
  const auto& kinematics =
      kinematics_history.back().contact_line_kinematics.front();
  const auto& previous_contact_wall =
      kinematics_history.front()
          .contact_stage->state.walls.front();
  const auto& current_contact_wall =
      kinematics_history.back()
          .contact_stage->state.walls.front();
  EXPECT_EQ(kinematics.boundary_marker, wall_marker);
  EXPECT_EQ(kinematics.previous_accepted_step, 2u);
  EXPECT_DOUBLE_EQ(
      kinematics.previous_accepted_time,
      svmp::FE::Real{0.10});
  EXPECT_DOUBLE_EQ(
      kinematics.previous_stage_time,
      svmp::FE::Real{0.10});
  EXPECT_EQ(
      kinematics.previous_stage_state_revision,
      kinematics_history.front()
          .contact_stage->stage_state_revision);
  EXPECT_EQ(
      kinematics.previous_snapshot_revision_key,
      kinematics_history.front()
          .contact_stage->geometry_revision.snapshot_revision_key);
  EXPECT_DOUBLE_EQ(
      kinematics.stage_time_interval,
      svmp::FE::Real{0.05});
  svmp::FE::Real expected_projected_displacement = 0.0;
  svmp::FE::Real projection_norm_squared = 0.0;
  for (std::size_t component = 0u; component < 3u; ++component) {
    EXPECT_DOUBLE_EQ(
        kinematics.previous_mean_contact_position[component],
        previous_contact_wall.mean_contact_position[component]);
    expected_projected_displacement +=
        (current_contact_wall.mean_contact_position[component] -
         previous_contact_wall.mean_contact_position[component]) *
        kinematics.projection_direction[component];
    projection_norm_squared +=
        kinematics.projection_direction[component] *
        kinematics.projection_direction[component];
  }
  EXPECT_NEAR(std::sqrt(projection_norm_squared), 1.0, 1.0e-13);
  EXPECT_NEAR(
      kinematics.projected_contact_centroid_speed,
      expected_projected_displacement /
          kinematics.stage_time_interval,
      1.0e-12);
  ASSERT_TRUE(current_contact_wall.mean_contact_speed.has_value());
  EXPECT_DOUBLE_EQ(
      kinematics.mean_fluid_contact_speed,
      *current_contact_wall.mean_contact_speed);
  EXPECT_NEAR(
      kinematics.fluid_minus_geometric_contact_speed,
      kinematics.mean_fluid_contact_speed -
          kinematics.projected_contact_centroid_speed,
      1.0e-12);
  ASSERT_NO_THROW(recordAcceptedFreeSurfaceDiscreteFunctionals(
      sim,
      /*accepted_step=*/3u,
      svmp::FE::Real{0.15},
      svmp::FE::Real{0.05},
      next_state_revision,
      next_state_revision,
      next_contact_stages));
  EXPECT_EQ(
      sim.fe_system->freeSurfaceDiscreteFunctionalHistory().size(), 2u);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     AcceptedSnapshotPrescribedFrameIsCompleteAndFailsClosed)
{
  constexpr int interface_marker = 1706;
  constexpr int wall_marker = 129;
  constexpr std::uint64_t revision = 41u;
  constexpr svmp::FE::GlobalIndex parent = 17;
  constexpr svmp::FE::Real half_pi =
      svmp::FE::Real{1.57079632679489661923132169163975144};
  constexpr svmp::FE::Real inv_sqrt_two =
      svmp::FE::Real{0.70710678118654752440084436210484904};

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
          .physical_point = {{0.25, 0.0, 0.0}},
          .physical_weight = 1.0,
          .normal = {{inv_sqrt_two, inv_sqrt_two, 0.0}},
          .boundary_normal = {{0.0, -1.0, 0.0}},
      });

  svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalParameters parameters;
  parameters.liquid_side =
      svmp::FE::geometry::CutIntegrationSide::Negative;
  parameters.young_wall_coefficients.push_back(
      svmp::FE::interfaces::FreeSurfaceYoungWallCoefficient{
          .boundary_marker = wall_marker,
          .equilibrium_contact_angle_radians = half_pi,
      });
  const auto constraint = makeAcceptedSnapshotWallConstraint(
      record,
      svmp::FE::level_set::LevelSetWallContactConstraintKind::
          PrescribedAngle,
      interface_marker,
      revision,
      parameters,
      /*dimension=*/2);
  EXPECT_EQ(constraint.parent_cell_global_id, parent);
  EXPECT_DOUBLE_EQ(constraint.target_angle_radians, half_pi);
  EXPECT_EQ(constraint.physical_wall_normal,
            (std::array<svmp::FE::Real, 3>{{0.0, -1.0, 0.0}}));
  EXPECT_EQ(constraint.accepted_contact_point,
            (std::array<svmp::FE::Real, 3>{{0.25, 0.0, 0.0}}));
  EXPECT_EQ(constraint.accepted_contact_line_tangent,
            (std::array<svmp::FE::Real, 3>{{0.0, 0.0, 1.0}}));

  const std::vector duplicates{constraint, constraint};
  const auto canonical = canonicalizeAcceptedWallConstraints(
      duplicates,
      svmp::MeshComm::world(),
      "Application prescribed-frame test");
  ASSERT_EQ(canonical.size(), 1u);
  auto conflict = constraint;
  conflict.accepted_contact_point[0] += svmp::FE::Real{0.125};
  EXPECT_THROW(
      (void)canonicalizeAcceptedWallConstraints(
          std::vector{constraint, conflict},
          svmp::MeshComm::world(),
          "Application prescribed-frame conflict test"),
      std::runtime_error);

  auto missing = record;
  missing.physical_rule.points.clear();
  EXPECT_THROW(
      (void)makeAcceptedSnapshotWallConstraint(
          missing,
          svmp::FE::level_set::LevelSetWallContactConstraintKind::
              PrescribedAngle,
          interface_marker,
          revision,
          parameters,
          /*dimension=*/2),
      std::runtime_error);
  auto stale = record;
  stale.physical_rule.free_surface_snapshot_revision_key = revision + 1u;
  EXPECT_THROW(
      (void)makeAcceptedSnapshotWallConstraint(
          stale,
          svmp::FE::level_set::LevelSetWallContactConstraintKind::
              PrescribedAngle,
          interface_marker,
          revision,
          parameters,
          /*dimension=*/2),
      std::runtime_error);
}

TEST(ApplicationDriverLevelSetWorkflows,
     PrescribedWallSnapshotDrivesEndpointReinitialization)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr int wall_marker = 29;
  constexpr int interface_marker = 706;
  auto mesh = makeWorkflowQuadPatch2x2Mesh();
  for (const auto face : mesh->local_mesh().boundary_faces()) {
    mesh->local_mesh().set_boundary_label(face, wall_marker);
  }
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
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1});
  svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalParameters parameters;
  parameters.liquid_side =
      svmp::FE::geometry::CutIntegrationSide::Negative;
  parameters.surface_tension = svmp::FE::Real{1.0};
  parameters.young_wall_coefficients.push_back(
      svmp::FE::interfaces::FreeSurfaceYoungWallCoefficient{
          .boundary_marker = wall_marker,
          .equilibrium_contact_angle_radians =
              svmp::FE::Real{1.57079632679489661923132169163975144},
      });
  system->declareFreeSurfaceDiscreteFunctional(
      svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration{
          .interface_marker = interface_marker,
          .level_set_field = phi,
          .geometry_domain_id = "prescribed_wall_maintenance",
          .parameters = parameters,
          .owner_component =
              "ApplicationDriverLevelSetWorkflows.PrescribedWallFixture",
      });
  svmp::FE::interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey key{};
  key.source =
      svmp::FE::interfaces::LevelSetInterfaceSource::fromField(phi);
  key.domain_id = "prescribed_wall_maintenance";
  key.isovalue = 0.0;
  key.interface_marker = interface_marker;
  key.boundary_marker = wall_marker;
  system->registerGeneratedEmbeddedInterfaceMarker(
      svmp::FE::interfaces::
          stableGeneratedInterfaceBoundaryIntersectionMarker(key));
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> vertex_values(mesh->n_vertices(), 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    vertex_values[vertex] =
        svmp::FE::Real{2.0} *
        (workflowVertexPoint(*mesh, vertex)[0] - svmp::FE::Real{0.8});
  }
  const auto coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      vertex_values,
      /*components=*/1u,
      "ApplicationDriver prescribed-wall maintenance phi");
  std::vector<svmp::FE::Real> solution(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, coefficients, solution);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>prescribed_wall_maintenance</Generated_interface_domain_id>
      <Interface_marker>706</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
  ASSERT_NO_THROW((void)refreshActiveCutIntegrationContextFromSolution(
      sim,
      *params,
      std::span<const svmp::FE::Real>(solution.data(), solution.size()),
      lifecycle,
      "application-driver-prescribed-wall-maintenance-test"));

  auto factory = svmp::FE::backends::BackendFactory::create(
      svmp::FE::backends::BackendKind::FSILS);
  ASSERT_NE(factory, nullptr);
  auto history = svmp::FE::timestepping::TimeHistory::allocate(
      *factory, sim.fe_system->dofHandler().getNumDofs());
  history.setTime(0.1);
  history.setDt(0.05);
  history.setPrevDt(0.05);
  history.setStepIndex(1);
  scatterFeOrderedSolution(history.u(), solution);
  scatterFeOrderedSolution(history.uPrev(), solution);
  scatterFeOrderedSolution(history.uPrev2(), solution);

  LevelSetMaintenanceRequest request{};
  request.level_set_field_name = "phi";
  request.reinitialization.enabled = true;
  request.reinitialization.cadence_steps = 1;
  request.reinitialization.max_iterations = 100;
  request.reinitialization.signed_distance_tolerance = 1.0e-10;
  std::vector<LevelSetMaintenanceRequest> requests{request};
  auto staged_solution = solution;
  const auto staged = stageLevelSetProjectionReinitialization(
      sim,
      history,
      request,
      phi,
      svmp::FE::Real{0.1},
      staged_solution,
      {},
      {},
      {});
  ASSERT_TRUE(staged.applied);
  ASSERT_TRUE(staged.repair.converged);
  ASSERT_EQ(staged.wall_context.local_constraints.size(), 2u);
  EXPECT_EQ(staged.wall_context.global_prescribed_contact_rules, 2u);
  for (const auto& constraint : staged.wall_context.local_constraints) {
    EXPECT_DOUBLE_EQ(
        constraint.target_angle_radians,
        svmp::FE::Real{1.57079632679489661923132169163975144});
    EXPECT_TRUE(acceptedWallConstraintFrameIsComplete(
        constraint,
        parameters,
        /*dimension=*/2));
    EXPECT_NE(std::abs(constraint.accepted_contact_line_tangent[2]), 0.0);
  }
  EXPECT_LE(staged.repair.max_prescribed_contact_value_residual,
            svmp::FE::Real{1.0e-9});
  EXPECT_LE(staged.repair.max_prescribed_contact_angle_error_radians,
            svmp::FE::Real{1.0e-12});
  EXPECT_DOUBLE_EQ(staged.repair.max_contact_line_displacement,
                   svmp::FE::Real{0.0});
  ::testing::Test::RecordProperty(
      "application_prescribed_target_angle_max_error_degrees",
      staged.repair.max_prescribed_contact_angle_error_radians *
          svmp::FE::Real{180.0} /
          std::acos(svmp::FE::Real{-1.0}));
  ::testing::Test::RecordProperty(
      "application_prescribed_target_contact_displacement_max",
      staged.repair.max_contact_line_displacement);
  testing::internal::CaptureStdout();
  const bool changed = applyLevelSetMaintenance(sim, history, requests);
  const auto output = testing::internal::GetCapturedStdout();
  ASSERT_TRUE(changed);
  EXPECT_NE(output.find("wall_contact_model=prescribed_angle"),
            std::string::npos);
  EXPECT_NE(output.find("prescribed_contact_rules=2"),
            std::string::npos);
  EXPECT_NE(output.find("dynamic_contact_rules=0"), std::string::npos);
  EXPECT_NE(output.find("max_prescribed_contact_value_residual="),
            std::string::npos);
  EXPECT_NE(output.find("max_prescribed_contact_angle_error_radians="),
            std::string::npos);
  EXPECT_NE(output.find("max_contact_line_displacement=0"),
            std::string::npos);
  EXPECT_NE(output.find("max_contact_angle_change_radians=0"),
            std::string::npos);

  const auto repaired = gatherFeOrderedSolution(history.u());
  const auto field_offset =
      static_cast<std::size_t>(sim.fe_system->fieldDofOffset(phi));
  const auto field_dofs = static_cast<std::size_t>(
      sim.fe_system->fieldDofHandler(phi).getNumDofs());
  ASSERT_LE(field_offset + field_dofs, repaired.size());
  for (std::size_t i = 0; i < field_dofs; ++i) {
    EXPECT_NEAR(repaired[field_offset + i],
                svmp::FE::Real{0.5} * solution[field_offset + i],
                1.0e-10);
  }
  mesh->event_bus().notify(svmp::MeshEvent::GeometryChanged);
  EXPECT_THROW(
      (void)resolveLevelSetWallAwareMaintenanceContext(
          sim,
          history,
          phi,
          svmp::FE::Real{0.1},
          {},
          {}),
      std::runtime_error);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     ActiveOnlyRetentionRejectsInactiveCutVolumeConsumerWithoutRules)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowBiquadraticQuadMesh();
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
      /*order=*/2);

  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1});
  system->addCutVolumeKernel(
      "equations",
      704,
      svmp::FE::geometry::CutIntegrationSide::Positive,
      phi,
      std::make_shared<WorkflowNoOpCellKernel>());
  ASSERT_NO_THROW(system->setup({}));
  EXPECT_EQ(system->cutVolumeKernelCount(
                704, svmp::FE::geometry::CutIntegrationSide::Positive),
            1u);

  std::vector<svmp::FE::Real> phi_vertex_values(mesh->n_vertices(), 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    phi_vertex_values[vertex] = workflowPhi(*mesh, vertex);
  }
  const auto phi_coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      std::span<const svmp::FE::Real>(phi_vertex_values.data(),
                                      phi_vertex_values.size()),
      1u,
      "ApplicationDriver active-only cut-retention audit phi");

  std::vector<svmp::FE::Real> solution(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, phi_coefficients, solution);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);

  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>active_only_interface</Generated_interface_domain_id>
      <Interface_marker>704</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
      <Small_cut_aggregation>false</Small_cut_aggregation>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
  const auto requests = application::core::activeCutVolumeRequests(*params);
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests.front().volume_retention,
            application::core::ActiveCutVolumeRetention::ActiveOnly);

  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
  try {
    (void)refreshActiveCutIntegrationContextFromSolution(
        sim,
        *params,
        std::span<const svmp::FE::Real>(solution.data(), solution.size()),
        lifecycle,
        "application-driver-active-only-cut-retention-audit-test");
    FAIL() << "Expected inactive-side cut-volume consumer diagnostic";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("Generated cut-volume consumer has no retained "
                           "quadrature rules"),
              std::string::npos);
    EXPECT_NE(message.find("marker=704"), std::string::npos);
    EXPECT_NE(message.find("logical_side=inactive"), std::string::npos);
    EXPECT_NE(message.find("cut_volume_side=Positive"), std::string::npos);
    EXPECT_NE(message.find("retained_volume_sides=active_only"),
              std::string::npos);
  }
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     VelocityExtensionProjectsHierarchicalTargetCoefficients)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowBiquadraticQuadMesh();
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/2,
      svmp::FE::BasisType::Hierarchical);
  auto vector_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, 2);

  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1});
  const auto source_velocity = system->addField(svmp::FE::systems::FieldSpec{
      .name = "Velocity",
      .space = vector_space,
      .components = 2});
  const auto target_velocity = system->addField(
      svmp::FE::systems::FieldSpec{
          .name = "LevelSetAdvectionVelocity",
          .space = vector_space,
          .components = 2,
          .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData});
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_vertex_values(mesh->n_vertices(), 0.0);
  std::vector<svmp::FE::Real> source_vertex_values(mesh->n_vertices() * 2u, 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    phi_vertex_values[vertex] = workflowPhi(*mesh, vertex);
    const auto velocity = workflowVelocity(*mesh, vertex);
    source_vertex_values[2u * vertex] = velocity[0];
    source_vertex_values[2u * vertex + 1u] = velocity[1];
  }

  const auto phi_coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      std::span<const svmp::FE::Real>(phi_vertex_values.data(),
                                      phi_vertex_values.size()),
      1u,
      "ApplicationDriver velocity extension hierarchical phi");
  const auto source_coefficients = projectWorkflowVertexValues(
      *system,
      source_velocity,
      std::span<const svmp::FE::Real>(source_vertex_values.data(),
                                      source_vertex_values.size()),
      2u,
      "ApplicationDriver velocity extension hierarchical source velocity");

  std::vector<svmp::FE::Real> solution(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, phi_coefficients, solution);
  writeWorkflowFieldSlice(*system,
                          source_velocity,
                          source_coefficients,
                          solution);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);

  svmp::FE::systems::SystemStateView state{};
  state.u = std::span<const svmp::FE::Real>(solution.data(), solution.size());

  LevelSetAdvectionVelocityRequest request{};
  request.level_set_field_name = "phi";
  request.source_velocity_field_name = "Velocity";
  request.target_velocity_field_name = "LevelSetAdvectionVelocity";
  request.extension_method = "nearest_active_vertex";
  request.active_side = application::core::LevelSetActiveSide::Negative;
  request.isovalue = 0.0;

  EXPECT_TRUE(updateLevelSetAdvectionVelocitiesFromState(
      sim,
      state,
      std::vector<LevelSetAdvectionVelocityRequest>{request}));

  const auto prescribed =
      sim.fe_system->prescribedFieldCoefficients(target_velocity);
  ASSERT_FALSE(prescribed.empty());

  std::vector<std::size_t> active_vertices;
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    if (phi_vertex_values[vertex] <= 0.0) {
      active_vertices.push_back(vertex);
    }
  }
  ASSERT_FALSE(active_vertices.empty());

  auto nearest_active_vertex = [&](std::size_t vertex) {
    if (phi_vertex_values[vertex] <= 0.0) {
      return vertex;
    }
    const auto point = workflowVertexPoint(*mesh, vertex);
    std::size_t best = active_vertices.front();
    svmp::FE::Real best_distance2 =
        std::numeric_limits<svmp::FE::Real>::infinity();
    for (const auto candidate : active_vertices) {
      const auto candidate_point = workflowVertexPoint(*mesh, candidate);
      svmp::FE::Real distance2 = 0.0;
      for (std::size_t d = 0; d < 2u; ++d) {
        const auto delta = point[d] - candidate_point[d];
        distance2 += delta * delta;
      }
      if (distance2 < best_distance2) {
        best_distance2 = distance2;
        best = candidate;
      }
    }
    return best;
  };

  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto expected_source = nearest_active_vertex(vertex);
    const auto expected = workflowVelocity(*mesh, expected_source);
    const auto point = workflowVertexPoint(*mesh, vertex);
    const auto value = sim.fe_system->evaluateFieldAtPoint(
        target_velocity,
        svmp::FE::systems::SystemStateView{},
        point);
    ASSERT_TRUE(value.has_value()) << "vertex " << vertex;
    EXPECT_NEAR((*value)[0], expected[0], 1.0e-10) << "vertex " << vertex;
    EXPECT_NEAR((*value)[1], expected[1], 1.0e-10) << "vertex " << vertex;
  }
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     WallCompatibleNormalVelocityExtensionMatchesGeneratedInterfaceTraceAndProjectsOuterDryWalls)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowThreeQuadStripMesh();
  constexpr svmp::label_t kWallLabel = 4343;
  constexpr int kInterfaceMarker = 706;
  mesh->register_label("wall", kWallLabel);
  for (const auto face : mesh->local_mesh().boundary_faces()) {
    mesh->set_boundary_label(face, kWallLabel);
  }
  const auto mesh_phi_field = svmp::MeshFields::attach_field(
      mesh->local_mesh(),
      svmp::EntityKind::Vertex,
      "phi",
      svmp::FieldScalarType::Float64,
      1);
  ASSERT_NE(svmp::MeshFields::field_data_as<svmp::real_t>(
                mesh->local_mesh(), mesh_phi_field),
            nullptr);
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/1);
  auto vector_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, 2);

  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1});
  const auto source_velocity = system->addField(svmp::FE::systems::FieldSpec{
      .name = "Velocity",
      .space = vector_space,
      .components = 2});
  const auto target_velocity = system->addField(
      svmp::FE::systems::FieldSpec{
          .name = "LevelSetAdvectionVelocity",
          .space = vector_space,
          .components = 2,
          .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData});
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_vertex_values(mesh->n_vertices(), 0.0);
  std::vector<svmp::FE::Real> source_vertex_values(mesh->n_vertices() * 2u, 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    phi_vertex_values[vertex] = workflowPhi(*mesh, vertex);
    const auto velocity = workflowVelocity(*mesh, vertex);
    source_vertex_values[2u * vertex] = velocity[0];
    source_vertex_values[2u * vertex + 1u] = velocity[1];
  }

  const auto phi_coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      std::span<const svmp::FE::Real>(phi_vertex_values.data(),
                                      phi_vertex_values.size()),
      1u,
      "ApplicationDriver nearest-interface extension phi");
  const auto source_coefficients = projectWorkflowVertexValues(
      *system,
      source_velocity,
      std::span<const svmp::FE::Real>(source_vertex_values.data(),
                                      source_vertex_values.size()),
      2u,
      "ApplicationDriver nearest-interface extension source velocity");

  std::vector<svmp::FE::Real> solution(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, phi_coefficients, solution);
  writeWorkflowFieldSlice(*system,
                          source_velocity,
                          source_coefficients,
                          solution);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);

  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>trace_support_interface</Generated_interface_domain_id>
      <Interface_marker>706</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
  const auto cut_report = refreshActiveCutIntegrationContextFromSolution(
      sim,
      *params,
      std::span<const svmp::FE::Real>(solution.data(), solution.size()),
      lifecycle,
      "application-driver-p1-trace-support-test");
  ASSERT_TRUE(cut_report.refreshed);
  ASSERT_GT(cut_report.interface_fragments, 0u);

  svmp::FE::systems::SystemStateView state{};
  state.u = std::span<const svmp::FE::Real>(solution.data(), solution.size());

  LevelSetAdvectionVelocityRequest request{};
  request.level_set_field_name = "phi";
  request.source_velocity_field_name = "Velocity";
  request.target_velocity_field_name = "LevelSetAdvectionVelocity";
  request.extension_method = "wall_compatible_normal";
  request.wall_face_names = {"wall"};
  request.wall_constraints = {{.face_name = "wall"}};
  request.active_side = application::core::LevelSetActiveSide::Negative;
  request.isovalue = 0.0;
  request.requested_interface_marker = kInterfaceMarker;
  request.active_cut_request_index = 0u;

  EXPECT_TRUE(updateLevelSetAdvectionVelocitiesFromState(
      sim,
      state,
      std::vector<LevelSetAdvectionVelocityRequest>{request}));

  const auto active_vertex = std::size_t{0};
  ASSERT_LE(phi_vertex_values[active_vertex], 0.0);
  const auto active_point = workflowVertexPoint(*mesh, active_vertex);
  const auto active_value = sim.fe_system->evaluateFieldAtPoint(
      target_velocity,
      svmp::FE::systems::SystemStateView{},
      active_point);
  ASSERT_TRUE(active_value.has_value());
  const auto active_expected = workflowVelocity(*mesh, active_vertex);
  EXPECT_NEAR((*active_value)[0], active_expected[0], 1.0e-10);
  EXPECT_NEAR((*active_value)[1], active_expected[1], 1.0e-10);

  // Vertices 1 and 5 are dry by nodal sign but support the retained cut cell.
  // They must be exact physical-velocity constraints, even though both lie on
  // the labelled wall.  Projecting either one would change the Q1 trace on
  // the free surface inside cell 0.
  for (const auto dry_trace_vertex : {std::size_t{1}, std::size_t{5}}) {
    ASSERT_GT(phi_vertex_values[dry_trace_vertex], 0.0);
    const auto point = workflowVertexPoint(*mesh, dry_trace_vertex);
    const auto value = sim.fe_system->evaluateFieldAtPoint(
        target_velocity,
        svmp::FE::systems::SystemStateView{},
        point);
    ASSERT_TRUE(value.has_value());
    const auto expected = workflowVelocity(*mesh, dry_trace_vertex);
    EXPECT_NEAR((*value)[0], expected[0], 1.0e-12)
        << "dry trace-support vertex " << dry_trace_vertex;
    EXPECT_NEAR((*value)[1], expected[1], 1.0e-12)
        << "dry trace-support vertex " << dry_trace_vertex;
  }

  const auto* context = sim.fe_system->cutIntegrationContext();
  ASSERT_NE(context, nullptr);
  std::size_t checked_interface_points = 0u;
  for (const auto* rule :
       context->interfaceRulesForMarker(kInterfaceMarker)) {
    ASSERT_NE(rule, nullptr);
    ASSERT_GE(rule->provenance.parent_entity, 0);
    const auto mapping = createCellGeometryMapping(
        sim.fe_system->meshAccess(), rule->provenance.parent_entity);
    ASSERT_NE(mapping, nullptr);
    for (const auto& qp : rule->points) {
      std::array<svmp::FE::Real, 3> point{
          qp.point[0], qp.point[1], qp.point[2]};
      if (rule->frame ==
          svmp::FE::geometry::CutGeometryFrame::Reference) {
        const auto physical =
            physicalCellPointAtReference(*mapping, qp.point);
        ASSERT_TRUE(physical.has_value());
        point = *physical;
      }
      const auto source_value = sim.fe_system->evaluateFieldAtPoint(
          source_velocity, state, point);
      const auto extension_value = sim.fe_system->evaluateFieldAtPoint(
          target_velocity,
          svmp::FE::systems::SystemStateView{},
          point);
      ASSERT_TRUE(source_value.has_value());
      ASSERT_TRUE(extension_value.has_value());
      EXPECT_NEAR((*extension_value)[0], (*source_value)[0], 1.0e-12)
          << "generated interface quadrature point "
          << checked_interface_points;
      EXPECT_NEAR((*extension_value)[1], (*source_value)[1], 1.0e-12)
          << "generated interface quadrature point "
          << checked_interface_points;
      ++checked_interface_points;
    }
  }
  EXPECT_GT(checked_interface_points, 0u);

  // Outside the cut-cell trace support, the existing graph extension and
  // wall projection remain in force.  Vertex 3 is a dry outer corner with two
  // independent wall normals, so its projected velocity is zero.
  const auto outer_dry_wall_vertex = std::size_t{3};
  ASSERT_GT(phi_vertex_values[outer_dry_wall_vertex], 0.0);
  const auto outer_point =
      workflowVertexPoint(*mesh, outer_dry_wall_vertex);
  const auto outer_value = sim.fe_system->evaluateFieldAtPoint(
      target_velocity,
      svmp::FE::systems::SystemStateView{},
      outer_point);
  ASSERT_TRUE(outer_value.has_value());
  EXPECT_NEAR((*outer_value)[0], 0.0, 1.0e-12);
  EXPECT_NEAR((*outer_value)[1], 0.0, 1.0e-12);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     WallCompatibleNormalVelocityExtensionRejectsHigherOrderFields)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowBiquadraticQuadMesh();
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/2,
      svmp::FE::BasisType::Hierarchical);
  auto vector_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, 2);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi", .space = scalar_space, .components = 1});
  system->addField(svmp::FE::systems::FieldSpec{
      .name = "Velocity", .space = vector_space, .components = 2});
  system->addField(svmp::FE::systems::FieldSpec{
      .name = "LevelSetAdvectionVelocity",
      .space = vector_space,
      .components = 2,
      .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData});
  ASSERT_NO_THROW(system->setup({}));

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);
  LevelSetAdvectionVelocityRequest request{};
  request.level_set_field_name = "phi";
  request.source_velocity_field_name = "Velocity";
  request.target_velocity_field_name = "LevelSetAdvectionVelocity";
  request.extension_method = "wall_compatible_normal";

  EXPECT_THROW(
      {
        try {
          (void)updateLevelSetAdvectionVelocitiesFromState(
              sim,
              svmp::FE::systems::SystemStateView{},
              std::vector<LevelSetAdvectionVelocityRequest>{request});
        } catch (const std::runtime_error& error) {
          EXPECT_NE(std::string(error.what()).find("fixed P1"),
                    std::string::npos);
          throw;
        }
      },
      std::runtime_error);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     WallCompatibleNormalVelocityExtensionRejectsMismatchedP1Layouts)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowThreeQuadStripMesh();
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/1);
  auto source_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, 2);
  auto target_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, 3);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi", .space = scalar_space, .components = 1});
  system->addField(svmp::FE::systems::FieldSpec{
      .name = "Velocity", .space = source_space, .components = 2});
  system->addField(svmp::FE::systems::FieldSpec{
      .name = "LevelSetAdvectionVelocity",
      .space = target_space,
      .components = 3,
      .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData});
  ASSERT_NO_THROW(system->setup({}));

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);
  LevelSetAdvectionVelocityRequest request{};
  request.level_set_field_name = "phi";
  request.source_velocity_field_name = "Velocity";
  request.target_velocity_field_name = "LevelSetAdvectionVelocity";
  request.extension_method = "wall_compatible_normal";

  EXPECT_THROW(
      {
        try {
          (void)updateLevelSetAdvectionVelocitiesFromState(
              sim,
              svmp::FE::systems::SystemStateView{},
              std::vector<LevelSetAdvectionVelocityRequest>{request});
        } catch (const std::runtime_error& error) {
          EXPECT_NE(std::string(error.what()).find(
                        "identical component layouts"),
                    std::string::npos);
          throw;
        }
      },
      std::runtime_error);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     AlgebraicWallCompatibleExtensionInvalidatesStaleMapWhenInterfaceDisappears)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowThreeQuadStripMesh();
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/1);
  auto vector_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, 2);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto physical_velocity = system->addField(
      svmp::FE::systems::FieldSpec{
          .name = "Velocity", .space = vector_space, .components = 2});
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi", .space = scalar_space, .components = 1});
  system->addOperator("level_set");

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
      *system, scalar_space, transport);

  const auto extension_velocity =
      system->findFieldByName("LevelSetAdvectionVelocity");
  ASSERT_NE(extension_velocity, svmp::FE::INVALID_FIELD_ID);
  const auto extension_kernel =
      svmp::FE::level_set::findLevelSetVelocityExtensionConstraintKernel(
          *system, "level_set", extension_velocity);
  ASSERT_TRUE(extension_kernel);
  std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow>
      stale_rows;
  for (svmp::FE::GlobalIndex vertex = 0;
       vertex < static_cast<svmp::FE::GlobalIndex>(mesh->n_vertices());
       ++vertex) {
    for (int component = 0; component < 2; ++component) {
      stale_rows.push_back(
          svmp::FE::level_set::VelocityExtensionConstraintRow{
              .vertex = vertex,
              .component = component,
              .dependencies = {
                  svmp::FE::level_set::VelocityExtensionDependency{
                      .field = svmp::FE::level_set::
                          VelocityExtensionDependencyField::SourceVelocity,
                      .vertex = vertex,
                      .component = component,
                      .coefficient = 1.0}}});
    }
  }
  extension_kernel->setFrozenRows(std::move(stale_rows), 1u);
  ASSERT_TRUE(extension_kernel->hasFrozenMap());
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_values(mesh->n_vertices(), -1.0);
  std::vector<svmp::FE::Real> velocity_values(mesh->n_vertices() * 2u, 0.0);
  const auto phi_coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      phi_values,
      1u,
      "ApplicationDriver disappeared-interface phi");
  const auto velocity_coefficients = projectWorkflowVertexValues(
      *system,
      physical_velocity,
      velocity_values,
      2u,
      "ApplicationDriver disappeared-interface velocity");
  std::vector<svmp::FE::Real> solution(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, phi_coefficients, solution);
  writeWorkflowFieldSlice(
      *system, physical_velocity, velocity_coefficients, solution);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);
  svmp::FE::systems::SystemStateView state{};
  state.u = solution;
  LevelSetAdvectionVelocityRequest request{};
  request.level_set_field_name = "phi";
  request.source_velocity_field_name = "Velocity";
  request.target_velocity_field_name = "LevelSetAdvectionVelocity";
  request.operator_tag = "level_set";
  request.extension_method = "wall_compatible_normal";
  request.enforce_wall_impermeability = false;

  EXPECT_THROW(
      {
        try {
          (void)updateLevelSetAdvectionVelocitiesFromState(
              sim,
              state,
              std::vector<LevelSetAdvectionVelocityRequest>{request});
        } catch (const std::runtime_error& error) {
          EXPECT_NE(std::string(error.what()).find(
                        "no resolved interface geometry samples"),
                    std::string::npos);
          throw;
        }
      },
      std::runtime_error);
  EXPECT_FALSE(extension_kernel->hasFrozenMap())
      << "A failed rebuild must not leave the previous interface map valid.";
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     AuthoritativeEmptyCutContextDoesNotFallBackToNodalCrossings)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr std::string_view kDomainId = "empty_authoritative_context";
  auto mesh = makeWorkflowThreeQuadStripMesh();
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/1);
  auto vector_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, 2);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi", .space = scalar_space, .components = 1});
  const auto physical_velocity = system->addField(
      svmp::FE::systems::FieldSpec{
          .name = "Velocity", .space = vector_space, .components = 2});
  system->addField(svmp::FE::systems::FieldSpec{
      .name = "LevelSetAdvectionVelocity",
      .space = vector_space,
      .components = 2,
      .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData});
  ASSERT_NO_THROW(system->setup({}));

  svmp::FE::interfaces::GeneratedInterfaceMarkerKey marker_key{};
  marker_key.source =
      svmp::FE::interfaces::LevelSetInterfaceSource::fromField(phi);
  marker_key.domain_id = std::string(kDomainId);
  marker_key.isovalue = 0.0;
  marker_key.requested_marker = -1;
  const int interface_marker =
      svmp::FE::interfaces::stableGeneratedInterfaceMarker(marker_key);

  std::vector<svmp::FE::Real> phi_values(mesh->n_vertices(), 0.0);
  std::vector<svmp::FE::Real> velocity_values(mesh->n_vertices() * 2u, 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    phi_values[vertex] = workflowPhi(*mesh, vertex);
    const auto velocity = workflowVelocity(*mesh, vertex);
    velocity_values[2u * vertex] = velocity[0];
    velocity_values[2u * vertex + 1u] = velocity[1];
  }
  const auto phi_coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      phi_values,
      1u,
      "ApplicationDriver empty-authoritative-context phi");
  const auto velocity_coefficients = projectWorkflowVertexValues(
      *system,
      physical_velocity,
      velocity_values,
      2u,
      "ApplicationDriver empty-authoritative-context velocity");
  std::vector<svmp::FE::Real> solution(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, phi_coefficients, solution);
  writeWorkflowFieldSlice(
      *system, physical_velocity, velocity_coefficients, solution);

  auto empty_context =
      std::make_shared<svmp::FE::assembly::CutIntegrationContext>();
  empty_context->setExpectedGeneratedSourceValueRevision(
      interface_marker, 1u);
  ASSERT_TRUE(empty_context->hasExpectedGeneratedSourceValueRevision(
      interface_marker));
  ASSERT_FALSE(empty_context->hasGeneratedInterfaceMarker(interface_marker));
  ASSERT_FALSE(empty_context->hasGeneratedVolumeMarker(interface_marker));
  system->registerGeneratedEmbeddedInterfaceMarker(interface_marker);
  system->setCutIntegrationContext(empty_context);

  LevelSetAdvectionVelocityRequest request{};
  request.level_set_field_name = "phi";
  request.source_velocity_field_name = "Velocity";
  request.target_velocity_field_name = "LevelSetAdvectionVelocity";
  request.domain_id = std::string(kDomainId);
  request.extension_method = "wall_compatible_normal";
  request.enforce_wall_impermeability = false;
  ASSERT_FALSE(request.active_cut_request_index.has_value());
  EXPECT_EQ(configuredInterfaceVelocityMarker(*system, request), std::nullopt)
      << "Integer-only marker registration must not authorize an unkeyed "
         "nodal request.";
  EXPECT_FALSE(hasAuthoritativeInterfaceVelocityContext(*system, request));
  request.active_cut_request_index = 0u;
  ASSERT_EQ(configuredInterfaceVelocityMarker(*system, request),
            std::optional<int>{interface_marker});
  ASSERT_TRUE(hasAuthoritativeInterfaceVelocityContext(*system, request));
  EXPECT_TRUE(interfaceVelocitySampleCandidateCells(*system, request).empty());
  EXPECT_FALSE(nodalVelocityExtensionInterfaceCells(
                   *mesh, phi_values, request.isovalue)
                   .empty())
      << "The fixture must contain a nodal crossing that the authoritative "
         "empty context suppresses.";

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);
  svmp::FE::systems::SystemStateView state{};
  state.u = solution;
  EXPECT_THROW(
      {
        try {
          (void)updateLevelSetAdvectionVelocitiesFromState(
              sim,
              state,
              std::vector<LevelSetAdvectionVelocityRequest>{request});
        } catch (const std::runtime_error& error) {
          EXPECT_NE(std::string(error.what()).find(
                        "no resolved interface geometry samples"),
                    std::string::npos);
          throw;
        }
      },
      std::runtime_error);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     VelocityExtensionGraphUsesCellEdgesWithoutQuadDiagonals)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  const auto mesh = makeDisconnectedWorkflowQuadPairMesh();
  const auto adjacency = velocityExtensionEdgeAdjacency(*mesh);
  ASSERT_EQ(adjacency.size(), 8u);

  EXPECT_EQ(adjacency[0], (std::vector<std::size_t>{1u, 3u}));
  EXPECT_EQ(adjacency[1], (std::vector<std::size_t>{0u, 2u}));
  EXPECT_EQ(adjacency[2], (std::vector<std::size_t>{1u, 3u}));
  EXPECT_EQ(adjacency[3], (std::vector<std::size_t>{0u, 2u}));
  EXPECT_EQ(adjacency[4], (std::vector<std::size_t>{5u, 7u}));
  EXPECT_EQ(adjacency[5], (std::vector<std::size_t>{4u, 6u}));
  EXPECT_EQ(adjacency[6], (std::vector<std::size_t>{5u, 7u}));
  EXPECT_EQ(adjacency[7], (std::vector<std::size_t>{4u, 6u}));
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     AlgebraicExtensionRefreshReprojectsStateAndChangesRevision)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowThreeQuadStripMesh();
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/1);
  auto vector_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, 2);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto source_velocity = system->addField(
      svmp::FE::systems::FieldSpec{
          .name = "Velocity", .space = vector_space, .components = 2});
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi", .space = scalar_space, .components = 1});
  system->addOperator("level_set");

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
      *system, scalar_space, transport);
  const auto extension_velocity =
      system->findFieldByName("LevelSetAdvectionVelocity");
  ASSERT_NE(extension_velocity, svmp::FE::INVALID_FIELD_ID);
  const auto extension_kernel =
      svmp::FE::level_set::findLevelSetVelocityExtensionConstraintKernel(
          *system, "level_set", extension_velocity);
  ASSERT_TRUE(extension_kernel);
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_values(mesh->n_vertices(), 0.0);
  std::vector<svmp::FE::Real> source_values(mesh->n_vertices() * 2u, 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    phi_values[vertex] = point[0] - 0.25;
    source_values[2u * vertex] = 2.0 + 3.0 * point[1];
    source_values[2u * vertex + 1u] = -1.0 + 0.5 * point[1];
  }
  const auto phi_coefficients = projectWorkflowVertexValues(
      *system, phi, phi_values, 1u, "algebraic extension refresh phi");
  const auto source_coefficients = projectWorkflowVertexValues(
      *system,
      source_velocity,
      source_values,
      2u,
      "algebraic extension refresh source");
  std::vector<svmp::FE::Real> initial(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 777.0);
  writeWorkflowFieldSlice(*system, phi, phi_coefficients, initial);
  writeWorkflowFieldSlice(
      *system, source_velocity, source_coefficients, initial);

  std::unique_ptr<svmp::FE::backends::BackendFactory> factory;
  try {
    factory = svmp::FE::backends::BackendFactory::create(
        svmp::FE::backends::BackendKind::FSILS);
  } catch (const std::exception&) {
    GTEST_SKIP() << "Requires an available FE vector backend.";
  }
  auto state_vector = factory->createVector(
      system->dofHandler().getNumDofs());
  ASSERT_TRUE(state_vector);
  scatterFeOrderedSolution(*state_vector, initial);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);
  svmp::FE::systems::SystemStateView state{};
  state.u_vector = state_vector.get();
  LevelSetAdvectionVelocityRequest request{};
  request.level_set_field_name = "phi";
  request.source_velocity_field_name = "Velocity";
  request.target_velocity_field_name = "LevelSetAdvectionVelocity";
  request.operator_tag = "level_set";
  request.extension_method = "wall_compatible_normal";
  request.extension_band_layers = 3;
  request.enforce_wall_impermeability = false;

  ASSERT_TRUE(updateLevelSetAdvectionVelocitiesFromState(
      sim, state, {request}));
  ASSERT_TRUE(extension_kernel->hasFrozenMap());
  const auto first_revision = extension_kernel->frozenMapRevision();
  EXPECT_NE(first_revision, 0u);
  const auto first_solution = gatherFeOrderedSolution(*state_vector);
  const auto extension_offset =
      sim.fe_system->fieldDofOffset(extension_velocity);
  const auto* extension_entity_map =
      sim.fe_system->fieldDofHandler(extension_velocity).getEntityDofMap();
  ASSERT_NE(extension_entity_map, nullptr);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto dofs = extension_entity_map->getVertexDofs(
        static_cast<svmp::FE::GlobalIndex>(vertex));
    ASSERT_EQ(dofs.size(), 2u);
    for (std::size_t component = 0; component < 2u; ++component) {
      EXPECT_NEAR(first_solution[static_cast<std::size_t>(
                      extension_offset + dofs[component])],
                  source_values[2u * vertex + component],
                  1.0e-12);
    }
  }

  phi_values.back() += 1.0e-5;
  const auto changed_phi_coefficients = projectWorkflowVertexValues(
      *sim.fe_system,
      phi,
      phi_values,
      1u,
      "algebraic extension changed phi");
  auto changed_solution = first_solution;
  writeWorkflowFieldSlice(
      *sim.fe_system, phi, changed_phi_coefficients, changed_solution);
  for (const auto local_dof :
       sim.fe_system->fieldDofHandler(extension_velocity)
           .getPartition()
           .locallyOwned()) {
    changed_solution[static_cast<std::size_t>(extension_offset + local_dof)] =
        -999.0;
  }
  scatterFeOrderedSolution(*state_vector, changed_solution);
  ASSERT_TRUE(updateLevelSetAdvectionVelocitiesFromState(
      sim, state, {request}));
  EXPECT_NE(extension_kernel->frozenMapRevision(), first_revision);
  const auto refreshed_solution = gatherFeOrderedSolution(*state_vector);
  for (const auto local_dof :
       sim.fe_system->fieldDofHandler(extension_velocity)
           .getPartition()
           .locallyOwned()) {
    EXPECT_NE(refreshed_solution[static_cast<std::size_t>(
                  extension_offset + local_dof)],
              -999.0);
  }
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     VelocityExtensionRegressionConditionEstimateDetectsNearSingularity)
{
  std::array<std::array<double, 4>, 4> matrix{};
  matrix[0][0] = 1.0;
  matrix[0][1] = 1.0 - 1.0e-12;
  matrix[1][0] = matrix[0][1];
  matrix[1][1] = 1.0;

  const double condition = estimateSymmetricConditionNumber(matrix, 2);
  const auto estimate =
      application::core::estimateSymmetricRankAndCondition(matrix, 2);
  EXPECT_TRUE(std::isfinite(condition));
  EXPECT_GT(condition, kVelocityExtensionMaxRegressionCondition);
  EXPECT_EQ(estimate.numerical_rank, 2);
  EXPECT_EQ(estimate.condition_estimate, condition);

  matrix[0][1] = 1.0;
  matrix[1][0] = 1.0;
  const auto singular =
      application::core::estimateSymmetricRankAndCondition(matrix, 2);
  EXPECT_EQ(singular.numerical_rank, 1);
  EXPECT_TRUE(std::isinf(singular.condition_estimate));
}

TEST(ApplicationDriverLevelSetWorkflows,
     ReducedD38MapFailureUsesBoundedRefreshNeutralFallback)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr double kUpperY = 1.00001;
  constexpr double kPhysicalVelocityScale = 0.04;
  const auto mesh = makeWorkflowSkewedExtensionTriangleMesh(kUpperY);

  // This thin tangential stencil is a deterministic reduction of the D38
  // failure archived under free_surface_review_completion_20260717.  The
  // earlier unguarded two-point affine evaluation is unique on this stencil:
  // its weights reproduce constants and the tangent coordinate at the dry
  // target, but their opposite signs turn O(0.04) data into O(10^3).
  const auto first_source_point = workflowVertexPoint(*mesh, 0u);
  const auto dry_target_point = workflowVertexPoint(*mesh, 1u);
  const auto second_source_point = workflowVertexPoint(*mesh, 2u);
  const double first_tangent =
      first_source_point[1] - dry_target_point[1];
  const double second_tangent =
      second_source_point[1] - dry_target_point[1];
  const double tangent_span = second_tangent - first_tangent;
  ASSERT_GT(std::abs(tangent_span), 0.0);
  const double old_first_weight = second_tangent / tangent_span;
  const double old_second_weight = -first_tangent / tangent_span;
  const double old_row_l1 =
      std::abs(old_first_weight) + std::abs(old_second_weight);
  const double old_dry_velocity =
      old_first_weight * kPhysicalVelocityScale -
      old_second_weight * kPhysicalVelocityScale;
  const double old_amplification =
      std::abs(old_dry_velocity) / kPhysicalVelocityScale;
  EXPECT_NEAR(old_first_weight + old_second_weight, 1.0, 1.0e-10);
  EXPECT_LT(old_second_weight, 0.0);
  EXPECT_GT(old_row_l1, 1.0e5);
  EXPECT_GT(std::abs(old_dry_velocity), 1.0e2);
  EXPECT_GT(old_amplification, 1.0e5);

  std::vector<double> phi(mesh->n_vertices(), 0.0);
  std::vector<double> source(mesh->n_vertices() * 2u, 0.0);
  std::vector<std::uint8_t> active(mesh->n_vertices(), 0u);
  for (std::size_t vertex = 0u; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    phi[vertex] = point[0] - 0.5;
    active[vertex] = phi[vertex] <= 0.0 ? 1u : 0u;
  }
  source[0u] = kPhysicalVelocityScale;
  source[4u] = -kPhysicalVelocityScale;

  const auto revision = application::core::velocityExtensionMapRevision(
      31u, 32u, 33u, 34u, 35u, phi, active);
  const auto build_snapshot = [&]() {
    return application::core::buildVelocityExtensionMapSnapshot(
        *mesh,
        svmp::MeshComm::self(),
        revision,
        phi,
        source,
        /*source_components=*/2u,
        active,
        /*target_components=*/2u,
        /*copy_components=*/2u,
        /*band_layers=*/1,
        /*enforce_wall_impermeability=*/false,
        std::span<const WallVelocityExtensionConstraint>{});
  };
  const auto first = build_snapshot();
  ASSERT_TRUE(first);
  ASSERT_EQ(first->report().regression_candidate_rows, 1u);
  EXPECT_EQ(first->report().regression_accepted_rows, 0u);
  EXPECT_EQ(first->report().bounded_fallback_rows, 1u);
  EXPECT_EQ(first->report().condition_rejected_rows, 1u);
  EXPECT_EQ(first->report().coefficient_rejected_rows, 0u);
  EXPECT_LE(first->report().max_abs_graph_coefficient,
            1.0 + kVelocityExtensionRowTolerance);
  EXPECT_LE(first->report().max_graph_row_l1,
            1.0 + kVelocityExtensionRowTolerance);
  EXPECT_LE(first->report().max_graph_row_sum_error,
            kVelocityExtensionRowTolerance);
  EXPECT_LE(first->report().max_negative_graph_coefficient,
            kVelocityExtensionCoefficientTolerance);
  EXPECT_LE(first->wetToDryAmplification(), 1.0 + 1.0e-12);
  ASSERT_EQ(first->preview().size(), source.size());
  EXPECT_LE(std::abs(first->preview()[2u]),
            kPhysicalVelocityScale + 1.0e-12);

  const auto dry_diagnostic = std::find_if(
      first->rowDiagnostics().begin(),
      first->rowDiagnostics().end(),
      [](const auto& diagnostic) {
        return diagnostic.local_vertex == 1;
      });
  ASSERT_NE(dry_diagnostic, first->rowDiagnostics().end());
  EXPECT_EQ(dry_diagnostic->disposition,
            application::core::VelocityExtensionRowDisposition::
                BoundedFallback);
  EXPECT_TRUE(dry_diagnostic->condition_rejected);
  EXPECT_TRUE(dry_diagnostic->bounded_fallback_used);
  EXPECT_EQ(dry_diagnostic->negative_weight_count, 0u);
  EXPECT_NEAR(dry_diagnostic->coefficient_sum,
              1.0,
              kVelocityExtensionRowTolerance);
  EXPECT_LE(dry_diagnostic->coefficient_l1,
            1.0 + kVelocityExtensionRowTolerance);
  EXPECT_LE(dry_diagnostic->preview_amplification, 1.0 + 1.0e-12);

  const auto maximum_map_residual = [&](const auto& snapshot) {
    double maximum = 0.0;
    for (const auto& row : snapshot.rows()) {
      if (row.vertex < 0 || row.component < 0 || row.component >= 2) {
        ADD_FAILURE() << "Invalid reduced D38 extension row";
        return std::numeric_limits<double>::infinity();
      }
      const auto row_vertex = static_cast<std::size_t>(row.vertex);
      const auto row_component = static_cast<std::size_t>(row.component);
      if (row_vertex >= mesh->n_vertices()) {
        ADD_FAILURE() << "Reduced D38 extension row is outside the mesh";
        return std::numeric_limits<double>::infinity();
      }
      double residual =
          snapshot.preview()[2u * row_vertex + row_component];
      for (const auto& dependency : row.dependencies) {
        if (dependency.vertex < 0 || dependency.component < 0 ||
            dependency.component >= 2) {
          ADD_FAILURE() << "Invalid reduced D38 extension dependency";
          return std::numeric_limits<double>::infinity();
        }
        const auto dependency_vertex =
            static_cast<std::size_t>(dependency.vertex);
        const auto dependency_component =
            static_cast<std::size_t>(dependency.component);
        if (dependency_vertex >= mesh->n_vertices()) {
          ADD_FAILURE()
              << "Reduced D38 extension dependency is outside the mesh";
          return std::numeric_limits<double>::infinity();
        }
        const std::span<const double> values =
            dependency.field == svmp::FE::level_set::
                                    VelocityExtensionDependencyField::
                                        SourceVelocity
                ? std::span<const double>(source)
                : snapshot.preview();
        residual -= dependency.coefficient *
                    values[2u * dependency_vertex + dependency_component];
      }
      maximum = std::max(maximum, std::abs(residual));
    }
    return maximum;
  };
  EXPECT_LE(maximum_map_residual(*first), 1.0e-14);

  const auto refreshed = build_snapshot();
  ASSERT_TRUE(refreshed);
  const auto change = application::core::compareVelocityExtensionMapSnapshots(
      *refreshed, first.get());
  EXPECT_TRUE(change.previous_available);
  EXPECT_FALSE(change.revision_changed);
  EXPECT_EQ(change.changed_owner_rows, 0u);
  EXPECT_EQ(change.component_assignment_changes, 0u);
  EXPECT_EQ(change.row_decision_changes, 0u);
  EXPECT_EQ(change.dependency_row_changes, 0u);
  EXPECT_EQ(change.maximum_coefficient_change, 0.0);
  EXPECT_EQ(change.preview_l2_change, 0.0);
  EXPECT_EQ(change.preview_linf_change, 0.0);
  EXPECT_LE(maximum_map_residual(*refreshed), 1.0e-14);

  RecordProperty("d38_archived_extension_norm", "135.759");
  RecordProperty("d38_archived_physical_velocity_norm", "0.0403971");
  RecordProperty("reduced_unguarded_row_l1",
                 std::to_string(old_row_l1));
  RecordProperty("reduced_unguarded_amplification",
                 std::to_string(old_amplification));
  RecordProperty("guarded_wet_to_dry_amplification",
                 std::to_string(first->wetToDryAmplification()));
  RecordProperty("same_state_refresh_preview_linf_change",
                 std::to_string(change.preview_linf_change));
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     VelocityExtensionOrientationMakesEitherRetainedSideNegative)
{
  const std::array<double, 3> level_set{{-0.25, 0.5, 1.25}};
  const auto negative = orientedLevelSetForVelocityExtension(
      level_set, 0.5, LevelSetActiveSide::Negative);
  const auto positive = orientedLevelSetForVelocityExtension(
      level_set, 0.5, LevelSetActiveSide::Positive);
  EXPECT_EQ(negative, (std::vector<double>{-0.75, 0.0, 0.75}));
  EXPECT_EQ(positive, (std::vector<double>{0.75, -0.0, -0.75}));

  const std::array<double, 1> nonfinite{{
      std::numeric_limits<double>::quiet_NaN()}};
  EXPECT_THROW(
      (void)orientedLevelSetForVelocityExtension(
          nonfinite, 0.0, LevelSetActiveSide::Negative),
      std::invalid_argument);
}

TEST(ApplicationDriverLevelSetWorkflows,
     WetExtensionRequestInfersOnlyZeroDirichletWalls)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Level_set_field_name>phi</Level_set_field_name>
    <Velocity_source>prescribed_data</Velocity_source>
    <Velocity_field_name>LevelSetAdvectionVelocity</Velocity_field_name>
    <Use_wet_extension_advection_velocity>true</Use_wet_extension_advection_velocity>
    <Source_velocity_field_name>Velocity</Source_velocity_field_name>
    <Wet_extension_advection_velocity_method>nearest_interface_point</Wet_extension_advection_velocity_method>
  </Add_equation>
  <Add_equation type="fluid">
    <Add_BC name="wall">
      <Type>Dir</Type>
      <Value>0.0</Value>
    </Add_BC>
    <Add_BC name="normal_wall">
      <Type>Dir</Type>
      <Value>0.0</Value>
      <Effective_direction>0 1</Effective_direction>
    </Add_BC>
    <Add_BC name="moving_lid">
      <Type>Dir</Type>
      <Value>1.0</Value>
    </Add_BC>
    <Add_BC name="outlet">
      <Type>Neu</Type>
      <Value>0.0</Value>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  const auto requests = levelSetAdvectionVelocityRequests(*params);
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests.front().extension_method, "wall_compatible_normal");
  EXPECT_EQ(requests.front().wall_face_names,
            (std::vector<std::string>{"wall", "normal_wall"}));
  ASSERT_EQ(requests.front().wall_constraints.size(), 2u);
  EXPECT_EQ(requests.front().wall_constraints[0].face_name, "wall");
  EXPECT_TRUE(
      requests.front().wall_constraints[0].effective_direction.empty());
  EXPECT_EQ(requests.front().wall_constraints[1].face_name, "normal_wall");
  EXPECT_EQ(requests.front().wall_constraints[1].effective_direction,
            (std::vector<int>{0, 1}));
}

TEST(ApplicationDriverLevelSetWorkflows,
     WetExtensionExplicitWallFailsClosedForNonzeroDirichletData)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Level_set_field_name>phi</Level_set_field_name>
    <Velocity_source>prescribed_data</Velocity_source>
    <Velocity_field_name>LevelSetAdvectionVelocity</Velocity_field_name>
    <Use_wet_extension_advection_velocity>true</Use_wet_extension_advection_velocity>
    <Source_velocity_field_name>Velocity</Source_velocity_field_name>
    <Wet_extension_wall_faces>moving_lid</Wet_extension_wall_faces>
  </Add_equation>
  <Add_equation type="fluid">
    <Add_BC name="moving_lid">
      <Type>Dir</Type>
      <Value>1.0</Value>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  EXPECT_THROW(
      {
        try {
          (void)levelSetAdvectionVelocityRequests(*params);
        } catch (const std::runtime_error& error) {
          EXPECT_NE(std::string(error.what()).find("nonzero"),
                    std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

TEST(ApplicationDriverLevelSetWorkflows,
     WetExtensionWallMasksComeOnlyFromOwningFluidEquation)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Level_set_field_name>phi</Level_set_field_name>
    <Velocity_source>prescribed_data</Velocity_source>
    <Velocity_field_name>LevelSetAdvectionVelocity</Velocity_field_name>
    <Use_wet_extension_advection_velocity>true</Use_wet_extension_advection_velocity>
    <Source_velocity_field_name>Velocity</Source_velocity_field_name>
  </Add_equation>
  <Add_equation type="heatS">
    <Add_BC name="wall">
      <Type>Dir</Type>
      <Value>0.0</Value>
      <Effective_direction>1 0</Effective_direction>
    </Add_BC>
    <Add_BC name="scalar_only">
      <Type>Dir</Type>
      <Value>0.0</Value>
    </Add_BC>
  </Add_equation>
  <Add_equation type="fluid">
    <Add_BC name="wall">
      <Type>Dir</Type>
      <Value>0.0</Value>
      <Effective_direction>0 1</Effective_direction>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  const auto requests = levelSetAdvectionVelocityRequests(*params);
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests.front().wall_face_names,
            (std::vector<std::string>{"wall"}));
  ASSERT_EQ(requests.front().wall_constraints.size(), 1u);
  EXPECT_EQ(requests.front().wall_constraints.front().effective_direction,
            (std::vector<int>{0, 1}));
}

TEST(ApplicationDriverLevelSetWorkflows,
     WetExtensionWallDiscoveryFailsWithoutOwningFluidEquation)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Level_set_field_name>phi</Level_set_field_name>
    <Velocity_source>prescribed_data</Velocity_source>
    <Velocity_field_name>LevelSetAdvectionVelocity</Velocity_field_name>
    <Use_wet_extension_advection_velocity>true</Use_wet_extension_advection_velocity>
    <Source_velocity_field_name>Velocity</Source_velocity_field_name>
  </Add_equation>
  <Add_equation type="heatS">
    <Add_BC name="wall">
      <Type>Dir</Type>
      <Value>0.0</Value>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  EXPECT_THROW(
      {
        try {
          (void)levelSetAdvectionVelocityRequests(*params);
        } catch (const std::runtime_error& error) {
          EXPECT_NE(std::string(error.what()).find("exactly one fluid"),
                    std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

TEST(ApplicationDriverLevelSetWorkflows,
     NormalBandVelocityExtensionHonorsStrongNoSlipAndNormalOnlyMasks)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr svmp::label_t kHorizontalWall = 4242;
  const auto mesh = makeWorkflowThreeQuadStripMesh();
  auto& local_mesh = mesh->local_mesh();
  for (const auto face : local_mesh.boundary_faces()) {
    const auto normal = local_mesh.face_normal(face);
    if (std::abs(normal[1]) > 0.9 * std::abs(normal[0])) {
      mesh->set_boundary_label(face, kHorizontalWall);
    }
  }

  std::vector<double> phi(mesh->n_vertices(), 0.0);
  std::vector<double> source(mesh->n_vertices() * 2u, 0.0);
  std::vector<std::uint8_t> active(mesh->n_vertices(), 0u);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    phi[vertex] = point[0] - 0.25;
    active[vertex] = phi[vertex] <= 0.0 ? 1u : 0u;
    source[2u * vertex] = 2.0 + point[1];
    source[2u * vertex + 1u] = 5.0 - 0.5 * point[1];
  }

  const std::vector<WallVelocityExtensionConstraint> normal_only{{
      .boundary_label = kHorizontalWall,
      .constrained_components = {false, true, false}}};
  std::vector<double> slip_extension;
  const auto slip_report = extendVelocityInLevelSetNormalBand(
      *mesh,
      svmp::MeshComm::self(),
      phi,
      source,
      /*source_components=*/2u,
      active,
      /*target_components=*/2u,
      /*copy_components=*/2u,
      /*band_layers=*/3,
      /*enforce_wall_impermeability=*/true,
      std::span<const WallVelocityExtensionConstraint>(normal_only),
      slip_extension);

  const std::vector<WallVelocityExtensionConstraint> no_slip{{
      .boundary_label = kHorizontalWall,
      .constrained_components = {true, true, true}}};
  std::vector<double> no_slip_extension;
  const auto no_slip_report = extendVelocityInLevelSetNormalBand(
      *mesh,
      svmp::MeshComm::self(),
      phi,
      source,
      /*source_components=*/2u,
      active,
      /*target_components=*/2u,
      /*copy_components=*/2u,
      /*band_layers=*/3,
      /*enforce_wall_impermeability=*/true,
      std::span<const WallVelocityExtensionConstraint>(no_slip),
      no_slip_extension);

  EXPECT_EQ(slip_report.vertices_outside_band, 0u);
  EXPECT_EQ(no_slip_report.vertices_outside_band, 0u);
  EXPECT_NEAR(slip_report.max_wall_normal_velocity, 0.0, 1.0e-12);
  EXPECT_NEAR(no_slip_report.max_wall_normal_velocity, 0.0, 1.0e-12);
  std::size_t checked_dry_wall_vertices = 0u;
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    if (active[vertex] != 0u) {
      continue;
    }
    const auto point = workflowVertexPoint(*mesh, vertex);
    EXPECT_NEAR(slip_extension[2u * vertex],
                2.0 + point[1],
                1.0e-11)
        << "normal-only wall must retain tangential extension at vertex "
        << vertex;
    EXPECT_NEAR(slip_extension[2u * vertex + 1u], 0.0, 1.0e-12)
        << "normal-only wall must remove its constrained component at vertex "
        << vertex;
    EXPECT_NEAR(no_slip_extension[2u * vertex], 0.0, 1.0e-12)
        << "no-slip wall must remove tangential extension at vertex " << vertex;
    EXPECT_NEAR(no_slip_extension[2u * vertex + 1u], 0.0, 1.0e-12)
        << "no-slip wall must remove normal extension at vertex " << vertex;
    ++checked_dry_wall_vertices;
  }
  EXPECT_EQ(checked_dry_wall_vertices, 6u);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     NormalBandVelocityExtensionUsesSourceRowsOnDryCutCellSupport)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  const auto mesh = makeWorkflowThreeQuadStripMesh();
  std::vector<double> phi(mesh->n_vertices(), 0.0);
  std::vector<double> source(mesh->n_vertices() * 2u, 0.0);
  std::vector<std::uint8_t> seed(mesh->n_vertices(), 0u);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    phi[vertex] = point[0] - 0.25;
    seed[vertex] = phi[vertex] <= 0.0 ? 1u : 0u;
    source[2u * vertex] = 1.25 + 0.75 * point[0] - 0.5 * point[1];
    source[2u * vertex + 1u] = -0.4 + 0.2 * point[0] + point[1];
  }

  // Cell 0 contains the retained Q1 interface.  Its x=1 vertices (1 and 5)
  // are dry by sign but are necessary basis support for the physical trace.
  const std::array<svmp::FE::MeshIndex, 1> cut_cells{{0}};
  ASSERT_EQ(markVelocityExtensionTraceSupportCells(
                *mesh,
                std::span<const svmp::FE::MeshIndex>(cut_cells),
                seed),
            2u);
  ASSERT_EQ(synchronizeVelocityExtensionTraceSupportMask(
                *mesh, svmp::MeshComm::self(), seed),
            4u);

  std::vector<double> extension;
  std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow> rows;
  const auto report = extendVelocityInLevelSetNormalBand(
      *mesh,
      svmp::MeshComm::self(),
      phi,
      source,
      /*source_components=*/2u,
      seed,
      /*target_components=*/2u,
      /*copy_components=*/2u,
      /*band_layers=*/3,
      /*enforce_wall_impermeability=*/false,
      std::span<const WallVelocityExtensionConstraint>{},
      extension,
      &rows);
  EXPECT_EQ(report.vertices_outside_band, 0u);

  auto find_row = [&](std::size_t vertex, int component)
      -> const svmp::FE::level_set::VelocityExtensionConstraintRow* {
    const auto found = std::find_if(
        rows.begin(), rows.end(), [&](const auto& row) {
          return row.vertex == static_cast<svmp::FE::GlobalIndex>(vertex) &&
                 row.component == component;
        });
    return found == rows.end() ? nullptr : &*found;
  };

  for (const auto dry_trace_vertex : {std::size_t{1}, std::size_t{5}}) {
    ASSERT_GT(phi[dry_trace_vertex], 0.0);
    for (int component = 0; component < 2; ++component) {
      const auto c = static_cast<std::size_t>(component);
      EXPECT_DOUBLE_EQ(extension[2u * dry_trace_vertex + c],
                       source[2u * dry_trace_vertex + c]);
      const auto* row = find_row(dry_trace_vertex, component);
      ASSERT_NE(row, nullptr);
      ASSERT_EQ(row->dependencies.size(), 1u);
      EXPECT_EQ(row->dependencies.front().field,
                svmp::FE::level_set::
                    VelocityExtensionDependencyField::SourceVelocity);
      EXPECT_EQ(row->dependencies.front().vertex,
                static_cast<svmp::FE::GlobalIndex>(dry_trace_vertex));
      EXPECT_EQ(row->dependencies.front().component, component);
      EXPECT_DOUBLE_EQ(row->dependencies.front().coefficient, 1.0);
    }
  }

  // Vertex 2 is dry and lies one graph layer beyond the cut-cell support, so
  // it must remain an extension dependency rather than a physical trace row.
  for (int component = 0; component < 2; ++component) {
    const auto* row = find_row(/*vertex=*/2u, component);
    ASSERT_NE(row, nullptr);
    ASSERT_FALSE(row->dependencies.empty());
    EXPECT_TRUE(std::all_of(
        row->dependencies.begin(), row->dependencies.end(),
        [](const auto& dependency) {
          return dependency.field ==
                 svmp::FE::level_set::
                     VelocityExtensionDependencyField::ExtensionVelocity;
        }));
  }
  EXPECT_LE(report.max_abs_graph_coefficient,
            1.0 + kVelocityExtensionRowTolerance);
  EXPECT_LE(report.max_graph_row_l1,
            1.0 + kVelocityExtensionRowTolerance);
  EXPECT_LE(report.max_graph_row_sum_error,
            kVelocityExtensionRowTolerance);
  EXPECT_LE(report.max_negative_graph_coefficient,
            kVelocityExtensionCoefficientTolerance);
  EXPECT_LE(report.max_extended_speed,
            report.max_seed_speed + 1.0e-12);
  for (const auto& row : rows) {
    if (row.dependencies.empty() ||
        row.dependencies.front().field !=
            svmp::FE::level_set::
                VelocityExtensionDependencyField::ExtensionVelocity) {
      continue;
    }
    double coefficient_sum = 0.0;
    double coefficient_l1 = 0.0;
    for (const auto& dependency : row.dependencies) {
      EXPECT_GE(dependency.coefficient,
                -kVelocityExtensionCoefficientTolerance);
      coefficient_sum += dependency.coefficient;
      coefficient_l1 += std::abs(dependency.coefficient);
    }
    EXPECT_NEAR(coefficient_sum, 1.0, kVelocityExtensionRowTolerance);
    EXPECT_LE(coefficient_l1, 1.0 + kVelocityExtensionRowTolerance);
  }
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     NormalBandVelocityExtensionReproducesTangentialAffineField)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr int subdivisions = 4;
  const auto mesh = makeWorkflowStructuredQuadMesh(subdivisions);
  std::vector<double> phi(mesh->n_vertices(), 0.0);
  std::vector<double> source(mesh->n_vertices() * 2u, 0.0);
  std::vector<std::uint8_t> active(mesh->n_vertices(), 0u);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    phi[vertex] = point[0] - 0.25;
    active[vertex] = phi[vertex] <= 0.0 ? 1u : 0u;
    // The exact normal extension for phi=x-0.25 is any field independent of
    // x.  Use two affine tangential components to exercise the local fit.
    source[2u * vertex] = 2.0 + 3.0 * point[1];
    source[2u * vertex + 1u] = -1.0 + 0.5 * point[1];
  }

  std::vector<double> extended;
  const auto report = extendVelocityInLevelSetNormalBand(
      *mesh,
      phi,
      source,
      /*source_components=*/2u,
      active,
      /*target_components=*/2u,
      /*copy_components=*/2u,
      /*band_layers=*/subdivisions,
      /*enforce_wall_impermeability=*/false,
      /*wall_boundary_labels=*/{},
      extended);

  EXPECT_EQ(report.vertices_outside_band, 0u);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    EXPECT_NEAR(extended[2u * vertex], 2.0 + 3.0 * point[1], 1.0e-11)
        << "vertex " << vertex;
    EXPECT_NEAR(extended[2u * vertex + 1u],
                -1.0 + 0.5 * point[1],
                1.0e-11)
        << "vertex " << vertex;
  }
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     VelocityExtensionMapSnapshotTracksEveryRevisionDomain)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  const auto mesh = makeWorkflowThreeQuadStripMesh();
  std::vector<double> phi(mesh->n_vertices(), 0.0);
  std::vector<double> source(mesh->n_vertices() * 2u, 0.0);
  std::vector<std::uint8_t> seed(mesh->n_vertices(), 0u);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    phi[vertex] = point[0] - 0.25;
    seed[vertex] = phi[vertex] <= 0.0 ? 1u : 0u;
    source[2u * vertex] = 2.0 + 3.0 * point[1];
    source[2u * vertex + 1u] = -1.0 + 0.5 * point[1];
  }
  const std::array<svmp::FE::MeshIndex, 1> cut_cells{{0}};
  ASSERT_EQ(markVelocityExtensionTraceSupportCells(
                *mesh,
                std::span<const svmp::FE::MeshIndex>(cut_cells),
                seed),
            2u);

  const auto revision = application::core::velocityExtensionMapRevision(
      /*mesh_geometry=*/11u,
      /*mesh_topology=*/12u,
      /*mesh_ownership=*/13u,
      /*mesh_numbering=*/14u,
      /*free_surface_geometry=*/15u,
      phi,
      seed);
  const auto snapshot =
      application::core::buildVelocityExtensionMapSnapshot(
          *mesh,
          svmp::MeshComm::self(),
          revision,
          phi,
          source,
          /*source_components=*/2u,
          seed,
          /*target_components=*/2u,
          /*copy_components=*/2u,
          /*band_layers=*/3,
          /*enforce_wall_impermeability=*/false,
          std::span<const WallVelocityExtensionConstraint>{});
  ASSERT_TRUE(snapshot);
  EXPECT_EQ(snapshot->revision(), revision);
  EXPECT_EQ(snapshot->preview().size(), source.size());
  EXPECT_EQ(snapshot->componentAssignment().size(), mesh->n_vertices());
  ASSERT_EQ(snapshot->rowDiagnostics().size(), mesh->n_vertices());
  EXPECT_GT(snapshot->report().max_extrapolation_distance, 0.0);
  EXPECT_LE(snapshot->report().max_constant_reproduction_error,
            kVelocityExtensionRowTolerance);
  EXPECT_LE(snapshot->report().max_linear_reproduction_error,
            kVelocityExtensionRowTolerance);
  EXPECT_LE(snapshot->wetToDryAmplification(), 1.0 + 1.0e-12);

  std::size_t trace_rows = 0u;
  std::size_t reconstructed_rows = 0u;
  for (const auto& diagnostic : snapshot->rowDiagnostics()) {
    EXPECT_NE(diagnostic.local_vertex, svmp::FE::INVALID_GLOBAL_INDEX);
    EXPECT_NE(diagnostic.global_vertex, svmp::INVALID_GID);
    EXPECT_TRUE(diagnostic.assigned);
    EXPECT_GE(diagnostic.extrapolation_distance, 0.0);
    EXPECT_LE(diagnostic.preview_amplification, 1.0 + 1.0e-12);
    if (diagnostic.disposition ==
        application::core::VelocityExtensionRowDisposition::TraceSeed) {
      ++trace_rows;
      EXPECT_EQ(diagnostic.band_layer, 0);
      EXPECT_EQ(diagnostic.numerical_rank, 1);
      EXPECT_NEAR(diagnostic.coefficient_sum, 1.0, 0.0);
      ASSERT_EQ(diagnostic.dependencies.size(), 1u);
      continue;
    }
    ++reconstructed_rows;
    EXPECT_TRUE(diagnostic.regression_attempted);
    EXPECT_EQ(diagnostic.numerical_rank,
              diagnostic.reconstruction_dimension);
    EXPECT_NEAR(diagnostic.coefficient_sum, 1.0,
                kVelocityExtensionRowTolerance);
    EXPECT_LE(diagnostic.coefficient_l1,
              1.0 + kVelocityExtensionRowTolerance);
    EXPECT_EQ(diagnostic.negative_weight_count, 0u);
    EXPECT_FALSE(diagnostic.dependencies.empty());
  }
  EXPECT_GT(trace_rows, 0u);
  EXPECT_GT(reconstructed_rows, 0u);

  auto detached_rows = snapshot->copyRows();
  ASSERT_FALSE(detached_rows.empty());
  detached_rows.clear();
  EXPECT_FALSE(snapshot->rows().empty());

  auto changed_phi = phi;
  changed_phi.back() += 1.0e-6;
  const auto phi_revision = application::core::velocityExtensionMapRevision(
      11u, 12u, 13u, 14u, 15u, changed_phi, seed);
  EXPECT_NE(phi_revision.key(), revision.key());

  auto changed_seed = seed;
  changed_seed.back() = changed_seed.back() == 0u ? 1u : 0u;
  const auto active_revision =
      application::core::velocityExtensionMapRevision(
          11u, 12u, 13u, 14u, 15u, phi, changed_seed);
  EXPECT_NE(active_revision.key(), revision.key());

  const auto geometry_revision =
      application::core::velocityExtensionMapRevision(
          16u, 12u, 13u, 14u, 15u, phi, seed);
  const auto topology_revision =
      application::core::velocityExtensionMapRevision(
          11u, 17u, 13u, 14u, 15u, phi, seed);
  const auto ownership_revision =
      application::core::velocityExtensionMapRevision(
          11u, 12u, 18u, 14u, 15u, phi, seed);
  const auto surface_revision =
      application::core::velocityExtensionMapRevision(
          11u, 12u, 13u, 14u, 19u, phi, seed);
  EXPECT_NE(geometry_revision.key(), revision.key());
  EXPECT_NE(topology_revision.key(), revision.key());
  EXPECT_NE(ownership_revision.key(), revision.key());
  EXPECT_NE(surface_revision.key(), revision.key());
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     VelocityExtensionMapArtifactRetainsRowsAndRevisionChanges)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  const auto mesh = makeWorkflowSkewedExtensionTriangleMesh();
  std::vector<double> phi(mesh->n_vertices(), 0.0);
  std::vector<double> source(mesh->n_vertices() * 2u, 0.0);
  std::vector<std::uint8_t> active(mesh->n_vertices(), 0u);
  for (std::size_t vertex = 0u; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    phi[vertex] = point[0] - 0.5;
    active[vertex] = phi[vertex] <= 0.0 ? 1u : 0u;
    source[2u * vertex] = 0.2;
    source[2u * vertex + 1u] = -0.1;
  }
  const auto first_revision = application::core::velocityExtensionMapRevision(
      11u, 12u, 13u, 14u, 15u, phi, active);
  const auto first = application::core::buildVelocityExtensionMapSnapshot(
      *mesh,
      svmp::MeshComm::self(),
      first_revision,
      phi,
      source,
      2u,
      active,
      2u,
      2u,
      1,
      false,
      std::span<const WallVelocityExtensionConstraint>{});
  ASSERT_TRUE(first);
  ASSERT_EQ(first->report().bounded_fallback_rows, 1u);
  ASSERT_EQ(first->report().coefficient_rejected_rows, 1u);

  const auto output_directory =
      std::filesystem::temp_directory_path() /
      ("svmp_velocity_extension_artifact_" +
       std::to_string(first_revision.key()));
  std::error_code cleanup_error;
  std::filesystem::remove_all(output_directory, cleanup_error);
  ASSERT_FALSE(cleanup_error);
  application::core::VelocityExtensionMapArtifactContext context{
      .level_set_field_name = "phi",
      .source_velocity_field_name = "Velocity",
      .target_velocity_field_name = "LevelSetAdvectionVelocity",
      .geometry_domain_id = "free_surface",
      .operator_tag = "level_set",
      .extension_method = "wall_compatible_normal",
      .retained_side = "LevelSetNegative",
      .accepted_step = 1u,
      .accepted_time = 0.1,
      .time_step = 0.1,
      .state_revision = 101u,
      .isovalue = 0.0,
      .extension_band_layers = 1,
      .enforce_wall_impermeability = false,
      .rank = 0,
      .ranks = 1,
  };
  const auto first_artifact =
      application::core::writeVelocityExtensionMapArtifact(
          output_directory, context, *first);
  ASSERT_TRUE(first_artifact.success) << first_artifact.diagnostic;
  EXPECT_EQ(first_artifact.owner_rows, mesh->n_vertices());
  EXPECT_EQ(first_artifact.constraint_rows, 2u * mesh->n_vertices());
  ASSERT_TRUE(std::filesystem::is_regular_file(first_artifact.path));
  std::ifstream first_input(first_artifact.path);
  ASSERT_TRUE(first_input.is_open());
  const std::string first_json{
      std::istreambuf_iterator<char>(first_input),
      std::istreambuf_iterator<char>()};
  EXPECT_NE(first_json.find("\"schema\":\"svmp.velocity_extension_map.v1\""),
            std::string::npos);
  EXPECT_NE(first_json.find("\"bounded_fallback_rows\":1"),
            std::string::npos);
  EXPECT_NE(first_json.find("\"coefficient_rejected\":true"),
            std::string::npos);
  EXPECT_NE(first_json.find("\"proposed_negative_weight_count\":1"),
            std::string::npos);
  EXPECT_NE(first_json.find("\"previous_available\":false"),
            std::string::npos);

  auto changed_phi = phi;
  changed_phi[1] += 0.05;
  auto changed_source = source;
  for (auto& value : changed_source) {
    value += 0.25;
  }
  const auto second_revision =
      application::core::velocityExtensionMapRevision(
          11u, 12u, 13u, 14u, 15u, changed_phi, active);
  const auto second = application::core::buildVelocityExtensionMapSnapshot(
      *mesh,
      svmp::MeshComm::self(),
      second_revision,
      changed_phi,
      changed_source,
      2u,
      active,
      2u,
      2u,
      1,
      false,
      std::span<const WallVelocityExtensionConstraint>{});
  ASSERT_TRUE(second);
  const auto change = application::core::compareVelocityExtensionMapSnapshots(
      *second, first.get());
  EXPECT_TRUE(change.previous_available);
  EXPECT_TRUE(change.revision_changed);
  EXPECT_TRUE(change.level_set_values_changed);
  EXPECT_EQ(change.common_owner_rows, mesh->n_vertices());
  EXPECT_EQ(change.added_owner_rows, 0u);
  EXPECT_EQ(change.removed_owner_rows, 0u);
  EXPECT_EQ(change.preview_values_compared, 2u * mesh->n_vertices());
  EXPECT_GT(change.preview_l2_change, 0.0);
  EXPECT_GT(change.preview_linf_change, 0.0);

  context.accepted_step = 2u;
  context.accepted_time = 0.2;
  context.state_revision = 102u;
  const auto second_artifact =
      application::core::writeVelocityExtensionMapArtifact(
          output_directory, context, *second, first.get());
  ASSERT_TRUE(second_artifact.success) << second_artifact.diagnostic;
  std::ifstream second_input(second_artifact.path);
  ASSERT_TRUE(second_input.is_open());
  const std::string second_json{
      std::istreambuf_iterator<char>(second_input),
      std::istreambuf_iterator<char>()};
  EXPECT_NE(second_json.find("\"previous_available\":true"),
            std::string::npos);
  EXPECT_NE(second_json.find("\"revision_changed\":true"),
            std::string::npos);
  EXPECT_NE(second_json.find("\"level_set_values\":true"),
            std::string::npos);
  const auto duplicate =
      application::core::writeVelocityExtensionMapArtifact(
          output_directory, context, *second, first.get());
  EXPECT_FALSE(duplicate.success);
  EXPECT_NE(duplicate.diagnostic.find("refuses to replace"),
            std::string::npos);
  cleanup_error.clear();
  EXPECT_EQ(std::filesystem::remove_all(output_directory, cleanup_error), 3u);
  EXPECT_FALSE(cleanup_error);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     NormalBandVelocityExtensionManufacturedRefinementConvergesAndProjectsWalls)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr svmp::label_t kHorizontalWall = 4351;
  constexpr double kInterfaceX = 0.30;
  const double pi = std::acos(-1.0);
  const auto exact_tangential_velocity = [pi](double y) {
    return std::sin(pi * y) + 0.2 * std::cos(2.0 * pi * y);
  };
  const auto exact_normal_velocity = [pi](double y) {
    return 0.15 * std::sin(pi * y);
  };

  struct RefinementError {
    double l2{0.0};
    double linf{0.0};
  };
  std::vector<RefinementError> errors;
  for (const int subdivisions : {8, 16, 32}) {
    SCOPED_TRACE("subdivisions=" + std::to_string(subdivisions));
    const auto mesh = makeWorkflowStructuredQuadMesh(subdivisions);
    for (const auto face : mesh->local_mesh().boundary_faces()) {
      const auto normal = mesh->local_mesh().face_normal(face);
      if (std::abs(normal[1]) > 0.9 * std::abs(normal[0])) {
        mesh->set_boundary_label(face, kHorizontalWall);
      }
    }

    std::vector<double> phi(mesh->n_vertices(), 0.0);
    std::vector<double> source(mesh->n_vertices() * 2u, 0.0);
    std::vector<std::uint8_t> active(mesh->n_vertices(), 0u);
    for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
      const auto point = workflowVertexPoint(*mesh, vertex);
      phi[vertex] = point[0] - kInterfaceX;
      active[vertex] = phi[vertex] <= 0.0 ? 1u : 0u;
      // For the planar interface, n=ex and this smooth field is the exact
      // solution of n.grad(u_ext)=0.  Its second component vanishes at the
      // horizontal walls, so it also manufactures the no-penetration trace
      // without making that component identically zero in the interior.
      source[2u * vertex] = exact_tangential_velocity(point[1]);
      source[2u * vertex + 1u] = exact_normal_velocity(point[1]);
    }

    const std::vector<WallVelocityExtensionConstraint> constraints{{
        .boundary_label = kHorizontalWall,
        .constrained_components = {false, true, false}}};
    std::vector<double> extended;
    const auto report = extendVelocityInLevelSetNormalBand(
        *mesh,
        svmp::MeshComm::self(),
        phi,
        source,
        /*source_components=*/2u,
        active,
        /*target_components=*/2u,
        /*copy_components=*/2u,
        /*band_layers=*/subdivisions,
        /*enforce_wall_impermeability=*/true,
        std::span<const WallVelocityExtensionConstraint>(constraints),
        extended);

    EXPECT_EQ(report.vertices_outside_band, 0u);
    EXPECT_GT(report.wall_projected_vertices, 0u);
    EXPECT_NEAR(report.max_wall_normal_velocity, 0.0, 1.0e-13);
    double squared_error = 0.0;
    double max_error = 0.0;
    std::size_t dry_vertices = 0u;
    for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
      const auto point = workflowVertexPoint(*mesh, vertex);
      const std::array<double, 2> exact{{
          exact_tangential_velocity(point[1]),
          exact_normal_velocity(point[1])}};
      if (active[vertex] != 0u) {
        EXPECT_NEAR(extended[2u * vertex], exact[0], 1.0e-14);
        EXPECT_NEAR(extended[2u * vertex + 1u], exact[1], 1.0e-14);
        continue;
      }
      for (std::size_t component = 0; component < 2u; ++component) {
        const double error =
            std::abs(extended[2u * vertex + component] - exact[component]);
        squared_error += error * error;
        max_error = std::max(max_error, error);
      }
      if (point[1] <= 1.0e-14 || point[1] >= 1.0 - 1.0e-14) {
        EXPECT_NEAR(extended[2u * vertex + 1u], 0.0, 1.0e-13);
      }
      ++dry_vertices;
    }
    ASSERT_GT(dry_vertices, 0u);
    errors.push_back({
        .l2 = std::sqrt(
            squared_error / static_cast<double>(2u * dry_vertices)),
        .linf = max_error});
    RecordProperty("extension_l2_N" + std::to_string(subdivisions),
                   std::to_string(errors.back().l2));
    RecordProperty("extension_linf_N" + std::to_string(subdivisions),
                   std::to_string(errors.back().linf));
  }

  ASSERT_EQ(errors.size(), 3u);
  for (std::size_t level = 1; level < errors.size(); ++level) {
    constexpr double exact_reproduction_tolerance = 5.0e-13;
    if (errors[level - 1u].linf <= exact_reproduction_tolerance &&
        errors[level].linf <= exact_reproduction_tolerance) {
      RecordProperty("extension_exact_reproduction_level" +
                         std::to_string(level),
                     "true");
      EXPECT_LE(errors[level - 1u].l2, exact_reproduction_tolerance);
      EXPECT_LE(errors[level].l2, exact_reproduction_tolerance);
      continue;
    }
    ASSERT_GT(errors[level].l2, 0.0);
    ASSERT_GT(errors[level].linf, 0.0);
    const double l2_rate =
        std::log(errors[level - 1u].l2 / errors[level].l2) /
        std::log(2.0);
    const double linf_rate =
        std::log(errors[level - 1u].linf / errors[level].linf) /
        std::log(2.0);
    RecordProperty("extension_l2_rate_level" + std::to_string(level),
                   std::to_string(l2_rate));
    RecordProperty("extension_linf_rate_level" + std::to_string(level),
                   std::to_string(linf_rate));
    EXPECT_GT(l2_rate, 0.75)
        << "coarse=" << errors[level - 1u].l2
        << " fine=" << errors[level].l2;
    EXPECT_GT(linf_rate, 0.60)
        << "coarse=" << errors[level - 1u].linf
        << " fine=" << errors[level].linf;
  }
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     NormalBandVelocityExtensionBandSweepKeepsFallbackRowsBounded)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr int subdivisions = 8;
  constexpr double tangential_velocity = 0.2;
  constexpr double normal_velocity = -0.1;
  const auto mesh = makeWorkflowStructuredQuadMesh(subdivisions);
  std::vector<double> phi(mesh->n_vertices(), 0.0);
  std::vector<double> source(mesh->n_vertices() * 2u, 0.0);
  std::vector<std::uint8_t> active(mesh->n_vertices(), 0u);
  for (std::size_t vertex = 0u; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    const double dx = point[0] - 0.35;
    const double dy = point[1] - 0.50;
    phi[vertex] = std::sqrt(dx * dx + dy * dy) - 0.22;
    active[vertex] = phi[vertex] <= 0.0 ? 1u : 0u;
    source[2u * vertex] = tangential_velocity;
    source[2u * vertex + 1u] = normal_velocity;
  }

  std::vector<std::size_t> vertices_outside_band;
  for (const int band_layers : {1, 2, 4, 8}) {
    SCOPED_TRACE("band_layers=" + std::to_string(band_layers));
    std::vector<double> extended;
    const auto report = extendVelocityInLevelSetNormalBand(
        *mesh,
        svmp::MeshComm::self(),
        phi,
        source,
        /*source_components=*/2u,
        active,
        /*target_components=*/2u,
        /*copy_components=*/2u,
        band_layers,
        /*enforce_wall_impermeability=*/false,
        std::span<const WallVelocityExtensionConstraint>{},
        extended);
    vertices_outside_band.push_back(report.vertices_outside_band);
    EXPECT_EQ(report.bounded_fallback_rows, 0u);
    EXPECT_LE(report.max_abs_graph_coefficient,
              1.0 + kVelocityExtensionRowTolerance);
    EXPECT_LE(report.max_graph_row_l1,
              1.0 + kVelocityExtensionRowTolerance);
    EXPECT_LE(report.max_graph_row_sum_error,
              kVelocityExtensionRowTolerance);
    EXPECT_LE(report.max_negative_graph_coefficient,
              kVelocityExtensionCoefficientTolerance);
    EXPECT_LE(report.max_constant_reproduction_error,
              kVelocityExtensionRowTolerance);
    EXPECT_LE(report.max_extended_speed,
              report.max_seed_speed + 1.0e-12);

    std::size_t observed_outside = 0u;
    for (std::size_t vertex = 0u;
         vertex < mesh->n_vertices(); ++vertex) {
      const bool outside =
          extended[2u * vertex] == 0.0 &&
          extended[2u * vertex + 1u] == 0.0;
      if (outside) {
        ++observed_outside;
        continue;
      }
      EXPECT_NEAR(extended[2u * vertex],
                  tangential_velocity,
                  1.0e-12);
      EXPECT_NEAR(extended[2u * vertex + 1u],
                  normal_velocity,
                  1.0e-12);
    }
    EXPECT_EQ(observed_outside, report.vertices_outside_band);
    RecordProperty(
        "extension_band_" + std::to_string(band_layers) +
            "_outside_vertices",
        std::to_string(report.vertices_outside_band));
    RecordProperty(
        "extension_band_" + std::to_string(band_layers) +
            "_bounded_fallback_rows",
        std::to_string(report.bounded_fallback_rows));
    RecordProperty(
        "extension_band_" + std::to_string(band_layers) +
            "_wet_to_dry_speed_ratio",
        std::to_string(
            report.max_extended_speed / report.max_seed_speed));
  }

  ASSERT_EQ(vertices_outside_band.size(), 4u);
  EXPECT_GT(vertices_outside_band[0], vertices_outside_band[1]);
  EXPECT_GT(vertices_outside_band[1], vertices_outside_band[2]);
  EXPECT_GT(vertices_outside_band[2], vertices_outside_band[3]);
  EXPECT_EQ(vertices_outside_band.back(), 0u);

  const auto fallback_mesh = makeWorkflowSkewedExtensionTriangleMesh();
  std::vector<double> fallback_phi(fallback_mesh->n_vertices(), 0.0);
  std::vector<double> fallback_source(
      fallback_mesh->n_vertices() * 2u, 0.0);
  std::vector<std::uint8_t> fallback_active(
      fallback_mesh->n_vertices(), 0u);
  for (std::size_t vertex = 0u;
       vertex < fallback_mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*fallback_mesh, vertex);
    fallback_phi[vertex] = point[0] - 0.5;
    fallback_active[vertex] =
        fallback_phi[vertex] <= 0.0 ? 1u : 0u;
    fallback_source[2u * vertex] = tangential_velocity;
    fallback_source[2u * vertex + 1u] = normal_velocity;
  }
  std::vector<double> fallback_extension;
  std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow>
      fallback_rows;
  std::vector<std::int64_t> fallback_components;
  std::vector<application::core::VelocityExtensionGraphRowDiagnostic>
      fallback_diagnostics;
  const auto fallback_report = extendVelocityInLevelSetNormalBand(
      *fallback_mesh,
      svmp::MeshComm::self(),
      fallback_phi,
      fallback_source,
      /*source_components=*/2u,
      fallback_active,
      /*target_components=*/2u,
      /*copy_components=*/2u,
      /*band_layers=*/1,
      /*enforce_wall_impermeability=*/false,
      std::span<const WallVelocityExtensionConstraint>{},
      fallback_extension,
      &fallback_rows,
      &fallback_components,
      &fallback_diagnostics);
  EXPECT_EQ(fallback_report.vertices_outside_band, 0u);
  EXPECT_EQ(fallback_report.regression_candidate_rows, 1u);
  EXPECT_EQ(fallback_report.regression_accepted_rows, 0u);
  EXPECT_EQ(fallback_report.bounded_fallback_rows, 1u);
  EXPECT_EQ(fallback_report.condition_rejected_rows, 0u);
  EXPECT_EQ(fallback_report.coefficient_rejected_rows, 1u);
  EXPECT_LE(fallback_report.max_abs_graph_coefficient,
            1.0 + kVelocityExtensionRowTolerance);
  EXPECT_LE(fallback_report.max_graph_row_l1,
            1.0 + kVelocityExtensionRowTolerance);
  EXPECT_LE(fallback_report.max_graph_row_sum_error,
            kVelocityExtensionRowTolerance);
  EXPECT_LE(fallback_report.max_negative_graph_coefficient,
            kVelocityExtensionCoefficientTolerance);
  EXPECT_LE(fallback_report.max_extended_speed,
            fallback_report.max_seed_speed + 1.0e-12);
  const auto fallback_diagnostic = std::find_if(
      fallback_diagnostics.begin(),
      fallback_diagnostics.end(),
      [](const auto& diagnostic) {
        return diagnostic.disposition ==
               application::core::VelocityExtensionRowDisposition::
                   BoundedFallback;
      });
  ASSERT_NE(fallback_diagnostic, fallback_diagnostics.end());
  EXPECT_TRUE(fallback_diagnostic->regression_attempted);
  EXPECT_FALSE(fallback_diagnostic->regression_accepted);
  EXPECT_TRUE(fallback_diagnostic->bounded_fallback_used);
  EXPECT_FALSE(fallback_diagnostic->condition_rejected);
  EXPECT_TRUE(fallback_diagnostic->coefficient_rejected);
  EXPECT_GT(fallback_diagnostic->proposed_negative_weight_count, 0u);
  EXPECT_GT(fallback_diagnostic->proposed_max_negative_coefficient, 0.0);
  EXPECT_EQ(fallback_diagnostic->negative_weight_count, 0u);
  EXPECT_NEAR(fallback_diagnostic->coefficient_sum, 1.0,
              kVelocityExtensionRowTolerance);
  EXPECT_LE(fallback_diagnostic->coefficient_l1,
            1.0 + kVelocityExtensionRowTolerance);
  EXPECT_LE(fallback_diagnostic->preview_amplification,
            1.0 + 1.0e-12);
  ASSERT_EQ(fallback_extension.size(), fallback_source.size());
  for (std::size_t index = 0u;
       index < fallback_extension.size(); index += 2u) {
    EXPECT_NEAR(fallback_extension[index],
                tangential_velocity,
                1.0e-12);
    EXPECT_NEAR(fallback_extension[index + 1u],
                normal_velocity,
                1.0e-12);
  }
  RecordProperty("extension_forced_bounded_fallback_rows",
                 std::to_string(fallback_report.bounded_fallback_rows));

  const auto ill_conditioned_mesh =
      makeWorkflowSkewedExtensionTriangleMesh(1.00001);
  std::vector<double> ill_conditioned_phi(
      ill_conditioned_mesh->n_vertices(), 0.0);
  std::vector<double> ill_conditioned_source(
      ill_conditioned_mesh->n_vertices() * 2u, 0.0);
  std::vector<std::uint8_t> ill_conditioned_active(
      ill_conditioned_mesh->n_vertices(), 0u);
  for (std::size_t vertex = 0u;
       vertex < ill_conditioned_mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*ill_conditioned_mesh, vertex);
    ill_conditioned_phi[vertex] = point[0] - 0.5;
    ill_conditioned_active[vertex] =
        ill_conditioned_phi[vertex] <= 0.0 ? 1u : 0u;
    ill_conditioned_source[2u * vertex] = tangential_velocity;
    ill_conditioned_source[2u * vertex + 1u] = normal_velocity;
  }
  std::vector<double> ill_conditioned_extension;
  std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow>
      ill_conditioned_rows;
  std::vector<std::int64_t> ill_conditioned_components;
  std::vector<application::core::VelocityExtensionGraphRowDiagnostic>
      ill_conditioned_diagnostics;
  const auto ill_conditioned_report = extendVelocityInLevelSetNormalBand(
      *ill_conditioned_mesh,
      svmp::MeshComm::self(),
      ill_conditioned_phi,
      ill_conditioned_source,
      /*source_components=*/2u,
      ill_conditioned_active,
      /*target_components=*/2u,
      /*copy_components=*/2u,
      /*band_layers=*/1,
      /*enforce_wall_impermeability=*/false,
      std::span<const WallVelocityExtensionConstraint>{},
      ill_conditioned_extension,
      &ill_conditioned_rows,
      &ill_conditioned_components,
      &ill_conditioned_diagnostics);
  EXPECT_EQ(ill_conditioned_report.regression_candidate_rows, 1u);
  EXPECT_EQ(ill_conditioned_report.regression_accepted_rows, 0u);
  EXPECT_EQ(ill_conditioned_report.bounded_fallback_rows, 1u);
  EXPECT_EQ(ill_conditioned_report.condition_rejected_rows, 1u);
  EXPECT_EQ(ill_conditioned_report.coefficient_rejected_rows, 0u);
  EXPECT_LE(ill_conditioned_report.max_graph_row_l1,
            1.0 + kVelocityExtensionRowTolerance);
  EXPECT_LE(ill_conditioned_report.max_graph_row_sum_error,
            kVelocityExtensionRowTolerance);
  EXPECT_LE(ill_conditioned_report.max_extended_speed,
            ill_conditioned_report.max_seed_speed + 1.0e-12);
  const auto condition_diagnostic = std::find_if(
      ill_conditioned_diagnostics.begin(),
      ill_conditioned_diagnostics.end(),
      [](const auto& diagnostic) {
        return diagnostic.condition_rejected;
      });
  ASSERT_NE(condition_diagnostic, ill_conditioned_diagnostics.end());
  EXPECT_EQ(condition_diagnostic->disposition,
            application::core::VelocityExtensionRowDisposition::
                BoundedFallback);
  EXPECT_TRUE(condition_diagnostic->bounded_fallback_used);
  EXPECT_FALSE(condition_diagnostic->coefficient_rejected);
  EXPECT_GT(condition_diagnostic->condition_estimate,
            kVelocityExtensionMaxRegressionCondition);
  EXPECT_EQ(condition_diagnostic->negative_weight_count, 0u);
  EXPECT_LE(condition_diagnostic->coefficient_l1,
            1.0 + kVelocityExtensionRowTolerance);
  RecordProperty("extension_condition_fallback_rows",
                 std::to_string(
                     ill_conditioned_report.condition_rejected_rows));
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     NormalBandVelocityExtensionDoesNotSwitchDisconnectedComponents)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  const auto mesh = makeDisconnectedWorkflowQuadPairMesh();
  std::vector<double> phi(mesh->n_vertices(), 0.0);
  std::vector<double> source(mesh->n_vertices() * 2u, 0.0);
  std::vector<std::uint8_t> active(mesh->n_vertices(), 0u);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    phi[vertex] = point[0] - 0.25;
    active[vertex] = phi[vertex] <= 0.0 ? 1u : 0u;
    const bool first_component = vertex < 4u;
    source[2u * vertex] = first_component ? 2.0 : -7.0;
    source[2u * vertex + 1u] = first_component ? 3.0 : 11.0;
  }

  std::vector<double> extended;
  const auto report = extendVelocityInLevelSetNormalBand(
      *mesh,
      phi,
      source,
      /*source_components=*/2u,
      active,
      /*target_components=*/2u,
      /*copy_components=*/2u,
      /*band_layers=*/1,
      /*enforce_wall_impermeability=*/false,
      /*wall_boundary_labels=*/{},
      extended);

  EXPECT_EQ(report.vertices_outside_band, 0u);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const bool first_component = vertex < 4u;
    EXPECT_NEAR(extended[2u * vertex], first_component ? 2.0 : -7.0,
                1.0e-12)
        << "vertex " << vertex;
    EXPECT_NEAR(extended[2u * vertex + 1u], first_component ? 3.0 : 11.0,
                1.0e-12)
        << "vertex " << vertex;
  }
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     NormalBandVelocityExtensionIgnoresReversedComponentNumbering)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  struct ExtensionResult {
    WallCompatibleVelocityExtensionResult report{};
    std::vector<std::array<double, 4>> samples{};
    std::int64_t left_component{-1};
    std::int64_t right_component{-1};
  };
  const auto run = [](const std::shared_ptr<svmp::Mesh>& mesh) {
    std::vector<double> phi(mesh->n_vertices(), 0.0);
    std::vector<double> source(mesh->n_vertices() * 2u, 0.0);
    std::vector<std::uint8_t> active(mesh->n_vertices(), 0u);
    for (std::size_t vertex = 0u; vertex < mesh->n_vertices(); ++vertex) {
      const auto point = workflowVertexPoint(*mesh, vertex);
      phi[vertex] = std::min(point[0] - 0.25,
                             4.0 - point[0] - 0.35);
      active[vertex] = phi[vertex] <= 0.0 ? 1u : 0u;
      const bool left = point[0] <= 2.0;
      source[2u * vertex] =
          left ? 2.0 + 3.0 * point[1] : -7.0 + 2.0 * point[1];
      source[2u * vertex + 1u] =
          left ? -1.0 + 0.5 * point[1] : 11.0 - 4.0 * point[1];
    }

    std::vector<double> extended;
    std::vector<std::int64_t> component_assignment;
    ExtensionResult result;
    result.report = extendVelocityInLevelSetNormalBand(
        *mesh,
        svmp::MeshComm::self(),
        phi,
        source,
        /*source_components=*/2u,
        active,
        /*target_components=*/2u,
        /*copy_components=*/2u,
        /*band_layers=*/4,
        /*enforce_wall_impermeability=*/false,
        std::span<const WallVelocityExtensionConstraint>{},
        extended,
        nullptr,
        &component_assignment,
        nullptr);
    for (std::size_t vertex = 0u; vertex < mesh->n_vertices(); ++vertex) {
      const auto point = workflowVertexPoint(*mesh, vertex);
      result.samples.push_back({{
          point[0],
          point[1],
          extended[2u * vertex],
          extended[2u * vertex + 1u],
      }});
      if (point[0] == 0.0 && result.left_component < 0) {
        result.left_component = component_assignment[vertex];
      }
      if (point[0] == 4.0 && result.right_component < 0) {
        result.right_component = component_assignment[vertex];
      }
    }
    std::sort(result.samples.begin(), result.samples.end());
    return result;
  };

  const auto forward = run(makeWorkflowFourQuadStripMesh());
  const auto reversed = run(makeWorkflowFourQuadStripMesh(true));
  ASSERT_EQ(forward.report.vertices_outside_band, 0u);
  ASSERT_EQ(reversed.report.vertices_outside_band, 0u);
  ASSERT_GT(forward.report.component_collision_vertices, 0u);
  EXPECT_EQ(reversed.report.component_collision_vertices,
            forward.report.component_collision_vertices);
  ASSERT_GE(forward.left_component, 0);
  ASSERT_GE(forward.right_component, 0);
  ASSERT_GE(reversed.left_component, 0);
  ASSERT_GE(reversed.right_component, 0);
  EXPECT_LT(forward.left_component, forward.right_component);
  EXPECT_GT(reversed.left_component, reversed.right_component);
  ASSERT_EQ(reversed.samples.size(), forward.samples.size());
  for (std::size_t sample = 0u; sample < forward.samples.size(); ++sample) {
    ASSERT_EQ(reversed.samples[sample][0], forward.samples[sample][0]);
    ASSERT_EQ(reversed.samples[sample][1], forward.samples[sample][1]);
    EXPECT_NEAR(reversed.samples[sample][2],
                forward.samples[sample][2],
                1.0e-12);
    EXPECT_NEAR(reversed.samples[sample][3],
                forward.samples[sample][3],
                1.0e-12);

    const double x = forward.samples[sample][0];
    const double y = forward.samples[sample][1];
    const bool left_branch = x < 1.95;
    EXPECT_NEAR(forward.samples[sample][2],
                left_branch ? 2.0 + 3.0 * y : -7.0 + 2.0 * y,
                1.0e-12);
    EXPECT_NEAR(forward.samples[sample][3],
                left_branch ? -1.0 + 0.5 * y : 11.0 - 4.0 * y,
                1.0e-12);
  }
  RecordProperty("forward_left_component",
                 std::to_string(forward.left_component));
  RecordProperty("forward_right_component",
                 std::to_string(forward.right_component));
  RecordProperty("reversed_left_component",
                 std::to_string(reversed.left_component));
  RecordProperty("reversed_right_component",
                 std::to_string(reversed.right_component));
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     NormalBandVelocityExtensionFailsClosedOnEquidistantComponentBands)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  const auto mesh = makeWorkflowFourQuadStripMesh();
  std::vector<double> phi(mesh->n_vertices(), 0.0);
  std::vector<double> source(mesh->n_vertices() * 2u, 0.0);
  std::vector<std::uint8_t> active(mesh->n_vertices(), 0u);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    // Two disconnected active components live at opposite ends of one
    // connected background mesh.  Their two-layer graph bands collide at
    // x=2, where the old unlabeled propagation blended both source fields.
    // Place both interfaces one quarter cell inside their end cells so that
    // x=2 is also equidistant from the two geometric interfaces.
    phi[vertex] = std::min(point[0] - 0.25,
                           4.0 - point[0] - 0.25);
    active[vertex] = phi[vertex] <= 0.0 ? 1u : 0u;
    const bool left_branch = point[0] <= 2.0;
    source[2u * vertex] = left_branch
                              ? 2.0 + 3.0 * point[1]
                              : -7.0 + 2.0 * point[1];
    source[2u * vertex + 1u] = left_branch
                                   ? -1.0 + 0.5 * point[1]
                                   : 11.0 - 4.0 * point[1];
  }

  std::vector<double> extended;
  EXPECT_THROW(
      {
        try {
          (void)extendVelocityInLevelSetNormalBand(
              *mesh,
              phi,
              source,
              /*source_components=*/2u,
              active,
              /*target_components=*/2u,
              /*copy_components=*/2u,
              /*band_layers=*/2,
              /*enforce_wall_impermeability=*/false,
              /*wall_boundary_labels=*/{},
              extended);
        } catch (const std::runtime_error& error) {
          EXPECT_NE(std::string(error.what()).find(
                        "unresolved equidistant active-component collision"),
                    std::string::npos);
          throw;
        }
      },
      std::runtime_error);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     NormalBandVelocityExtensionHonorsGraphLayerCutoff)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  const auto mesh = makeWorkflowThreeQuadStripMesh();
  std::vector<double> phi(mesh->n_vertices(), 0.0);
  std::vector<double> source(mesh->n_vertices() * 2u, 0.0);
  std::vector<std::uint8_t> active(mesh->n_vertices(), 0u);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    phi[vertex] = point[0] - 0.25;
    active[vertex] = phi[vertex] <= 0.0 ? 1u : 0u;
    source[2u * vertex] = 4.0;
    source[2u * vertex + 1u] = -2.0;
  }

  std::vector<double> extended;
  const auto report = extendVelocityInLevelSetNormalBand(
      *mesh,
      phi,
      source,
      /*source_components=*/2u,
      active,
      /*target_components=*/2u,
      /*copy_components=*/2u,
      /*band_layers=*/1,
      /*enforce_wall_impermeability=*/false,
      /*wall_boundary_labels=*/{},
      extended);

  EXPECT_EQ(report.vertices_outside_band, 4u);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    if (point[0] <= 1.0) {
      EXPECT_NEAR(extended[2u * vertex], 4.0, 1.0e-12)
          << "vertex " << vertex;
      EXPECT_NEAR(extended[2u * vertex + 1u], -2.0, 1.0e-12)
          << "vertex " << vertex;
    } else {
      EXPECT_NEAR(extended[2u * vertex], 0.0, 1.0e-12)
          << "vertex " << vertex;
      EXPECT_NEAR(extended[2u * vertex + 1u], 0.0, 1.0e-12)
          << "vertex " << vertex;
    }
  }
#endif
}

application::core::LevelSetAuthoritativeFunctionalValue
makeMaintenanceFunctionalValue(
    double total_potential,
    std::uint64_t snapshot_revision,
    std::uint64_t mesh_topology_revision = 19u,
    std::uint64_t cut_topology_revision = 23u,
    int interface_marker = 407)
{
  return application::core::LevelSetAuthoritativeFunctionalValue{
      .interface_marker = interface_marker,
      .snapshot_revision = snapshot_revision,
      .mesh_topology_revision = mesh_topology_revision,
      .cut_topology_revision = cut_topology_revision,
      .liquid_volume = 1.25,
      .liquid_gas_area = 2.5,
      .wetted_wall_area = 0.75,
      .contact_measure = 0.5,
      .surface_energy = 3.0,
      .young_wall_energy = -0.25,
      .volume_constraint_potential = total_potential - 2.75,
      .total_potential = total_potential,
  };
}

application::core::LevelSetAuthoritativeFunctionalValue
makeModeledMaintenanceFunctionalValue(
    double total_potential,
    std::uint64_t snapshot_revision,
    double kinetic_energy,
    double gravitational_energy,
    double gravitational_potential_power)
{
  auto value = makeMaintenanceFunctionalValue(
      total_potential, snapshot_revision);
  value.kinetic_energy = kinetic_energy;
  value.gravitational_energy = gravitational_energy;
  value.gravitational_potential_power =
      gravitational_potential_power;
  value.modeled_stored_energy =
      kinetic_energy + gravitational_energy +
      value.surface_energy + value.young_wall_energy;
  return value;
}

application::core::LevelSetMaintenanceWorkTransaction
makeMaintenanceWorkTransaction(std::uint64_t transaction_id)
{
  return application::core::LevelSetMaintenanceWorkTransaction{
      .transaction_id = transaction_id,
      .step = 8u,
      .attempt = 2u,
      .time = 0.4,
      .dt = 0.05,
      .declared_stage =
          application::core::LevelSetMaintenanceDeclaredStage::
              AcceptedEndpointPostStep,
      .extension_map_revision = 313u,
  };
}

TEST(ApplicationDriverLevelSetWorkflows,
     MaintenanceWorkLedgerPublishesReinitializationOnlyAtCommit)
{
  application::core::LevelSetMaintenanceWorkLedger ledger;
  ledger.beginTransaction(makeMaintenanceWorkTransaction(101u));
  ledger.stageRow(
      application::core::LevelSetMaintenanceWorkSubstage::
          Reinitialization,
      1001u,
      1002u,
      {makeMaintenanceFunctionalValue(2.0, 501u)},
      {makeMaintenanceFunctionalValue(2.125, 502u)});

  ASSERT_EQ(ledger.trialRows().size(), 1u);
  EXPECT_TRUE(ledger.acceptedRows().empty());
  EXPECT_EQ(
      ledger.trialRows().front().status,
      application::core::LevelSetMaintenanceWorkStatus::Trial);
  EXPECT_DOUBLE_EQ(
      ledger.trialRows().front().accepted_numerical_work, 0.0);

  ledger.commitTransaction();
  ASSERT_EQ(ledger.acceptedRows().size(), 1u);
  const auto& row = ledger.acceptedRows().front();
  EXPECT_EQ(
      row.status,
      application::core::LevelSetMaintenanceWorkStatus::Accepted);
  EXPECT_EQ(
      row.substage,
      application::core::LevelSetMaintenanceWorkSubstage::
          Reinitialization);
  EXPECT_EQ(row.step, 8u);
  EXPECT_EQ(row.attempt, 2u);
  EXPECT_DOUBLE_EQ(row.time, 0.4);
  EXPECT_DOUBLE_EQ(row.dt, 0.05);
  ASSERT_TRUE(row.extension_map_revision_before.has_value());
  ASSERT_TRUE(row.extension_map_revision_after.has_value());
  EXPECT_EQ(*row.extension_map_revision_before, 313u);
  EXPECT_EQ(*row.extension_map_revision_after, 313u);
  EXPECT_DOUBLE_EQ(row.numerical_work, 0.125);
  EXPECT_DOUBLE_EQ(row.accepted_numerical_work, 0.125);
  ASSERT_EQ(ledger.acceptedAttempts().size(), 1u);
  EXPECT_EQ(
      ledger.acceptedAttempts().front().status,
      application::core::LevelSetMaintenanceWorkStatus::Accepted);
  EXPECT_EQ(ledger.acceptedAttempts().front().row_count, 1u);
  EXPECT_DOUBLE_EQ(
      ledger.acceptedAttempts().front().accepted_numerical_work,
      0.125);
}

TEST(ApplicationDriverLevelSetWorkflows,
     MaintenanceWorkLedgerKeepsReinitializationAndCorrectionAdditive)
{
  application::core::LevelSetMaintenanceWorkLedger ledger;
  ledger.beginTransaction(makeMaintenanceWorkTransaction(102u));
  ledger.stageRow(
      application::core::LevelSetMaintenanceWorkSubstage::
          Reinitialization,
      2001u,
      2002u,
      {makeMaintenanceFunctionalValue(3.0, 601u)},
      {makeMaintenanceFunctionalValue(3.25, 602u)});
  ledger.stageRow(
      application::core::LevelSetMaintenanceWorkSubstage::
          GlobalCorrection,
      2002u,
      2003u,
      {makeMaintenanceFunctionalValue(3.25, 602u)},
      {makeMaintenanceFunctionalValue(3.10, 603u)});
  ledger.commitTransaction();

  ASSERT_EQ(ledger.acceptedRows().size(), 2u);
  EXPECT_EQ(
      ledger.acceptedRows()[0].substage,
      application::core::LevelSetMaintenanceWorkSubstage::
          Reinitialization);
  EXPECT_EQ(
      ledger.acceptedRows()[1].substage,
      application::core::LevelSetMaintenanceWorkSubstage::
          GlobalCorrection);
  EXPECT_EQ(
      ledger.acceptedRows()[0].algebraic_state_revision_after,
      ledger.acceptedRows()[1].algebraic_state_revision_before);
  EXPECT_EQ(
      ledger.acceptedRows()[0].after,
      ledger.acceptedRows()[1].before);
  const auto accepted_sum =
      ledger.acceptedRows()[0].accepted_numerical_work +
      ledger.acceptedRows()[1].accepted_numerical_work;
  EXPECT_NEAR(accepted_sum, 0.10, 1.0e-15);
}

TEST(ApplicationDriverLevelSetWorkflows,
     MaintenanceWorkLedgerSeparatesModeledStoredEnergyFromConstraintPotential)
{
  application::core::LevelSetMaintenanceWorkLedger ledger;
  ledger.beginTransaction(makeMaintenanceWorkTransaction(109u));
  ledger.stageRow(
      application::core::LevelSetMaintenanceWorkSubstage::
          Reinitialization,
      2501u,
      2502u,
      {makeModeledMaintenanceFunctionalValue(
          /*total_potential=*/4.0,
          /*snapshot_revision=*/651u,
          /*kinetic_energy=*/1.0,
          /*gravitational_energy=*/2.0,
          /*gravitational_potential_power=*/-0.5)},
      {makeModeledMaintenanceFunctionalValue(
          /*total_potential=*/4.0,
          /*snapshot_revision=*/652u,
          /*kinetic_energy=*/1.5,
          /*gravitational_energy=*/1.8,
          /*gravitational_potential_power=*/-0.4)});

  ASSERT_EQ(ledger.trialRows().size(), 1u);
  EXPECT_DOUBLE_EQ(ledger.trialRows().front().numerical_work, 0.0);
  ASSERT_TRUE(
      ledger.trialRows()
          .front()
          .modeled_energy_numerical_work.has_value());
  EXPECT_NEAR(
      *ledger.trialRows()
           .front()
           .modeled_energy_numerical_work,
      0.3,
      1.0e-15);
  ledger.commitTransaction();

  ASSERT_EQ(ledger.acceptedRows().size(), 1u);
  ASSERT_TRUE(
      ledger.acceptedRows()
          .front()
          .accepted_modeled_energy_numerical_work.has_value());
  EXPECT_NEAR(
      *ledger.acceptedRows()
           .front()
           .accepted_modeled_energy_numerical_work,
      0.3,
      1.0e-15);
  ASSERT_EQ(ledger.acceptedAttempts().size(), 1u);
  ASSERT_TRUE(
      ledger.acceptedAttempts()
          .front()
          .accepted_modeled_energy_numerical_work.has_value());
  EXPECT_NEAR(
      *ledger.acceptedAttempts()
           .front()
           .accepted_modeled_energy_numerical_work,
      0.3,
      1.0e-15);
  const auto& reinitialization_breakdown =
      ledger.acceptedAttempts()
          .front()
          .modeled_energy_breakdown;
  EXPECT_EQ(
      reinitialization_breakdown.reinitialization.row_count, 1u);
  ASSERT_TRUE(
      reinitialization_breakdown.reinitialization
          .modeled_energy_change.has_value());
  ASSERT_TRUE(
      reinitialization_breakdown.reinitialization
          .accepted_modeled_energy_change.has_value());
  EXPECT_NEAR(
      *reinitialization_breakdown.reinitialization
           .modeled_energy_change,
      0.3,
      1.0e-15);
  EXPECT_NEAR(
      *reinitialization_breakdown.reinitialization
           .accepted_modeled_energy_change,
      0.3,
      1.0e-15);
  EXPECT_EQ(
      reinitialization_breakdown.numerical_maintenance_total
          .row_count,
      1u);
  ASSERT_TRUE(
      reinitialization_breakdown.numerical_maintenance_total
          .accepted_modeled_energy_change.has_value());
  EXPECT_NEAR(
      *reinitialization_breakdown.numerical_maintenance_total
           .accepted_modeled_energy_change,
      0.3,
      1.0e-15);
  EXPECT_EQ(reinitialization_breakdown.transport.row_count, 0u);
  EXPECT_FALSE(
      reinitialization_breakdown.transport
          .modeled_energy_change.has_value());

  application::core::LevelSetMaintenanceWorkLedger staged_ledger;
  auto staged_transaction = makeMaintenanceWorkTransaction(112u);
  staged_transaction.declared_stage =
      application::core::LevelSetMaintenanceDeclaredStage::
          ProspectiveAcceptedEndpoint;
  staged_ledger.beginTransaction(staged_transaction);
  const auto before_transport =
      makeModeledMaintenanceFunctionalValue(
          /*total_potential=*/4.0,
          /*snapshot_revision=*/661u,
          /*kinetic_energy=*/1.0,
          /*gravitational_energy=*/2.0,
          /*gravitational_potential_power=*/-0.5);
  const auto after_transport =
      makeModeledMaintenanceFunctionalValue(
          /*total_potential=*/4.0,
          /*snapshot_revision=*/662u,
          /*kinetic_energy=*/1.4,
          /*gravitational_energy=*/1.8,
          /*gravitational_potential_power=*/-0.4);
  const auto after_limiting =
      makeModeledMaintenanceFunctionalValue(
          /*total_potential=*/4.0,
          /*snapshot_revision=*/663u,
          /*kinetic_energy=*/1.2,
          /*gravitational_energy=*/1.7,
          /*gravitational_potential_power=*/-0.35);
  staged_ledger.stageRow(
      application::core::LevelSetMaintenanceWorkSubstage::Transport,
      2551u,
      2552u,
      {before_transport},
      {after_transport});
  staged_ledger.stageRow(
      application::core::LevelSetMaintenanceWorkSubstage::Limiting,
      2552u,
      2553u,
      {after_transport},
      {after_limiting});
  staged_ledger.commitTransaction();

  ASSERT_EQ(staged_ledger.acceptedAttempts().size(), 1u);
  const auto& staged_attempt =
      staged_ledger.acceptedAttempts().front();
  ASSERT_TRUE(
      staged_attempt.modeled_energy_numerical_work.has_value());
  EXPECT_NEAR(
      *staged_attempt.modeled_energy_numerical_work,
      -0.1,
      1.0e-15);
  const auto& staged_breakdown =
      staged_attempt.modeled_energy_breakdown;
  EXPECT_EQ(staged_breakdown.transport.row_count, 1u);
  ASSERT_TRUE(
      staged_breakdown.transport.modeled_energy_change.has_value());
  EXPECT_NEAR(
      *staged_breakdown.transport.modeled_energy_change,
      0.2,
      1.0e-15);
  EXPECT_EQ(staged_breakdown.limiting.row_count, 1u);
  ASSERT_TRUE(
      staged_breakdown.limiting
          .accepted_modeled_energy_change.has_value());
  EXPECT_NEAR(
      *staged_breakdown.limiting
           .accepted_modeled_energy_change,
      -0.3,
      1.0e-15);
  EXPECT_EQ(
      staged_breakdown.numerical_maintenance_total.row_count,
      1u);
  ASSERT_TRUE(
      staged_breakdown.numerical_maintenance_total
          .accepted_modeled_energy_change.has_value());
  EXPECT_NEAR(
      *staged_breakdown.numerical_maintenance_total
           .accepted_modeled_energy_change,
      -0.3,
      1.0e-15);

  const auto after_global_correction =
      makeModeledMaintenanceFunctionalValue(
          /*total_potential=*/4.0,
          /*snapshot_revision=*/664u,
          /*kinetic_energy=*/1.25,
          /*gravitational_energy=*/1.75,
          /*gravitational_potential_power=*/-0.3);
  staged_ledger.beginTransaction(
      makeMaintenanceWorkTransaction(113u));
  staged_ledger.stageRow(
      application::core::LevelSetMaintenanceWorkSubstage::
          GlobalCorrection,
      2553u,
      2554u,
      {after_limiting},
      {after_global_correction});
  staged_ledger.commitTransaction();

  const auto step_account =
      application::core::
          aggregateLevelSetMaintenanceAcceptedStepEnergy(
              staged_ledger.acceptedAttempts(),
              staged_ledger.acceptedRows());
  ASSERT_TRUE(step_account.has_value());
  EXPECT_EQ(step_account->step, 8u);
  EXPECT_EQ(step_account->attempt, 2u);
  EXPECT_DOUBLE_EQ(step_account->time, 0.4);
  EXPECT_DOUBLE_EQ(step_account->dt, 0.05);
  EXPECT_EQ(step_account->transaction_count, 2u);
  EXPECT_EQ(step_account->row_count, 3u);
  ASSERT_TRUE(step_account->maintenance_start.has_value());
  ASSERT_TRUE(step_account->post_transport.has_value());
  ASSERT_TRUE(step_account->maintenance_end.has_value());
  EXPECT_EQ(
      step_account->maintenance_start
          ->algebraic_state_revision,
      2551u);
  EXPECT_EQ(
      step_account->post_transport
          ->algebraic_state_revision,
      2552u);
  EXPECT_EQ(
      step_account->maintenance_end
          ->algebraic_state_revision,
      2554u);
  ASSERT_TRUE(
      step_account->maintenance_start->modeled_stored_energy
          .has_value());
  ASSERT_TRUE(
      step_account->post_transport->modeled_stored_energy
          .has_value());
  ASSERT_TRUE(
      step_account->post_transport
          ->gravitational_potential_power.has_value());
  EXPECT_DOUBLE_EQ(
      *step_account->post_transport
           ->gravitational_potential_power,
      -0.4);
  EXPECT_FALSE(
      step_account->post_transport
          ->surface_wall_potential_power.has_value());
  ASSERT_TRUE(
      step_account->maintenance_end->modeled_stored_energy
          .has_value());
  EXPECT_NEAR(
      *step_account->post_transport->modeled_stored_energy -
          *step_account->maintenance_start
               ->modeled_stored_energy,
      0.2,
      1.0e-15);
  EXPECT_NEAR(
      *step_account->maintenance_end->modeled_stored_energy -
          *step_account->post_transport->modeled_stored_energy,
      -0.2,
      1.0e-15);
  ASSERT_TRUE(
      step_account->physical_transport_endpoint_residual
          .has_value());
  ASSERT_TRUE(
      step_account->numerical_maintenance_endpoint_residual
          .has_value());
  EXPECT_NEAR(
      *step_account->physical_transport_endpoint_residual,
      0.0,
      1.0e-15);
  EXPECT_NEAR(
      *step_account->numerical_maintenance_endpoint_residual,
      0.0,
      1.0e-15);
  const auto physical_channels =
      application::core::
          evaluateLevelSetMaintenancePhysicalEndpointChannels(
              *step_account,
              /*preceding_gravitational_energy=*/1.9,
              /*preceding_surface_wall_energy=*/2.5);
  ASSERT_TRUE(
      physical_channels.surface_wall_energy_change.has_value());
  EXPECT_NEAR(
      *physical_channels.surface_wall_energy_change,
      0.25,
      1.0e-15);
  EXPECT_FALSE(
      physical_channels.surface_transport_coupling_work
          .has_value());
  ASSERT_TRUE(
      physical_channels.gravitational_energy_change.has_value());
  ASSERT_TRUE(
      physical_channels.gravitational_transport_coupling_work
          .has_value());
  EXPECT_NEAR(
      *physical_channels.gravitational_energy_change,
      -0.1,
      1.0e-15);
  EXPECT_NEAR(
      *physical_channels.gravitational_transport_coupling_work,
      -0.08,
      1.0e-15);
  EXPECT_FALSE(
      physical_channels.bulk_viscous_dissipation_rate
          .has_value());
  EXPECT_FALSE(
      physical_channels.external_pressure_work.has_value());
  ASSERT_TRUE(
      step_account->modeled_energy_breakdown.transport
          .accepted_modeled_energy_change.has_value());
  EXPECT_NEAR(
      *step_account->modeled_energy_breakdown.transport
           .accepted_modeled_energy_change,
      0.2,
      1.0e-15);
  ASSERT_TRUE(
      step_account->modeled_energy_breakdown.global_correction
          .accepted_modeled_energy_change.has_value());
  EXPECT_NEAR(
      *step_account->modeled_energy_breakdown.global_correction
           .accepted_modeled_energy_change,
      0.1,
      1.0e-15);
  ASSERT_TRUE(
      step_account->modeled_energy_breakdown
          .numerical_maintenance_total
          .accepted_modeled_energy_change.has_value());
  EXPECT_NEAR(
      *step_account->modeled_energy_breakdown
           .numerical_maintenance_total
           .accepted_modeled_energy_change,
      -0.2,
      1.0e-15);

  const std::vector<
      application::core::LevelSetMaintenanceWorkAttempt>
      no_attempts;
  const std::vector<
      application::core::LevelSetMaintenanceWorkRow>
      no_rows;
  EXPECT_FALSE(
      application::core::
          aggregateLevelSetMaintenanceAcceptedStepEnergy(
              no_attempts, no_rows)
              .has_value());
  auto malformed_attempts = staged_ledger.acceptedAttempts();
  malformed_attempts.back().status =
      application::core::LevelSetMaintenanceWorkStatus::Rejected;
  EXPECT_THROW(
      application::core::
          aggregateLevelSetMaintenanceAcceptedStepEnergy(
              malformed_attempts, staged_ledger.acceptedRows()),
      std::invalid_argument);
  auto discontinuous_rows = staged_ledger.acceptedRows();
  discontinuous_rows.back().algebraic_state_revision_before =
      9999u;
  EXPECT_THROW(
      application::core::
          aggregateLevelSetMaintenanceAcceptedStepEnergy(
              staged_ledger.acceptedAttempts(),
              discontinuous_rows),
      std::invalid_argument);
  auto empty_functional_rows = staged_ledger.acceptedRows();
  empty_functional_rows.front().before.clear();
  empty_functional_rows.front().after.clear();
  EXPECT_THROW(
      application::core::
          aggregateLevelSetMaintenanceAcceptedStepEnergy(
              staged_ledger.acceptedAttempts(),
              empty_functional_rows),
      std::invalid_argument);
  auto nontelescoping_rows = staged_ledger.acceptedRows();
  ASSERT_TRUE(
      nontelescoping_rows.back()
          .after.front()
          .kinetic_energy.has_value());
  ASSERT_TRUE(
      nontelescoping_rows.back()
          .after.front()
          .modeled_stored_energy.has_value());
  *nontelescoping_rows.back()
       .after.front()
       .kinetic_energy += 0.25;
  *nontelescoping_rows.back()
       .after.front()
       .modeled_stored_energy += 0.25;
  EXPECT_THROW(
      application::core::
          aggregateLevelSetMaintenanceAcceptedStepEnergy(
              staged_ledger.acceptedAttempts(),
              nontelescoping_rows),
      std::invalid_argument);
}

TEST(ApplicationDriverLevelSetWorkflows,
     MaintenanceWorkLedgerCountsOneBulkOwnerAcrossMultipleInterfaces)
{
  application::core::LevelSetMaintenanceWorkLedger ledger;
  ledger.beginTransaction(makeMaintenanceWorkTransaction(110u));
  auto secondary_before = makeMaintenanceFunctionalValue(
      /*total_potential=*/1.5,
      /*snapshot_revision=*/653u,
      /*mesh_topology_revision=*/19u,
      /*cut_topology_revision=*/23u,
      /*interface_marker=*/408);
  secondary_before.surface_energy = 1.0;
  secondary_before.young_wall_energy = -0.1;
  secondary_before.volume_constraint_potential =
      secondary_before.total_potential -
      secondary_before.surface_energy -
      secondary_before.young_wall_energy;
  auto secondary_after = makeMaintenanceFunctionalValue(
      /*total_potential=*/1.75,
      /*snapshot_revision=*/654u,
      /*mesh_topology_revision=*/19u,
      /*cut_topology_revision=*/23u,
      /*interface_marker=*/408);
  secondary_after.surface_energy = 1.2;
  secondary_after.young_wall_energy = -0.05;
  secondary_after.volume_constraint_potential =
      secondary_after.total_potential -
      secondary_after.surface_energy -
      secondary_after.young_wall_energy;

  ledger.stageRow(
      application::core::LevelSetMaintenanceWorkSubstage::
          Reinitialization,
      2601u,
      2602u,
      {makeModeledMaintenanceFunctionalValue(
           /*total_potential=*/4.0,
           /*snapshot_revision=*/651u,
           /*kinetic_energy=*/1.0,
           /*gravitational_energy=*/2.0,
           /*gravitational_potential_power=*/-0.5),
       secondary_before},
      {makeModeledMaintenanceFunctionalValue(
           /*total_potential=*/4.0,
           /*snapshot_revision=*/652u,
           /*kinetic_energy=*/1.5,
           /*gravitational_energy=*/1.8,
           /*gravitational_potential_power=*/-0.4),
       secondary_after});

  ASSERT_EQ(ledger.trialRows().size(), 1u);
  ASSERT_TRUE(
      ledger.trialRows()
          .front()
          .modeled_energy_numerical_work.has_value());
  EXPECT_NEAR(
      *ledger.trialRows()
           .front()
           .modeled_energy_numerical_work,
      0.55,
      2.0e-15);
  ledger.commitTransaction();
  ASSERT_EQ(ledger.acceptedRows().size(), 1u);
  ASSERT_TRUE(
      ledger.acceptedRows()
          .front()
          .accepted_modeled_energy_numerical_work.has_value());
  EXPECT_NEAR(
      *ledger.acceptedRows()
           .front()
           .accepted_modeled_energy_numerical_work,
      0.55,
      2.0e-15);
}

TEST(ApplicationDriverLevelSetWorkflows,
     MaintenanceWorkLedgerRejectsDuplicateBulkEnergyOwners)
{
  application::core::LevelSetMaintenanceWorkLedger ledger;
  ledger.beginTransaction(makeMaintenanceWorkTransaction(111u));
  auto second_before = makeModeledMaintenanceFunctionalValue(
      /*total_potential=*/2.0,
      /*snapshot_revision=*/655u,
      /*kinetic_energy=*/0.5,
      /*gravitational_energy=*/0.75,
      /*gravitational_potential_power=*/-0.2);
  second_before.interface_marker = 408;
  auto second_after = second_before;
  second_after.snapshot_revision = 656u;

  EXPECT_THROW(
      ledger.stageRow(
          application::core::LevelSetMaintenanceWorkSubstage::
              Reinitialization,
          2701u,
          2702u,
          {makeModeledMaintenanceFunctionalValue(
               /*total_potential=*/4.0,
               /*snapshot_revision=*/651u,
               /*kinetic_energy=*/1.0,
               /*gravitational_energy=*/2.0,
               /*gravitational_potential_power=*/-0.5),
           second_before},
          {makeModeledMaintenanceFunctionalValue(
               /*total_potential=*/4.0,
               /*snapshot_revision=*/652u,
               /*kinetic_energy=*/1.5,
               /*gravitational_energy=*/1.8,
               /*gravitational_potential_power=*/-0.4),
           second_after}),
      std::invalid_argument);
  EXPECT_TRUE(ledger.trialRows().empty());
}

TEST(ApplicationDriverLevelSetWorkflows,
     MaintenanceWorkLedgerReportsSameStateAsZeroWork)
{
  application::core::LevelSetMaintenanceWorkLedger ledger;
  ledger.beginTransaction(makeMaintenanceWorkTransaction(103u));
  const auto state = makeMaintenanceFunctionalValue(4.5, 701u);
  ledger.stageRow(
      application::core::LevelSetMaintenanceWorkSubstage::
          GlobalCorrection,
      3001u,
      3001u,
      {state},
      {state});
  ledger.commitTransaction();

  ASSERT_EQ(ledger.acceptedRows().size(), 1u);
  EXPECT_DOUBLE_EQ(ledger.acceptedRows().front().numerical_work, 0.0);
  EXPECT_DOUBLE_EQ(
      ledger.acceptedRows().front().accepted_numerical_work, 0.0);
  EXPECT_EQ(
      ledger.acceptedRows().front().snapshot_set_revision_before,
      ledger.acceptedRows().front().snapshot_set_revision_after);
  EXPECT_EQ(
      ledger.acceptedRows().front().mesh_topology_set_revision_before,
      ledger.acceptedRows().front().mesh_topology_set_revision_after);
  EXPECT_EQ(
      ledger.acceptedRows().front().cut_topology_set_revision_before,
      ledger.acceptedRows().front().cut_topology_set_revision_after);
}

TEST(ApplicationDriverLevelSetWorkflows,
     MaintenanceWorkLedgerRollbackPublishesNoAcceptedRow)
{
  application::core::LevelSetMaintenanceWorkLedger ledger;
  ledger.beginTransaction(makeMaintenanceWorkTransaction(104u));
  ledger.stageRow(
      application::core::LevelSetMaintenanceWorkSubstage::
          Reinitialization,
      4001u,
      4002u,
      {makeModeledMaintenanceFunctionalValue(
          5.0, 801u, 1.0, 2.0, -0.5)},
      {makeModeledMaintenanceFunctionalValue(
          5.75, 802u, 1.2, 2.1, -0.4)});
  ledger.rejectTransaction();

  EXPECT_TRUE(ledger.trialRows().empty());
  EXPECT_TRUE(ledger.acceptedRows().empty());
  ASSERT_EQ(ledger.rejectedRows().size(), 1u);
  EXPECT_EQ(
      ledger.rejectedRows().front().status,
      application::core::LevelSetMaintenanceWorkStatus::Rejected);
  EXPECT_DOUBLE_EQ(
      ledger.rejectedRows().front().numerical_work, 0.75);
  EXPECT_DOUBLE_EQ(
      ledger.rejectedRows().front().accepted_numerical_work, 0.0);
  ASSERT_TRUE(
      ledger.rejectedRows()
          .front()
          .modeled_energy_numerical_work.has_value());
  EXPECT_NEAR(
      *ledger.rejectedRows()
           .front()
           .modeled_energy_numerical_work,
      0.3,
      1.0e-15);
  ASSERT_TRUE(
      ledger.rejectedRows()
          .front()
          .accepted_modeled_energy_numerical_work.has_value());
  EXPECT_DOUBLE_EQ(
      *ledger.rejectedRows()
           .front()
           .accepted_modeled_energy_numerical_work,
      0.0);
  ASSERT_EQ(ledger.rejectedAttempts().size(), 1u);
  EXPECT_EQ(
      ledger.rejectedAttempts().front().status,
      application::core::LevelSetMaintenanceWorkStatus::Rejected);
  EXPECT_EQ(ledger.rejectedAttempts().front().row_count, 1u);
  EXPECT_DOUBLE_EQ(
      ledger.rejectedAttempts().front().accepted_numerical_work,
      0.0);
  ASSERT_TRUE(
      ledger.rejectedAttempts()
          .front()
          .accepted_modeled_energy_numerical_work.has_value());
  EXPECT_DOUBLE_EQ(
      *ledger.rejectedAttempts()
           .front()
           .accepted_modeled_energy_numerical_work,
      0.0);
  const auto& rejected_breakdown =
      ledger.rejectedAttempts()
          .front()
          .modeled_energy_breakdown;
  EXPECT_EQ(rejected_breakdown.reinitialization.row_count, 1u);
  ASSERT_TRUE(
      rejected_breakdown.reinitialization
          .modeled_energy_change.has_value());
  ASSERT_TRUE(
      rejected_breakdown.reinitialization
          .accepted_modeled_energy_change.has_value());
  EXPECT_NEAR(
      *rejected_breakdown.reinitialization
           .modeled_energy_change,
      0.3,
      1.0e-15);
  EXPECT_DOUBLE_EQ(
      *rejected_breakdown.reinitialization
           .accepted_modeled_energy_change,
      0.0);
  ASSERT_TRUE(
      rejected_breakdown.numerical_maintenance_total
          .accepted_modeled_energy_change.has_value());
  EXPECT_DOUBLE_EQ(
      *rejected_breakdown.numerical_maintenance_total
           .accepted_modeled_energy_change,
      0.0);
}

TEST(ApplicationDriverLevelSetWorkflows,
     MaintenanceWorkLedgerRejectsDiscontinuousRows)
{
  application::core::LevelSetMaintenanceWorkLedger ledger;
  ledger.beginTransaction(makeMaintenanceWorkTransaction(105u));
  const auto initial =
      makeMaintenanceFunctionalValue(6.0, 901u, 21u, 31u);
  const auto intermediate =
      makeMaintenanceFunctionalValue(6.2, 902u, 21u, 31u);
  const auto final =
      makeMaintenanceFunctionalValue(6.1, 903u, 21u, 31u);
  ledger.stageRow(
      application::core::LevelSetMaintenanceWorkSubstage::
          Reinitialization,
      5001u,
      5002u,
      {initial},
      {intermediate},
      314u);

  EXPECT_THROW(
      ledger.stageRow(
          application::core::LevelSetMaintenanceWorkSubstage::
              GlobalCorrection,
          5999u,
          5003u,
          {intermediate},
          {final}),
      std::invalid_argument);
  auto discontinuous_functional = intermediate;
  discontinuous_functional.total_potential += 0.25;
  EXPECT_THROW(
      ledger.stageRow(
          application::core::LevelSetMaintenanceWorkSubstage::
              GlobalCorrection,
          5002u,
          5003u,
          {discontinuous_functional},
          {final}),
      std::invalid_argument);
  ASSERT_TRUE(ledger.transactionActive());
  ASSERT_EQ(ledger.trialRows().size(), 1u);

  ledger.stageRow(
      application::core::LevelSetMaintenanceWorkSubstage::
          GlobalCorrection,
      5002u,
      5003u,
      {intermediate},
      {final});
  ledger.commitTransaction();
  ASSERT_EQ(ledger.acceptedRows().size(), 2u);
  ASSERT_TRUE(
      ledger.acceptedRows()[1].extension_map_revision_before.has_value());
  ASSERT_TRUE(
      ledger.acceptedRows()[1].extension_map_revision_after.has_value());
  EXPECT_EQ(
      *ledger.acceptedRows()[1].extension_map_revision_before, 314u);
  EXPECT_EQ(
      *ledger.acceptedRows()[1].extension_map_revision_after, 314u);
}

TEST(ApplicationDriverLevelSetWorkflows,
     MaintenanceWorkLedgerPublishesZeroRowAttemptOutcomes)
{
  application::core::LevelSetMaintenanceWorkLedger ledger;
  ledger.beginTransaction(makeMaintenanceWorkTransaction(106u));
  ledger.rejectTransaction();
  EXPECT_TRUE(ledger.rejectedRows().empty());
  ASSERT_EQ(ledger.rejectedAttempts().size(), 1u);
  EXPECT_EQ(ledger.rejectedAttempts().front().row_count, 0u);
  EXPECT_DOUBLE_EQ(
      ledger.rejectedAttempts().front().numerical_work, 0.0);
  EXPECT_DOUBLE_EQ(
      ledger.rejectedAttempts().front().accepted_numerical_work,
      0.0);
  EXPECT_FALSE(
      ledger.rejectedAttempts()
          .front()
          .modeled_energy_numerical_work.has_value());
  EXPECT_FALSE(
      ledger.rejectedAttempts()
          .front()
          .accepted_modeled_energy_numerical_work.has_value());
  EXPECT_EQ(
      ledger.rejectedAttempts()
          .front()
          .modeled_energy_breakdown
          .numerical_maintenance_total.row_count,
      0u);
  EXPECT_FALSE(
      ledger.rejectedAttempts()
          .front()
          .modeled_energy_breakdown
          .numerical_maintenance_total.modeled_energy_change
          .has_value());

  ledger.beginTransaction(makeMaintenanceWorkTransaction(107u));
  ledger.commitTransaction();
  EXPECT_TRUE(ledger.acceptedRows().empty());
  ASSERT_EQ(ledger.acceptedAttempts().size(), 1u);
  EXPECT_EQ(ledger.acceptedAttempts().front().row_count, 0u);
  EXPECT_DOUBLE_EQ(
      ledger.acceptedAttempts().front().accepted_numerical_work,
      0.0);
  EXPECT_FALSE(
      ledger.acceptedAttempts()
          .front()
          .modeled_energy_numerical_work.has_value());
  EXPECT_FALSE(
      ledger.acceptedAttempts()
          .front()
          .accepted_modeled_energy_numerical_work.has_value());
  EXPECT_EQ(
      ledger.acceptedAttempts()
          .front()
          .modeled_energy_breakdown.transport.row_count,
      0u);
  const auto zero_row_step_account =
      application::core::
          aggregateLevelSetMaintenanceAcceptedStepEnergy(
              ledger.acceptedAttempts(), ledger.acceptedRows());
  ASSERT_TRUE(zero_row_step_account.has_value());
  EXPECT_EQ(zero_row_step_account->transaction_count, 1u);
  EXPECT_EQ(zero_row_step_account->row_count, 0u);
  EXPECT_FALSE(
      zero_row_step_account->maintenance_start.has_value());
  EXPECT_FALSE(
      zero_row_step_account->post_transport.has_value());
  EXPECT_FALSE(
      zero_row_step_account->maintenance_end.has_value());
  EXPECT_FALSE(
      zero_row_step_account
          ->numerical_maintenance_endpoint_residual.has_value());
  EXPECT_THROW(
      ledger.beginTransaction(makeMaintenanceWorkTransaction(107u)),
      std::invalid_argument);
}

TEST(ApplicationDriverLevelSetWorkflows,
     MaintenanceWorkLedgerRequiresExplicitCutTopologyProvenance)
{
  application::core::LevelSetMaintenanceWorkLedger ledger;
  ledger.beginTransaction(makeMaintenanceWorkTransaction(108u));
  auto missing_topology =
      makeMaintenanceFunctionalValue(7.0, 1001u);
  missing_topology.cut_topology_revision = 0u;
  EXPECT_THROW(
      ledger.stageRow(
          application::core::LevelSetMaintenanceWorkSubstage::
              Reinitialization,
          6001u,
          6002u,
          {missing_topology},
          {makeMaintenanceFunctionalValue(7.1, 1002u)}),
      std::invalid_argument);
  EXPECT_TRUE(ledger.trialRows().empty());
  ledger.rejectTransaction();
  ASSERT_EQ(ledger.rejectedAttempts().size(), 1u);
  EXPECT_EQ(ledger.rejectedAttempts().front().row_count, 0u);
}

TEST(ApplicationDriverLevelSetWorkflows,
     MaintenanceWorkLedgerRejectsInvalidFunctionalIdentityBeforeStaging)
{
  application::core::LevelSetMaintenanceWorkLedger initial_epoch_ledger;
  initial_epoch_ledger.beginTransaction(
      makeMaintenanceWorkTransaction(111u));
  auto initial_epoch_before =
      makeMaintenanceFunctionalValue(7.0, 1001u);
  initial_epoch_before.mesh_topology_revision = 0u;
  auto initial_epoch_after =
      makeMaintenanceFunctionalValue(7.1, 1002u);
  initial_epoch_after.mesh_topology_revision = 0u;
  EXPECT_NO_THROW(
      initial_epoch_ledger.stageRow(
          application::core::LevelSetMaintenanceWorkSubstage::
              Reinitialization,
          6101u,
          6102u,
          {initial_epoch_before},
          {initial_epoch_after}));
  ASSERT_EQ(initial_epoch_ledger.trialRows().size(), 1u);
  EXPECT_NE(
      initial_epoch_ledger.trialRows()
          .front()
          .mesh_topology_set_revision_before,
      0u);
  EXPECT_EQ(
      initial_epoch_ledger.trialRows()
          .front()
          .mesh_topology_set_revision_before,
      initial_epoch_ledger.trialRows()
          .front()
          .mesh_topology_set_revision_after);
  initial_epoch_ledger.rejectTransaction();

  application::core::LevelSetMaintenanceWorkLedger ledger;
  ledger.beginTransaction(makeMaintenanceWorkTransaction(112u));
  const auto valid =
      makeMaintenanceFunctionalValue(7.1, 1002u);

  EXPECT_THROW(
      ledger.stageRow(
          static_cast<
              application::core::LevelSetMaintenanceWorkSubstage>(
              255u),
          6101u,
          6102u,
          {makeMaintenanceFunctionalValue(7.0, 1001u)},
          {valid}),
      std::invalid_argument);
  EXPECT_TRUE(ledger.trialRows().empty());

  auto negative_measure =
      makeMaintenanceFunctionalValue(7.0, 1001u);
  negative_measure.liquid_volume = -0.25;
  EXPECT_THROW(
      ledger.stageRow(
          application::core::LevelSetMaintenanceWorkSubstage::
              Reinitialization,
          6101u,
          6102u,
          {negative_measure},
          {valid}),
      std::invalid_argument);
  EXPECT_TRUE(ledger.trialRows().empty());

  auto inconsistent_potential =
      makeMaintenanceFunctionalValue(7.0, 1001u);
  inconsistent_potential.total_potential += 0.125;
  EXPECT_THROW(
      ledger.stageRow(
          application::core::LevelSetMaintenanceWorkSubstage::
              Reinitialization,
          6101u,
          6102u,
          {inconsistent_potential},
          {valid}),
      std::invalid_argument);
  EXPECT_TRUE(ledger.trialRows().empty());

  ledger.rejectTransaction();
  ASSERT_EQ(ledger.rejectedAttempts().size(), 1u);
  EXPECT_EQ(ledger.rejectedAttempts().front().row_count, 0u);
}

TEST(ApplicationDriverLevelSetWorkflows,
     NonconvergedReinitializationDoesNotModifyAcceptedHistory)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowQuadPatch2x2Mesh();
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/1);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1,
  });
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_vertex_values(mesh->n_vertices(), 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    // Nodal interpolation of this off-center circle gives four connected cut
    // cells with different local gradient magnitudes.  In a fixed continuous
    // P1 space, independently normalizing those gradients would require
    // different positive cell multipliers, but continuity forces their shared
    // vertex values to agree.  Exact zero-set preservation and exact
    // redistancing are therefore not simultaneously representable here.
    const auto dx = point[0] - svmp::FE::Real{0.8};
    const auto dy = point[1] - svmp::FE::Real{0.9};
    phi_vertex_values[vertex] =
        dx * dx + dy * dy - svmp::FE::Real{0.49};
  }
  const auto coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      phi_vertex_values,
      /*components=*/1u,
      "ApplicationDriver nonconverged reinitialization phi");
  std::vector<svmp::FE::Real> initial(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, coefficients, initial);

  // Verify the low-level operation deterministically produces a geometrically
  // bounded but nonconverged candidate with the production defaults.
  svmp::FE::level_set::LevelSetReinitializationOptions defaults{};
  std::vector<svmp::FE::Real> candidate;
  const auto repair =
      svmp::FE::level_set::repairLevelSetSignedDistanceByProjection(
          *system,
          phi,
          defaults,
          initial,
          candidate);
  ASSERT_TRUE(repair.success) << repair.diagnostic;
  EXPECT_FALSE(repair.converged);
  EXPECT_TRUE(repair.zero_set_bound_satisfied);
  EXPECT_LE(repair.max_interface_displacement,
            defaults.max_zero_set_displacement + 1.0e-12);
  EXPECT_GT(repair.max_signed_distance_error,
            defaults.signed_distance_tolerance);

  std::unique_ptr<svmp::FE::backends::BackendFactory> factory;
  try {
    factory = svmp::FE::backends::BackendFactory::create(
        svmp::FE::backends::BackendKind::Eigen);
  } catch (const std::exception&) {
    GTEST_SKIP() << "Requires the Eigen FE backend.";
  }
  ASSERT_NE(factory, nullptr);
  auto history = svmp::FE::timestepping::TimeHistory::allocate(
      *factory,
      system->dofHandler().getNumDofs());
  history.setStepIndex(1);
  scatterFeOrderedSolution(history.u(), initial);
  scatterFeOrderedSolution(history.uPrev(), initial);
  scatterFeOrderedSolution(history.uPrev2(), initial);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);

  LevelSetMaintenanceRequest request{};
  request.level_set_field_name = "phi";
  request.reinitialization.enabled = true;
  request.reinitialization.cadence_steps = 1;
  std::vector<LevelSetMaintenanceRequest> requests{request};

  testing::internal::CaptureStdout();
  const bool changed = applyLevelSetMaintenance(sim, history, requests);
  const auto output = testing::internal::GetCapturedStdout();

  EXPECT_FALSE(changed);
  EXPECT_NE(output.find("reason=nonconverged"), std::string::npos);
  EXPECT_EQ(gatherFeOrderedSolution(history.u()), initial);
  EXPECT_EQ(gatherFeOrderedSolution(history.uPrev()), initial);
  EXPECT_EQ(gatherFeOrderedSolution(history.uPrev2()), initial);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     ConvergedMaintenanceAppliesOneRepresentationDeltaToEveryHistoryLevel)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowQuadPatch2x2Mesh();
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/1);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1,
  });
  ASSERT_NO_THROW(system->setup({}));

  const auto make_plane = [&](svmp::FE::Real offset,
                              svmp::FE::Real gradient_scale) {
    std::vector<svmp::FE::Real> values(mesh->n_vertices(), 0.0);
    for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
      values[vertex] = gradient_scale *
          (workflowVertexPoint(*mesh, vertex)[0] - offset);
    }
    const auto coefficients = projectWorkflowVertexValues(
        *system,
        phi,
        values,
        /*components=*/1u,
        "ApplicationDriver maintenance-history plane");
    std::vector<svmp::FE::Real> solution(
        static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
    writeWorkflowFieldSlice(*system, phi, coefficients, solution);
    return solution;
  };
  const auto current_before = make_plane(0.80, 2.0);
  const auto previous_before = make_plane(0.72, 2.0);
  const auto previous2_before = make_plane(0.63, 2.0);

  std::unique_ptr<svmp::FE::backends::BackendFactory> factory;
  try {
    factory = svmp::FE::backends::BackendFactory::create(
        svmp::FE::backends::BackendKind::FSILS);
  } catch (const std::exception&) {
    GTEST_SKIP() << "Requires an available FE vector backend.";
  }
  auto history = svmp::FE::timestepping::TimeHistory::allocate(
      *factory,
      system->dofHandler().getNumDofs(),
      /*history_depth=*/2,
      /*allocate_second_order_state=*/true);
  history.setStepIndex(1);
  history.setDt(0.1);
  history.setPrevDt(0.1);
  scatterFeOrderedSolution(history.u(), current_before);
  scatterFeOrderedSolution(history.uPrev(), previous_before);
  scatterFeOrderedSolution(history.uPrev2(), previous2_before);
  std::vector<svmp::FE::Real> rate_before(current_before.size(), 0.375);
  scatterFeOrderedSolution(history.uDot(), rate_before);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);

  LevelSetMaintenanceRequest request{};
  request.level_set_field_name = "phi";
  request.reinitialization.enabled = true;
  request.reinitialization.cadence_steps = 1;
  // With the production relaxation factor (0.3), twenty iterations leave an
  // O(10^-3) residual for this factor-of-two planar distortion.  Give the
  // projection enough iterations to meet the deliberately strict tolerance
  // so this test exercises the converged maintenance/history path.
  request.reinitialization.max_iterations = 100;
  request.reinitialization.signed_distance_tolerance = 1.0e-10;
  std::vector<LevelSetMaintenanceRequest> requests{request};
  std::vector<application::core::LevelSetMaintenanceWorkSubstage>
      observed_substages;
  std::vector<std::pair<std::uint64_t, std::uint64_t>>
      observed_state_revisions;
  const LevelSetMaintenanceStageObserver observe_stage =
      [&](application::core::LevelSetMaintenanceWorkSubstage substage,
          std::span<const svmp::FE::Real> before,
          std::span<const svmp::FE::Real> after) {
        observed_substages.push_back(substage);
        observed_state_revisions.emplace_back(
            levelSetMaintenanceAlgebraicRevision(before),
            levelSetMaintenanceAlgebraicRevision(after));
      };

  testing::internal::CaptureStdout();
  const bool changed = applyLevelSetMaintenance(
      sim,
      history,
      requests,
      {},
      {},
      {},
      nullptr,
      {},
      observe_stage);
  const auto output = testing::internal::GetCapturedStdout();
  ASSERT_TRUE(changed);
  EXPECT_NE(output.find("temporal_increments=preserved"), std::string::npos);
  ASSERT_EQ(observed_substages.size(), 1u);
  EXPECT_EQ(
      observed_substages.front(),
      application::core::LevelSetMaintenanceWorkSubstage::
          Reinitialization);
  ASSERT_EQ(observed_state_revisions.size(), 1u);
  EXPECT_NE(
      observed_state_revisions.front().first,
      observed_state_revisions.front().second);

  const auto current_after = gatherFeOrderedSolution(history.u());
  const auto previous_after = gatherFeOrderedSolution(history.uPrev());
  const auto previous2_after = gatherFeOrderedSolution(history.uPrev2());
  ASSERT_EQ(current_after.size(), current_before.size());
  bool saw_nonzero_delta = false;
  for (std::size_t i = 0; i < current_after.size(); ++i) {
    const auto current_delta = current_after[i] - current_before[i];
    saw_nonzero_delta = saw_nonzero_delta ||
                        std::abs(current_delta) > 1.0e-12;
    EXPECT_NEAR(previous_after[i] - previous_before[i],
                current_delta,
                1.0e-12)
        << "global DOF " << i;
    EXPECT_NEAR(previous2_after[i] - previous2_before[i],
                current_delta,
                1.0e-12)
        << "global DOF " << i;
    EXPECT_NEAR((current_after[i] - previous_after[i]),
                (current_before[i] - previous_before[i]),
                1.0e-12)
        << "current/previous increment at global DOF " << i;
    EXPECT_NEAR((previous_after[i] - previous2_after[i]),
                (previous_before[i] - previous2_before[i]),
                1.0e-12)
        << "previous/older increment at global DOF " << i;
  }
  EXPECT_TRUE(saw_nonzero_delta);
  EXPECT_EQ(gatherFeOrderedSolution(history.uDot()), rate_before);
#endif
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
namespace {

class ApplicationDriverConservativePhaseCandidatesTest
    : public ::testing::Test {
protected:
  void SetUp() override
  {
    mesh_ = makeWorkflowTriangleMesh();
    (void)svmp::MeshFields::attach_field(
        mesh_->local_mesh(),
        svmp::EntityKind::Vertex,
        "phi",
        svmp::FieldScalarType::Float64,
        1);
    auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
        svmp::FE::ElementType::Triangle3,
        /*order=*/1);
    auto system =
        std::make_unique<svmp::FE::systems::FESystem>(mesh_);
    phi_ = system->addField(svmp::FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    phase_ = system->addField(svmp::FE::systems::FieldSpec{
        .name = "phase",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system->setup({}));

    sim_.primary_mesh = mesh_;
    sim_.fe_system = std::move(system);
    sim_.backend = svmp::FE::backends::BackendFactory::create(
        svmp::FE::backends::BackendKind::FSILS);
    ASSERT_NE(sim_.backend, nullptr);
    auto allocated_history =
        svmp::FE::timestepping::TimeHistory::allocate(
            *sim_.backend,
            sim_.fe_system->dofHandler().getNumDofs(),
            /*history_depth=*/2,
            /*allocate_second_order_state=*/true);
    sim_.time_history =
        std::make_unique<svmp::FE::timestepping::TimeHistory>(
            std::move(allocated_history));
    history().setDt(0.05);
    history().setPrevDt(0.05);

    std::vector<svmp::FE::Real> phi_vertex_values(
        mesh_->n_vertices(), svmp::FE::Real{0.0});
    for (std::size_t vertex = 0u; vertex < mesh_->n_vertices(); ++vertex) {
      phi_vertex_values[vertex] =
          workflowVertexPoint(*mesh_, vertex)[0] - svmp::FE::Real{0.75};
    }
    const auto phi_coefficients = projectWorkflowVertexValues(
        *sim_.fe_system,
        phi_,
        phi_vertex_values,
        /*components=*/1u,
        "ApplicationDriver conservative phase phi");
    std::vector<svmp::FE::Real> initial(solutionSize(),
                                         svmp::FE::Real{0.0});
    writeWorkflowFieldSlice(
        *sim_.fe_system, phi_, phi_coefficients, initial);
    scatterFeOrderedSolution(history().u(), initial);
    scatterFeOrderedSolution(history().uPrev(), initial);
    scatterFeOrderedSolution(history().uPrev2(), initial);
    std::vector<svmp::FE::Real> initial_rates(
        solutionSize(), svmp::FE::Real{0.5});
    scatterFeOrderedSolution(history().uDot(), initial_rates);
    scatterFeOrderedSolution(history().uDDot(), initial_rates);

    params_ = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Level_set_field_name>phi</Level_set_field_name>
    <Velocity_source>constant</Velocity_source>
    <Constant_velocity>0.0 0.0 0.0</Constant_velocity>
    <Enable_conservative_phase_transport>true</Enable_conservative_phase_transport>
    <Conservative_phase_field_name>phase</Conservative_phase_field_name>
  </Add_equation>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>conservative_phase_interface</Generated_interface_domain_id>
      <Interface_marker>911</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
    active_requests_ = activeCutVolumeRequests(*params_);
    ASSERT_EQ(active_requests_.size(), 1u);
    requests_ = levelSetMaintenanceRequests(*params_);
    ASSERT_EQ(requests_.size(), 1u);
    ASSERT_TRUE(requests_.front().conservative_phase.enabled);
    ASSERT_TRUE(requests_.front().volume_cut_request.has_value());

    const auto initial_refresh = refreshActiveCutIntegrationContextCached(
        sim_,
        *params_,
        history().u(),
        lifecycle_,
        refresh_cache_,
        "application-driver-conservative-phase-initial");
    ASSERT_TRUE(initial_refresh.refreshed);
    ASSERT_NO_THROW(initializeConservativePhaseStates(sim_, requests_));
    ASSERT_TRUE(requests_.front().conservative_phase_initialized);
    initialized_solution_ = gatherFeOrderedSolution(history().u());
  }

  [[nodiscard]] svmp::FE::timestepping::TimeHistory& history()
  {
    return *sim_.time_history;
  }

  struct PreparedConservativePhaseCandidateStage {
    std::vector<ConservativePhaseFixedGraphBinding> fixed_graph_bindings{};
    ConservativePhaseCandidateStageSnapshot snapshot{};
    std::uint64_t expected_attempt{1u};
  };

  [[nodiscard]] PreparedConservativePhaseCandidateStage
  prepareConservativePhaseCandidateStage()
  {
    const auto comm = activeFESystemCommunicator(*sim_.fe_system);
    PreparedConservativePhaseCandidateStage prepared;
    prepared.fixed_graph_bindings =
        captureConservativePhaseFixedGraphBindings(
            *sim_.fe_system, requests_, comm);
    const auto& mesh = sim_.fe_system->meshAccess();
    svmp::FE::timestepping::CandidateStageObservation observation;
    observation.scheme =
        svmp::FE::timestepping::SchemeKind::BackwardEuler;
    observation.temporal_order = 1;
    observation.step_index = history().stepIndex() + 1;
    observation.attempt_index = 0;
    observation.step_start_time = history().time();
    observation.step_end_time = history().time() + history().dt();
    observation.state_time = observation.step_end_time;
    observation.rate_time = observation.step_end_time;
    observation.dt = history().dt();
    observation.mesh_revision =
        svmp::FE::timestepping::CandidateStageMeshRevision{
            .geometry_revision = mesh.geometryRevision(),
            .topology_revision = mesh.topologyRevision(),
            .ownership_revision = mesh.ownershipRevision(),
            .numbering_revision = mesh.numberingRevision(),
            .field_layout_revision = mesh.fieldLayoutRevision(),
            .label_revision = mesh.labelRevision(),
            .active_configuration_epoch =
                mesh.activeConfigurationEpoch(),
            .coordinate_configuration_key =
                mesh.coordinateConfigurationKey(),
        };
    observation.state_vector = &history().u();
    observation.rate_vector = &history().uDot();
    prepared.snapshot = buildConservativePhaseCandidateStageSnapshot(
        sim_,
        history(),
        requests_,
        prepared.fixed_graph_bindings,
        observation,
        comm);
    return prepared;
  }

  [[nodiscard]] ConservativePhaseCandidateResult
  applyPreparedConservativePhaseCandidate(
      PreparedConservativePhaseCandidateStage& prepared,
      const ConservativePhaseContactStageBuilder& contact_stage_builder = {},
      const LevelSetMaintenanceStageObserver& observe_stage = {})
  {
    return applyConservativePhaseCandidates(
        sim_,
        history(),
        requests_,
        *params_,
        lifecycle_,
        refresh_cache_,
        active_requests_,
        prepared.snapshot,
        prepared.fixed_graph_bindings,
        prepared.expected_attempt,
        contact_stage_builder,
        observe_stage);
  }

  [[nodiscard]] ConservativePhaseCandidateResult
  applyPreparedConservativePhaseCandidate(
      const ConservativePhaseContactStageBuilder& contact_stage_builder = {},
      const LevelSetMaintenanceStageObserver& observe_stage = {})
  {
    auto prepared = prepareConservativePhaseCandidateStage();
    return applyPreparedConservativePhaseCandidate(
        prepared, contact_stage_builder, observe_stage);
  }

  [[nodiscard]] std::size_t solutionSize() const
  {
    return static_cast<std::size_t>(
        sim_.fe_system->dofHandler().getNumDofs());
  }

  [[nodiscard]] std::size_t fieldOffset(svmp::FE::FieldId field) const
  {
    const auto offset = sim_.fe_system->fieldDofOffset(field);
    if (offset < 0) {
      throw std::runtime_error(
          "ApplicationDriver conservative phase test has no field offset");
    }
    return static_cast<std::size_t>(offset);
  }

  [[nodiscard]] std::size_t fieldCount(svmp::FE::FieldId field) const
  {
    return static_cast<std::size_t>(
        sim_.fe_system->fieldDofHandler(field).getNumDofs());
  }

  [[nodiscard]] std::vector<svmp::FE::Real> fieldSlice(
      std::span<const svmp::FE::Real> solution,
      svmp::FE::FieldId field) const
  {
    const auto offset = fieldOffset(field);
    const auto count = fieldCount(field);
    if (offset + count > solution.size()) {
      throw std::runtime_error(
          "ApplicationDriver conservative phase test slice is out of range");
    }
    return std::vector<svmp::FE::Real>(
        solution.begin() + static_cast<std::ptrdiff_t>(offset),
        solution.begin() + static_cast<std::ptrdiff_t>(offset + count));
  }

  void refreshCurrentCandidate(const char* provenance)
  {
    (void)refreshActiveCutIntegrationContextCached(
        sim_,
        *params_,
        history().u(),
        lifecycle_,
        refresh_cache_,
        provenance);
  }

  std::shared_ptr<svmp::Mesh> mesh_{};
  svmp::FE::FieldId phi_{svmp::FE::INVALID_FIELD_ID};
  svmp::FE::FieldId phase_{svmp::FE::INVALID_FIELD_ID};
  application::core::SimulationComponents sim_{};
  std::unique_ptr<Parameters> params_{};
  std::vector<ActiveCutVolumeRequest> active_requests_{};
  std::vector<LevelSetMaintenanceRequest> requests_{};
  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle_{};
  ActiveCutContextRefreshCache refresh_cache_{};
  std::vector<svmp::FE::Real> initialized_solution_{};
};

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       ActiveCutRefreshObserverSeesVectorAndSpanCachePaths)
{
  TransientCutTopologyAttemptTracker tracker;
  std::size_t observations = 0u;
  std::size_t rebuilds = 0u;
  std::size_t cache_hits = 0u;
  refresh_cache_.observer =
      [&](const ActiveCutContextRefreshReport& report,
          std::string_view provenance) {
        ++observations;
        rebuilds += report.refreshed ? 1u : 0u;
        cache_hits += report.refreshed ? 0u : 1u;
        tracker.observe(report, provenance);
      };

  tracker.beginAttempt();
  const auto before_solve = refreshActiveCutIntegrationContextCached(
      sim_,
      *params_,
      history().u(),
      lifecycle_,
      refresh_cache_,
      "before_physics_solve");
  EXPECT_FALSE(before_solve.refreshed);
  ASSERT_NO_THROW(tracker.requireAcceptedBaseline());
  const auto accepted_topology_key = before_solve.topology_key;
  ASSERT_NE(accepted_topology_key, 0u);

  auto vector_candidate = gatherFeOrderedSolution(history().u());
  const auto offset = fieldOffset(phi_);
  const auto count = fieldCount(phi_);
  for (std::size_t i = 0u; i < count; ++i) {
    vector_candidate[offset + i] += svmp::FE::Real{1.0e-7};
  }
  scatterFeOrderedSolution(history().u(), vector_candidate);
  const auto vector_rebuild = refreshActiveCutIntegrationContextCached(
      sim_,
      *params_,
      history().u(),
      lifecycle_,
      refresh_cache_,
      "accepted_newton_iterate");
  EXPECT_TRUE(vector_rebuild.refreshed);
  EXPECT_EQ(vector_rebuild.topology_key, accepted_topology_key);

  auto span_candidate = gatherFeOrderedSolution(history().u());
  for (std::size_t i = 0u; i < count; ++i) {
    span_candidate[offset + i] += svmp::FE::Real{1.0e-7};
  }
  const auto span_rebuild =
      refreshActiveCutIntegrationContextFromSolutionCached(
          sim_,
          *params_,
          span_candidate,
          lifecycle_,
          refresh_cache_,
          "accepted_step_maintenance_candidate",
          "staged_fe_solution");
  EXPECT_TRUE(span_rebuild.refreshed);
  EXPECT_EQ(span_rebuild.topology_key, accepted_topology_key);

  const auto final_cache_hit =
      refreshActiveCutIntegrationContextFromSolutionCached(
          sim_,
          *params_,
          span_candidate,
          lifecycle_,
          refresh_cache_,
          "final_candidate_topology_gate",
          "staged_fe_solution");
  EXPECT_FALSE(final_cache_hit.refreshed);
  EXPECT_EQ(final_cache_hit.topology_key, accepted_topology_key);
  EXPECT_EQ(observations, 4u);
  EXPECT_EQ(rebuilds, 2u);
  EXPECT_EQ(cache_hits, 2u);
  EXPECT_FALSE(tracker.attemptTainted());
  EXPECT_FALSE(
      tracker.candidateMustReject(refresh_cache_.topology_key));
  EXPECT_NO_THROW(
      tracker.completeAttempt(refresh_cache_.topology_key));
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       InitializesEveryHistoryLevelAndOnlyItsRateSlices)
{
  const auto current = gatherFeOrderedSolution(history().u());
  const auto previous = gatherFeOrderedSolution(history().uPrev());
  const auto older = gatherFeOrderedSolution(history().uPrev2());
  const auto current_phase = fieldSlice(current, phase_);
  EXPECT_EQ(current_phase, fieldSlice(previous, phase_));
  EXPECT_EQ(current_phase, fieldSlice(older, phase_));
  for (const auto value : current_phase) {
    EXPECT_GE(value, svmp::FE::Real{0.0});
    EXPECT_LE(value, svmp::FE::Real{1.0});
  }

  const auto rate = gatherFeOrderedSolution(history().uDot());
  const auto acceleration = gatherFeOrderedSolution(history().uDDot());
  for (const auto value : fieldSlice(rate, phase_)) {
    EXPECT_DOUBLE_EQ(value, svmp::FE::Real{0.0});
  }
  for (const auto value : fieldSlice(acceleration, phase_)) {
    EXPECT_DOUBLE_EQ(value, svmp::FE::Real{0.0});
  }
  for (const auto value : fieldSlice(rate, phi_)) {
    EXPECT_DOUBLE_EQ(value, svmp::FE::Real{0.5});
  }
  for (const auto value : fieldSlice(acceleration, phi_)) {
    EXPECT_DOUBLE_EQ(value, svmp::FE::Real{0.5});
  }
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       ExplicitPointwiseWallVelocityContractIsMarkedUnsupported)
{
  EXPECT_FALSE(
      hasExplicitUnsupportedConservativePhasePointwiseWallContract(
          requests_));
  requests_.front()
      .pointwise_impermeable_velocity_tolerance_explicitly_requested =
      true;
  EXPECT_TRUE(
      hasExplicitUnsupportedConservativePhasePointwiseWallContract(
          requests_));
  requests_.front().conservative_phase.enabled = false;
  EXPECT_FALSE(
      hasExplicitUnsupportedConservativePhasePointwiseWallContract(
          requests_));
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       StagesAndCommitsTheTransportedPhaseAgainstAuthoritativeGeometry)
{
  auto raw_candidate = initialized_solution_;
  const auto phase_offset = fieldOffset(phase_);
  for (std::size_t i = 0u; i < fieldCount(phase_); ++i) {
    raw_candidate[phase_offset + i] =
        svmp::FE::Real{0.9} -
        svmp::FE::Real{0.05} * static_cast<svmp::FE::Real>(i);
  }
  scatterFeOrderedSolution(history().u(), raw_candidate);
  refreshCurrentCandidate(
      "application-driver-conservative-phase-commit-raw");
  const auto previous_phase = fieldSlice(
      gatherFeOrderedSolution(history().uPrev()), phase_);
  std::vector<application::core::LevelSetMaintenanceWorkSubstage>
      observed_substages;

  auto result = applyPreparedConservativePhaseCandidate(
      {},
      [&](application::core::LevelSetMaintenanceWorkSubstage substage,
          std::span<const svmp::FE::Real>,
          std::span<const svmp::FE::Real>) {
        observed_substages.push_back(substage);
      });
  EXPECT_TRUE(result.accept_step);
  EXPECT_TRUE(result.changed);
  ASSERT_NE(result.geometry_transaction, nullptr);
  ASSERT_GE(observed_substages.size(), 2u);
  EXPECT_EQ(
      observed_substages[0],
      application::core::LevelSetMaintenanceWorkSubstage::Transport);
  EXPECT_EQ(
      observed_substages[1],
      application::core::LevelSetMaintenanceWorkSubstage::Limiting);
  EXPECT_NE(
      std::find(
          observed_substages.begin(),
          observed_substages.end(),
          application::core::LevelSetMaintenanceWorkSubstage::
              GeometryReconciliation),
      observed_substages.end());
  EXPECT_EQ(fieldSlice(gatherFeOrderedSolution(history().u()), phase_),
            previous_phase);
  EXPECT_EQ(fieldSlice(gatherFeOrderedSolution(history().uPrev()), phase_),
            previous_phase);
  ASSERT_NO_THROW(result.geometry_transaction->commit());
  EXPECT_TRUE(result.geometry_transaction->publicationStarted());
  EXPECT_FALSE(result.geometry_transaction->active());
  EXPECT_THROW(
      result.geometry_transaction->rollback(), std::logic_error);
  result.geometry_transaction.reset();
  EXPECT_FALSE(sim_.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle_.transactionActive());

  const auto projection = projectCurrentConservativePhaseGeometry(
      *sim_.fe_system, requests_.front());
  ASSERT_TRUE(projection.success) << projection.diagnostic;
  auto& graph = requireCurrentConservativePhaseGraph(
      *sim_.fe_system, requests_.front());
  svmp::FE::Real accepted_measure = svmp::FE::Real{0.0};
  const auto accepted_phase = fieldSlice(
      gatherFeOrderedSolution(history().u()), phase_);
  for (std::size_t i = 0u; i < graph.nodes; ++i) {
    accepted_measure += graph.lumped_control_volume[i] * accepted_phase[i];
  }
  EXPECT_NEAR(projection.retained_liquid_measure,
              accepted_measure,
              1.0e-10);
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       RollbackRemainsActiveAndRetriesAfterCutContextCallbackFailure)
{
  const auto* context_before =
      sim_.fe_system->cutIntegrationContext();
  ASSERT_NE(context_before, nullptr);
  const auto lifecycle_revision_before = lifecycle_.valueRevision();
  const auto constraint_revision_before =
      sim_.fe_system->constraintLayoutRevision();
  const auto sparsity_revision_before =
      sim_.fe_system->sparsityPatternRevision();
  const auto constraint_count_before =
      sim_.fe_system->constraints().numConstraints();
  const auto refresh_cache_before = refresh_cache_;
  const auto mesh_revisions_before =
      mesh_->event_bus().revision_state();
  const auto phi_handle = mesh_->field_handle(
      svmp::EntityKind::Vertex, "phi");
  const auto phi_value_count =
      mesh_->field_components(phi_handle) *
      mesh_->field_entity_count(phi_handle);
  const auto* phi_before_data =
      static_cast<const double*>(mesh_->field_data(phi_handle));
  ASSERT_NE(phi_before_data, nullptr);
  const std::vector<double> phi_before(
      phi_before_data, phi_before_data + phi_value_count);

  auto candidate = gatherFeOrderedSolution(history().u());
  candidate[fieldOffset(phi_)] += svmp::FE::Real{0.2};
  LevelSetMaintenanceGeometryTransaction transaction(
      sim_, lifecycle_, refresh_cache_, active_requests_);
  ASSERT_NO_THROW(
      (void)transaction.refresh(*params_, candidate));
  ASSERT_TRUE(transaction.active());
  ASSERT_TRUE(
      sim_.fe_system->cutIntegrationContextTransactionActive());
  ASSERT_TRUE(lifecycle_.transactionActive());

  auto throw_once = std::make_shared<bool>(true);
  auto callback_calls = std::make_shared<int>(0);
  sim_.fe_system->addCutIntegrationContextUpdateCallback(
      svmp::FE::systems::CutIntegrationContextUpdateCallback{
          .name = "application-driver-rollback-retry-test",
          .callback =
              [throw_once, callback_calls](
                  const svmp::FE::assembly::CutIntegrationContext*) {
                ++*callback_calls;
                if (*throw_once) {
                  *throw_once = false;
                  throw std::runtime_error(
                      "injected one-shot cut-context rollback failure");
                }
              },
      });

  EXPECT_THROW(transaction.rollback(), std::runtime_error);
  EXPECT_TRUE(transaction.active());
  EXPECT_TRUE(
      sim_.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle_.transactionActive());

  ASSERT_NO_THROW(transaction.rollback());
  EXPECT_FALSE(transaction.active());
  EXPECT_FALSE(
      sim_.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle_.transactionActive());
  EXPECT_GE(*callback_calls, 2);
  EXPECT_EQ(sim_.fe_system->cutIntegrationContext(), context_before);
  EXPECT_EQ(lifecycle_.valueRevision(), lifecycle_revision_before);
  EXPECT_EQ(
      sim_.fe_system->constraintLayoutRevision(),
      constraint_revision_before);
  EXPECT_EQ(
      sim_.fe_system->sparsityPatternRevision(),
      sparsity_revision_before);
  EXPECT_EQ(
      sim_.fe_system->constraints().numConstraints(),
      constraint_count_before);
  EXPECT_EQ(
      mesh_->event_bus().revision_state().reference_geometry,
      mesh_revisions_before.reference_geometry);
  EXPECT_EQ(
      mesh_->event_bus().revision_state().current_geometry,
      mesh_revisions_before.current_geometry);
  EXPECT_EQ(
      mesh_->event_bus().revision_state().reference_rebase,
      mesh_revisions_before.reference_rebase);
  EXPECT_EQ(
      mesh_->event_bus().revision_state().geometry,
      mesh_revisions_before.geometry);
  EXPECT_EQ(
      mesh_->event_bus().revision_state().topology,
      mesh_revisions_before.topology);
  EXPECT_EQ(
      mesh_->event_bus().revision_state().ownership,
      mesh_revisions_before.ownership);
  EXPECT_EQ(
      mesh_->event_bus().revision_state().numbering,
      mesh_revisions_before.numbering);
  EXPECT_EQ(
      mesh_->event_bus().revision_state().field_layout,
      mesh_revisions_before.field_layout);
  EXPECT_EQ(
      mesh_->event_bus().revision_state().labels,
      mesh_revisions_before.labels);
  EXPECT_EQ(
      mesh_->event_bus().revision_state().active_configuration,
      mesh_revisions_before.active_configuration);
  const auto* phi_after_data =
      static_cast<const double*>(mesh_->field_data(phi_handle));
  ASSERT_NE(phi_after_data, nullptr);
  EXPECT_EQ(
      std::vector<double>(
          phi_after_data, phi_after_data + phi_value_count),
      phi_before);
  EXPECT_EQ(
      refresh_cache_.last_signature.has_value(),
      refresh_cache_before.last_signature.has_value());
  if (refresh_cache_.last_signature.has_value()) {
    EXPECT_EQ(
        *refresh_cache_.last_signature,
        *refresh_cache_before.last_signature);
  }
  EXPECT_EQ(
      refresh_cache_.last_vector_signature.has_value(),
      refresh_cache_before.last_vector_signature.has_value());
  if (refresh_cache_.last_vector_signature.has_value()) {
    EXPECT_EQ(
        *refresh_cache_.last_vector_signature,
        *refresh_cache_before.last_vector_signature);
  }
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       TransportOnlyKeepsAnAlreadyConsistentGeometryUnchanged)
{
  requests_.front().reinitialization.enabled = false;
  requests_.front().conservative_phase.reconcile_geometry = false;

  auto result = applyPreparedConservativePhaseCandidate();
  ASSERT_TRUE(result.accept_step);
  EXPECT_FALSE(result.changed);
  ASSERT_EQ(result.maintenance_ledgers.size(), 1u);
  const auto& ledger = result.maintenance_ledgers.front();
  EXPECT_TRUE(ledger.transport_stage.success);
  EXPECT_FALSE(ledger.reinitialization_due);
  EXPECT_FALSE(ledger.reinitialization_applied);
  EXPECT_EQ(ledger.reconciliation.iterations, 0);
  EXPECT_LE(ledger.post_correction_mismatch.maximum_nodal_residual,
            svmp::FE::Real{1.0e-10});
  ASSERT_NE(result.geometry_transaction, nullptr);
  ASSERT_NO_THROW(result.geometry_transaction->commit());
  result.geometry_transaction.reset();
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       ImmutableEndpointVelocityIsConsumedOnceAndRetainedInTheStageLedger)
{
  requests_.front().reinitialization.enabled = false;
  requests_.front().conservative_phase.reconcile_geometry = false;
  auto prepared = prepareConservativePhaseCandidateStage();
  ASSERT_EQ(prepared.snapshot.requests.size(), requests_.size());
  ASSERT_FALSE(
      prepared.snapshot.requests.front().sampled_nodal_velocity.empty());
  const auto sampled_velocity_revision =
      svmp::FE::level_set::levelSetP1PhaseVelocityContentRevision(
          prepared.snapshot.requests.front().sampled_nodal_velocity);

  auto result = applyPreparedConservativePhaseCandidate(prepared);

  EXPECT_TRUE(
      prepared.snapshot.requests.front().sampled_nodal_velocity.empty());
  ASSERT_EQ(result.maintenance_ledgers.size(), requests_.size());
  const auto& ledger = result.maintenance_ledgers.front();
  ASSERT_TRUE(ledger.split_stage_provenance.has_value());
  EXPECT_NE(ledger.split_stage_provenance->operator_state_revision, 0u);
  EXPECT_EQ(
      ledger.split_stage_provenance->nodal_velocity_revision,
      sampled_velocity_revision);
  EXPECT_EQ(
      svmp::FE::level_set::levelSetP1PhaseVelocityContentRevision(
          ledger.transport_stage.sampled_nodal_velocity),
      sampled_velocity_revision);
  EXPECT_EQ(ledger.split_stage_provenance->attempt, 1u);
  ASSERT_NE(result.geometry_transaction, nullptr);
  ASSERT_NO_THROW(result.geometry_transaction->commit());
  result.geometry_transaction.reset();
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       StaleRetryAttemptSnapshotFailsBeforeCandidateMutation)
{
  auto prepared = prepareConservativePhaseCandidateStage();
  ++prepared.snapshot.attempt;
  const auto before = gatherFeOrderedSolution(history().u());

  EXPECT_THROW(
      (void)applyPreparedConservativePhaseCandidate(prepared),
      std::runtime_error);

  EXPECT_EQ(gatherFeOrderedSolution(history().u()), before);
  EXPECT_FALSE(sim_.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle_.transactionActive());
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       FixedBackgroundRevisionDriftFailsBeforeCandidateMutation)
{
  auto prepared = prepareConservativePhaseCandidateStage();
  const auto solution_before = gatherFeOrderedSolution(history().u());
  mesh_->event_bus().notify(svmp::MeshEvent::GeometryChanged);
  const auto* cut_context_before_apply =
      sim_.fe_system->cutIntegrationContext();
  const auto lifecycle_revision_before_apply = lifecycle_.valueRevision();

  EXPECT_THROW(
      (void)applyPreparedConservativePhaseCandidate(prepared),
      std::runtime_error);

  EXPECT_EQ(gatherFeOrderedSolution(history().u()), solution_before);
  EXPECT_EQ(
      sim_.fe_system->cutIntegrationContext(), cut_context_before_apply);
  EXPECT_EQ(lifecycle_.valueRevision(), lifecycle_revision_before_apply);
  EXPECT_FALSE(sim_.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle_.transactionActive());
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       RequestDriftDoesNotResampleTheImmutableEndpointVelocity)
{
  auto prepared = prepareConservativePhaseCandidateStage();
  requests_.front().velocity.constant_value[0] = svmp::FE::Real{0.25};
  const auto before = gatherFeOrderedSolution(history().u());

  EXPECT_THROW(
      (void)applyPreparedConservativePhaseCandidate(prepared),
      std::runtime_error);

  EXPECT_EQ(gatherFeOrderedSolution(history().u()), before);
  EXPECT_FALSE(sim_.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle_.transactionActive());
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       ClosedDomainRejectsDiscretePhaseBoundaryTransfer)
{
  requests_.front().reinitialization.enabled = false;
  requests_.front().conservative_phase.reconcile_geometry = false;
  requests_.front().conservative_phase.enforce_courant_limit = false;
  requests_.front().velocity.constant_value = {
      svmp::FE::Real{0.25}, svmp::FE::Real{0.0}, svmp::FE::Real{0.0}};
  const auto before = gatherFeOrderedSolution(history().u());

  try {
    auto result = applyPreparedConservativePhaseCandidate();
    if (result.geometry_transaction) {
      result.geometry_transaction->rollback();
    }
    FAIL() << "Expected nonzero closed-domain discrete q boundary transfer to be rejected";
  } catch (const std::runtime_error& error) {
    const std::string diagnostic = error.what();
    EXPECT_NE(
        diagnostic.find("discrete q boundary flux above its invariant tolerance"),
        std::string::npos);
    EXPECT_NE(
        diagnostic.find("not a pointwise velocity-normal test"),
        std::string::npos);
  }

  EXPECT_EQ(gatherFeOrderedSolution(history().u()), before);
  EXPECT_FALSE(sim_.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle_.transactionActive());
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       ReconciliationOnlyRepairsTransportedGeometryLocally)
{
  auto raw_candidate = initialized_solution_;
  const auto phi_offset = fieldOffset(phi_);
  for (std::size_t i = 0u; i < fieldCount(phi_); ++i) {
    raw_candidate[phi_offset + i] *= svmp::FE::Real{1.5};
  }
  raw_candidate[phi_offset] += svmp::FE::Real{0.01};
  scatterFeOrderedSolution(history().u(), raw_candidate);
  refreshCurrentCandidate(
      "application-driver-conservative-phase-reconciliation-only-raw");
  requests_.front().reinitialization.enabled = false;
  requests_.front().conservative_phase.reconcile_geometry = true;

  auto result = applyPreparedConservativePhaseCandidate();
  ASSERT_TRUE(result.accept_step);
  EXPECT_TRUE(result.changed);
  ASSERT_EQ(result.maintenance_ledgers.size(), 1u);
  const auto& ledger = result.maintenance_ledgers.front();
  EXPECT_TRUE(ledger.transport_stage.success);
  EXPECT_FALSE(ledger.reinitialization_due);
  EXPECT_FALSE(ledger.reinitialization_applied);
  EXPECT_DOUBLE_EQ(ledger.raw_post_transport_geometry_measure,
                   ledger.post_reinitialization_geometry_measure);
  EXPECT_GT(ledger.post_reinitialization_mismatch.maximum_nodal_residual,
            ledger.post_correction_mismatch.maximum_nodal_residual +
                svmp::FE::Real{1.0e-8});
  EXPECT_GT(ledger.reconciliation.iterations, 0);
  EXPECT_LE(ledger.post_correction_mismatch.maximum_nodal_residual,
            svmp::FE::Real{1.0e-10});
  ASSERT_NE(result.geometry_transaction, nullptr);
  ASSERT_NO_THROW(result.geometry_transaction->commit());
  result.geometry_transaction.reset();
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       ReinitializesBeforeLocalReconciliationAndRetainsEveryStage)
{
  auto raw_candidate = initialized_solution_;
  const auto phi_offset = fieldOffset(phi_);
  for (std::size_t i = 0u; i < fieldCount(phi_); ++i) {
    raw_candidate[phi_offset + i] *= svmp::FE::Real{2.0};
  }
  raw_candidate[phi_offset] += svmp::FE::Real{0.01};
  scatterFeOrderedSolution(history().u(), raw_candidate);
  refreshCurrentCandidate(
      "application-driver-conservative-phase-reinitialization-raw");

  auto& reinitialization = requests_.front().reinitialization;
  reinitialization.enabled = true;
  reinitialization.cadence_steps = 1;
  reinitialization.max_iterations = 100;
  reinitialization.signed_distance_tolerance = 1.0e-10;

  testing::internal::CaptureStdout();
  auto result = applyPreparedConservativePhaseCandidate();
  const auto output = testing::internal::GetCapturedStdout();

  ASSERT_TRUE(result.accept_step);
  ASSERT_TRUE(result.changed);
  ASSERT_NE(result.geometry_transaction, nullptr);
  ASSERT_EQ(result.maintenance_ledgers.size(), requests_.size());
  const auto& ledger = result.maintenance_ledgers.front();
  EXPECT_TRUE(ledger.transport_stage.success);
  EXPECT_TRUE(ledger.transport_stage.correction.success);
  EXPECT_TRUE(
      ledger.transport_stage.correction.component_balance_satisfied);
  EXPECT_TRUE(ledger.transport_stage.correction
                  .component_measure_closure_satisfied);
  EXPECT_DOUBLE_EQ(
      ledger.transport_stage.correction.component_activity_tolerance,
      requests_.front().conservative_phase.component_activity_tolerance);
  EXPECT_EQ(ledger.transport_stage.correction.nodes.size(),
            fieldCount(phase_));
  EXPECT_FALSE(ledger.transport_stage.correction.edges.empty());
  EXPECT_FALSE(ledger.transport_stage.correction.components.empty());
  EXPECT_TRUE(ledger.reinitialization_due);
  EXPECT_TRUE(ledger.reinitialization_applied);
  EXPECT_TRUE(ledger.reinitialization.success);
  EXPECT_TRUE(ledger.reinitialization.converged);
  EXPECT_GT(ledger.reinitialization.max_abs_update,
            svmp::FE::Real{0.0});
  EXPECT_NEAR(ledger.raw_post_transport_phase_measure,
              ledger.post_limit_phase_measure,
              svmp::FE::Real{1.0e-12});
  EXPECT_GT(
      ledger.post_reinitialization_mismatch.maximum_nodal_residual,
      ledger.post_correction_mismatch.maximum_nodal_residual +
          svmp::FE::Real{1.0e-8});
  EXPECT_LE(ledger.post_correction_mismatch.maximum_nodal_residual,
            svmp::FE::Real{1.0e-10});
  EXPECT_NE(output.find("raw_post_transport_phase_measure="),
            std::string::npos);
  EXPECT_NE(output.find("post_limit_phase_measure="),
            std::string::npos);
  EXPECT_NE(output.find("post_reinitialization_geometry_measure="),
            std::string::npos);
  EXPECT_NE(output.find("post_correction_geometry_measure="),
            std::string::npos);
  EXPECT_NE(output.find("transport_nodes="), std::string::npos);
  EXPECT_NE(output.find("transport_edges="), std::string::npos);
  EXPECT_NE(output.find("transport_components="), std::string::npos);
  EXPECT_NE(output.find("transport_component_activity_tolerance="),
            std::string::npos);
  EXPECT_NE(output.find("transport_subthreshold_component_present="),
            std::string::npos);
  EXPECT_NE(
      output.find(
          "transport_limited_component_transfer_closure_residual="),
      std::string::npos);
  EXPECT_NE(output.find("diagnostic=conservative_phase_component_ledger"),
            std::string::npos);

  const auto projection = projectCurrentConservativePhaseGeometry(
      *sim_.fe_system, requests_.front());
  ASSERT_TRUE(projection.success) << projection.diagnostic;
  auto& graph = requireCurrentConservativePhaseGraph(
      *sim_.fe_system, requests_.front());
  const auto accepted_phase = fieldSlice(
      gatherFeOrderedSolution(history().u()), phase_);
  ASSERT_EQ(projection.liquid_phase_mass.size(), graph.nodes);
  for (std::size_t node = 0u; node < graph.nodes; ++node) {
    EXPECT_NEAR(projection.liquid_phase_mass[node],
                graph.lumped_control_volume[node] * accepted_phase[node],
                svmp::FE::Real{1.0e-10});
  }

  ASSERT_NO_THROW(result.geometry_transaction->commit());
  result.geometry_transaction.reset();
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       WritesFluxArtifactOnlyAfterAnAcceptedCadenceStep)
{
  const auto unique = std::chrono::steady_clock::now()
                          .time_since_epoch()
                          .count();
  const auto output_directory =
      std::filesystem::temp_directory_path() /
      ("svmp-conservative-phase-workflow-artifact-" +
       std::to_string(unique));
  params_->general_simulation_parameters.save_results_in_folder.set(
      output_directory.string());
  auto& phase_options = requests_.front().conservative_phase;
  phase_options.write_flux_artifacts = true;
  phase_options.flux_artifact_cadence_steps = 2;
  phase_options.fixed_flux_regions =
      svmp::FE::level_set::parseLevelSetPhaseRegionBoxes(
          "test_film|wall_film|*|*|*|*|*|*");

  auto result = applyPreparedConservativePhaseCandidate();
  ASSERT_TRUE(result.accept_step);
  ASSERT_NE(result.geometry_transaction, nullptr);
  ASSERT_NO_THROW(result.geometry_transaction->commit());
  result.geometry_transaction.reset();

  ASSERT_NO_THROW(writeAcceptedConservativePhaseArtifacts(
      *params_,
      requests_,
      result,
      1u,
      svmp::FE::Real{0.05},
      svmp::FE::Real{0.05},
      history().u().valueRevision(),
      svmp::MeshComm::world()));
  EXPECT_FALSE(std::filesystem::exists(
      output_directory / "conservative_phase_flux"));

  phase_options.flux_artifact_cadence_steps = 1;
  ASSERT_NO_THROW(writeAcceptedConservativePhaseArtifacts(
      *params_,
      requests_,
      result,
      1u,
      svmp::FE::Real{0.05},
      svmp::FE::Real{0.05},
      history().u().valueRevision(),
      svmp::MeshComm::world()));
  const auto artifact_path =
      output_directory / "conservative_phase_flux" /
      "conservative_phase_flux_phase_step_00000001.json";
  ASSERT_TRUE(std::filesystem::is_regular_file(artifact_path));
  std::ifstream input(artifact_path);
  ASSERT_TRUE(input.is_open());
  const std::string contents{
      std::istreambuf_iterator<char>{input},
      std::istreambuf_iterator<char>{}};
  EXPECT_NE(contents.find(
                "\"maintenance_ordering\":\"conservative_phase_transport_then_raw_geometry_rebuild_then_wall_aware_reinitialization_then_local_geometry_reconciliation_then_validation_then_commit\""),
            std::string::npos);
  EXPECT_NE(contents.find("\"raw_post_transport_phase_measure\":"),
            std::string::npos);
  EXPECT_NE(contents.find("\"post_reinitialization_phase_measure\":"),
            std::string::npos);
  EXPECT_NE(contents.find("\"post_correction_phase_measure\":"),
            std::string::npos);
  EXPECT_NE(contents.find("\"tracked_regions\":1"),
            std::string::npos);
  EXPECT_NE(contents.find(
                "\"regions\":[{\"name\":\"test_film\",\"kind\":\"wall_film\""),
            std::string::npos);
  EXPECT_THROW(
      writeAcceptedConservativePhaseArtifacts(
          *params_,
          requests_,
          result,
          1u,
          svmp::FE::Real{0.05},
          svmp::FE::Real{0.05},
          history().u().valueRevision(),
          svmp::MeshComm::world()),
      std::runtime_error);

  const auto duplicate_request = requests_.front();
  ASSERT_TRUE(result.maintenance_ledgers.front()
                  .split_stage_provenance.has_value());
  auto& second_step_provenance =
      *result.maintenance_ledgers.front().split_stage_provenance;
  second_step_provenance.prospective_step = 2u;
  second_step_provenance.step_start_time = svmp::FE::Real{0.05};
  second_step_provenance.step_end_time = svmp::FE::Real{0.10};
  second_step_provenance.q_input_time = svmp::FE::Real{0.05};
  second_step_provenance.velocity_state_time = svmp::FE::Real{0.10};
  second_step_provenance.time_step = svmp::FE::Real{0.05};
  const auto duplicate_ledger = result.maintenance_ledgers.front();
  requests_.push_back(duplicate_request);
  result.maintenance_ledgers.push_back(duplicate_ledger);
  const auto second_step_artifact_path =
      output_directory / "conservative_phase_flux" /
      "conservative_phase_flux_phase_step_00000002.json";
  EXPECT_THROW(
      writeAcceptedConservativePhaseArtifacts(
          *params_,
          requests_,
          result,
          2u,
          svmp::FE::Real{0.10},
          svmp::FE::Real{0.05},
          history().u().valueRevision(),
          svmp::MeshComm::world()),
      std::runtime_error);
  EXPECT_FALSE(std::filesystem::exists(second_step_artifact_path))
      << "A later artifact failure must remove every earlier file from the "
         "same accepted-step batch.";
  requests_.pop_back();
  result.maintenance_ledgers.pop_back();

  std::error_code cleanup_error;
  std::filesystem::remove_all(output_directory, cleanup_error);
  EXPECT_FALSE(cleanup_error);
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       NonconvergedReinitializationRejectsAndRestoresGeometry)
{
  auto raw_candidate = initialized_solution_;
  const auto phi_offset = fieldOffset(phi_);
  for (std::size_t i = 0u; i < fieldCount(phi_); ++i) {
    raw_candidate[phi_offset + i] *= svmp::FE::Real{4.0};
  }
  scatterFeOrderedSolution(history().u(), raw_candidate);
  refreshCurrentCandidate(
      "application-driver-conservative-phase-reinitialization-reject-raw");
  const auto* raw_context = sim_.fe_system->cutIntegrationContext();
  ASSERT_NE(raw_context, nullptr);
  const auto lifecycle_revision = lifecycle_.valueRevision();

  auto& reinitialization = requests_.front().reinitialization;
  reinitialization.enabled = true;
  reinitialization.cadence_steps = 1;
  reinitialization.max_iterations = 1;
  reinitialization.pseudo_time_step_scale = 1.0e-3;
  reinitialization.signed_distance_tolerance = 1.0e-14;

  const auto result = applyPreparedConservativePhaseCandidate();
  EXPECT_FALSE(result.accept_step);
  EXPECT_FALSE(result.changed);
  EXPECT_EQ(result.geometry_transaction, nullptr);
  ASSERT_EQ(result.maintenance_ledgers.size(), requests_.size());
  EXPECT_TRUE(result.maintenance_ledgers.front().reinitialization_due);
  EXPECT_TRUE(result.maintenance_ledgers.front().reinitialization.success);
  EXPECT_FALSE(result.maintenance_ledgers.front().reinitialization.converged);
  EXPECT_EQ(gatherFeOrderedSolution(history().u()), raw_candidate);
  EXPECT_EQ(sim_.fe_system->cutIntegrationContext(), raw_context);
  EXPECT_EQ(lifecycle_.valueRevision(), lifecycle_revision);
  EXPECT_FALSE(sim_.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle_.transactionActive());
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       PostAcceptanceMaintenanceDoesNotRepeatCandidateOwnedRepair)
{
  requests_.front().reinitialization.enabled = true;
  requests_.front().reinitialization.cadence_steps = 1;
  const auto before = gatherFeOrderedSolution(history().u());

  EXPECT_FALSE(applyLevelSetMaintenance(sim_, history(), requests_));
  EXPECT_EQ(gatherFeOrderedSolution(history().u()), before);
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       ContactParentProtectionRejectsAnIncompatibleGeometryTarget)
{
  auto& graph = requireCurrentConservativePhaseGraph(
      *sim_.fe_system, requests_.front());
  const auto& mesh_access = sim_.fe_system->meshAccess();
  const auto parent_global_id = mesh_access.globalEntityIdsAvailable()
      ? mesh_access.getCellGlobalId(0)
      : svmp::FE::GlobalIndex{0};
  const std::array<
      svmp::FE::level_set::LevelSetWallContactConstraint, 1>
      constraints{{
          svmp::FE::level_set::LevelSetWallContactConstraint{
              .kind = svmp::FE::level_set::
                  LevelSetWallContactConstraintKind::PrescribedAngle,
              .interface_marker = 911,
              .boundary_marker = 41,
              .parent_cell_global_id = parent_global_id,
              .geometry_revision = 1u,
          },
      }};
  const auto protected_nodes = conservativePhaseContactProtectedNodes(
      *sim_.fe_system,
      requests_.front(),
      graph,
      constraints);
  ASSERT_EQ(protected_nodes.size(), graph.nodes);
  EXPECT_EQ(std::count(protected_nodes.begin(), protected_nodes.end(), 1u),
            static_cast<std::ptrdiff_t>(graph.nodes));

  auto raw_candidate = initialized_solution_;
  raw_candidate[fieldOffset(phi_)] += svmp::FE::Real{0.01};
  scatterFeOrderedSolution(history().u(), raw_candidate);
  refreshCurrentCandidate(
      "application-driver-conservative-phase-contact-protection-raw");
  auto candidate = gatherFeOrderedSolution(history().u());
  const auto candidate_before = candidate;
  const auto previous_phase = fieldSlice(
      gatherFeOrderedSolution(history().uPrev()), phase_);
  std::vector<svmp::FE::Real> target_mass(
      graph.nodes, svmp::FE::Real{0.0});
  for (std::size_t node = 0u; node < graph.nodes; ++node) {
    target_mass[node] =
        graph.lumped_control_volume[node] * previous_phase[node];
  }

  LevelSetMaintenanceGeometryTransaction transaction(
      sim_, lifecycle_, refresh_cache_, active_requests_);
  const auto reconciliation = reconcileConservativePhaseGeometry(
      sim_,
      requests_.front(),
      *params_,
      target_mass,
      candidate,
      transaction,
      protected_nodes);
  EXPECT_FALSE(reconciliation.success);
  EXPECT_FALSE(reconciliation.target_reached);
  EXPECT_EQ(reconciliation.contact_protected_nodes, graph.nodes);
  EXPECT_GT(reconciliation.maximum_removed_contact_increment,
            svmp::FE::Real{0.0});
  EXPECT_NE(reconciliation.diagnostic.find(
                "without changing accepted wall-contact parent cells"),
            std::string::npos);
  EXPECT_EQ(candidate, candidate_before);
  ASSERT_NO_THROW(transaction.rollback());
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       LaterRejectionRestoresTheRawCandidateAndEveryGeometryRevision)
{
  auto raw_candidate = initialized_solution_;
  raw_candidate[fieldOffset(phi_)] += svmp::FE::Real{0.01};
  const auto phase_offset = fieldOffset(phase_);
  for (std::size_t i = 0u; i < fieldCount(phase_); ++i) {
    raw_candidate[phase_offset + i] = svmp::FE::Real{0.8};
  }
  scatterFeOrderedSolution(history().u(), raw_candidate);
  refreshCurrentCandidate(
      "application-driver-conservative-phase-rollback-raw");
  const auto* raw_context = sim_.fe_system->cutIntegrationContext();
  ASSERT_NE(raw_context, nullptr);
  const auto lifecycle_revision = lifecycle_.valueRevision();
  const auto constraint_revision =
      sim_.fe_system->constraintLayoutRevision();
  const auto sparsity_revision = sim_.fe_system->sparsityPatternRevision();
  const auto cache_before = refresh_cache_;
  requests_.front().reinitialization.enabled = true;
  requests_.front().reinitialization.cadence_steps = 1;
  requests_.front().reinitialization.max_iterations = 100;
  requests_.front().reinitialization.signed_distance_tolerance = 1.0e-10;

  auto result = applyPreparedConservativePhaseCandidate();
  EXPECT_TRUE(result.accept_step);
  EXPECT_TRUE(result.changed);
  ASSERT_NE(result.geometry_transaction, nullptr);
  ASSERT_EQ(result.maintenance_ledgers.size(), requests_.size());
  EXPECT_TRUE(result.maintenance_ledgers.front().reinitialization_due);
  EXPECT_TRUE(result.maintenance_ledgers.front().reinitialization_applied);
  EXPECT_TRUE(result.maintenance_ledgers.front().reinitialization.converged);
  EXPECT_TRUE(sim_.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_TRUE(lifecycle_.transactionActive());
  EXPECT_NE(sim_.fe_system->cutIntegrationContext(), raw_context);

  const auto staged_solution = gatherFeOrderedSolution(history().u());
  const auto staged_phi = fieldSlice(staged_solution, phi_);
  const auto raw_phi = fieldSlice(raw_candidate, phi_);
  ASSERT_EQ(staged_phi.size(), raw_phi.size());
  std::vector<svmp::FE::Real> coefficient_updates(staged_phi.size());
  for (std::size_t node = 0u; node < staged_phi.size(); ++node) {
    coefficient_updates[node] = staged_phi[node] - raw_phi[node];
  }
  const auto [minimum_update, maximum_update] = std::minmax_element(
      coefficient_updates.begin(), coefficient_updates.end());
  ASSERT_NE(minimum_update, coefficient_updates.end());
  ASSERT_NE(maximum_update, coefficient_updates.end());
  EXPECT_GT(*maximum_update - *minimum_update,
            svmp::FE::Real{1.0e-5});

  const auto staged_projection = projectCurrentConservativePhaseGeometry(
      *sim_.fe_system, requests_.front());
  ASSERT_TRUE(staged_projection.success) << staged_projection.diagnostic;
  auto& graph = requireCurrentConservativePhaseGraph(
      *sim_.fe_system, requests_.front());
  const auto staged_phase = fieldSlice(staged_solution, phase_);
  ASSERT_EQ(staged_projection.liquid_phase_mass.size(), graph.nodes);
  ASSERT_EQ(staged_phase.size(), graph.nodes);
  for (std::size_t node = 0u; node < graph.nodes; ++node) {
    EXPECT_NEAR(staged_projection.liquid_phase_mass[node],
                graph.lumped_control_volume[node] * staged_phase[node],
                svmp::FE::Real{1.0e-10});
  }

  ASSERT_NO_THROW(rollbackConservativePhaseCandidate(history(), result));
  EXPECT_EQ(gatherFeOrderedSolution(history().u()), raw_candidate);
  EXPECT_EQ(sim_.fe_system->cutIntegrationContext(), raw_context);
  EXPECT_EQ(lifecycle_.valueRevision(), lifecycle_revision);
  EXPECT_EQ(sim_.fe_system->constraintLayoutRevision(),
            constraint_revision);
  EXPECT_EQ(sim_.fe_system->sparsityPatternRevision(),
            sparsity_revision);
  EXPECT_FALSE(sim_.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle_.transactionActive());
  ASSERT_EQ(refresh_cache_.last_signature.has_value(),
            cache_before.last_signature.has_value());
  if (refresh_cache_.last_signature.has_value()) {
    EXPECT_TRUE(*refresh_cache_.last_signature ==
                *cache_before.last_signature);
  }
  ASSERT_EQ(refresh_cache_.last_vector_signature.has_value(),
            cache_before.last_vector_signature.has_value());
  if (refresh_cache_.last_vector_signature.has_value()) {
    EXPECT_TRUE(*refresh_cache_.last_vector_signature ==
                *cache_before.last_vector_signature);
  }
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       CourantRejectionLeavesTheRawCandidateAndGeometryUntouched)
{
  history().setDt(1.0);
  auto raw_candidate = initialized_solution_;
  const auto phase_offset = fieldOffset(phase_);
  for (std::size_t i = 0u; i < fieldCount(phase_); ++i) {
    raw_candidate[phase_offset + i] = svmp::FE::Real{0.7};
  }
  scatterFeOrderedSolution(history().u(), raw_candidate);
  refreshCurrentCandidate(
      "application-driver-conservative-phase-courant-raw");
  const auto* raw_context = sim_.fe_system->cutIntegrationContext();
  requests_.front().velocity.constant_value = {
      svmp::FE::Real{10.0}, svmp::FE::Real{0.0}, svmp::FE::Real{0.0}};
  requests_.front().conservative_phase
      .impermeable_normal_velocity_tolerance = svmp::FE::Real{2.0};

  const auto result = applyPreparedConservativePhaseCandidate();
  EXPECT_FALSE(result.accept_step);
  EXPECT_FALSE(result.changed);
  EXPECT_EQ(result.geometry_transaction, nullptr);
  EXPECT_EQ(gatherFeOrderedSolution(history().u()), raw_candidate);
  EXPECT_EQ(sim_.fe_system->cutIntegrationContext(), raw_context);
  EXPECT_FALSE(sim_.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle_.transactionActive());
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       GeometryDisplacementRejectionLeavesTheRawCandidateAndGeometryUntouched)
{
  auto raw_candidate = initialized_solution_;
  raw_candidate[fieldOffset(phi_)] += svmp::FE::Real{0.5};
  scatterFeOrderedSolution(history().u(), raw_candidate);
  refreshCurrentCandidate(
      "application-driver-conservative-phase-displacement-raw");
  const auto* raw_context = sim_.fe_system->cutIntegrationContext();
  ASSERT_NE(raw_context, nullptr);
  const auto lifecycle_revision = lifecycle_.valueRevision();

  const auto result = applyPreparedConservativePhaseCandidate();
  EXPECT_FALSE(result.accept_step);
  EXPECT_FALSE(result.changed);
  EXPECT_EQ(result.geometry_transaction, nullptr);
  EXPECT_EQ(gatherFeOrderedSolution(history().u()), raw_candidate);
  EXPECT_EQ(sim_.fe_system->cutIntegrationContext(), raw_context);
  EXPECT_EQ(lifecycle_.valueRevision(), lifecycle_revision);
  EXPECT_FALSE(sim_.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle_.transactionActive());
}

class ApplicationDriverBoundPreservingCandidatesTest
    : public ::testing::Test {
protected:
  void SetUp() override
  {
    mesh_ = makeWorkflowQuadPatch2x2Mesh();
    auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
        svmp::FE::ElementType::Quad4,
        /*order=*/1);
    auto system =
        std::make_unique<svmp::FE::systems::FESystem>(mesh_);
    phi_ = system->addField(svmp::FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system->setup({}));

    factory_ = svmp::FE::backends::BackendFactory::create(
        svmp::FE::backends::BackendKind::FSILS);
    ASSERT_NE(factory_, nullptr);
    history_ = svmp::FE::timestepping::TimeHistory::allocate(
        *factory_,
        system->dofHandler().getNumDofs(),
        /*history_depth=*/2,
        /*allocate_second_order_state=*/true);
    history_.setDt(0.2);
    history_.setPrevDt(0.2);

    sim_.primary_mesh = mesh_;
    sim_.fe_system = std::move(system);
  }

  [[nodiscard]] std::size_t solutionSize() const
  {
    return static_cast<std::size_t>(
        sim_.fe_system->dofHandler().getNumDofs());
  }

  [[nodiscard]] std::size_t phiOffset() const
  {
    const auto offset = sim_.fe_system->fieldDofOffset(phi_);
    if (offset < 0) {
      throw std::runtime_error(
          "ApplicationDriver bound-preserving test has no phi offset");
    }
    return static_cast<std::size_t>(offset);
  }

  [[nodiscard]] LevelSetMaintenanceRequest requestWithVelocity(
      std::array<svmp::FE::Real, 3> velocity) const
  {
    LevelSetMaintenanceRequest request{};
    request.level_set_field_name = "phi";
    request.velocity.source =
        svmp::FE::level_set::LevelSetVelocitySource::ConstantVector;
    request.velocity.constant_value = velocity;
    request.bound_preserving.enabled = true;
    return request;
  }

  void setCandidateState(std::span<const svmp::FE::Real> previous,
                         std::span<const svmp::FE::Real> candidate,
                         std::span<const svmp::FE::Real> rates)
  {
    ASSERT_EQ(previous.size(), solutionSize());
    ASSERT_EQ(candidate.size(), solutionSize());
    ASSERT_EQ(rates.size(), solutionSize());
    scatterFeOrderedSolution(history_.uPrev(), previous);
    scatterFeOrderedSolution(history_.uPrev2(), previous);
    scatterFeOrderedSolution(history_.u(), candidate);
    scatterFeOrderedSolution(history_.uDot(), rates);
  }

  std::shared_ptr<svmp::Mesh> mesh_{};
  svmp::FE::FieldId phi_{svmp::FE::INVALID_FIELD_ID};
  std::unique_ptr<svmp::FE::backends::BackendFactory> factory_{};
  svmp::FE::timestepping::TimeHistory history_{};
  application::core::SimulationComponents sim_{};
};

TEST_F(ApplicationDriverBoundPreservingCandidatesTest,
       LimitedCandidateRequestsNonlinearRetryWithoutMutatingHistory)
{
  std::vector<svmp::FE::Real> previous(solutionSize(), 1.0);
  auto raw_candidate = previous;
  const auto limited_dof = phiOffset();
  raw_candidate[limited_dof] = -2.0;
  std::vector<svmp::FE::Real> rates(solutionSize(), 0.0);
  for (std::size_t i = 0; i < rates.size(); ++i) {
    rates[i] = 0.25 + 0.1 * static_cast<svmp::FE::Real>(i);
  }
  setCandidateState(previous, raw_candidate, rates);

  testing::internal::CaptureStdout();
  const auto result = applyLevelSetBoundPreservingCandidates(
      sim_,
      history_,
      std::vector<LevelSetMaintenanceRequest>{
          requestWithVelocity({0.0, 0.0, 0.0})});
  const auto output = testing::internal::GetCapturedStdout();

  EXPECT_FALSE(result.accept_step);
  EXPECT_TRUE(result.changed);
  EXPECT_NE(output.find(
                "reason=bound_preserving_limiter_requires_nonlinear_retry"),
            std::string::npos);
  EXPECT_EQ(gatherFeOrderedSolution(history_.u()), raw_candidate);
  EXPECT_EQ(gatherFeOrderedSolution(history_.uDot()), rates);
  EXPECT_EQ(gatherFeOrderedSolution(history_.uPrev()), previous);
}

TEST_F(ApplicationDriverBoundPreservingCandidatesTest,
       CourantViolationReturnsRetryableRejectionWithoutMutatingCandidate)
{
  history_.setDt(0.75);
  std::vector<svmp::FE::Real> previous(solutionSize(), 1.0);
  const auto raw_candidate = previous;
  std::vector<svmp::FE::Real> rates(solutionSize(), -0.25);
  setCandidateState(previous, raw_candidate, rates);

  auto request = requestWithVelocity({2.0, 0.0, 0.0});
  request.bound_preserving.bound_tolerance = 10.0;
  request.bound_preserving.courant_tolerance = 1.0e-12;
  request.bound_preserving.enforce_impermeable_boundaries = false;
  testing::internal::CaptureStdout();
  const auto result = applyLevelSetBoundPreservingCandidates(
      sim_,
      history_,
      std::vector<LevelSetMaintenanceRequest>{request});
  const auto output = testing::internal::GetCapturedStdout();

  EXPECT_FALSE(result.accept_step);
  EXPECT_FALSE(result.changed);
  EXPECT_NE(output.find("reason=bound_preserving_courant_contract"),
            std::string::npos);
  EXPECT_EQ(gatherFeOrderedSolution(history_.u()), raw_candidate);
  EXPECT_EQ(gatherFeOrderedSolution(history_.uDot()), rates);
  EXPECT_EQ(gatherFeOrderedSolution(history_.uPrev()), previous);
}

TEST_F(ApplicationDriverBoundPreservingCandidatesTest,
       ImpermeableWallNormalVelocityFailsClosedWithoutMutatingCandidate)
{
  history_.setDt(0.1);
  std::vector<svmp::FE::Real> previous(solutionSize(), 1.0);
  const auto raw_candidate = previous;
  std::vector<svmp::FE::Real> rates(solutionSize(), 0.125);
  setCandidateState(previous, raw_candidate, rates);

  try {
    (void)applyLevelSetBoundPreservingCandidates(
        sim_,
        history_,
        std::vector<LevelSetMaintenanceRequest>{
            requestWithVelocity({0.0, 0.5, 0.0})});
    FAIL() << "A nonzero normal wall velocity must fail closed";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find(
                  "incompatible impermeable-wall velocity"),
              std::string::npos);
  }

  EXPECT_EQ(gatherFeOrderedSolution(history_.u()), raw_candidate);
  EXPECT_EQ(gatherFeOrderedSolution(history_.uDot()), rates);
  EXPECT_EQ(gatherFeOrderedSolution(history_.uPrev()), previous);
}

TEST_F(ApplicationDriverBoundPreservingCandidatesTest,
       InBoundsNontrivialCandidatePassesWithoutLimiterOrRateChanges)
{
  std::vector<svmp::FE::Real> vertex_values(mesh_->n_vertices(), 0.0);
  for (std::size_t vertex = 0; vertex < mesh_->n_vertices(); ++vertex) {
    vertex_values[vertex] = workflowVertexPoint(*mesh_, vertex)[0];
  }
  const auto phi_coefficients = projectWorkflowVertexValues(
      *sim_.fe_system,
      phi_,
      vertex_values,
      /*components=*/1u,
      "ApplicationDriver bound-preserving pass-through phi");
  std::vector<svmp::FE::Real> previous(solutionSize(), 0.0);
  writeWorkflowFieldSlice(
      *sim_.fe_system, phi_, phi_coefficients, previous);
  auto raw_candidate = previous;
  for (std::size_t i = 0; i < phi_coefficients.size(); ++i) {
    raw_candidate[phiOffset() + i] *= 0.5;
  }
  ASSERT_NE(raw_candidate, previous);
  std::vector<svmp::FE::Real> rates(solutionSize(), 0.375);
  setCandidateState(previous, raw_candidate, rates);

  const auto result = applyLevelSetBoundPreservingCandidates(
      sim_,
      history_,
      std::vector<LevelSetMaintenanceRequest>{
          requestWithVelocity({0.0, 0.0, 0.0})});

  EXPECT_TRUE(result.accept_step);
  EXPECT_FALSE(result.changed);
  EXPECT_EQ(gatherFeOrderedSolution(history_.u()), raw_candidate);
  EXPECT_EQ(gatherFeOrderedSolution(history_.uDot()), rates);
  EXPECT_EQ(gatherFeOrderedSolution(history_.uPrev()), previous);
}

void addValidComponentTransferLedger(
    svmp::FE::level_set::LevelSetGlobalShiftCorrectionResult& result)
{
  result.negative_component_topology_preserved = true;
  result.negative_component_volume_transfers.push_back(
      svmp::FE::level_set::LevelSetComponentVolumeTransfer{
          .component_global_vertex_id = 0,
      });
}

TEST(ApplicationDriverLevelSetVolumeCorrection,
     ReportsAuthoritativeFreeSurfacePotentialChange)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowTriangleMesh();
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Triangle3,
      /*order=*/1);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1,
  });
  svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalParameters parameters;
  parameters.surface_tension = 1.0;
  parameters.volume_multiplier = 0.5;
  system->declareFreeSurfaceDiscreteFunctional(
      svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration{
          .interface_marker = 808,
          .level_set_field = phi,
          .geometry_domain_id = "volume_work_fixture",
          .parameters = parameters,
          .owner_component =
              "ApplicationDriverLevelSetVolumeCorrection.WorkFixture",
      });
  ASSERT_NO_THROW(system->setup({}));
  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);

  const auto make_state = [](std::uint64_t source_revision,
                             std::uint64_t snapshot_revision,
                             svmp::FE::Real liquid_volume,
                             svmp::FE::Real liquid_gas_area,
                             svmp::FE::Real wetted_wall_area,
                             svmp::FE::Real contact_measure,
                             svmp::FE::Real surface_energy,
                             svmp::FE::Real wall_energy,
                             svmp::FE::Real volume_potential,
                             svmp::FE::Real total_potential) {
    svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState state;
    state.interface_marker = 808;
    state.geometry_revision.source_id = "field:0";
    state.geometry_revision.domain_id = "volume_work_fixture";
    state.geometry_revision.interface_marker = 808;
    state.geometry_revision.source_value_revision = source_revision;
    state.geometry_revision.snapshot_revision_key = snapshot_revision;
    state.state.snapshot_revision_key = snapshot_revision;
    state.state.surface_tension = 1.0;
    state.state.volume_multiplier = 0.5;
    state.state.owned_liquid_volume = liquid_volume;
    state.state.owned_liquid_gas_area = liquid_gas_area;
    state.state.owned_wetted_wall_area = wetted_wall_area;
    state.state.owned_contact_measure = contact_measure;
    state.state.liquid_gas_surface_energy = surface_energy;
    state.state.young_wall_energy = wall_energy;
    state.state.volume_constraint_potential = volume_potential;
    state.state.total_potential = total_potential;
    return state;
  };
  const std::vector<
      svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState>
      before{make_state(
          1u, 101u, 0.5, 2.0, 0.5, 0.25, 2.0, -0.5, 0.25, 1.75)};
  const std::vector<
      svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState>
      after{make_state(
          2u, 102u, 0.375, 2.25, 0.75, 0.125, 2.25, -0.375,
          0.125, 2.0)};
  LevelSetVolumeCorrectionMaintenanceEvent event;
  event.level_set_field = phi;
  event.level_set_field_name = "phi";
  event.completed_step = 4;
  event.correction.correction_applied = true;
  event.correction.applied_shift = 0.125;
  event.correction.total_component_volume_transfer = -0.125;
  event.correction.negative_component_volume_transfers.push_back(
      svmp::FE::level_set::LevelSetComponentVolumeTransfer{
          .component_global_vertex_id = 0,
          .initial_negative_volume = 0.5,
          .corrected_negative_volume = 0.375,
          .volume_transfer = -0.125,
      });
  const std::vector<LevelSetVolumeCorrectionMaintenanceEvent> events{event};

  testing::internal::CaptureStdout();
  ASSERT_NO_THROW(logLevelSetVolumeCorrectionFreeSurfaceWork(
      sim, events, before, after));
  const auto output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find("diagnostic=level_set_volume_correction_work"),
            std::string::npos);
  EXPECT_NE(output.find("scope=global_shift_only"), std::string::npos);
  EXPECT_NE(output.find("numerical_work_sign=energy_after_minus_before"),
            std::string::npos);
  EXPECT_NE(output.find("free_surface_functional_count=1"),
            std::string::npos);
  EXPECT_NE(output.find("initial_snapshot_revision=101"),
            std::string::npos);
  EXPECT_NE(output.find("corrected_snapshot_revision=102"),
            std::string::npos);
  EXPECT_NE(output.find("surface_energy_change="), std::string::npos);
  EXPECT_NE(output.find("young_wall_energy_change="), std::string::npos);
  EXPECT_NE(output.find("volume_constraint_potential_change="),
            std::string::npos);
  EXPECT_NE(output.find("liquid_volume_change=-0.125"),
            std::string::npos);
  EXPECT_NE(output.find("surface_energy_change=0.25"),
            std::string::npos);
  EXPECT_NE(output.find("young_wall_energy_change=0.125"),
            std::string::npos);
  EXPECT_NE(output.find("volume_constraint_potential_change=-0.125"),
            std::string::npos);
  EXPECT_NE(output.find("numerical_free_surface_work=0.25"),
            std::string::npos);
#endif
}

TEST(ApplicationDriverLevelSetVolumeCorrection,
     CandidateGeometryFailureRestoresCompleteMaintenanceTransaction)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr int interface_marker = 809;
  auto mesh = makeWorkflowTriangleMesh();
  (void)svmp::MeshFields::attach_field(
      mesh->local_mesh(),
      svmp::EntityKind::Vertex,
      "phi",
      svmp::FieldScalarType::Float64,
      1);
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Triangle3,
      /*order=*/1);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1,
  });
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_vertex_values(mesh->n_vertices(), 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    phi_vertex_values[vertex] =
        workflowVertexPoint(*mesh, vertex)[0] - svmp::FE::Real{0.5};
  }
  const auto coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      phi_vertex_values,
      /*components=*/1u,
      "ApplicationDriver maintenance transaction phi");
  std::vector<svmp::FE::Real> initial(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, coefficients, initial);

  std::unique_ptr<svmp::FE::backends::BackendFactory> factory;
  try {
    factory = svmp::FE::backends::BackendFactory::create(
        svmp::FE::backends::BackendKind::FSILS);
  } catch (const std::exception&) {
    GTEST_SKIP() << "Requires an available FE vector backend.";
  }
  auto history = svmp::FE::timestepping::TimeHistory::allocate(
      *factory,
      system->dofHandler().getNumDofs(),
      /*history_depth=*/2,
      /*allocate_second_order_state=*/true);
  history.setStepIndex(1);
  history.setDt(0.1);
  history.setPrevDt(0.1);
  scatterFeOrderedSolution(history.u(), initial);
  scatterFeOrderedSolution(history.uPrev(), initial);
  scatterFeOrderedSolution(history.uPrev2(), initial);
  std::vector<svmp::FE::Real> rates(initial.size(), 0.375);
  scatterFeOrderedSolution(history.uDot(), rates);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>transaction_interface</Generated_interface_domain_id>
      <Interface_marker>809</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
  const auto active_requests = activeCutVolumeRequests(*params);
  ASSERT_EQ(active_requests.size(), 1u);
  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
  ActiveCutContextRefreshCache refresh_cache;
  const auto initial_refresh_report =
      refreshActiveCutIntegrationContextFromSolutionCached(
          sim,
          *params,
          initial,
          lifecycle,
          refresh_cache,
          "application-driver-maintenance-transaction-initial");
  ASSERT_TRUE(initial_refresh_report.refreshed);
  ASSERT_NE(initial_refresh_report.topology_key, 0u);

  const auto* original_context = sim.fe_system->cutIntegrationContext();
  ASSERT_NE(original_context, nullptr);
  ASSERT_TRUE(original_context->hasGeneratedInterfaceMarker(interface_marker));
  const auto lifecycle_revision_before = lifecycle.valueRevision();
  const auto constraint_revision_before =
      sim.fe_system->constraintLayoutRevision();
  const auto sparsity_revision_before =
      sim.fe_system->sparsityPatternRevision();
  const auto constraint_count_before =
      sim.fe_system->constraints().numConstraints();
  const auto mesh_revisions_before = mesh->event_bus().revision_state();
  const auto refresh_cache_before = refresh_cache;
  const auto mesh_phi_handle = mesh->field_handle(
      svmp::EntityKind::Vertex, "phi");
  const auto mesh_phi_count =
      mesh->field_components(mesh_phi_handle) *
      mesh->field_entity_count(mesh_phi_handle);
  const auto* mesh_phi_data_before =
      static_cast<const double*>(mesh->field_data(mesh_phi_handle));
  ASSERT_NE(mesh_phi_data_before, nullptr);
  const std::vector<double> mesh_phi_before(
      mesh_phi_data_before, mesh_phi_data_before + mesh_phi_count);

  const auto current_revision_before = history.u().valueRevision();
  const auto previous_revision_before = history.uPrev().valueRevision();
  const auto previous2_revision_before = history.uPrev2().valueRevision();
  const auto rate_revision_before = history.uDot().valueRevision();
  const auto current_before = gatherFeOrderedSolution(history.u());
  const auto previous_before = gatherFeOrderedSolution(history.uPrev());
  const auto previous2_before = gatherFeOrderedSolution(history.uPrev2());
  const auto rates_before = gatherFeOrderedSolution(history.uDot());

  LevelSetMaintenanceRequest request{};
  request.level_set_field_name = "phi";
  request.volume_correction.enabled = true;
  request.volume_correction.cadence_steps = 1;
  request.volume_correction.use_initial_negative_volume_as_target = false;
  request.volume_correction.target_negative_volume = 0.36;
  request.volume_correction.minimum_relative_volume_error = 0.0;
  request.volume_correction.maximum_interface_displacement_fraction = 0.10;
  request.volume_correction
      .maximum_cumulative_interface_displacement_fraction = 1.0;
  std::vector<LevelSetMaintenanceRequest> requests{request};
  std::vector<LevelSetVolumeCorrectionMaintenanceEvent> published_events;
  std::unique_ptr<LevelSetMaintenanceGeometryTransaction>
      geometry_transaction;
  bool candidate_context_replaced = false;
  const LevelSetMaintenanceCandidateValidator reject_candidate =
      [&](std::span<const svmp::FE::Real> candidate,
          std::span<const LevelSetVolumeCorrectionMaintenanceEvent> events) {
        if (events.size() != 1u) {
          throw std::runtime_error(
              "injected maintenance candidate had incomplete event coverage");
        }
        geometry_transaction =
            std::make_unique<LevelSetMaintenanceGeometryTransaction>(
                sim, lifecycle, refresh_cache, active_requests);
        const auto report = geometry_transaction->refresh(*params, candidate);
        candidate_context_replaced =
            report.refreshed &&
            sim.fe_system->cutIntegrationContext() != original_context;
        throw std::runtime_error(
            "injected post-refresh maintenance validation failure");
      };

  testing::internal::CaptureStdout();
  EXPECT_THROW(
      (void)applyLevelSetMaintenance(
          sim,
          history,
          requests,
          {},
          {},
          {},
          &published_events,
          reject_candidate),
      std::runtime_error);
  const auto output = testing::internal::GetCapturedStdout();
  ASSERT_NE(geometry_transaction, nullptr);
  ASSERT_NO_THROW(geometry_transaction->rollback());
  EXPECT_TRUE(candidate_context_replaced);
  EXPECT_EQ(output.find("Level-set volume corrected"), std::string::npos);
  EXPECT_TRUE(published_events.empty());
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_FALSE(requests.front().volume_target_initialized);
  EXPECT_DOUBLE_EQ(
      requests.front().cumulative_volume_correction_interface_displacement,
      0.0);
  EXPECT_DOUBLE_EQ(
      requests.front().cumulative_volume_correction_contact_line_displacement,
      0.0);

  EXPECT_EQ(gatherFeOrderedSolution(history.u()), current_before);
  EXPECT_EQ(gatherFeOrderedSolution(history.uPrev()), previous_before);
  EXPECT_EQ(gatherFeOrderedSolution(history.uPrev2()), previous2_before);
  EXPECT_EQ(gatherFeOrderedSolution(history.uDot()), rates_before);
  EXPECT_EQ(history.u().valueRevision(), current_revision_before);
  EXPECT_EQ(history.uPrev().valueRevision(), previous_revision_before);
  EXPECT_EQ(history.uPrev2().valueRevision(), previous2_revision_before);
  EXPECT_EQ(history.uDot().valueRevision(), rate_revision_before);

  EXPECT_EQ(sim.fe_system->cutIntegrationContext(), original_context);
  EXPECT_FALSE(sim.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle.transactionActive());
  EXPECT_EQ(lifecycle.valueRevision(), lifecycle_revision_before);
  EXPECT_EQ(sim.fe_system->constraintLayoutRevision(),
            constraint_revision_before);
  EXPECT_EQ(sim.fe_system->sparsityPatternRevision(),
            sparsity_revision_before);
  EXPECT_EQ(sim.fe_system->constraints().numConstraints(),
            constraint_count_before);
  ASSERT_EQ(refresh_cache.last_signature.has_value(),
            refresh_cache_before.last_signature.has_value());
  if (refresh_cache.last_signature.has_value()) {
    EXPECT_TRUE(*refresh_cache.last_signature ==
                *refresh_cache_before.last_signature);
  }
  ASSERT_EQ(refresh_cache.last_vector_signature.has_value(),
            refresh_cache_before.last_vector_signature.has_value());
  if (refresh_cache.last_vector_signature.has_value()) {
    EXPECT_TRUE(*refresh_cache.last_vector_signature ==
                *refresh_cache_before.last_vector_signature);
  }

  const auto mesh_revisions_after = mesh->event_bus().revision_state();
  EXPECT_EQ(mesh_revisions_after.geometry, mesh_revisions_before.geometry);
  EXPECT_EQ(mesh_revisions_after.reference_geometry,
            mesh_revisions_before.reference_geometry);
  EXPECT_EQ(mesh_revisions_after.current_geometry,
            mesh_revisions_before.current_geometry);
  EXPECT_EQ(mesh_revisions_after.reference_rebase,
            mesh_revisions_before.reference_rebase);
  EXPECT_EQ(mesh_revisions_after.topology, mesh_revisions_before.topology);
  EXPECT_EQ(mesh_revisions_after.ownership, mesh_revisions_before.ownership);
  EXPECT_EQ(mesh_revisions_after.numbering, mesh_revisions_before.numbering);
  EXPECT_EQ(mesh_revisions_after.field_layout,
            mesh_revisions_before.field_layout);
  EXPECT_EQ(mesh_revisions_after.labels, mesh_revisions_before.labels);
  EXPECT_EQ(mesh_revisions_after.active_configuration,
            mesh_revisions_before.active_configuration);
  const auto* mesh_phi_data_after =
      static_cast<const double*>(mesh->field_data(mesh_phi_handle));
  ASSERT_NE(mesh_phi_data_after, nullptr);
  EXPECT_EQ(std::vector<double>(
                mesh_phi_data_after, mesh_phi_data_after + mesh_phi_count),
            mesh_phi_before);

  const auto cached_report =
      refreshActiveCutIntegrationContextFromSolutionCached(
          sim,
          *params,
          initial,
          lifecycle,
          refresh_cache,
          "application-driver-maintenance-transaction-restored");
  EXPECT_FALSE(cached_report.refreshed);
  EXPECT_EQ(
      cached_report.topology_key,
      initial_refresh_report.topology_key);
  EXPECT_EQ(sim.fe_system->cutIntegrationContext(), original_context);
  EXPECT_EQ(lifecycle.valueRevision(), lifecycle_revision_before);

  std::vector<svmp::FE::Real> committed_candidate;
  bool committed_candidate_refreshed = false;
  bool forced_certificate_refresh_rebuilt_same_topology = false;
  const LevelSetMaintenanceCandidateValidator accept_candidate =
      [&](std::span<const svmp::FE::Real> candidate,
          std::span<const LevelSetVolumeCorrectionMaintenanceEvent> events) {
        if (events.size() != 1u) {
          throw std::runtime_error(
              "accepted maintenance candidate had incomplete event coverage");
        }
        committed_candidate.assign(candidate.begin(), candidate.end());
        geometry_transaction =
            std::make_unique<LevelSetMaintenanceGeometryTransaction>(
                sim, lifecycle, refresh_cache, active_requests);
        const auto report = geometry_transaction->refresh(*params, candidate);
        const auto forced_report =
            geometry_transaction->refresh(
                *params,
                candidate,
                /*force_rebuild=*/true);
        committed_candidate_refreshed =
            report.refreshed &&
            sim.fe_system->cutIntegrationContext() != original_context;
        forced_certificate_refresh_rebuilt_same_topology =
            forced_report.refreshed &&
            forced_report.topology_key == report.topology_key;
      };
  ASSERT_TRUE(applyLevelSetMaintenance(
      sim,
      history,
      requests,
      {},
      {},
      {},
      &published_events,
      accept_candidate));
  ASSERT_NE(geometry_transaction, nullptr);
  ASSERT_NO_THROW(geometry_transaction->commit());
  EXPECT_TRUE(committed_candidate_refreshed);
  EXPECT_TRUE(forced_certificate_refresh_rebuilt_same_topology);
  EXPECT_FALSE(sim.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle.transactionActive());
  EXPECT_NE(sim.fe_system->cutIntegrationContext(), original_context);
  ASSERT_EQ(published_events.size(), 1u);
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_TRUE(requests.front().volume_target_initialized);
  EXPECT_EQ(gatherFeOrderedSolution(history.u()), committed_candidate);
  EXPECT_EQ(gatherFeOrderedSolution(history.uPrev()), committed_candidate);
  EXPECT_EQ(gatherFeOrderedSolution(history.uPrev2()), committed_candidate);
  EXPECT_EQ(gatherFeOrderedSolution(history.uDot()), rates_before);
  EXPECT_EQ(history.uDot().valueRevision(), rate_revision_before);
  const auto committed_cached_report =
      refreshActiveCutIntegrationContextFromSolutionCached(
          sim,
          *params,
          committed_candidate,
          lifecycle,
          refresh_cache,
          "application-driver-maintenance-transaction-committed");
  EXPECT_FALSE(committed_cached_report.refreshed);
  EXPECT_NE(committed_cached_report.topology_key, 0u);
#endif
}

TEST(ApplicationDriverLevelSetVolumeCorrection,
     CumulativeDisplacementBudgetRejectsBeforeAccountingExcessEvent)
{
  LevelSetMaintenanceRequest request{};
  request.level_set_field_name = "phi";
  request.volume_correction
      .maximum_cumulative_interface_displacement_fraction = 0.10;

  svmp::FE::level_set::LevelSetGlobalShiftCorrectionResult result{};
  result.correction_applied = true;
  result.minimum_edge_length = 1.0;
  result.max_interface_displacement = 0.04;
  result.max_contact_line_displacement = 0.03;
  addValidComponentTransferLedger(result);

  ASSERT_NO_THROW(accountAppliedLevelSetVolumeCorrection(request, result));
  ASSERT_NO_THROW(accountAppliedLevelSetVolumeCorrection(request, result));
  EXPECT_DOUBLE_EQ(
      request.cumulative_volume_correction_interface_displacement, 0.08);
  EXPECT_DOUBLE_EQ(
      request.cumulative_volume_correction_contact_line_displacement, 0.06);
  EXPECT_DOUBLE_EQ(
      request.volume_correction_reference_minimum_edge_length, 1.0);

  result.max_interface_displacement = 0.03;
  result.max_contact_line_displacement = 0.02;
  EXPECT_THROW(
      accountAppliedLevelSetVolumeCorrection(request, result),
      std::runtime_error);
  EXPECT_DOUBLE_EQ(
      request.cumulative_volume_correction_interface_displacement, 0.08);
  EXPECT_DOUBLE_EQ(
      request.cumulative_volume_correction_contact_line_displacement, 0.06);
}

TEST(ApplicationDriverLevelSetVolumeCorrection,
     CumulativeBudgetUsesSmallestObservedEdgeAndIgnoresSkippedEvents)
{
  LevelSetMaintenanceRequest request{};
  request.level_set_field_name = "phi";
  request.volume_correction
      .maximum_cumulative_interface_displacement_fraction = 0.10;

  svmp::FE::level_set::LevelSetGlobalShiftCorrectionResult skipped{};
  skipped.correction_applied = false;
  accountAppliedLevelSetVolumeCorrection(request, skipped);
  EXPECT_DOUBLE_EQ(
      request.cumulative_volume_correction_interface_displacement, 0.0);

  auto applied = skipped;
  applied.correction_applied = true;
  applied.minimum_edge_length = 1.0;
  applied.max_interface_displacement = 0.04;
  applied.max_contact_line_displacement = 0.02;
  addValidComponentTransferLedger(applied);
  ASSERT_NO_THROW(accountAppliedLevelSetVolumeCorrection(request, applied));

  applied.minimum_edge_length = 0.5;
  applied.max_interface_displacement = 0.02;
  applied.max_contact_line_displacement = 0.01;
  EXPECT_THROW(
      accountAppliedLevelSetVolumeCorrection(request, applied),
      std::runtime_error);
  EXPECT_DOUBLE_EQ(
      request.cumulative_volume_correction_interface_displacement, 0.04);
  EXPECT_DOUBLE_EQ(
      request.volume_correction_reference_minimum_edge_length, 1.0);
}

TEST(ApplicationDriverLevelSetVolumeCorrection,
     CumulativeContactLineBudgetRejectsBeforeMutatingAccountingState)
{
  LevelSetMaintenanceRequest request{};
  request.level_set_field_name = "phi";
  request.volume_correction
      .maximum_cumulative_interface_displacement_fraction = 0.10;
  request.volume_correction_reference_minimum_edge_length = 1.0;
  request.cumulative_volume_correction_interface_displacement = 0.02;
  request.cumulative_volume_correction_contact_line_displacement = 0.08;
  request.volume_target_initialized = true;
  request.volume_target = 0.375;

  svmp::FE::level_set::LevelSetGlobalShiftCorrectionResult result{};
  result.correction_applied = true;
  result.minimum_edge_length = 1.0;
  result.max_interface_displacement = 0.01;
  result.max_contact_line_displacement = 0.03;
  addValidComponentTransferLedger(result);

  const auto reference_edge_before =
      request.volume_correction_reference_minimum_edge_length;
  const auto interface_history_before =
      request.cumulative_volume_correction_interface_displacement;
  const auto contact_line_history_before =
      request.cumulative_volume_correction_contact_line_displacement;
  const auto target_initialized_before = request.volume_target_initialized;
  const auto target_before = request.volume_target;

  try {
    accountAppliedLevelSetVolumeCorrection(request, result);
    FAIL() << "Expected the contact-line cumulative path to exceed the budget";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("limiting_path=contact_line"), std::string::npos);
    EXPECT_NE(message.find("prospective_interface="), std::string::npos);
    EXPECT_NE(message.find("prospective_contact_line="), std::string::npos);
  }

  EXPECT_DOUBLE_EQ(
      request.volume_correction_reference_minimum_edge_length,
      reference_edge_before);
  EXPECT_DOUBLE_EQ(
      request.cumulative_volume_correction_interface_displacement,
      interface_history_before);
  EXPECT_DOUBLE_EQ(
      request.cumulative_volume_correction_contact_line_displacement,
      contact_line_history_before);
  EXPECT_EQ(request.volume_target_initialized, target_initialized_before);
  EXPECT_DOUBLE_EQ(request.volume_target, target_before);
}

TEST(ApplicationDriverLevelSetVolumeCorrection,
     LaterContactLineOnlyBudgetRejectionRollsBackEarlierRequestAndHistory)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowTriangleMesh();
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Triangle3,
      /*order=*/1);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1,
  });
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_vertex_values(mesh->n_vertices(), 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    phi_vertex_values[vertex] =
        workflowVertexPoint(*mesh, vertex)[0] - svmp::FE::Real{0.5};
  }
  const auto coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      phi_vertex_values,
      /*components=*/1u,
      "ApplicationDriver cumulative contact-line budget phi");
  std::vector<svmp::FE::Real> initial(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, coefficients, initial);

  std::unique_ptr<svmp::FE::backends::BackendFactory> factory;
  try {
    factory = svmp::FE::backends::BackendFactory::create(
        svmp::FE::backends::BackendKind::FSILS);
  } catch (const std::exception&) {
    GTEST_SKIP() << "Requires an available FE vector backend.";
  }
  auto history = svmp::FE::timestepping::TimeHistory::allocate(
      *factory,
      system->dofHandler().getNumDofs(),
      /*history_depth=*/2,
      /*allocate_second_order_state=*/true);
  history.setStepIndex(1);
  history.setDt(0.1);
  history.setPrevDt(0.1);
  scatterFeOrderedSolution(history.u(), initial);
  scatterFeOrderedSolution(history.uPrev(), initial);
  scatterFeOrderedSolution(history.uPrev2(), initial);
  std::vector<svmp::FE::Real> rates(initial.size(), 0.375);
  scatterFeOrderedSolution(history.uDot(), rates);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);

  LevelSetMaintenanceRequest first_request{};
  first_request.level_set_field_name = "phi";
  first_request.volume_correction.enabled = true;
  first_request.volume_correction.cadence_steps = 1;
  first_request.volume_correction.use_initial_negative_volume_as_target = false;
  first_request.volume_correction.target_negative_volume = 0.36;
  first_request.volume_correction.minimum_relative_volume_error = 0.0;
  first_request.volume_correction.maximum_interface_displacement_fraction =
      0.10;
  first_request.volume_correction
      .maximum_cumulative_interface_displacement_fraction = 1.0;
  first_request.volume_target_initialized = true;
  first_request.volume_target = 0.36;

  std::vector<LevelSetMaintenanceRequest> successful_requests{
      first_request};
  std::vector<LevelSetVolumeCorrectionMaintenanceEvent> successful_events;
  testing::internal::CaptureStdout();
  const bool successful_change = applyLevelSetMaintenance(
      sim,
      history,
      successful_requests,
      {},
      {},
      {},
      &successful_events);
  const auto successful_output = testing::internal::GetCapturedStdout();
  ASSERT_TRUE(successful_change);
  ASSERT_EQ(successful_events.size(), 1u);
  EXPECT_EQ(successful_events.front().level_set_field, phi);
  EXPECT_EQ(successful_events.front().level_set_field_name, "phi");
  EXPECT_TRUE(successful_events.front().correction.correction_applied);
  EXPECT_NE(successful_output.find(
                "max_contact_angle_change_radians=0"),
            std::string::npos);
  EXPECT_NE(successful_output.find(
                "negative_component_topology_preserved=true"),
            std::string::npos);
  EXPECT_NE(successful_output.find("negative_component_count=1"),
            std::string::npos);
  EXPECT_NE(successful_output.find("component_global_vertex_id="),
            std::string::npos);
  EXPECT_NE(successful_output.find("component_volume_transfer="),
            std::string::npos);
  scatterFeOrderedSolution(history.u(), initial);
  scatterFeOrderedSolution(history.uPrev(), initial);
  scatterFeOrderedSolution(history.uPrev2(), initial);
  scatterFeOrderedSolution(history.uDot(), rates);

  auto rejecting_request = first_request;
  rejecting_request.volume_correction.target_negative_volume = 0.35;
  rejecting_request.volume_correction
      .maximum_cumulative_interface_displacement_fraction = 0.10;
  rejecting_request.volume_target = 0.35;
  rejecting_request.volume_correction_reference_minimum_edge_length = 1.0;
  rejecting_request.cumulative_volume_correction_interface_displacement = 0.0;
  rejecting_request.cumulative_volume_correction_contact_line_displacement =
      0.095;
  std::vector<LevelSetMaintenanceRequest> requests{
      first_request,
      rejecting_request};

  const auto current_before = gatherFeOrderedSolution(history.u());
  const auto previous_before = gatherFeOrderedSolution(history.uPrev());
  const auto previous2_before = gatherFeOrderedSolution(history.uPrev2());
  const auto rates_before = gatherFeOrderedSolution(history.uDot());
  auto rejected_events = successful_events;

  testing::internal::CaptureStdout();
  std::string rejection_message;
  try {
    (void)applyLevelSetMaintenance(
        sim,
        history,
        requests,
        {},
        {},
        {},
        &rejected_events);
  } catch (const std::runtime_error& error) {
    rejection_message = error.what();
  }
  const auto output = testing::internal::GetCapturedStdout();
  EXPECT_NE(rejection_message.find("limiting_path=contact_line"),
            std::string::npos);
  EXPECT_EQ(output.find("Level-set volume corrected"), std::string::npos);
  EXPECT_EQ(output.find("Level-set maintenance synchronized"),
            std::string::npos);
  EXPECT_TRUE(rejected_events.empty());

  EXPECT_EQ(gatherFeOrderedSolution(history.u()), current_before);
  EXPECT_EQ(gatherFeOrderedSolution(history.uPrev()), previous_before);
  EXPECT_EQ(gatherFeOrderedSolution(history.uPrev2()), previous2_before);
  EXPECT_EQ(gatherFeOrderedSolution(history.uDot()), rates_before);
  ASSERT_EQ(requests.size(), 2u);
  EXPECT_DOUBLE_EQ(
      requests[0].volume_correction_reference_minimum_edge_length,
      first_request.volume_correction_reference_minimum_edge_length);
  EXPECT_DOUBLE_EQ(
      requests[0].cumulative_volume_correction_interface_displacement,
      first_request.cumulative_volume_correction_interface_displacement);
  EXPECT_DOUBLE_EQ(
      requests[0].cumulative_volume_correction_contact_line_displacement,
      first_request.cumulative_volume_correction_contact_line_displacement);
  EXPECT_DOUBLE_EQ(
      requests[1].volume_correction_reference_minimum_edge_length,
      rejecting_request.volume_correction_reference_minimum_edge_length);
  EXPECT_DOUBLE_EQ(
      requests[1].cumulative_volume_correction_interface_displacement,
      rejecting_request.cumulative_volume_correction_interface_displacement);
  EXPECT_DOUBLE_EQ(
      requests[1].cumulative_volume_correction_contact_line_displacement,
      rejecting_request.cumulative_volume_correction_contact_line_displacement);
#endif
}

} // namespace
#else
TEST(ApplicationDriverBoundPreservingCandidates,
     RequiresMeshIntegration)
{
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
}
#endif
