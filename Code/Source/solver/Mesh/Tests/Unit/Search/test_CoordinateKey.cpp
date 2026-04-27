#include "gtest/gtest.h"

#include "Core/MeshBase.h"
#include "Search/CoordinateKey.h"

#include <cmath>
#include <vector>

namespace {

svmp::MeshBase make_tetra_mesh()
{
  svmp::MeshBase mesh;
  std::vector<svmp::real_t> x = {
      0.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0,
  };
  std::vector<svmp::offset_t> offsets = {0, 4};
  std::vector<svmp::index_t> conn = {0, 1, 2, 3};

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Tetra;
  shape.num_corners = 4;
  shape.order = 1;
  mesh.build_from_arrays(3, x, offsets, conn, {shape});
  return mesh;
}

} // namespace

TEST(CoordinateKey, CanonicalizesSignedZero)
{
  const std::vector<svmp::real_t> positive_zero = {0.0, 1.0, 2.0};
  const std::vector<svmp::real_t> negative_zero = {-0.0, 1.0, 2.0};

  const auto a = svmp::search::make_coordinate_key(positive_zero, 3, 0);
  const auto b = svmp::search::make_coordinate_key(negative_zero, 3, 0);

  EXPECT_EQ(a, b);
  EXPECT_EQ(svmp::search::CoordinateKeyHash{}(a),
            svmp::search::CoordinateKeyHash{}(b));
}

TEST(CoordinateKey, DimensionParticipatesInKey)
{
  const std::vector<svmp::real_t> coords = {1.0, 2.0, 3.0};

  const auto two_d = svmp::search::make_coordinate_key(coords, 2, 0);
  const auto three_d = svmp::search::make_coordinate_key(coords, 3, 0);

  EXPECT_NE(two_d, three_d);
}

TEST(VertexCoordinateLocator, FindsVerticesFromExternalCoordinateArray)
{
  const auto mesh = make_tetra_mesh();
  const svmp::search::VertexCoordinateLocator locator(mesh);

  const std::vector<svmp::real_t> face_coords = {
      0.0, 1.0, 0.0,
      1.0, 0.0, 0.0,
  };

  EXPECT_EQ(locator.find(face_coords, 3, 0), 2);
  EXPECT_EQ(locator.find(face_coords, 3, 1), 1);
}

TEST(VertexCoordinateLocator, ReturnsInvalidIndexForMissingCoordinate)
{
  const auto mesh = make_tetra_mesh();
  const svmp::search::VertexCoordinateLocator locator(mesh);

  const std::vector<svmp::real_t> face_coords = {0.5, 0.5, 0.5};

  EXPECT_EQ(locator.find(face_coords, 3, 0), svmp::INVALID_INDEX);
}
