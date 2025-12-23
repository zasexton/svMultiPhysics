/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file test_InterfaceMesh.cpp
 * @brief Unit tests for InterfaceMesh (trace mesh extraction + orientation + fields)
 */

#include "../../../Core/InterfaceMesh.h"
#include "../../../Core/MeshBase.h"
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <vector>

namespace svmp::test {

#define ASSERT(cond)                                                                               \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << " : " #cond "\n";     \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                                \
    }                                                                                              \
  } while (0)

#define ASSERT_NEAR(a, b, tol) ASSERT(std::abs((a) - (b)) <= (tol))

static MeshBase create_unit_hex() {
  MeshBase mesh;

  std::vector<real_t> coords = {
      0, 0, 0,  // 0
      1, 0, 0,  // 1
      1, 1, 0,  // 2
      0, 1, 0,  // 3
      0, 0, 1,  // 4
      1, 0, 1,  // 5
      1, 1, 1,  // 6
      0, 1, 1   // 7
  };

  std::vector<offset_t> offsets = {0, 8};
  std::vector<index_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<CellShape> shapes = {{CellFamily::Hex, 8, 1}};

  mesh.build_from_arrays(3, coords, offsets, conn, shapes);
  mesh.finalize();
  return mesh;
}

static void test_build_and_geometry() {
  MeshBase vol = create_unit_hex();

  const label_t wall = 7;
  for (index_t f : vol.boundary_faces()) {
    vol.set_boundary_label(f, wall);
  }

  InterfaceMesh iface = InterfaceMesh::build_from_boundary_label(vol, wall, true);

  ASSERT(iface.n_faces() == vol.boundary_faces().size());
  ASSERT(iface.n_faces() == 6);
  ASSERT(iface.n_vertices() == 8);
  ASSERT(iface.spatial_dim() == 3);

  for (index_t lf = 0; lf < static_cast<index_t>(iface.n_faces()); ++lf) {
    ASSERT(iface.volume_cell(lf) == 0);
    ASSERT(iface.local_face_in_cell(lf) >= 0);
    ASSERT(iface.face_orientation(lf) >= 0);

    const real_t area = iface.face_area(lf, vol);
    ASSERT_NEAR(area, 1.0, 1e-12);

    const auto c = iface.face_center(lf, vol);
    for (int d = 0; d < 3; ++d) {
      ASSERT(c[d] >= -1e-12);
      ASSERT(c[d] <= 1.0 + 1e-12);
    }

    const auto n = iface.face_normal(lf, vol);
    const real_t norm = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    ASSERT(norm > 0.0);
  }
}

static void test_interface_fields() {
  MeshBase vol = create_unit_hex();
  const label_t wall = 7;
  for (index_t f : vol.boundary_faces()) {
    vol.set_boundary_label(f, wall);
  }

  InterfaceMesh iface = InterfaceMesh::build_from_boundary_label(vol, wall, true);

  auto h = iface.attach_field("flux", FieldScalarType::Float64, 3);
  ASSERT(h.id != 0);
  ASSERT(iface.field_bytes_per_entity(h) == 3 * sizeof(real_t));

  auto* data = iface.field_data_as<real_t>(h);
  ASSERT(data != nullptr);
  data[0] = 123.0;
  ASSERT_NEAR(data[0], 123.0, 0.0);

  // Attaching again returns the same handle.
  auto h2 = iface.attach_field("flux", FieldScalarType::Float64, 3);
  ASSERT(h2.id == h.id);
}

} // namespace svmp::test

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  svmp::test::test_build_and_geometry();
  svmp::test::test_interface_fields();

  if (rank == 0) {
    std::cout << "InterfaceMesh tests PASSED\n";
  }

  MPI_Finalize();
  return 0;
}
