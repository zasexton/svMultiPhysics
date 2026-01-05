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
 * @file test_DistributedSemanticsSerial.cpp
 * @brief Serial (non-mpiexec) test validating owned/shared/ghost helpers are consistent.
 */

#include "../../../Mesh.h"
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

namespace svmp::test {

#define ASSERT(cond)                                                                               \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << " : " #cond "\n";     \
      std::abort();                                                                                \
    }                                                                                              \
  } while (0)

static std::shared_ptr<MeshBase> create_single_hex() {
  auto mesh = std::make_shared<MeshBase>();

  std::vector<real_t> coords = {
      0.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      1.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0,
      1.0, 0.0, 1.0,
      1.0, 1.0, 1.0,
      0.0, 1.0, 1.0,
  };

  std::vector<offset_t> offsets = {0, 8};
  std::vector<index_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<CellShape> shapes = {{CellFamily::Hex, 8, 1}};

  mesh->build_from_arrays(3, coords, offsets, conn, shapes);
  mesh->finalize();
  return mesh;
}

static void test_serial_semantics() {
  auto base = create_single_hex();
  Mesh mesh(base, MeshComm::world());

  ASSERT(mesh.rank() == 0);
  ASSERT(mesh.world_size() == 1);

  ASSERT(mesh.n_cells() == 1);
  ASSERT(mesh.n_vertices() == 8);
  ASSERT(mesh.n_faces() == 6);

  ASSERT(mesh.n_owned_cells() == 1);
  ASSERT(mesh.n_shared_cells() == 0);
  ASSERT(mesh.n_ghost_cells() == 0);

  ASSERT(mesh.n_owned_vertices() == 8);
  ASSERT(mesh.n_shared_vertices() == 0);
  ASSERT(mesh.n_ghost_vertices() == 0);

  ASSERT(mesh.n_owned_faces() == 6);
  ASSERT(mesh.n_shared_faces() == 0);
  ASSERT(mesh.n_ghost_faces() == 0);

  ASSERT(mesh.n_owned_vertices() + mesh.n_shared_vertices() + mesh.n_ghost_vertices() == mesh.n_vertices());
  ASSERT(mesh.n_owned_cells() + mesh.n_shared_cells() + mesh.n_ghost_cells() == mesh.n_cells());
  ASSERT(mesh.n_owned_faces() + mesh.n_shared_faces() + mesh.n_ghost_faces() == mesh.n_faces());

  ASSERT(mesh.owned_vertices().size() == mesh.n_owned_vertices());
  ASSERT(mesh.shared_vertices().size() == mesh.n_shared_vertices());
  ASSERT(mesh.ghost_vertices().size() == mesh.n_ghost_vertices());

  ASSERT(mesh.owned_cells().size() == mesh.n_owned_cells());
  ASSERT(mesh.shared_cells().size() == mesh.n_shared_cells());
  ASSERT(mesh.ghost_cells().size() == mesh.n_ghost_cells());

  ASSERT(mesh.owned_faces().size() == mesh.n_owned_faces());
  ASSERT(mesh.shared_faces().size() == mesh.n_shared_faces());
  ASSERT(mesh.ghost_faces().size() == mesh.n_ghost_faces());

  ASSERT(mesh.global_n_cells() == 1);
  ASSERT(mesh.global_n_vertices() == 8);
  ASSERT(mesh.global_n_faces() == 6);
}

} // namespace svmp::test

int main() {
  svmp::test::test_serial_semantics();
  std::cout << "DistributedMesh serial semantics tests PASSED\n";
  return 0;
}

