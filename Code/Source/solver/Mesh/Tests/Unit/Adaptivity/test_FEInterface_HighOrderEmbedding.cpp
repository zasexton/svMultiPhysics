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

#include <gtest/gtest.h>

#include "../../../Adaptivity/AdaptivityManager.h"
#include "../../../Adaptivity/FEInterface.h"
#include "../../../Adaptivity/HighOrderEmbedding.h"
#include "../../../Core/MeshBase.h"

#include <memory>
#include <vector>

using namespace svmp;

namespace {

class DummyFE : public AdaptivityFEInterface {
public:
  mutable int embedding_calls = 0;
  mutable int adapted_calls = 0;

  bool get_high_order_embedding(const HighOrderEmbeddingKey& key, HighOrderEmbedding& out) const override {
    embedding_calls++;
    if (key.parent_family != CellFamily::Line) return false;
    if (key.parent_order != 3) return false;
    if (key.parent_num_nodes != 4) return false;

    out.children.clear();
    out.children.resize(2);

    // Child 0 (left half): [0, 1/2, 1/6, 1/3] in the standard "endpoints then interior" ordering.
    {
      HighOrderChildEmbedding child;
      child.child_family = CellFamily::Line;
      child.child_order = 3;
      child.child_num_nodes = 4;
      child.child_node_parent_weights.row_offsets = {0, 1, 3, 5, 6};
      child.child_node_parent_weights.col_indices = {0, 0, 1, 0, 1, 2};
      child.child_node_parent_weights.values = {
          1.0,
          0.5, 0.5,
          5.0 / 6.0, 1.0 / 6.0,
          1.0,
      };
      out.children[0] = std::move(child);
    }

    // Child 1 (right half): [1/2, 1, 2/3, 5/6] in the standard ordering.
    {
      HighOrderChildEmbedding child;
      child.child_family = CellFamily::Line;
      child.child_order = 3;
      child.child_num_nodes = 4;
      child.child_node_parent_weights.row_offsets = {0, 2, 3, 4, 6};
      child.child_node_parent_weights.col_indices = {0, 1, 1, 3, 0, 1};
      child.child_node_parent_weights.values = {
          0.5, 0.5,
          1.0,
          1.0,
          1.0 / 6.0, 5.0 / 6.0,
      };
      out.children[1] = std::move(child);
    }

    return true;
  }

  void on_mesh_adapted(const MeshBase&, const MeshBase&, const RefinementDelta&, const AdaptivityOptions&) override {
    adapted_calls++;
  }
};

MeshBase make_order3_line_mesh() {
  MeshBase mesh(1);

  // Node ordering: endpoints first, then interior nodes (p=3 has 2 interior nodes).
  // Vertex 0: x=0
  // Vertex 1: x=1
  // Vertex 2: x=1/3
  // Vertex 3: x=2/3
  std::vector<real_t> X = {
      0.0,
      1.0,
      static_cast<real_t>(1.0 / 3.0),
      static_cast<real_t>(2.0 / 3.0),
  };

  std::vector<offset_t> offsets = {0, 4};
  std::vector<index_t> cell2v = {0, 1, 2, 3};
  std::vector<CellShape> shapes(1);
  shapes[0].family = CellFamily::Line;
  shapes[0].order = 3;
  shapes[0].num_corners = 2;

  mesh.build_from_arrays(1, X, offsets, cell2v, shapes);
  mesh.finalize();
  return mesh;
}

} // namespace

TEST(HighOrderEmbeddingCacheTest, RequestsOnceAndCaches) {
  HighOrderEmbeddingCache cache;
  DummyFE fe;

  HighOrderEmbeddingKey key;
  key.parent_family = CellFamily::Line;
  key.parent_order = 3;
  key.parent_num_nodes = 4;
  key.spec = RefinementSpec{RefinementPattern::RED, 0};

  (void)cache.get_or_request(key, &fe);
  (void)cache.get_or_request(key, &fe);
  EXPECT_EQ(fe.embedding_calls, 1);
}

TEST(AdaptivityFEInterfaceTest, RefinesOrder3LineViaFEEmbedding) {
  auto fe = std::make_shared<DummyFE>();

  AdaptivityOptions options;
  options.enable_refinement = true;
  options.enable_coarsening = false;
  options.max_refinement_level = 1;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
  options.conformity_mode = AdaptivityOptions::ConformityMode::ALLOW_HANGING_NODES;
  options.check_quality = false;
  options.enforce_quality_after_refinement = false;
  options.verbosity = 0;

  AdaptivityManager mgr(options);
  mgr.set_fe_interface(fe);

  auto mesh = make_order3_line_mesh();
  std::vector<bool> marks(mesh.n_cells(), true);

  auto result = mgr.refine(mesh, marks, nullptr);
  ASSERT_TRUE(result.success);
  ASSERT_NE(result.refinement_delta, nullptr);

  EXPECT_EQ(fe->embedding_calls, 1);
  EXPECT_EQ(fe->adapted_calls, 1);

  // One line refined -> two children.
  EXPECT_EQ(mesh.n_cells(), 2u);

  // Parent has 4 nodes; refinement introduces 3 new nodes (1/2, 1/6, 5/6).
  EXPECT_EQ(mesh.n_vertices(), 7u);

  // Parent GIDs are preserved; new vertex GIDs are appended.
  EXPECT_NE(mesh.global_to_local_vertex(0), INVALID_INDEX);
  EXPECT_NE(mesh.global_to_local_vertex(1), INVALID_INDEX);
  EXPECT_NE(mesh.global_to_local_vertex(2), INVALID_INDEX);
  EXPECT_NE(mesh.global_to_local_vertex(3), INVALID_INDEX);
  EXPECT_NE(mesh.global_to_local_vertex(4), INVALID_INDEX);
  EXPECT_NE(mesh.global_to_local_vertex(5), INVALID_INDEX);
  EXPECT_NE(mesh.global_to_local_vertex(6), INVALID_INDEX);

  EXPECT_EQ(result.refinement_delta->refined_cells.size(), 1u);
  EXPECT_EQ(result.refinement_delta->refined_cells[0].parent_cell_gid, 0);
  EXPECT_EQ(result.refinement_delta->refined_cells[0].child_cell_gids.size(), 2u);
  EXPECT_EQ(result.refinement_delta->new_vertices.size(), 3u);
}

