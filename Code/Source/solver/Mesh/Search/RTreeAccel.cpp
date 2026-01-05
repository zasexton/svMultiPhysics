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

#include "RTreeAccel.h"
#include "SearchBuilders.h"
#include "../Core/MeshBase.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stack>

namespace svmp {
using namespace search;

// ---- Building ----

void RTreeAccel::build(const MeshBase& mesh,
                       Configuration cfg,
                       const MeshSearch::SearchConfig& config) {
  clear();

  built_cfg_ = cfg;
  stats_ = SearchStats();
  auto start_time = std::chrono::steady_clock::now();

  // Extract mesh data based on configuration
  if (cfg == Configuration::Reference) {
    vertex_coords_ = SearchBuilders::extract_vertex_coords(mesh);
  } else {
    vertex_coords_ = SearchBuilders::extract_deformed_coords(mesh);
  }

  // Build cell data
  cell_indices_.resize(mesh.n_cells());
  std::iota(cell_indices_.begin(), cell_indices_.end(), 0);
  cell_aabbs_ = SearchBuilders::compute_cell_aabbs(mesh, vertex_coords_);

  // Prepare entries for bulk loading
  std::vector<std::pair<index_t, search::AABB>> cell_entries;
  cell_entries.reserve(cell_indices_.size());
  for (size_t i = 0; i < cell_indices_.size(); ++i) {
    cell_entries.emplace_back(cell_indices_[i], cell_aabbs_[i]);
  }

  // Build main R-Tree for cells using STR bulk loading
  if (!cell_entries.empty()) {
    root_ = str_build(cell_entries, 0, static_cast<int>(cell_entries.size()), 0);
    tree_height_ = get_tree_height(root_.get());
  }

  // Build separate R-Tree for vertices (for nearest neighbor queries)
  if (config.primary_use == MeshSearch::QueryType::NearestNeighbor) {
    vertex_indices_.resize(vertex_coords_.size());
    std::iota(vertex_indices_.begin(), vertex_indices_.end(), 0);

    std::vector<std::pair<index_t, search::AABB>> vertex_entries;
    vertex_entries.reserve(vertex_indices_.size());

    for (size_t i = 0; i < vertex_indices_.size(); ++i) {
      const auto& v = vertex_coords_[i];
      search::AABB point_bounds(v, v);
      // Expand slightly to avoid degenerate AABBs
      for (int j = 0; j < 3; ++j) {
        point_bounds.min[j] -= 1e-10;
        point_bounds.max[j] += 1e-10;
      }
      vertex_entries.emplace_back(vertex_indices_[i], point_bounds);
    }

    if (!vertex_entries.empty()) {
      vertex_root_ = str_build(vertex_entries, 0, static_cast<int>(vertex_entries.size()), 0);
    }
  }

  // Extract boundary triangles for ray intersection
  if (config.primary_use == MeshSearch::QueryType::RayIntersection) {
    auto tris = SearchBuilders::extract_boundary_triangles(mesh, vertex_coords_);
    boundary_triangles_.clear();
    boundary_triangles_.reserve(tris.size());
    for (const auto& t : tris) {
      BoundaryTriangle bt;
      bt.vertices = t.vertices;
      // Compute flat normal
      auto e1 = search::subtract(bt.vertices[1], bt.vertices[0]);
      auto e2 = search::subtract(bt.vertices[2], bt.vertices[0]);
      bt.normal = search::cross(e1, e2);
      bt.face_id = t.face_id;
      boundary_triangles_.push_back(bt);
    }
  }

  // Compute statistics
  if (root_) {
    stats_.n_nodes = count_nodes(root_.get());
    if (vertex_root_) {
      stats_.n_nodes += count_nodes(vertex_root_.get());
    }
    stats_.tree_depth = tree_height_;
    stats_.memory_bytes = compute_memory_usage();
  }

  auto end_time = std::chrono::steady_clock::now();
  stats_.build_time_ms = std::chrono::duration<double, std::milli>(
      end_time - start_time).count();

  is_built_ = true;
}

void RTreeAccel::clear() {
  root_.reset();
  vertex_root_.reset();
  vertex_coords_.clear();
  cell_aabbs_.clear();
  cell_indices_.clear();
  vertex_indices_.clear();
  boundary_triangles_.clear();
  tree_height_ = 0;
  is_built_ = false;
  stats_ = SearchStats();
}

// ---- Dynamic insertion ----

void RTreeAccel::insert(index_t entity_id, const search::AABB& bounds) {
  if (!root_) {
    // Create new root
    root_ = std::make_unique<RTreeNode>(true, 0);
    root_->entries.emplace_back(entity_id, bounds);
    root_->update_mbr();
    tree_height_ = 1;
  } else {
    // Find leaf and insert
    RTreeNode* leaf = choose_leaf(root_.get(), bounds, 0);
    insert_entry(leaf, entity_id, bounds);

    // Adjust tree
    adjust_tree(leaf);

    // Check if root needs to be split
    if (root_->is_full()) {
      auto split_result = rstar_split(root_.get());
      auto new_root = std::make_unique<RTreeNode>(false, root_->level + 1);
      new_root->children.push_back(std::move(split_result.first));
      new_root->children.push_back(std::move(split_result.second));
      new_root->children[0]->parent = new_root.get();
      new_root->children[1]->parent = new_root.get();
      new_root->update_mbr();
      root_ = std::move(new_root);
      tree_height_++;
    }
  }

  is_built_ = true;
  tree_height_ = root_ ? get_tree_height(root_.get()) : 0;
  stats_.n_nodes = count_nodes(root_.get());
  stats_.tree_depth = tree_height_;
  stats_.memory_bytes = compute_memory_usage();
}

void RTreeAccel::remove(index_t entity_id, const search::AABB& bounds) {
  if (!root_) return;

  // Find and remove entry
  RTreeNode* leaf = find_leaf(root_.get(), entity_id, bounds);
  if (!leaf) return;

  // Remove entry from leaf
  auto it = std::find_if(leaf->entries.begin(), leaf->entries.end(),
                         [entity_id](const RTreeNode::Entry& e) {
                           return e.id == entity_id;
                         });

  if (it != leaf->entries.end()) {
    leaf->entries.erase(it);
    leaf->update_mbr();

    // Condense tree
    condense_tree(leaf);

    // Shorten tree if root has only one child
    if (!root_->is_leaf && root_->children.size() == 1) {
      root_ = std::move(root_->children[0]);
      root_->parent = nullptr;
      tree_height_--;
    }
  }

  tree_height_ = root_ ? get_tree_height(root_.get()) : 0;
  stats_.n_nodes = count_nodes(root_.get());
  stats_.tree_depth = tree_height_;
  stats_.memory_bytes = compute_memory_usage();
}

// ---- Bulk loading (STR) ----

void RTreeAccel::bulk_load(const std::vector<std::pair<index_t, search::AABB>>& entries) {
  if (entries.empty()) return;

  clear();

  auto mutable_entries = entries;  // Copy for sorting
  root_ = str_build(mutable_entries, 0, static_cast<int>(entries.size()), 0);
  tree_height_ = get_tree_height(root_.get());
  is_built_ = true;

  stats_.n_nodes = count_nodes(root_.get());
  stats_.tree_depth = tree_height_;
  stats_.memory_bytes = compute_memory_usage();
}

std::unique_ptr<RTreeAccel::RTreeNode> RTreeAccel::str_build(
    std::vector<std::pair<index_t, search::AABB>>& entries,
    int start, int end, int level) {

  int n = end - start;
  if (n <= 0) return nullptr;

  // Create leaf if few enough entries
  if (n <= RTreeNode::MAX_ENTRIES) {
    auto leaf = std::make_unique<RTreeNode>(true, 0);
    for (int i = start; i < end; ++i) {
      leaf->entries.emplace_back(entries[i].first, entries[i].second);
    }
    leaf->update_mbr();
    return leaf;
  }

  // Calculate grid dimensions
  int node_capacity = RTreeNode::MAX_ENTRIES;
  int n_nodes = (n + node_capacity - 1) / node_capacity;
  int n_slices = static_cast<int>(std::ceil(std::pow(n_nodes, 1.0/3.0)));

  // Sort by X coordinate
  std::sort(entries.begin() + start, entries.begin() + end,
           [](const auto& a, const auto& b) {
             return a.second.center()[0] < b.second.center()[0];
           });

  // Create node for this level
  auto node = std::make_unique<RTreeNode>(false, level);

  // Divide into X slices
  int slice_size = (n + n_slices - 1) / n_slices;
  for (int x_slice = 0; x_slice < n_slices && start < end; ++x_slice) {
    int slice_start = start;
    int slice_end = std::min(start + slice_size, end);

    // Sort this X-slice by Y coordinate
    std::sort(entries.begin() + slice_start, entries.begin() + slice_end,
             [](const auto& a, const auto& b) {
               return a.second.center()[1] < b.second.center()[1];
             });

    // Divide into Y strips
    int strip_size = (slice_end - slice_start + n_slices - 1) / n_slices;
    for (int y_strip = 0; y_strip < n_slices && slice_start < slice_end; ++y_strip) {
      int strip_start = slice_start;
      int strip_end = std::min(slice_start + strip_size, slice_end);

      // Sort this Y-strip by Z coordinate
      std::sort(entries.begin() + strip_start, entries.begin() + strip_end,
               [](const auto& a, const auto& b) {
                 return a.second.center()[2] < b.second.center()[2];
               });

      // Create nodes for Z-sorted groups
      int group_size = node_capacity;
      for (int i = strip_start; i < strip_end; i += group_size) {
        int group_end = std::min(i + group_size, strip_end);

        if (level == 0) {
          // Create leaf node
          auto leaf = std::make_unique<RTreeNode>(true, 0);
          for (int j = i; j < group_end; ++j) {
            leaf->entries.emplace_back(entries[j].first, entries[j].second);
          }
          leaf->update_mbr();
          leaf->parent = node.get();
          node->children.push_back(std::move(leaf));
        } else {
          // Recursively build subtree
          auto child = str_build(entries, i, group_end, level - 1);
          if (child) {
            child->parent = node.get();
            node->children.push_back(std::move(child));
          }
        }
      }

      slice_start = strip_end;
    }

    start = slice_end;
  }

  node->update_mbr();
  return node;
}

// ---- Tree operations ----

RTreeAccel::RTreeNode* RTreeAccel::choose_leaf(RTreeNode* node,
                                               const search::AABB& bounds,
                                               int target_level) {
  if (node->level == target_level) {
    return node;
  }

  // Choose child with minimum area enlargement
  real_t min_enlargement = std::numeric_limits<real_t>::infinity();
  real_t min_area = std::numeric_limits<real_t>::infinity();
  RTreeNode* best_child = nullptr;

  for (auto& child : node->children) {
    real_t enlargement = compute_enlargement(child->mbr, bounds);
    auto expanded = child->mbr;
    expanded.expand(bounds);
    real_t area = expanded.volume();

    if (enlargement < min_enlargement ||
        (enlargement == min_enlargement && area < min_area)) {
      min_enlargement = enlargement;
      min_area = area;
      best_child = child.get();
    }
  }

  return choose_leaf(best_child, bounds, target_level);
}

void RTreeAccel::insert_entry(RTreeNode* leaf, index_t id, const search::AABB& bounds) {
  leaf->entries.emplace_back(id, bounds);
  leaf->update_mbr();
}

void RTreeAccel::insert_child(RTreeNode* node, std::unique_ptr<RTreeNode> child) {
  child->parent = node;
  node->children.push_back(std::move(child));
  node->update_mbr();
}

void RTreeAccel::adjust_tree(RTreeNode* node, std::unique_ptr<RTreeNode> split_node) {
  while (node != root_.get()) {
    RTreeNode* parent = node->parent;
    parent->update_mbr();

    if (split_node) {
      insert_child(parent, std::move(split_node));
      if (parent->is_full()) {
        auto split_result = rstar_split(parent);
        split_node = std::move(split_result.second);
        // Update parent with first half
        *parent = std::move(*split_result.first);
      } else {
        split_node = nullptr;
      }
    }

    node = parent;
  }

  // Handle root split
  if (split_node) {
    auto new_root = std::make_unique<RTreeNode>(false, root_->level + 1);
    root_->parent = new_root.get();
    split_node->parent = new_root.get();
    new_root->children.push_back(std::move(root_));
    new_root->children.push_back(std::move(split_node));
    new_root->update_mbr();
    root_ = std::move(new_root);
    tree_height_++;
  }
}

// ---- Node splitting ----

RTreeAccel::SplitResult RTreeAccel::rstar_split(RTreeNode* node) {
  // For R*-tree, we try different axis and choose the best
  // For simplicity, using quadratic split here
  return quadratic_split(node);
}

RTreeAccel::SplitResult RTreeAccel::quadratic_split(RTreeNode* node) {
  SplitResult result;
  result.first = std::make_unique<RTreeNode>(node->is_leaf, node->level);
  result.second = std::make_unique<RTreeNode>(node->is_leaf, node->level);

  // Pick seeds
  auto [seed1, seed2] = pick_seeds_quadratic(node);

  // Assign seeds to groups
  if (node->is_leaf) {
    result.first->entries.push_back(node->entries[seed1]);
    result.second->entries.push_back(node->entries[seed2]);
  } else {
    result.first->children.push_back(std::move(node->children[seed1]));
    result.second->children.push_back(std::move(node->children[seed2]));
    result.first->children.back()->parent = result.first.get();
    result.second->children.back()->parent = result.second.get();
  }

  result.first->update_mbr();
  result.second->update_mbr();

  // Mark seeds as assigned
  size_t n = node->size();
  std::vector<bool> assigned(n, false);
  assigned[seed1] = true;
  assigned[seed2] = true;
  size_t n_assigned = 2;

  // Assign remaining entries
  while (n_assigned < n) {
    // Check if one group needs all remaining entries
    size_t group1_size = result.first->size();
    size_t group2_size = result.second->size();
    size_t remaining = n - n_assigned;

    if (group1_size + remaining <= RTreeNode::MIN_ENTRIES) {
      // Assign all remaining to group 1
      for (size_t i = 0; i < n; ++i) {
        if (!assigned[i]) {
          if (node->is_leaf) {
            result.first->entries.push_back(node->entries[i]);
          } else {
            node->children[i]->parent = result.first.get();
            result.first->children.push_back(std::move(node->children[i]));
          }
        }
      }
      break;
    }

    if (group2_size + remaining <= RTreeNode::MIN_ENTRIES) {
      // Assign all remaining to group 2
      for (size_t i = 0; i < n; ++i) {
        if (!assigned[i]) {
          if (node->is_leaf) {
            result.second->entries.push_back(node->entries[i]);
          } else {
            node->children[i]->parent = result.second.get();
            result.second->children.push_back(std::move(node->children[i]));
          }
        }
      }
      break;
    }

    // Pick next entry
    int next = pick_next_quadratic(node, assigned, result.first->mbr, result.second->mbr);
    assigned[next] = true;
    n_assigned++;

    // Determine which group to add to
    search::AABB entry_bounds;
    if (node->is_leaf) {
      entry_bounds = node->entries[next].bounds;
    } else {
      entry_bounds = node->children[next]->mbr;
    }

    real_t enlargement1 = compute_enlargement(result.first->mbr, entry_bounds);
    real_t enlargement2 = compute_enlargement(result.second->mbr, entry_bounds);

    if (enlargement1 < enlargement2 ||
        (enlargement1 == enlargement2 && group1_size < group2_size)) {
      // Add to group 1
      if (node->is_leaf) {
        result.first->entries.push_back(node->entries[next]);
      } else {
        node->children[next]->parent = result.first.get();
        result.first->children.push_back(std::move(node->children[next]));
      }
      result.first->update_mbr();
    } else {
      // Add to group 2
      if (node->is_leaf) {
        result.second->entries.push_back(node->entries[next]);
      } else {
        node->children[next]->parent = result.second.get();
        result.second->children.push_back(std::move(node->children[next]));
      }
      result.second->update_mbr();
    }
  }

  result.first->update_mbr();
  result.second->update_mbr();

  return result;
}

std::pair<int, int> RTreeAccel::pick_seeds_quadratic(RTreeNode* node) {
  size_t n = node->size();
  real_t max_waste = -std::numeric_limits<real_t>::infinity();
  int seed1 = 0, seed2 = 1;

  // Find pair with maximum wasted space
  for (size_t i = 0; i < n - 1; ++i) {
    search::AABB mbr_i = node->is_leaf ? node->entries[i].bounds : node->children[i]->mbr;

    for (size_t j = i + 1; j < n; ++j) {
      search::AABB mbr_j = node->is_leaf ? node->entries[j].bounds : node->children[j]->mbr;

      search::AABB combined = mbr_i;
      combined.expand(mbr_j);

      real_t waste = combined.volume() - mbr_i.volume() - mbr_j.volume();

      if (waste > max_waste) {
        max_waste = waste;
        seed1 = static_cast<int>(i);
        seed2 = static_cast<int>(j);
      }
    }
  }

  return {seed1, seed2};
}

int RTreeAccel::pick_next_quadratic(RTreeNode* node,
                                    const std::vector<bool>& assigned,
                                    const search::AABB& mbr1,
                                    const search::AABB& mbr2) {
  real_t max_diff = -std::numeric_limits<real_t>::infinity();
  int best = -1;

  for (size_t i = 0; i < node->size(); ++i) {
    if (assigned[i]) continue;

    search::AABB entry_mbr = node->is_leaf ? node->entries[i].bounds : node->children[i]->mbr;

    real_t enlargement1 = compute_enlargement(mbr1, entry_mbr);
    real_t enlargement2 = compute_enlargement(mbr2, entry_mbr);
    real_t diff = std::abs(enlargement1 - enlargement2);

    if (diff > max_diff) {
      max_diff = diff;
      best = static_cast<int>(i);
    }
  }

  return best;
}

real_t RTreeAccel::compute_enlargement(const search::AABB& mbr,
                                       const search::AABB& new_mbr) const {
  search::AABB expanded = mbr;
  expanded.expand(new_mbr);
  return expanded.volume() - mbr.volume();
}

real_t RTreeAccel::compute_overlap(const search::AABB& mbr1,
                                  const search::AABB& mbr2) const {
  search::AABB intersection;
  for (int i = 0; i < 3; ++i) {
    intersection.min[i] = std::max(mbr1.min[i], mbr2.min[i]);
    intersection.max[i] = std::min(mbr1.max[i], mbr2.max[i]);
    if (intersection.min[i] > intersection.max[i]) {
      return 0.0;  // No overlap
    }
  }
  return intersection.volume();
}

// ---- Deletion support ----

RTreeAccel::RTreeNode* RTreeAccel::find_leaf(RTreeNode* node,
                                             index_t id,
                                             const search::AABB& bounds) {
  if (!node) return nullptr;

  if (node->is_leaf) {
    // Check if this leaf contains the entry
    for (const auto& entry : node->entries) {
      if (entry.id == id) {
        return node;
      }
    }
    return nullptr;
  }

  // Search children
  for (auto& child : node->children) {
    if (child->mbr.overlaps(bounds)) {
      RTreeNode* result = find_leaf(child.get(), id, bounds);
      if (result) return result;
    }
  }

  return nullptr;
}

void RTreeAccel::condense_tree(RTreeNode* leaf) {
  RTreeNode* node = leaf;
  std::vector<std::unique_ptr<RTreeNode>> orphaned_nodes;

  while (node != root_.get()) {
    RTreeNode* parent = node->parent;

    if (node->is_underfull()) {
      // Remove node from parent and collect for reinsertion
      auto it = std::find_if(parent->children.begin(), parent->children.end(),
                            [node](const auto& child) {
                              return child.get() == node;
                            });

      if (it != parent->children.end()) {
        orphaned_nodes.push_back(std::move(*it));
        parent->children.erase(it);
      }
    }

    parent->update_mbr();
    node = parent;
  }

  // Reinsert orphaned entries
  reinsert_orphans(orphaned_nodes);
}

void RTreeAccel::reinsert_orphans(const std::vector<std::unique_ptr<RTreeNode>>& orphans) {
  for (const auto& orphan : orphans) {
    if (orphan->is_leaf) {
      for (const auto& entry : orphan->entries) {
        insert(entry.id, entry.bounds);
      }
    } else {
      // Reinsert all entries from subtree
      std::stack<RTreeNode*> stack;
      stack.push(orphan.get());

      while (!stack.empty()) {
        RTreeNode* current = stack.top();
        stack.pop();

        if (current->is_leaf) {
          for (const auto& entry : current->entries) {
            insert(entry.id, entry.bounds);
          }
        } else {
          for (const auto& child : current->children) {
            stack.push(child.get());
          }
        }
      }
    }
  }
}

// ---- Point location ----

PointLocateResult RTreeAccel::locate_point(const MeshBase& mesh,
                                          const std::array<real_t,3>& point,
                                          index_t hint_cell) const {
  PointLocateResult result;
  result.found = false;
  result.cell_id = -1;

  if (!is_built_ || !root_) {
    return result;
  }

  stats_.n_queries++;
  auto start_time = std::chrono::steady_clock::now();

  // Check hint first
  if (hint_cell >= 0 && hint_cell < mesh.n_cells()) {
    if (SearchBuilders::point_in_cell(mesh, vertex_coords_, hint_cell, point,
                                      result.parametric_coords)) {
      result.found = true;
      result.cell_id = hint_cell;

      auto end_time = std::chrono::steady_clock::now();
      stats_.total_query_time_ms += std::chrono::duration<double, std::milli>(
          end_time - start_time).count();
      return result;
    }
  }

  // Search R-Tree
  locate_point_recursive(root_.get(), mesh, point, result);

  auto end_time = std::chrono::steady_clock::now();
  stats_.total_query_time_ms += std::chrono::duration<double, std::milli>(
      end_time - start_time).count();

  return result;
}

void RTreeAccel::locate_point_recursive(RTreeNode* node,
                                        const MeshBase& mesh,
                                        const std::array<real_t,3>& point,
                                        PointLocateResult& result) const {
  if (!node || !node->mbr.contains(point)) {
    return;
  }

  stats_.n_node_visits++;

  if (node->is_leaf) {
    // Check entries in leaf
    for (const auto& entry : node->entries) {
      if (entry.bounds.contains(point)) {
        index_t cell_id = entry.id;
        if (SearchBuilders::point_in_cell(mesh, vertex_coords_, cell_id, point,
                                         result.parametric_coords)) {
          result.found = true;
          result.cell_id = cell_id;
          return;
        }
      }
    }
  } else {
    // Search children
    for (const auto& child : node->children) {
      locate_point_recursive(child.get(), mesh, point, result);
      if (result.found) return;
    }
  }
}

std::vector<PointLocateResult> RTreeAccel::locate_points(
    const MeshBase& mesh,
    const std::vector<std::array<real_t,3>>& points) const {

  std::vector<PointLocateResult> results;
  results.reserve(points.size());

  for (const auto& point : points) {
    results.push_back(locate_point(mesh, point));
  }

  return results;
}

// ---- Nearest neighbor ----

std::pair<index_t, real_t> RTreeAccel::nearest_vertex(
    const MeshBase& mesh,
    const std::array<real_t,3>& point) const {

  if (!is_built_ || vertex_coords_.empty()) {
    return {-1, std::numeric_limits<real_t>::infinity()};
  }

  stats_.n_queries++;
  auto start_time = std::chrono::steady_clock::now();

  index_t best_idx = -1;
  real_t best_dist_sq = std::numeric_limits<real_t>::infinity();

  // Use vertex tree if available, otherwise use main tree
  RTreeNode* search_root = vertex_root_ ? vertex_root_.get() : root_.get();

  if (search_root) {
    nearest_neighbor_recursive(search_root, point, best_idx, best_dist_sq);
  }

  auto end_time = std::chrono::steady_clock::now();
  stats_.total_query_time_ms += std::chrono::duration<double, std::milli>(
      end_time - start_time).count();

  return {best_idx, std::sqrt(best_dist_sq)};
}

void RTreeAccel::nearest_neighbor_recursive(RTreeNode* node,
                                           const std::array<real_t,3>& point,
                                           index_t& best_idx,
                                           real_t& best_dist_sq) const {
  if (!node) return;

  // Check if this node can contain a closer point
  real_t min_dist_sq = min_distance_to_mbr(node->mbr, point);
  if (min_dist_sq >= best_dist_sq) {
    return;  // Prune this branch
  }

  stats_.n_node_visits++;

  if (node->is_leaf) {
    // Check vertices in leaf
    for (const auto& entry : node->entries) {
      index_t vid = entry.id;
      if (vid < vertex_coords_.size()) {
        const auto& v = vertex_coords_[vid];
        real_t dist_sq = search::norm_squared(search::subtract(v, point));
        if (dist_sq < best_dist_sq) {
          best_dist_sq = dist_sq;
          best_idx = vid;
        }
      }
    }
  } else {
    // Visit children in order of minimum distance
    std::vector<std::pair<real_t, RTreeNode*>> children_dist;
    for (const auto& child : node->children) {
      real_t dist = min_distance_to_mbr(child->mbr, point);
      children_dist.emplace_back(dist, child.get());
    }

    std::sort(children_dist.begin(), children_dist.end());

    for (const auto& [dist, child] : children_dist) {
      if (dist < best_dist_sq) {
        nearest_neighbor_recursive(child, point, best_idx, best_dist_sq);
      }
    }
  }
}

std::vector<std::pair<index_t, real_t>> RTreeAccel::k_nearest_vertices(
    const MeshBase& mesh,
    const std::array<real_t,3>& point,
    size_t k) const {

  if (!is_built_ || vertex_coords_.empty() || k == 0) {
    return {};
  }

  stats_.n_queries++;
  auto start_time = std::chrono::steady_clock::now();

  // Max heap to track k nearest
  std::priority_queue<std::pair<real_t, index_t>> max_heap;

  RTreeNode* search_root = vertex_root_ ? vertex_root_.get() : root_.get();
  if (search_root) {
    k_nearest_search(search_root, point, k, max_heap);
  }

  // Extract results
  std::vector<std::pair<index_t, real_t>> results;
  while (!max_heap.empty()) {
    auto [dist_sq, idx] = max_heap.top();
    max_heap.pop();
    results.push_back({idx, std::sqrt(dist_sq)});
  }

  std::reverse(results.begin(), results.end());

  auto end_time = std::chrono::steady_clock::now();
  stats_.total_query_time_ms += std::chrono::duration<double, std::milli>(
      end_time - start_time).count();

  return results;
}

void RTreeAccel::k_nearest_search(RTreeNode* root,
                                  const std::array<real_t,3>& point,
                                  size_t k,
                                  std::priority_queue<std::pair<real_t, index_t>>& max_heap) const {
  if (!root) return;

  // Priority queue for nodes to visit
  std::priority_queue<NNEntry, std::vector<NNEntry>, std::greater<NNEntry>> pq;
  pq.push({root, 0.0});

  while (!pq.empty()) {
    NNEntry entry = pq.top();
    pq.pop();

    // Prune if too far
    if (!max_heap.empty() && max_heap.size() >= k &&
        entry.min_dist >= max_heap.top().first) {
      continue;
    }

    stats_.n_node_visits++;

    if (entry.node->is_leaf) {
      // Check vertices in leaf
      for (const auto& e : entry.node->entries) {
        index_t vid = e.id;
        if (vid < vertex_coords_.size()) {
          const auto& v = vertex_coords_[vid];
          real_t dist_sq = search::norm_squared(search::subtract(v, point));

          if (max_heap.size() < k) {
            max_heap.push({dist_sq, vid});
          } else if (dist_sq < max_heap.top().first) {
            max_heap.pop();
            max_heap.push({dist_sq, vid});
          }
        }
      }
    } else {
      // Add children to priority queue
      for (const auto& child : entry.node->children) {
        real_t min_dist_sq = min_distance_to_mbr(child->mbr, point);
        if (max_heap.size() < k || min_dist_sq < max_heap.top().first) {
          pq.push({child.get(), min_dist_sq});
        }
      }
    }
  }
}

std::vector<index_t> RTreeAccel::vertices_in_radius(
    const MeshBase& mesh,
    const std::array<real_t,3>& point,
    real_t radius) const {

  if (!is_built_ || vertex_coords_.empty()) {
    return {};
  }

  stats_.n_queries++;
  auto start_time = std::chrono::steady_clock::now();

  std::vector<index_t> results;
  real_t radius_sq = radius * radius;

  RTreeNode* search_root = vertex_root_ ? vertex_root_.get() : root_.get();
  if (search_root) {
    radius_search_recursive(search_root, point, radius_sq, results);
  }

  auto end_time = std::chrono::steady_clock::now();
  stats_.total_query_time_ms += std::chrono::duration<double, std::milli>(
      end_time - start_time).count();

  return results;
}

void RTreeAccel::radius_search_recursive(RTreeNode* node,
                                        const std::array<real_t,3>& point,
                                        real_t radius_sq,
                                        std::vector<index_t>& results) const {
  if (!node) return;

  // Check if sphere overlaps node
  real_t min_dist_sq = min_distance_to_mbr(node->mbr, point);
  if (min_dist_sq > radius_sq) {
    return;
  }

  stats_.n_node_visits++;

  if (node->is_leaf) {
    // Check vertices in leaf
    for (const auto& entry : node->entries) {
      index_t vid = entry.id;
      if (vid < vertex_coords_.size()) {
        const auto& v = vertex_coords_[vid];
        real_t dist_sq = search::norm_squared(search::subtract(v, point));
        if (dist_sq <= radius_sq) {
          results.push_back(vid);
        }
      }
    }
  } else {
    // Search children
    for (const auto& child : node->children) {
      radius_search_recursive(child.get(), point, radius_sq, results);
    }
  }
}

// ---- Ray intersection ----

RayIntersectResult RTreeAccel::intersect_ray(
    const MeshBase& mesh,
    const std::array<real_t,3>& origin,
    const std::array<real_t,3>& direction,
    real_t max_distance) const {

  RayIntersectResult result;
  result.hit = false;
  result.distance = max_distance;

  if (!is_built_ || !root_ || boundary_triangles_.empty()) {
    return result;
  }

  stats_.n_queries++;
  auto start_time = std::chrono::steady_clock::now();

  search::Ray ray(origin, direction);
  ray.max_t = max_distance;

  std::vector<RayIntersectResult> all_hits;
  ray_traverse_recursive(root_.get(), ray, all_hits);

  // Find closest hit
  for (const auto& hit : all_hits) {
    if (hit.hit && hit.distance < result.distance) {
      result = hit;
    }
  }

  auto end_time = std::chrono::steady_clock::now();
  stats_.total_query_time_ms += std::chrono::duration<double, std::milli>(
      end_time - start_time).count();

  return result;
}

std::vector<RayIntersectResult> RTreeAccel::intersect_ray_all(
    const MeshBase& mesh,
    const std::array<real_t,3>& origin,
    const std::array<real_t,3>& direction,
    real_t max_distance) const {

  std::vector<RayIntersectResult> results;

  if (!is_built_ || !root_ || boundary_triangles_.empty()) {
    return results;
  }

  stats_.n_queries++;
  auto start_time = std::chrono::steady_clock::now();

  search::Ray ray(origin, direction);
  ray.max_t = max_distance;

  ray_traverse_recursive(root_.get(), ray, results);

  // Sort by distance
  std::sort(results.begin(), results.end(),
           [](const RayIntersectResult& a, const RayIntersectResult& b) {
             return a.distance < b.distance;
           });

  auto end_time = std::chrono::steady_clock::now();
  stats_.total_query_time_ms += std::chrono::duration<double, std::milli>(
      end_time - start_time).count();

  return results;
}

void RTreeAccel::ray_traverse_recursive(RTreeNode* node,
                                        const search::Ray& ray,
                                        std::vector<RayIntersectResult>& results) const {
  if (!node) return;

  // Check ray-MBR intersection
  real_t t_near, t_far;
  if (!ray_intersects_mbr(ray, node->mbr, t_near, t_far)) {
    return;
  }

  stats_.n_node_visits++;

  if (node->is_leaf) {
    // Test triangles in leaf
    for (const auto& entry : node->entries) {
      if (entry.id < boundary_triangles_.size()) {
        const auto& tri = boundary_triangles_[entry.id];

        real_t t;
        std::array<real_t,3> hit_point;
        std::array<real_t,2> bary;

        if (ray_triangle_intersection(ray, tri, t, hit_point, bary)) {
          RayIntersectResult result;
          result.hit = true;
          result.distance = t;
          result.hit_point = hit_point;
          result.normal = tri.normal;
          result.face_id = tri.face_id;
          result.barycentric = {bary[0], bary[1], 1.0 - bary[0] - bary[1]};
          results.push_back(result);
        }
      }
    }
  } else {
    // Traverse children
    for (const auto& child : node->children) {
      ray_traverse_recursive(child.get(), ray, results);
    }
  }
}

// ---- Region queries ----

std::vector<index_t> RTreeAccel::cells_in_box(
    const MeshBase& mesh,
    const std::array<real_t,3>& box_min,
    const std::array<real_t,3>& box_max) const {

  if (!is_built_ || !root_) {
    return {};
  }

  stats_.n_queries++;
  auto start_time = std::chrono::steady_clock::now();

  search::AABB query_box(box_min, box_max);
  std::vector<index_t> results;

  search_recursive(root_.get(), query_box, results);

  auto end_time = std::chrono::steady_clock::now();
  stats_.total_query_time_ms += std::chrono::duration<double, std::milli>(
      end_time - start_time).count();

  return results;
}

void RTreeAccel::search_recursive(RTreeNode* node,
                                 const search::AABB& query,
                                 std::vector<index_t>& results) const {
  if (!node || !node->mbr.overlaps(query)) {
    return;
  }

  stats_.n_node_visits++;

  if (node->is_leaf) {
    // Check entries in leaf
    for (const auto& entry : node->entries) {
      if (entry.bounds.overlaps(query)) {
        results.push_back(entry.id);
      }
    }
  } else {
    // Search children
    for (const auto& child : node->children) {
      search_recursive(child.get(), query, results);
    }
  }
}

std::vector<index_t> RTreeAccel::cells_in_sphere(
    const MeshBase& mesh,
    const std::array<real_t,3>& center,
    real_t radius) const {

  if (!is_built_ || !root_) {
    return {};
  }

  stats_.n_queries++;
  auto start_time = std::chrono::steady_clock::now();

  std::unordered_set<index_t> result_set;

  sphere_search_recursive(root_.get(), center, radius, result_set);

  std::vector<index_t> results(result_set.begin(), result_set.end());

  auto end_time = std::chrono::steady_clock::now();
  stats_.total_query_time_ms += std::chrono::duration<double, std::milli>(
      end_time - start_time).count();

  return results;
}

void RTreeAccel::sphere_search_recursive(RTreeNode* node,
                                        const std::array<real_t,3>& center,
                                        real_t radius,
                                        std::unordered_set<index_t>& results) const {
  if (!node) return;

  // Check if sphere overlaps MBR
  if (min_distance_to_mbr(node->mbr, center) > radius) {
    return;
  }

  stats_.n_node_visits++;

  if (node->is_leaf) {
    // Check entries in leaf
    real_t radius_sq = radius * radius;

    for (const auto& entry : node->entries) {
      auto closest = entry.bounds.closest_point(center);
      real_t dist_sq = search::norm_squared(search::subtract(closest, center));
      if (dist_sq <= radius_sq) {
        results.insert(entry.id);
      }
    }
  } else {
    // Search children
    for (const auto& child : node->children) {
      sphere_search_recursive(child.get(), center, radius, results);
    }
  }
}

// ---- Helper methods ----

real_t RTreeAccel::min_distance_to_mbr(const search::AABB& mbr,
                                       const std::array<real_t,3>& point) const {
  auto closest = mbr.closest_point(point);
  return search::norm_squared(search::subtract(closest, point));
}

real_t RTreeAccel::max_distance_to_mbr(const search::AABB& mbr,
                                       const std::array<real_t,3>& point) const {
  std::array<real_t,3> farthest;
  for (int i = 0; i < 3; ++i) {
    real_t d_min = std::abs(point[i] - mbr.min[i]);
    real_t d_max = std::abs(point[i] - mbr.max[i]);
    farthest[i] = d_min > d_max ? mbr.min[i] : mbr.max[i];
  }
  return search::norm_squared(search::subtract(farthest, point));
}

bool RTreeAccel::ray_intersects_mbr(const search::Ray& ray,
                                    const search::AABB& mbr,
                                    real_t& t_near,
                                    real_t& t_far) const {
  t_near = 0.0;
  t_far = ray.max_t;

  for (int i = 0; i < 3; ++i) {
    real_t inv_dir = 1.0 / ray.direction[i];
    real_t t0 = (mbr.min[i] - ray.origin[i]) * inv_dir;
    real_t t1 = (mbr.max[i] - ray.origin[i]) * inv_dir;

    if (inv_dir < 0.0) {
      std::swap(t0, t1);
    }

    t_near = std::max(t_near, t0);
    t_far = std::min(t_far, t1);

    if (t_near > t_far) {
      return false;
    }
  }

  return true;
}

bool RTreeAccel::ray_triangle_intersection(const search::Ray& ray,
                                          const BoundaryTriangle& tri,
                                          real_t& t,
                                          std::array<real_t,3>& hit_point,
                                          std::array<real_t,2>& bary) const {
  // Möller–Trumbore intersection algorithm
  const auto& v0 = tri.vertices[0];
  const auto& v1 = tri.vertices[1];
  const auto& v2 = tri.vertices[2];

  auto edge1 = search::subtract(v1, v0);
  auto edge2 = search::subtract(v2, v0);
  auto h = search::cross(ray.direction, edge2);
  real_t a = search::dot(edge1, h);

  if (std::abs(a) < 1e-10) {
    return false;  // Ray parallel to triangle
  }

  real_t f = 1.0 / a;
  auto s = search::subtract(ray.origin, v0);
  real_t u = f * search::dot(s, h);

  if (u < 0.0 || u > 1.0) {
    return false;
  }

  auto q = search::cross(s, edge1);
  real_t v = f * search::dot(ray.direction, q);

  if (v < 0.0 || u + v > 1.0) {
    return false;
  }

  t = f * search::dot(edge2, q);

  if (t > 0.0 && t <= ray.max_t) {
    bary[0] = u;
    bary[1] = v;
    hit_point = ray.point_at(t);
    return true;
  }

  return false;
}

// ---- Statistics ----

size_t RTreeAccel::count_nodes(RTreeNode* node) const {
  if (!node) return 0;

  size_t count = 1;
  if (!node->is_leaf) {
    for (const auto& child : node->children) {
      count += count_nodes(child.get());
    }
  }
  return count;
}

int RTreeAccel::get_tree_height(RTreeNode* node) const {
  if (!node) return 0;
  if (node->is_leaf) return 1;

  int max_height = 0;
  for (const auto& child : node->children) {
    max_height = std::max(max_height, get_tree_height(child.get()));
  }
  return max_height + 1;
}

size_t RTreeAccel::compute_memory_usage() const {
  size_t total = 0;

  // Tree nodes
  if (root_) {
    total += count_nodes(root_.get()) * sizeof(RTreeNode);
  }
  if (vertex_root_) {
    total += count_nodes(vertex_root_.get()) * sizeof(RTreeNode);
  }

  // Cached data
  total += vertex_coords_.size() * sizeof(std::array<real_t,3>);
  total += cell_aabbs_.size() * sizeof(search::AABB);
  total += cell_indices_.size() * sizeof(index_t);
  total += vertex_indices_.size() * sizeof(index_t);
  total += boundary_triangles_.size() * sizeof(BoundaryTriangle);

  return total;
}

bool RTreeAccel::validate_tree(RTreeNode* node) const {
  if (!node) node = root_.get();
  if (!node) return true;

  // Check node constraints
  if (node != root_.get()) {
    if (node->size() < RTreeNode::MIN_ENTRIES) {
      return false;  // Underfull
    }
  }

  if (node->size() > RTreeNode::MAX_ENTRIES) {
    return false;  // Overfull
  }

  // Check MBR correctness
  search::AABB computed_mbr;
  bool first = true;

  if (node->is_leaf) {
    for (const auto& entry : node->entries) {
      if (first) {
        computed_mbr = entry.bounds;
        first = false;
      } else {
        computed_mbr.expand(entry.bounds);
      }
    }
  } else {
    for (const auto& child : node->children) {
      if (first) {
        computed_mbr = child->mbr;
        first = false;
      } else {
        computed_mbr.expand(child->mbr);
      }

      // Recursively validate children
      if (!validate_tree(child.get())) {
        return false;
      }
    }
  }

  // Check if MBR matches
  const real_t eps = 1e-6;
  for (int i = 0; i < 3; ++i) {
    if (std::abs(node->mbr.min[i] - computed_mbr.min[i]) > eps ||
        std::abs(node->mbr.max[i] - computed_mbr.max[i]) > eps) {
      return false;  // MBR mismatch
    }
  }

  return true;
}

} // namespace svmp
