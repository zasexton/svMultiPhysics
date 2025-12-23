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

#include "Conformity.h"
#include "../Core/MeshBase.h"
#include <algorithm>

namespace svmp {

ClosureConformityEnforcer::ClosureConformityEnforcer(const Config& config)
    : config_(config) {}

NonConformity ClosureConformityEnforcer::check_conformity(
    const MeshBase& mesh,
    const std::vector<MarkType>& marks) const {
  (void)mesh;
  (void)marks;
  return {};
}

size_t ClosureConformityEnforcer::enforce_conformity(
    const MeshBase& mesh,
    std::vector<MarkType>& marks,
    const AdaptivityOptions& options) const {
  (void)mesh;
  (void)marks;
  (void)options;
  return 0;
}

std::map<size_t, std::map<size_t, double>> ClosureConformityEnforcer::generate_constraints(
    const MeshBase& mesh,
    const NonConformity& non_conformity) const {
  (void)mesh;
  (void)non_conformity;
  return {};
}

bool ClosureConformityEnforcer::is_edge_conforming(
    const MeshBase& mesh,
    size_t v1, size_t v2,
    const std::vector<MarkType>& marks) const {
  (void)mesh;
  (void)v1;
  (void)v2;
  (void)marks;
  return true;
}

bool ClosureConformityEnforcer::is_face_conforming(
    const MeshBase& mesh,
    const std::vector<size_t>& face_vertices,
    const std::vector<MarkType>& marks) const {
  (void)mesh;
  (void)face_vertices;
  (void)marks;
  return true;
}

std::vector<size_t> ClosureConformityEnforcer::find_edge_elements(
    const MeshBase& mesh,
    size_t v1, size_t v2) const {
  (void)mesh;
  (void)v1;
  (void)v2;
  return {};
}

std::vector<size_t> ClosureConformityEnforcer::find_face_elements(
    const MeshBase& mesh,
    const std::vector<size_t>& face_vertices) const {
  (void)mesh;
  (void)face_vertices;
  return {};
}

void ClosureConformityEnforcer::mark_for_closure(
    std::vector<MarkType>& marks,
    size_t elem_id) const {
  if (elem_id < marks.size() && marks[elem_id] == MarkType::NONE) {
    marks[elem_id] = MarkType::REFINE;
  }
}

HangingNodeConformityEnforcer::HangingNodeConformityEnforcer(const Config& config)
    : config_(config) {}

NonConformity HangingNodeConformityEnforcer::check_conformity(
    const MeshBase& mesh,
    const std::vector<MarkType>& marks) const {
  (void)mesh;
  (void)marks;
  return {};
}

size_t HangingNodeConformityEnforcer::enforce_conformity(
    const MeshBase& mesh,
    std::vector<MarkType>& marks,
    const AdaptivityOptions& options) const {
  (void)mesh;
  (void)marks;
  (void)options;
  return 0;
}

std::map<size_t, std::map<size_t, double>> HangingNodeConformityEnforcer::generate_constraints(
    const MeshBase& mesh,
    const NonConformity& non_conformity) const {
  (void)mesh;
  (void)non_conformity;
  return {};
}

std::vector<HangingNode> HangingNodeConformityEnforcer::find_edge_hanging_nodes(
    const MeshBase& mesh,
    size_t v1, size_t v2,
    const std::vector<MarkType>& marks) const {
  (void)mesh;
  (void)v1;
  (void)v2;
  (void)marks;
  return {};
}

std::vector<HangingNode> HangingNodeConformityEnforcer::find_face_hanging_nodes(
    const MeshBase& mesh,
    const std::vector<size_t>& face_vertices,
    const std::vector<MarkType>& marks) const {
  (void)mesh;
  (void)face_vertices;
  (void)marks;
  return {};
}

std::map<size_t, double> HangingNodeConformityEnforcer::generate_node_constraint(
    const MeshBase& mesh,
    const HangingNode& node) const {
  (void)mesh;
  return node.constraints;
}

MinimalClosureEnforcer::MinimalClosureEnforcer(const Config& config)
    : config_(config) {}

NonConformity MinimalClosureEnforcer::check_conformity(
    const MeshBase& mesh,
    const std::vector<MarkType>& marks) const {
  (void)mesh;
  (void)marks;
  return {};
}

size_t MinimalClosureEnforcer::enforce_conformity(
    const MeshBase& mesh,
    std::vector<MarkType>& marks,
    const AdaptivityOptions& options) const {
  (void)mesh;
  (void)marks;
  (void)options;
  return 0;
}

std::map<size_t, std::map<size_t, double>> MinimalClosureEnforcer::generate_constraints(
    const MeshBase& mesh,
    const NonConformity& non_conformity) const {
  (void)mesh;
  (void)non_conformity;
  return {};
}

std::vector<std::pair<size_t, RefinementPattern>> MinimalClosureEnforcer::compute_minimal_closure(
    const MeshBase& mesh,
    const std::vector<MarkType>& marks,
    const NonConformity& non_conformity) const {
  (void)mesh;
  (void)marks;
  (void)non_conformity;
  return {};
}

double MinimalClosureEnforcer::compute_closure_cost(
    const std::vector<std::pair<size_t, RefinementPattern>>& closure) const {
  (void)closure;
  return 0.0;
}

std::unique_ptr<ConformityEnforcer> ConformityEnforcerFactory::create(
    const AdaptivityOptions& options) {
  switch (options.conformity_mode) {
    case AdaptivityOptions::ConformityMode::ENFORCE_CONFORMING:
      return create_closure();
    case AdaptivityOptions::ConformityMode::ALLOW_HANGING_NODES:
      return create_hanging_node();
    case AdaptivityOptions::ConformityMode::MINIMAL_CLOSURE:
      return create_minimal_closure();
  }
  return create_closure();
}

std::unique_ptr<ConformityEnforcer> ConformityEnforcerFactory::create_closure(
    const ClosureConformityEnforcer::Config& config) {
  return std::make_unique<ClosureConformityEnforcer>(config);
}

std::unique_ptr<ConformityEnforcer> ConformityEnforcerFactory::create_hanging_node(
    const HangingNodeConformityEnforcer::Config& config) {
  return std::make_unique<HangingNodeConformityEnforcer>(config);
}

std::unique_ptr<ConformityEnforcer> ConformityEnforcerFactory::create_minimal_closure(
    const MinimalClosureEnforcer::Config& config) {
  return std::make_unique<MinimalClosureEnforcer>(config);
}

bool ConformityUtils::is_mesh_conforming(const MeshBase& mesh) {
  (void)mesh;
  return true;
}

std::vector<HangingNode> ConformityUtils::find_hanging_nodes(const MeshBase& mesh) {
  (void)mesh;
  return {};
}

size_t ConformityUtils::check_level_difference(
    const MeshBase& mesh,
    size_t elem1,
    size_t elem2) {
  (void)mesh;
  (void)elem1;
  (void)elem2;
  return 0;
}

void ConformityUtils::apply_constraints(
    std::vector<double>& solution,
    const std::map<size_t, std::map<size_t, double>>& constraints) {
  (void)solution;
  (void)constraints;
}

void ConformityUtils::eliminate_constraints(
    std::vector<std::vector<double>>& matrix,
    std::vector<double>& rhs,
    const std::map<size_t, std::map<size_t, double>>& constraints) {
  (void)matrix;
  (void)rhs;
  (void)constraints;
}

void ConformityUtils::write_nonconformity_to_field(
    MeshFields& fields,
    const MeshBase& mesh,
    const NonConformity& non_conformity) {
  (void)fields;
  (void)mesh;
  (void)non_conformity;
}

} // namespace svmp

