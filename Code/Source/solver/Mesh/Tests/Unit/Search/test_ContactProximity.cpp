#include "Search/ContactProximity.h"

#include "Core/MeshBase.h"
#include "Topology/CellShape.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {

constexpr svmp::label_t kSlaveLabel = 501;
constexpr svmp::label_t kMasterLabel = 502;

svmp::CellShape quad_shape() {
  return {svmp::CellFamily::Quad, 4, 1};
}

svmp::CellShape line_shape() {
  return {svmp::CellFamily::Line, 2, 1};
}

svmp::MeshBase make_vertex_mesh(
    const std::array<svmp::real_t, 3>& x,
    svmp::label_t label,
    svmp::gid_t vertex_gid) {
  svmp::MeshBase mesh;
  const std::vector<svmp::real_t> x_ref = {x[0], x[1], x[2]};
  mesh.build_from_arrays(3, x_ref, {0}, {}, {});
  mesh.set_vertex_gids({vertex_gid});
  mesh.set_vertex_label(0, label);
  return mesh;
}

svmp::MeshBase make_edge_mesh(
    const std::array<svmp::real_t, 3>& a,
    const std::array<svmp::real_t, 3>& b,
    svmp::label_t label,
    svmp::gid_t edge_gid) {
  svmp::MeshBase mesh;
  const std::vector<svmp::real_t> x_ref = {
      a[0], a[1], a[2],
      b[0], b[1], b[2]};
  mesh.build_from_arrays(3, x_ref, {0}, {}, {});
  mesh.set_edges_from_arrays(std::vector<std::array<svmp::index_t, 2>>{{{0, 1}}});
  mesh.set_edge_gids({edge_gid});
  mesh.set_edge_label(0, label);
  return mesh;
}

svmp::MeshBase make_quad_surface(double x, svmp::label_t label, svmp::gid_t face_gid) {
  svmp::MeshBase mesh;
  const std::vector<svmp::real_t> x_ref = {
      x, 0.0, 0.0,
      x, 1.0, 0.0,
      x, 1.0, 1.0,
      x, 0.0, 1.0};
  mesh.build_from_arrays(3, x_ref, {0}, {}, {});
  mesh.set_faces_from_arrays({quad_shape()}, {0, 4}, {0, 1, 2, 3},
                             {{{svmp::INVALID_INDEX, svmp::INVALID_INDEX}}});
  mesh.set_face_gids({face_gid});
  mesh.set_boundary_label(0, label);
  return mesh;
}

svmp::MeshBase make_two_face_self_contact_surface() {
  svmp::MeshBase mesh;
  const std::vector<svmp::real_t> x_ref = {
      0.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 1.0,
      0.0, 0.0, 1.0,
      0.2, 0.0, 0.0,
      0.2, 1.0, 0.0,
      0.2, 1.0, 1.0,
      0.2, 0.0, 1.0};
  mesh.build_from_arrays(3, x_ref, {0}, {}, {});
  const std::vector<svmp::offset_t> face_offsets = {0, 4, 8};
  const std::vector<svmp::index_t> faces = {
      0, 1, 2, 3,
      4, 7, 6, 5};
  mesh.set_faces_from_arrays({quad_shape(), quad_shape()}, face_offsets, faces,
                             {{{svmp::INVALID_INDEX, svmp::INVALID_INDEX}},
                              {{svmp::INVALID_INDEX, svmp::INVALID_INDEX}}});
  mesh.set_face_gids({900, 901});
  mesh.set_boundary_label(0, kSlaveLabel);
  mesh.set_boundary_label(1, kSlaveLabel);
  return mesh;
}

svmp::search::ContactCandidateOptions active_options(double radius = 1.0) {
  svmp::search::ContactCandidateOptions options;
  options.search_radius = radius;
  options.activation_distance = 0.3;
  options.only_nearest_per_source = false;
  options.include_inactive_candidates = true;
  options.remove_duplicate_pairs = true;
  return options;
}

svmp::search::ContactPairState state_for_gap(
    double gap,
    double activation_distance,
    bool include_inactive_candidates) {
  auto slave = make_quad_surface(0.0, kSlaveLabel, 100);
  auto master = make_quad_surface(gap, kMasterLabel, 200);

  auto options = active_options();
  options.activation_distance = activation_distance;
  options.include_inactive_candidates = include_inactive_candidates;

  svmp::search::ContactProximityRegistry registry;
  registry.register_contact(
      "state-contact",
      svmp::search::ContactSurfaceSpec::from_mesh(slave, kSlaveLabel),
      svmp::search::ContactSurfaceSpec::from_mesh(master, kMasterLabel),
      options);

  auto map = registry.build_trial_map("state-contact");
  EXPECT_EQ(map.pairs.size(), 1u);
  return map.pairs.empty() ? svmp::search::ContactPairState::Rejected
                           : map.pairs.front().state;
}

} // namespace

TEST(ContactProximityTest, BuildsFaceContactCandidateWithProjectionAndProvenance) {
  auto slave = make_quad_surface(0.0, kSlaveLabel, 100);
  auto master = make_quad_surface(0.2, kMasterLabel, 200);

  svmp::search::ContactProximityRegistry registry;
  registry.register_contact(
      "surface-contact",
      svmp::search::ContactSurfaceSpec::from_mesh(
          slave, kSlaveLabel, svmp::search::ContactEntityKind::Surface,
          svmp::Configuration::Reference, "slave"),
      svmp::search::ContactSurfaceSpec::from_mesh(
          master, kMasterLabel, svmp::search::ContactEntityKind::Surface,
          svmp::Configuration::Reference, "master"),
      active_options());

  auto map = registry.build_trial_map("surface-contact");
  ASSERT_EQ(map.state, svmp::search::ContactTransactionState::TrialIterate);
  ASSERT_EQ(map.pairs.size(), 1u);
  EXPECT_TRUE(map.valid_for_current_revisions());
  EXPECT_EQ(map.active_pair_count(), 1u);

  const auto& pair = map.pairs.front();
  EXPECT_EQ(pair.lifecycle_stage, svmp::search::ContactLifecycleStage::Classified);
  EXPECT_EQ(pair.state, svmp::search::ContactPairState::Active);
  EXPECT_EQ(pair.provenance.source_surface_name, "slave");
  EXPECT_EQ(pair.provenance.target_surface_name, "master");
  EXPECT_EQ(pair.provenance.source_label, kSlaveLabel);
  EXPECT_EQ(pair.provenance.target_label, kMasterLabel);
  EXPECT_EQ(pair.provenance.source_gid, 100);
  EXPECT_EQ(pair.provenance.target_gid, 200);
  EXPECT_EQ(pair.provenance.canonical_owner_rank, 0);
  EXPECT_TRUE(pair.projection.valid);
  EXPECT_NEAR(pair.projection.unsigned_gap, 0.2, 1.0e-12);
  EXPECT_EQ(pair.projection.side, "positive");
  EXPECT_TRUE(pair.projection.tangential_frame_valid);
  EXPECT_DOUBLE_EQ(pair.projection.tangential_slip_magnitude, 0.0);
  EXPECT_TRUE(std::isfinite(pair.projection.source_local_coordinates[0]));
  EXPECT_TRUE(std::isfinite(pair.projection.target_local_coordinates[0]));
  EXPECT_NEAR(pair.projection.tangent0[0] * pair.projection.source_normal[0] +
              pair.projection.tangent0[1] * pair.projection.source_normal[1] +
              pair.projection.tangent0[2] * pair.projection.source_normal[2], 0.0, 1.0e-12);
  EXPECT_NEAR(pair.projection.tangential_reference0[0] * pair.projection.source_normal[0] +
              pair.projection.tangential_reference0[1] * pair.projection.source_normal[1] +
              pair.projection.tangential_reference0[2] * pair.projection.source_normal[2], 0.0, 1.0e-12);
}

TEST(ContactProximityTest, DuplicateFilteringProvidesCanonicalSelfContactPairs) {
  auto mesh = make_two_face_self_contact_surface();

  svmp::search::ContactCandidateOptions options = active_options();
  options.activation_distance = 0.5;
  options.allow_self_pairs = false;
  options.remove_duplicate_pairs = true;

  svmp::search::ContactSurfaceSpec side =
      svmp::search::ContactSurfaceSpec::from_mesh(
          mesh, kSlaveLabel, svmp::search::ContactEntityKind::Surface,
          svmp::Configuration::Reference, "self");
  side.allow_self_contact = true;

  svmp::search::ContactProximityRegistry registry;
  registry.register_contact("self-contact", side, side, options);

  auto map = registry.build_trial_map("self-contact");
  ASSERT_EQ(map.pairs.size(), 1u);
  EXPECT_TRUE(map.has_diagnostic(svmp::search::ContactDiagnosticCode::DuplicatePairRemoved));
  const auto& pair = map.pairs.front();
  EXPECT_NE(pair.provenance.source_gid, pair.provenance.target_gid);
  EXPECT_EQ(pair.provenance.canonical_owner_rank, 0);
}

TEST(ContactProximityTest, ProjectionStateCoversVertexEdgeAndShellContactEntities) {
  auto vertex = make_vertex_mesh({{0.0, 0.0, 0.0}}, kSlaveLabel, 300);
  auto edge = make_edge_mesh({{0.2, -1.0, 0.0}}, {{0.2, 1.0, 0.0}}, kMasterLabel, 400);

  svmp::search::ContactProximityRegistry registry;
  registry.register_contact(
      "vertex-edge",
      svmp::search::ContactSurfaceSpec::from_mesh(
          vertex, kSlaveLabel, svmp::search::ContactEntityKind::Vertex,
          svmp::Configuration::Reference, "slave-vertex"),
      svmp::search::ContactSurfaceSpec::from_mesh(
          edge, kMasterLabel, svmp::search::ContactEntityKind::Edge,
          svmp::Configuration::Reference, "master-edge"),
      active_options());

  auto map = registry.build_trial_map("vertex-edge", {}, 1.25);
  ASSERT_EQ(map.pairs.size(), 1u);
  const auto& vertex_edge = map.pairs.front();
  EXPECT_EQ(vertex_edge.provenance.source_kind, svmp::search::ContactEntityKind::Vertex);
  EXPECT_EQ(vertex_edge.provenance.target_kind, svmp::search::ContactEntityKind::Edge);
  EXPECT_EQ(vertex_edge.provenance.time_level, 1.25);
  EXPECT_NEAR(vertex_edge.projection.unsigned_gap, 0.2, 1.0e-12);
  EXPECT_NEAR(vertex_edge.projection.target_local_coordinates[0], 0.0, 1.0e-12);
  EXPECT_TRUE(vertex_edge.projection.tangential_frame_valid);

  auto slave_shell = make_quad_surface(0.0, kSlaveLabel, 500);
  auto master_shell = make_quad_surface(0.3, kMasterLabel, 600);
  auto source = svmp::search::ContactSurfaceSpec::from_mesh(
      slave_shell, kSlaveLabel, svmp::search::ContactEntityKind::Shell,
      svmp::Configuration::Reference, "slave-shell");
  auto target = svmp::search::ContactSurfaceSpec::from_mesh(
      master_shell, kMasterLabel, svmp::search::ContactEntityKind::Shell,
      svmp::Configuration::Reference, "master-shell");
  source.shell_thickness = 0.1;
  target.shell_thickness = 0.1;

  svmp::search::ContactProximityRegistry shell_registry;
  auto shell_options = active_options();
  shell_options.activation_distance = 0.25;
  shell_registry.register_contact("shell-contact", source, target, shell_options);

  auto shell_map = shell_registry.build_trial_map("shell-contact");
  ASSERT_EQ(shell_map.pairs.size(), 1u);
  const auto& shell_pair = shell_map.pairs.front();
  EXPECT_EQ(shell_pair.provenance.source_kind, svmp::search::ContactEntityKind::Shell);
  EXPECT_EQ(shell_pair.provenance.target_kind, svmp::search::ContactEntityKind::Shell);
  EXPECT_NEAR(shell_pair.projection.shell_thickness_offset, 0.1, 1.0e-12);
  EXPECT_NEAR(shell_pair.projection.unsigned_gap, 0.2, 1.0e-12);
  EXPECT_EQ(shell_pair.state, svmp::search::ContactPairState::Active);
}

TEST(ContactProximityTest, ClassifiesActiveInactiveAndProjectedStates) {
  EXPECT_EQ(state_for_gap(0.2, 0.3, true), svmp::search::ContactPairState::Active);
  EXPECT_EQ(state_for_gap(0.2, 0.1, true), svmp::search::ContactPairState::Inactive);
  EXPECT_EQ(state_for_gap(0.2, 0.1, false), svmp::search::ContactPairState::Projected);
}

TEST(ContactProximityTest, RevisionAndExternalLayoutChangesInvalidateContactState) {
  auto slave = make_quad_surface(0.0, kSlaveLabel, 100);
  auto master = make_quad_surface(0.2, kMasterLabel, 200);
  slave.set_current_coords(slave.X_ref());
  master.set_current_coords(master.X_ref());

  svmp::search::ContactExternalRevisions external;
  external.fe_space_revision = 1;
  external.fe_dof_layout_revision = 2;

  svmp::search::ContactProximityRegistry registry;
  registry.register_contact(
      "surface-contact",
      svmp::search::ContactSurfaceSpec::from_mesh(
          slave, kSlaveLabel, svmp::search::ContactEntityKind::Surface,
          svmp::Configuration::Current, "slave"),
      svmp::search::ContactSurfaceSpec::from_mesh(
          master, kMasterLabel, svmp::search::ContactEntityKind::Surface,
          svmp::Configuration::Current, "master"),
      active_options());

  auto map = registry.build_trial_map("surface-contact", external);
  ASSERT_TRUE(map.valid_for_current_revisions(external));

  auto moved = master.X_ref();
  for (std::size_t i = 0; i < moved.size(); i += 3) {
    moved[i] += 0.1;
  }
  master.set_current_coords(moved);
  EXPECT_FALSE(map.valid_for_current_revisions(external));

  master.set_current_coords(master.X_ref());
  auto fresh = registry.build_trial_map("surface-contact", external);
  ASSERT_TRUE(fresh.valid_for_current_revisions(external));
  external.fe_dof_layout_revision = 3;
  EXPECT_FALSE(fresh.valid_for_current_revisions(external));
}

TEST(ContactProximityTest, RevisionDomainsIndependentlyInvalidateCommittedContactMaps) {
  auto expect_invalid_after = [](auto mutator) {
    auto slave = make_quad_surface(0.0, kSlaveLabel, 100);
    auto master = make_quad_surface(0.2, kMasterLabel, 200);

    svmp::search::ContactProximityRegistry registry;
    registry.register_contact(
        "surface-contact",
        svmp::search::ContactSurfaceSpec::from_mesh(slave, kSlaveLabel),
        svmp::search::ContactSurfaceSpec::from_mesh(master, kMasterLabel),
        active_options());

    registry.commit_map(registry.build_trial_map("surface-contact"));
    ASSERT_TRUE(registry.committed_map_valid("surface-contact"));

    mutator(slave, master);
    EXPECT_FALSE(registry.committed_map_valid("surface-contact"));
  };

  expect_invalid_after([](auto& slave, auto&) {
    slave.set_boundary_label(0, kSlaveLabel + 10);
  });
  expect_invalid_after([](auto&, auto& master) {
    master.set_face_gids({201});
  });
  expect_invalid_after([](auto& slave, auto&) {
    slave.attach_field(svmp::EntityKind::Vertex, "contact-tag",
                       svmp::FieldScalarType::Float64, 1);
  });
  expect_invalid_after([](auto& slave, auto&) {
    slave.use_current_configuration();
  });
  expect_invalid_after([](auto& slave, auto&) {
    slave.set_faces_from_arrays({quad_shape()}, {0, 4}, {0, 1, 2, 3},
                                {{{svmp::INVALID_INDEX, svmp::INVALID_INDEX}}});
  });
}

TEST(ContactProximityTest, TrialCommitRollbackRestartAndReinitializationAreExplicit) {
  auto slave = make_quad_surface(0.0, kSlaveLabel, 100);
  auto master = make_quad_surface(0.2, kMasterLabel, 200);

  svmp::search::ContactProximityRegistry registry;
  registry.register_contact(
      "surface-contact",
      svmp::search::ContactSurfaceSpec::from_mesh(slave, kSlaveLabel),
      svmp::search::ContactSurfaceSpec::from_mesh(master, kMasterLabel),
      active_options());

  auto trial = registry.build_trial_map("surface-contact");
  ASSERT_EQ(trial.pairs.size(), 1u);
  const auto trial_revision = trial.revision_key();

  auto rollback = trial;
  rollback.rollback_trial();
  EXPECT_EQ(rollback.state, svmp::search::ContactTransactionState::RolledBack);
  EXPECT_TRUE(rollback.pairs.empty());

  registry.commit_map(trial);
  ASSERT_TRUE(registry.committed_map_valid("surface-contact"));
  const auto* committed = registry.committed_map("surface-contact");
  ASSERT_NE(committed, nullptr);
  EXPECT_EQ(committed->state, svmp::search::ContactTransactionState::AcceptedTimeStep);
  EXPECT_EQ(committed->revision_key(), trial_revision);

  auto restart = committed->restart_metadata();
  EXPECT_EQ(restart.name, "surface-contact");
  EXPECT_EQ(restart.pair_count, 1u);
  EXPECT_EQ(restart.active_pair_count, 1u);
  EXPECT_EQ(restart.accepted_state, svmp::search::ContactTransactionState::AcceptedTimeStep);

  auto remesh_state = *committed;
  remesh_state.reinitialize_after_remesh_or_repartition("remesh replaced contact surfaces");
  EXPECT_TRUE(remesh_state.pairs.empty());
  EXPECT_TRUE(remesh_state.has_diagnostic(
      svmp::search::ContactDiagnosticCode::ReinitializedAfterRemeshOrRepartition));
  EXPECT_EQ(remesh_state.state,
            svmp::search::ContactTransactionState::AcceptedRemeshRezoneState);
}

TEST(ContactProximityTest, RestartMetadataRebuildsEquivalentAcceptedState) {
  auto slave = make_quad_surface(0.0, kSlaveLabel, 100);
  auto master = make_quad_surface(0.2, kMasterLabel, 200);

  svmp::search::ContactProximityRegistry registry;
  registry.register_contact(
      "surface-contact",
      svmp::search::ContactSurfaceSpec::from_mesh(slave, kSlaveLabel),
      svmp::search::ContactSurfaceSpec::from_mesh(master, kMasterLabel),
      active_options());
  registry.commit_map(registry.build_trial_map("surface-contact"));
  ASSERT_NE(registry.committed_map("surface-contact"), nullptr);
  const auto restart = registry.committed_map("surface-contact")->restart_metadata();

  svmp::search::ContactProximityRegistry rebuilt_registry;
  rebuilt_registry.register_contact(
      "surface-contact",
      svmp::search::ContactSurfaceSpec::from_mesh(slave, kSlaveLabel),
      svmp::search::ContactSurfaceSpec::from_mesh(master, kMasterLabel),
      active_options());
  rebuilt_registry.commit_map(rebuilt_registry.build_trial_map("surface-contact"));
  ASSERT_NE(rebuilt_registry.committed_map("surface-contact"), nullptr);
  const auto rebuilt = rebuilt_registry.committed_map("surface-contact")->restart_metadata();

  EXPECT_EQ(rebuilt.source_revision_key, restart.source_revision_key);
  EXPECT_EQ(rebuilt.target_revision_key, restart.target_revision_key);
  EXPECT_EQ(rebuilt.contact_revision_key, restart.contact_revision_key);
  EXPECT_EQ(rebuilt.candidate_generation_epoch, restart.candidate_generation_epoch);
  EXPECT_EQ(rebuilt.active_set_epoch, restart.active_set_epoch);
  EXPECT_EQ(rebuilt.pair_count, restart.pair_count);
  EXPECT_EQ(rebuilt.active_pair_count, restart.active_pair_count);
}

TEST(ContactProximityTest, SearchRadiusMissProducesActionableDiagnostic) {
  auto slave = make_quad_surface(0.0, kSlaveLabel, 100);
  auto master = make_quad_surface(0.2, kMasterLabel, 200);
  auto options = active_options(0.05);

  svmp::search::ContactProximityRegistry registry;
  registry.register_contact(
      "surface-contact",
      svmp::search::ContactSurfaceSpec::from_mesh(slave, kSlaveLabel),
      svmp::search::ContactSurfaceSpec::from_mesh(master, kMasterLabel),
      options);

  const auto map = registry.build_trial_map("surface-contact");
  EXPECT_TRUE(map.pairs.empty());
  EXPECT_TRUE(map.has_diagnostic(svmp::search::ContactDiagnosticCode::SearchRadiusMiss));
}

TEST(ContactProximityTest, DiagnosticsDistinguishNoContactProjectionFailureStaleAndUnsupportedTopology) {
  auto slave = make_quad_surface(0.0, kSlaveLabel, 100);
  auto master = make_quad_surface(0.2, kMasterLabel, 200);

  svmp::search::ContactProximityRegistry no_contact_registry;
  no_contact_registry.register_contact(
      "no-contact",
      svmp::search::ContactSurfaceSpec::from_mesh(slave, kSlaveLabel + 1),
      svmp::search::ContactSurfaceSpec::from_mesh(master, kMasterLabel),
      active_options());
  const auto no_contact = no_contact_registry.build_trial_map("no-contact");
  EXPECT_TRUE(no_contact.pairs.empty());
  EXPECT_TRUE(no_contact.has_diagnostic(svmp::search::ContactDiagnosticCode::NoContact));

  auto finite_vertex = make_vertex_mesh({{0.0, 0.0, 0.0}}, kSlaveLabel, 10);
  const auto nan = std::numeric_limits<svmp::real_t>::quiet_NaN();
  auto invalid_vertex = make_vertex_mesh({{nan, 0.0, 0.0}}, kMasterLabel, 11);
  svmp::search::ContactProximityRegistry projection_registry;
  projection_registry.register_contact(
      "projection-failure",
      svmp::search::ContactSurfaceSpec::from_mesh(
          finite_vertex, kSlaveLabel, svmp::search::ContactEntityKind::Vertex),
      svmp::search::ContactSurfaceSpec::from_mesh(
          invalid_vertex, kMasterLabel, svmp::search::ContactEntityKind::Vertex),
      active_options());
  const auto projection_failure = projection_registry.build_trial_map("projection-failure");
  EXPECT_TRUE(projection_failure.has_diagnostic(
      svmp::search::ContactDiagnosticCode::ProjectionFailure));

  svmp::search::ContactProximityRegistry stale_registry;
  stale_registry.register_contact(
      "stale-contact",
      svmp::search::ContactSurfaceSpec::from_mesh(slave, kSlaveLabel),
      svmp::search::ContactSurfaceSpec::from_mesh(master, kMasterLabel),
      active_options());
  auto stale = stale_registry.build_trial_map("stale-contact");
  ASSERT_EQ(stale.pairs.size(), 1u);
  stale.mark_stale();
  EXPECT_EQ(stale.pairs_in_state(svmp::search::ContactPairState::Stale).size(), 1u);
  EXPECT_TRUE(stale.has_diagnostic(svmp::search::ContactDiagnosticCode::StaleRevision));

  svmp::search::ContactProximityRegistry unsupported_registry;
  unsupported_registry.register_contact(
      "unsupported-edge-contact",
      svmp::search::ContactSurfaceSpec::from_mesh(
          slave, svmp::INVALID_LABEL, svmp::search::ContactEntityKind::Edge),
      svmp::search::ContactSurfaceSpec::from_mesh(
          master, svmp::INVALID_LABEL, svmp::search::ContactEntityKind::Edge),
      active_options());
  const auto unsupported = unsupported_registry.build_trial_map("unsupported-edge-contact");
  EXPECT_TRUE(unsupported.pairs.empty());
  EXPECT_TRUE(unsupported.has_diagnostic(
      svmp::search::ContactDiagnosticCode::UnsupportedTopology));
}
