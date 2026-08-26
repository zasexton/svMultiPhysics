#ifndef SVMP_APPLICATION_TESTS_NATIVE_MANUFACTURED_CHANNEL_MPI_HARNESS_H
#define SVMP_APPLICATION_TESTS_NATIVE_MANUFACTURED_CHANNEL_MPI_HARNESS_H

#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Backends/Interfaces/DofPermutation.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Systems/FESystem.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Fields/MeshFields.h"
#include "Mesh/Mesh.h"
#include "Parameters.h"
#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "tinyxml2.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace native_manufactured_channel_mpi {

namespace channel =
    svmp::Physics::formulations::navier_stokes;

inline constexpr int interface_marker = 27410;
inline constexpr int inlet_marker = 27411;
inline constexpr int outlet_marker = 27412;
inline constexpr int side_wall_marker = 27413;
inline constexpr int anchor_marker = 27414;
inline constexpr int other_marker = 27415;
inline constexpr svmp::FE::Real channel_length = 2.0;
inline constexpr svmp::FE::Real window_height = 1.0;
inline constexpr svmp::FE::Real channel_depth = 1.0;
inline constexpr int aggregation_overlap_layers = 8;

struct ChannelArrays {
  std::vector<svmp::real_t> coordinates{};
  std::vector<svmp::offset_t> offsets{};
  std::vector<svmp::index_t> connectivity{};
  std::vector<svmp::CellShape> shapes{};
};

[[nodiscard]] inline ChannelArrays makeChannelArrays(
    int upper_subdivisions)
{
  if (upper_subdivisions < 2) {
    throw std::invalid_argument(
        "distributed native channel requires at least two upper layers");
  }

  constexpr int cells_x = 2;
  constexpr int cells_z = 1;
  std::vector<svmp::real_t> y_coordinates;
  y_coordinates.reserve(
      static_cast<std::size_t>(upper_subdivisions + 4));
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

  ChannelArrays arrays;
  arrays.coordinates.reserve(
      static_cast<std::size_t>(nodes_x * nodes_y * nodes_z * 3));
  for (int k = 0; k < nodes_z; ++k) {
    for (int j = 0; j < nodes_y; ++j) {
      for (int i = 0; i < nodes_x; ++i) {
        arrays.coordinates.push_back(
            channel_length * static_cast<svmp::real_t>(i) /
            static_cast<svmp::real_t>(cells_x));
        arrays.coordinates.push_back(
            y_coordinates[static_cast<std::size_t>(j)]);
        arrays.coordinates.push_back(
            channel_depth * static_cast<svmp::real_t>(k) /
            static_cast<svmp::real_t>(cells_z));
      }
    }
  }

  constexpr std::array<std::array<std::size_t, 4>, 6>
      outlet_tetrahedra{{
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

  arrays.offsets.push_back(0);
  const int cells_y = nodes_y - 1;
  const auto cell_count = static_cast<std::size_t>(
      cells_x * cells_y * cells_z *
      static_cast<int>(outlet_tetrahedra.size()));
  arrays.offsets.reserve(cell_count + 1u);
  arrays.connectivity.reserve(4u * cell_count);
  arrays.shapes.reserve(cell_count);
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
        const auto& tetrahedra =
            i == 0 ? inlet_tetrahedra : outlet_tetrahedra;
        for (const auto& tetrahedron : tetrahedra) {
          for (const auto local_vertex : tetrahedron) {
            arrays.connectivity.push_back(nodes[local_vertex]);
          }
          arrays.offsets.push_back(
              static_cast<svmp::offset_t>(arrays.connectivity.size()));
          arrays.shapes.push_back(
              svmp::CellShape{svmp::CellFamily::Tetra, 4, 1});
        }
      }
    }
  }
  return arrays;
}

inline void registerChannelLabels(svmp::MeshBase& mesh)
{
  mesh.register_label("native_channel_inlet",
                      static_cast<svmp::label_t>(inlet_marker));
  mesh.register_label("native_channel_outlet",
                      static_cast<svmp::label_t>(outlet_marker));
  mesh.register_label("native_channel_side_wall",
                      static_cast<svmp::label_t>(side_wall_marker));
  mesh.register_label("native_channel_anchor",
                      static_cast<svmp::label_t>(anchor_marker));
  mesh.register_label("native_channel_other",
                      static_cast<svmp::label_t>(other_marker));
}

inline void labelChannelBoundary(
    svmp::Mesh& mesh,
    int upper_subdivisions)
{
  auto& local_mesh = mesh.local_mesh();
  registerChannelLabels(local_mesh);
  constexpr svmp::FE::Real tolerance = 1.0e-12;
  const auto on_plane = [](svmp::FE::Real value,
                           svmp::FE::Real target) {
    return std::abs(value - target) <= tolerance;
  };
  std::array<unsigned long long, 5> local_counts{};
  for (const auto face : local_mesh.boundary_faces()) {
    const auto vertices = local_mesh.face_vertices(face);
    if (vertices.size() != 3u) {
      throw std::runtime_error(
          "distributed native channel has a nontriangular boundary face");
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
      const auto point = local_mesh.get_vertex_coords(vertex);
      on_inlet = on_inlet && on_plane(point[0], 0.0);
      on_outlet = on_outlet && on_plane(point[0], channel_length);
      on_side_wall =
          on_side_wall && on_plane(point[2], channel_depth);
      on_anchor = on_anchor && on_plane(point[1], -2.0);
      minimum_y = std::min(
          minimum_y, static_cast<svmp::FE::Real>(point[1]));
      maximum_y = std::max(
          maximum_y, static_cast<svmp::FE::Real>(point[1]));
    }
    const bool in_measured_window =
        minimum_y >= -tolerance &&
        maximum_y <= window_height + tolerance;

    int marker = other_marker;
    std::size_t marker_index = 4u;
    if (on_inlet && in_measured_window) {
      marker = inlet_marker;
      marker_index = 0u;
    } else if (on_outlet && in_measured_window) {
      marker = outlet_marker;
      marker_index = 1u;
    } else if (on_side_wall && in_measured_window) {
      marker = side_wall_marker;
      marker_index = 2u;
    } else if (on_anchor) {
      marker = anchor_marker;
      marker_index = 3u;
    }
    mesh.set_boundary_label(face, static_cast<svmp::label_t>(marker));
    if (mesh.is_owned_face(face)) {
      ++local_counts[marker_index];
    }
  }

  std::array<unsigned long long, 5> global_counts{};
  MPI_Allreduce(local_counts.data(),
                global_counts.data(),
                static_cast<int>(global_counts.size()),
                MPI_UNSIGNED_LONG_LONG,
                MPI_SUM,
                mesh.mpi_comm());
  const auto expected_end_faces =
      static_cast<unsigned long long>(2 * upper_subdivisions);
  const auto expected_side_faces =
      static_cast<unsigned long long>(4 * upper_subdivisions);
  constexpr unsigned long long expected_anchor_faces = 4u;
  if (global_counts[0] != expected_end_faces ||
      global_counts[1] != expected_end_faces ||
      global_counts[2] != expected_side_faces ||
      global_counts[3] != expected_anchor_faces ||
      global_counts[4] == 0u) {
    throw std::runtime_error(
        "distributed native channel boundary labeling is incomplete");
  }
}

inline void attachChannelLevelSetField(svmp::Mesh& mesh)
{
  auto& local_mesh = mesh.local_mesh();
  const auto field = svmp::MeshFields::attach_field(
      local_mesh,
      svmp::EntityKind::Vertex,
      "phi_native_channel",
      svmp::FieldScalarType::Float64,
      1);
  auto* values = svmp::MeshFields::field_data_as<svmp::real_t>(
      local_mesh, field);
  if (values == nullptr) {
    throw std::runtime_error(
        "distributed native channel level-set allocation failed");
  }
  for (std::size_t vertex = 0u;
       vertex < local_mesh.n_vertices();
       ++vertex) {
    values[vertex] =
        local_mesh.get_vertex_coords(
            static_cast<svmp::index_t>(vertex))[1] -
        svmp::real_t{0.5};
  }
}

[[nodiscard]] inline std::shared_ptr<svmp::Mesh>
makePartitionedChannelMesh(
    int upper_subdivisions,
    std::string partition_method)
{
  if (partition_method != "block" && partition_method != "metis") {
    throw std::invalid_argument(
        "distributed native channel partition method is unsupported");
  }
  const auto arrays = makeChannelArrays(upper_subdivisions);
  auto mesh =
      std::make_shared<svmp::Mesh>(svmp::MeshComm(MPI_COMM_WORLD));
  mesh->build_from_arrays_global_and_partition(
      /*spatial_dim=*/3,
      arrays.coordinates,
      arrays.offsets,
      arrays.connectivity,
      arrays.shapes,
      svmp::PartitionHint::Cells,
      /*ghost_layers=*/aggregation_overlap_layers,
      {{"partition_method", std::move(partition_method)}});
  labelChannelBoundary(*mesh, upper_subdivisions);
  attachChannelLevelSetField(*mesh);
  return mesh;
}

[[nodiscard]] inline std::array<svmp::FE::Real, 3>
vertexPoint(const svmp::Mesh& mesh, std::size_t vertex)
{
  const auto& coordinates = mesh.X_ref();
  std::array<svmp::FE::Real, 3> point{};
  for (std::size_t component = 0u; component < point.size(); ++component) {
    point[component] = static_cast<svmp::FE::Real>(
        coordinates[3u * vertex + component]);
  }
  return point;
}

[[nodiscard]] inline std::unique_ptr<Parameters>
parseParameters(const std::string& xml)
{
  tinyxml2::XMLDocument document;
  if (document.Parse(xml.c_str()) != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error(document.ErrorStr());
  }
  auto* root =
      document.FirstChildElement(Parameters::FSI_FILE.c_str());
  if (root == nullptr) {
    throw std::runtime_error(
        "distributed native channel configuration lacks its root element");
  }
  auto parameters = std::make_unique<Parameters>();
  parameters->set_equation_values(root);
  return parameters;
}

inline void allreduceInPlace(
    std::span<svmp::FE::Real> values,
    MPI_Comm communicator)
{
  if (values.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(
        "distributed native channel reduction exceeds the MPI count limit");
  }
  std::vector<svmp::FE::Real> reduced(values.size(), 0.0);
  MPI_Allreduce(values.data(),
                reduced.data(),
                static_cast<int>(values.size()),
                MPI_DOUBLE,
                MPI_SUM,
                communicator);
  std::copy(reduced.begin(), reduced.end(), values.begin());
}

struct TracePatchEvidence {
  bool localized_support_patch{false};
  svmp::FE::GlobalIndex root_cell_gid{svmp::FE::INVALID_GLOBAL_INDEX};
  std::vector<svmp::FE::GlobalIndex> support_cell_gids{};
  std::vector<std::array<svmp::FE::GlobalIndex, 2>>
      boundary_rule_physical_keys{};
  std::size_t boundary_rule_count{0u};
  std::size_t raw_support_dof_count{0u};
  std::size_t terminal_tangent_dof_count{0u};
  std::size_t rigid_mode_candidate_count{0u};
  std::size_t structural_rigid_mode_count{0u};
  std::size_t rigid_mode_constraint_rank{0u};
  std::size_t maximum_cell_support_overlap{0u};
  svmp::FE::Real retained_support_physical_volume{0.0};
  svmp::FE::Real generated_boundary_physical_measure{0.0};
  svmp::FE::Real directly_proven_upper_bound{0.0};
  svmp::FE::analysis::GeneratedBoundaryRigidModeQuotientStatus
      rigid_mode_quotient_status{
          svmp::FE::analysis::GeneratedBoundaryRigidModeQuotientStatus::
              NotApplicable};
  svmp::FE::math::DenseExactDyadicProofInput proof_input{
      svmp::FE::math::DenseExactDyadicProofInput::DenseBinary64Matrix};
  bool exact_rigid_factor_action_proven{false};
  bool denominator_positive_definite_proven{false};
  bool numerator_positive_semidefinite_proven{false};
  bool upper_inequality_proven{false};
  bool exact_factorized_materialization_proven{false};
  bool exact_sparse_map_applied{false};
  bool exact_common_kernel_proven{false};
  std::size_t exact_dimension{0u};
  std::size_t denominator_rank{0u};
  std::size_t numerator_rank{0u};
  std::size_t numerator_gram_block_count{0u};
  std::size_t denominator_gram_block_count{0u};
  std::size_t numerator_gram_row_count{0u};
  std::size_t denominator_gram_row_count{0u};
  std::size_t numerator_weight_term_count{0u};
  std::size_t denominator_weight_term_count{0u};
  std::size_t factorized_input_dimension{0u};
  std::size_t exact_common_kernel_nullity{0u};
};

struct Sample {
  svmp::FE::Real target_wet_fraction{0.0};
  std::array<svmp::FE::Real, 3> active_measures{};
  std::array<svmp::FE::Real, 3> parent_measures{};
  std::array<std::size_t, 3> active_rule_counts{};
  std::array<std::size_t, 3> retained_active_rule_counts{};
  std::array<int, 3> active_markers{{-1, -1, -1}};
  std::array<svmp::FE::Real, 3> operator_work{};
  std::array<std::size_t, 3> generated_route_term_counts{};
  std::size_t physical_role_boundary_term_count{0u};
  std::size_t trace_patch_count{0u};
  std::size_t trace_localized_support_patch_count{0u};
  std::size_t trace_maximum_factorized_input_dimension{0u};
  std::size_t trace_boundary_rule_count{0u};
  std::size_t trace_maximum_support_overlap{0u};
  std::uint64_t trace_certificate_digest{0u};
  svmp::FE::Real trace_global_conservative_upper_bound{0.0};
  svmp::FE::Real trace_grouped_symmetric_ratio{0.0};
  std::vector<TracePatchEvidence> trace_partition_invariant_patches{};
  bool trace_revision_match{false};
  bool trace_factorized_proof_valid{false};
};

class Harness {
public:
  using ActiveSide = svmp::FE::geometry::CutIntegrationSide;

  inline static constexpr svmp::FE::Real inlet_traction = 1.25;
  inline static constexpr svmp::FE::Real outlet_pressure = 1.2;
  inline static constexpr svmp::FE::Real prescribed_side_velocity = 0.4;
  inline static constexpr svmp::FE::Real viscosity = 0.02;
  inline static constexpr svmp::FE::Real nitsche_gamma = 16.0;
  inline static constexpr svmp::FE::Real side_facet_normal_scale =
      2.0 / 3.0;

  Harness(ActiveSide active_side,
          int upper_subdivisions,
          std::string partition_method)
      : active_side_(active_side),
        upper_subdivisions_(upper_subdivisions),
        mesh_(makePartitionedChannelMesh(
            upper_subdivisions,
            std::move(partition_method))),
        system_(std::make_unique<svmp::FE::systems::FESystem>(mesh_))
  {
    if (active_side_ != ActiveSide::Negative &&
        active_side_ != ActiveSide::Positive) {
      throw std::invalid_argument(
          "distributed native channel requires a volume active side");
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);

    auto pressure_space = svmp::FE::spaces::SpaceFactory::create_h1(
        svmp::FE::ElementType::Tetra4, 1);
    auto velocity_space =
        svmp::FE::spaces::SpaceFactory::create_vector_h1(
            svmp::FE::ElementType::Tetra4, 1, 3);
    level_set_ = system_->addField(svmp::FE::systems::FieldSpec{
        .name = "phi_native_channel",
        .space = pressure_space,
        .components = 1,
    });

    channel::IncompressibleNavierStokesVMSOptions options;
    options.symmetric_nitsche_energy_qualification_scope =
        channel::SymmetricNitscheEnergyQualificationScope::
            JointLowLevelPrerequisite;
    options.velocity_field_name = "u_native_channel";
    options.pressure_field_name = "p_native_channel";
    options.density = 1.0;
    options.viscosity = viscosity;
    options.enable_convection = false;
    options.enable_vms = false;
    options.jit_policy.enable = false;
    options.velocity_dirichlet.push_back(
        channel::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
            .boundary_marker = anchor_marker,
            .value = {0.0, 0.0, 0.0},
        });
    options.velocity_dirichlet_weak.push_back(
        channel::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
            .boundary_marker = side_wall_marker,
            .value = {0.0, 0.0, prescribed_side_velocity},
        });
    options.traction_neumann.push_back(
        channel::IncompressibleNavierStokesVMSOptions::TractionNeumannBC{
            .boundary_marker = inlet_marker,
            .traction = {0.0, inlet_traction, 0.0},
        });
    options.pressure_outflow.push_back(
        channel::IncompressibleNavierStokesVMSOptions::PressureOutflowBC{
            .boundary_marker = outlet_marker,
            .pressure = outlet_pressure,
            .backflow_beta = 0.0,
        });
    options.free_surface.push_back(
        channel::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation =
                channel::FreeSurfaceImplementation::UnfittedLevelSet,
            .interface_marker = interface_marker,
            .level_set_field_name = "phi_native_channel",
            .generated_interface_domain_id =
                "native_manufactured_channel",
            .generated_interface_geometry = "LinearCorner",
            .level_set_isovalue = 0.0,
            .active_domain =
                active_side_ == ActiveSide::Negative
                    ? channel::FreeSurfaceActiveDomain::LevelSetNegative
                    : channel::FreeSurfaceActiveDomain::LevelSetPositive,
            .active_domain_method =
                channel::FreeSurfaceActiveDomainMethod::CutVolume,
            .external_pressure = 0.0,
            .surface_tension = 0.0,
            .use_level_set_curvature = false,
            .cut_cell_stabilization = {.enabled = false},
            .small_cut_aggregation = true,
        });
    options.nitsche_gamma = nitsche_gamma;
    options.nitsche_symmetric = true;
    options.nitsche_scale_with_p = false;

    channel::IncompressibleNavierStokesVMSModule module(
        velocity_space, pressure_space, std::move(options));
    module.registerOn(*system_);

    svmp::FE::systems::SetupOptions setup;
    setup.assembler_name = "ParallelAssembler";
    setup.assembly_options.ghost_policy =
        svmp::FE::assembly::GhostPolicy::ReverseScatter;
    setup.assembly_options.deterministic = true;
    setup.assembly_options.overlap_communication = false;
    setup.dof_options.global_numbering =
        svmp::FE::dofs::GlobalNumberingMode::OwnerContiguous;
    setup.dof_options.ownership =
        svmp::FE::dofs::OwnershipStrategy::LowestRank;
    setup.dof_options.my_rank = rank_;
    setup.dof_options.world_size = size_;
    setup.dof_options.mpi_comm = MPI_COMM_WORLD;
    setup.use_backend_row_ownership_for_assembly = true;
    setup.retain_serial_sparsity = false;
    system_->setup(setup);

    velocity_ = system_->findFieldByName("u_native_channel");
    pressure_ = system_->findFieldByName("p_native_channel");
    if (velocity_ == svmp::FE::INVALID_FIELD_ID ||
        pressure_ == svmp::FE::INVALID_FIELD_ID) {
      throw std::runtime_error(
          "distributed native channel fluid fields are unavailable");
    }
    if (!system_->dofPermutation()) {
      throw std::runtime_error(
          "distributed native channel has no row ownership permutation");
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
    params_ = parseParameters(
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
)xml");

    sim_.primary_mesh = mesh_;
    sim_.fe_system = std::move(system_);
  }

  [[nodiscard]] Sample sample(svmp::FE::Real wet_fraction)
  {
    if (!(wet_fraction >= 0.0) || !(wet_fraction <= 1.0) ||
        !std::isfinite(wet_fraction)) {
      throw std::invalid_argument(
          "distributed native channel wet fraction is outside [0,1]");
    }
    const svmp::FE::Real exterior_offset =
        0.05 / static_cast<svmp::FE::Real>(upper_subdivisions_);
    const svmp::FE::Real interface_height =
        wet_fraction == 0.0
            ? -exterior_offset
            : wet_fraction == 1.0
                  ? window_height + exterior_offset
                  : wet_fraction;

    auto coefficients = vertexFieldCoefficients(
        level_set_,
        1u,
        [&](std::size_t vertex, std::size_t) {
          const auto y = vertexPoint(*mesh_, vertex)[1];
          return active_side_ == ActiveSide::Negative
                     ? y - interface_height
                     : interface_height - y;
        });
    writeFieldSlice(level_set_, coefficients, solution_);
    previous_ = solution_;

    const auto refresh_report =
        refreshActiveCutIntegrationContextFromSolution(
            sim_,
            *params_,
            solution_,
            lifecycle_,
            "native-manufactured-channel-mpi-test");
    if (!refresh_report.refreshed ||
        refresh_report.value_revision == 0u) {
      throw std::runtime_error(
          "distributed native channel refresh produced no revision");
    }
    const auto* context = sim_.fe_system->cutIntegrationContext();
    if (context == nullptr ||
        context->freeSurfaceGeometrySnapshots().size() != 1u) {
      throw std::runtime_error(
          "distributed native channel has no unique geometry snapshot");
    }
    const auto snapshot =
        context->freeSurfaceGeometrySnapshots().front();
    if (!snapshot) {
      throw std::runtime_error(
          "distributed native channel geometry snapshot is null");
    }

    Sample result;
    result.target_wet_fraction = wet_fraction;
    constexpr std::array<int, 3> physical_markers{{
        inlet_marker,
        outlet_marker,
        side_wall_marker,
    }};
    std::array<unsigned long long, 6> local_counts{};
    for (std::size_t role = 0u; role < physical_markers.size(); ++role) {
      const svmp::FE::interfaces::GeneratedActiveBoundaryDomain*
          selected = nullptr;
      for (const auto& active : snapshot->activeBoundaryDomains()) {
        if (active.request().boundary_marker == physical_markers[role] &&
            active.request().side == active_side_) {
          selected = &active;
        }
      }
      if (selected == nullptr) {
        throw std::runtime_error(
            "distributed native channel boundary partition is incomplete");
      }
      result.active_markers[role] = selected->marker();
      for (const auto& fragment : selected->fragments()) {
        if (fragment.owner_rank == rank_) {
          ++local_counts[role];
        }
      }
      for (const auto& rule :
           context->interfaceRulesForMarker(result.active_markers[role])) {
        if (rule != nullptr &&
            rule->provenance.owner_rank == rank_) {
          ++local_counts[3u + role];
        }
      }
    }
    std::array<unsigned long long, 6> global_counts{};
    MPI_Allreduce(local_counts.data(),
                  global_counts.data(),
                  static_cast<int>(global_counts.size()),
                  MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    for (std::size_t role = 0u; role < physical_markers.size(); ++role) {
      result.active_rule_counts[role] =
          static_cast<std::size_t>(global_counts[role]);
      result.retained_active_rule_counts[role] =
          static_cast<std::size_t>(global_counts[3u + role]);
    }

    std::array<svmp::FE::Real, 6> local_measures{};
    const auto selected_role =
        active_side_ == ActiveSide::Negative
            ? svmp::FE::interfaces::FreeSurfaceGeometryRuleRole::
                  NegativeExteriorBoundary
            : svmp::FE::interfaces::FreeSurfaceGeometryRuleRole::
                  PositiveExteriorBoundary;
    for (const auto& record : snapshot->rules()) {
      if (!record.locally_owned) {
        continue;
      }
      const auto marker = std::find(
          physical_markers.begin(),
          physical_markers.end(),
          record.physical_boundary_marker);
      if (marker == physical_markers.end()) {
        continue;
      }
      if (record.role !=
              svmp::FE::interfaces::FreeSurfaceGeometryRuleRole::
                  NegativeExteriorBoundary &&
          record.role !=
              svmp::FE::interfaces::FreeSurfaceGeometryRuleRole::
                  PositiveExteriorBoundary) {
        continue;
      }
      const auto role = static_cast<std::size_t>(
          std::distance(physical_markers.begin(), marker));
      local_measures[role] += record.physical_rule.physical_measure;
      if (record.role == selected_role) {
        local_measures[3u + role] +=
            record.physical_rule.physical_measure;
      }
    }
    std::array<svmp::FE::Real, 6> global_measures{};
    MPI_Allreduce(local_measures.data(),
                  global_measures.data(),
                  static_cast<int>(global_measures.size()),
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    for (std::size_t role = 0u; role < physical_markers.size(); ++role) {
      result.parent_measures[role] = global_measures[role];
      result.active_measures[role] = global_measures[3u + role];
    }

    const auto& definition =
        sim_.fe_system->operatorDefinition("equations");
    for (const auto& term : definition.boundary) {
      result.physical_role_boundary_term_count +=
          static_cast<std::size_t>(
              std::find(physical_markers.begin(),
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

    const auto records =
        sim_.fe_system->generatedBoundaryNitscheTraceCertificates();
    if (records.size() != 1u) {
      throw std::runtime_error(
          "distributed native channel requires one trace certificate");
    }
    const auto& trace = records.front();
    result.trace_patch_count = trace.certificate.certified_patch_count;
    result.trace_localized_support_patch_count =
        trace.certificate.localized_support_patch_count;
    result.trace_boundary_rule_count =
        trace.certificate.generated_boundary_rule_count;
    result.trace_maximum_support_overlap =
        trace.certificate.maximum_support_overlap;
    result.trace_certificate_digest =
        trace.certificate.canonical_certificate_digest;
    result.trace_global_conservative_upper_bound =
        trace.certificate.global_conservative_upper_bound;
    result.trace_grouped_symmetric_ratio =
        trace.grouped_symmetric_trace_to_penalty_ratio;

    // Boundary-fragment stable IDs bind the current ownership revision and
    // face numbering, so they are not cross-repartition identities. Gather a
    // physical key from each owning rank instead: the canonical parent-cell
    // GID and its cell-local facet index.
    std::vector<std::uint64_t> local_boundary_key_words;
    const auto& mesh_access = sim_.fe_system->meshAccess();
    for (const auto* rule :
         context->interfaceRulesForMarker(result.active_markers[2])) {
      if (rule == nullptr ||
          rule->provenance.owner_rank != rank_) {
        continue;
      }
      const auto local_cell = static_cast<svmp::FE::GlobalIndex>(
          rule->provenance.parent_entity);
      const auto local_face = static_cast<svmp::FE::GlobalIndex>(
          rule->provenance.parent_boundary_entity);
      const auto local_facet =
          mesh_access.getLocalFaceIndex(local_face, local_cell);
      if (rule->provenance.cut_topology_revision == 0u ||
          rule->provenance.parent_entity_global_id < 0 ||
          local_facet < 0) {
        throw std::runtime_error(
            "distributed native channel boundary rule lacks a physical key");
      }
      local_boundary_key_words.push_back(
          rule->provenance.cut_topology_revision);
      local_boundary_key_words.push_back(static_cast<std::uint64_t>(
          rule->provenance.parent_entity_global_id));
      local_boundary_key_words.push_back(
          static_cast<std::uint64_t>(local_facet));
    }
    if (local_boundary_key_words.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error(
          "distributed native channel boundary-key payload is too large");
    }
    const int local_boundary_word_count =
        static_cast<int>(local_boundary_key_words.size());
    std::vector<int> boundary_word_counts(
        static_cast<std::size_t>(size_), 0);
    MPI_Allgather(&local_boundary_word_count,
                  1,
                  MPI_INT,
                  boundary_word_counts.data(),
                  1,
                  MPI_INT,
                  MPI_COMM_WORLD);
    std::vector<int> boundary_word_offsets(
        static_cast<std::size_t>(size_), 0);
    std::size_t total_boundary_word_count = 0u;
    for (int peer = 0; peer < size_; ++peer) {
      const auto peer_index = static_cast<std::size_t>(peer);
      if (boundary_word_counts[peer_index] < 0 ||
          boundary_word_counts[peer_index] % 3 != 0 ||
          total_boundary_word_count >
              static_cast<std::size_t>(std::numeric_limits<int>::max()) -
                  static_cast<std::size_t>(
                      boundary_word_counts[peer_index])) {
        throw std::runtime_error(
            "distributed native channel boundary-key counts are invalid");
      }
      boundary_word_offsets[peer_index] =
          static_cast<int>(total_boundary_word_count);
      total_boundary_word_count += static_cast<std::size_t>(
          boundary_word_counts[peer_index]);
    }
    if (total_boundary_word_count !=
        3u * trace.certificate.generated_boundary_rule_count) {
      throw std::runtime_error(
          "distributed native channel boundary-key count is incomplete");
    }
    std::vector<std::uint64_t> global_boundary_key_words(
        total_boundary_word_count, 0u);
    MPI_Allgatherv(local_boundary_key_words.data(),
                   local_boundary_word_count,
                   MPI_UINT64_T,
                   global_boundary_key_words.data(),
                   boundary_word_counts.data(),
                   boundary_word_offsets.data(),
                   MPI_UINT64_T,
                   MPI_COMM_WORLD);
    std::unordered_map<
        std::uint64_t,
        std::array<svmp::FE::GlobalIndex, 2>>
        physical_key_by_stable_id;
    physical_key_by_stable_id.reserve(
        trace.certificate.generated_boundary_rule_count);
    for (std::size_t word = 0u;
         word < global_boundary_key_words.size();
         word += 3u) {
      const auto stable_id = global_boundary_key_words[word];
      const auto cell_gid = global_boundary_key_words[word + 1u];
      const auto local_facet = global_boundary_key_words[word + 2u];
      if (stable_id == 0u ||
          cell_gid > static_cast<std::uint64_t>(
                         std::numeric_limits<svmp::FE::GlobalIndex>::max()) ||
          local_facet > static_cast<std::uint64_t>(
                            std::numeric_limits<svmp::FE::GlobalIndex>::max()) ||
          !physical_key_by_stable_id
               .emplace(
                   stable_id,
                   std::array<svmp::FE::GlobalIndex, 2>{{
                       static_cast<svmp::FE::GlobalIndex>(cell_gid),
                       static_cast<svmp::FE::GlobalIndex>(local_facet)}})
               .second) {
        throw std::runtime_error(
            "distributed native channel boundary-key map is invalid");
      }
    }
    for (const auto& patch : trace.certificate.patches) {
      const auto& exact = patch.generalized_bound.exact_dyadic;
      std::vector<std::array<svmp::FE::GlobalIndex, 2>>
          boundary_rule_physical_keys;
      boundary_rule_physical_keys.reserve(
          patch.boundary_rule_stable_ids.size());
      for (const auto stable_id : patch.boundary_rule_stable_ids) {
        const auto found = physical_key_by_stable_id.find(stable_id);
        if (found == physical_key_by_stable_id.end()) {
          throw std::runtime_error(
              "distributed native channel trace patch lacks a physical boundary key");
        }
        boundary_rule_physical_keys.push_back(found->second);
      }
      std::sort(boundary_rule_physical_keys.begin(),
                boundary_rule_physical_keys.end());
      if (std::adjacent_find(boundary_rule_physical_keys.begin(),
                             boundary_rule_physical_keys.end()) !=
          boundary_rule_physical_keys.end()) {
        throw std::runtime_error(
            "distributed native channel trace patch repeats a physical boundary key");
      }
      result.trace_partition_invariant_patches.push_back(
          TracePatchEvidence{
              .localized_support_patch = patch.localized_support_patch,
              .root_cell_gid = patch.root_cell_gid,
              .support_cell_gids = patch.support_cell_gids,
              .boundary_rule_physical_keys =
                  std::move(boundary_rule_physical_keys),
              .boundary_rule_count = patch.boundary_rule_stable_ids.size(),
              .raw_support_dof_count = patch.raw_support_dof_count,
              .terminal_tangent_dof_count =
                  patch.terminal_tangent_dof_count,
              .rigid_mode_candidate_count =
                  patch.rigid_mode_candidate_count,
              .structural_rigid_mode_count =
                  patch.structural_rigid_mode_count,
              .rigid_mode_constraint_rank =
                  patch.rigid_mode_constraint_rank,
              .maximum_cell_support_overlap =
                  patch.maximum_cell_support_overlap,
              .retained_support_physical_volume =
                  patch.retained_support_physical_volume,
              .generated_boundary_physical_measure =
                  patch.generated_boundary_physical_measure,
              .directly_proven_upper_bound =
                  exact.directly_proven_upper_bound,
              .rigid_mode_quotient_status =
                  patch.rigid_mode_quotient_status,
              .proof_input = exact.proof_input,
              .exact_rigid_factor_action_proven =
                  patch.exact_rigid_factor_action_proven,
              .denominator_positive_definite_proven =
                  exact.denominator_positive_definite_proven,
              .numerator_positive_semidefinite_proven =
                  exact.numerator_positive_semidefinite_proven,
              .upper_inequality_proven = exact.upper_inequality_proven,
              .exact_factorized_materialization_proven =
                  exact.exact_factorized_materialization_proven,
              .exact_sparse_map_applied = exact.exact_sparse_map_applied,
              .exact_common_kernel_proven =
                  exact.exact_common_kernel_proven,
              .exact_dimension = exact.dimension,
              .denominator_rank = exact.denominator_rank,
              .numerator_rank = exact.numerator_rank,
              .numerator_gram_block_count =
                  exact.numerator_gram_block_count,
              .denominator_gram_block_count =
                  exact.denominator_gram_block_count,
              .numerator_gram_row_count =
                  exact.numerator_gram_row_count,
              .denominator_gram_row_count =
                  exact.denominator_gram_row_count,
              .numerator_weight_term_count =
                  exact.numerator_weight_term_count,
              .denominator_weight_term_count =
                  exact.denominator_weight_term_count,
              .factorized_input_dimension =
                  exact.factorized_input_dimension,
              .exact_common_kernel_nullity =
                  exact.exact_common_kernel_nullity});
      result.trace_maximum_factorized_input_dimension =
          std::max(
              result.trace_maximum_factorized_input_dimension,
              patch.generalized_bound.exact_dyadic
                  .factorized_input_dimension);
    }
    std::sort(
        result.trace_partition_invariant_patches.begin(),
        result.trace_partition_invariant_patches.end(),
        [](const auto& left, const auto& right) {
          if (left.root_cell_gid != right.root_cell_gid) {
            return left.root_cell_gid < right.root_cell_gid;
          }
          if (left.support_cell_gids != right.support_cell_gids) {
            return left.support_cell_gids < right.support_cell_gids;
          }
          if (left.boundary_rule_physical_keys !=
              right.boundary_rule_physical_keys) {
            return left.boundary_rule_physical_keys <
                   right.boundary_rule_physical_keys;
          }
          if (left.localized_support_patch !=
              right.localized_support_patch) {
            return left.localized_support_patch <
                   right.localized_support_patch;
          }
          return left.boundary_rule_count < right.boundary_rule_count;
        });
    result.trace_revision_match =
        trace.policy.physical_boundary_marker == side_wall_marker &&
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
        std::all_of(
            trace.certificate.patches.begin(),
            trace.certificate.patches.end(),
            [](const auto& patch) {
              const auto& exact =
                  patch.generalized_bound.exact_dyadic;
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
          "distributed native channel assembly failed: " +
          assembly.error_message);
    }

    std::array<svmp::FE::Real, 3> local_work{};
    const auto permutation = sim_.fe_system->dofPermutation();
    if (!permutation ||
        permutation->forward.size() !=
            static_cast<std::size_t>(dof_count) ||
        permutation->owner_rank.size() !=
            static_cast<std::size_t>(dof_count)) {
      throw std::runtime_error(
          "distributed native channel row ownership is incomplete");
    }
    for (svmp::FE::GlobalIndex row = 0; row < dof_count; ++row) {
      const auto index = static_cast<std::size_t>(row);
      const auto backend_row = permutation->forward[index];
      if (backend_row < 0 ||
          static_cast<std::size_t>(backend_row) >=
              permutation->owner_rank.size()) {
        throw std::runtime_error(
            "distributed native channel row permutation is invalid");
      }
      if (permutation->owner_rank[
              static_cast<std::size_t>(backend_row)] != rank_) {
        continue;
      }
      for (std::size_t role = 0u; role < probes_.size(); ++role) {
        local_work[role] += residual[row] * probes_[role][index];
      }
    }
    MPI_Allreduce(local_work.data(),
                  result.operator_work.data(),
                  static_cast<int>(local_work.size()),
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    return result;
  }

  [[nodiscard]] std::vector<int> partitionOwners() const
  {
    const auto global_count =
        makeChannelArrays(upper_subdivisions_).shapes.size();
    std::vector<int> local(global_count, -1);
    const auto& gids = mesh_->local_mesh().cell_gids();
    for (std::size_t cell = 0u; cell < mesh_->n_cells(); ++cell) {
      if (!mesh_->is_owned_cell(static_cast<svmp::index_t>(cell))) {
        continue;
      }
      const auto gid = gids[cell];
      if (gid < 0 || static_cast<std::size_t>(gid) >= local.size()) {
        throw std::runtime_error(
            "distributed native channel cell GID is outside the fixture");
      }
      local[static_cast<std::size_t>(gid)] = rank_;
    }
    std::vector<int> global(global_count, -1);
    MPI_Allreduce(local.data(),
                  global.data(),
                  static_cast<int>(global.size()),
                  MPI_INT,
                  MPI_MAX,
                  MPI_COMM_WORLD);
    if (std::find(global.begin(), global.end(), -1) != global.end()) {
      throw std::runtime_error(
          "distributed native channel partition omitted a global cell");
    }
    return global;
  }

  [[nodiscard]] static constexpr svmp::FE::Real
  expectedFullForceWork() noexcept
  {
    return -inlet_traction * window_height * channel_depth;
  }

  [[nodiscard]] static constexpr svmp::FE::Real
  expectedFullFluxWork() noexcept
  {
    return outlet_pressure * window_height * channel_depth;
  }

  [[nodiscard]] static constexpr svmp::FE::Real
  expectedFullPenaltyWork() noexcept
  {
    return -nitsche_gamma * viscosity / side_facet_normal_scale *
           prescribed_side_velocity * channel_length * window_height;
  }

private:
  template <class Value>
  [[nodiscard]] std::vector<svmp::FE::Real>
  vertexFieldCoefficients(
      svmp::FE::FieldId field,
      std::size_t components,
      Value&& value) const
  {
    const auto& dofs = sim_.fe_system
                           ? sim_.fe_system->fieldDofHandler(field)
                           : system_->fieldDofHandler(field);
    const auto* entity_map = dofs.getEntityDofMap();
    if (entity_map == nullptr) {
      throw std::runtime_error(
          "distributed native channel field has no entity DOF map");
    }
    std::vector<svmp::FE::Real> coefficients(
        static_cast<std::size_t>(dofs.getNumDofs()), 0.0);
    for (std::size_t vertex = 0u; vertex < mesh_->n_vertices(); ++vertex) {
      const auto vertex_dofs = entity_map->getVertexDofs(
          static_cast<svmp::FE::GlobalIndex>(vertex));
      if (vertex_dofs.size() != components) {
        throw std::runtime_error(
            "distributed native channel field has unexpected vertex DOFs");
      }
      for (std::size_t component = 0u;
           component < components;
           ++component) {
        const auto dof = vertex_dofs[component];
        if (dof < 0 ||
            static_cast<std::size_t>(dof) >= coefficients.size()) {
          throw std::runtime_error(
              "distributed native channel field DOF is invalid");
        }
        if (dofs.getDofMap().isOwnedDof(dof)) {
          coefficients[static_cast<std::size_t>(dof)] =
              value(vertex, component);
        }
      }
    }
    allreduceInPlace(coefficients, MPI_COMM_WORLD);
    return coefficients;
  }

  void writeFieldSlice(
      svmp::FE::FieldId field,
      std::span<const svmp::FE::Real> coefficients,
      std::vector<svmp::FE::Real>& solution) const
  {
    const auto& system = sim_.fe_system ? *sim_.fe_system : *system_;
    const auto offset = system.fieldDofOffset(field);
    if (offset < 0 ||
        static_cast<std::size_t>(offset) + coefficients.size() >
            solution.size()) {
      throw std::runtime_error(
          "distributed native channel field slice is outside the solution");
    }
    std::copy(coefficients.begin(),
              coefficients.end(),
              solution.begin() + offset);
  }

  [[nodiscard]] std::vector<svmp::FE::Real>
  constantVelocityProbe(
      const std::array<svmp::FE::Real, 3>& value) const
  {
    const auto coefficients = vertexFieldCoefficients(
        velocity_,
        value.size(),
        [&](std::size_t, std::size_t component) {
          return value[component];
        });
    const auto& system = *system_;
    std::vector<svmp::FE::Real> probe(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()),
        0.0);
    const auto offset = system.fieldDofOffset(velocity_);
    if (offset < 0 ||
        static_cast<std::size_t>(offset) + coefficients.size() >
            probe.size()) {
      throw std::runtime_error(
          "distributed native channel probe slice is invalid");
    }
    std::copy(coefficients.begin(),
              coefficients.end(),
              probe.begin() + offset);
    return probe;
  }

  ActiveSide active_side_{ActiveSide::Negative};
  int upper_subdivisions_{2};
  int rank_{0};
  int size_{1};
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

} // namespace native_manufactured_channel_mpi

#endif
