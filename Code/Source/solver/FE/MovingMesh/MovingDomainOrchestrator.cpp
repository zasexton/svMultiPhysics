/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "MovingMesh/MovingDomainOrchestrator.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#include "Core/FEException.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Motion/MotionQuality.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace svmp::FE::moving_mesh {
namespace {

std::string normalized(std::string_view value)
{
    std::string out(value);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        if (c == '-' || c == ' ') {
            return '_';
        }
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

bool parse_bool(std::string_view value)
{
    const auto v = normalized(value);
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

std::vector<std::string> split_list(std::string_view value)
{
    std::vector<std::string> out;
    std::string current;
    for (const char c : value) {
        if (c == ',' || c == ';' || c == '|') {
            if (!current.empty()) {
                out.push_back(current);
                current.clear();
            }
        } else if (!std::isspace(static_cast<unsigned char>(c))) {
            current.push_back(c);
        }
    }
    if (!current.empty()) {
        out.push_back(current);
    }
    return out;
}

std::array<real_t, 3> parse_triplet(std::string_view value)
{
    const auto parts = split_list(value);
    FE_THROW_IF(parts.empty() || parts.size() > 3, InvalidArgumentException,
                "moving-domain config: expected one to three numeric values");

    std::array<real_t, 3> out{{0.0, 0.0, 0.0}};
    for (std::size_t i = 0; i < parts.size(); ++i) {
        out[i] = static_cast<real_t>(std::stod(parts[i]));
    }
    return out;
}

std::array<bool, 3> parse_component_mask(std::string_view value)
{
    const auto v = normalized(value);
    if (v.empty() || v == "all" || v == "xyz" || v == "0,1,2") {
        return {{true, true, true}};
    }
    if (v == "xy") {
        return {{true, true, false}};
    }
    if (v == "xz") {
        return {{true, false, true}};
    }
    if (v == "yz") {
        return {{false, true, true}};
    }
    if (v == "none") {
        return {{false, false, false}};
    }

    std::array<bool, 3> mask{{false, false, false}};
    const auto parts = split_list(value);
    if (parts.size() == 1u && normalized(parts.front()).size() > 1u) {
        for (const char c : normalized(parts.front())) {
            if (c == 'x' || c == '0') {
                mask[0] = true;
            } else if (c == 'y' || c == '1') {
                mask[1] = true;
            } else if (c == 'z' || c == '2') {
                mask[2] = true;
            } else {
                throw std::invalid_argument("moving-domain config: unknown boundary component '" +
                                            std::string(value) + "'");
            }
        }
    } else {
        for (const auto& part : parts) {
            const auto p = normalized(part);
            if (p == "x" || p == "0") {
                mask[0] = true;
            } else if (p == "y" || p == "1") {
                mask[1] = true;
            } else if (p == "z" || p == "2") {
                mask[2] = true;
            } else if (p == "all" || p == "xyz") {
                mask = {{true, true, true}};
            } else {
                throw std::invalid_argument("moving-domain config: unknown boundary component '" + part + "'");
            }
        }
    }
    return mask;
}

BoundaryMotionValueMode parse_boundary_motion_value_mode(std::string_view value)
{
    const auto v = normalized(value);
    if (v.empty() || v == "constant_displacement" || v == "displacement") {
        return BoundaryMotionValueMode::ConstantDisplacement;
    }
    if (v == "constant_velocity" || v == "velocity") {
        return BoundaryMotionValueMode::ConstantVelocity;
    }
    if (v == "function") {
        return BoundaryMotionValueMode::Function;
    }
    throw std::invalid_argument("moving-domain config: unknown boundary motion value mode '" +
                                std::string(value) + "'");
}

GeometryRegularizationModel parse_geometry_regularization_model(std::string_view value)
{
    const auto v = normalized(value);
    if (v.empty() || v == "harmonic" || v == "laplace") {
        return GeometryRegularizationModel::Harmonic;
    }
    if (v == "artificial_pseudo_elastic" || v == "pseudo_elastic" ||
        v == "artificial_elastic") {
        return GeometryRegularizationModel::ArtificialPseudoElastic;
    }
    throw std::invalid_argument("moving-domain config: unknown geometry regularization model '" +
                                std::string(value) + "'");
}

GeometryRegularizationWeightMode parse_geometry_regularization_weight_mode(std::string_view value)
{
    const auto v = normalized(value);
    if (v.empty() || v == "uniform") {
        return GeometryRegularizationWeightMode::Uniform;
    }
    if (v == "element_size" || v == "size") {
        return GeometryRegularizationWeightMode::ElementSize;
    }
    if (v == "boundary_distance" || v == "distance") {
        return GeometryRegularizationWeightMode::BoundaryDistance;
    }
    if (v == "vertex_field" || v == "field") {
        return GeometryRegularizationWeightMode::VertexField;
    }
    throw std::invalid_argument("moving-domain config: unknown geometry regularization weight mode '" +
                                std::string(value) + "'");
}

GeometryConstraintMode parse_geometry_constraint_mode(std::string_view value)
{
    const auto v = normalized(value);
    if (v.empty() || v == "strong_dirichlet" || v == "dirichlet" || v == "strong") {
        return GeometryConstraintMode::StrongDirichlet;
    }
    throw std::invalid_argument("moving-domain config: unknown geometry constraint mode '" +
                                std::string(value) + "'");
}

MotionBackendModel resolved_backend_model(const MovingDomainConfig& config)
{
    if (config.backend_model != MotionBackendModel::None) {
        return config.backend_model;
    }
    if (config.mode == MovingMeshMode::PrescribedMotion) {
        return MotionBackendModel::Prescribed;
    }
    if (config.mode == MovingMeshMode::FEBackedSmoothing) {
        return MotionBackendModel::FEGeometryRegularization;
    }
    return MotionBackendModel::None;
}

double configured_step_scale(const motion::MotionConfig& config)
{
    double step_scale = config.max_step_scale;
    if (!(step_scale > 0.0) || !std::isfinite(step_scale)) {
        step_scale = 1.0;
    }
    return std::min(step_scale, 1.0);
}

class PrescribedMotionBackend final : public motion::IMotionBackend {
public:
    [[nodiscard]] const char* name() const noexcept override
    {
        return "FE::MovingMesh::PrescribedMotionBackend";
    }

    motion::MotionSolveResult solve(const motion::MotionSolveRequest& request) override
    {
        motion::MotionSolveResult result;
        if (!request.displacement.valid()) {
            result.success = false;
            result.message = "prescribed mesh motion requires a valid displacement field";
            return result;
        }
        if (!request.dirichlet_bcs || request.dirichlet_bcs->empty()) {
            result.success = false;
            result.message = "prescribed mesh motion requires at least one boundary motion entry";
            return result;
        }

        auto& mesh = request.mesh.local_mesh();
        const int dim = mesh.dim();
        if (dim <= 0) {
            result.success = false;
            result.message = "prescribed mesh motion requires a positive mesh dimension";
            return result;
        }

        const auto& x = (request.geometry_config == Configuration::Current && mesh.has_current_coords())
            ? mesh.X_cur()
            : mesh.X_ref();

        std::fill(request.displacement.data,
                  request.displacement.data +
                      request.displacement.n_entities * request.displacement.components,
                  real_t(0));
        if (request.velocity.valid()) {
            std::fill(request.velocity.data,
                      request.velocity.data +
                          request.velocity.n_entities * request.velocity.components,
                      real_t(0));
        }

        std::vector<unsigned char> touched(request.displacement.n_entities, 0);
        for (const auto& bc : *request.dirichlet_bcs) {
            const auto faces = (bc.boundary_label == INVALID_LABEL)
                ? mesh.boundary_faces()
                : mesh.faces_with_label(bc.boundary_label);
            if (faces.empty()) {
                std::ostringstream os;
                os << "prescribed mesh motion boundary label " << bc.boundary_label
                   << " matched no boundary faces";
                result.success = false;
                result.message = os.str();
                return result;
            }

            for (const index_t face : faces) {
                auto [verts, n_face_vertices] = mesh.face_vertices_span(face);
                for (std::size_t fv = 0; fv < n_face_vertices; ++fv) {
                    const auto vertex = static_cast<std::size_t>(verts[fv]);
                    if (vertex >= request.displacement.n_entities) {
                        result.success = false;
                        result.message = "prescribed mesh motion references an invalid boundary vertex";
                        return result;
                    }

                    std::array<real_t, 3> xyz{{0.0, 0.0, 0.0}};
                    const auto base = vertex * static_cast<std::size_t>(dim);
                    for (int d = 0; d < dim; ++d) {
                        xyz[static_cast<std::size_t>(d)] =
                            x[base + static_cast<std::size_t>(d)];
                    }

                    const auto value = bc.value
                        ? bc.value(xyz, request.dt, request.step_scale)
                        : std::array<real_t, 3>{{0.0, 0.0, 0.0}};
                    const auto disp_base = vertex * request.displacement.components;
                    const auto vel_base = vertex * request.velocity.components;
                    for (int d = 0; d < dim; ++d) {
                        const auto di = static_cast<std::size_t>(d);
                        if (!bc.component_mask[di]) {
                            continue;
                        }
                        const real_t next = value[di];
                        if (!std::isfinite(next)) {
                            result.success = false;
                            result.message = "prescribed mesh motion produced a non-finite displacement";
                            return result;
                        }
                        request.displacement.data[disp_base + di] = next;
                        if (request.velocity.valid() && request.dt > 0.0 && request.step_scale > 0.0 &&
                            request.velocity.components >= request.displacement.components) {
                            request.velocity.data[vel_base + di] =
                                next / static_cast<real_t>(request.dt * request.step_scale);
                        }
                    }
                    touched[vertex] = 1;
                }
            }
        }

        result.success = true;
        result.wrote_velocity = request.velocity.valid() && request.dt > 0.0;
        result.message = "prescribed mesh motion applied to " +
                         std::to_string(std::count(touched.begin(), touched.end(), 1)) +
                         " boundary vertices";
        return result;
    }
};

motion::MotionDirichletBC make_motion_bc(const BoundaryMotionConfig& config)
{
    motion::MotionDirichletBC bc;
    bc.boundary_label = config.boundary_label;
    bc.component_mask = config.component_mask;
    if (config.value_mode == BoundaryMotionValueMode::Function) {
        bc.value = config.function;
    } else if (config.value_mode == BoundaryMotionValueMode::ConstantVelocity) {
        const auto v = config.value;
        bc.value = [v](const std::array<real_t, 3>&, double dt, double step_scale) {
            return std::array<real_t, 3>{{
                static_cast<real_t>(v[0] * dt * step_scale),
                static_cast<real_t>(v[1] * dt * step_scale),
                static_cast<real_t>(v[2] * dt * step_scale),
            }};
        };
    } else {
        const auto u = config.value;
        bc.value = [u](const std::array<real_t, 3>&, double, double step_scale) {
            return std::array<real_t, 3>{{
                static_cast<real_t>(u[0] * step_scale),
                static_cast<real_t>(u[1] * step_scale),
                static_cast<real_t>(u[2] * step_scale),
            }};
        };
    }
    return bc;
}

void fill_quality_diagnostics(MovingDomainDiagnostics& diagnostics, const Mesh& mesh)
{
    if (!mesh.local_mesh().has_current_coords()) {
        return;
    }
    const auto quality = motion::evaluate_motion_quality(mesh, Configuration::Current);
    diagnostics.minimum_quality_jacobian = quality.min_jacobian;
    diagnostics.minimum_quality_angle_degrees = quality.min_angle_deg;
    diagnostics.maximum_quality_skewness = quality.max_skewness;
    diagnostics.has_inverted_cells = quality.has_inverted_cells;
}

} // namespace

const char* to_string(MovingMeshMode mode) noexcept
{
    switch (mode) {
        case MovingMeshMode::Disabled: return "disabled";
        case MovingMeshMode::PrescribedMotion: return "prescribed_motion";
        case MovingMeshMode::FEBackedSmoothing: return "fe_backed_smoothing";
        case MovingMeshMode::CoupledMonolithic: return "coupled_monolithic";
    }
    return "unknown";
}

const char* to_string(MotionBackendModel model) noexcept
{
    switch (model) {
        case MotionBackendModel::None: return "none";
        case MotionBackendModel::Prescribed: return "prescribed";
        case MotionBackendModel::FEGeometryRegularization: return "fe_geometry_regularization";
        case MotionBackendModel::External: return "external";
    }
    return "unknown";
}

const char* to_string(BoundaryMotionValueMode mode) noexcept
{
    switch (mode) {
        case BoundaryMotionValueMode::ConstantDisplacement: return "constant_displacement";
        case BoundaryMotionValueMode::ConstantVelocity: return "constant_velocity";
        case BoundaryMotionValueMode::Function: return "function";
    }
    return "unknown";
}

const char* to_string(MovingDomainAdvancePoint point) noexcept
{
    switch (point) {
        case MovingDomainAdvancePoint::BeforePhysicsSolve: return "before_physics_solve";
        case MovingDomainAdvancePoint::BeforeNonlinearIteration: return "before_nonlinear_iteration";
        case MovingDomainAdvancePoint::AfterAcceptedNonlinearState: return "after_accepted_nonlinear_state";
        case MovingDomainAdvancePoint::BeforeRemesh: return "before_remesh";
        case MovingDomainAdvancePoint::BeforeCheckpoint: return "before_checkpoint";
        case MovingDomainAdvancePoint::BeforeOutput: return "before_output";
    }
    return "unknown";
}

MovingMeshMode parse_moving_mesh_mode(std::string_view value)
{
    const auto v = normalized(value);
    if (v.empty() || v == "disabled" || v == "off" || v == "none" || v == "static") {
        return MovingMeshMode::Disabled;
    }
    if (v == "prescribed" || v == "prescribed_motion") {
        return MovingMeshMode::PrescribedMotion;
    }
    if (v == "fe_backed_smoothing" || v == "fe_smoothing" || v == "smoothing") {
        return MovingMeshMode::FEBackedSmoothing;
    }
    if (v == "coupled_monolithic" || v == "monolithic") {
        return MovingMeshMode::CoupledMonolithic;
    }
    throw std::invalid_argument("parse_moving_mesh_mode: unknown mode '" + std::string(value) + "'");
}

MotionBackendModel parse_motion_backend_model(std::string_view value)
{
    const auto v = normalized(value);
    if (v.empty() || v == "none" || v == "disabled") {
        return MotionBackendModel::None;
    }
    if (v == "prescribed" || v == "prescribed_motion") {
        return MotionBackendModel::Prescribed;
    }
    if (v == "fe_geometry_regularization" || v == "geometry_regularization" || v == "fe_smoothing") {
        return MotionBackendModel::FEGeometryRegularization;
    }
    if (v == "external" || v == "injected") {
        return MotionBackendModel::External;
    }
    throw std::invalid_argument("parse_motion_backend_model: unknown backend model '" + std::string(value) + "'");
}

Configuration parse_coordinate_configuration(std::string_view value)
{
    const auto v = normalized(value);
    if (v.empty() || v == "reference" || v == "ref") {
        return Configuration::Reference;
    }
    if (v == "current" || v == "deformed") {
        return Configuration::Current;
    }
    throw std::invalid_argument("parse_coordinate_configuration: unknown coordinate configuration '" +
                                std::string(value) + "'");
}

MovingDomainConfig moving_domain_config_from_kv(
    const std::unordered_map<std::string, std::string>& kv)
{
    MovingDomainConfig config;
    if (auto it = kv.find("mesh_motion.mode"); it != kv.end()) {
        config.mode = parse_moving_mesh_mode(it->second);
    }
    if (auto it = kv.find("mesh_motion.coordinate_configuration"); it != kv.end()) {
        config.fe_coordinate_configuration = parse_coordinate_configuration(it->second);
    }
    if (auto it = kv.find("mesh_motion.backend"); it != kv.end()) {
        config.backend_model = parse_motion_backend_model(it->second);
    }
    if (auto it = kv.find("mesh_motion.notify_fe_systems"); it != kv.end()) {
        config.notify_fe_systems = parse_bool(it->second);
    }
    if (auto it = kv.find("mesh_motion.exchange_ghost_coordinates"); it != kv.end()) {
        config.exchange_ghost_coordinates = parse_bool(it->second);
    }
    if (auto it = kv.find("mesh_motion.max_step_scale"); it != kv.end()) {
        config.motion.max_step_scale = std::stod(it->second);
    }
    if (auto it = kv.find("mesh_motion.max_substeps"); it != kv.end()) {
        config.motion.max_substeps = std::stoi(it->second);
    }
    if (auto it = kv.find("mesh_motion.enable_quality_guard"); it != kv.end()) {
        config.motion.enable_quality_guard = parse_bool(it->second);
    }
    if (auto it = kv.find("mesh_motion.enforce_quality_thresholds"); it != kv.end()) {
        config.motion.enforce_quality_thresholds = parse_bool(it->second);
    }
    if (auto it = kv.find("mesh_motion.quality_min_jacobian"); it != kv.end()) {
        config.motion.quality_min_jacobian = std::stod(it->second);
    }
    if (auto it = kv.find("mesh_motion.quality_min_angle_degrees"); it != kv.end()) {
        config.motion.quality_min_angle_deg = std::stod(it->second);
    }
    if (auto it = kv.find("mesh_motion.quality_max_skewness"); it != kv.end()) {
        config.motion.quality_max_skewness = std::stod(it->second);
    }

    if (auto it = kv.find("mesh_motion.geometry_regularization.model"); it != kv.end()) {
        config.geometry_regularization.model = parse_geometry_regularization_model(it->second);
    }
    if (auto it = kv.find("mesh_motion.geometry_regularization.weight_mode"); it != kv.end()) {
        config.geometry_regularization.weight_mode = parse_geometry_regularization_weight_mode(it->second);
    }
    if (auto it = kv.find("mesh_motion.geometry_regularization.constraint_mode"); it != kv.end()) {
        config.geometry_regularization.constraint_mode = parse_geometry_constraint_mode(it->second);
    }
    if (auto it = kv.find("mesh_motion.geometry_regularization.max_linear_iterations"); it != kv.end()) {
        config.geometry_regularization.max_linear_iterations = std::stoi(it->second);
    }
    if (auto it = kv.find("mesh_motion.geometry_regularization.relative_tolerance"); it != kv.end()) {
        config.geometry_regularization.relative_tolerance = static_cast<Real>(std::stod(it->second));
    }
    if (auto it = kv.find("mesh_motion.geometry_regularization.absolute_tolerance"); it != kv.end()) {
        config.geometry_regularization.absolute_tolerance = static_cast<Real>(std::stod(it->second));
    }
    if (auto it = kv.find("mesh_motion.geometry_regularization.constraint_tolerance"); it != kv.end()) {
        config.geometry_regularization.constraint_tolerance = static_cast<Real>(std::stod(it->second));
    }
    if (auto it = kv.find("mesh_motion.geometry_regularization.artificial_stiffness_scale"); it != kv.end()) {
        config.geometry_regularization.artificial_stiffness_scale = static_cast<Real>(std::stod(it->second));
    }
    if (auto it = kv.find("mesh_motion.geometry_regularization.artificial_pseudo_elastic_blend"); it != kv.end()) {
        config.geometry_regularization.artificial_pseudo_elastic_blend = static_cast<Real>(std::stod(it->second));
    }
    if (auto it = kv.find("mesh_motion.geometry_regularization.boundary_distance_floor"); it != kv.end()) {
        config.geometry_regularization.boundary_distance_floor = static_cast<Real>(std::stod(it->second));
    }
    if (auto it = kv.find("mesh_motion.geometry_regularization.minimum_weight"); it != kv.end()) {
        config.geometry_regularization.minimum_weight = static_cast<Real>(std::stod(it->second));
    }
    if (auto it = kv.find("mesh_motion.geometry_regularization.vertex_weight_field"); it != kv.end()) {
        config.geometry_regularization.vertex_weight_field = it->second;
    }
    if (auto it = kv.find("mesh_motion.geometry_regularization.write_velocity"); it != kv.end()) {
        config.geometry_regularization.write_velocity = parse_bool(it->second);
    }

    std::size_t boundary_count = 0;
    if (auto it = kv.find("mesh_motion.boundary.count"); it != kv.end()) {
        boundary_count = static_cast<std::size_t>(std::stoul(it->second));
    }
    for (std::size_t i = 0; i < boundary_count; ++i) {
        const std::string prefix = "mesh_motion.boundary." + std::to_string(i) + ".";
        BoundaryMotionConfig bc;
        if (auto it = kv.find(prefix + "label"); it != kv.end()) {
            bc.boundary_label = static_cast<label_t>(std::stoll(it->second));
        }
        if (auto it = kv.find(prefix + "components"); it != kv.end()) {
            bc.component_mask = parse_component_mask(it->second);
        }
        if (auto it = kv.find(prefix + "mode"); it != kv.end()) {
            bc.value_mode = parse_boundary_motion_value_mode(it->second);
        }
        if (auto it = kv.find(prefix + "value"); it != kv.end()) {
            bc.value = parse_triplet(it->second);
        }
        config.boundary_motion.push_back(std::move(bc));
    }
    return config;
}

std::vector<motion::MotionDirichletBC>
make_motion_dirichlet_bcs(const std::vector<BoundaryMotionConfig>& configs)
{
    std::vector<motion::MotionDirichletBC> out;
    out.reserve(configs.size());
    for (const auto& config : configs) {
        out.push_back(make_motion_bc(config));
    }
    return out;
}

MovingDomainOrchestrator::MovingDomainOrchestrator(std::shared_ptr<Mesh> mesh,
                                                   MovingDomainConfig config)
    : mesh_(std::move(mesh))
    , config_(std::move(config))
{
    FE_CHECK_NOT_NULL(mesh_.get(), "MovingDomainOrchestrator: mesh");
}

void MovingDomainOrchestrator::configure(MovingDomainConfig config)
{
    config_ = std::move(config);
}

void MovingDomainOrchestrator::setExternalBackend(std::shared_ptr<motion::IMotionBackend> backend)
{
    external_backend_ = std::move(backend);
}

std::shared_ptr<motion::IMotionBackend> MovingDomainOrchestrator::buildBackend()
{
    const MotionBackendModel model = resolved_backend_model(config_);

    switch (model) {
        case MotionBackendModel::None:
            return nullptr;
        case MotionBackendModel::Prescribed:
            return std::make_shared<PrescribedMotionBackend>();
        case MotionBackendModel::FEGeometryRegularization:
            return make_geometry_regularization_motion_backend(config_.geometry_regularization);
        case MotionBackendModel::External:
            FE_CHECK_NOT_NULL(external_backend_.get(),
                              "MovingDomainOrchestrator: external backend requested but not set");
            return external_backend_;
    }
    return nullptr;
}

std::vector<motion::MotionDirichletBC> MovingDomainOrchestrator::buildBoundaryConditions() const
{
    return make_motion_dirichlet_bcs(config_.boundary_motion);
}

void MovingDomainOrchestrator::notifySystems(std::span<systems::FESystem* const> systems)
{
    if (!config_.notify_fe_systems) {
        return;
    }
    for (auto* system : systems) {
        if (!system) {
            continue;
        }
        system->notifyMeshGeometryAdvanced();
        ++diagnostics_.notified_fe_systems;
    }
}

void MovingDomainOrchestrator::validateSystemCoordinateConfiguration(
    std::span<systems::FESystem* const> systems) const
{
    for (const auto* system : systems) {
        if (!system) {
            continue;
        }
        FE_THROW_IF(system->coordinateConfiguration() != config_.fe_coordinate_configuration,
                    InvalidArgumentException,
                    "MovingDomainOrchestrator: FE system coordinate configuration does not match moving-domain configuration");
    }
}

MovingDomainDiagnostics MovingDomainOrchestrator::advance(
    MovingDomainAdvancePoint point,
    double time,
    double dt,
    std::span<systems::FESystem* const> systems)
{
    FE_CHECK_NOT_NULL(mesh_.get(), "MovingDomainOrchestrator::advance: mesh");
    validateSystemCoordinateConfiguration(systems);

    diagnostics_ = MovingDomainDiagnostics{};
    diagnostics_.mode = to_string(config_.mode);
    diagnostics_.backend_model = to_string(resolved_backend_model(config_));
    diagnostics_.advance_point = to_string(point);
    diagnostics_.time = time;
    diagnostics_.dt = dt;
    diagnostics_.accepted_step_scale = configured_step_scale(config_.motion);
    diagnostics_.boundary_condition_count = config_.boundary_motion.size();
    diagnostics_.geometry_revision_before = mesh_->local_mesh().geometry_revision();

    if (config_.mode == MovingMeshMode::Disabled) {
        if (config_.fe_coordinate_configuration == Configuration::Reference) {
            mesh_->local_mesh().use_reference_configuration();
        }
        diagnostics_.success = true;
        diagnostics_.message = "moving mesh disabled";
        diagnostics_.geometry_revision_after = mesh_->local_mesh().geometry_revision();
        return diagnostics_;
    }

    if (config_.mode == MovingMeshMode::CoupledMonolithic) {
        diagnostics_.success = false;
        diagnostics_.message = "coupled/monolithic mesh motion is configured but not supported by this orchestration phase";
        diagnostics_.geometry_revision_after = mesh_->local_mesh().geometry_revision();
        return diagnostics_;
    }

    auto backend = buildBackend();
    FE_CHECK_NOT_NULL(backend.get(), "MovingDomainOrchestrator::advance: backend");
    const auto* regularization_backend =
        dynamic_cast<const GeometryRegularizationMotionBackend*>(backend.get());
    diagnostics_.backend_name = backend->name();

    auto bcs = buildBoundaryConditions();
    motion::MeshMotion motion(*mesh_);
    motion.set_config(config_.motion);
    motion.set_backend(backend);
    motion.set_dirichlet_bcs(std::move(bcs));

    const bool ok = motion.advance(dt);
    diagnostics_.success = ok;
    diagnostics_.rolled_back = !ok;
    diagnostics_.geometry_revision_after = mesh_->local_mesh().geometry_revision();
    diagnostics_.advanced_geometry =
        ok && diagnostics_.geometry_revision_after != diagnostics_.geometry_revision_before;
    if (regularization_backend) {
        diagnostics_.accepted_step_scale =
            static_cast<double>(regularization_backend->last_diagnostics().accepted_step_scale);
    }

    if (ok) {
        if (config_.fe_coordinate_configuration == Configuration::Current) {
            mesh_->local_mesh().use_current_configuration();
        } else {
            mesh_->local_mesh().use_reference_configuration();
        }
        if (config_.exchange_ghost_coordinates && mesh_->world_size() > 1 &&
            mesh_->local_mesh().has_current_coords()) {
            mesh_->update_exchange_ghost_coordinates(Configuration::Current);
        }
        notifySystems(systems);
        fill_quality_diagnostics(diagnostics_, *mesh_);
        diagnostics_.message = "moving mesh advanced before physics solve";
    } else {
        diagnostics_.message = "moving mesh advancement failed and was rolled back";
    }

    return diagnostics_;
}

std::function<bool(timestepping::TimeHistory&, double, double)>
MovingDomainOrchestrator::makeBeforePhysicsSolveCallback(std::vector<systems::FESystem*> system_ptrs)
{
    return [this, system_ptrs = std::move(system_ptrs)](timestepping::TimeHistory&, double solve_time, double dt) {
        auto system_span = std::span<systems::FESystem* const>(system_ptrs.data(), system_ptrs.size());
        const auto diagnostics = advance(MovingDomainAdvancePoint::BeforePhysicsSolve,
                                         solve_time,
                                         dt,
                                         system_span);
        return diagnostics.success;
    };
}

std::unique_ptr<systems::FESystem>
make_moving_domain_fe_system(std::shared_ptr<Mesh> mesh,
                             const MovingDomainConfig& config)
{
    FE_CHECK_NOT_NULL(mesh.get(), "make_moving_domain_fe_system: mesh");
    return std::make_unique<systems::FESystem>(std::move(mesh),
                                               config.fe_coordinate_configuration);
}

} // namespace svmp::FE::moving_mesh

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
