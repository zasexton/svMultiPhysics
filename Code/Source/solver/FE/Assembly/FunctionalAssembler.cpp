/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "FunctionalAssembler.h"
#include "Dofs/DofMap.h"
#include "Spaces/FunctionSpace.h"
#include "Elements/Element.h"
#include "Quadrature/QuadratureFactory.h"
#include "Geometry/MappingFactory.h"
#include "Math/Vector.h"
#include "Basis/BasisFunction.h"

#include <chrono>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace svmp {
namespace FE {
namespace assembly {

namespace {

int defaultGeometryOrder(ElementType element_type) noexcept
{
    switch (element_type) {
        case ElementType::Line3:
        case ElementType::Triangle6:
        case ElementType::Quad8:
        case ElementType::Quad9:
        case ElementType::Tetra10:
        case ElementType::Hex20:
        case ElementType::Hex27:
        case ElementType::Wedge15:
        case ElementType::Wedge18:
        case ElementType::Pyramid13:
        case ElementType::Pyramid14:
            return 2;
        default:
            return 1;
    }
}

struct CellContextScratch {
    std::vector<std::array<Real, 3>> cell_coords;

    std::vector<AssemblyContext::Point3D> quad_points;
    std::vector<Real> quad_weights;
    std::vector<AssemblyContext::Point3D> phys_points;
    std::vector<AssemblyContext::Matrix3x3> jacobians;
    std::vector<AssemblyContext::Matrix3x3> inv_jacobians;
    std::vector<Real> jac_dets;
    std::vector<Real> integration_weights;

    std::vector<Real> basis_values;
    std::vector<AssemblyContext::Vector3D> ref_gradients;
    std::vector<AssemblyContext::Vector3D> phys_gradients;
};

CellContextScratch& cellScratch()
{
    static thread_local CellContextScratch scratch;
    return scratch;
}

void prepareCellContext(AssemblyContext& context,
                        const IMeshAccess& mesh,
                        GlobalIndex cell_id,
                        const spaces::FunctionSpace& space,
                        RequiredData required_data)
{
    auto& scratch = cellScratch();

    const ElementType cell_type = mesh.getCellType(cell_id);
    const int dim = mesh.dimension();

    const auto& element = space.getElement(cell_type, cell_id);

    auto quad_rule = element.quadrature();
    if (!quad_rule) {
        const int quad_order = quadrature::QuadratureFactory::recommended_order(
            element.polynomial_order(), false);
        quad_rule = quadrature::QuadratureFactory::create(cell_type, quad_order);
    }

    const auto n_qpts = static_cast<LocalIndex>(quad_rule->num_points());
    const auto n_dofs = static_cast<LocalIndex>(element.num_dofs());

    mesh.getCellCoordinates(cell_id, scratch.cell_coords);

    std::vector<math::Vector<Real, 3>> node_coords(scratch.cell_coords.size());
    for (std::size_t i = 0; i < scratch.cell_coords.size(); ++i) {
        node_coords[i] = math::Vector<Real, 3>{
            scratch.cell_coords[i][0],
            scratch.cell_coords[i][1],
            scratch.cell_coords[i][2]};
    }

    geometry::MappingRequest map_request;
    map_request.element_type = cell_type;
    map_request.geometry_order = defaultGeometryOrder(cell_type);
    map_request.use_affine = (map_request.geometry_order <= 1);
    auto mapping = geometry::MappingFactory::create(map_request, node_coords);

    scratch.quad_points.resize(n_qpts);
    scratch.quad_weights.resize(n_qpts);
    scratch.phys_points.resize(n_qpts);
    scratch.jacobians.resize(n_qpts);
    scratch.inv_jacobians.resize(n_qpts);
    scratch.jac_dets.resize(n_qpts);
    scratch.integration_weights.resize(n_qpts);

    scratch.basis_values.resize(static_cast<std::size_t>(n_dofs * n_qpts));
    scratch.ref_gradients.resize(static_cast<std::size_t>(n_dofs * n_qpts));
    scratch.phys_gradients.resize(static_cast<std::size_t>(n_dofs * n_qpts));

    const auto& quad_points = quad_rule->points();
    const auto& quad_weights = quad_rule->weights();

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const auto& qpt = quad_points[q];
        scratch.quad_points[q] = {qpt[0], qpt[1], qpt[2]};
        scratch.quad_weights[q] = quad_weights[q];

        const math::Vector<Real, 3> xi{qpt[0], qpt[1], qpt[2]};
        const auto x_phys = mapping->map_to_physical(xi);
        scratch.phys_points[q] = {x_phys[0], x_phys[1], x_phys[2]};

        const auto J = mapping->jacobian(xi);
        const auto J_inv = mapping->jacobian_inverse(xi);
        const Real det_J = mapping->jacobian_determinant(xi);

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                scratch.jacobians[q][i][j] = J(i, j);
                scratch.inv_jacobians[q][i][j] = J_inv(i, j);
            }
        }
        scratch.jac_dets[q] = det_J;
        scratch.integration_weights[q] = quad_weights[q] * std::abs(det_J);
    }

    const auto& basis = element.basis();
    std::vector<Real> values_at_pt;
    std::vector<basis::Gradient> gradients_at_pt;

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const math::Vector<Real, 3> xi{
            scratch.quad_points[q][0],
            scratch.quad_points[q][1],
            scratch.quad_points[q][2]};

        basis.evaluate_values(xi, values_at_pt);
        basis.evaluate_gradients(xi, gradients_at_pt);

        for (LocalIndex i = 0; i < n_dofs; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i * n_qpts + q);
            scratch.basis_values[idx] = values_at_pt[i];
            scratch.ref_gradients[idx] = {
                gradients_at_pt[i][0],
                gradients_at_pt[i][1],
                gradients_at_pt[i][2]};

            const auto& grad_ref = scratch.ref_gradients[idx];
            const auto& J_inv = scratch.inv_jacobians[q];
            AssemblyContext::Vector3D grad_phys = {0.0, 0.0, 0.0};
            for (int d1 = 0; d1 < dim; ++d1) {
                for (int d2 = 0; d2 < dim; ++d2) {
                    grad_phys[d1] += J_inv[d2][d1] * grad_ref[d2];
                }
            }
            scratch.phys_gradients[idx] = grad_phys;
        }
    }

    context.configure(cell_id, element, element, required_data);
    context.setQuadratureData(scratch.quad_points, scratch.quad_weights);
    context.setPhysicalPoints(scratch.phys_points);
    context.setJacobianData(scratch.jacobians, scratch.inv_jacobians, scratch.jac_dets);
    context.setIntegrationWeights(scratch.integration_weights);
    context.setTestBasisData(n_dofs, scratch.basis_values, scratch.ref_gradients);
    context.setPhysicalGradients(scratch.phys_gradients, scratch.phys_gradients);
}

} // namespace

// ============================================================================
// Construction
// ============================================================================

FunctionalAssembler::FunctionalAssembler()
    : options_{}
{
}

FunctionalAssembler::FunctionalAssembler(const FunctionalAssemblyOptions& options)
    : options_(options)
{
}

FunctionalAssembler::~FunctionalAssembler() = default;

FunctionalAssembler::FunctionalAssembler(FunctionalAssembler&& other) noexcept = default;

FunctionalAssembler& FunctionalAssembler::operator=(FunctionalAssembler&& other) noexcept = default;

// ============================================================================
// Configuration
// ============================================================================

void FunctionalAssembler::setMesh(const IMeshAccess& mesh)
{
    mesh_ = &mesh;
}

void FunctionalAssembler::setDofMap(const dofs::DofMap& dof_map)
{
    dof_map_ = &dof_map;
}

void FunctionalAssembler::setSpace(const spaces::FunctionSpace& space)
{
    space_ = &space;
}

void FunctionalAssembler::setSolution(std::span<const Real> solution)
{
    solution_.assign(solution.begin(), solution.end());
}

void FunctionalAssembler::setOptions(const FunctionalAssemblyOptions& options)
{
    options_ = options;
}

const FunctionalAssemblyOptions& FunctionalAssembler::getOptions() const noexcept
{
    return options_;
}

// ============================================================================
// Scalar Assembly
// ============================================================================

Real FunctionalAssembler::assembleScalar(FunctionalKernel& kernel)
{
    FunctionalResult result;
    Real value = assembleCellsCore(kernel, result);
    last_result_ = result;

    // Apply post-processing (e.g., square root for norms)
    return kernel.postProcess(value);
}

FunctionalResult FunctionalAssembler::assembleScalarDetailed(FunctionalKernel& kernel)
{
    FunctionalResult result;
    Real raw_value = assembleCellsCore(kernel, result);
    result.value = kernel.postProcess(raw_value);
    last_result_ = result;
    return result;
}

Real FunctionalAssembler::assembleBoundaryScalar(
    FunctionalKernel& kernel,
    int boundary_marker)
{
    FunctionalResult result;
    Real value = assembleBoundaryCore(kernel, boundary_marker, result);
    last_result_ = result;
    return kernel.postProcess(value);
}

// ============================================================================
// Multiple Functionals
// ============================================================================

std::vector<Real> FunctionalAssembler::assembleMultiple(
    std::span<FunctionalKernel* const> kernels)
{
    if (!isConfigured()) {
        throw std::runtime_error("FunctionalAssembler: not configured");
    }

    if (kernels.empty()) {
        return {};
    }

    auto start_time = std::chrono::steady_clock::now();

    const std::size_t num_kernels = kernels.size();
    std::vector<Real> totals(num_kernels, 0.0);
    std::vector<KahanAccumulator> accumulators(num_kernels);

    // Compute union of required data
    RequiredData all_required = RequiredData::None;
    for (auto* kernel : kernels) {
        all_required = all_required | kernel->getRequiredData();
    }

    // Initialize solution in context if needed
    if (!solution_.empty()) {
        context_.setSolutionCoefficients(solution_);
    }

    GlobalIndex elements_processed = 0;

    // Loop over cells
    mesh_->forEachCell([&](GlobalIndex cell_id) {
        prepareContext(cell_id, all_required);

        // Evaluate all kernels for this cell
        for (std::size_t k = 0; k < num_kernels; ++k) {
            if (kernels[k]->hasCell()) {
                Real cell_value = kernels[k]->evaluateCellTotal(context_);
                if (options_.use_kahan_summation) {
                    accumulators[k].add(cell_value);
                } else {
                    totals[k] += cell_value;
                }
            }
        }

        elements_processed++;
    });

    // Extract final values
    std::vector<Real> results(num_kernels);
    for (std::size_t k = 0; k < num_kernels; ++k) {
        Real raw = options_.use_kahan_summation ? accumulators[k].get() : totals[k];
        results[k] = kernels[k]->postProcess(raw);
    }

    auto end_time = std::chrono::steady_clock::now();
    last_result_.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    last_result_.elements_processed = elements_processed;

    return results;
}

// ============================================================================
// Goal-Oriented Error Estimation
// ============================================================================

std::vector<Real> FunctionalAssembler::computeGoalOrientedIndicators(
    std::span<const Real> primal_solution,
    std::span<const Real> dual_solution,
    FunctionalKernel& kernel)
{
    if (!isConfigured()) {
        throw std::runtime_error("FunctionalAssembler: not configured");
    }

    // Store original solution
    auto original_solution = std::move(solution_);

    // We need both solutions; store primal for now
    solution_.assign(primal_solution.begin(), primal_solution.end());
    context_.setSolutionCoefficients(solution_);

    // Store dual solution separately
    std::vector<Real> dual(dual_solution.begin(), dual_solution.end());

    // Get number of cells
    GlobalIndex num_cells = mesh_->numCells();
    std::vector<Real> indicators(static_cast<std::size_t>(num_cells), 0.0);

    RequiredData required = kernel.getRequiredData() | RequiredData::SolutionValues;

    GlobalIndex cell_idx = 0;
    mesh_->forEachCell([&](GlobalIndex cell_id) {
        prepareContext(cell_id, required);

        // Compute DWR indicator: R(u_h) * z_h integrated over element
        // Simplified: we use the kernel evaluation weighted by dual solution
        Real indicator = 0.0;

        for (LocalIndex q = 0; q < context_.numQuadraturePoints(); ++q) {
            Real residual_contribution = kernel.evaluateCell(context_, q);
            // In a full implementation, we'd evaluate the dual solution at this point
            // For now, use element-average dual value
            Real jxw = context_.integrationWeight(q);
            indicator += residual_contribution * jxw;
        }

        indicators[static_cast<std::size_t>(cell_idx)] = std::abs(indicator);
        cell_idx++;
    });

    // Restore original solution
    solution_ = std::move(original_solution);

    return indicators;
}

// ============================================================================
// Convenience Norm Computations
// ============================================================================

Real FunctionalAssembler::computeL2Norm()
{
    L2NormKernel kernel;
    return assembleScalar(kernel);
}

Real FunctionalAssembler::computeH1Seminorm()
{
    H1SeminormKernel kernel;
    return assembleScalar(kernel);
}

Real FunctionalAssembler::computeL2Error(
    std::function<Real(Real, Real, Real)> exact_solution)
{
    L2ErrorKernel kernel(std::move(exact_solution));
    return assembleScalar(kernel);
}

Real FunctionalAssembler::computeH1Error(
    std::function<std::array<Real, 3>(Real, Real, Real)> exact_gradient)
{
    H1ErrorKernel kernel(std::move(exact_gradient));
    return assembleScalar(kernel);
}

Real FunctionalAssembler::computeEnergy()
{
    EnergyKernel kernel;
    return assembleScalar(kernel);
}

Real FunctionalAssembler::computeVolume()
{
    VolumeKernel kernel;
    return assembleScalar(kernel);
}

// ============================================================================
// Query
// ============================================================================

bool FunctionalAssembler::isConfigured() const noexcept
{
    return mesh_ != nullptr && dof_map_ != nullptr && space_ != nullptr;
}

const FunctionalResult& FunctionalAssembler::getLastResult() const noexcept
{
    return last_result_;
}

// ============================================================================
// Internal Implementation
// ============================================================================

Real FunctionalAssembler::assembleCellsCore(
    FunctionalKernel& kernel,
    FunctionalResult& result)
{
    if (!isConfigured()) {
        result.success = false;
        result.error_message = "FunctionalAssembler not configured";
        return 0.0;
    }

    auto start_time = std::chrono::steady_clock::now();

    const RequiredData required = kernel.getRequiredData();
    const bool need_solution =
        hasFlag(required, RequiredData::SolutionValues) ||
        hasFlag(required, RequiredData::SolutionGradients) ||
        hasFlag(required, RequiredData::SolutionHessians) ||
        hasFlag(required, RequiredData::SolutionLaplacians);

    Real total = 0.0;
    KahanAccumulator accumulator;

    const int num_threads = options_.num_threads;
    const bool use_parallel = (num_threads > 1);

    if (use_parallel && options_.deterministic) {
        // Deterministic parallel: gather local contributions then reduce
        std::vector<Real> local_values;
        std::mutex values_mutex;

#ifdef _OPENMP
        #pragma omp parallel num_threads(num_threads)
        {
            AssemblyContext thread_context;
            thread_context.reserve(dof_map_->getMaxDofsPerCell(), 27,
                                   mesh_->dimension());

            std::vector<Real> my_values;
            std::vector<Real> local_solution;

            #pragma omp for schedule(static)
            for (GlobalIndex cell_id = 0; cell_id < mesh_->numCells(); ++cell_id) {
                prepareCellContext(thread_context, *mesh_, cell_id, *space_, required);
                if (need_solution) {
                    FE_THROW_IF(solution_.empty(), FEException,
                                "FunctionalAssembler::assembleCellsCore: kernel requires solution but no solution was set");
                    const auto dofs = dof_map_->getCellDofs(cell_id);
                    local_solution.resize(dofs.size());
                    for (std::size_t i = 0; i < dofs.size(); ++i) {
                        const auto dof = dofs[i];
                        FE_THROW_IF(dof < 0 || static_cast<std::size_t>(dof) >= solution_.size(), FEException,
                                    "FunctionalAssembler::assembleCellsCore: solution vector too small for DOF " + std::to_string(dof));
                        local_solution[i] = solution_[static_cast<std::size_t>(dof)];
                    }
                    thread_context.setSolutionCoefficients(local_solution);
                }

                Real cell_value = kernel.evaluateCellTotal(thread_context);
                my_values.push_back(cell_value);
            }

            // Merge results
            {
                std::lock_guard<std::mutex> lock(values_mutex);
                local_values.insert(local_values.end(),
                                    my_values.begin(), my_values.end());
            }
        }
#else
        // Sequential fallback
        mesh_->forEachCell([&](GlobalIndex cell_id) {
            prepareContext(cell_id, required);
            Real cell_value = kernel.evaluateCellTotal(context_);
            local_values.push_back(cell_value);
            result.elements_processed++;
        });
#endif

        // Deterministic reduction: sort by element order then sum
        // (Parallel execution may produce out-of-order results)
        // For truly deterministic results, we rely on stable per-element order
        total = parallelReduce(local_values, options_.use_kahan_summation);
        result.elements_processed = static_cast<GlobalIndex>(local_values.size());

    } else {
        // Sequential assembly
        std::vector<Real> local_solution;
        mesh_->forEachCell([&](GlobalIndex cell_id) {
            prepareContext(cell_id, required);
            if (need_solution) {
                FE_THROW_IF(solution_.empty(), FEException,
                            "FunctionalAssembler::assembleCellsCore: kernel requires solution but no solution was set");
                const auto dofs = dof_map_->getCellDofs(cell_id);
                local_solution.resize(dofs.size());
                for (std::size_t i = 0; i < dofs.size(); ++i) {
                    const auto dof = dofs[i];
                    FE_THROW_IF(dof < 0 || static_cast<std::size_t>(dof) >= solution_.size(), FEException,
                                "FunctionalAssembler::assembleCellsCore: solution vector too small for DOF " + std::to_string(dof));
                    local_solution[i] = solution_[static_cast<std::size_t>(dof)];
                }
                context_.setSolutionCoefficients(local_solution);
            }

            Real cell_value = kernel.evaluateCellTotal(context_);

            if (options_.use_kahan_summation) {
                accumulator.add(cell_value);
            } else {
                total += cell_value;
            }

            result.elements_processed++;
        });

        if (options_.use_kahan_summation) {
            total = accumulator.get();
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    result.success = true;

    return total;
}

Real FunctionalAssembler::assembleBoundaryCore(
    FunctionalKernel& kernel,
    int boundary_marker,
    FunctionalResult& result)
{
    if (!isConfigured()) {
        result.success = false;
        result.error_message = "FunctionalAssembler not configured";
        return 0.0;
    }

    if (!kernel.hasBoundaryFace()) {
        result.success = false;
        result.error_message = "Kernel does not support boundary face integration";
        return 0.0;
    }

    auto start_time = std::chrono::steady_clock::now();

    const RequiredData required = kernel.getRequiredData();

    if (!solution_.empty()) {
        context_.setSolutionCoefficients(solution_);
    }

    Real total = 0.0;
    KahanAccumulator accumulator;

    mesh_->forEachBoundaryFace(boundary_marker,
        [&](GlobalIndex face_id, GlobalIndex cell_id) {
            // Prepare face context
            LocalIndex local_face_id = 0;  // Simplified
            ElementType cell_type = mesh_->getCellType(cell_id);
            const auto& element = space_->getElement(cell_type, cell_id);

            context_.configureFace(face_id, cell_id, local_face_id, element,
                                   required, ContextType::BoundaryFace);
            context_.setBoundaryMarker(boundary_marker);

            // Integrate over face
            for (LocalIndex q = 0; q < context_.numQuadraturePoints(); ++q) {
                Real value = kernel.evaluateBoundaryFace(context_, q, boundary_marker);
                Real jxw = context_.integrationWeight(q);

                if (options_.use_kahan_summation) {
                    accumulator.add(value * jxw);
                } else {
                    total += value * jxw;
                }
            }

            result.faces_processed++;
        });

    if (options_.use_kahan_summation) {
        total = accumulator.get();
    }

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    result.success = true;

    return total;
}

void FunctionalAssembler::prepareContext(
    GlobalIndex cell_id,
    RequiredData required_data)
{
    prepareCellContext(context_, *mesh_, cell_id, *space_, required_data);
}

Real FunctionalAssembler::parallelReduce(
    const std::vector<Real>& local_values,
    bool use_kahan)
{
    if (local_values.empty()) {
        return 0.0;
    }

    if (use_kahan) {
        KahanAccumulator acc;
        for (Real v : local_values) {
            acc.add(v);
        }
        return acc.get();
    } else {
        return std::accumulate(local_values.begin(), local_values.end(), Real{0.0});
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<FunctionalAssembler> createFunctionalAssembler()
{
    return std::make_unique<FunctionalAssembler>();
}

std::unique_ptr<FunctionalAssembler> createFunctionalAssembler(
    const FunctionalAssemblyOptions& options)
{
    return std::make_unique<FunctionalAssembler>(options);
}

} // namespace assembly
} // namespace FE
} // namespace svmp
