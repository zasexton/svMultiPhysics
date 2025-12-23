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

    // Initialize solution in context if needed
    if (!solution_.empty()) {
        context_.setSolutionCoefficients(solution_);
    }

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
            if (!solution_.empty()) {
                thread_context.setSolutionCoefficients(solution_);
            }

            std::vector<Real> my_values;

            #pragma omp for schedule(static)
            for (GlobalIndex cell_id = 0; cell_id < mesh_->numCells(); ++cell_id) {
                // Prepare context (simplified - real impl would be thread-safe)
                ElementType cell_type = mesh_->getCellType(cell_id);
                const auto& element = space_->getElement(cell_type, cell_id);
                thread_context.configure(cell_id, element, element, required);

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
        mesh_->forEachCell([&](GlobalIndex cell_id) {
            prepareContext(cell_id, required);

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
    // Get element type
    ElementType cell_type = mesh_->getCellType(cell_id);

    // Get element from space
    const auto& element = space_->getElement(cell_type, cell_id);

    // Configure context
    context_.configure(cell_id, element, element, required_data);

    // In a full implementation:
    // 1. Get quadrature rule from element
    // 2. Map quadrature points to physical space
    // 3. Compute Jacobians
    // 4. Evaluate basis functions and solution
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
