/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "AssemblyKernel.h"
#include "AssemblyContext.h"

#include <algorithm>
#include <cmath>

namespace svmp {
namespace FE {
namespace assembly {

// ============================================================================
// AssemblyKernel Default Implementations
// ============================================================================

void AssemblyKernel::computeBoundaryFace(
    const AssemblyContext& /*ctx*/,
    int /*boundary_marker*/,
    KernelOutput& /*output*/)
{
    // Default: no boundary face contribution
}

void AssemblyKernel::computeInteriorFace(
    const AssemblyContext& /*ctx_minus*/,
    const AssemblyContext& /*ctx_plus*/,
    KernelOutput& /*output_minus*/,
    KernelOutput& /*output_plus*/,
    KernelOutput& /*coupling_minus_plus*/,
    KernelOutput& /*coupling_plus_minus*/)
{
    // Default: no interior face contribution
}

// ============================================================================
// MassKernel Implementation
// ============================================================================

MassKernel::MassKernel(Real coefficient)
    : coefficient_(coefficient)
{
}

RequiredData MassKernel::getRequiredData() const
{
    return RequiredData::BasisValues | RequiredData::IntegrationWeights;
}

void MassKernel::computeCell(const AssemblyContext& ctx, KernelOutput& output)
{
    const auto n_dofs = ctx.numTestDofs();
    const auto n_qpts = ctx.numQuadraturePoints();

    output.reserve(n_dofs, n_dofs, true, false);

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const Real w = coefficient_ * ctx.integrationWeight(q);

        for (LocalIndex i = 0; i < n_dofs; ++i) {
            const Real phi_i = ctx.basisValue(i, q);

            for (LocalIndex j = 0; j < n_dofs; ++j) {
                const Real phi_j = ctx.basisValue(j, q);
                output.matrixEntry(i, j) += w * phi_i * phi_j;
            }
        }
    }
}

// ============================================================================
// StiffnessKernel Implementation
// ============================================================================

StiffnessKernel::StiffnessKernel(Real coefficient)
    : coefficient_(coefficient)
{
}

RequiredData StiffnessKernel::getRequiredData() const
{
    return RequiredData::PhysicalGradients | RequiredData::IntegrationWeights;
}

void StiffnessKernel::computeCell(const AssemblyContext& ctx, KernelOutput& output)
{
    const auto n_dofs = ctx.numTestDofs();
    const auto n_qpts = ctx.numQuadraturePoints();

    output.reserve(n_dofs, n_dofs, true, false);

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const Real w = coefficient_ * ctx.integrationWeight(q);

        for (LocalIndex i = 0; i < n_dofs; ++i) {
            const auto grad_i = ctx.physicalGradient(i, q);

            for (LocalIndex j = 0; j < n_dofs; ++j) {
                const auto grad_j = ctx.physicalGradient(j, q);

                // Dot product of gradients
                Real dot = grad_i[0] * grad_j[0] +
                           grad_i[1] * grad_j[1] +
                           grad_i[2] * grad_j[2];

                output.matrixEntry(i, j) += w * dot;
            }
        }
    }
}

// ============================================================================
// SourceKernel Implementation
// ============================================================================

SourceKernel::SourceKernel(SourceFunction source)
    : source_(std::move(source))
{
}

SourceKernel::SourceKernel(Real constant_source)
    : source_([constant_source](Real, Real, Real) { return constant_source; })
{
}

RequiredData SourceKernel::getRequiredData() const
{
    return RequiredData::BasisValues | RequiredData::IntegrationWeights |
           RequiredData::PhysicalPoints;
}

void SourceKernel::computeCell(const AssemblyContext& ctx, KernelOutput& output)
{
    const auto n_dofs = ctx.numTestDofs();
    const auto n_qpts = ctx.numQuadraturePoints();

    output.reserve(n_dofs, n_dofs, false, true);

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const Real w = ctx.integrationWeight(q);
        const auto x = ctx.physicalPoint(q);
        const Real f = source_(x[0], x[1], x[2]);

        for (LocalIndex i = 0; i < n_dofs; ++i) {
            const Real phi_i = ctx.basisValue(i, q);
            output.vectorEntry(i) += w * f * phi_i;
        }
    }
}

// ============================================================================
// PoissonKernel Implementation
// ============================================================================

PoissonKernel::PoissonKernel(SourceFunction source)
    : source_(std::move(source))
{
}

PoissonKernel::PoissonKernel(Real constant_source)
    : source_([constant_source](Real, Real, Real) { return constant_source; })
{
}

RequiredData PoissonKernel::getRequiredData() const
{
    return RequiredData::BasisValues | RequiredData::PhysicalGradients |
           RequiredData::IntegrationWeights | RequiredData::PhysicalPoints;
}

void PoissonKernel::computeCell(const AssemblyContext& ctx, KernelOutput& output)
{
    const auto n_dofs = ctx.numTestDofs();
    const auto n_qpts = ctx.numQuadraturePoints();

    output.reserve(n_dofs, n_dofs, true, true);

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const Real w = ctx.integrationWeight(q);
        const auto x = ctx.physicalPoint(q);
        const Real f = source_(x[0], x[1], x[2]);

        for (LocalIndex i = 0; i < n_dofs; ++i) {
            const Real phi_i = ctx.basisValue(i, q);
            const auto grad_i = ctx.physicalGradient(i, q);

            // RHS: (f, phi_i)
            output.vectorEntry(i) += w * f * phi_i;

            // Stiffness: (grad phi_j, grad phi_i)
            for (LocalIndex j = 0; j < n_dofs; ++j) {
                const auto grad_j = ctx.physicalGradient(j, q);

                Real dot = grad_i[0] * grad_j[0] +
                           grad_i[1] * grad_j[1] +
                           grad_i[2] * grad_j[2];

                output.matrixEntry(i, j) += w * dot;
            }
        }
    }
}

// ============================================================================
// CompositeKernel Implementation
// ============================================================================

void CompositeKernel::addKernel(std::shared_ptr<AssemblyKernel> kernel, Real scale)
{
    kernels_.push_back({std::move(kernel), scale});
}

RequiredData CompositeKernel::getRequiredData() const
{
    RequiredData combined = RequiredData::None;
    for (const auto& entry : kernels_) {
        combined |= entry.kernel->getRequiredData();
    }
    return combined;
}

void CompositeKernel::computeCell(const AssemblyContext& ctx, KernelOutput& output)
{
    const auto n_test = ctx.numTestDofs();
    const auto n_trial = ctx.numTrialDofs();

    // Determine what we need
    bool need_matrix = false;
    bool need_vector = false;
    for (const auto& entry : kernels_) {
        if (!entry.kernel->isVectorOnly()) need_matrix = true;
        if (!entry.kernel->isMatrixOnly()) need_vector = true;
    }

    output.reserve(n_test, n_trial, need_matrix, need_vector);

    for (const auto& entry : kernels_) {
        temp_output_.clear();
        entry.kernel->computeCell(ctx, temp_output_);

        // Accumulate matrix
        if (temp_output_.has_matrix && need_matrix) {
            for (LocalIndex i = 0; i < n_test; ++i) {
                for (LocalIndex j = 0; j < n_trial; ++j) {
                    output.matrixEntry(i, j) += entry.scale * temp_output_.matrixEntry(i, j);
                }
            }
        }

        // Accumulate vector
        if (temp_output_.has_vector && need_vector) {
            for (LocalIndex i = 0; i < n_test; ++i) {
                output.vectorEntry(i) += entry.scale * temp_output_.vectorEntry(i);
            }
        }
    }
}

void CompositeKernel::computeBoundaryFace(
    const AssemblyContext& ctx,
    int boundary_marker,
    KernelOutput& output)
{
    const auto n_test = ctx.numTestDofs();
    const auto n_trial = ctx.numTrialDofs();

    bool need_matrix = false;
    bool need_vector = false;
    for (const auto& entry : kernels_) {
        if (entry.kernel->hasBoundaryFace()) {
            if (!entry.kernel->isVectorOnly()) need_matrix = true;
            if (!entry.kernel->isMatrixOnly()) need_vector = true;
        }
    }

    output.reserve(n_test, n_trial, need_matrix, need_vector);

    for (const auto& entry : kernels_) {
        if (!entry.kernel->hasBoundaryFace()) continue;

        temp_output_.clear();
        entry.kernel->computeBoundaryFace(ctx, boundary_marker, temp_output_);

        if (temp_output_.has_matrix && need_matrix) {
            for (LocalIndex i = 0; i < n_test; ++i) {
                for (LocalIndex j = 0; j < n_trial; ++j) {
                    output.matrixEntry(i, j) += entry.scale * temp_output_.matrixEntry(i, j);
                }
            }
        }

        if (temp_output_.has_vector && need_vector) {
            for (LocalIndex i = 0; i < n_test; ++i) {
                output.vectorEntry(i) += entry.scale * temp_output_.vectorEntry(i);
            }
        }
    }
}

bool CompositeKernel::hasBoundaryFace() const noexcept
{
    for (const auto& entry : kernels_) {
        if (entry.kernel->hasBoundaryFace()) return true;
    }
    return false;
}

} // namespace assembly
} // namespace FE
} // namespace svmp
