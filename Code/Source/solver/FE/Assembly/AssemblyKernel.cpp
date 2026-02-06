/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "AssemblyKernel.h"
#include "AssemblyContext.h"
#include "Math/SIMD.h"

#include <algorithm>
#include <array>
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

void AssemblyKernel::computeInterfaceFace(
    const AssemblyContext& ctx_minus,
    const AssemblyContext& ctx_plus,
    int /*interface_marker*/,
    KernelOutput& output_minus,
    KernelOutput& output_plus,
    KernelOutput& coupling_minus_plus,
    KernelOutput& coupling_plus_minus)
{
    // Default: treat interface faces like interior faces if the kernel implements them.
    computeInteriorFace(ctx_minus, ctx_plus,
                        output_minus, output_plus,
                        coupling_minus_plus, coupling_plus_minus);
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
    const auto n_test = ctx.numTestDofs();
    const auto n_trial = ctx.numTrialDofs();
    const auto n_qpts = ctx.numQuadraturePoints();

    output.reserve(n_test, n_trial, true, false);

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const Real w = coefficient_ * ctx.integrationWeight(q);

        for (LocalIndex i = 0; i < n_test; ++i) {
            const Real phi_i = ctx.basisValue(i, q);

            for (LocalIndex j = 0; j < n_trial; ++j) {
                const Real phi_j = ctx.trialBasisValue(j, q);
                output.matrixEntry(i, j) += w * phi_i * phi_j;
            }
        }
    }
}

void MassKernel::computeCellBatch(std::span<const AssemblyContext* const> contexts,
                                  std::span<KernelOutput> outputs)
{
    const std::size_t n = std::min(contexts.size(), outputs.size());
    using simd_ops = math::simd::SIMDOps<Real>;
    using vec_t = simd_ops::vec_type;
    constexpr std::size_t lane_width = simd_ops::vec_size;

    std::size_t begin = 0u;
    while (begin < n) {
        std::array<const AssemblyContext*, lane_width> lane_ctx{};
        std::array<KernelOutput*, lane_width> lane_out{};
        std::size_t active = 0u;
        while (begin < n && active < lane_width) {
            if (contexts[begin] != nullptr) {
                lane_ctx[active] = contexts[begin];
                lane_out[active] = &outputs[begin];
                ++active;
            }
            ++begin;
        }

        if (active == 0u) {
            continue;
        }

        const LocalIndex n_test = lane_ctx[0]->numTestDofs();
        const LocalIndex n_trial = lane_ctx[0]->numTrialDofs();
        const LocalIndex n_qpts = lane_ctx[0]->numQuadraturePoints();

        bool homogeneous = true;
        for (std::size_t lane = 1u; lane < active; ++lane) {
            homogeneous = homogeneous &&
                          (lane_ctx[lane]->numTestDofs() == n_test) &&
                          (lane_ctx[lane]->numTrialDofs() == n_trial) &&
                          (lane_ctx[lane]->numQuadraturePoints() == n_qpts);
        }
        if (!homogeneous) {
            for (std::size_t lane = 0u; lane < active; ++lane) {
                computeCell(*lane_ctx[lane], *lane_out[lane]);
            }
            continue;
        }

        for (std::size_t lane = 0u; lane < active; ++lane) {
            lane_out[lane]->reserve(n_test, n_trial, /*need_matrix=*/true, /*need_vector=*/false);
        }

        std::array<Real, lane_width> weights{};
        std::array<Real, lane_width> phi_i_lanes{};
        std::array<Real, lane_width> phi_j_lanes{};
        std::array<Real, lane_width> contrib{};

        for (LocalIndex q = 0; q < n_qpts; ++q) {
            for (std::size_t lane = 0u; lane < lane_width; ++lane) {
                weights[lane] = (lane < active) ? coefficient_ * lane_ctx[lane]->integrationWeight(q) : Real(0);
            }
            const vec_t wv = simd_ops::loadu(weights.data());

            for (LocalIndex i = 0; i < n_test; ++i) {
                for (std::size_t lane = 0u; lane < lane_width; ++lane) {
                    phi_i_lanes[lane] = (lane < active) ? lane_ctx[lane]->basisValue(i, q) : Real(0);
                }
                const vec_t phi_i_v = simd_ops::loadu(phi_i_lanes.data());

                for (LocalIndex j = 0; j < n_trial; ++j) {
                    for (std::size_t lane = 0u; lane < lane_width; ++lane) {
                        phi_j_lanes[lane] = (lane < active) ? lane_ctx[lane]->trialBasisValue(j, q) : Real(0);
                    }
                    const vec_t phi_j_v = simd_ops::loadu(phi_j_lanes.data());
                    const vec_t cv = simd_ops::mul(wv, simd_ops::mul(phi_i_v, phi_j_v));
                    simd_ops::storeu(contrib.data(), cv);
                    for (std::size_t lane = 0u; lane < active; ++lane) {
                        lane_out[lane]->matrixEntry(i, j) += contrib[lane];
                    }
                }
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
    const auto n_test = ctx.numTestDofs();
    const auto n_trial = ctx.numTrialDofs();
    const auto n_qpts = ctx.numQuadraturePoints();

    output.reserve(n_test, n_trial, true, false);

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const Real w = coefficient_ * ctx.integrationWeight(q);

        for (LocalIndex i = 0; i < n_test; ++i) {
            const auto grad_i = ctx.physicalGradient(i, q);

            for (LocalIndex j = 0; j < n_trial; ++j) {
                const auto grad_j = ctx.trialPhysicalGradient(j, q);

                // Dot product of gradients
                Real dot = grad_i[0] * grad_j[0] +
                           grad_i[1] * grad_j[1] +
                           grad_i[2] * grad_j[2];

                output.matrixEntry(i, j) += w * dot;
            }
        }
    }
}

void StiffnessKernel::computeCellBatch(std::span<const AssemblyContext* const> contexts,
                                       std::span<KernelOutput> outputs)
{
    const std::size_t n = std::min(contexts.size(), outputs.size());
    using simd_ops = math::simd::SIMDOps<Real>;
    using vec_t = simd_ops::vec_type;
    constexpr std::size_t lane_width = simd_ops::vec_size;

    std::size_t begin = 0u;
    while (begin < n) {
        std::array<const AssemblyContext*, lane_width> lane_ctx{};
        std::array<KernelOutput*, lane_width> lane_out{};
        std::size_t active = 0u;
        while (begin < n && active < lane_width) {
            if (contexts[begin] != nullptr) {
                lane_ctx[active] = contexts[begin];
                lane_out[active] = &outputs[begin];
                ++active;
            }
            ++begin;
        }

        if (active == 0u) {
            continue;
        }

        const LocalIndex n_test = lane_ctx[0]->numTestDofs();
        const LocalIndex n_trial = lane_ctx[0]->numTrialDofs();
        const LocalIndex n_qpts = lane_ctx[0]->numQuadraturePoints();

        bool homogeneous = true;
        for (std::size_t lane = 1u; lane < active; ++lane) {
            homogeneous = homogeneous &&
                          (lane_ctx[lane]->numTestDofs() == n_test) &&
                          (lane_ctx[lane]->numTrialDofs() == n_trial) &&
                          (lane_ctx[lane]->numQuadraturePoints() == n_qpts);
        }
        if (!homogeneous) {
            for (std::size_t lane = 0u; lane < active; ++lane) {
                computeCell(*lane_ctx[lane], *lane_out[lane]);
            }
            continue;
        }

        for (std::size_t lane = 0u; lane < active; ++lane) {
            lane_out[lane]->reserve(n_test, n_trial, /*need_matrix=*/true, /*need_vector=*/false);
        }

        std::array<Real, lane_width> weights{};
        std::array<Real, lane_width> grad_i0{};
        std::array<Real, lane_width> grad_i1{};
        std::array<Real, lane_width> grad_i2{};
        std::array<Real, lane_width> grad_j0{};
        std::array<Real, lane_width> grad_j1{};
        std::array<Real, lane_width> grad_j2{};
        std::array<Real, lane_width> contrib{};

        for (LocalIndex q = 0; q < n_qpts; ++q) {
            for (std::size_t lane = 0u; lane < lane_width; ++lane) {
                weights[lane] = (lane < active) ? coefficient_ * lane_ctx[lane]->integrationWeight(q) : Real(0);
            }
            const vec_t wv = simd_ops::loadu(weights.data());

            for (LocalIndex i = 0; i < n_test; ++i) {
                for (std::size_t lane = 0u; lane < lane_width; ++lane) {
                    if (lane < active) {
                        const auto gi = lane_ctx[lane]->physicalGradient(i, q);
                        grad_i0[lane] = gi[0];
                        grad_i1[lane] = gi[1];
                        grad_i2[lane] = gi[2];
                    } else {
                        grad_i0[lane] = Real(0);
                        grad_i1[lane] = Real(0);
                        grad_i2[lane] = Real(0);
                    }
                }
                const vec_t gi0v = simd_ops::loadu(grad_i0.data());
                const vec_t gi1v = simd_ops::loadu(grad_i1.data());
                const vec_t gi2v = simd_ops::loadu(grad_i2.data());

                for (LocalIndex j = 0; j < n_trial; ++j) {
                    for (std::size_t lane = 0u; lane < lane_width; ++lane) {
                        if (lane < active) {
                            const auto gj = lane_ctx[lane]->trialPhysicalGradient(j, q);
                            grad_j0[lane] = gj[0];
                            grad_j1[lane] = gj[1];
                            grad_j2[lane] = gj[2];
                        } else {
                            grad_j0[lane] = Real(0);
                            grad_j1[lane] = Real(0);
                            grad_j2[lane] = Real(0);
                        }
                    }
                    const vec_t gj0v = simd_ops::loadu(grad_j0.data());
                    const vec_t gj1v = simd_ops::loadu(grad_j1.data());
                    const vec_t gj2v = simd_ops::loadu(grad_j2.data());

                    const vec_t dotv =
                        simd_ops::add(simd_ops::mul(gi0v, gj0v),
                                      simd_ops::add(simd_ops::mul(gi1v, gj1v),
                                                    simd_ops::mul(gi2v, gj2v)));
                    const vec_t cv = simd_ops::mul(wv, dotv);
                    simd_ops::storeu(contrib.data(), cv);
                    for (std::size_t lane = 0u; lane < active; ++lane) {
                        lane_out[lane]->matrixEntry(i, j) += contrib[lane];
                    }
                }
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

void SourceKernel::computeCellBatch(std::span<const AssemblyContext* const> contexts,
                                    std::span<KernelOutput> outputs)
{
    const std::size_t n = std::min(contexts.size(), outputs.size());
    using simd_ops = math::simd::SIMDOps<Real>;
    using vec_t = simd_ops::vec_type;
    constexpr std::size_t lane_width = simd_ops::vec_size;

    std::size_t begin = 0u;
    while (begin < n) {
        std::array<const AssemblyContext*, lane_width> lane_ctx{};
        std::array<KernelOutput*, lane_width> lane_out{};
        std::size_t active = 0u;
        while (begin < n && active < lane_width) {
            if (contexts[begin] != nullptr) {
                lane_ctx[active] = contexts[begin];
                lane_out[active] = &outputs[begin];
                ++active;
            }
            ++begin;
        }

        if (active == 0u) {
            continue;
        }

        const LocalIndex n_dofs = lane_ctx[0]->numTestDofs();
        const LocalIndex n_qpts = lane_ctx[0]->numQuadraturePoints();

        bool homogeneous = true;
        for (std::size_t lane = 1u; lane < active; ++lane) {
            homogeneous = homogeneous &&
                          (lane_ctx[lane]->numTestDofs() == n_dofs) &&
                          (lane_ctx[lane]->numQuadraturePoints() == n_qpts);
        }
        if (!homogeneous) {
            for (std::size_t lane = 0u; lane < active; ++lane) {
                computeCell(*lane_ctx[lane], *lane_out[lane]);
            }
            continue;
        }

        for (std::size_t lane = 0u; lane < active; ++lane) {
            lane_out[lane]->reserve(n_dofs, n_dofs, /*need_matrix=*/false, /*need_vector=*/true);
        }

        std::array<Real, lane_width> weighted_source{};
        std::array<Real, lane_width> phi_i_lanes{};
        std::array<Real, lane_width> contrib{};
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            for (std::size_t lane = 0u; lane < lane_width; ++lane) {
                if (lane < active) {
                    const auto x = lane_ctx[lane]->physicalPoint(q);
                    weighted_source[lane] =
                        lane_ctx[lane]->integrationWeight(q) * source_(x[0], x[1], x[2]);
                } else {
                    weighted_source[lane] = Real(0);
                }
            }
            const vec_t srcv = simd_ops::loadu(weighted_source.data());

            for (LocalIndex i = 0; i < n_dofs; ++i) {
                for (std::size_t lane = 0u; lane < lane_width; ++lane) {
                    phi_i_lanes[lane] = (lane < active) ? lane_ctx[lane]->basisValue(i, q) : Real(0);
                }
                const vec_t phiv = simd_ops::loadu(phi_i_lanes.data());
                const vec_t cv = simd_ops::mul(srcv, phiv);
                simd_ops::storeu(contrib.data(), cv);
                for (std::size_t lane = 0u; lane < active; ++lane) {
                    lane_out[lane]->vectorEntry(i) += contrib[lane];
                }
            }
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
    const auto n_test = ctx.numTestDofs();
    const auto n_trial = ctx.numTrialDofs();
    const auto n_qpts = ctx.numQuadraturePoints();

    output.reserve(n_test, n_trial, true, true);

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const Real w = ctx.integrationWeight(q);
        const auto x = ctx.physicalPoint(q);
        const Real f = source_(x[0], x[1], x[2]);

        for (LocalIndex i = 0; i < n_test; ++i) {
            const Real phi_i = ctx.basisValue(i, q);
            const auto grad_i = ctx.physicalGradient(i, q);

            // RHS: (f, phi_i)
            output.vectorEntry(i) += w * f * phi_i;

            // Stiffness: (grad phi_j, grad phi_i)
            for (LocalIndex j = 0; j < n_trial; ++j) {
                const auto grad_j = ctx.trialPhysicalGradient(j, q);

                Real dot = grad_i[0] * grad_j[0] +
                           grad_i[1] * grad_j[1] +
                           grad_i[2] * grad_j[2];

                output.matrixEntry(i, j) += w * dot;
            }
        }
    }
}

void PoissonKernel::computeCellBatch(std::span<const AssemblyContext* const> contexts,
                                     std::span<KernelOutput> outputs)
{
    const std::size_t n = std::min(contexts.size(), outputs.size());
    using simd_ops = math::simd::SIMDOps<Real>;
    using vec_t = simd_ops::vec_type;
    constexpr std::size_t lane_width = simd_ops::vec_size;

    std::size_t begin = 0u;
    while (begin < n) {
        std::array<const AssemblyContext*, lane_width> lane_ctx{};
        std::array<KernelOutput*, lane_width> lane_out{};
        std::size_t active = 0u;
        while (begin < n && active < lane_width) {
            if (contexts[begin] != nullptr) {
                lane_ctx[active] = contexts[begin];
                lane_out[active] = &outputs[begin];
                ++active;
            }
            ++begin;
        }

        if (active == 0u) {
            continue;
        }

        const LocalIndex n_test = lane_ctx[0]->numTestDofs();
        const LocalIndex n_trial = lane_ctx[0]->numTrialDofs();
        const LocalIndex n_qpts = lane_ctx[0]->numQuadraturePoints();

        bool homogeneous = true;
        for (std::size_t lane = 1u; lane < active; ++lane) {
            homogeneous = homogeneous &&
                          (lane_ctx[lane]->numTestDofs() == n_test) &&
                          (lane_ctx[lane]->numTrialDofs() == n_trial) &&
                          (lane_ctx[lane]->numQuadraturePoints() == n_qpts);
        }
        if (!homogeneous) {
            for (std::size_t lane = 0u; lane < active; ++lane) {
                computeCell(*lane_ctx[lane], *lane_out[lane]);
            }
            continue;
        }

        for (std::size_t lane = 0u; lane < active; ++lane) {
            lane_out[lane]->reserve(n_test, n_trial, /*need_matrix=*/true, /*need_vector=*/true);
        }

        std::array<Real, lane_width> weights{};
        std::array<Real, lane_width> weighted_source{};
        std::array<Real, lane_width> phi_i_lanes{};
        std::array<Real, lane_width> grad_i0{};
        std::array<Real, lane_width> grad_i1{};
        std::array<Real, lane_width> grad_i2{};
        std::array<Real, lane_width> grad_j0{};
        std::array<Real, lane_width> grad_j1{};
        std::array<Real, lane_width> grad_j2{};
        std::array<Real, lane_width> contrib{};

        for (LocalIndex q = 0; q < n_qpts; ++q) {
            for (std::size_t lane = 0u; lane < lane_width; ++lane) {
                if (lane < active) {
                    const auto x = lane_ctx[lane]->physicalPoint(q);
                    weights[lane] = lane_ctx[lane]->integrationWeight(q);
                    weighted_source[lane] = weights[lane] * source_(x[0], x[1], x[2]);
                } else {
                    weights[lane] = Real(0);
                    weighted_source[lane] = Real(0);
                }
            }
            const vec_t wv = simd_ops::loadu(weights.data());
            const vec_t srcv = simd_ops::loadu(weighted_source.data());

            for (LocalIndex i = 0; i < n_test; ++i) {
                for (std::size_t lane = 0u; lane < lane_width; ++lane) {
                    if (lane < active) {
                        phi_i_lanes[lane] = lane_ctx[lane]->basisValue(i, q);
                        const auto gi = lane_ctx[lane]->physicalGradient(i, q);
                        grad_i0[lane] = gi[0];
                        grad_i1[lane] = gi[1];
                        grad_i2[lane] = gi[2];
                    } else {
                        phi_i_lanes[lane] = Real(0);
                        grad_i0[lane] = Real(0);
                        grad_i1[lane] = Real(0);
                        grad_i2[lane] = Real(0);
                    }
                }
                const vec_t phiv = simd_ops::loadu(phi_i_lanes.data());
                const vec_t vec_contrib_v = simd_ops::mul(srcv, phiv);
                simd_ops::storeu(contrib.data(), vec_contrib_v);
                for (std::size_t lane = 0u; lane < active; ++lane) {
                    lane_out[lane]->vectorEntry(i) += contrib[lane];
                }

                const vec_t gi0v = simd_ops::loadu(grad_i0.data());
                const vec_t gi1v = simd_ops::loadu(grad_i1.data());
                const vec_t gi2v = simd_ops::loadu(grad_i2.data());

                for (LocalIndex j = 0; j < n_trial; ++j) {
                    for (std::size_t lane = 0u; lane < lane_width; ++lane) {
                        if (lane < active) {
                            const auto gj = lane_ctx[lane]->trialPhysicalGradient(j, q);
                            grad_j0[lane] = gj[0];
                            grad_j1[lane] = gj[1];
                            grad_j2[lane] = gj[2];
                        } else {
                            grad_j0[lane] = Real(0);
                            grad_j1[lane] = Real(0);
                            grad_j2[lane] = Real(0);
                        }
                    }
                    const vec_t gj0v = simd_ops::loadu(grad_j0.data());
                    const vec_t gj1v = simd_ops::loadu(grad_j1.data());
                    const vec_t gj2v = simd_ops::loadu(grad_j2.data());

                    const vec_t dotv =
                        simd_ops::add(simd_ops::mul(gi0v, gj0v),
                                      simd_ops::add(simd_ops::mul(gi1v, gj1v),
                                                    simd_ops::mul(gi2v, gj2v)));
                    const vec_t mat_contrib_v = simd_ops::mul(wv, dotv);
                    simd_ops::storeu(contrib.data(), mat_contrib_v);
                    for (std::size_t lane = 0u; lane < active; ++lane) {
                        lane_out[lane]->matrixEntry(i, j) += contrib[lane];
                    }
                }
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

void CompositeKernel::computeCellBatch(std::span<const AssemblyContext* const> contexts,
                                       std::span<KernelOutput> outputs)
{
    const std::size_t n = std::min(contexts.size(), outputs.size());
    if (n == 0u) {
        return;
    }

    bool need_matrix = false;
    bool need_vector = false;
    for (const auto& entry : kernels_) {
        if (!entry.kernel->isVectorOnly()) need_matrix = true;
        if (!entry.kernel->isMatrixOnly()) need_vector = true;
    }

    for (std::size_t idx = 0; idx < n; ++idx) {
        const auto* ctx = contexts[idx];
        if (ctx == nullptr) {
            continue;
        }
        outputs[idx].reserve(ctx->numTestDofs(), ctx->numTrialDofs(), need_matrix, need_vector);
    }

    std::vector<KernelOutput> temp_outputs(n);

    for (const auto& entry : kernels_) {
        if (entry.kernel == nullptr) {
            continue;
        }

        for (auto& out : temp_outputs) {
            out.clear();
        }

        if (entry.kernel->supportsCellBatch()) {
            entry.kernel->computeCellBatch(contexts.subspan(0, n), std::span<KernelOutput>(temp_outputs.data(), n));
        } else {
            for (std::size_t idx = 0; idx < n; ++idx) {
                const auto* ctx = contexts[idx];
                if (ctx == nullptr) {
                    continue;
                }
                entry.kernel->computeCell(*ctx, temp_outputs[idx]);
            }
        }

        for (std::size_t idx = 0; idx < n; ++idx) {
            if (contexts[idx] == nullptr) {
                continue;
            }

            auto& out = outputs[idx];
            const auto& tmp = temp_outputs[idx];
            const auto n_test = out.n_test_dofs;
            const auto n_trial = out.n_trial_dofs;

            if (tmp.has_matrix && need_matrix) {
                for (LocalIndex i = 0; i < n_test; ++i) {
                    for (LocalIndex j = 0; j < n_trial; ++j) {
                        out.matrixEntry(i, j) += entry.scale * tmp.matrixEntry(i, j);
                    }
                }
            }

            if (tmp.has_vector && need_vector) {
                for (LocalIndex i = 0; i < n_test; ++i) {
                    out.vectorEntry(i) += entry.scale * tmp.vectorEntry(i);
                }
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
