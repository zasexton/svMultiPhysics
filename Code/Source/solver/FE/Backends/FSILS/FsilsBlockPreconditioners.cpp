/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/FSILS/FsilsBlockPreconditioners.h"

#include "Core/FEException.h"
#include "Core/Logger.h"

#include "Backends/FSILS/liner_solver/fils_struct.hpp"

#include "Array.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <utility>

namespace svmp {
namespace FE {
namespace backends {

// ---------------------------------------------------------------------------
// Dense block inverse
// ---------------------------------------------------------------------------

int invertDenseBlock(int d, const double* block, double* inverse)
{
    const std::size_t dd = static_cast<std::size_t>(d) * static_cast<std::size_t>(d);
    std::vector<double> a(block, block + dd);
    std::vector<double> inv(dd, 0.0);
    for (int i = 0; i < d; ++i) {
        inv[static_cast<std::size_t>(i * d + i)] = 1.0;
    }
    double scale = 0.0;
    for (const double v : a) {
        scale = std::max(scale, std::abs(v));
    }
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        // A vanishing (or nonfinite) block carries no information: use identity.
        std::copy(inv.begin(), inv.end(), inverse);
        return d;
    }
    const double tiny = std::numeric_limits<double>::epsilon() * scale * static_cast<double>(d);
    int regularized = 0;
    for (int col = 0; col < d; ++col) {
        int piv = col;
        double best = std::abs(a[static_cast<std::size_t>(col * d + col)]);
        for (int r = col + 1; r < d; ++r) {
            const double v = std::abs(a[static_cast<std::size_t>(r * d + col)]);
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        if (piv != col) {
            for (int c = 0; c < d; ++c) {
                std::swap(a[static_cast<std::size_t>(col * d + c)], a[static_cast<std::size_t>(piv * d + c)]);
                std::swap(inv[static_cast<std::size_t>(col * d + c)], inv[static_cast<std::size_t>(piv * d + c)]);
            }
        }
        double pivot = a[static_cast<std::size_t>(col * d + col)];
        if (!(std::abs(pivot) > tiny)) {
            pivot = (pivot < 0.0) ? -tiny : tiny;
            a[static_cast<std::size_t>(col * d + col)] = pivot;
            ++regularized;
        }
        const double inv_pivot = 1.0 / pivot;
        for (int c = 0; c < d; ++c) {
            a[static_cast<std::size_t>(col * d + c)] *= inv_pivot;
            inv[static_cast<std::size_t>(col * d + c)] *= inv_pivot;
        }
        for (int r = 0; r < d; ++r) {
            if (r == col) {
                continue;
            }
            const double f = a[static_cast<std::size_t>(r * d + col)];
            if (f == 0.0) {
                continue;
            }
            for (int c = 0; c < d; ++c) {
                a[static_cast<std::size_t>(r * d + c)] -= f * a[static_cast<std::size_t>(col * d + c)];
                inv[static_cast<std::size_t>(r * d + c)] -= f * inv[static_cast<std::size_t>(col * d + c)];
            }
        }
    }
    std::copy(inv.begin(), inv.end(), inverse);
    return regularized;
}

// ---------------------------------------------------------------------------
// Aggregation-multigrid right preconditioner
// ---------------------------------------------------------------------------

class FsilsKrylovPreconditioner::Applier final : public fe_fsi_linear_solver::FSILS_rightPreconditioner {
public:
    explicit Applier(const FsilsKrylovPreconditioner& owner) : owner_(owner) {}
    void apply(const Array<double>& in, Array<double>& out) const override { owner_.apply(in, out); }

private:
    const FsilsKrylovPreconditioner& owner_;
};

FsilsKrylovPreconditioner::FsilsKrylovPreconditioner()
    : applier_(std::make_unique<Applier>(*this))
{
}

FsilsKrylovPreconditioner::~FsilsKrylovPreconditioner() = default;

void FsilsKrylovPreconditioner::setAmgOptions(const FsilsAmgOptions& options)
{
    amg_options_ = options;
}

void FsilsKrylovPreconditioner::setNodeKeys(std::vector<std::uint64_t> owned_keys)
{
    node_keys_ = std::move(owned_keys);
}

fe_fsi_linear_solver::FSILS_rightPreconditionerHook FsilsKrylovPreconditioner::makeHook()
{
    fe_fsi_linear_solver::FSILS_rightPreconditionerHook hook;
    hook.prepare = [this](const fe_fsi_linear_solver::FSILS_lhsType& lhs, int dof, const Array<double>& Val) {
        return prepare(lhs, dof, Val);
    };
    hook.finish = [this](int iterations, double residual_reduction, bool converged) {
        finish(iterations, residual_reduction, converged);
    };
    return hook;
}

const fe_fsi_linear_solver::FSILS_rightPreconditioner*
FsilsKrylovPreconditioner::prepare(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                   int dof,
                                   const Array<double>& Val)
{
    if (dof <= 0 || lhs.nNo <= 0) {
        return nullptr;
    }
    const auto t0 = std::chrono::steady_clock::now();
    dof_ = dof;
    n_owned_ = static_cast<int>(lhs.mynNo);
    if (!amg_) {
        amg_ = std::make_unique<FsilsAmgHierarchy>();
    }
    const bool have_keys = node_keys_.size() >= static_cast<std::size_t>(lhs.mynNo);
    const std::span<const std::uint64_t> keys =
        have_keys ? std::span<const std::uint64_t>(node_keys_) : std::span<const std::uint64_t>{};
    amg_->build(lhs, dof, Val.data(), keys, amg_options_);
    stats_.regularized_pivots = amg_->stats().regularized_pivots;
    stats_.last_setup_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    stats_.setup_seconds += stats_.last_setup_seconds;
    return applier_.get();
}

void FsilsKrylovPreconditioner::apply(const Array<double>& in, Array<double>& out) const
{
    const std::size_t owned = static_cast<std::size_t>(n_owned_) * static_cast<std::size_t>(dof_);
    const double* src = in.data();
    double* dst = out.data();
    work_.resize(owned);
    std::copy(src, src + owned, work_.data());
    amg_->apply(work_.data(), dst);
    const std::size_t total = static_cast<std::size_t>(out.size());
    for (std::size_t i = owned; i < total; ++i) {
        dst[i] = 0.0;
    }
}

void FsilsKrylovPreconditioner::applyForTesting(const Array<double>& in, Array<double>& out) const
{
    apply(in, out);
}

void FsilsKrylovPreconditioner::finish(int iterations, double residual_reduction, bool converged)
{
    ++stats_.solves;
    stats_.last_iterations = iterations;

    std::ostringstream oss;
    oss << "FsilsKrylovPreconditioner: diagnostic=fsils_right_preconditioner"
        << " kind=amg"
        << " iterations=" << iterations
        << " converged=" << (converged ? 1 : 0)
        << " residual_reduction=" << residual_reduction
        << " setup_s=" << stats_.last_setup_seconds
        << " regularized_pivots=" << stats_.regularized_pivots
        << " solves=" << stats_.solves;
    if (amg_) {
        oss << " amg " << amg_->summary();
    }
    FE_LOG_INFO(oss.str());
}

} // namespace backends
} // namespace FE
} // namespace svmp
