/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TimeStepping/TimeHistory.h"

#include "Core/FEException.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace svmp {
namespace FE {
namespace timestepping {

namespace {

void copySpan(std::span<Real> dst, std::span<const Real> src)
{
    FE_CHECK_ARG(dst.size() == src.size(), "TimeHistory: size mismatch in copySpan");
    std::copy(src.begin(), src.end(), dst.begin());
}

void repackVector(backends::GenericVector& dst, backends::GenericVector& src)
{
    FE_CHECK_ARG(dst.size() == src.size(), "TimeHistory: size mismatch in repackVector");

    auto src_view = src.createAssemblyView();
    auto dst_view = dst.createAssemblyView();
    FE_CHECK_NOT_NULL(src_view.get(), "TimeHistory: repack src view");
    FE_CHECK_NOT_NULL(dst_view.get(), "TimeHistory: repack dst view");

    dst_view->beginAssemblyPhase();
    for (GlobalIndex dof = 0; dof < dst.size(); ++dof) {
        dst_view->addVectorEntry(dof, src_view->getVectorEntry(dof), assembly::AddMode::Insert);
    }
    dst_view->finalizeAssembly();
}

} // namespace

TimeHistory::TimeHistory(std::unique_ptr<backends::GenericVector> u,
                         std::unique_ptr<backends::GenericVector> u_prev,
                         std::unique_ptr<backends::GenericVector> u_prev2)
{
    std::vector<std::unique_ptr<backends::GenericVector>> hist;
    hist.reserve(2);
    hist.push_back(std::move(u_prev));
    hist.push_back(std::move(u_prev2));
    *this = TimeHistory(std::move(u), std::move(hist));
}

TimeHistory::TimeHistory(std::unique_ptr<backends::GenericVector> u,
                         std::vector<std::unique_ptr<backends::GenericVector>> history)
    : u_(std::move(u))
    , history_(std::move(history))
{
    FE_CHECK_NOT_NULL(u_.get(), "TimeHistory::u");
    FE_THROW_IF(history_.empty(), InvalidArgumentException, "TimeHistory: history depth must be >= 1");
    for (std::size_t i = 0; i < history_.size(); ++i) {
        FE_CHECK_NOT_NULL(history_[i].get(), "TimeHistory::history");
        FE_THROW_IF(history_[i]->size() != u_->size(), InvalidArgumentException,
                    "TimeHistory: vector sizes must match");
    }

    dt_history_.assign(history_.size(), 0.0);
    refreshHistoryViews();
}

std::unique_ptr<backends::GenericVector>
TimeHistory::cloneLike(const backends::BackendFactory& factory, GlobalIndex size)
{
    auto v = factory.createVector(size);
    FE_CHECK_NOT_NULL(v.get(), "TimeHistory::allocate vector");
    return v;
}

TimeHistory TimeHistory::allocate(const backends::BackendFactory& factory,
                                  GlobalIndex size,
                                  int history_depth,
                                  bool allocate_second_order_state)
{
    FE_THROW_IF(history_depth <= 0, InvalidArgumentException,
                "TimeHistory::allocate: history_depth must be > 0");

    auto u = cloneLike(factory, size);
    u->zero();

    std::vector<std::unique_ptr<backends::GenericVector>> hist;
    hist.reserve(static_cast<std::size_t>(history_depth));
    for (int i = 0; i < history_depth; ++i) {
        auto v = cloneLike(factory, size);
        v->zero();
        hist.push_back(std::move(v));
    }

    TimeHistory history(std::move(u), std::move(hist));
    if (allocate_second_order_state) {
        history.u_dot_ = cloneLike(factory, size);
        history.u_ddot_ = cloneLike(factory, size);
        history.u_dot_->zero();
        history.u_ddot_->zero();
    }
    return history;
}

void TimeHistory::ensureSecondOrderState(const backends::BackendFactory& factory)
{
    if (hasSecondOrderState()) {
        return;
    }

    const auto size = u().size();
    u_dot_ = cloneLike(factory, size);
    u_ddot_ = cloneLike(factory, size);
    u_dot_->zero();
    u_ddot_->zero();
}

backends::GenericVector& TimeHistory::u()
{
    FE_CHECK_NOT_NULL(u_.get(), "TimeHistory::u");
    return *u_;
}

const backends::GenericVector& TimeHistory::u() const
{
    FE_CHECK_NOT_NULL(u_.get(), "TimeHistory::u");
    return *u_;
}

backends::GenericVector& TimeHistory::uPrev()
{
    return uPrevK(1);
}

const backends::GenericVector& TimeHistory::uPrev() const
{
    return uPrevK(1);
}

backends::GenericVector& TimeHistory::uPrev2()
{
    return uPrevK(2);
}

const backends::GenericVector& TimeHistory::uPrev2() const
{
    return uPrevK(2);
}

backends::GenericVector& TimeHistory::uPrevK(int k)
{
    FE_THROW_IF(k <= 0, InvalidArgumentException, "TimeHistory::uPrevK: k must be >= 1");
    FE_THROW_IF(history_.size() < static_cast<std::size_t>(k), InvalidArgumentException,
                "TimeHistory::uPrevK: requested history state is not allocated");
    FE_CHECK_NOT_NULL(history_[static_cast<std::size_t>(k - 1)].get(), "TimeHistory::history");
    return *history_[static_cast<std::size_t>(k - 1)];
}

const backends::GenericVector& TimeHistory::uPrevK(int k) const
{
    FE_THROW_IF(k <= 0, InvalidArgumentException, "TimeHistory::uPrevK: k must be >= 1");
    FE_THROW_IF(history_.size() < static_cast<std::size_t>(k), InvalidArgumentException,
                "TimeHistory::uPrevK: requested history state is not allocated");
    FE_CHECK_NOT_NULL(history_[static_cast<std::size_t>(k - 1)].get(), "TimeHistory::history");
    return *history_[static_cast<std::size_t>(k - 1)];
}

backends::GenericVector& TimeHistory::uDot()
{
    FE_THROW_IF(!u_dot_, InvalidArgumentException, "TimeHistory::uDot: uDot is not allocated");
    return *u_dot_;
}

const backends::GenericVector& TimeHistory::uDot() const
{
    FE_THROW_IF(!u_dot_, InvalidArgumentException, "TimeHistory::uDot: uDot is not allocated");
    return *u_dot_;
}

backends::GenericVector& TimeHistory::uDDot()
{
    FE_THROW_IF(!u_ddot_, InvalidArgumentException, "TimeHistory::uDDot: uDDot is not allocated");
    return *u_ddot_;
}

const backends::GenericVector& TimeHistory::uDDot() const
{
    FE_THROW_IF(!u_ddot_, InvalidArgumentException, "TimeHistory::uDDot: uDDot is not allocated");
    return *u_ddot_;
}

std::span<Real> TimeHistory::uSpan()
{
    return u().localSpan();
}

std::span<const Real> TimeHistory::uSpan() const
{
    return u().localSpan();
}

std::span<const Real> TimeHistory::uPrevSpan() const
{
    return uPrev().localSpan();
}

std::span<const Real> TimeHistory::uPrev2Span() const
{
    return uPrev2().localSpan();
}

std::span<const Real> TimeHistory::uPrevKSpan(int k) const
{
    return uPrevK(k).localSpan();
}

std::span<Real> TimeHistory::uDotSpan()
{
    return uDot().localSpan();
}

std::span<const Real> TimeHistory::uDotSpan() const
{
    return uDot().localSpan();
}

std::span<Real> TimeHistory::uDDotSpan()
{
    return uDDot().localSpan();
}

std::span<const Real> TimeHistory::uDDotSpan() const
{
    return uDDot().localSpan();
}

void TimeHistory::setDtHistory(std::span<const double> dt_history)
{
    FE_THROW_IF(dt_history.size() != dt_history_.size(),
                InvalidArgumentException,
                "TimeHistory::setDtHistory: size mismatch (expected " + std::to_string(dt_history_.size()) +
                    ", got " + std::to_string(dt_history.size()) + ")");

    dt_history_.assign(dt_history.begin(), dt_history.end());
    if (!dt_history_.empty()) {
        const double v0 = dt_history_[0];
        if (v0 > 0.0 && std::isfinite(v0)) {
            dt_prev_ = v0;
        }
    }
}

bool TimeHistory::dtHistoryIsValid(int required_entries) const noexcept
{
    if (required_entries <= 0) {
        return true;
    }
    if (required_entries > static_cast<int>(dt_history_.size())) {
        return false;
    }
    for (int i = 0; i < required_entries; ++i) {
        const double v = dt_history_[static_cast<std::size_t>(i)];
        if (!(v > 0.0) || !std::isfinite(v)) {
            return false;
        }
    }
    return true;
}

void TimeHistory::updateGhosts()
{
    u().updateGhosts();
    if (u_dot_) {
        u_dot_->updateGhosts();
    }
    if (u_ddot_) {
        u_ddot_->updateGhosts();
    }
    for (auto& h : history_) {
        FE_CHECK_NOT_NULL(h.get(), "TimeHistory::updateGhosts history");
        h->updateGhosts();
    }
}

void TimeHistory::primeDtHistory(double dt_default) noexcept
{
    if (dt_history_.empty()) {
        return;
    }

    double fill = dt_default;
    if (!(fill > 0.0) || !std::isfinite(fill)) {
        fill = dt_prev_;
    }
    if (!(fill > 0.0) || !std::isfinite(fill)) {
        fill = dt_;
    }
    if (!(fill > 0.0) || !std::isfinite(fill)) {
        return;
    }

    for (auto& dt_hist : dt_history_) {
        if (!(dt_hist > 0.0) || !std::isfinite(dt_hist)) {
            dt_hist = fill;
        }
    }

    if (dt_prev_ > 0.0 && std::isfinite(dt_prev_)) {
        dt_history_[0] = dt_prev_;
    }
}

void TimeHistory::repack(const backends::BackendFactory& factory)
{
    const auto size = u().size();
    for (std::size_t i = 0; i < history_.size(); ++i) {
        FE_THROW_IF(size != history_[i]->size(),
                    InvalidArgumentException,
                    "TimeHistory::repack: vector sizes must match");
    }

    auto new_u = cloneLike(factory, size);
    repackVector(*new_u, u());

    std::unique_ptr<backends::GenericVector> new_u_dot{};
    std::unique_ptr<backends::GenericVector> new_u_ddot{};
    if (u_dot_) {
        FE_THROW_IF(size != u_dot_->size(),
                    InvalidArgumentException,
                    "TimeHistory::repack: uDot size must match u size");
        new_u_dot = cloneLike(factory, size);
        repackVector(*new_u_dot, *u_dot_);
    }
    if (u_ddot_) {
        FE_THROW_IF(size != u_ddot_->size(),
                    InvalidArgumentException,
                    "TimeHistory::repack: uDDot size must match u size");
        new_u_ddot = cloneLike(factory, size);
        repackVector(*new_u_ddot, *u_ddot_);
    }

    std::vector<std::unique_ptr<backends::GenericVector>> new_hist;
    new_hist.reserve(history_.size());
    for (std::size_t i = 0; i < history_.size(); ++i) {
        auto v = cloneLike(factory, size);
        repackVector(*v, *history_[i]);
        new_hist.push_back(std::move(v));
    }

    u_ = std::move(new_u);
    u_dot_ = std::move(new_u_dot);
    u_ddot_ = std::move(new_u_ddot);
    history_ = std::move(new_hist);
    refreshHistoryViews();
}

void TimeHistory::acceptStep(double accepted_dt)
{
    FE_THROW_IF(!(accepted_dt > 0.0), InvalidArgumentException, "TimeHistory::acceptStep: dt must be > 0");

    FE_THROW_IF(history_.empty(), InvalidArgumentException, "TimeHistory::acceptStep: no history storage");

    const auto cur = u().localSpan();

    for (std::size_t i = history_.size(); i-- > 1;) {
        copySpan(history_[i]->localSpan(), history_[i - 1]->localSpan());
    }
    copySpan(history_[0]->localSpan(), cur);

    dt_prev_ = accepted_dt;
    if (!dt_history_.empty()) {
        for (std::size_t i = dt_history_.size(); i-- > 1;) {
            dt_history_[i] = dt_history_[i - 1];
        }
        dt_history_[0] = accepted_dt;
    }
    dt_ = accepted_dt;
    time_ += accepted_dt;
    step_index_ += 1;
}

void TimeHistory::resetCurrentToPrevious()
{
    auto cur = u().localSpan();
    FE_THROW_IF(history_.empty(), InvalidArgumentException, "TimeHistory::resetCurrentToPrevious: no history storage");
    copySpan(cur, history_[0]->localSpan());
}

void TimeHistory::refreshHistoryViews()
{
    u_history_spans_.resize(history_.size());
    for (std::size_t i = 0; i < history_.size(); ++i) {
        u_history_spans_[i] = history_[i]->localSpan();
    }
    if (dt_history_.size() != history_.size()) {
        dt_history_.assign(history_.size(), 0.0);
    }
    if (!dt_history_.empty()) {
        dt_history_[0] = dt_prev_;
    }
}

} // namespace timestepping
} // namespace FE
} // namespace svmp
