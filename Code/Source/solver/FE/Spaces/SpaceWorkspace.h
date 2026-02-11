/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_SPACEWORKSPACE_H
#define SVMP_FE_SPACES_SPACEWORKSPACE_H

/**
 * @file SpaceWorkspace.h
 * @brief Thread-local workspace for temporary FE vectors/matrices
 *
 * This utility provides reusable buffers to reduce dynamic allocations in
 * space operations. It is intentionally minimal and header-only in terms
 * of interface; implementation lives in SpaceWorkspace.cpp.
 */

#include "Core/Types.h"
#include <deque>
#include <vector>

namespace svmp {
namespace FE {
namespace spaces {

class SpaceWorkspace {
public:
    /// Get a reusable std::vector<Real> of at least the requested size
    std::vector<Real>& get_vector(std::size_t size, int slot = 0);

    /// Thread-local workspace accessor
    static SpaceWorkspace& local();

private:
    std::deque<std::vector<Real>> vectors_;
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_SPACEWORKSPACE_H

