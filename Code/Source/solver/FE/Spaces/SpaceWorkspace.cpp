/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/SpaceWorkspace.h"

namespace svmp {
namespace FE {
namespace spaces {

SpaceWorkspace& SpaceWorkspace::local() {
    thread_local SpaceWorkspace ws;
    return ws;
}

std::vector<Real>& SpaceWorkspace::get_vector(std::size_t size, int slot) {
    if (slot < 0) {
        slot = 0;
    }
    const std::size_t islot = static_cast<std::size_t>(slot);
    if (vectors_.size() <= islot) {
        vectors_.resize(islot + 1);
    }
    auto& v = vectors_[islot];
    if (v.capacity() < size) {
        v.clear();
        v.reserve(size);
    }
    v.resize(size);
    return v;
}

} // namespace spaces
} // namespace FE
} // namespace svmp

