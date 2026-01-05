/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Index.h"

#include <atomic>

namespace svmp {
namespace FE {
namespace forms {

int Index::nextId() noexcept
{
    static std::atomic<int> counter{0};
    return counter.fetch_add(1, std::memory_order_relaxed);
}

Index::Index(std::string name, IndexSet set)
    : id_(nextId()), name_(std::move(name)), set_(set)
{
}

} // namespace forms
} // namespace FE
} // namespace svmp

