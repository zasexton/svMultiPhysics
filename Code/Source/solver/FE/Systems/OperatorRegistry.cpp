/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/OperatorRegistry.h"

#include <algorithm>

namespace svmp {
namespace FE {
namespace systems {

void OperatorRegistry::addOperator(OperatorTag tag)
{
    FE_THROW_IF(tag.empty(), InvalidArgumentException, "OperatorRegistry::addOperator: empty tag");
    FE_THROW_IF(ops_.count(tag) > 0, InvalidArgumentException,
                "OperatorRegistry::addOperator: operator '" + tag + "' already exists");
    OperatorDefinition def;
    def.tag = std::move(tag);
    ops_.emplace(def.tag, std::move(def));
}

bool OperatorRegistry::has(const OperatorTag& tag) const noexcept
{
    return ops_.count(tag) > 0;
}

OperatorDefinition& OperatorRegistry::get(const OperatorTag& tag)
{
    auto it = ops_.find(tag);
    FE_THROW_IF(it == ops_.end(), InvalidArgumentException,
                "OperatorRegistry::get: unknown operator '" + tag + "'");
    return it->second;
}

const OperatorDefinition& OperatorRegistry::get(const OperatorTag& tag) const
{
    auto it = ops_.find(tag);
    FE_THROW_IF(it == ops_.end(), InvalidArgumentException,
                "OperatorRegistry::get: unknown operator '" + tag + "'");
    return it->second;
}

std::vector<OperatorTag> OperatorRegistry::list() const
{
    std::vector<OperatorTag> tags;
    tags.reserve(ops_.size());
    for (const auto& kv : ops_) {
        tags.push_back(kv.first);
    }
    std::sort(tags.begin(), tags.end());
    return tags;
}

OperatorRegistry::Snapshot OperatorRegistry::snapshot() const
{
    Snapshot snap;
    snap.ops.reserve(ops_.size());
    for (const auto& [tag, def] : ops_) {
        snap.ops.push_back({
            tag,
            def.cells.size(),
            def.boundary.size(),
            def.interior.size(),
            def.interface_faces.size(),
            def.global.size()
        });
    }
    return snap;
}

void OperatorRegistry::rollback(const Snapshot& snap)
{
    // Build a set of tags that existed at snapshot time
    std::unordered_map<OperatorTag, const Snapshot::OpSizes*> snap_map;
    for (const auto& os : snap.ops) {
        snap_map[os.tag] = &os;
    }

    // Remove operators added after snapshot
    for (auto it = ops_.begin(); it != ops_.end(); ) {
        if (snap_map.count(it->first) == 0) {
            it = ops_.erase(it);
        } else {
            ++it;
        }
    }

    // Truncate term vectors back to snapshot sizes
    for (const auto& os : snap.ops) {
        auto it = ops_.find(os.tag);
        if (it == ops_.end()) continue;
        auto& def = it->second;
        if (def.cells.size() > os.cells) def.cells.resize(os.cells);
        if (def.boundary.size() > os.boundary) def.boundary.resize(os.boundary);
        if (def.interior.size() > os.interior) def.interior.resize(os.interior);
        if (def.interface_faces.size() > os.interface_faces) def.interface_faces.resize(os.interface_faces);
        if (def.global.size() > os.global) def.global.resize(os.global);
    }
}

} // namespace systems
} // namespace FE
} // namespace svmp
