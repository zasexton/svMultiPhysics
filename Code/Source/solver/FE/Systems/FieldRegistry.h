/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_FIELDREGISTRY_H
#define SVMP_FE_SYSTEMS_FIELDREGISTRY_H

#include "Core/Types.h"
#include "Core/FEException.h"

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {

namespace spaces {
class FunctionSpace;
}

namespace systems {

struct FieldSpec {
    std::string name;
    std::shared_ptr<const spaces::FunctionSpace> space;
    int components{1};
};

struct FieldRecord {
    FieldId id{INVALID_FIELD_ID};
    std::string name;
    std::shared_ptr<const spaces::FunctionSpace> space;
    int components{1};

    // Transient metadata (derived from registered kernels containing `dt(...)`)
    bool time_dependent{false};
    int max_time_derivative_order{0};
};

class FieldRegistry {
public:
    FieldId add(FieldSpec spec);
    [[nodiscard]] const FieldRecord& get(FieldId id) const;
    void markTimeDependent(FieldId id, int max_order);
    [[nodiscard]] FieldId findByName(std::string_view name) const noexcept;
    [[nodiscard]] bool has(FieldId id) const noexcept;
    [[nodiscard]] std::size_t size() const noexcept { return fields_.size(); }
    [[nodiscard]] const std::vector<FieldRecord>& records() const noexcept { return fields_; }

private:
    std::vector<FieldRecord> fields_;
    std::unordered_map<std::string, FieldId> name_to_id_;
    FieldId next_id_{0};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FIELDREGISTRY_H
