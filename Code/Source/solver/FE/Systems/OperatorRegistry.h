/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_OPERATORREGISTRY_H
#define SVMP_FE_SYSTEMS_OPERATORREGISTRY_H

#include "Core/Types.h"
#include "Core/FEException.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {

namespace assembly {
class AssemblyKernel;
}

namespace systems {

using OperatorTag = std::string;
class GlobalKernel;

struct CellTerm {
    FieldId test_field{INVALID_FIELD_ID};
    FieldId trial_field{INVALID_FIELD_ID};
    std::shared_ptr<assembly::AssemblyKernel> kernel;
};

struct BoundaryTerm {
    int marker{0};
    FieldId test_field{INVALID_FIELD_ID};
    FieldId trial_field{INVALID_FIELD_ID};
    std::shared_ptr<assembly::AssemblyKernel> kernel;
};

struct InteriorFaceTerm {
    FieldId test_field{INVALID_FIELD_ID};
    FieldId trial_field{INVALID_FIELD_ID};
    std::shared_ptr<assembly::AssemblyKernel> kernel;
};

struct InterfaceFaceTerm {
    int marker{0};
    FieldId test_field{INVALID_FIELD_ID};
    FieldId trial_field{INVALID_FIELD_ID};
    std::shared_ptr<assembly::AssemblyKernel> kernel;
};

struct OperatorDefinition {
    OperatorTag tag;
    std::vector<CellTerm> cells;
    std::vector<BoundaryTerm> boundary;
    std::vector<InteriorFaceTerm> interior;
    std::vector<InterfaceFaceTerm> interface_faces;
    std::vector<std::shared_ptr<GlobalKernel>> global;
};

class OperatorRegistry {
public:
    void addOperator(OperatorTag tag);
    [[nodiscard]] bool has(const OperatorTag& tag) const noexcept;
    [[nodiscard]] OperatorDefinition& get(const OperatorTag& tag);
    [[nodiscard]] const OperatorDefinition& get(const OperatorTag& tag) const;
    [[nodiscard]] std::vector<OperatorTag> list() const;

    // ---- Snapshot/rollback for transactional installation ----

    /// Lightweight snapshot of term vector sizes for each operator.
    /// Used to roll back partial kernel registrations on failure.
    struct Snapshot {
        struct OpSizes {
            OperatorTag tag;
            std::size_t cells{0};
            std::size_t boundary{0};
            std::size_t interior{0};
            std::size_t interface_faces{0};
            std::size_t global{0};
        };
        std::vector<OpSizes> ops;
    };

    /// Capture the current state of all operators' term counts.
    [[nodiscard]] Snapshot snapshot() const;

    /// Rollback all operators to the sizes captured in a previous snapshot.
    /// Terms added after the snapshot are removed; operators added after the
    /// snapshot are removed entirely.
    void rollback(const Snapshot& snap);

private:
    std::unordered_map<OperatorTag, OperatorDefinition> ops_;
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_OPERATORREGISTRY_H
