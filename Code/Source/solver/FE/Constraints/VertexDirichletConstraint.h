#ifndef SVMP_FE_CONSTRAINTS_VERTEXDIRICHLETCONSTRAINT_H
#define SVMP_FE_CONSTRAINTS_VERTEXDIRICHLETCONSTRAINT_H

/**
 * @file VertexDirichletConstraint.h
 * @brief Setup-time Dirichlet constraints specified by mesh vertex IDs
 */

#include "Constraints/SystemConstraint.h"
#include "Core/Types.h"

#include <vector>

namespace svmp {
namespace FE {
namespace constraints {

struct VertexDirichletValue {
    GlobalIndex vertex_id{-1};
    Real value{0.0};
};

enum class VertexIdMode {
    GlobalVertexGid,
    LocalVertexId,
};

class VertexDirichletConstraint final : public ISystemConstraint {
public:
    VertexDirichletConstraint(FieldId field,
                              std::vector<VertexDirichletValue> values,
                              VertexIdMode mode = VertexIdMode::GlobalVertexGid);

    void apply(const systems::FESystem& system, AffineConstraints& constraints) override;

    bool updateValues(const systems::FESystem& system,
                      AffineConstraints& constraints,
                      double time,
                      double dt) override;

    [[nodiscard]] bool isTimeDependent() const noexcept override { return false; }

private:
    FieldId field_{INVALID_FIELD_ID};
    std::vector<VertexDirichletValue> values_{};
    VertexIdMode mode_{VertexIdMode::GlobalVertexGid};
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_VERTEXDIRICHLETCONSTRAINT_H
