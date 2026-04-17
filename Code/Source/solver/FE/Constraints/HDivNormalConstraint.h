/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_HDIVNORMALCONSTRAINT_H
#define SVMP_FE_CONSTRAINTS_HDIVNORMALCONSTRAINT_H

/**
 * @file HDivNormalConstraint.h
 * @brief Essential (strong) boundary constraint for H(div) normal traces
 *
 * This constraint enforces a normal boundary condition:
 *   n · B = g(x,t)  on boundary marker Γ
 * by constraining the H(div) face-trace coefficients associated with the
 * marked boundary facets. The prescribed scalar trace data is interpolated or
 * projected in the face trace space before being lowered into affine
 * constraints.
 */

#include "Constraints/SystemConstraint.h"
#include "Forms/FormExpr.h"

#include <memory>
#include <vector>

namespace svmp {
namespace FE {
namespace geometry {
class GeometryMapping;
}
namespace spaces {
class TraceSpace;
}
namespace constraints {

class HDivNormalConstraint final : public ISystemConstraint {
public:
    HDivNormalConstraint(FieldId field, int boundary_marker);
    HDivNormalConstraint(FieldId field, int boundary_marker, forms::FormExpr value);

    void apply(const systems::FESystem& system, AffineConstraints& constraints) override;

    bool updateValues(const systems::FESystem& system,
                      AffineConstraints& constraints,
                      double time,
                      double dt) override;

    [[nodiscard]] bool isTimeDependent() const noexcept override { return is_time_dependent_; }

private:
    struct FaceWorkItem {
        std::shared_ptr<spaces::TraceSpace> trace_space{};
        std::shared_ptr<const geometry::GeometryMapping> cell_mapping{};
        std::vector<GlobalIndex> global_dofs{};
    };

    void buildFaceWorkItems_(const systems::FESystem& system);
    std::vector<Real> faceValues_(const FaceWorkItem& item, double time, double dt) const;

    FieldId field_{INVALID_FIELD_ID};
    int boundary_marker_{-1};
    forms::FormExpr value_{};
    bool is_time_dependent_{false};
    std::vector<GlobalIndex> dofs_{};
    std::vector<FaceWorkItem> face_work_items_{};
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_HDIVNORMALCONSTRAINT_H
