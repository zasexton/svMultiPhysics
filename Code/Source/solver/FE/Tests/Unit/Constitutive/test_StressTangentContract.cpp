/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Constitutive/StressTangentContract.h"

namespace svmp {
namespace FE {
namespace constitutive {
namespace test {

TEST(StressTangentContract, CoversReferenceAndSpatialStressMeasures)
{
    EXPECT_TRUE(isReferenceStress(StressMeasure::FirstPiolaKirchhoff));
    EXPECT_TRUE(isReferenceStress(StressMeasure::SecondPiolaKirchhoff));
    EXPECT_TRUE(isSpatialStress(StressMeasure::Cauchy));
    EXPECT_TRUE(isSpatialStress(StressMeasure::Kirchhoff));

    EXPECT_EQ(naturalUpdateFrame(StressMeasure::FirstPiolaKirchhoff),
              ConstitutiveUpdateFrame::Reference);
    EXPECT_EQ(naturalUpdateFrame(StressMeasure::Kirchhoff),
              ConstitutiveUpdateFrame::Current);
}

TEST(StressTangentContract, RejectsFrameInconsistentTangentContracts)
{
    StressTangentContract pk1{};
    pk1.stress_measure = StressMeasure::FirstPiolaKirchhoff;
    pk1.tangent_measure = TangentMeasure::Material;
    pk1.input_measure = KinematicInputMeasure::DeformationGradient;
    pk1.update_frame = ConstitutiveUpdateFrame::Reference;
    pk1.provides_consistent_tangent = true;
    EXPECT_TRUE(isStressTangentContractValid(pk1));

    pk1.tangent_measure = TangentMeasure::Spatial;
    EXPECT_FALSE(isStressTangentContractValid(pk1));

    StressTangentContract cauchy{};
    cauchy.stress_measure = StressMeasure::Cauchy;
    cauchy.tangent_measure = TangentMeasure::Spatial;
    cauchy.update_frame = ConstitutiveUpdateFrame::Current;
    EXPECT_TRUE(isStressTangentContractValid(cauchy));

    cauchy.tangent_measure = TangentMeasure::Material;
    EXPECT_FALSE(isStressTangentContractValid(cauchy));
}

TEST(StressTangentContract, NamesAreStableForRestartAndDiagnostics)
{
    EXPECT_STREQ(stressMeasureName(StressMeasure::FirstPiolaKirchhoff),
                 "FirstPiolaKirchhoff");
    EXPECT_STREQ(stressMeasureName(StressMeasure::SecondPiolaKirchhoff),
                 "SecondPiolaKirchhoff");
    EXPECT_STREQ(stressMeasureName(StressMeasure::Cauchy), "Cauchy");
    EXPECT_STREQ(stressMeasureName(StressMeasure::Kirchhoff), "Kirchhoff");
    EXPECT_STREQ(tangentMeasureName(TangentMeasure::Mixed), "Mixed");
    EXPECT_STREQ(kinematicInputMeasureName(KinematicInputMeasure::AlmansiStrain),
                 "AlmansiStrain");
    EXPECT_STREQ(constitutiveUpdateFrameName(ConstitutiveUpdateFrame::Current),
                 "Current");
}

} // namespace test
} // namespace constitutive
} // namespace FE
} // namespace svmp
