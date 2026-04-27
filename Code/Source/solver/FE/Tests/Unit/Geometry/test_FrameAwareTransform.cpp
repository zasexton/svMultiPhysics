#include "FE/Geometry/FrameAwareTransform.h"

#include <gtest/gtest.h>

#include <cmath>

using namespace svmp::FE;
using namespace svmp::FE::geometry;

namespace {

math::Matrix<Real, 3, 3> diag(Real a, Real b, Real c)
{
    math::Matrix<Real, 3, 3> J{};
    J(0, 0) = a;
    J(1, 1) = b;
    J(2, 2) = c;
    return J;
}

math::Matrix<Real, 3, 3> rotationZ90()
{
    math::Matrix<Real, 3, 3> R{};
    R(0, 1) = Real(-1);
    R(1, 0) = Real(1);
    R(2, 2) = Real(1);
    return R;
}

void expectVectorNear(const math::Vector<Real, 3>& a,
                      const math::Vector<Real, 3>& b,
                      Real tol = Real(1e-12))
{
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(a[i], b[i], tol);
    }
}

void expectMatrixNear(const math::Matrix<Real, 3, 3>& a,
                      const math::Matrix<Real, 3, 3>& b,
                      Real tol = Real(1e-12))
{
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(a(i, j), b(i, j), tol);
        }
    }
}

} // namespace

TEST(FrameAwareTransform, DeclaresMovingGeometrySemanticsForFieldFamilies)
{
    const auto h1 = FrameAwareTransform::semantics(FEFieldTransformFamily::H1Vector);
    EXPECT_TRUE(h1.component_value);
    EXPECT_FALSE(h1.uses_piola);
    EXPECT_FALSE(h1.preserves_normal_flux);

    const auto hdiv = FrameAwareTransform::semantics(FEFieldTransformFamily::HDiv);
    EXPECT_TRUE(hdiv.uses_piola);
    EXPECT_TRUE(hdiv.preserves_normal_flux);
    EXPECT_FALSE(hdiv.preserves_tangential_circulation);

    const auto hcurl = FrameAwareTransform::semantics(FEFieldTransformFamily::HCurl);
    EXPECT_TRUE(hcurl.uses_piola);
    EXPECT_FALSE(hcurl.preserves_normal_flux);
    EXPECT_TRUE(hcurl.preserves_tangential_circulation);

    EXPECT_STREQ(FrameAwareTransform::familyName(FEFieldTransformFamily::ShellDirector),
                 "ShellDirector");
}

TEST(FrameAwareTransform, H1ScalarAndVectorValuesAreNotPiolaTransformed)
{
    const auto frame = DeformationFrame::fromJacobian(diag(2.0, 3.0, 4.0));
    EXPECT_DOUBLE_EQ(FrameAwareTransform::pushForwardScalar(4.5), 4.5);
    EXPECT_DOUBLE_EQ(FrameAwareTransform::pullBackScalar(-2.0), -2.0);

    const math::Vector<Real, 3> v{1.0, -2.0, 3.0};
    expectVectorNear(FrameAwareTransform::pushForwardValue(FEFieldTransformFamily::H1Vector, v, frame), v);
    expectVectorNear(FrameAwareTransform::pullBackValue(FEFieldTransformFamily::H1Vector, v, frame), v);
}

TEST(FrameAwareTransform, HDivPiolaRoundtripPreservesNormalFlux)
{
    const auto frame = DeformationFrame::fromJacobian(diag(2.0, 3.0, 4.0));
    const math::Vector<Real, 3> v_ref{5.0, -1.0, 2.0};

    const auto v_cur = FrameAwareTransform::hdivPushForward(v_ref, frame);
    const auto v_back = FrameAwareTransform::hdivPullBack(v_cur, frame);
    expectVectorNear(v_back, v_ref);

    const math::Vector<Real, 3> N_ref{1.0, 0.0, 0.0};
    const Real ref_measure = 5.0;
    const auto surface = FrameAwareTransform::nansonSurfaceTransform(N_ref, ref_measure, frame);

    const Real flux_ref = v_ref.dot(N_ref) * ref_measure;
    const Real flux_cur = v_cur.dot(surface.normal) * surface.measure;
    EXPECT_NEAR(flux_cur, flux_ref, 1e-12);
}

TEST(FrameAwareTransform, HCurlPiolaRoundtripPreservesTangentialCirculation)
{
    const auto frame = DeformationFrame::fromJacobian(diag(2.0, 3.0, 4.0));
    const math::Vector<Real, 3> v_ref{2.0, 4.0, 6.0};

    const auto v_cur = FrameAwareTransform::hcurlPushForward(v_ref, frame);
    const auto v_back = FrameAwareTransform::hcurlPullBack(v_cur, frame);
    expectVectorNear(v_back, v_ref);

    const math::Vector<Real, 3> t_ref{0.0, 1.0, 0.0};
    const Real ref_length = 7.0;
    const auto t_current_unnormalized = frame.J * t_ref;
    const auto t_current = t_current_unnormalized / t_current_unnormalized.norm();
    const Real current_length = t_current_unnormalized.norm() * ref_length;

    const Real circ_ref = v_ref.dot(t_ref) * ref_length;
    const Real circ_cur = v_cur.dot(t_current) * current_length;
    EXPECT_NEAR(circ_cur, circ_ref, 1e-12);
}

TEST(FrameAwareTransform, TensorTransformsRoundtripForRigidRotationAndStretch)
{
    const auto stretch = DeformationFrame::fromJacobian(diag(2.0, 3.0, 4.0));
    const auto rotation = DeformationFrame::fromJacobian(rotationZ90());

    math::Matrix<Real, 3, 3> T{};
    T(0, 0) = 1.0;
    T(0, 1) = 2.0;
    T(1, 0) = -3.0;
    T(1, 1) = 4.0;
    T(2, 2) = 5.0;

    for (const auto transform : {TensorFrameTransform::Covariant,
                                 TensorFrameTransform::Contravariant,
                                 TensorFrameTransform::Piola,
                                 TensorFrameTransform::InversePiola}) {
        const auto stretched = FrameAwareTransform::pushForwardTensor(transform, T, stretch);
        const auto stretch_back = FrameAwareTransform::pullBackTensor(transform, stretched, stretch);
        expectMatrixNear(stretch_back, T, 1e-11);

        const auto rotated = FrameAwareTransform::pushForwardTensor(transform, T, rotation);
        const auto rotate_back = FrameAwareTransform::pullBackTensor(transform, rotated, rotation);
        expectMatrixNear(rotate_back, T, 1e-12);
    }
}

TEST(FrameAwareTransform, ProjectorsAndDirectionalConstraintsRotateWithFrame)
{
    const auto frame = DeformationFrame::fromJacobian(rotationZ90());
    const math::Vector<Real, 3> N_ref{1.0, 0.0, 0.0};
    const math::Vector<Real, 3> t_ref{0.0, 1.0, 0.0};
    const math::Vector<Real, 3> value{2.0, 3.0, 4.0};

    const auto normal = FrameAwareTransform::transformDirectionalComponent(
        DirectionalComponent::Normal, N_ref, frame);
    ASSERT_TRUE(normal.valid);
    expectVectorNear(normal.direction, math::Vector<Real, 3>{0.0, 1.0, 0.0});
    expectVectorNear(normal.projector * value, math::Vector<Real, 3>{0.0, 3.0, 0.0});

    const auto tangential = FrameAwareTransform::transformDirectionalComponent(
        DirectionalComponent::Tangential, N_ref, frame);
    ASSERT_TRUE(tangential.valid);
    expectVectorNear(tangential.projector * value, math::Vector<Real, 3>{2.0, 0.0, 4.0});

    const auto tangent0 = FrameAwareTransform::transformDirectionalComponent(
        DirectionalComponent::SurfaceTangent0, N_ref, frame, t_ref);
    ASSERT_TRUE(tangent0.valid);
    EXPECT_NEAR(tangent0.direction.norm(), 1.0, 1e-12);
    EXPECT_NEAR(tangent0.direction.dot(normal.direction), 0.0, 1e-12);
}

TEST(FrameAwareTransform, AllDirectionalComponentsHaveFrameAwareProjectors)
{
    const auto frame = DeformationFrame::fromJacobian(rotationZ90());
    const math::Vector<Real, 3> N_ref{1.0, 0.0, 0.0};
    const math::Vector<Real, 3> t_ref{0.0, 1.0, 0.0};
    const math::Vector<Real, 3> director_ref{1.0, 1.0, 0.0};
    const math::Vector<Real, 3> value{2.0, 3.0, 4.0};

    const auto full = FrameAwareTransform::transformDirectionalComponent(
        DirectionalComponent::Full, N_ref, frame);
    ASSERT_TRUE(full.valid);
    EXPECT_STREQ(full.semantic_name.c_str(), "Full");
    expectVectorNear(full.projector * value, value);

    const auto interface_normal = FrameAwareTransform::transformDirectionalComponent(
        DirectionalComponent::InterfaceNormal, N_ref, frame);
    ASSERT_TRUE(interface_normal.valid);
    EXPECT_STREQ(interface_normal.semantic_name.c_str(), "InterfaceNormal");
    expectVectorNear(interface_normal.direction, math::Vector<Real, 3>{0.0, 1.0, 0.0});
    expectVectorNear(interface_normal.projector * value, math::Vector<Real, 3>{0.0, 3.0, 0.0});

    const auto interface_tangential = FrameAwareTransform::transformDirectionalComponent(
        DirectionalComponent::InterfaceTangential, N_ref, frame);
    ASSERT_TRUE(interface_tangential.valid);
    EXPECT_STREQ(interface_tangential.semantic_name.c_str(), "InterfaceTangential");
    expectVectorNear(interface_tangential.direction, interface_normal.direction);
    expectVectorNear(interface_tangential.projector * value, math::Vector<Real, 3>{2.0, 0.0, 4.0});

    const auto tangent0 = FrameAwareTransform::transformDirectionalComponent(
        DirectionalComponent::SurfaceTangent0, N_ref, frame, t_ref);
    const auto tangent1 = FrameAwareTransform::transformDirectionalComponent(
        DirectionalComponent::SurfaceTangent1, N_ref, frame, t_ref);
    ASSERT_TRUE(tangent0.valid);
    ASSERT_TRUE(tangent1.valid);
    EXPECT_STREQ(tangent1.semantic_name.c_str(), "SurfaceTangent1");
    EXPECT_NEAR(tangent1.direction.norm(), 1.0, 1e-12);
    EXPECT_NEAR(tangent1.direction.dot(interface_normal.direction), 0.0, 1e-12);
    EXPECT_NEAR(tangent1.direction.dot(tangent0.direction), 0.0, 1e-12);
    expectVectorNear(tangent1.direction, math::Vector<Real, 3>{0.0, 0.0, 1.0});

    const auto shell_director = FrameAwareTransform::transformDirectionalComponent(
        DirectionalComponent::ShellDirector, director_ref, frame);
    ASSERT_TRUE(shell_director.valid);
    EXPECT_STREQ(shell_director.semantic_name.c_str(), "ShellDirector");
    EXPECT_NEAR(shell_director.direction.norm(), 1.0, 1e-12);
    expectVectorNear(shell_director.direction,
                     math::Vector<Real, 3>{-1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0), 0.0});
    expectVectorNear(shell_director.projector * shell_director.direction, shell_director.direction);
}

TEST(FrameAwareTransform, SurfaceAndShellTransformsUseSharedFrameUtilities)
{
    const auto frame = DeformationFrame::fromJacobian(diag(2.0, 3.0, 4.0));

    const math::Vector<Real, 3> N_ref{0.0, 0.0, 1.0};
    const auto surface = FrameAwareTransform::nansonSurfaceTransform(N_ref, 2.0, frame);
    expectVectorNear(surface.normal, math::Vector<Real, 3>{0.0, 0.0, 1.0});
    EXPECT_NEAR(surface.measure, 12.0, 1e-12);
    expectVectorNear(surface.oriented_measure_vector, math::Vector<Real, 3>{0.0, 0.0, 12.0});

    const auto shell_director =
        FrameAwareTransform::pushForwardShellDirector(math::Vector<Real, 3>{1.0, 1.0, 0.0}, frame);
    EXPECT_NEAR(shell_director.norm(), 1.0, 1e-12);
    expectVectorNear(shell_director,
                     math::Vector<Real, 3>{2.0 / std::sqrt(13.0), 3.0 / std::sqrt(13.0), 0.0});

    const auto surface_frame = FrameAwareTransform::surfaceFrame(surface.normal,
                                                                math::Vector<Real, 3>{1.0, 0.0, 0.0},
                                                                shell_director);
    ASSERT_TRUE(surface_frame.valid);
    EXPECT_NEAR(surface_frame.normal.dot(surface_frame.tangent0), 0.0, 1e-12);
    EXPECT_NEAR(surface_frame.normal.dot(surface_frame.tangent1), 0.0, 1e-12);
    EXPECT_NEAR(surface_frame.tangent0.dot(surface_frame.tangent1), 0.0, 1e-12);
}
