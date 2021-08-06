#include <vector>
#include <SLPlane.h>
#include <SLMat4.h>
#include <SLFrustum.h>

void SLFrustum::viewToFrustumPlanes(SLPlane *planes, const SLMat4f &P, const SLMat4f &V)
{
    SLMat4f A = P * V;
    // Order is T B L R N F
    viewToFrustumPlanes(planes, A);
}

void SLFrustum::viewToFrustumPlanes(SLPlane *planes, const SLMat4f &A)
{
    // set the A,B,C & D coefficient for each plane
    // Order is T B L R N F
    planes[0].setCoefficients(-A.m(1) + A.m(3),
                              -A.m(5) + A.m(7),
                              -A.m(9) + A.m(11),
                              -A.m(13) + A.m(15));
    planes[1].setCoefficients(A.m(1) + A.m(3),
                              A.m(5) + A.m(7),
                              A.m(9) + A.m(11),
                              A.m(13) + A.m(15));
    planes[2].setCoefficients(A.m(0) + A.m(3),
                              A.m(4) + A.m(7),
                              A.m(8) + A.m(11),
                              A.m(12) + A.m(15));
    planes[3].setCoefficients(-A.m(0) + A.m(3),
                              -A.m(4) + A.m(7),
                              -A.m(8) + A.m(11),
                              -A.m(12) + A.m(15));
    planes[4].setCoefficients(A.m(2) + A.m(3),
                              A.m(6) + A.m(7),
                              A.m(10) + A.m(11),
                              A.m(14) + A.m(15));
    planes[5].setCoefficients(-A.m(2) + A.m(3),
                              -A.m(6) + A.m(7),
                              -A.m(10) + A.m(11),
                              -A.m(14) + A.m(15));
}

void SLFrustum::getPointsEyeSpace(SLVec3f *points, float fovV, float ratio, float near, float far)
{
    SLfloat t = tan(Utils::DEG2RAD * fovV * 0.5f) * near;   // top
    SLfloat b = -t;                                         // bottom
    SLfloat r = ratio * t;                                  // right
    SLfloat l = -r;                                         // left
    points[0] = (SLVec3f(r, t, -near));
    points[1] = (SLVec3f(r, b, -near));
    points[2] = (SLVec3f(l, t, -near));
    points[3] = (SLVec3f(l, b, -near));

    t = tan(Utils::DEG2RAD * fovV * 0.5f) * far;           // top
    b = -t;                                                // bottom
    r = ratio * t;                                         // right
    l = -r;                                                // left

    points[4] = (SLVec3f(r, t, -far));
    points[5] = (SLVec3f(r, b, -far));
    points[6] = (SLVec3f(l, t, -far));
    points[7] = (SLVec3f(l, b, -far));
}
