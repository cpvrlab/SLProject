#ifndef SLFRUSTUM
#define SLFRUSTUM

class SLFrustum
{
public:
    static void viewToFrustumPlanes(SLPlane * planes, const SLMat4f &P, const SLMat4f &V);
    static void viewToFrustumPlanes(SLPlane * planes, const SLMat4f &A);
    static void getPointsEyeSpace(SLVec3f * points, float fovV, float ratio, float near, float far);
};


#endif

