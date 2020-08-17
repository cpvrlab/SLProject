#include "Camera.h"
/*
bool Camera::isInFrustum(SLAABBox* aabb)
{
    // check the 6 planes of the frustum
    for (auto& i : _plane)
    {
        SLfloat distance = i.distToPoint(aabb->centerWS());
        if (distance < -aabb->radiusWS())
        {
            aabb->isVisible(false);
            return false;
        }
    }
    aabb->isVisible(true);

    // Calculate squared dist. from AABB's center to viewer for blend sorting.
    SLVec3f viewToCenter(_wm.translation() - aabb->centerWS());
    aabb->sqrViewDist(viewToCenter.lengthSqr());
    return true;
}


void Camera::setFrustumPlanes()
{
    // TODO: Find better solution/alternative without GL
    SLGLState* stateGL = SLGLState::instance();
    SLMat4f    A(stateGL->projectionMatrix * stateGL->viewMatrix);

    // set the A,B,C & D coeffitient for each plane
    _plane[0].setCoefficients(-A.m(1) + A.m(3),
                              -A.m(5) + A.m(7),
                              -A.m(9) + A.m(11),
                              -A.m(13) + A.m(15));
    _plane[1].setCoefficients(A.m(1) + A.m(3),
                              A.m(5) + A.m(7),
                              A.m(9) + A.m(11),
                              A.m(13) + A.m(15));
    _plane[2].setCoefficients(A.m(0) + A.m(3),
                              A.m(4) + A.m(7),
                              A.m(8) + A.m(11),
                              A.m(12) + A.m(15));
    _plane[3].setCoefficients(-A.m(0) + A.m(3),
                              -A.m(4) + A.m(7),
                              -A.m(8) + A.m(11),
                              -A.m(12) + A.m(15));
    _plane[4].setCoefficients(A.m(2) + A.m(3),
                              A.m(6) + A.m(7),
                              A.m(10) + A.m(11),
                              A.m(14) + A.m(15));
    _plane[5].setCoefficients(-A.m(2) + A.m(3),
                              -A.m(6) + A.m(7),
                              -A.m(10) + A.m(11),
                              -A.m(14) + A.m(15));
}
*/
