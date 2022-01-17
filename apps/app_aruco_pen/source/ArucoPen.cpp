//#############################################################################
//  File:      ArucoPen.cpp
//  Date:      September 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "ArucoPen.h"
#include "app/AppArucoPen.h"
#include "app/AppArucoPenROSNode.h"

//-----------------------------------------------------------------------------
ArucoPen::~ArucoPen()
{
    delete _trackingSystem;
}
//-----------------------------------------------------------------------------
SLbool ArucoPen::onKeyPress(const SLKey key,
                            const SLKey mod)
{
    if (key == '1')
    {
        _state = Tracing;
        return true;
    }

    if (key == '2')
    {
        AppArucoPenROSNode::instance().publishKeyEvent(rosPosition(),
                                                       rosOrientation());
        return true;
    }

    if (key == K_F6)
    {
        SLVec3f position = tipPosition();
        float   distance = liveDistance();

        SL_LOG("ArUco Pen");
        SL_LOG("\tPosition: %s", position.toString(", ", 4).c_str());

        if (_positionPrintedOnce)
        {
            SL_LOG("\tDistance: %.2fcm", distance * 100.0f);
        }
        else
        {
            SL_LOG("\tTake a second measurement to calculate the distance");
            _positionPrintedOnce = true;
        }

        _lastPrintedPosition = position;
        _lastDistance        = distance;

        return true;
    }

    return false;
}
//-----------------------------------------------------------------------------
SLbool ArucoPen::onKeyRelease(const SLKey key,
                              const SLKey mod)
{
    if (key == '1')
    {
        _state = Idle;
        return true;
    }

    return false;
}
//-----------------------------------------------------------------------------
SLVec3f ArucoPen::tipPosition()
{
    float tipOffset = -(0.147f - 0.025f + 0.002f);

    CVMatx44f worldMatrix = _trackingSystem->worldMatrix();

    float offsetX = worldMatrix.val[1] * tipOffset;
    float offsetY = worldMatrix.val[5] * tipOffset;
    float offsetZ = worldMatrix.val[9] * tipOffset;

    SLVec3f position(worldMatrix.val[3] + offsetX,
                     worldMatrix.val[7] + offsetY,
                     worldMatrix.val[11] + offsetZ);
    return position;
}
//-----------------------------------------------------------------------------
SLQuat4f ArucoPen::orientation()
{
    CVMatx44f worldMatrix = _trackingSystem->worldMatrix();
    // clang-format off
    SLMat3f rotMatrix(worldMatrix.val[0], worldMatrix.val[1], worldMatrix.val[2],
                      worldMatrix.val[4], worldMatrix.val[5], worldMatrix.val[6],
                      worldMatrix.val[8], worldMatrix.val[9], worldMatrix.val[10]);
    // clang-format on

    SLQuat4f orientation;
    orientation.fromMat3(rotMatrix);
    return orientation;
}
//-----------------------------------------------------------------------------
SLVec3f ArucoPen::rosPosition()
{
    // ROS coordinate system: (-z, y, -x)

    SLVec3f p = tipPosition();
    return {-p.z, p.y, -p.x};
}
//-----------------------------------------------------------------------------
SLQuat4f ArucoPen::rosOrientation()
{
    // ROS coordinate system: (-z, y, -x)
    // Source: https://stackoverflow.com/questions/18818102/convert-quaternion-representing-rotation-from-one-coordinate-system-to-another

    SLQuat4f o = orientation();
    return {o.z(), -o.y(), o.x(), o.w()};
}
//-----------------------------------------------------------------------------
SLfloat ArucoPen::liveDistance()
{
    if (!_positionPrintedOnce)
    {
        return 0.0f;
    }

    SLVec3f position = tipPosition();
    return position.distance(_lastPrintedPosition);
}
//-----------------------------------------------------------------------------
SLfloat ArucoPen::lastDistance() const
{
    return _lastDistance;
}
//-----------------------------------------------------------------------------
void ArucoPen::trackingSystem(TrackingSystem* trackingSystem)
{
    delete _trackingSystem;
    _trackingSystem = trackingSystem;

    // Switch to the first accepted provider if the current one isn't accepted by the tracking system
    CVCaptureProvider* currentProvider = AppArucoPen::instance().currentCaptureProvider();

    if (_trackingSystem->isAcceptedProvider(currentProvider))
    {
        return;
    }

    for (CVCaptureProvider* provider : AppArucoPen::instance().captureProviders())
    {
        SL_LOG("TESTING");
        if(_trackingSystem->isAcceptedProvider(provider))
        {
            SL_LOG("ACCEPTED!");
            AppArucoPen::instance().currentCaptureProvider(provider);
            return;
        }
    }

    SL_EXIT_MSG("No accepted capture provider was found for this system");
}