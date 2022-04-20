//#############################################################################
//  File:      ArucoPen.cpp
//  Date:      September 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <TrackedPen.h>
#include <app/AppPenTracking.h>
#include <app/AppPenTrackingROSNode.h>
#include <CoordSystemConversions.h>
#include <AppDemo.h>
#include <SLProjectScene.h>
#include <stdexcept>

//-----------------------------------------------------------------------------
TrackedPen::~TrackedPen()
{
    delete _trackingSystem;
}
//-----------------------------------------------------------------------------
SLbool TrackedPen::onKeyPress(const SLKey key,
                              const SLKey mod)
{
    if (key == '1')
    {
        _state = Tracing;
        return true;
    }

    if (key == '2')
    {
        AppPenTrackingROSNode::instance().publishKeyEvent(rosPosition(),
                                                         rosOrientation());
        return true;
    }

    if (key == '3')
    {
        AppPenTrackingROSNode::instance().publishSelection(rosPosition());
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
SLbool TrackedPen::onKeyRelease(const SLKey key,
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
SLMat4f TrackedPen::headTransform()
{
    CVMatx44f m = _trackingSystem->worldMatrix();
    // clang-format off
    return {m.val[ 0], m.val[ 1], m.val[ 2], m.val[ 3],
            m.val[ 4], m.val[ 5], m.val[ 6], m.val[ 7],
            m.val[ 8], m.val[ 9], m.val[10], m.val[11],
            m.val[12], m.val[13], m.val[14], m.val[15]};
    // clang-format on
}
//-----------------------------------------------------------------------------
SLVec3f TrackedPen::tipPosition()
{
    CVMatx44f worldMatrix = _trackingSystem->worldMatrix();

    float tipOffset = -length();
    float offsetX   = worldMatrix.val[1] * tipOffset;
    float offsetY   = worldMatrix.val[5] * tipOffset;
    float offsetZ   = worldMatrix.val[9] * tipOffset;

    SLVec3f position(worldMatrix.val[3] + offsetX,
                     worldMatrix.val[7] + offsetY,
                     worldMatrix.val[11] + offsetZ);
    return position;
}
//-----------------------------------------------------------------------------
SLQuat4f TrackedPen::orientation()
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
SLVec3f TrackedPen::rosPosition()
{
    // ROS coordinate system: (-z, y, -x)

    SLVec3f p = tipPosition();
    return {p.z, p.y, p.x};
}
//-----------------------------------------------------------------------------
SLQuat4f TrackedPen::rosOrientation()
{
    // ROS coordinate system: (-z, y, -x)
    // Source: https://stackoverflow.com/questions/18818102/convert-quaternion-representing-rotation-from-one-coordinate-system-to-another

    SLQuat4f o = orientation();
    return {o.z(), o.y(), o.x(), -o.w()};
}
//-----------------------------------------------------------------------------
SLfloat TrackedPen::liveDistance()
{
    if (!_positionPrintedOnce)
    {
        return 0.0f;
    }

    SLVec3f position = tipPosition();
    return position.distance(_lastPrintedPosition);
}
//-----------------------------------------------------------------------------
SLfloat TrackedPen::lastDistance() const
{
    return _lastDistance;
}
//-----------------------------------------------------------------------------
void TrackedPen::trackingSystem(TrackingSystem* trackingSystem)
{
    // Switch to the first accepted provider if the current one isn't accepted by the tracking system
    CVCaptureProvider* currentProvider = AppPenTracking::instance().currentCaptureProvider();

    if (trackingSystem->isAcceptedProvider(currentProvider))
    {
        updateTrackingSystem(trackingSystem);
        return;
    }

    for (CVCaptureProvider* provider : AppPenTracking::instance().captureProviders())
    {
        if (trackingSystem->isAcceptedProvider(provider))
        {
            updateTrackingSystem(trackingSystem);
            AppPenTracking::instance().currentCaptureProvider(provider);
            return;
        }
    }

    throw std::runtime_error("No capture provider accepted by this tracking system was found!");
}
//-----------------------------------------------------------------------------
void TrackedPen::updateTrackingSystem(TrackingSystem* trackingSystem)
{
    if(_trackingSystem)
    {
        AppDemo::scene->root3D()->removeChild(_trackingSystem->penNode());
        delete _trackingSystem;
    }

    _trackingSystem = trackingSystem;

    trackingSystem->createPenNode();
    AppDemo::scene->root3D()->addChild(trackingSystem->penNode());
}
//-----------------------------------------------------------------------------