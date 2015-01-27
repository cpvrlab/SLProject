#include <SLLeapTool.h>


SLQuat4f SLLeapTool::toolRotation() const
{
    Leap::Vector dir = _tool.direction();
    SLVec3f slDir(dir.x, dir.y, dir.z);
    return SLQuat4f(-SLVec3f::AXISZ, slDir);
}
SLVec3f SLLeapTool::toolTipPosition() const
{
    Leap::Vector pos = _tool.tipPosition();
    return SLVec3f(pos.x, pos.y, pos.z) *0.01f; // @todo add the unit scaling as a constant define
}
SLVec3f SLLeapTool::toolTipVelocity() const
{
    Leap::Vector vel = _tool.tipVelocity();
    return SLVec3f(vel.x, vel.y, vel.z);
}
