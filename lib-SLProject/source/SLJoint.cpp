
#include <stdafx.h>
#include <SLJoint.h>
#include <SLSkeleton.h>


SLJoint::SLJoint(SLuint handle, SLSkeleton* creator)
: _handle(handle), _creator(creator), SLNode("Unnamed Joint")
{ }

SLJoint::SLJoint(const SLstring& name, SLuint handle, SLSkeleton* creator)
: _handle(handle), _creator(creator), SLNode(name)
{ }


SLJoint* SLJoint::createChild(SLuint handle)
{
    SLJoint* joint = _creator->createJoint(handle);
    addChild(joint);
    return joint;
}

SLJoint* SLJoint::createChild(const SLstring& name, SLuint handle)
{
    SLJoint* joint = _creator->createJoint(name, handle);
    addChild(joint);
    return joint;
}

// set a new offset matrix
void SLJoint::offsetMat(const SLMat4f& mat)
{
    _offsetMat = mat;
}

// 
SLMat4f SLJoint::calculateFinalMat()
{
    return updateAndGetWM() * _offsetMat;
}