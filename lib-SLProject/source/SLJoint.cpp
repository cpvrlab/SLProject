
#include <stdafx.h>
#include <SLJoint.h>
#include <SLSkeleton.h>


//-----------------------------------------------------------------------------
/*! @todo document
*/
SLJoint::SLJoint(SLuint handle, SLSkeleton* creator)
: _handle(handle), _creator(creator), SLNode("Unnamed Joint"), _radius(0)
{ }

//-----------------------------------------------------------------------------
/*! @todo document
*/
SLJoint::SLJoint(const SLstring& name, SLuint handle, SLSkeleton* creator)
: _handle(handle), _creator(creator), SLNode(name), _radius(0)
{ }


//-----------------------------------------------------------------------------
/*! @todo document
*/
SLJoint* SLJoint::createChild(SLuint handle)
{
    SLJoint* joint = _creator->createJoint(handle);
    addChild(joint);
    return joint;
}

//-----------------------------------------------------------------------------
/*! @todo document
*/
SLJoint* SLJoint::createChild(const SLstring& name, SLuint handle)
{
    SLJoint* joint = _creator->createJoint(name, handle);
    addChild(joint);
    return joint;
}

//-----------------------------------------------------------------------------
/*! @todo document
*/
void SLJoint::offsetMat(const SLMat4f& mat)
{
    _offsetMat = mat;
}


//-----------------------------------------------------------------------------
/*! @todo document
*/
void SLJoint::calcMaxRadius(const SLVec3f& vec)
{
    //
    SLVec3f boneSpaceVec = _offsetMat * vec;
    _radius = max(_radius, boneSpaceVec.length());
}

//-----------------------------------------------------------------------------
/*! @todo document
*/
SLMat4f SLJoint::calculateFinalMat()
{
    return updateAndGetWM() * _offsetMat;
}