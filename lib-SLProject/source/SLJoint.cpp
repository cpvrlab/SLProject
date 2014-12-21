
#include <stdafx.h>
#include <SLJoint.h>
#include <SLSkeleton.h>


//-----------------------------------------------------------------------------
/*! Constructor
*/
SLJoint::SLJoint(SLuint handle, SLSkeleton* creator)
: _handle(handle), _creator(creator), SLNode("Unnamed Joint"), _radius(0)
{ }

//-----------------------------------------------------------------------------
/*! Constructor
*/
SLJoint::SLJoint(const SLstring& name, SLuint handle, SLSkeleton* creator)
: _handle(handle), _creator(creator), SLNode(name), _radius(0)
{ }


//-----------------------------------------------------------------------------
/*! Creation function to create a new child joint for this joint.
*/
SLJoint* SLJoint::createChild(SLuint handle)
{
    SLJoint* joint = _creator->createJoint(handle);
    addChild(joint);
    return joint;
}

//-----------------------------------------------------------------------------
/*! Creation function to create a new child joint for this joint.
*/
SLJoint* SLJoint::createChild(const SLstring& name, SLuint handle)
{
    SLJoint* joint = _creator->createJoint(name, handle);
    addChild(joint);
    return joint;
}

//-----------------------------------------------------------------------------
/*! Getter for the offset matrix of this specific joint.
*/
void SLJoint::offsetMat(const SLMat4f& mat)
{
    _offsetMat = mat;
}


//-----------------------------------------------------------------------------
/*! Updates the current max radius with the input vertex position in joint space.
*/
void SLJoint::calcMaxRadius(const SLVec3f& vec)
{
    //
    SLVec3f boneSpaceVec = _offsetMat * vec;
    _radius = max(_radius, boneSpaceVec.length());
}

//-----------------------------------------------------------------------------
/*! Getter that calculates the final joint transform matrix.
*/
SLMat4f SLJoint::calculateFinalMat()
{
    return updateAndGetWM() * _offsetMat;
}