//#############################################################################
//  File:      SLJoint.cpp
//  Author:    Marc Wacker
//  Date:      Autumn 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLJoint.h>
#include <SLSkeleton.h>

//-----------------------------------------------------------------------------
SLJoint::SLJoint(SLuint handle, SLSkeleton* creator)
: _handle(handle), _creator(creator), SLNode("Unnamed Joint")
{
}
//-----------------------------------------------------------------------------
SLJoint::SLJoint(const SLstring& name, SLuint handle, SLSkeleton* creator)
: _handle(handle), _creator(creator), SLNode(name)
{
}
//-----------------------------------------------------------------------------
SLJoint* SLJoint::createChild(SLuint handle)
{
    SLJoint* joint = _creator->createJoint(handle);
    addChild(joint);
    return joint;
}
//-----------------------------------------------------------------------------
SLJoint* SLJoint::createChild(const SLstring& name, SLuint handle)
{
    SLJoint* joint = _creator->createJoint(name, handle);
    addChild(joint);
    return joint;
}
//-----------------------------------------------------------------------------
// set a new offset matrix
void SLJoint::offsetMat(const SLMat4f& mat)
{
    _offsetMat = mat;
}
//-----------------------------------------------------------------------------
// 
SLMat4f SLJoint::calculateFinalMat()
{
    return updateAndGetWM() * _offsetMat;
}
//-----------------------------------------------------------------------------
