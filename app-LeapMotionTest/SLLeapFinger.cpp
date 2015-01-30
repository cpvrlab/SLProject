//#############################################################################
//  File:      SLLeapFinger.cpp
//  Author:    Marc Wacker
//  Date:      January 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include  <stdafx.h>
#include <SLLeapFinger.h>

SLLeapFinger::SLLeapFinger(Leap::Finger::Type type)
{
    _fingerType = type;
}


void SLLeapFinger::leapHand(const Leap::Hand& hand)
{
    _hand = hand;
    _finger = _hand.fingers()[_fingerType];
}


SLVec3f SLLeapFinger::tipPosition() const
{
    Leap::Vector pos = _finger.tipPosition();
    return SLVec3f(pos.x, pos.y, pos.z) * 0.01f; // @todo add the unit scaling as a constant define
}

SLVec3f SLLeapFinger::jointPosition(SLint joint) const
{
    if (joint >= numBones())
        return tipPosition();

    
    Leap::Bone::Type type = static_cast<Leap::Bone::Type>(joint);
    Leap::Vector pos = _finger.bone(type).prevJoint();
    
    return SLVec3f(pos.x, pos.y, pos.z) * 0.01f; // @todo add the unit scaling as a constant define
}

SLVec3f SLLeapFinger::boneCenter(SLint boneType) const
{
    Leap::Bone::Type type = static_cast<Leap::Bone::Type>(boneType);
    Leap::Vector pos = _finger.bone(type).center();
    
    return SLVec3f(pos.x, pos.y, pos.z) * 0.01f; // @todo add the unit scaling as a constant define
}

SLVec3f SLLeapFinger::boneDirection(SLint boneType) const
{
    Leap::Bone::Type type = static_cast<Leap::Bone::Type>(boneType);
    Leap::Vector dir = _finger.bone(type).direction();
    
    return SLVec3f(dir.x, dir.y, dir.z);
}

SLQuat4f SLLeapFinger::boneRotation(SLint boneType) const
{    
    Leap::Bone::Type type = static_cast<Leap::Bone::Type>(boneType);
    Leap::Vector& bX = _finger.bone(type).basis().xBasis;
    Leap::Vector& bY = _finger.bone(type).basis().yBasis;
    Leap::Vector& bZ = _finger.bone(type).basis().zBasis;
    
    SLMat3f basis(bX.x, bY.x, bZ.x,
                  bX.y, bY.y, bZ.y,
                  bX.z, bY.z, bZ.z);
    
    // @note we don't seem to have to convert the left hand from a LH system to a RH system

    SLVec3f slUp = basis * SLVec3f::AXISY;
    SLVec3f slForward = basis * -SLVec3f::AXISZ;
    
    return SLQuat4f::fromLookRotation(slForward, slUp);
}