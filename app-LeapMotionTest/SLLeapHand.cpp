//#############################################################################
//  File:      SLLeapHand.cpp
//  Author:    Marc Wacker
//  Date:      January 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include  <stdafx.h>
#include <SLLeapHand.h>

SLLeapHand::SLLeapHand()
{
    for (SLint i = 0; i < 5; ++i)
        _fingers.push_back(SLLeapFinger((Leap::Finger::Type)i));
}

SLVec3f SLLeapHand::palmPosition() const
{
    Leap::Vector pos = _hand.palmPosition();
    return SLVec3f(pos.x, pos.y, pos.z) *0.01f; // @todo add the unit scaling as a constant define
}


SLQuat4f SLLeapHand::palmRotation() const
{
    Leap::Vector& bX = _hand.basis().xBasis;
    Leap::Vector& bY = _hand.basis().yBasis;
    Leap::Vector& bZ = _hand.basis().zBasis;
    // @note    We enter the Leap::Matrix's row vectors as
    //          column vectors in the SLMat3 since
    //          Leap::Matrix is row major and this saves
    //          us an additional call to SLMat3.transpose
    SLMat3f basis(bX.x, bY.x, bZ.x,
                  bX.y, bY.y, bZ.y,
                  bX.z, bY.z, bZ.z);
    
    // @note we don't seem to have to convert the left hand from a LH system to a RH system

    SLVec3f slUp = basis * SLVec3f::AXISY;
    SLVec3f slForward = basis * -SLVec3f::AXISZ;
    
    return SLQuat4f::fromLookRotation(slForward, slUp);
}


SLVec3f SLLeapHand::wristPosition() const
{
    Leap::Vector pos = _hand.wristPosition();
    return SLVec3f(pos.x, pos.y, pos.z) *0.01f; // @todo add the unit scaling as a constant define
}
SLVec3f SLLeapHand::elbowPosition() const
{
    Leap::Vector pos = _hand.arm().elbowPosition();
    return SLVec3f(pos.x, pos.y, pos.z) *0.01f; // @todo add the unit scaling as a constant define
}
    
SLVec3f SLLeapHand::armCenter() const
{
    Leap::Vector pos = _hand.arm().center();
    return SLVec3f(pos.x, pos.y, pos.z) *0.01f; // @todo add the unit scaling as a constant define
}
SLVec3f SLLeapHand::armDirection() const
{
    return SLVec3f();
}
SLQuat4f SLLeapHand::armRotation() const
{
    Leap::Vector& bX = _hand.arm().basis().xBasis;
    Leap::Vector& bY = _hand.arm().basis().yBasis;
    Leap::Vector& bZ = _hand.arm().basis().zBasis;
    
    SLMat3f basis(bX.x, bY.x, bZ.x,
                  bX.y, bY.y, bZ.y,
                  bX.z, bY.z, bZ.z);
    
    // @note we don't seem to have to convert the left hand from a LH system to a RH system

    SLVec3f slUp = basis * SLVec3f::AXISY;
    SLVec3f slForward = basis * -SLVec3f::AXISZ;
    
    return SLQuat4f::fromLookRotation(slForward, slUp);
}

void SLLeapHand::leapHand(const Leap::Hand& hand)
{
    _hand = hand;
    for (SLint i = 0; i < _fingers.size(); ++i)
        _fingers[i].leapHand(hand);
}

float SLLeapHand::pinchStrength() const
{
    return _hand.pinchStrength();
}

float SLLeapHand::grabStrength() const
{
    return _hand.grabStrength();
}