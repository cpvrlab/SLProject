
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
    /*
    Leap::Bone::Type type = static_cast<Leap::Bone::Type>(boneType);
    Leap::Vector& bX = _finger.bone(type).basis().xBasis;
    Leap::Vector& bY = _finger.bone(type).basis().yBasis;
    Leap::Vector& bZ = _finger.bone(type).basis().zBasis;
    SLMat3f basis(bX.x, bY.x, bZ.x,
                  bX.y, bY.y, bZ.y,
                  bX.z, bY.z, bZ.z);

    if (_hand.isLeft()) 
    {
        SLMat3f flipZ;
        flipZ.scale(1, 1, -1);
        basis = basis * flipZ;
    }
    
    return SLQuat4f(basis);*/

    Leap::Bone::Type type = static_cast<Leap::Bone::Type>(boneType);
    Leap::Vector& bX = _finger.bone(type).basis().xBasis;
    Leap::Vector& bY = _finger.bone(type).basis().yBasis;
    Leap::Vector& bZ = _finger.bone(type).basis().zBasis;
    
    Leap::Vector up = _finger.bone(type).basis().transformDirection(Leap::Vector(0, 1, 0));
    Leap::Vector forward = _finger.bone(type).basis().transformDirection(Leap::Vector(0, 0, -1));
    /*
    if (_hand.isLeft()) 
    {
        up.z *= -1;
        forward.z *= -1;
    }*/
    
    return TestUtil::QuaternionLookRotation(SLVec3f(forward.x, forward.y, forward.z), SLVec3f(up.x, up.y, up.z));
}