
#include <stdafx.h>
#include <SLBone.h>
#include <SLSkeleton.h>


SLBone::SLBone(SLuint handle, SLSkeleton* creator)
: _handle(handle), _creator(creator), SLNode("Unnamed Bone")
{ }

SLBone::SLBone(const SLstring& name, SLuint handle, SLSkeleton* creator)
: _handle(handle), _creator(creator), SLNode(name)
{ }


SLBone* SLBone::createChild(SLuint handle)
{
    SLBone* bone = _creator->createBone(handle);
    addChild(bone);
    return bone;
}

SLBone* SLBone::createChild(const SLstring& name, SLuint handle)
{
    SLBone* bone = _creator->createBone(name, handle);
    addChild(bone);
    return bone;
}

// set a new offset matrix
void SLBone::offsetMat(const SLMat4f& mat)
{
    _offsetMat = mat;
}

// 
SLMat4f SLBone::calculateFinalMat()
{
    return updateAndGetWM() * _offsetMat;
}