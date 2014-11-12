
#include <stdafx.h>
#include <SLSkeleton.h>
#include <SLScene.h>


SLSkeleton::SLSkeleton()
{
    SLScene::current->skeletons().push_back(this);
}

SLBone* SLSkeleton::createBone(SLuint handle)
{
    SLBone* result = new SLBone(handle, this);
    
    assert((handle >= _boneList.size() || (handle < _boneList.size() && _boneList[handle] == NULL)) && "Trying to create a bone with an already existing handle.");

    if (_boneList.size() <= handle)
        _boneList.resize(handle+1);
    
    _boneList[handle] = result;
    return result;
}

SLBone* SLSkeleton::getBone(SLuint handle)
{
    assert(handle < _boneList.size() && "Index out of bounds");
    return _boneList[handle];
}

void SLSkeleton::getBoneWorldMatrices(SLMat4f* boneWM)
{
    // @todo this is asking for a crash...
    for (SLint i = 0; i < _boneList.size(); i++)
    {
        boneWM[i] = _boneList[i]->updateAndGetWM() * _boneList[i]->offsetMat();
    }
}

void SLSkeleton::root(SLBone* bone)
{
    if (_root)

    _root = bone;
}

void SLSkeleton::addAnimation(SLAnimation* anim)
{
    _animations[anim->name()] = anim;
}

void SLSkeleton::reset()
{
    for (SLint i = 0; i < _boneList.size(); i++)
        _boneList[i]->resetToInitialState();
}