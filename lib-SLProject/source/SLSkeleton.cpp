
#include <stdafx.h>
#include <SLSkeleton.h>
#include <SLScene.h>
#include <SLAnimationState.h>


SLSkeleton::SLSkeleton()
{
    SLScene::current->skeletons().push_back(this);
}

SLBone* SLSkeleton::createBone(SLuint handle)
{
    ostringstream oss;
    oss << "Bone " << handle;
    return createBone(oss.str(), handle);
}

SLBone* SLSkeleton::createBone(const SLstring& name, SLuint handle)
{
    SLBone* result = new SLBone(name, handle, this);
    
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

SLBone* SLSkeleton::getBone(const SLstring& name)
{
    if (!_root) return NULL;

    SLBone* result = _root->find<SLBone>(name);
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


void SLSkeleton::updateAnimations()
{
    SLScene* scene = SLScene::current;

    // @todo don't do this if we don't have any enabled animations
    reset();

    map<SLstring, SLAnimationState*>::iterator it;
    for (it = _animationStates.begin(); it != _animationStates.end(); it++)
    {
        SLAnimationState* state = it->second;
        if (state->enabled())
        {
            // state->advanceTime(scene->elapsedTime()); // get elapsed time
            state->advanceTime(0.016f); // temporary test val
            state->parentAnimation()->apply(this, state->localTime(), state->weight());
        }
    }
}