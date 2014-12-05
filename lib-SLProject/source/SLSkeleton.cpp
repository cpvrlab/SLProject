
#include <stdafx.h>
#include <SLSkeleton.h>
#include <SLScene.h>
#include <SLAnimationState.h>


SLSkeleton::SLSkeleton()
{
    SLScene::current->animManager().addSkeleton(this);
}

SLSkeleton::~SLSkeleton()
{
    delete _root;

    map<SLstring, SLAnimation*>::iterator it1;
    for (it1 = _animations.begin(); it1 != _animations.end(); it1++)
        delete it1->second;
    
    map<SLstring, SLAnimationState*>::iterator it2;
    for (it2 = _animationStates.begin(); it2 != _animationStates.end(); it2++)
        delete it2->second;
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


SLAnimationState* SLSkeleton::getAnimationState(const SLstring& name)
{
    if (_animationStates.find(name) != _animationStates.end())
        return _animationStates[name];

    else if (_animations.find(name) != _animations.end())
    {
        _animationStates[name] = new SLAnimationState(_animations[name]);
        return _animationStates[name];
    }

    return NULL;
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
    //reset();

    map<SLstring, SLAnimationState*>::iterator it;
    for (it = _animationStates.begin(); it != _animationStates.end(); it++)
    {
        SLAnimationState* state = it->second;
        if (state->enabled())
        {reset();
            state->advanceTime(scene->elapsedTimeSec());
            state->parentAnimation()->apply(this, state->localTime(), state->weight());
        }
    }
}