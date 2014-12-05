
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

SLJoint* SLSkeleton::createJoint(SLuint handle)
{
    ostringstream oss;
    oss << "Joint " << handle;
    return createJoint(oss.str(), handle);
}

SLJoint* SLSkeleton::createJoint(const SLstring& name, SLuint handle)
{
    SLJoint* result = new SLJoint(name, handle, this);
    
    assert((handle >= _jointList.size() || (handle < _jointList.size() && _jointList[handle] == NULL)) && "Trying to create a joint with an already existing handle.");

    if (_jointList.size() <= handle)
        _jointList.resize(handle+1);
    
    _jointList[handle] = result;
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

SLJoint* SLSkeleton::getJoint(SLuint handle)
{
    assert(handle < _jointList.size() && "Index out of bounds");
    return _jointList[handle];
}

SLJoint* SLSkeleton::getJoint(const SLstring& name)
{
    if (!_root) return NULL;

    SLJoint* result = _root->find<SLJoint>(name);
    return result;
}

void SLSkeleton::getJointWorldMatrices(SLMat4f* jointWM)
{
    // @todo this is asking for a crash...
    for (SLint i = 0; i < _jointList.size(); i++)
    {
        jointWM[i] = _jointList[i]->updateAndGetWM() * _jointList[i]->offsetMat();
    }
}

void SLSkeleton::root(SLJoint* joint)
{
    if (_root)

    _root = joint;
}

void SLSkeleton::addAnimation(SLAnimation* anim)
{
    _animations[anim->name()] = anim;
}

void SLSkeleton::reset()
{
    for (SLint i = 0; i < _jointList.size(); i++)
        _jointList[i]->resetToInitialState();
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