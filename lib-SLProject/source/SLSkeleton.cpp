//#############################################################################
//  File:      SLSkeleton.cpp
//  Author:    Marc Wacker
//  Date:      Autumn 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLSkeleton.h>
#include <SLScene.h>
#include <SLAnimationState.h>

//-----------------------------------------------------------------------------
SLSkeleton::SLSkeleton()
{
    SLScene::current->animManager().addSkeleton(this);
}

//-----------------------------------------------------------------------------
SLSkeleton::~SLSkeleton()
{
    delete _root;

    SLMAnimation::iterator it1;
    for (it1 = _animations.begin(); it1 != _animations.end(); it1++)
        delete it1->second;
    
    SLMAnimationState::iterator it2;
    for (it2 = _animationStates.begin(); it2 != _animationStates.end(); it2++)
        delete it2->second;
}

//-----------------------------------------------------------------------------
SLJoint* SLSkeleton::createJoint(SLuint handle)
{
    ostringstream oss;
    oss << "Joint " << handle;
    return createJoint(oss.str(), handle);
}

//-----------------------------------------------------------------------------
SLJoint* SLSkeleton::createJoint(const SLstring& name, SLuint handle)
{
    SLJoint* result = new SLJoint(name, handle, this);
    
    assert((handle >= _joints.size() ||
           (handle < _joints.size() && _joints[handle] == NULL)) &&
          "Trying to create a joint with an already existing handle.");

    if (_joints.size() <= handle)
        _joints.resize(handle+1);
    
    _joints[handle] = result;
    return result;
}

//-----------------------------------------------------------------------------
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
//-----------------------------------------------------------------------------
SLJoint* SLSkeleton::getJoint(SLuint handle)
{
    assert(handle < _joints.size() && "Index out of bounds");
    return _joints[handle];
}
//-----------------------------------------------------------------------------
SLJoint* SLSkeleton::getJoint(const SLstring& name)
{
    if (!_root) return NULL;

    SLJoint* result = _root->find<SLJoint>(name);
    return result;
}
//-----------------------------------------------------------------------------
void SLSkeleton::getJointWorldMatrices(SLMat4f* jointWM)
{
    // @todo this is asking for a crash...
    for (SLint i = 0; i < _joints.size(); i++)
    {
        jointWM[i] = _joints[i]->updateAndGetWM() * _joints[i]->offsetMat();
    }
}
//-----------------------------------------------------------------------------
void SLSkeleton::root(SLJoint* joint)
{
    if (_root)
        _root = joint;
}
//-----------------------------------------------------------------------------
void SLSkeleton::addAnimation(SLAnimation* anim)
{
    _animations[anim->name()] = anim;
}
//-----------------------------------------------------------------------------
void SLSkeleton::reset()
{
    // mark the skeleton as changed
    changed(true);

    // update all joints
    for (SLint i = 0; i < _joints.size(); i++)
        _joints[i]->resetToInitialState();
}
//-----------------------------------------------------------------------------
void SLSkeleton::updateAnimations()
{
    SLScene* scene = SLScene::current;

    /// @todo IMPORTANT: don't do this if we don't have any enabled animations,
    /// current workaround won't allow for blending!
    //reset();

    // first advance time on all animations
    _changed = false;
    SLMAnimationState::iterator it;
    for (it = _animationStates.begin(); it != _animationStates.end(); it++)
    {
        SLAnimationState* state = it->second;
        if (state->enabled())
        {
            state->advanceTime(scene->elapsedTimeSec());
            // mark skeleton as changed if a state is different
            if (state->changed())
                _changed = true;
        }
    }

    // return if nothing changed
    if (!_changed)
        return;

    // reset the skeleton and apply all enabled animations
    reset();

    for (it = _animationStates.begin(); it != _animationStates.end(); it++)
    {
        SLAnimationState* state = it->second;
        if (state->enabled())
        {
            state->parentAnimation()->apply(this, state->localTime(), state->weight());
            state->changed(false); // remove changed dirty flag from the state again
        }
    }
}
//-----------------------------------------------------------------------------
