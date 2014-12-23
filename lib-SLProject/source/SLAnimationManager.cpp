//#############################################################################
//  File:      SLAnimationManager.cpp
//  Author:    Marc Wacker
//  Date:      Autumn 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLScene.h>
#include <SLAnimation.h>
#include <SLAnimationState.h>
#include <SLAnimationManager.h>
#include <SLSkeleton.h>

//-----------------------------------------------------------------------------
//! destructor
SLAnimationManager::~SLAnimationManager()
{
    clear();
}

//-----------------------------------------------------------------------------
//! Clears and deletes all node animations and skeletons
void SLAnimationManager::clear()
{
    SLMAnimation::iterator it;
    for (it = _nodeAnimations.begin(); it != _nodeAnimations.end(); it++)
        delete it->second;
    _nodeAnimations.clear();
    
    SLMAnimationState::iterator it2;
    for (it2 = _nodeAnimationStates.begin(); it2 != _nodeAnimationStates.end(); it2++)
        delete it2->second;
    _nodeAnimationStates.clear();

    for (SLint i = 0; i < _skeletons.size(); ++i)
        delete _skeletons[i];
    _skeletons.clear();
}

//-----------------------------------------------------------------------------
/*! Creates a new node animation
    @param  duration    length of the animation
*/
SLAnimation* SLAnimationManager::createNodeAnimation(SLfloat duration)
{
    SLuint index = (SLuint)_nodeAnimations.size();
    ostringstream oss;

    do
    {   
        oss.clear();
        oss << "NodeAnimation_" << index;
        index++;
    }
    while (_nodeAnimations.find(oss.str()) != _nodeAnimations.end());
     
    return createNodeAnimation(oss.str(), duration);
}

//-----------------------------------------------------------------------------
/*! Creates a new node animation
    @param  name        the animation name
    @param  duration    length of the animation
*/
SLAnimation* SLAnimationManager::createNodeAnimation(const SLstring& name, SLfloat duration)
{
    assert(_nodeAnimations.find(name) == _nodeAnimations.end() &&
           "node animation with same name already exists!");
    SLAnimation* anim = new SLAnimation(name, duration);
    _nodeAnimations[name] = anim;
    return anim;
}

//-----------------------------------------------------------------------------
//! Returns the state of a node animation by name if it exists.
SLAnimationState* SLAnimationManager::getNodeAnimationState(const SLstring& name)
{
    if (_nodeAnimationStates.find(name) != _nodeAnimationStates.end())
        return _nodeAnimationStates[name];

    else if (_nodeAnimations.find(name) != _nodeAnimations.end())
    {
        _nodeAnimationStates[name] = new SLAnimationState(_nodeAnimations[name]);
        return _nodeAnimationStates[name];
    }

    return NULL;
}

//-----------------------------------------------------------------------------
//! Advances the time of all enabled animation states.
void SLAnimationManager::update()
{
    SLScene* s = SLScene::current;

    // return if animations are off
    if (s->stopAnimations())
        return;

    // advance time for node animations and apply them
    // @todo currently we can't blend between normal node animations because we reset them
    // per state. so the last state that affects a node will have its animation applied.
    // we need to save the states differently if we want them.

    SLMAnimationState::iterator it;
    for (it = _nodeAnimationStates.begin(); it != _nodeAnimationStates.end(); it++)
    {
        SLAnimationState* state = it->second;
        if (state->enabled())
        {
            state->parentAnimation()->resetNodes(); 
            state->advanceTime(s->elapsedTimeSec());
            state->parentAnimation()->apply(state->localTime(), state->weight());
        }
    }
    
    // update the skeletons seperately 
    for (SLint i = 0; i < _skeletons.size(); ++i)
    {
        _skeletons[i]->updateAnimations();
    }
}

//-----------------------------------------------------------------------------
