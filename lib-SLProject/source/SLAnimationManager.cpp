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
#include <SLAnimationPlay.h>
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
    
    SLMAnimationPlay::iterator it2;
    for (it2 = _nodeAnimationPlays.begin(); it2 != _nodeAnimationPlays.end(); it2++)
        delete it2->second;
    _nodeAnimationPlays.clear();

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
//! Returns the play of a node animation by name if it exists.
SLAnimationPlay* SLAnimationManager::getNodeAnimationPlay(const SLstring& name)
{
    if (_nodeAnimationPlays.find(name) != _nodeAnimationPlays.end())
        return _nodeAnimationPlays[name];

    else if (_nodeAnimations.find(name) != _nodeAnimations.end())
    {
        _nodeAnimationPlays[name] = new SLAnimationPlay(_nodeAnimations[name]);
        return _nodeAnimationPlays[name];
    }

    return NULL;
}

//-----------------------------------------------------------------------------
//! Advances the time of all enabled animation plays.
SLbool SLAnimationManager::update(SLfloat elapsedTimeSec)
{
    SLbool updated = false;

    // advance time for node animations and apply them
    // @todo currently we can't blend between normal node animations because we reset them
    // per play. so the last plays that affects a node will have its animation applied.
    // we need to save the plays differently if we want them.

    SLMAnimationPlay::iterator it;
    for (it = _nodeAnimationPlays.begin(); it != _nodeAnimationPlays.end(); it++)
    {
        SLAnimationPlay* play = it->second;
        if (play->enabled())
        {
            play->parentAnimation()->resetNodes();
            play->advanceTime(elapsedTimeSec);
            play->parentAnimation()->apply(play->localTime(), play->weight());
            updated = true;
        }
    }
    
    // update the skeletons seperately 
    for (SLint i = 0; i < _skeletons.size(); ++i)
    {
        updated |= _skeletons[i]->updateAnimations(elapsedTimeSec);
    }
    return updated;
}

//-----------------------------------------------------------------------------
