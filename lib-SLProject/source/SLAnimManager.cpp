//#############################################################################
//  File:      SLAnimManager.cpp
//  Author:    Marc Wacker
//  Date:      Autumn 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif
#include <SLScene.h>
#include <SLAnimation.h>
#include <SLAnimPlayback.h>
#include <SLAnimManager.h>
#include <SLSkeleton.h>

//-----------------------------------------------------------------------------
//! destructor
SLAnimManager::~SLAnimManager()
{
    clear();
}

//-----------------------------------------------------------------------------
//! Clears and deletes all node animations and skeletons
void SLAnimManager::clear()
{
    for (auto it : _nodeAnimations) delete it.second;
    _nodeAnimations.clear();
    
    for (auto it : _nodeAnimPlaybacks) delete it.second;
    _nodeAnimPlaybacks.clear();

    for (auto skeleton : _skeletons) delete skeleton;
    _skeletons.clear();
}

//-----------------------------------------------------------------------------
/*! Creates a new node animation
    @param  duration    length of the animation
*/
SLAnimation* SLAnimManager::createNodeAnimation(SLfloat duration)
{
    SLuint index = (SLuint)_nodeAnimations.size();
    ostringstream oss;

    do
    {   oss.clear();
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
SLAnimation* SLAnimManager::createNodeAnimation(const SLstring& name, SLfloat duration)
{
    assert(_nodeAnimations.find(name) == _nodeAnimations.end() &&
           "node animation with same name already exists!");
    SLAnimation* anim = new SLAnimation(name, duration);
    _nodeAnimations[name] = anim;
    return anim;
}

//-----------------------------------------------------------------------------
//! Returns the playback of a node animation by name if it exists.
SLAnimPlayback* SLAnimManager::getNodeAnimPlayack(const SLstring& name)
{
    if (_nodeAnimPlaybacks.find(name) != _nodeAnimPlaybacks.end())
        return _nodeAnimPlaybacks[name];

    else if (_nodeAnimations.find(name) != _nodeAnimations.end())
    {
        _nodeAnimPlaybacks[name] = new SLAnimPlayback(_nodeAnimations[name]);
        return _nodeAnimPlaybacks[name];
    }

    return nullptr;
}

//-----------------------------------------------------------------------------
//! Advances the time of all enabled animation plays.
SLbool SLAnimManager::update(SLfloat elapsedTimeSec)
{
    SLbool updated = false;

    // advance time for node animations and apply them
    // @todo currently we can't blend between normal node animations because we reset them
    // per animplayback. so the last playback that affects a node will have its animation applied.
    // we need to save the playback differently if we want to blend them.

    for (auto it : _nodeAnimPlaybacks)
    {
        SLAnimPlayback* playback = it.second;
        if (playback->enabled())
        {
            playback->parentAnimation()->resetNodes();
            playback->advanceTime(elapsedTimeSec);
            playback->parentAnimation()->apply(playback->localTime(), playback->weight());
            updated = true;
        }
    }
    
    // update the skeletons seperately 
    for (auto skeleton : _skeletons)
        updated |= skeleton->updateAnimations(elapsedTimeSec);
    
    return updated;
}

//-----------------------------------------------------------------------------
