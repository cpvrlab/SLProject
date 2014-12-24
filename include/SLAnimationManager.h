//#############################################################################
//  File:      SLAnimationManager.h
//  Author:    Marc Wacker
//  Date:      Autumn 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################


#ifndef SLANIMATIONMANAGER_H
#define SLANIMATIONMANAGER_H

#include <stdafx.h>
#include <SLAnimation.h>
#include <SLAnimationManager.h>
#include <SLAnimationPlay.h>
#include <SLSkeleton.h>

//-----------------------------------------------------------------------------
//! SLAnimationManager is the central class for all animation handling.
/*! 
    Keeps a list of all skeleton instances.
    Also keeps a map of simple node animations that affect normal 
    SLNodes in the scene graph.
    The manager is responsible for advancing the time of the enabled
    animations and to manage their life time.
*/
class SLAnimationManager
{
public:
    ~SLAnimationManager();
    
    void            addSkeleton(SLSkeleton* skel) { _skeletons.push_back(skel); }
    void            addNodeAnimation(SLAnimation* anim);
    SLbool          hasNodeAnimations() { return (_nodeAnimations.size() > 0); }
    SLAnimationPlay* getNodeAnimationPlay(const SLstring& name);

    SLAnimation*    createNodeAnimation(SLfloat duration);
    SLAnimation*    createNodeAnimation(const SLstring& name, SLfloat duration);

    // @todo find a better way to give access to the animation names to external stuff (like the gui)
    SLMAnimation    animations() { return _nodeAnimations; }
    SLVSkeleton&    skeletons() { return _skeletons; }

    SLbool          update(SLfloat elapsedTimeSec);
    void            clear();

private:
    SLVSkeleton         _skeletons;             //!< all skeleton instances
    SLMAnimation        _nodeAnimations;        //!< node animations
    SLMAnimationPlay    _nodeAnimationPlays;   //!< node animation plays
};
//-----------------------------------------------------------------------------
#endif
