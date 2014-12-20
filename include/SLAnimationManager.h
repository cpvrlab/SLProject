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
#include <SLAnimationState.h>

class SLSkeleton;

//-----------------------------------------------------------------------------
class SLAnimationManager
{
public:
    SLAnimationManager();
    ~SLAnimationManager();
    
    void            addSkeleton(SLSkeleton* skel) { _skeletons.push_back(skel); }
    void            addNodeAnimation(SLAnimation* anim);
    SLbool          hasNodeAnimations() { return (_nodeAnimations.size() > 0); }
    vector<SLSkeleton*>& skeletons() { return _skeletons; }
    SLAnimationState* getNodeAnimationState(const SLstring& name); // get the state for a specific animation

    // @todo find a better way to give access to the animation names to external stuff (like the gui)
    SLMAnimation    animations() { return _nodeAnimations; }

    void update(); // updates all active animations

    void clear();

private:
    // at the moment we keep the states seperated by their application type
    // this means that we need to create them differently and that only the 
    // manager knows which state affects what type of animation
    vector<SLSkeleton*> _skeletons;             //!< all skeleton instances
    SLMAnimation        _nodeAnimations;        //!< node animations
    SLMAnimationState   _nodeAnimationStates;   //!< node animation states
};
//-----------------------------------------------------------------------------
#endif
