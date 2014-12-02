//#############################################################################
//  File:      SLAnimation.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################


#ifndef SLANIMATIONMANAGER_H
#define SLANIMATIONMANAGER_H


#include <stdafx.h>

class SLAnimation;
class SLAnimationState;

class SLAnimationManager
{
public:
    SLAnimationManager();
    ~SLAnimationManager();
    
    void addSkeleton(SLSkeleton* skel) { _skeletons.push_back(skel); }
    void addNodeAnimation(SLAnimation* anim);
    SLbool hasNodeAnimations() { return (_nodeAnimations.size() > 0); }
    SLVSkeleton& skeletons() { return _skeletons; }
    SLAnimationState* createNodeAnimationState(SLAnimation* parent, SLfloat weight = 1.0f);
    SLAnimationState* getAnimationState(const SLstring& name); // get the state for a specific animation

    // @todo find a better way to give access to the animation names to external stuff (like the gui)
    map<SLstring, SLAnimation*> animations() { return _nodeAnimations; }

    void update(); // updates all active animations

    void clear();

private:
    // at the moment we keep the states seperated by their application type
    // this means that we need to create them differently and that only the 
    // manager knows which state affects what type of animation
    SLVSkeleton                 _skeletons;             //!< all skeleton instances
    map<SLstring, SLAnimation*> _nodeAnimations;        //!< node animations
    vector<SLAnimationState*>   _nodeAnimationStates;   //!< node animation states
};

#endif