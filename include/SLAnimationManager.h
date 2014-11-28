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
    
    SLAnimationState* createNodeAnimationState(SLAnimation* parent, SLfloat weight = 1.0f);

    void update(); // updates all active animations

private:
    // at the moment we keep the states seperated by their application type
    // this means that we need to create them differently and that only the 
    // manager knows which state affects what type of animation
    SLVSkeleton                 _skeletons;
    vector<SLAnimationState*>   _nodeAnimationStates;
};

#endif