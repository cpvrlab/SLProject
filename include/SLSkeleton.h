//#############################################################################
//  File:      SLAnimation.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################


#ifndef SLSKELETON_H
#define SLSKELETON_H


#include <stdafx.h>
#include <SLBone.h>
#include <SLAnimation.h>

class SLAnimationState;

class SLSkeleton
{
public:
    SLSkeleton();
    ~SLSkeleton();

    void update();
    
    // creates a new bone that belongs to this skeleton
    // handle must be unique for this skeleton and also contiguous
    SLBone* createBone(SLuint handle);
    SLBone* createBone(const SLstring& name, SLuint handle);
    
    SLAnimationState* getAnimationState(const SLstring& name);

    void        loadAnimation(const SLstring& file); // import a seperate animation that works with this skeleton

    SLBone*     getBone(SLuint handle);
    SLBone*     getBone(const SLstring& name);
    SLint       numBones() const { return _boneList.size(); }
    void        getBoneWorldMatrices(SLMat4f* boneWM);
    void        root(SLBone* bone);
    SLBone*     root() { return _root; }
    void        addAnimation(SLAnimation* anim);
    SLint       numAnimations() const { return _animations.size(); }
    void        reset();

    void        updateAnimations();
    
protected:
    SLBone*     _root;
    vector<SLBone*> _boneList; //!< bone map for fast acces of bones
    map<SLstring, SLAnimation*> _animations;
    map<SLstring, SLAnimationState*> _animationStates;
};

typedef std::vector<SLSkeleton*> SLVSkeleton;

#endif