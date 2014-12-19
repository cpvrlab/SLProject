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
#include <SLJoint.h>
#include <SLAnimation.h>
#include <SLAnimationState.h>

class SLAnimationState;

//-----------------------------------------------------------------------------
//! SLSkeleton keeps track of a skeletons joint and animations
/*! 
    An SLSkeleton keeps track of its SLJoints in a tree structure.
    It is very similar to how a scene graph works. 
    SLAnimations for this skeleton are also kept in this class. The SLAnimations
    have tracks corresponding to the individual SLJoints in the skeleton.

    @note   The current implementation doesn't support multiple instances of the same
            skeleton animation. It is however not that far away from supporting it.
            Currently the SLSkeleton class keeps both a SLAnimation map
            and an SLAnimationState map. We can split this into two classes
            by creating an SLSkeletonInstance class we that keeps the 
            SLAnimationState map that references its parent SLSkeleton
            we would be able to create multiple SLSkeletonInstance instances
            that use the same SLAnimations but with different states.

            This leaves the problem of SLMesh that is not able to be instantiated
            without copying the data into a completely seperate SLMesh. But the
            solution for SLMesh would take the same approach by creating a 
            mesh instance class that is able to use SLSkeletonInstance.
*/
class SLSkeleton
{
public:
    SLSkeleton();
    ~SLSkeleton();

    void update();
    
    // creates a new joint that belongs to this skeleton
    // handle must be unique for this skeleton and also contiguous
    SLJoint*            createJoint(SLuint handle);
    SLJoint*            createJoint(const SLstring& name, SLuint handle);
    
    SLAnimationState*   getAnimationState(const SLstring& name);

    void                loadAnimation(const SLstring& file); // import a seperate animation that works with this skeleton

    SLJoint*            getJoint(SLuint handle);
    SLJoint*            getJoint(const SLstring& name);
    SLint               numJoints() const { return (SLint)_jointList.size(); }
    void                getJointWorldMatrices(SLMat4f* jointWM);
    void                root(SLJoint* joint);
    SLJoint*            root() { return _root; }
    SLAnimation*        createAnimation(const SLstring& name, SLfloat duration);
    SLint               numAnimations() const { return (SLint)_animations.size(); }
    void                reset();

    SLbool              changed() const { return _changed; }
    void                changed(SLbool changed) { _changed = changed; }
    
    const SLVec3f&      minOS();
    const SLVec3f&      maxOS();

    // @todo this accessor was needed by the qt gui, is there a better way?
    map<SLstring, SLAnimation*> animations() { return _animations; }

    void                updateAnimations();
    
protected:
    SLJoint*            _root;              //!< root joint
    vector<SLJoint*>    _jointList;         //!< joint list for fast access and index to joint mapping
    map<SLstring, SLAnimation*> _animations;        //!< animations for this skeleton
    map<SLstring, SLAnimationState*> _animationStates;  //!< animation states for this skeleton
    SLbool              _changed;           //!< did this skeleton change this frame (attribute for skeleton instance)
    SLVec3f             _minOS;             //!< min point in os space for this skeleton (attribute for skeleton instance)
    SLVec3f             _maxOS;             //!< max point in os space for this skeleton (attribute for skeleton instance)
    SLbool              _minMaxOutOfDate;

    void updateMinMax();

};

typedef std::vector<SLSkeleton*> SLVSkeleton;

#endif