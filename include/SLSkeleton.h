//#############################################################################
//  File:      SLSkeleton.h
//  Author:    Marc Wacker
//  Date:      Autumn 2014
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
#include <SLAnimationManager.h>

//-----------------------------------------------------------------------------
/** @todo add documentation to skeleton class (explain how to achieve multiple
 * instances of the same skeletal animations with one skeleton)
*/
class SLSkeleton
{
public:
                SLSkeleton();
               ~SLSkeleton();

    void        update();
    
    // creates a new joint that belongs to this skeleton
    // handle must be unique for this skeleton and also contiguous
    SLJoint*    createJoint(SLuint handle);
    SLJoint*    createJoint(const SLstring& name, SLuint handle);
    
    SLAnimationState* getAnimationState(const SLstring& name);

    void        loadAnimation   (const SLstring& file); // import a seperate animation that works with this skeleton

    SLJoint*    getJoint        (SLuint handle);
    SLJoint*    getJoint        (const SLstring& name);
    SLint       numJoints       () const { return (SLint)_joints.size(); }
    void        getJointWorldMatrices(SLMat4f* jointWM);
    void        root            (SLJoint* joint);
    SLJoint*    root            () { return _root; }
    void        addAnimation    (SLAnimation* anim);
    SLint       numAnimations   () const { return (SLint)_animations.size(); }
    void        reset           ();

    SLbool      changed         () const { return _changed; }
    void        changed         (SLbool changed) { _changed = changed; }


    // @todo find a better way to give access to the animation names to external stuff (like the gui)
    SLMAnimation animations() { return _animations; }

    void        updateAnimations();
    
protected:
    SLJoint*            _root;
    SLVJoint            _joints; //!< joint map for fast acces of joints
    SLMAnimation        _animations;
    SLMAnimationState   _animationStates;
    SLbool              _changed;   //!< did this skeleton change this frame (attribute for skeleton instance)
};
//-----------------------------------------------------------------------------
typedef std::vector<SLSkeleton*> SLVSkeleton;
//-----------------------------------------------------------------------------

#endif
