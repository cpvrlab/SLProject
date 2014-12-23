//#############################################################################
//  File:      SLAnimation.h
//  Author:    Marc Wacker
//  Date:      Autumn 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLANIMATION_H
#define SLANIMATION_H

#include <stdafx.h>
#include <SLAnimationTrack.h>
#include <SLJoint.h>
#include <SLEnums.h>

class SLSkeleton;

//-----------------------------------------------------------------------------
//! SLAnimation is the
/*! 
    SLAnimation is a container for multiple types of SLAnimationTracks that 
    should be kept together. For example a walk animation would consist of
    all the SLAnimationTracks that make a SLSkeleton walk.
    The SLAnimation is also the one that knows the length of the animation.
*/
class SLAnimation
{
public:
                        SLAnimation(const SLstring& name, SLfloat duration);
                       ~SLAnimation();
    
            SLfloat     nextKeyframeTime(SLfloat time);
            SLfloat     prevKeyframeTime(SLfloat time);
            SLbool      affectsNode (SLNode* node);
            void        apply       (SLfloat time,
                                     SLfloat weight = 1.0f,
                                     SLfloat scale = 1.0f);
            void        applyToNode (SLNode* node,
                                     SLfloat time,
                                     SLfloat weight = 1.0f,
                                     SLfloat scale = 1.0f);
            void        apply       (SLSkeleton* skel,
                                     SLfloat time,
                                     SLfloat weight = 1.0f,
                                     SLfloat scale = 1.0f);
            void        resetNodes  ();

    // static creator 
    static SLAnimation* createAnimation(const SLstring& name,
                                        SLfloat duration,
                                        SLbool enabled = true,
                                        SLEasingCurve easing = EC_linear,
                                        SLAnimLooping looping = AL_loop);
    // track creators
    SLNodeAnimationTrack* createNodeAnimationTrack();
    SLNodeAnimationTrack* createNodeAnimationTrack(SLuint handle);
    SLNodeAnimationTrack* createSimpleTranslationNodeTrack(SLNode* target, const SLVec3f& endPos);
    SLNodeAnimationTrack* createSimpleRotationNodeTrack(SLNode* target, SLfloat angleDeg, const SLVec3f& axis);
    SLNodeAnimationTrack* createSimpleScalingNodeTrack(SLNode* target, const SLVec3f& endScale);
    SLNodeAnimationTrack* createEllipticNodeTrack(SLNode* target, 
                                                  SLfloat radiusA, SLAxis axisA,
                                                  SLfloat radiusB, SLAxis axisB);
    // Getters
    const   SLstring&   name        () { return _name; }
            SLfloat     length      () const { return _length; }

    // Setters
            void        name        (const SLstring& name) { _name = name; }
            void        length      (SLfloat length);

protected:
    SLstring                _name;              //!< name of the animation
    SLfloat                 _length;            //!< duration of the animation
    SLMNodeAnimationTrack   _nodeAnimTracks;    //!< map of all the node tracks in this animation
};
//-----------------------------------------------------------------------------
typedef vector<SLAnimation*>        SLVAnimation;
typedef map<SLstring, SLAnimation*> SLMAnimation;
//-----------------------------------------------------------------------------
#endif


