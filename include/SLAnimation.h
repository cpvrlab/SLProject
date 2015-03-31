//#############################################################################
//  File:      SLAnimation.h
//  Author:    Marc Wacker
//  Date:      Autumn 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLANIMATION_H
#define SLANIMATION_H

#include <stdafx.h>
#include <SLAnimTrack.h>
#include <SLJoint.h>
#include <SLEnums.h>

class SLSkeleton;

//-----------------------------------------------------------------------------
//! SLAnimation is the base container for all animation data.
/*! 
SLAnimation is a container for multiple SLAnimTrack that build an animation.
E.g. a walk animation would consist of all the SLAnimTrack that make a
SLSkeleton walk. It also knows the length of the animation.

An animation for a SLSkeleton with n joints must consist of 1 to n
SLNodeAnimTrack. The SLAnimation class keeps a map with index -> SLNodeAnimTrack
pairs, the index for the SLNodeAnimTrack must match the index of a bone in the
target SLSkeleton. This method allows us to animate multiple identical, or similar
SLSkeletons with the same SLAnimation.
*/
class SLAnimation
{
public:
                        SLAnimation     (const SLstring& name, SLfloat duration);
                       ~SLAnimation     ();
    
            SLfloat     nextKeyframeTime(SLfloat time);
            SLfloat     prevKeyframeTime(SLfloat time);
            SLbool      affectsNode     (SLNode* node);
            void        apply           (SLfloat time,
                                         SLfloat weight = 1.0f,
                                         SLfloat scale = 1.0f);
            void        applyToNode     (SLNode* node,
                                         SLfloat time,
                                         SLfloat weight = 1.0f,
                                         SLfloat scale = 1.0f);
            void        apply           (SLSkeleton* skel,
                                         SLfloat time,
                                         SLfloat weight = 1.0f,
                                         SLfloat scale = 1.0f);
            void        resetNodes      ();

    // static creator 
    static SLAnimation* create          (const SLstring& name,
                                         SLfloat duration,
                                         SLbool enabled = true,
                                         SLEasingCurve easing = EC_linear,
                                         SLAnimLooping looping = AL_loop);
    // track creators
    SLNodeAnimTrack*    createNodeAnimationTrack        ();
    SLNodeAnimTrack*    createNodeAnimationTrack        (SLuint handle);
    SLNodeAnimTrack*    createSimpleTranslationNodeTrack(SLNode* target, const SLVec3f& endPos);
    SLNodeAnimTrack*    createSimpleRotationNodeTrack   (SLNode* target, SLfloat angleDeg, const SLVec3f& axis);
    SLNodeAnimTrack*    createSimpleScalingNodeTrack    (SLNode* target, const SLVec3f& endScale);
    SLNodeAnimTrack*    createEllipticNodeTrack         (SLNode* target,
                                                         SLfloat radiusA, SLAxis axisA,
                                                         SLfloat radiusB, SLAxis axisB);
    // Getters
    const   SLstring&   name            () { return _name; }
            SLfloat     lengthSec       () const { return _lengthSec; }

    // Setters
            void        name            (const SLstring& name) { _name = name; }
            void        lengthSec       (SLfloat lengthSec);

protected:
    SLstring            _name;              //!< name of the animation
    SLfloat             _lengthSec;            //!< duration of the animation in seconds
    SLMNodeAnimTrack    _nodeAnimTracks;    //!< map of all the node tracks in this animation
};
//-----------------------------------------------------------------------------
typedef vector<SLAnimation*>        SLVAnimation;
typedef map<SLstring, SLAnimation*> SLMAnimation;
//-----------------------------------------------------------------------------
#endif


