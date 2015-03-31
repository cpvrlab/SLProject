//#############################################################################
//  File:      SLAnimTrack.h
//  Author:    Marc Wacker
//  Date:      Autumn 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLANIMTRACK_H
#define SLANIMTRACK_H

#include <stdafx.h>
#include <SLEnums.h>
#include <SLKeyframe.h>

class SLNode;
class SLAnimation;
class SLCurve;

//-----------------------------------------------------------------------------
//! Abstract base class for SLAnimationTracks providing time and keyframe functions
/*! 
An animation track is a specialized track that affects a single SLNode or an
SLJoint of an SLSkeleton by interpolating its transform. It holds therefore a
list of SLKeyframe. For a smooth motion it can interpolate the transform at a
given time between two neighboring SLKeyframe.
*/
class SLAnimTrack
{
public:
                        SLAnimTrack             (SLAnimation* parent);
                       ~SLAnimTrack             ();

            SLKeyframe* createKeyframe          (SLfloat time);   // create and add a new keyframe
            SLfloat     getKeyframesAtTime      (SLfloat time,
                                                 SLKeyframe** k1,
                                                 SLKeyframe** k2) const;
    virtual void        calcInterpolatedKeyframe(SLfloat time,
                                                 SLKeyframe* keyframe) const = 0; // we need a way to get an output value for a time we put in
    virtual void        apply                   (SLfloat time,
                                                 SLfloat weight = 1.0f,
                                                 SLfloat scale = 1.0f) = 0; // applies the animation clip to its target
            SLint       numKeyframes            () const { return (SLint)_keyframes.size(); }
            SLKeyframe* keyframe                (SLint index);
protected:
    /// Keyframe creator function for derived implementations
    virtual SLKeyframe* createKeyframeImpl      (SLfloat time) = 0;
    
    SLAnimation*        _animation;     //!< parent animation that created this track
    SLVKeyframe         _keyframes;     //!< keyframe list for this track
};

//-----------------------------------------------------------------------------
//! Specialized animation track for node animations
/*! 
    Allows for translation, scale and rotation parameters to be animated.
    Also allows for either linear or bezier interpolation of the position
    parameter in the track.
*/
class SLNodeAnimTrack : public SLAnimTrack
{
public:
                        SLNodeAnimTrack         (SLAnimation* parent);
                       ~SLNodeAnimTrack         ();

   SLTransformKeyframe* createNodeKeyframe      (SLfloat time);
    
            void        animatedNode            (SLNode* target) { _animatedNode = target; }
            SLNode*     animatedNode            () { return _animatedNode; }

    virtual void        calcInterpolatedKeyframe(SLfloat time, SLKeyframe* keyframe) const;
    virtual void        apply                   (SLfloat time, SLfloat weight = 1.0f, SLfloat scale = 1.0f);
    virtual void        applyToNode             (SLNode* node, SLfloat time, SLfloat weight = 1.0f, SLfloat scale = 1.0f);
    
            void        interpolationCurve      (SLCurve* curve);
            void        translationInterpolation(SLAnimInterpolation interp) { _translationInterpolation = interp; }

protected:       
    void                buildInterpolationCurve () const;
    virtual SLKeyframe* createKeyframeImpl      (SLfloat time);

    SLNode*             _animatedNode;              //!< the target node for this track_nodeID
    mutable SLCurve*    _interpolationCurve;        //!< the translation interpolation curve
    SLAnimInterpolation _translationInterpolation;  //!< interpolation mode for translations (bezier or linear)
    SLbool              _rebuildInterpolationCurve; //!< dirty flag of the bezier curve
};
//-----------------------------------------------------------------------------
typedef map<SLuint, SLNodeAnimTrack*> SLMNodeAnimTrack;
//-----------------------------------------------------------------------------
#endif








