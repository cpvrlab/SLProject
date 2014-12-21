//#############################################################################
//  File:      SLAnimationTrack.h
//  Author:    Marc Wacker
//  Date:      Autumn 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLANIMATIONTRACK_H
#define SLANIMATIONTRACK_H

#include <stdafx.h>
#include <SLEnums.h>
#include <SLKeyframe.h>

class SLNode;
class SLAnimation;
class SLCurve;

//-----------------------------------------------------------------------------
//! Abstract base class for SLAnimationTracks providing time and keyframe functions
/*! 
    An animation track is a specialized track that affects one object or value
    at most. For example a track in a skeleton animation will affect one
    joint at a time.
*/
class SLAnimationTrack
{
public:
                        SLAnimationTrack        (SLAnimation* parent,
                                                 SLuint handle);
                       ~SLAnimationTrack        ();

            SLKeyframe* createKeyframe          (SLfloat time);   // create and add a new keyframe

            SLfloat     getKeyframesAtTime      (SLfloat time,
                                                 SLKeyframe** k1,
                                                 SLKeyframe** k2) const;
    virtual void        calcInterpolatedKeyframe(SLfloat time,
                                                 SLKeyframe* keyframe) const = 0; // we need a way to get an output value for a time we put in
    virtual void        apply                   (SLfloat time,
                                                 SLfloat weight = 1.0f,
                                                 SLfloat scale = 1.0f) = 0; // applies the animation clip to its target
            SLuint      numKeyframes            () { return (SLuint)_keyframes.size(); }
            SLKeyframe* keyframe                (SLuint index);
protected:
    /// Keyframe creator function for derived implementations
    virtual SLKeyframe* createKeyframeImpl(SLfloat time) = 0;
    
    SLAnimation*        _parent;            //!< parent animation that created this track
    SLuint              _handle;            //!< unique handle for this track inside its parent animation
    SLVKeyframe         _keyframes;         //!< keyframe list for this track

};


//-----------------------------------------------------------------------------
/*! Specialized SLAnimationTrack for node animations */
class SLNodeAnimationTrack : public SLAnimationTrack
{
public:
    SLNodeAnimationTrack(SLAnimation* parent, SLuint handle);    
    ~SLNodeAnimationTrack();

    SLTransformKeyframe* createNodeKeyframe(SLfloat time);
    
    void                animationTarget(SLNode* target) { _animationTarget = target; }
    SLNode*             animationTarget() { return _animationTarget; }

    virtual void        calcInterpolatedKeyframe(SLfloat time, SLKeyframe* keyframe) const;

    virtual void        apply(SLfloat time, SLfloat weight = 1.0f, SLfloat scale = 1.0f);
    virtual void        applyToNode(SLNode* node, SLfloat time, SLfloat weight = 1.0f, SLfloat scale = 1.0f);
    
    void                interpolationCurve(SLCurve* curve);
    void                translationInterpolation(SLAnimInterpolation interp) { _translationInterpolation = interp; }

protected:
    SLNode*                 _animationTarget;           //!< the default target for this track
    SLAnimInterpolation     _translationInterpolation;  //!< interpolation mode for translations (bezier or linear)
    SLbool                  _rebuildInterpolationCurve; //!< dirty flag of the bezier curve
    mutable SLCurve*        _interpolationCurve;        //!< the translation interpolation curve


    void                buildInterpolationCurve() const;
    virtual SLKeyframe* createKeyframeImpl(SLfloat time);
};
//-----------------------------------------------------------------------------
typedef map<SLuint, SLNodeAnimationTrack*> SLMNodeAnimationTrack;
//-----------------------------------------------------------------------------
#endif








