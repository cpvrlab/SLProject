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

// @todo order the keyframe sets based on their time value
// @todo provide an iterator over the keyframes

//-----------------------------------------------------------------------------
//! Virtual baseclass for specialized animation clips.
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

protected:
            void        keyframesChanged        ();
    virtual SLKeyframe* createKeyframeImpl      (SLfloat time) = 0;

    SLAnimation*       _parent;
    SLuint             _handle;     //!< unique handle for this track inside its parent animation
    SLVKeyframe        _keyframes;
    SLfloat            _localTime;
};
//-----------------------------------------------------------------------------
//! Specialized animation track for animating nodes
class SLNodeAnimationTrack : public SLAnimationTrack
{
public:
    SLNodeAnimationTrack(SLAnimation* parent, SLuint handle);    

    // static creator functions for common animation types (old SLAnimation constructors)
    static SLNodeAnimationTrack* createEllipticTrack(SLAnimation* parent,
                                                     SLfloat radiusA, SLAxis axisA,
                                                     SLfloat radiusB, SLAxis axisB);

    SLTransformKeyframe* createNodeKeyframe(SLfloat time);
    
    void            animationTarget(SLNode* target) { _animationTarget = target; }
    SLNode*         animationTarget() { return _animationTarget; }

    virtual void    calcInterpolatedKeyframe(SLfloat time, SLKeyframe* keyframe) const;
    // apply this track to a specified node
    virtual void    apply(SLfloat time, SLfloat weight = 1.0f, SLfloat scale = 1.0f);
    virtual void    applyToNode(SLNode* node, SLfloat time, SLfloat weight = 1.0f, SLfloat scale = 1.0f);

protected:
    SLNode*             _animationTarget;   //!< the default target for this track
    SLAnimInterpolation _translationInterpolation;
    SLbool              _rebuildInterpolationCurve;
    SLCurve*            _interpolationCurve;

    void                buildInterpolationCurve() const;

    virtual SLKeyframe* createKeyframeImpl(SLfloat time);
};
//-----------------------------------------------------------------------------
typedef map<SLuint, SLNodeAnimationTrack*> SLMNodeAnimationTrack;
//-----------------------------------------------------------------------------
#endif








