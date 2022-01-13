//#############################################################################
//  File:      SLAnimTrack.h
//  Date:      Autumn 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marc Wacker, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLANIMTRACK_h
#define SLANIMTRACK_h

#include <map>

#include <SLEnums.h>
#include <SLAnimKeyframe.h>

class SLNode;
class SLAnimation;
class SLCurve;
class SLSceneView;

//-----------------------------------------------------------------------------
//! Abstract base class for SLAnimationTracks providing time and keyframe functions
/*!
An animation track is a specialized track that affects a single SLNode or an
SLJoint of an SLAnimSkeleton by interpolating its transform. It holds therefore a
list of SLKeyframe. For a smooth motion it can interpolate the transform at a
given time between two neighboring SLKeyframe.
*/
class SLAnimTrack
{
public:
    SLAnimTrack(SLAnimation* parent);
    virtual ~SLAnimTrack();

    SLAnimKeyframe* createKeyframe(SLfloat time); // create and add a new keyframe
    SLfloat         getKeyframesAtTime(SLfloat          time,
                                       SLAnimKeyframe** k1,
                                       SLAnimKeyframe** k2) const;
    virtual void    calcInterpolatedKeyframe(SLfloat         time,
                                             SLAnimKeyframe* keyframe) const = 0; // we need a way to get an output value for a time we put in
    virtual void    apply(SLfloat time,
                          SLfloat weight = 1.0f,
                          SLfloat scale  = 1.0f)                              = 0;
    virtual void    drawVisuals(SLSceneView* sv)                             = 0;
    SLint           numKeyframes() const { return (SLint)_keyframes.size(); }
    SLAnimKeyframe* keyframe(SLint index);

protected:
    /// Keyframe creator function for derived implementations
    virtual SLAnimKeyframe* createKeyframeImpl(SLfloat time) = 0;

    SLAnimation* _animation; //!< parent animation that created this track
    SLVKeyframe  _keyframes; //!< keyframe list for this track
};

//-----------------------------------------------------------------------------
//! Specialized animation track for node animations
/*!
    Allows for translation, scale and rotation parameters to be animated.
    Also allows for either linear or Bezier interpolation of the position
    parameter in the track.
*/
class SLNodeAnimTrack : public SLAnimTrack
{
public:
    SLNodeAnimTrack(SLAnimation* parent);
    virtual ~SLNodeAnimTrack();

    SLTransformKeyframe* createNodeKeyframe(SLfloat time);

    void    animatedNode(SLNode* target) { _animatedNode = target; }
    SLNode* animatedNode() { return _animatedNode; }

    virtual void calcInterpolatedKeyframe(SLfloat time, SLAnimKeyframe* keyframe) const;
    virtual void apply(SLfloat time, SLfloat weight = 1.0f, SLfloat scale = 1.0f);
    virtual void applyToNode(SLNode* node, SLfloat time, SLfloat weight = 1.0f, SLfloat scale = 1.0f);
    virtual void drawVisuals(SLSceneView* sv);

    void interpolationCurve(SLCurve* curve);
    void translationInterpolation(SLAnimInterpolation interp) { _translationInterpolation = interp; }

protected:
    void                    buildInterpolationCurve() const;
    virtual SLAnimKeyframe* createKeyframeImpl(SLfloat time);

    SLNode*             _animatedNode;              //!< the target node for this track_nodeID
    mutable SLCurve*    _interpolationCurve;        //!< the translation interpolation curve
    SLAnimInterpolation _translationInterpolation;  //!< interpolation mode for translations (Bezier or linear)
    SLbool              _rebuildInterpolationCurve; //!< dirty flag of the Bezier curve
};
//-----------------------------------------------------------------------------
typedef std::map<SLuint, SLNodeAnimTrack*> SLMNodeAnimTrack;
//-----------------------------------------------------------------------------
#endif
