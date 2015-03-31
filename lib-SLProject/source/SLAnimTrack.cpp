//#############################################################################
//  File:      SLAnimTrack.cpp
//  Author:    Marc Wacker
//  Date:      Autumn 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif
#include <SLAnimTrack.h>
#include <SLAnimation.h>
#include <SLNode.h>
#include <SLCurveBezier.h>

//-----------------------------------------------------------------------------
/*! Constructor
*/
SLAnimTrack::SLAnimTrack(SLAnimation* animation)
            :_animation(animation)
{ }

//-----------------------------------------------------------------------------
/*! Destructor
*/
SLAnimTrack::~SLAnimTrack()
{
    for (auto kf : _keyframes) delete kf;
}

//-----------------------------------------------------------------------------
/*! Creates a new keyframed with the passed in timestamp.
    @note   It is required that the keyframes are created in chronological order.
            since we currently don't sort the keyframe list they have to be sorted
            before being created.
*/
SLKeyframe* SLAnimTrack::createKeyframe(SLfloat time)
{
    SLKeyframe* kf = createKeyframeImpl(time);
    _keyframes.push_back(kf);
    return kf;
}

//-----------------------------------------------------------------------------
/*! Getter for keyframes by index.
*/
SLKeyframe* SLAnimTrack::keyframe(SLint index)
{
    if (index < 0 || index >= numKeyframes())
        return nullptr;

    return _keyframes[index];
}

//-----------------------------------------------------------------------------
/*! Get the two keyframes to the left or the right of the passed in timestamp.
    If keyframes will wrap around, if there is no keyframe after the passed in time
    then the k2 result will be the first keyframe in the list.
    If only one keyframe exists the two values will be equivalent.
*/
SLfloat SLAnimTrack::getKeyframesAtTime(SLfloat time,
                                             SLKeyframe** k1,
                                             SLKeyframe** k2) const
{
    SLfloat t1, t2;
    SLint numKf = (SLint)_keyframes.size();
    float animationLength = _animation->lengthSec();

    assert(animationLength > 0.0f && "Animation length is invalid.");

    *k1 = *k2 = nullptr;
        
    // no keyframes or only one keyframe in animation, early out
    if (numKf == 0)
        return 0.0f;

    if (numKf < 2)
    {   *k1 = *k2 = _keyframes[0];
        return 0.0f;
    }

    // wrap time
    if (time > animationLength)
        time = fmod(time, animationLength);

    // @todo is it really required of us to check if time is < 0.0f here? Or should this be done higher up?
    while (time < 0.0f)
        time += animationLength;
        
    // search lower bound kf for given time
    // kf list must be sorted by time at this point
    // @todo we could use std::lower_bounds here
    // @todo this could be implemented much nicer
    //      use the following algorithm:
    //      1. find the keyframe that comes after the 'time' parameter
    //      2. if there is no keyframe after 'time' then set keframe 2 to the first keyframe in the list
    //          set t2 to animationLength + the time of the keyframe
    //      3. if there is a keyframe after 'time' then set keyframe 2 to that keyframe
    //          set t2 to the time of the keyframe
    //      4. now find the keyframe before keyframe 2 (if we use iterators here this is trivial!)
    //         set keyframe 1 to the keyframe found before keyframe 2
    SLint kfIndex = 0;
    for (SLint i = 0; i < numKf; ++i)
    {
        SLKeyframe* cur = _keyframes[i];

        if (cur->time() <= time)
        {   *k1 = cur;
            kfIndex = i;
        }
    }

    // time is than first kf
    if (*k1 == nullptr) 
    {   *k1 = _keyframes.back();
        // as long as k1 is in the back
    }

    t1 = (*k1)->time();

    if (*k1 == _keyframes.back())
    {   *k2 = _keyframes.front();
        t2 = animationLength + (*k2)->time();
    } else
    {   *k2 = _keyframes[kfIndex+1];
        t2 = (*k2)->time();
    }

    
    if (t1 == t2)
        return 0.0f;

    /// @todo   do we want to consider the edge case below or do we want imported animations to have
    ///         to have a keyframe at 0.0 time?
    ///         Is there a better solution for this problem?
    ///         e.x: the astroboy animation looks wrong when doing this (but thats because it is **** and kf0 and kfn dont match up...
    //
    // if an animation doesn't have a keyframe at 0.0 and 
    // k1 is the last keyframe and k2 is the first keyframe
    // and the current timestamp is just above zero in the timeline
    // 
    // like this:
    //   0.0                                animationLenth
    //    |-.--*----------*----*------*------|~~~~*
    //      ^  ^                      ^           ^
    //      |  t2                     t1          t2' // t2' is where we put the t2 value if its a wrap around!
    //     time 
    // then the calculation below wont work because time < t1.
    //
    //
    if (time < t1)
        time += animationLength;
        
    return (time - t1) / (t2 - t1);
}


//-----------------------------------------------------------------------------
/*! Constructor for specialized NodeAnimationTrack
*/
SLNodeAnimTrack::SLNodeAnimTrack(SLAnimation* animation)
                     :SLAnimTrack(animation),
                      _animatedNode(nullptr),
                      _interpolationCurve(nullptr),
                      _translationInterpolation(AI_Linear),
                      _rebuildInterpolationCurve(true)
{ }

//-----------------------------------------------------------------------------
/*! Destructor
*/
SLNodeAnimTrack::~SLNodeAnimTrack()
{
    if (_interpolationCurve)
        delete _interpolationCurve;
}

//-----------------------------------------------------------------------------
/*! Creates a new SLTransformKeyframe at 'time'.
*/
SLTransformKeyframe* SLNodeAnimTrack::createNodeKeyframe(SLfloat time)
{
    return static_cast<SLTransformKeyframe*>(createKeyframe(time));
}

//-----------------------------------------------------------------------------
/*! Calculates a new keyframe based on the input time and interpolation functions.
*/
void SLNodeAnimTrack::calcInterpolatedKeyframe(SLfloat time,
                                                    SLKeyframe* keyframe) const
{
    SLKeyframe* k1;
    SLKeyframe* k2;

    SLfloat t = getKeyframesAtTime(time, &k1, &k2);
    
    if (k1 == nullptr)
        return;

    SLTransformKeyframe* kfOut = static_cast<SLTransformKeyframe*>(keyframe);
    SLTransformKeyframe* kf1   = static_cast<SLTransformKeyframe*>(k1);
    SLTransformKeyframe* kf2   = static_cast<SLTransformKeyframe*>(k2);
    

    SLVec3f base = kf1->translation();
    SLVec3f translation;
    if (_translationInterpolation == AI_Linear)
        translation = base + (kf2->translation() - base) * t; 
    else
    {   if (_rebuildInterpolationCurve)
            buildInterpolationCurve();
        translation = _interpolationCurve->evaluate(time);
    }

    kfOut->translation(translation);
    
    SLQuat4f rotation;
    rotation = kf1->rotation().slerp(kf2->rotation(), t); // @todo provide a 2 parameter implementation for lerp, slerp etc.
    kfOut->rotation(rotation);

    base = kf1->scale();
    SLVec3f scale;
    scale = base + (kf2->scale() - base) * t;
    kfOut->scale(scale);
}

//-----------------------------------------------------------------------------
/*! Applies the animation with the input timestamp to the set animation target if it exists.
*/
void SLNodeAnimTrack::apply(SLfloat time, SLfloat weight, SLfloat scale)
{
    if (_animatedNode)
        applyToNode(_animatedNode, time, weight, scale);
}

//-----------------------------------------------------------------------------
/*! Applies the animation to the input node with the input timestamp and weight.
*/
void SLNodeAnimTrack::applyToNode(SLNode* node,
                                       SLfloat time,
                                       SLfloat weight,
                                       SLfloat scale)
{
    if (node == nullptr)
        return;

    SLTransformKeyframe kf(0, time);
    calcInterpolatedKeyframe(time, &kf);

    SLVec3f translation = kf.translation() * weight * scale;
    node->translate(translation, TS_Parent);

    // @todo update the slerp and lerp impelemtation for quaternions
    //       there is currently no early out for 1.0 and 0.0 inputs
    //       also provide a non OO version.
    SLQuat4f rotation = SLQuat4f().slerp(kf.rotation(), weight);
    node->rotate(rotation, TS_Parent);

    SLVec3f scl = kf.scale();// @todo find a good way to combine scale animations, we can't just scale them by a weight factor...
    node->scale(scl);
}

//-----------------------------------------------------------------------------
/*! Rebuilds the translation interpolation bezier curve.
*/
void SLNodeAnimTrack::buildInterpolationCurve() const
{
    if (numKeyframes() > 1)
    {
        if (_interpolationCurve) delete _interpolationCurve;

        // Build curve data w. cummulated times
        SLVec3f* points = new SLVec3f[numKeyframes()];
        SLfloat* times  = new SLfloat[numKeyframes()];
        SLfloat  curTime = 0;
        for (SLint i=0; i<numKeyframes(); ++i)
        {   times[i] = _keyframes[i]->time();
            points[i] = ((SLTransformKeyframe*)_keyframes[i])->translation();
        }

        // create curve and delete temp arrays again
        _interpolationCurve = new SLCurveBezier(points, times, (SLint)numKeyframes());
        delete[] points;
        delete[] times;
    }
}

//-----------------------------------------------------------------------------
/*! Implementation for the keyframe creation function.
*/
SLKeyframe* SLNodeAnimTrack::createKeyframeImpl(SLfloat time)
{
    return new SLTransformKeyframe(this, time);
}
    
//-----------------------------------------------------------------------------
/*! setter for the interpoilation curve
*/
void SLNodeAnimTrack::interpolationCurve(SLCurve* curve)
{
    if (_interpolationCurve)
        delete _interpolationCurve;

    _interpolationCurve = curve;
    _rebuildInterpolationCurve = false;
}
//-----------------------------------------------------------------------------
