
#include <stdafx.h>
#include <SLAnimationTrack.h>
#include <SLAnimation.h>
#include <SLNode.h>

SLAnimationTrack::SLAnimationTrack(SLAnimation* parent, SLuint handle)
: _parent(parent), _handle(handle)
{ }

SLAnimationTrack::~SLAnimationTrack()
{
    for (SLint i = 0; i < _keyframeList.size(); ++i)
        delete _keyframeList[i];
}

SLKeyframe* SLAnimationTrack::createKeyframe(SLfloat time)
{
    SLKeyframe* kf = createKeyframeImpl(time);
    _keyframeList.push_back(kf);

    return kf;
}

void SLAnimationTrack::keyframesChanged()
{
    // @todo ...
}


// @todo support the special case of animationtracks that contain invalid keyframes
//       animationLength:   |----------------|
//       Keyframes:          *                    *
//                           k1                   k2
//  the animation above has two keyframes, but one is invalid so this function should only return
//  the first keyframe and an interpolation value of 0.0.
// @todo this function could be implemented better, rework it
SLfloat SLAnimationTrack::getKeyframesAtTime(SLfloat time, SLKeyframe** k1, SLKeyframe** k2) const
{
    SLfloat t1, t2;
    SLint numKf = _keyframeList.size();
    float animationLength = _parent->length();

    assert(animationLength > 0.0f && "Animation length is invalid.");

    *k1 = *k2 = NULL;
        
    // no keyframes or only one keyframe in animation, early out
    if (numKf == 0)
        return 0.0f;
    if (numKf < 2)
    {
        *k1 = *k2 = _keyframeList[0];
        return 0.0f;
    }

    // wrap time
    if (time > animationLength)
        time = fmod(time, animationLength);


    // search lower bound kf for given time
    // kf list must be sorted by time at this point
    // @todo we could use std::lower_bounds here
    SLint kfIndex = 0;
    for (SLint i = 0; i < numKf; ++i)
    {
        SLKeyframe* cur = _keyframeList[i];

        if (cur->time() < time)
        {
            *k1 = cur;
            kfIndex = i;
        }
    }

    if (*k1 == _keyframeList.back())
    {
        *k2 = _keyframeList.front();
        t2 = animationLength + (*k2)->time();
    }
    else
    {
        *k2 = _keyframeList[kfIndex+1];
        t2 = (*k2)->time();
    }

    t1 = (*k1)->time();


    if (t1 == t2)
    {
        return 0.0f;
    }
        
    //return 0.5f*sin(tempReturnVal*SL_PI - SL_PI*0.5f) + 0.5f; 

    return (time - t1) / (t2 -t1);
}


SLNodeAnimationTrack::SLNodeAnimationTrack(SLAnimation* parent, SLuint handle)
: SLAnimationTrack(parent, handle)
{ }


SLTransformKeyframe* SLNodeAnimationTrack::createNodeKeyframe(SLfloat time)
{
    return static_cast<SLTransformKeyframe*>(createKeyframe(time));
}

void SLNodeAnimationTrack::calcInterpolatedKeyframe(SLfloat time, SLKeyframe* keyframe) const
{
    SLKeyframe* k1;
    SLKeyframe* k2;

    SLfloat t = getKeyframesAtTime(time, &k1, &k2);
    
    if (k1 == NULL)
        return;

    SLTransformKeyframe* kfOut = static_cast<SLTransformKeyframe*>(keyframe);
    SLTransformKeyframe* kf1 = static_cast<SLTransformKeyframe*>(k1);
    SLTransformKeyframe* kf2 = static_cast<SLTransformKeyframe*>(k2);
    
    // @todo optimize interpolation for all parameters
    // @todo provide more customization for the interpolation
    SLVec3f base = kf1->translation();
    SLVec3f translation;
    translation = base + (kf2->translation() - base) * t; // @todo implement general interpolation functions

    kfOut->translation(translation);
    
    SLQuat4f rotation;
    rotation = kf1->rotation().slerp(kf2->rotation(), t); // @todo provide a 2 parameter implementation for lerp, slerp etc.
    kfOut->rotation(rotation);

    base = kf1->scale();
    SLVec3f scale;
    scale = base + (kf2->scale() - base) * t;
    kfOut->scale(scale);
}

void SLNodeAnimationTrack::apply(SLfloat time, SLfloat weight, SLfloat scale)
{
    applyToNode(_animationTarget, time, weight, scale);
}

void SLNodeAnimationTrack::applyToNode(SLNode* node, SLfloat time, SLfloat weight, SLfloat scale)
{
    if (node == NULL)
        return;

    SLTransformKeyframe kf(0, time);
    calcInterpolatedKeyframe(time, &kf);

    SLVec3f translation = kf.translation() * weight * scale;
    node->translate(translation, TS_Parent);

    // @todo update the slerp and lerp impelemtation for quaternions
    //       there is currently no early out for 1.0 and 0.0 inputs
    //       also provide a non OO version.
    // @todo define quaternion constants for identity quats
    SLQuat4f rotation = SLQuat4f().slerp(kf.rotation(), weight);
    node->rotate(rotation, TS_Parent);

    SLVec3f scl = kf.scale() * weight * scale;
    node->scale(scl);
}

SLKeyframe* SLNodeAnimationTrack::createKeyframeImpl(SLfloat time)
{
    return new SLTransformKeyframe(this, time);
}