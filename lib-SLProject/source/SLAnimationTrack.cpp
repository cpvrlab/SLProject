
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
    // @todo this could be implemented much nicer
    //      use the following algorithm:
    //      1. find the keyframe that comes after the 'time' parameter
    //      2. if there is no keyframe after the 'time' then set keframe 2 to the first keyframe in the list
    //          set t2 to animationLength + the time of the keyframe
    //      3. if there is a keyframe after 'time' then set keyframe 2 to that keyframe
    //          set t2 to the time of the keyframe
    //      4. now find the keyframe before keyframe 2 (if we use iterators here this is trivial!)
    //         set keyframe 1 to the keyframe found before keyframe 2
    SLint kfIndex = 0;
    for (SLint i = 0; i < numKf; ++i)
    {
        SLKeyframe* cur = _keyframeList[i];

        if (cur->time() <= time)
        {
            *k1 = cur;
            kfIndex = i;
        }
    }

    // time is than first kf
    if (*k1 == NULL) {
        *k1 = _keyframeList.back();
        // as long as k1 is in the back
    }

    t1 = (*k1)->time();


    SLbool stopCondition = (*k1) == NULL;
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

    
    if (t1 == t2)
    {
        return 0.0f;
    }

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

    SLfloat t = (time - t1) / (t2 - t1);
    bool test = t<0 || 1<t;
    //return 0.5f*sin(tempReturnVal*SL_PI - SL_PI*0.5f) + 0.5f; 

    
    
    // temporary test code below (prints out left and right keyframe for a track
    // was used to check which keyframe interpolation failed
    // @todo remove this when finished
    static int kfL = -1;
    static int kfR = -1;
    static int kfL_last = -1;
    static int kfR_last = -1;
    
    for (int i = 0; i < _keyframeList.size(); i++)
    {
        if (_keyframeList[i] == *k1)
            kfL = i;
        if (_keyframeList[i] == *k2)
            kfR = i;
    }

    if (kfL != kfL_last) {
        std::cout << "left: " << kfL;
        kfL_last = kfL;
    }
    if (kfR != kfR_last) {
        std::cout << "   right: " << kfR << "\n" << std::endl;
        kfR_last = kfR;
    }


    return t;
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

    SLVec3f scl = kf.scale();// @todo find a good way to combine scale animations, we can't just scale them by a weight factor...
    node->scale(scl);
}

SLKeyframe* SLNodeAnimationTrack::createKeyframeImpl(SLfloat time)
{
    return new SLTransformKeyframe(this, time);
}