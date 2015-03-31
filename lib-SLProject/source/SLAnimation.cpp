//#############################################################################
//  File:      SLAnimation.cpp
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
#include <SLScene.h>
#include <SLAnimation.h>
#include <SLAnimManager.h>
#include <SLSkeleton.h>
#include <SLCurveBezier.h>

//-----------------------------------------------------------------------------
/*! Constructor
*/
SLAnimation::SLAnimation(const SLstring& name, SLfloat duration)
            : _name(name), _lengthSec(duration)
{ 
}

//-----------------------------------------------------------------------------
/*! Destructor
*/
SLAnimation::~SLAnimation()
{
    for (auto it : _nodeAnimTracks)
        delete it.second;
}

//-----------------------------------------------------------------------------
/*! Setter for the animation length
*/
void SLAnimation::lengthSec(SLfloat lengthSec)
{
    _lengthSec = lengthSec;
}

//-----------------------------------------------------------------------------
/*! Returns the timestamp for the next keyframe in all of the tracks.
*/
SLfloat SLAnimation::nextKeyframeTime(SLfloat time)
{
    // find the closest keyframe time to the right
    SLfloat result = _lengthSec;
    SLKeyframe* kf1;
    SLKeyframe* kf2;
    
    for (auto it : _nodeAnimTracks)
    {   it.second->getKeyframesAtTime(time, &kf1, &kf2);
        if (kf2->time() < result && kf2->time() >= time)
            result = kf2->time();
    }

    return result;
}

//-----------------------------------------------------------------------------
/*! Returns the timestamp for the previous keyframe in all of the tracks.
*/
SLfloat SLAnimation::prevKeyframeTime(SLfloat time)
{
    // find the closest keyframe time to the right
    SLfloat result = 0.0;
    SLKeyframe* kf1;
    SLKeyframe* kf2;

    // shift the time a little bit to the left or else the getKeyframesAtTime function
    // would return the same keyframe over and over again
    time -= 0.01f; 
    if (time <= 0.0f)
        return 0.0f;

    for (auto it : _nodeAnimTracks)
    {   it.second->getKeyframesAtTime(time, &kf1, &kf2);
        if (kf1->time() > result && kf1->time() <= time)
            result = kf1->time();
    }

    return result;
}

//-----------------------------------------------------------------------------
/*! Returns true if node is the animationTarget of any of the SLNodeAnimationTracks
in this animation.
*/
SLbool SLAnimation::affectsNode(SLNode* node)
{
    for (auto it : _nodeAnimTracks)
        if (it.second->animatedNode() == node)
            return true;

    return false;
}

//-----------------------------------------------------------------------------
/*! Creates a new SLNodeAnimationTrack with the next free handle.
*/
SLNodeAnimTrack* SLAnimation::createNodeAnimationTrack()
{
    SLuint freeIndex = 0;
    
    auto it = _nodeAnimTracks.begin();
    for (; it != _nodeAnimTracks.end() && freeIndex == it->first; ++it, ++freeIndex)
    { }

    return createNodeAnimationTrack(freeIndex);
}

//-----------------------------------------------------------------------------
/*! Creates a new SLNodeAnimationTrack with the passed in handle.
*/
SLNodeAnimTrack* SLAnimation::createNodeAnimationTrack(SLuint id)
{
    // track with same handle already exists
    if (_nodeAnimTracks.find(id) != _nodeAnimTracks.end())
        return nullptr;

    _nodeAnimTracks[id] = new SLNodeAnimTrack(this);

    return _nodeAnimTracks[id];
}

//-----------------------------------------------------------------------------
/*! Applies all animation tracks for the passed in timestamp, weight and scale.
*/
void SLAnimation::apply(SLfloat time, SLfloat weight , SLfloat scale)
{
    for (auto it : _nodeAnimTracks)
        it.second->apply(time, weight, scale);
}

//-----------------------------------------------------------------------------
/*! Applies all node tracks of this animation on a single node
*/
void SLAnimation::applyToNode(SLNode* node, SLfloat time, SLfloat weight, SLfloat scale)
{
    for (auto it : _nodeAnimTracks)
        it.second->applyToNode(node, time, weight, scale);
}

//-----------------------------------------------------------------------------
/*! Applies all the tracks to their respective joints in the passed in skeleton.
*/
void SLAnimation::apply(SLSkeleton* skel, SLfloat time, SLfloat weight, SLfloat scale)
{
    for (auto it : _nodeAnimTracks)
    {   SLJoint* joint = skel->getJoint(it.first);
        it.second->applyToNode(joint, time, weight, scale);
    }

}

//-----------------------------------------------------------------------------
/*! Resets all default animation targets to their initial state.
*/
void SLAnimation::resetNodes()
{
    for (auto it : _nodeAnimTracks)
        it.second->animatedNode()->resetToInitialState();
}

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/*! Creates new SLAnimation istance for node animations. It will already create and set parameters
for the respective SLAnimPlayback.
*/
SLAnimation* SLAnimation::create(const SLstring& name,
                                          SLfloat duration,
                                          SLbool enabled,
                                          SLEasingCurve easing,
                                          SLAnimLooping looping)
{
    SLAnimation* anim = SLScene::current->animManager().createNodeAnimation(name, duration);
    SLAnimPlayback* playback = SLScene::current->animManager().getNodeAnimPlayack(anim->name());
    playback->enabled(enabled);
    playback->easing(easing);
    playback->loop(looping);
    return anim;
}

//-----------------------------------------------------------------------------
/*! Specialized SLNodeAnimationTrack creator for a two keyframe translation animation
*/
SLNodeAnimTrack* SLAnimation::createSimpleTranslationNodeTrack(SLNode* target,
                                                                    const SLVec3f& endPos)
{
    SLNodeAnimTrack* track = createNodeAnimationTrack();
    target->setInitialState();
    track->animatedNode(target);
    track->createNodeKeyframe(0.0f); // create zero kf
    track->createNodeKeyframe(lengthSec())->translation(endPos); // create end scale keyframe
    return track;
}

//-----------------------------------------------------------------------------
/*! Specialized SLNodeAnimationTrack creator for a two keyframe rotation animation
*/
SLNodeAnimTrack* SLAnimation::createSimpleRotationNodeTrack(SLNode* target,
                                                                 SLfloat angleDeg,
                                                                 const SLVec3f& axis)
{
    SLNodeAnimTrack* track = createNodeAnimationTrack();
    target->setInitialState();
    track->animatedNode(target);
    track->createNodeKeyframe(0.0f); // create zero kf
    track->createNodeKeyframe(lengthSec())->rotation(SLQuat4f(angleDeg, axis)); // create end scale keyframe
    return track;
}
  
//-----------------------------------------------------------------------------
/*! Specialized SLNodeAnimationTrack creator for a two keyframe scaling animation
*/
SLNodeAnimTrack* SLAnimation::createSimpleScalingNodeTrack(SLNode* target,
                                                                const SLVec3f& endScale)
{    
    SLNodeAnimTrack* track = createNodeAnimationTrack();
    target->setInitialState();
    track->animatedNode(target);
    track->createNodeKeyframe(0.0f); // create zero kf
    track->createNodeKeyframe(lengthSec())->scale(endScale); // create end scale keyframe
    return track;
}
  
//-----------------------------------------------------------------------------
/*! Specialized SLNodeAnimationTrack creator for an elliptic node animation
*/
SLNodeAnimTrack* SLAnimation::createEllipticNodeTrack(SLNode* target,
                                                           SLfloat radiusA, SLAxis axisA,
                                                           SLfloat radiusB, SLAxis axisB)
{
    assert(axisA!=axisB && radiusA>0 && radiusB>0);
    SLNodeAnimTrack* track = createNodeAnimationTrack();
    target->setInitialState();
    track->animatedNode(target);

    /* The ellipse is defined by 5 keyframes: A,B,C,D and again A

        c2----B----c1

    c3                 c0
    ¦                   ¦
    ¦         ¦         ¦
    C       --0--       A
    ¦         ¦         ¦
    ¦                   ¦
    c4                 c7 

        c5----D----c6
    */

    SLVec3f A(0,0,0); A.comp[axisA] =  radiusA;
    SLVec3f B(0,0,0); B.comp[axisB] =  radiusB;
    SLVec3f C(0,0,0); C.comp[axisA] = -radiusA;
    SLVec3f D(0,0,0); D.comp[axisB] = -radiusB;

    // Control points with the magic factor kappa for control points
    SLfloat k = 0.5522847498f;

    SLVec3f controls[8];
    for (SLint i=0; i<8; ++i) controls[i].set(0,0,0);
    controls[0].comp[axisA] = radiusA; controls[0].comp[axisB] = k *  radiusB;
    controls[1].comp[axisB] = radiusB; controls[1].comp[axisA] = k *  radiusA;
    controls[2].comp[axisB] = radiusB; controls[2].comp[axisA] = k * -radiusA;
    controls[3].comp[axisA] =-radiusA; controls[3].comp[axisB] = k *  radiusB;
    controls[4].comp[axisA] =-radiusA; controls[4].comp[axisB] = k * -radiusB;
    controls[5].comp[axisB] =-radiusB; controls[5].comp[axisA] = k * -radiusA;
    controls[6].comp[axisB] =-radiusB; controls[6].comp[axisA] = k *  radiusA;
    controls[7].comp[axisA] = radiusA; controls[7].comp[axisB] = k * -radiusB;

    // Add keyframes
    SLfloat t4 = lengthSec() / 4.0f;
    track->createNodeKeyframe(0.0f * t4)->translation(A);
    track->createNodeKeyframe(1.0f * t4)->translation(B);
    track->createNodeKeyframe(2.0f * t4)->translation(C);
    track->createNodeKeyframe(3.0f * t4)->translation(D);
    track->createNodeKeyframe(4.0f * t4)->translation(A);


    // Build curve data w. cummulated times
    SLVec3f* points = new SLVec3f[track->numKeyframes()];
    SLfloat* times  = new SLfloat[track->numKeyframes()];
    for (SLint i=0; i<track->numKeyframes(); ++i)
    {   SLTransformKeyframe* kf = (SLTransformKeyframe*)track->keyframe(i);
        points[i] =kf->translation();
        times[i] = kf->time();
    }

    // create curve and delete temp arrays again
    track->interpolationCurve(new SLCurveBezier(points, times, (SLint)track->numKeyframes(), controls));
    track->translationInterpolation(AI_Bezier);

    delete[] points;
    delete[] times;


    return track;
}
//-----------------------------------------------------------------------------
