//#############################################################################
//  File:      SLAnimation.cpp
//  Author:    Marc Wacker
//  Date:      Autumn 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLScene.h>
#include <SLAnimation.h>
#include <SLAnimationManager.h>
#include <SLSkeleton.h>
#include <SLCurveBezier.h>

//-----------------------------------------------------------------------------
/*! Constructor
*/
SLAnimation::SLAnimation(const SLstring& name, SLfloat duration)
            : _name(name), _length(duration)
{ 
}

//-----------------------------------------------------------------------------
/*! Destructor
*/
SLAnimation::~SLAnimation()
{
    SLMNodeAnimationTrack::iterator it = _nodeAnimations.begin();
    for (; it != _nodeAnimations.end(); it++)
        delete it->second;
}

//-----------------------------------------------------------------------------
/*! Setter for the animation length
*/
void SLAnimation::length(SLfloat length)
{
    // @todo notify the animations track to optimize their keyframes
    _length = length;
}

//-----------------------------------------------------------------------------
/*! Returns the timestamp for the next keyframe in all of the tracks.
*/
SLfloat SLAnimation::nextKeyframeTime(SLfloat time)
{
    // find the closest keyframe time to the right
    SLfloat result = _length;
    SLKeyframe* kf1;
    SLKeyframe* kf2;
    
    SLMNodeAnimationTrack::iterator it = _nodeAnimations.begin();
    for (; it != _nodeAnimations.end(); it++)
    {
        it->second->getKeyframesAtTime(time, &kf1, &kf2);
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

    SLMNodeAnimationTrack::iterator it = _nodeAnimations.begin();
    for (; it != _nodeAnimations.end(); it++)
    {
        it->second->getKeyframesAtTime(time, &kf1, &kf2);
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
    map<SLuint, SLNodeAnimationTrack*>::iterator it = _nodeAnimations.begin();
    for (; it != _nodeAnimations.end(); ++it)
    {
        if (it->second->animationTarget() == node)
            return true;
    }

    return false;
}

//-----------------------------------------------------------------------------
/*! Creates a new SLNodeAnimationTrack with the next free handle.
*/
SLNodeAnimationTrack* SLAnimation::createNodeAnimationTrack()
{
    SLuint freeIndex = 0;
    
    map<SLuint, SLNodeAnimationTrack*>::iterator it = _nodeAnimations.begin();
    for (; it != _nodeAnimations.end() && freeIndex == it->first; ++it, ++freeIndex)
    { }

    return createNodeAnimationTrack(freeIndex);
}

//-----------------------------------------------------------------------------
/*! Creates a new SLNodeAnimationTrack with the passed in handle.
*/
SLNodeAnimationTrack* SLAnimation::createNodeAnimationTrack(SLuint handle)
{
    // track with same handle already exists
    // @todo provide a function that generates the handle automatically
    if (_nodeAnimations.find(handle) != _nodeAnimations.end())
        return NULL;

    _nodeAnimations[handle] = new SLNodeAnimationTrack(this, handle);

    return _nodeAnimations[handle];
}

//-----------------------------------------------------------------------------
/*! Applies all animation tracks for the passed in timestamp, weight and scale.
*/
void SLAnimation::apply(SLfloat time, SLfloat weight , SLfloat scale)
{
    SLMNodeAnimationTrack::iterator it = _nodeAnimations.begin();
    for (; it != _nodeAnimations.end(); it++)
        it->second->apply(time, weight, scale);
}

//-----------------------------------------------------------------------------
/*! Applies all node tracks of this animation on a single node
*/
void SLAnimation::applyToNode(SLNode* node, SLfloat time, SLfloat weight, SLfloat scale)
{
    SLMNodeAnimationTrack::iterator it = _nodeAnimations.begin();
    for (; it != _nodeAnimations.end(); it++)
        it->second->applyToNode(node, time, weight, scale);
}

//-----------------------------------------------------------------------------
/*! Applies all the tracks to their respective joints in the passed in skeleton.
*/
void SLAnimation::apply(SLSkeleton* skel, SLfloat time, SLfloat weight, SLfloat scale)
{
    SLMNodeAnimationTrack::iterator it = _nodeAnimations.begin();
    for (; it != _nodeAnimations.end(); it++)
    {
        SLJoint* joint = skel->getJoint(it->first);
        it->second->applyToNode(joint, time, weight, scale);
    }

}

//-----------------------------------------------------------------------------
/*! Resets all default animation targets to their initial state.
*/
void SLAnimation::resetNodes()
{
    SLMNodeAnimationTrack::iterator it = _nodeAnimations.begin();
    for (; it != _nodeAnimations.end(); it++)
        it->second->animationTarget()->resetToInitialState();
}

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/*! Creates new SLAnimation istance for node animations. It will already create and set parameters
for the respective SLAnimationState.
*/
SLAnimation* SLAnimation::createAnimation(const SLstring& name,
                                          SLfloat duration,
                                          SLbool enabled,
                                          SLEasingCurve easing,
                                          SLAnimLooping looping)
{
    SLAnimation* anim = SLScene::current->animManager().createNodeAnimation(name, duration);
    SLAnimationState* state = SLScene::current->animManager().getNodeAnimationState(anim->name());
    state->enabled(enabled);
    state->easing(easing);
    state->loop(looping);
    return anim;
}

//-----------------------------------------------------------------------------
/*! Specialized SLNodeAnimationTrack creator for a two keyframe translation animation
*/
SLNodeAnimationTrack* SLAnimation::createSimpleTranslationNodeTrack(SLNode* target,
                                                                    const SLVec3f& endPos)
{
    SLNodeAnimationTrack* track = createNodeAnimationTrack();
    target->setInitialState();
    track->animationTarget(target);
    track->createNodeKeyframe(0.0f); // create zero kf
    track->createNodeKeyframe(length())->translation(endPos); // create end scale keyframe
    return track;
}

//-----------------------------------------------------------------------------
/*! Specialized SLNodeAnimationTrack creator for a two keyframe rotation animation
*/
SLNodeAnimationTrack* SLAnimation::createSimpleRotationNodeTrack(SLNode* target,
                                                                 SLfloat angleDeg,
                                                                 const SLVec3f& axis)
{
    SLNodeAnimationTrack* track = createNodeAnimationTrack();
    target->setInitialState();
    track->animationTarget(target);
    track->createNodeKeyframe(0.0f); // create zero kf
    track->createNodeKeyframe(length())->rotation(SLQuat4f(angleDeg, axis)); // create end scale keyframe
    return track;
}
  
//-----------------------------------------------------------------------------
/*! Specialized SLNodeAnimationTrack creator for a two keyframe scaling animation
*/
SLNodeAnimationTrack* SLAnimation::createSimpleScalingNodeTrack(SLNode* target,
                                                                const SLVec3f& endScale)
{    
    SLNodeAnimationTrack* track = createNodeAnimationTrack();
    target->setInitialState();
    track->animationTarget(target);
    track->createNodeKeyframe(0.0f); // create zero kf
    track->createNodeKeyframe(length())->scale(endScale); // create end scale keyframe
    return track;
}
  
//-----------------------------------------------------------------------------
/*! Specialized SLNodeAnimationTrack creator for an elliptic node animation
*/
SLNodeAnimationTrack* SLAnimation::createEllipticNodeTrack(SLNode* target, 
                                                           SLfloat radiusA, SLAxis axisA,
                                                           SLfloat radiusB, SLAxis axisB)
{
    assert(axisA!=axisB && radiusA>0 && radiusB>0);
    SLNodeAnimationTrack* track = createNodeAnimationTrack();
    target->setInitialState();
    track->animationTarget(target);

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
    SLfloat t4 = length() / 4.0f;
    track->createNodeKeyframe(0.0f * t4)->translation(A);
    track->createNodeKeyframe(1.0f * t4)->translation(B);
    track->createNodeKeyframe(2.0f * t4)->translation(C);
    track->createNodeKeyframe(3.0f * t4)->translation(D);
    track->createNodeKeyframe(4.0f * t4)->translation(A);


    // Build curve data w. cummulated times
    SLVec3f* points = new SLVec3f[track->numKeyframes()];
    SLfloat* times  = new SLfloat[track->numKeyframes()];
    for (SLuint i=0; i<track->numKeyframes(); ++i)
    {   
        SLTransformKeyframe* kf = (SLTransformKeyframe*)track->keyframe(i);
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
