//#############################################################################
//   File:      SLAnimation.cpp
//   Date:      Autumn 2014
//   Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//   Authors:   Marc Wacker, Marcus Hudritsch
//   License:   This software is provided under the GNU General Public License
//              Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <math/SLCurveBezier.h>
#include <SLScene.h>

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
    SLfloat         result = _lengthSec;
    SLAnimKeyframe* kf1;
    SLAnimKeyframe* kf2;

    for (auto it : _nodeAnimTracks)
    {
        it.second->getKeyframesAtTime(time, &kf1, &kf2);
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
    SLfloat         result = 0.0;
    SLAnimKeyframe* kf1;
    SLAnimKeyframe* kf2;

    // shift the time a little bit to the left or else the getKeyframesAtTime function
    // would return the same keyframe over and over again
    time -= 0.01f;
    if (time <= 0.0f)
        return 0.0f;

    for (auto it : _nodeAnimTracks)
    {
        it.second->getKeyframesAtTime(time, &kf1, &kf2);
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
SLNodeAnimTrack* SLAnimation::createNodeAnimTrack()
{
    SLuint freeIndex = 0;

    auto it = _nodeAnimTracks.begin();
    for (; it != _nodeAnimTracks.end() && freeIndex == it->first; ++it, ++freeIndex)
    {
    }

    return createNodeAnimTrack(freeIndex);
}
//-----------------------------------------------------------------------------
/*! Creates a new SLNodeAnimationTrack with the passed in track id.
 */
SLNodeAnimTrack* SLAnimation::createNodeAnimTrack(SLuint trackID)
{
    // track with same handle already exists
    if (_nodeAnimTracks.find(trackID) != _nodeAnimTracks.end())
        return nullptr;

    _nodeAnimTracks[trackID] = new SLNodeAnimTrack(this);

    return _nodeAnimTracks[trackID];
}
//-----------------------------------------------------------------------------
/*! Applies all animation tracks for the passed in timestamp, weight and scale.
 */
void SLAnimation::apply(SLfloat time, SLfloat weight, SLfloat scale)
{
    for (auto it : _nodeAnimTracks)
        it.second->apply(time, weight, scale);
}
//-----------------------------------------------------------------------------
/*! Applies all node tracks of this animation on a single node
 */
void SLAnimation::applyToNode(SLNode* node,
                              SLfloat time,
                              SLfloat weight,
                              SLfloat scale)
{
    for (auto it : _nodeAnimTracks)
        it.second->applyToNode(node, time, weight, scale);
}
//-----------------------------------------------------------------------------
/*! Applies all the tracks to their respective joints in the passed in skeleton.
 */
void SLAnimation::apply(SLAnimSkeleton* skel,
                        SLfloat         time,
                        SLfloat         weight,
                        SLfloat         scale)
{
    for (auto it : _nodeAnimTracks)
    {
        SLJoint* joint = skel->getJoint(it.first);
        it.second->applyToNode(joint, time, weight, scale);
    }
}
//-----------------------------------------------------------------------------
/*! Draws the visualizations of all node tracks
 */
void SLAnimation::drawNodeVisuals(SLSceneView* sv)
{
    for (auto it : _nodeAnimTracks)
        it.second->drawVisuals(sv);
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
/*! Specialized SLNodeAnimationTrack creator for a two keyframe translation animation
 */
SLNodeAnimTrack* SLAnimation::createNodeAnimTrackForTranslation(SLNode*        target,
                                                                const SLVec3f& endPos)
{
    SLNodeAnimTrack* track = createNodeAnimTrack();
    target->setInitialState();
    track->animatedNode(target);
    track->createNodeKeyframe(0.0f);
    track->createNodeKeyframe(lengthSec())->translation(endPos);
    return track;
}
//-----------------------------------------------------------------------------
/*! Specialized SLNodeAnimationTrack creator for a two keyframe rotation
 * animation from 0° to angleDeg.
 */
SLNodeAnimTrack* SLAnimation::createNodeAnimTrackForRotation(SLNode*        target,
                                                             SLfloat        angleDeg,
                                                             const SLVec3f& axis)
{
    SLNodeAnimTrack* track = createNodeAnimTrack();
    target->setInitialState();
    track->animatedNode(target);
    track->createNodeKeyframe(0.0f);
    track->createNodeKeyframe(lengthSec())->rotation(SLQuat4f(angleDeg, axis));
    return track;
}
//-----------------------------------------------------------------------------
/*! Specialized SLNodeAnimationTrack creator for 2 keyframes at angleDeg0 and
 * angleDeg1.
 */
SLNodeAnimTrack* SLAnimation::createNodeAnimTrackForRotation2(SLNode*        target,
                                                              SLfloat        angleDeg0,
                                                              SLfloat        angleDeg1,
                                                              const SLVec3f& axis)
{
    SLNodeAnimTrack* track = createNodeAnimTrack();
    target->setInitialState();
    track->animatedNode(target);

    SLTransformKeyframe* frame0 = track->createNodeKeyframe(0.0f);
    frame0->rotation(SLQuat4f(angleDeg0, axis));

    SLTransformKeyframe* frame1 = track->createNodeKeyframe(lengthSec());
    frame1->rotation(SLQuat4f(angleDeg1, axis));

    return track;
}
//-----------------------------------------------------------------------------
/*! Specialized SLNodeAnimationTrack creator for 3 keyframes at angleDeg0,
 * angleDeg1 and angleDeg2.
 */
SLNodeAnimTrack* SLAnimation::createNodeAnimTrackForRotation3(SLNode*        target,
                                                              SLfloat        angleDeg0,
                                                              SLfloat        angleDeg1,
                                                              SLfloat        angleDeg2,
                                                              const SLVec3f& axis)
{
    SLNodeAnimTrack* track = createNodeAnimTrack();
    target->setInitialState();
    track->animatedNode(target);

    SLTransformKeyframe* frame0 = track->createNodeKeyframe(0.0f);
    frame0->rotation(SLQuat4f(angleDeg0, axis));

    SLTransformKeyframe* frame1 = track->createNodeKeyframe(lengthSec() * 0.5f);
    frame1->rotation(SLQuat4f(angleDeg1, axis));

    SLTransformKeyframe* frame2 = track->createNodeKeyframe(lengthSec());
    frame2->rotation(SLQuat4f(angleDeg2, axis));

    return track;
}
//-----------------------------------------------------------------------------
/*! Specialized SLNodeAnimationTrack creator for 4 keyframes at angleDeg0,
 * angleDeg1, angleDeg2 and angleDeg3.
 */
SLNodeAnimTrack* SLAnimation::createNodeAnimTrackForRotation4(SLNode*        target,
                                                              SLfloat        angleDeg0,
                                                              SLfloat        angleDeg1,
                                                              SLfloat        angleDeg2,
                                                              SLfloat        angleDeg3,
                                                              const SLVec3f& axis)
{
    SLNodeAnimTrack* track = createNodeAnimTrack();
    target->setInitialState();
    track->animatedNode(target);

    SLTransformKeyframe* frame0 = track->createNodeKeyframe(0.0f);
    frame0->rotation(SLQuat4f(angleDeg0, axis));

    SLTransformKeyframe* frame1 = track->createNodeKeyframe(lengthSec() * 0.3333f);
    frame1->rotation(SLQuat4f(angleDeg1, axis));

    SLTransformKeyframe* frame2 = track->createNodeKeyframe(lengthSec() * 0.6666f);
    frame2->rotation(SLQuat4f(angleDeg2, axis));

    SLTransformKeyframe* frame3 = track->createNodeKeyframe(lengthSec());
    frame3->rotation(SLQuat4f(angleDeg3, axis));

    return track;
}
//-----------------------------------------------------------------------------
/*! Specialized SLNodeAnimationTrack creator for a 360 deg. node rotation track
 * with 3 keyframes from 0° to 180° to 360°.
 */
SLNodeAnimTrack* SLAnimation::createNodeAnimTrackForRotation360(SLNode*        target,
                                                                const SLVec3f& axis)
{
    SLNodeAnimTrack* track = createNodeAnimTrack();
    target->setInitialState();
    track->animatedNode(target);

    SLTransformKeyframe* frame0 = track->createNodeKeyframe(0.0f);
    frame0->rotation(SLQuat4f(0.0f, axis));

    SLTransformKeyframe* frame1 = track->createNodeKeyframe(lengthSec() * 0.5f);
    frame1->rotation(SLQuat4f(180.0f, axis));

    SLTransformKeyframe* frame2 = track->createNodeKeyframe(lengthSec());
    frame2->rotation(SLQuat4f(360.0f, axis));

    return track;
}
//-----------------------------------------------------------------------------
/*! Specialized SLNodeAnimationTrack creator for a two keyframe scaling animation
 */
SLNodeAnimTrack* SLAnimation::createNodeAnimTrackForScaling(SLNode*        target,
                                                            const SLVec3f& endScale)
{
    SLNodeAnimTrack* track = createNodeAnimTrack();
    target->setInitialState();
    track->animatedNode(target);
    track->createNodeKeyframe(0.0f);
    track->createNodeKeyframe(lengthSec())->scale(endScale);
    return track;
}

//-----------------------------------------------------------------------------
/*! Specialized SLNodeAnimationTrack creator for an elliptic node animation
 */
SLNodeAnimTrack* SLAnimation::createNodeAnimTrackForEllipse(SLNode* target,
                                                            SLfloat radiusA,
                                                            SLAxis  axisA,
                                                            SLfloat radiusB,
                                                            SLAxis  axisB)
{
    assert(axisA != axisB && radiusA > 0 && radiusB > 0);
    SLNodeAnimTrack* track = createNodeAnimTrack();
    target->setInitialState();
    track->animatedNode(target);

    /* The ellipse is defined by 5 keyframes: A,B,C,D and again A

        c2----B----c1
    c3                 c0
    |                   |
    |         |         |
    C       --0--       A
    |         |         |
    |                   |
    c4                 c7
        c5----D----c6
    */

    SLVec3f A(0, 0, 0);
    A.comp[axisA] = radiusA;
    SLVec3f B(0, 0, 0);
    B.comp[axisB] = radiusB;
    SLVec3f C(0, 0, 0);
    C.comp[axisA] = -radiusA;
    SLVec3f D(0, 0, 0);
    D.comp[axisB] = -radiusB;

    // Control points with the magic factor kappa for control points
    SLfloat k = 0.5522847498f;

    SLVVec3f controls;
    controls.resize(8);
    for (SLuint i = 0; i < controls.size(); ++i)
        controls[i].set(0, 0, 0);
    controls[0].comp[axisA] = radiusA;
    controls[0].comp[axisB] = k * radiusB;
    controls[1].comp[axisB] = radiusB;
    controls[1].comp[axisA] = k * radiusA;
    controls[2].comp[axisB] = radiusB;
    controls[2].comp[axisA] = k * -radiusA;
    controls[3].comp[axisA] = -radiusA;
    controls[3].comp[axisB] = k * radiusB;
    controls[4].comp[axisA] = -radiusA;
    controls[4].comp[axisB] = k * -radiusB;
    controls[5].comp[axisB] = -radiusB;
    controls[5].comp[axisA] = k * -radiusA;
    controls[6].comp[axisB] = -radiusB;
    controls[6].comp[axisA] = k * radiusA;
    controls[7].comp[axisA] = radiusA;
    controls[7].comp[axisB] = k * -radiusB;

    // Add keyframes
    SLfloat t4 = lengthSec() / 4.0f;
    track->createNodeKeyframe(0.0f * t4)->translation(A);
    track->createNodeKeyframe(1.0f * t4)->translation(B);
    track->createNodeKeyframe(2.0f * t4)->translation(C);
    track->createNodeKeyframe(3.0f * t4)->translation(D);
    track->createNodeKeyframe(4.0f * t4)->translation(A);

    // Build curve data w. cumulated times
    SLVVec4f points;
    points.resize((SLulong)track->numKeyframes());
    for (SLuint i = 0; i < (SLuint)track->numKeyframes(); ++i)
    {
        SLTransformKeyframe* kf = (SLTransformKeyframe*)track->keyframe((SLint)i);
        points[i].set(kf->translation().x,
                      kf->translation().y,
                      kf->translation().z,
                      kf->time());
    }

    // create curve and delete temp arrays again
    track->interpolationCurve(new SLCurveBezier(points, controls));
    track->translationInterpolation(AI_bezier);

    return track;
}
//-----------------------------------------------------------------------------
