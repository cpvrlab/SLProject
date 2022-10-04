//#############################################################################
//  File:      SLAnimation.h
//  Date:      Autumn 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marc Wacker, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLANIMATION_H
#define SLANIMATION_H

#include <SLAnimTrack.h>
#include <SLEnums.h>
#include <SLJoint.h>

class SLAnimSkeleton;

//-----------------------------------------------------------------------------
//! SLAnimation is the base container for all animation data.
/*!
SLAnimation is a container for multiple SLAnimTrack that build an animation.
E.g. a walk animation would consist of all the SLAnimTrack that make a
SLAnimSkeleton walk. It also knows the length of the animation.

An animation for a SLAnimSkeleton with n joints must consist of 1 to n
SLNodeAnimTrack. The SLAnimation class keeps a map with index -> SLNodeAnimTrack
pairs, the index for the SLNodeAnimTrack must match the index of a bone in the
target SLAnimSkeleton. This method allows us to animate multiple identical, or similar
SLSkeletons with the same SLAnimation.
*/
class SLAnimation
{
public:
    SLAnimation(const SLstring& name, SLfloat duration);
    ~SLAnimation();

    SLfloat nextKeyframeTime(SLfloat time);
    SLfloat prevKeyframeTime(SLfloat time);
    SLbool  affectsNode(SLNode* node);
    void    apply(SLfloat time,
                  SLfloat weight = 1.0f,
                  SLfloat scale  = 1.0f);
    void    applyToNode(SLNode* node,
                        SLfloat time,
                        SLfloat weight = 1.0f,
                        SLfloat scale  = 1.0f);
    void    apply(SLAnimSkeleton* skel,
                  SLfloat         time,
                  SLfloat         weight = 1.0f,
                  SLfloat         scale  = 1.0f);
    void    resetNodes();
    void    drawNodeVisuals(SLSceneView* sv);

    // track creators
    SLNodeAnimTrack* createNodeAnimTrack();
    SLNodeAnimTrack* createNodeAnimTrack(SLuint trackID);
    SLNodeAnimTrack* createNodeAnimTrackForTranslation(SLNode* target, const SLVec3f& endPos);
    SLNodeAnimTrack* createNodeAnimTrackForRotation(SLNode* target, SLfloat angleDeg1, const SLVec3f& axis);
    SLNodeAnimTrack* createNodeAnimTrackForRotation2(SLNode* target, SLfloat angleDeg0, SLfloat angleDeg1, const SLVec3f& axis);
    SLNodeAnimTrack* createNodeAnimTrackForRotation3(SLNode* target, SLfloat angleDeg0, SLfloat angleDeg1, SLfloat angleDeg2, const SLVec3f& axis);
    SLNodeAnimTrack* createNodeAnimTrackForRotation4(SLNode* target, SLfloat angleDeg0, SLfloat angleDeg1, SLfloat angleDeg2, SLfloat angleDeg3, const SLVec3f& axis);
    SLNodeAnimTrack* createNodeAnimTrackForRotation360(SLNode* target, const SLVec3f& axis);
    SLNodeAnimTrack* createNodeAnimTrackForScaling(SLNode* target, const SLVec3f& endScale);
    SLNodeAnimTrack* createNodeAnimTrackForEllipse(SLNode* target,
                                                   SLfloat radiusA,
                                                   SLAxis  axisA,
                                                   SLfloat radiusB,
                                                   SLAxis  axisB);
    // Getters
    const SLstring& name() { return _name; }
    SLfloat         lengthSec() const { return _lengthSec; }

    // Setters
    void name(const SLstring& name) { _name = name; }
    void lengthSec(SLfloat lengthSec);

protected:
    SLstring         _name;           //!< name of the animation
    SLfloat          _lengthSec;      //!< duration of the animation in seconds
    SLMNodeAnimTrack _nodeAnimTracks; //!< map of all the node tracks in this animation
};
//-----------------------------------------------------------------------------
typedef vector<SLAnimation*>             SLVAnimation;
typedef std::map<SLstring, SLAnimation*> SLMAnimation;
//-----------------------------------------------------------------------------
#endif
