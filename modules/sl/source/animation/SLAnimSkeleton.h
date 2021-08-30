//#############################################################################
//  File:      SLAnimSkeleton.h
//  Date:      Autumn 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Authors:   Marc Wacker, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLSKELETON_H
#define SLSKELETON_H

#include <SLAnimPlayback.h>
#include <SLAnimation.h>
#include <SLJoint.h>

class SLAnimManager;
class SLSceneView;

//-----------------------------------------------------------------------------
//! SLAnimSkeleton keeps track of a skeletons joints and animations
/*!
An SLAnimSkeleton is used to animate a hierarchical object like a human being.
An SLAnimSkeleton keeps track of its bones (SLJoints) in a tree structure and
points with _root to the root node of the skeleton hierarchy.
An SLAnimSkeleton is not actively transforming any SLNode in the scenegraph.
It just keeps track of its transformed SLJoint.
A mesh that is associated with a skeleton transforms all its vertices every
frame by the joint weights. Every vertex of a mesh has weights for four joints
by which it can be influenced.

SLAnimations for this skeleton are also kept in this class. The SLAnimations
have tracks corresponding to the individual SLJoints in the skeleton.

@note   The current implementation doesn't support multiple instances of the same
        skeleton animation. It is however not that far away from supporting it.
        Currently the SLAnimSkeleton class keeps both a SLAnimation map
        and an SLAnimPlayback map. We can split this into two classes
        by creating an SLSkeletonInstance class we that keeps the
        SLAnimPlayback map that references its parent SLAnimSkeleton
        we would be able to create multiple SLSkeletonInstance instances
        that use the same SLAnimations but with different states.

        This leaves the problem of SLMesh that is not able to be instantiated
        without copying the data into a completely separate SLMesh. But the
        solution for SLMesh would take the same approach by creating a
        mesh instance class that is able to use SLSkeletonInstance.

@note   The current version of the SLAssimpImporter only supports the loading of a single animation.
        This limitation is mainly because there are very few 3D programs
        that make use of the possibility to export multiple animations in
        a single file. This means we would need to extend our importer to
        be able to load more animations for an already loaded skeleton.
*/
class SLAnimSkeleton
{
public:
    SLAnimSkeleton();
    ~SLAnimSkeleton();

    SLJoint*     createJoint(SLuint id);
    SLJoint*     createJoint(const SLstring& name, SLuint id);
    SLAnimation* createAnimation(SLAnimManager& aniMan, const SLstring& name, SLfloat duration);

    void loadAnimation(const SLstring& file);
    void addAnimation(SLAnimation* anim);
    void getJointMatrices(SLVMat4f& jointWM);
    void reset();

    // Getters
    SLAnimPlayback* animPlayback(const SLstring& name);
    SLMAnimation    animations() { return _animations; }
    SLint           numAnimations() const { return (SLint)_animations.size(); }
    SLJoint*        getJoint(SLuint id);
    SLJoint*        getJoint(const SLstring& name);
    SLint           numJoints() const { return (SLint)_joints.size(); }
    const SLVJoint& joints() const { return _joints; }
    SLJoint*        rootJoint() { return _rootJoint; }
    SLbool          changed() const { return _changed; }
    const SLVec3f&  minOS();
    const SLVec3f&  maxOS();

    // Setters
    void rootJoint(SLJoint* joint) { _rootJoint = joint; }
    void changed(SLbool changed)
    {
        _changed         = changed;
        _minMaxOutOfDate = true;
    }

    SLbool updateAnimations(SLfloat elapsedTimeSec);

protected:
    void updateMinMax();

    SLJoint*        _rootJoint;       //!< pointer to the root joint of skeleton
    SLVJoint        _joints;          //!< joint vector for fast access and index to joint mapping
    SLMAnimation    _animations;      //!< map of animations for this skeleton
    SLMAnimPlayback _animPlaybacks;   //!< map of animation playbacks for this skeleton
    SLbool          _changed;         //!< did this skeleton change this frame (attribute for skeleton instance)
    SLVec3f         _minOS;           //!< min point in os for this skeleton (attribute for skeleton instance)
    SLVec3f         _maxOS;           //!< max point in os for this skeleton (attribute for skeleton instance)
    SLbool          _minMaxOutOfDate; //!< dirty flag aabb rebuild
};
//-----------------------------------------------------------------------------
typedef vector<SLAnimSkeleton*> SLVSkeleton;
//-----------------------------------------------------------------------------

#endif
