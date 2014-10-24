
#ifndef TEADSADASD
#define TEADSADASD

#include <stdafx.h>
#include "SLNode.h"

//
///** Interface for important node functions that need to be added with the update 
//@todo  add variations of the functions below setter for position that takes x, y, z floats etc... */
//class INewNode
//{
//public:
//
//    //// SETTERS
//    /** Set the position of this node.
//    @param  pos         new position the object should be in.
//    @param  space       the space we want to set the position in. */
//    virtual void position(const SLVec3f& pos, SLTransformSpace space = TS_Parent) = 0;
//
//    /** set the node's rotation.
//    @param  rot         the rotation to set the node's rotation to.
//    @param  space       relative space to apply the rotation in. */
//    virtual void rotation(const SLQuat4f& rot, SLTransformSpace space = TS_Parent) = 0;
//
//    /** set the node's scale.
//    @param  scale       the amount to scale the object by.
//    @param  space       relative space to apply the scale in. */
//    virtual void scale(const SLVec3f& scale, SLTransformSpace space = TS_Parent) = 0;
//
//
//
//
//    //// GETTERS
//    
//    /** get position 
//    @param  space       the relative space we want the return value to be in */
//    virtual const SLVec3f& position(SLTransformSpace space = TS_Parent) const = 0;
//    
//    /** get rotation
//    @param  space       the relative space we want the return value to be in. */
//    virtual const SLQuat4f& rotation(SLTransformSpace space = TS_Parent) const = 0;
//
//    /** get scale 
//    @param  space       the relative space we want the return value to be in. */
//    virtual const SLVec3f& scale(SLTransformSpace space = TS_Parent) const = 0;
//    
//    /** get the current right direction of the node (-Z axis)
//    @param  space       the relative space we want the return value to be in. */
//    virtual SLVec3f forward(SLTransformSpace space = TS_Parent) const = 0;
//
//    /** get the current right direction of the node (Y axis)
//    @param  space       the relative space we want the return value to be in. */
//    virtual SLVec3f up(SLTransformSpace space = TS_Parent) const = 0;
//
//    /** get the current right direction of the node (X axis) 
//    @param  space       the relative space we want the return value to be in. */
//    virtual SLVec3f right(SLTransformSpace space = TS_Parent) const = 0;
//
//    /** get the current world matrix for the node. */
//    virtual const SLMat4f& worldMat() const = 0;
//
//    /** get the current local matrix for the node. */
//    virtual const SLMat4f& localMat() const = 0;
//
//
//
//
//    //// MOVEMENT FUNCTIONS
//    /** Move this node.
//    @param  dist        the distance this node should move.
//    @param  space       the space we want to set the position in. */
//    virtual void move(const SLVec3f& dist, SLTransformSpace space = TS_Parent) = 0;
//
//    /** Move this node relative to its current orientation
//    @param  dist        the distace this object should move. */
//    virtual void moveRel(const SLVec3f& dist) = 0;
//
//    /** rotate the object around \up so that it faces the position passed in.
//    @param  point       position the object should look at.
//    @param  up          up axis the object should orient itself around.
//    @param  space       the space in which \point and \up are interpreted in. */
//    virtual void lookAtPos(const SLVec3f& point, const SLVec3f& up = SLVec3f::AXISY, SLTransformSpace space = TS_Parent) = 0;
//    
//    /** rotate the object around \up so that it faces in the set direction.
//    @param  point       position the object should look at.
//    @param  up          up axis the object should orient itself around.
//    @param  space       the space in which \point and \up are interpreted in. */
//    virtual void lookAtDir(const SLVec3f& point, const SLVec3f& up = SLVec3f::AXISY, SLTransformSpace space = TS_Parent) = 0;
//    
//    /** rotate the object around an axis.
//    @param  angleDeg    angle in degree.
//    @param  axis        axis to rotate around.
//    @param  space       the relative space we want to apply our rotation in. */
//    virtual void rotate(const SLfloat angleDeg, const SLVec3f& axis, SLTransformSpace space = TS_Parent) = 0;
//
//    /** rotate the object around an axis
//    @param  rot         the rotation to rotate this object by.
//    @param  space       the relative space we want to apply our rotation in. */
//    virtual void rotate(const SLQuat4f& rot, SLTransformSpace space = TS_Parent) = 0;
//    
//    /** rotate around the forward axis (roll)
//    @param  angleDeg     roll angle in degrees. */
//    virtual void roll(const SLfloat angleDeg) = 0;
//
//    /** rotate around the up axis (yaw)
//    @param  angleDeg     yaw angle in degrees. */
//    virtual void yaw(const SLfloat angleDeg) = 0;
//
//    /** rotate around the right axis (pitch)
//    @param  angleDeg     pitch angle in degrees. */
//    virtual void pitch(const SLfloat angleDeg) = 0;
//};


/** A completely reworked SLNode class (still inheriting SLNode to test it in the normal scene graph) */
class NewNodeTRS : public SLNode
{
public:
	NewNodeTRS(const SLstring& name = "NodeTRS")
        : SLNode(name), _position(0, 0, 0), _scale(1, 1, 1)
	{     
    }

    // SLNode functions that are no longer required OR that need a slight tweak
    void update() {}
    void findDirtyRec() {}
    void updateRec() {}
    const SLMat4f& wm();
    bool animateRec(SLfloat timeMS);
    SLAABBox& updateAABBRec();
    void markDirty();


    // new functions
    void updateIfDirty();
    // marks a node's parent dirty. if that node wants to get its world transform
    // it has to recalculate it. the local transform however is still cached.
    void markWorldMatDirty();

    //// implementation of the INewNode interface
    //virtual void position(const SLVec3f& pos, SLTransformSpace space = TS_Parent);
    //virtual void rotation(const SLQuat4f& rot, SLTransformSpace space = TS_Parent);
    //virtual void scale(const SLVec3f& scale, SLTransformSpace space = TS_Parent);

    //virtual const SLVec3f& position(SLTransformSpace space = TS_Parent) const;
    //virtual const SLQuat4f& rotation(SLTransformSpace space = TS_Parent) const;
    //virtual const SLVec3f& scale(SLTransformSpace space = TS_Parent) const;
    //virtual SLVec3f forward(SLTransformSpace space = TS_Parent) const;
    //virtual SLVec3f up(SLTransformSpace space = TS_Parent) const;
    //virtual SLVec3f right(SLTransformSpace space = TS_Parent) const;
    //virtual const SLMat4f& worldMat() const;
    //virtual const SLMat4f& localMat() const;

    //virtual void move(const SLVec3f& dist, SLTransformSpace space = TS_Parent);
    //virtual void moveRel(const SLVec3f& dist);
    //virtual void lookAtPos(const SLVec3f& point, const SLVec3f& up = SLVec3f::AXISY, SLTransformSpace space = TS_Parent);
    //virtual void lookAtDir(const SLVec3f& point, const SLVec3f& up = SLVec3f::AXISY, SLTransformSpace space = TS_Parent);
    //virtual void rotate(const SLfloat angleDeg, const SLVec3f& axis, SLTransformSpace space = TS_Parent);
    //virtual void rotate(const SLQuat4f& rot, SLTransformSpace space = TS_Parent);
    //virtual void roll(const SLfloat angleDeg);
    //virtual void yaw(const SLfloat angleDeg);
    //virtual void pitch(const SLfloat angleDeg);
    
private:
    // I opted to use a new member for the local matrix to demonstrate that it 
    // is more clear this way.
    mutable SLMat4f     _localMat;
	mutable SLbool      _isLocalMatOutOfDate;  // TODO: give this a better name maybe? isLocalUpToDate, isLocalDirty...
    mutable SLMat4f     _worldMat; // we could just use _wm but for now we use this new member just to be save 
	mutable SLbool      _isWorldMatOutOfDate;

	SLVec3f     _position;
	SLVec3f     _scale;
	SLQuat4f    _rotation;

    // helper functions to update the world and local matrix
    // TODO: the functions below change the values of _worldMat and _localMat
    //       to hold the values in _position, _scale and _rotation.
    //       so technically the functions below change a member and are by definition not const...
    //       however, we need the getters for local and worldMat to be const in case we pass around const references
    //       what do?
    void updateLocalMat() const;
    void updateWorldMat() const;
};


#endif