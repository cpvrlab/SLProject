
#include <stdafx.h>
#include "NewNodeTRS.h"
#include "SLBox.h"
#include "SLGLTexture.h"
#include "SLMaterial.h"
#include "SLAnimation.h"

// Temporary helper function here:
// we need a way to efficiently set a matrix
// using a translation, rotation and scale
// PROBLEM: all current math classes are templates and header only
//          Mat4 would need to include Quat4, but Quat4 also needs to inlcude Mat4
//          I don't believe there is an easy soluton to add a setMatrix(const SLVec3& trans, const SLQuat4& rot, const SLVec3& scale)
//          function in SLMat4.h

void setMatrixTRS(SLMat4f& mat, const SLVec3f& translation, const SLQuat4f rotation, const SLVec3f& scale)
{
    SLMat3f rot3x3 = rotation.toMat3();
    
        // found something while doing this, Mat3::_m is public but Mat4::_m is not. maybe add operator[] for access?, why make m private then though?..   
    mat.setMatrix(
        scale.x * rot3x3._m[0], scale.y * rot3x3[3], scale.z * rot3x3[6], translation.x,
        scale.x * rot3x3._m[1], scale.y * rot3x3[4], scale.z * rot3x3[7], translation.y,
        scale.x * rot3x3._m[2], scale.y * rot3x3[5], scale.z * rot3x3[8], translation.z,
        0,                      0,                   0,                   1
        );
}

const SLMat4f& NewNodeTRS::wm()
{
   return _wm;
}

SLbool NewNodeTRS::animateRec(SLfloat timeMS)
{
   SLbool gotAnimated = false;

   if (!_drawBits.get(SL_DB_ANIMOFF))
   {  
      if (_animation && !_animation->isFinished()) 
      {  _animation->animate(this, timeMS);
         gotAnimated = true;
         // added line here
         markDirty(); // marking ourselfes and all children dirty
      }

      // animate children nodes for groups or group derived classes
      for (SLint i=0; i<_children.size(); ++i)
         if (_children[i]->animateRec(timeMS)) gotAnimated = true;
   }
   return gotAnimated;
}

SLAABBox& NewNodeTRS::updateAABBRec()
{
    // update the wm and wmI in this current node since updateAABBRec needs them
    updateIfDirty();
    return SLNode::updateAABBRec();
}

void NewNodeTRS::markDirty()
{
   _isLocalMatOutOfDate = true;

   // if our local mat is out of date then
   // our world mat is out of date too
   // + the world mat for all children is out of date
   markWorldMatDirty();
}

void NewNodeTRS::markWorldMatDirty()
{
    // only mark branches that aren't already dirty
    if(_isWorldMatOutOfDate)
        return;

    _isWorldMatOutOfDate = true;

    for(auto child : _children)
        ((NewNodeTRS*)child)->markWorldMatDirty();
}

void NewNodeTRS::updateIfDirty()
{
    if(_isLocalMatOutOfDate)
        updateLocalMat();

    if(_isWorldMatOutOfDate)
        updateWorldMat();
}


void NewNodeTRS::updateLocalMat() const
{
    setMatrixTRS(_localMat, _position, _rotation, _scale);
    _isLocalMatOutOfDate = false;
}

void NewNodeTRS::updateWorldMat() const
{
    //if (_parent)
    //{
    //    _worldMat = ((NewNodeTRS*)_parent)->worldMat() * localMat();
    //}
    //else
    //{
    //    _worldMat = localMat();
    //}

    //_wm.setMatrix(_worldMat);
    //_wmI.setMatrix(_wm);
    //_wmI.invert();
    //_wmN.setMatrix(_wmI.mat3());
    //_wmN.transpose();

    //_isWorldMatOutOfDate = false;
}

//
//void NewNodeTRS::position(const SLVec3f& pos, SLTransformSpace space)
//{
//    if(space == TS_Parent)
//        _position = pos;
//    // TODO: add world space position setter
//    markDirty();
//}
//
//void NewNodeTRS::rotation(const SLQuat4f& rot, SLTransformSpace space)
//{
//    if (space == TS_Parent) 
//        _rotation = rot;
//    // TODO: add world space rotation setter
//
//    markDirty();
//}
//
//void NewNodeTRS::scale(const SLVec3f& scale, SLTransformSpace space)
//{
//    if(space == TS_Parent)
//        _scale = scale;
//    // TODO: add world space scale setter (allow for shearing!)
//    markDirty();
//}
//
//  
//const SLVec3f& NewNodeTRS::position(SLTransformSpace space) const
//{
//    // TODO: add possibility to get world or local
//    return _position;
//}
//
//const SLQuat4f& NewNodeTRS::rotation(SLTransformSpace space) const
//{    
//    // TODO: add possibility to get world or local
//    return _rotation;
//}
//
//const SLVec3f& NewNodeTRS::scale(SLTransformSpace space) const
//{
//    // TODO: add possiblity to get world or local
//    return _scale;
//}
//
//SLVec3f NewNodeTRS::forward(SLTransformSpace space) const
//{
//    // TODO: add forward, up, right directions to the cache (like unity Transform)
//    return _rotation.rotate(-SLVec3f::AXISZ);
//}
//
//SLVec3f NewNodeTRS::up(SLTransformSpace space) const
//{
//    // TODO: add forward, up, right directions to the cache (like unity Transform)
//    return _rotation.rotate(SLVec3f::AXISY);
//}
//
//SLVec3f NewNodeTRS::right(SLTransformSpace space) const
//{
//    // TODO: add forward, up, right directions to the cache (like unity Transform)
//    return _rotation.rotate(SLVec3f::AXISX);
//}
//
//const SLMat4f& NewNodeTRS::worldMat() const
//{
//    if(_isWorldMatOutOfDate)
//        updateWorldMat();
//
//    return _worldMat;
//}
//
//const SLMat4f& NewNodeTRS::localMat() const
//{
//    if (_isLocalMatOutOfDate)
//        updateLocalMat();
//
//    return _localMat;
//}
//
//
//void NewNodeTRS::move(const SLVec3f& dist, SLTransformSpace space)
//{
//    if (space == TS_Parent)
//        position(_position + dist);
//    // TODO: add world space implementation
//}
//
//void NewNodeTRS::moveRel(const SLVec3f& dist)
//{
//    SLVec3f moved = _rotation.rotate(dist);
//    position(_position + moved);
//}
//
//void NewNodeTRS::lookAtPos(const SLVec3f& point, const SLVec3f& up, SLTransformSpace space)
//{
//    SLVec3f newDir = point - _position;
//    lookAtDir(newDir, up, space);
//}
//
//void NewNodeTRS::lookAtDir(const SLVec3f& point, const SLVec3f& up, SLTransformSpace space)
//{
//    //TODO: implement
//}
//
//void NewNodeTRS::rotate(const SLfloat angleDeg, const SLVec3f& axis, SLTransformSpace space)
//{
//    SLQuat4f rot(angleDeg, axis);
//
//    rotate(rot, space);
//}
//
//void NewNodeTRS::rotate(const SLQuat4f& rot, SLTransformSpace space)
//{
//    if (space == TS_Parent)
//        rotation(_rotation * rot);
//    // TODO: add world space implementation
//}
//
//void NewNodeTRS::roll(const SLfloat angleDeg)
//{
//    rotate(angleDeg, forward());
//}
//
//void NewNodeTRS::yaw(const SLfloat angleDeg)
//{
//    rotate(angleDeg, up());
//}
//
//void NewNodeTRS::pitch(const SLfloat angleDeg)
//{
//    rotate(angleDeg, right());
//}





