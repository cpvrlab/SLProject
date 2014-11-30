//#############################################################################
//  File:      SLNode.cpp
//  Author:    Marc Wacker, Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLNode.h>
#include <SLSceneView.h>
#include <SLRay.h>
#include <SLCamera.h>
#include <SLLightSphere.h>

//-----------------------------------------------------------------------------
/*! 
Default constructor just setting the name. 
*/ 
SLNode::SLNode(SLstring name) : SLObject(name)
{
    _parent = 0;
    _depth = 1;
    _om.identity();
    _wm.identity();
    _wmI.identity();
    _wmN.identity();
    _isWMUpToDate = false;
}
//-----------------------------------------------------------------------------
/*! 
Destructor deletes all children recursively and the animation.
The meshes are not deleted. They are deleted at the end by the SLScene mesh
vector. The entire scenegraph is deleted by deleting the SLScene::_root3D node.
Nodes that are not in the scenegraph will not be deleted at scene destruction.
*/ 
SLNode::~SLNode()
{  
    //SL_LOG("~SLNode: %s\n", name().c_str());

    for (int i=0; i<_children.size(); ++i)
        delete _children[i];
    _children.clear();
}
//-----------------------------------------------------------------------------
/*!
Adds a child node to the children vector
*/
void SLNode::addChild(SLNode* child)
{
    assert(child);
    assert(this != child);

    // remove the node from it's old parent
    if(child->parent())
        child->parent()->deleteChild(child);

    _children.push_back(child);
    child->parent(this);
}
//-----------------------------------------------------------------------------
/*!
Inserts a child node in the children vector after the 
afterC node.
*/
bool SLNode::insertChild(SLNode* insertC, SLNode* afterC)
{
    assert(insertC && afterC);
    assert(insertC != afterC);

    SLVNode::iterator found = find(_children.begin(), _children.end(), afterC);
    if (found != _children.end())
    {   _children.insert(found, insertC);
        insertC->parent(this);
        return true;
    }
    return false;
}
//-----------------------------------------------------------------------------
/*!
Deletes all child nodes.
*/
void SLNode::deleteChildren()
{
    for (int i=0; i<_children.size(); ++i)
        delete _children[i];
    _children.clear();
}
//-----------------------------------------------------------------------------
/*!
Deletes the last child in the child vector.
*/
bool SLNode::deleteChild()
{
    if (_children.size() > 0)
    {  
        delete _children[_children.size()-1];
        _children.pop_back();
        return true;
    }
    return false;
}
//-----------------------------------------------------------------------------
/*!
Deletes a child from the child vector.
*/
bool SLNode::deleteChild(SLNode* child)
{
    assert(child);
    for (SLint i=0; i<_children.size(); ++i)
    {   if (_children[i]==child)
        {   _children.erase(_children.begin()+i);
            delete child;
            return true;
        }
    }
    return false;
}
//-----------------------------------------------------------------------------
/*!
Searches for a child with the name 'name' and deletes it.
*/
bool SLNode::deleteChild(SLstring name)
{
    assert(name!="");
    SLNode* found = findChild<SLNode>(name);
    if (found) return deleteChild(found);
    return false;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/*!
Sets the parent for this node and updates its depth
*/
void SLNode::parent(SLNode* p)
{
    _parent = p;

    if(_parent)
        _depth = _parent->depth() + 1;
    else
        _depth = 1;
}
//-----------------------------------------------------------------------------
/*!
Flags this node for an update. This function is called 
automatically if the local transform of the node or of its parent changed.
Nodes that are flagged for updating will recalculate their world transform
the next time it is requested by updateAndGetWM(). 
*/
void SLNode::needUpdate()
{
    // stop if we reach a node that is already flagged.
    if (!_isWMUpToDate)
        return;

    _isWMUpToDate = false;

    // mark the WM of the children dirty since their parent just changed
    for (SLint i=0; i<_children.size(); ++i)
        _children[i]->needUpdate();

    // flag AABB for an update
    needAABBUpdate();
}
//-----------------------------------------------------------------------------
/*!
Flags this node for a wm update. It is almost
identical to the needUpdate function but it won't flag AABBs.

This function is currently not in use but could give a slight performance
boost if it was called instead of needUpdate for the children of a 
node that changed.
*/
void SLNode::needWMUpdate()
{
    // stop if we reach a node that is already flagged.
    if (!_isWMUpToDate)
        return;

    _isWMUpToDate = false;
    _isAABBUpToDate = false;

    // mark the WM of the children dirty since their parent just changed
    for (SLint i=0; i<_children.size(); ++i)
        _children[i]->needWMUpdate();
}
//-----------------------------------------------------------------------------
/*!
A helper function that updates the current _wm to reflectthe local matrix. 
recursively calls the updateAndGetWM of the node's parent.

@note
This function is const because it has to be called from inside the updateAndGetWM
function which has to be const. Since we only update the private cache of this
class it is ok.
*/
void SLNode::updateWM() const
{
    if (_parent)
        _wm.setMatrix(_parent->updateAndGetWM() * _om); 
    else
        _wm.setMatrix(_om);
    
    _wmI.setMatrix(_wm);
    _wmI.invert();
    _wmN.setMatrix(_wm.mat3());

    _isWMUpToDate = true;
}
//-----------------------------------------------------------------------------
/*!
Will retrieve the current world matrix for this node.
If the world matrix is out of date it will update it and return a current result.
*/
const SLMat4f& SLNode::updateAndGetWM() const
{
    if (!_isWMUpToDate)
        updateWM();

    return _wm;
}
//-----------------------------------------------------------------------------
/*!
Will retrieve the current world inverse matrix for this node.
If the world matrix is out of date it will update it and return a current result.
*/
const SLMat4f& SLNode::updateAndGetWMI() const
{
    if (!_isWMUpToDate)
        updateWM();

    return _wmI;
}
//-----------------------------------------------------------------------------
/*!
Will retrieve the current world normal matrix for this node.
If the world matrix is out of date it will update it and return a current result.
*/
const SLMat3f& SLNode::updateAndGetWMN() const
{
    if (!_isWMUpToDate)
        updateWM();

    return _wmN;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/*!
sets the position of this node to pos in 'relativeTo' space.

@note using TS_Local for this function yields the same result as calling
translate(pos, TS_Local)
*/
void SLNode::position(const SLVec3f& pos, SLTransformSpace relativeTo)
{
    if (relativeTo == TS_World && _parent)
    {   // transform position to local space
        SLVec3f localPos = _parent->updateAndGetWMI() * pos;
        _om.translation(localPos);
    }
    else
        _om.translation(pos);

    needUpdate();
}
//-----------------------------------------------------------------------------
/*!
sets the rotation of this node. The axis parameter
will be transformed into 'relativeTo' space. So an passing in an axis
of (0, 1, 0) with TS_Local will rotate the node around its own up axis.
*/
void SLNode::rotation(SLfloat angleDeg, const SLVec3f& axis,
                      SLTransformSpace relativeTo)
{    
    if (_parent && relativeTo == TS_World)
    {
        SLMat4f rot;
        rot.translate(updateAndGetWM().translation());
        rot.rotation(angleDeg, axis);
        rot.translate(-updateAndGetWM().translation());
        _om = _parent->updateAndGetWMI() * rot * updateAndGetWM();
        needUpdate();
        return;
    }
    else if (relativeTo == TS_Parent)
    {   // relative to parent, reset current rotation and just rotate again
        _om.rotation(0, 0, 0, 0);
        rotate(angleDeg, axis, relativeTo);
    }
    else
    {
        // in TS_Local everything is relative to our current orientation
        rotate(angleDeg, axis, relativeTo);
    }

}
//-----------------------------------------------------------------------------
/*!
scales this.

@note this is not a setter but a scale modifier.
@note this modifier doesn't allow for different transform spaces, so there 
isn't the possiblity for shearing an object currently.
*/         
void SLNode::scale(const SLVec3f& scale)
{
    _om.scale(scale);
    needUpdate();
}

//-----------------------------------------------------------------------------
/*!
Moves the node by the vector 'delta' relative to the space expressed by 'relativeTo'.
*/
void SLNode::translate(const SLVec3f& delta, SLTransformSpace relativeTo)
{
    switch (relativeTo)
    {
        case TS_Local:
            _om.translate(delta);
            break;

        case TS_World:
            if (_parent)
            {   SLVec3f localVec = _parent->updateAndGetWMI().mat3() * delta;
                _om.translation(localVec + _om.translation());
            }
            else 
                _om.translation(delta + _om.translation());
            break;

        case TS_Parent:
            _om.translation(delta + _om.translation());
            break;
    }

    needUpdate();
}
//-----------------------------------------------------------------------------
/*!
Rotates the node around its local origin relative to the space expressed by 'relativeTo'.
*/
void SLNode::rotate(SLfloat angleDeg, const SLVec3f& axis,
                    SLTransformSpace relativeTo)
{
    if (relativeTo == TS_Local)
    {
        _om.rotate(angleDeg, axis);
    }
    else if (_parent && relativeTo == TS_World)
    {
        SLMat4f rot;
        rot.translate(updateAndGetWM().translation());
        rot.rotate(angleDeg, axis);
        rot.translate(-updateAndGetWM().translation());

        _om = _parent->updateAndGetWMI() * rot * updateAndGetWM();
    }
    else // relativeTo == TS_Parent || relativeTo == TS_World && !_parent
    {            
        SLMat4f rot;
        rot.translate(position());
        rot.rotate(angleDeg, axis);
        rot.translate(-position());
                
        _om.setMatrix(rot * _om);
    }

    needUpdate();
}
//-----------------------------------------------------------------------------
/*!
Rotates the node around its local origin relative to the space expressed by 'relativeTo'.
*/
void SLNode::rotate(const SLQuat4f& rot, SLTransformSpace relativeTo)
{
    SLMat4f rotation = rot.toMat4(); 

    
    if (relativeTo == TS_Local)
    {
        _om *= rotation;
    }
    else if (_parent && relativeTo == TS_World)
    {
        SLMat4f rot;
        rot.translate(updateAndGetWM().translation());
        rot.multiply(rotation);
        rot.translate(-updateAndGetWM().translation());

        _om = _parent->updateAndGetWMI() * rot * updateAndGetWM();
    }
    else // relativeTo == TS_Parent || relativeTo == TS_World && !_parent
    {            
        SLMat4f rot;
        rot.translate(position());
        rot.multiply(rotation);
        rot.translate(-position());
                
        _om.setMatrix(rot * _om);
    }

    needUpdate();
}
//-----------------------------------------------------------------------------
/*!
Rotates the node around an arbitrary point. The 'axis' and 'point' parameter
are relative to the space described by 'relativeTo'. 
*/
void SLNode::rotateAround(const SLVec3f& point, SLVec3f& axis,
                          SLfloat angleDeg, SLTransformSpace relativeTo)
{
    SLVec3f localPoint = point;
    SLVec3f localAxis = axis;
    
    if (relativeTo == TS_World && _parent)
    {
        localPoint = _parent->updateAndGetWMI() * point;
        localAxis = _parent->updateAndGetWMI().mat3() * axis;
    }

    SLMat4f rot;
    rot.translate(localPoint);
    rot.rotate(angleDeg, localAxis);
    rot.translate(-localPoint);

    if (relativeTo == TS_Local)
        _om.setMatrix(_om * rot);
    else
        _om.setMatrix(rot * _om);

    needUpdate();
}
//-----------------------------------------------------------------------------
/*!
Rotates the object so that it's forward vector is pointing towards the 'target' 
point. Default forward is -Z. The 'relativeTo' parameter defines in what space
the 'target' parameter is to be interpreted in. 
*/
void SLNode::lookAt(const SLVec3f& target, const SLVec3f& up,
                    SLTransformSpace relativeTo)
{
    SLVec3f pos = position();
    SLVec3f dir;
    SLVec3f localUp = up;

    if (relativeTo == TS_World && _parent)
    {
        SLVec3f localTarget = _parent->updateAndGetWMI() * target;
        localUp = _parent->updateAndGetWMI().mat3() * up;
        dir = localTarget - position();
    }
    else if (relativeTo == TS_Local)
        dir = _om * target - position();
    else
        dir = target - position();

    dir.normalize();
    
    SLfloat cosAngle = localUp.dot(dir);
    
    // dir and up are parallel and facing in the same direction 
    // or facing in opposite directions.
    // in this case we just rotate the up vector by 90° around
    // our current right vector
    // @todo This check might make more sense to be in Mat3.posAtUp
    if (fabs(cosAngle-1.0) <= FLT_EPSILON || fabs(cosAngle+1.0) <= FLT_EPSILON)
    {
        SLMat3f rot;
        rot.rotation(-90.0f, right());

        localUp = rot * localUp;
    }

    _om.posAtUp(pos, pos+dir, localUp);

    needUpdate();
}



//-----------------------------------------------------------------------------
/*! 
Saves the current position as the initial state
*/

void SLNode::setInitialState()
{
    _initialOM = _om;
}
//-----------------------------------------------------------------------------
/*! 
sesets this object to its initial state
*/
void SLNode::resetToInitialState()
{
    _om = _initialOM;
    needUpdate();
}
