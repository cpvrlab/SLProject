//#############################################################################
//  File:      SLNode.cpp
//  Author:    Marc Wacker, Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLNode.h>
#include <SLAnimation.h>
#include <SLSceneView.h>
#include <SLRay.h>
#include <SLCamera.h>
#include <SLLightSphere.h>
#include <SLLightRect.h>

//-----------------------------------------------------------------------------
/*! 
Default constructor just setting the name. 
*/ 
SLNode::SLNode(SLstring name) : SLObject(name)
{  
    _stateGL = SLGLState::getInstance();
    _parent = 0;
    _depth = 1;
    _om.identity();
    _wm.identity();
    _wmI.identity();
    _wmN.identity();
    _drawBits.allOff();
    _animation = 0;
    _isWMUpToDate = false;
    _isAABBUpToDate = false;
}
//-----------------------------------------------------------------------------
/*! 
Constructor with a mesh pointer and name.
*/ 
SLNode::SLNode(SLMesh* mesh, SLstring name) : SLObject(name)
{  
    _stateGL = SLGLState::getInstance();
    _parent = 0;
    _depth = 1;
    _om.identity();
    _wm.identity();
    _wmI.identity();
    _wmN.identity();
    _drawBits.allOff();
    _animation = 0;
    _isWMUpToDate = false;
    _isAABBUpToDate = false;
    
    addMesh(mesh);
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

    for (auto child : _children) delete child;
    _children.clear();

    if (_animation) 
        delete _animation;
}



//-----------------------------------------------------------------------------
/*! 
Simply adds a mesh to its mesh pointer vector of the node.
*/ 
void SLNode::addMesh(SLMesh* mesh)
{
    if (!mesh)
        return;

    if (std::find(_meshes.begin(), _meshes.end(), mesh) != _meshes.end())
        return;

    // Take over mesh name if node name is default name
    if (_name == "Node" && mesh->name() != "Mesh")
        _name = mesh->name() + "-Node";

    _meshes.push_back(mesh);
    mesh->init(this);
}
//-----------------------------------------------------------------------------
/*! 
Inserts a mesh pointer in the mesh pointer vector after the
specified afterM pointer.
*/ 
bool SLNode::insertMesh(SLMesh* insertM, SLMesh* afterM)
{
    assert(insertM && afterM);
    assert(insertM != afterM);

    auto found = std::find(_meshes.begin(), _meshes.end(), afterM);
    if (found != _meshes.end())
    {   _meshes.insert(found, insertM);
        insertM->init(this);

        // Take over mesh name if node name is default name
        if (_name == "Node" && insertM->name() != "Mesh")
            _name = insertM->name() + "-Node";

        return true;
    }
    return false;
}
//-----------------------------------------------------------------------------
/*! 
Removes the last mesh.
*/
bool SLNode::removeMesh()
{
    if (_meshes.size() > 0)
    {   _meshes.pop_back();
        return true;
    }
    return false;
}
//-----------------------------------------------------------------------------
/*! 
Removes the specified mesh from the vector.
*/
bool SLNode::removeMesh(SLMesh* mesh)
{
    assert(mesh);
    for (SLint i=0; i<_meshes.size(); ++i)
    {   if (_meshes[i]==mesh)
        {   _meshes.erase(_meshes.begin()+i);
            return true;
        }
    }
    return false;
}
//-----------------------------------------------------------------------------
/*! 
Removes the specified mesh by name from the vector.
*/
bool SLNode::removeMesh(SLstring name)
{
    assert(name!="");
    SLMesh* found = findMesh(name);
    if (found) return removeMesh(found);
    return false;
}
//-----------------------------------------------------------------------------
/*! 
SLNode::findMesh finds the specified mesh by name.
*/
SLMesh* SLNode::findMesh(SLstring name)
{  
    assert(name!="");
    for (auto mesh : _meshes)
        if (mesh->name() == name) return mesh;
    return 0;
}
//-----------------------------------------------------------------------------
/*!
Returns true if the node contains the provided mesh
*/
SLbool SLNode::containsMesh(const SLMesh* mesh)
{
    for (auto m : _meshes)
        if (m == mesh)
            return true;

    return false;
}
//-----------------------------------------------------------------------------
/*! 
DrawMeshes draws the meshes by just calling the SLMesh::draw method.
See also the SLNode::drawRec method for more information. There are two 
possibilities to guarantee that the meshes of a node are transformed correctly:
<ul>
<li>
<b>Flat drawing</b>: Before the SLNode::drawMeshes is called we must multiply the
nodes world matrix (SLNode::_wm) to the OpenGL modelview matrix
(SLGLState::modelViewMatrix). The flat drawing method is slightly faster and 
the order of drawing doesn't matter anymore. This method is used within 
SLSceneView::draw3D to draw first a list of all opaque meshes and the a list
of all meshes with a transparent material.
</li>
<li>
<b>Recursive drawing</b>: By calling SLNode::drawRec all meshes are drawn with
SLNode::drawMeshes with only the object matrix applied before drawing. After
the meshes the drawRec method is called on each children node. By pushing 
the OpenGL modelviewmatrix before on a stack this method is also refered as
stack drawing.
</li>
</ul>
*/
void SLNode::drawMeshes(SLSceneView* sv)
{
    for (auto mesh : _meshes)
        mesh->draw(sv, this);
}
//-----------------------------------------------------------------------------



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

    auto found = std::find(_children.begin(), _children.end(), afterC);
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
    {   delete _children[_children.size()-1];
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
/*!
Searches for all nodes that contain the provided mesh
*/
vector<SLNode*> SLNode::findChildren(const SLMesh* mesh,
                                     SLbool findRecursive)
{
    vector<SLNode*> list;
    findChildrenHelper(mesh, list, findRecursive);

    return list;
}
//-----------------------------------------------------------------------------
/*!
Helper function of findChildren for meshes
*/
void SLNode::findChildrenHelper(const SLMesh* mesh,
                                vector<SLNode*>& list,
                                SLbool findRecursive)
{
    for (auto child : _children)
    {   if (child->containsMesh(mesh))
            list.push_back(child);
        if (findRecursive)
            child->findChildrenHelper(mesh, list, findRecursive);
    }
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
/*!
Does the view frustum culling by checking whether the AABB is 
inside the view frustum. The check is done in world space. If a AABB is visible
the nodes children are checked recursively.
If a node containes meshes with alpha blended materials it is added to the 
blendedNodes vector. If not it is added to the opaqueNodes vector.
*/
void SLNode::cullRec(SLSceneView* sv)  
{     
    // Do frustum culling for all shapes except cameras & lights
    if (sv->doFrustumCulling() &&
        typeid(*this)!=typeid(SLCamera) &&
        typeid(*this)!=typeid(SLLightSphere) &&
        typeid(*this)!=typeid(SLLightRect))
        sv->camera()->isInFrustum(&_aabb);
    else _aabb.isVisible(true);

    // Cull the group nodes recursively
    if (_aabb.isVisible())
    {  
        for (auto child : _children)
            child->cullRec(sv);
      
        // for leaf nodes add them to the blended or opaque vector
        if (_aabb.hasAlpha())
             sv->blendNodes()->push_back(this);
        else sv->opaqueNodes()->push_back(this);
    }
}
//-----------------------------------------------------------------------------
/*!
Draws the the nodes meshes with SLNode::drawMeshes and calls 
recursively the drawRec method of the nodes children. 
The nodes object matrix (SLNode::_om) is multiplied before the meshes are drawn. 
This recursive drawing is more expensive than the flat drawing with the 
opaqueNodes vector because of the additional matrix multiplications. 
The order of drawing doesn't matter in flat drawing because the world 
matrix (SLNode::_wm) is used for transform. See also SLNode::drawMeshes.
The drawRec method is <b>still used</b> for the rendering of the 2D menu!
*/
void SLNode::drawRec(SLSceneView* sv)
{    
    bool rootButton = (_name == ">");
    bool loadSceneButton = (_name == "Load Scene >");
    // Do frustum culling for all shapes except cameras & lights
    if (sv->doFrustumCulling() && !_aabb.isVisible()) return; 
   
    _stateGL->pushModelViewMatrix();
    _stateGL->modelViewMatrix.multiply(_om.m());
    _stateGL->buildInverseAndNormalMatrix();
   
    ///////////////
    drawMeshes(sv);
    ///////////////
   
    for (auto child : _children)
        child->drawRec(sv);

    _stateGL->popModelViewMatrix();

    // Draw axis aligned bounding box
    SLbool showBBOX = sv->drawBit(SL_DB_BBOX) || drawBit(SL_DB_BBOX);
    SLbool showAXIS = sv->drawBit(SL_DB_AXIS) || drawBit(SL_DB_AXIS);
    SLbool showSELECT = drawBit(SL_DB_SELECTED);
    if (showBBOX || showAXIS || showSELECT)
    {  
        _stateGL->pushModelViewMatrix();
        _stateGL->modelViewMatrix.setMatrix(sv->camera()->updateAndGetVM().m());
      
        // Draw AABB of all other shapes only
        if (showBBOX && !showSELECT)
        {   if (_meshes.size() > 0)
                 _aabb.drawWS(SLCol3f(1,0,0));
            else _aabb.drawWS(SLCol3f(1,0,1));
        }

        if (showAXIS)
            _aabb.drawAxisWS();

        // Draw AABB if shapes is selected
        if (showSELECT)
            _aabb.drawWS(SLCol3f(1,1,0));

        _stateGL->popModelViewMatrix(); 
    }
}
//-----------------------------------------------------------------------------
/*!
Updates the statistic numbers of the passed SLNodeStats struct
and calls recursively the same method for all children.
*/
void SLNode::statsRec(SLNodeStats &stats)  
{  
    stats.numBytes += sizeof(SLNode);
    stats.numNodes++;

    if (_children.size() == 0)
         stats.numLeafNodes++;
    else stats.numGroupNodes++;
     
    for (auto mesh : _meshes) mesh->addStats(stats);
    for (auto child : _children) child->statsRec(stats);
}
//-----------------------------------------------------------------------------
/*!
Intersects the nodes meshes with the given ray. The intersection
test is only done if the AABB is intersected. The ray-mesh intersection is
done in the nodes object space. The rays origin and direction is therefore 
transformed into the object space.
*/
bool SLNode::hitRec(SLRay* ray)
{
    assert(ray != 0);

    // Do not test hidden nodes
    if (_drawBits.get(SL_DB_HIDDEN)) 
        return false;

    // Do not test origin node for shadow rays 
    if (this==ray->originNode && ray->type==SHADOW) 
        return false;
   
    // Check first AABB for intersection
    if (!_aabb.isHitInWS(ray)) 
        return false;


    SLbool wasHit = false;
   
    // Transform ray to object space for non-groups
    if (_meshes.size() > 0)    
    {  
        // transform origin position to object space
        ray->originOS.set(updateAndGetWMI().multVec(ray->origin));
         
        // transform the direction only with the linear sub matrix
        ray->setDirOS(_wmI.mat3() * ray->dir);

        // test all meshes
        for (auto mesh : _meshes)
        {   if (mesh->hit(ray, this) && !wasHit) 
                wasHit = true;
            if (ray->isShaded()) 
                return true;
        }
    }

    // Test children nodes
    for (auto child : _children)
    {   if (child->hitRec(ray) && !wasHit) 
            wasHit = true;
        if (ray->isShaded()) 
            return true;
    }

    return wasHit;
}
//-----------------------------------------------------------------------------
/*! 
Copies the nodes meshes and children recursively.
*/ 
SLNode* SLNode::copyRec()
{
    SLNode* copy = new SLNode(name());
    copy->_om = _om;
    copy->_depth = _depth;
    copy->_isAABBUpToDate = _isAABBUpToDate;
    copy->_isAABBUpToDate = _isWMUpToDate;
    copy->_drawBits = _drawBits;
    copy->_aabb = _aabb;

    if (_animation) 
         copy->_animation = new SLAnimation(*_animation);
    else copy->_animation = 0;

    for (auto mesh : _meshes) copy->addMesh(mesh);
    for (auto child : _children) copy->addChild(child->copyRec());
   
    return copy;
}
//-----------------------------------------------------------------------------
/*!
Sets the parent for this node and updates its depth
*/
void SLNode::parent(SLNode* p)
{
    _parent = p;

    if(_parent)
         _depth = _parent->depth() + 1;
    else _depth = 1;
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
    for (auto child : _children)
        child->needUpdate();

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

    // mark the WM of the children dirty since their parent just changed
    for (auto child : _children)
        child->needWMUpdate();
}
//-----------------------------------------------------------------------------
/*!
Flags this node's AABB for an update. If a node 
changed we need to update it's world space AABB. This needs to also be propagated
up the parent chain since the AABB of a node incorperates the AABB's of child
nodes.
*/
void SLNode::needAABBUpdate()
{
    // stop if we reach a node that is already flagged.
    if (!_isAABBUpToDate)
        return;

    _isAABBUpToDate = false;

    // flag parent's for an AABB update too since they need to
    // merge the child AABBs
    if (_parent)
        _parent->needAABBUpdate();
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
/*!
Updates the axis aligned bounding box in world space. 
*/
SLAABBox& SLNode::updateAABBRec()
{
    if (_isAABBUpToDate)
        return _aabb;

    // empty the AABB (= max negative AABB)
    if (_meshes.size() > 0 || _children.size() > 0)
    {   _aabb.minWS(SLVec3f( FLT_MAX, FLT_MAX, FLT_MAX));
        _aabb.maxWS(SLVec3f(-FLT_MAX,-FLT_MAX,-FLT_MAX));  
    }

    // Build or update AABB of meshes & merge them to the nodes aabb in WS
    for (auto mesh : _meshes)
    {   SLAABBox aabbMesh;
        mesh->buildAABB(aabbMesh, updateAndGetWM());
        _aabb.mergeWS(aabbMesh);
    }
    
    // Merge children in WS
    for (auto child : _children)
        if (typeid(*child)!=typeid(SLCamera))
            _aabb.mergeWS(child->updateAABBRec());

    // We need min & max also in OS for the uniform grid intersection in OS
    _aabb.fromWStoOS(_aabb.minWS(), _aabb.maxWS(), updateAndGetWMI());

    // For visualizing the nodes orientation we finally update the axis in WS
    _aabb.updateAxisWS(updateAndGetWM());

    _isAABBUpToDate = true;
    return _aabb;
}
//-----------------------------------------------------------------------------
/*!
prints the node name with the names of the meshes recursively
*/
void SLNode::dumpRec()
{
    // dump node
    for (SLint i = 0; i < _depth; ++i) cout << "   ";
    cout << "Node: " << _name << endl;

    // dump meshes of node
    if (_meshes.size() > 0)
    {   for (auto mesh : _meshes)
        {   for (SLint i = 0; i < _depth; ++i) cout << "   ";
            cout << "- Mesh: " << mesh->name();
            cout << ", " << mesh->numI*3 << " tri";
            if (mesh->mat)
            cout << ", Mat: " << mesh->mat->name();
            cout << endl;
        }
    }

    // dump children nodes
    for (auto child : _children)
        child->dumpRec();
}
//-----------------------------------------------------------------------------
/*!
recursively sets the specified drawbit on or off.
See also SLDrawBits.
*/
void SLNode::setDrawBitsRec(SLuint bit, SLbool state)
{
    _drawBits.set(bit, state);
    for (auto child : _children)
        child->setDrawBitsRec(bit, state);
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
/*!
sets the position of this node to pos in 'relativeTo' space.
@note using TS_Object for this function yields the same result as calling
translate(pos, TS_Object)
*/
void SLNode::translation(const SLVec3f& pos, SLTransformSpace relativeTo)
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
of (0, 1, 0) with TS_Object will rotate the node around its own up axis.
*/
void SLNode::rotation(const SLQuat4f& rot, 
                      SLTransformSpace relativeTo)
{   
    SLMat4f rotation = rot.toMat4(); 
    
    if (_parent && relativeTo == TS_World)
    {
        // get the inverse parent rotation to remove it from our current rotation
        // we want the input quaternion to absolutely set our new rotation relative 
        // to the world axes
        SLMat4f parentRotInv = _parent->updateAndGetWMI();
        parentRotInv.translation(0, 0, 0);
        
        // set the om rotation to the inverse of the parents rotation to achieve a
        // 0, 0, 0 relative rotation in world space
        _om.rotation(0, 0, 0, 0);
        _om *= parentRotInv;
        needUpdate();
        rotate(rot, relativeTo);
    }
    else if (relativeTo == TS_Parent)
    {   // relative to parent, reset current rotation and just rotate again
        _om.rotation(0, 0, 0, 0);
        needUpdate();
        rotate(rot, relativeTo);
    }
    else
    {
        // in TS_Object everything is relative to our current orientation
        _om.rotation(0, 0, 0, 0);        
        _om *= rotation;
        needUpdate();
    }
}
//-----------------------------------------------------------------------------
/*!
sets the rotation of this node. The axis parameter
will be transformed into 'relativeTo' space. So a passing in an axis
of (0, 1, 0) with TS_Object will rotate the node around its own up axis.
*/
void SLNode::rotation(SLfloat angleDeg, const SLVec3f& axis, 
                      SLTransformSpace relativeTo)
{    
    SLQuat4f rot(angleDeg, axis);
    rotation(rot, relativeTo);
}
//-----------------------------------------------------------------------------
/*!
Sets the scaling of current object matrix
@note this modifier doesn't allow for different transform spaces, so there
isn't the possiblity for shearing an object currently.
*/
void SLNode::scaling(const SLVec3f& scaling)
{
    _om.scaling(scaling);
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
        case TS_Object:
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
    SLQuat4f rot(angleDeg, axis);
    rotate(rot, relativeTo);
}
//-----------------------------------------------------------------------------
/*!
Rotates the node around its local origin relative to the space expressed by 'relativeTo'.
*/
void SLNode::rotate(const SLQuat4f& rot, SLTransformSpace relativeTo)
{
    SLQuat4f norm = rot.normalized();
    SLMat4f rotation = rot.toMat4();


    if (relativeTo == TS_Object)
    {
        _om *= rotation;
    }
    else if (_parent && relativeTo == TS_World)
    {
        SLMat4f rot;
        rot.translate(updateAndGetWM().translation());
        rot.multiply(rotation);
        rot.translate(-updateAndGetWM().translation());

        _om = _parent->_wm.inverse() * rot * updateAndGetWM();
    }
    else // relativeTo == TS_Parent || relativeTo == TS_World && !_parent
    {
        SLMat4f rot;
        rot.translate(translation());
        rot.multiply(rotation);
        rot.translate(-translation());

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

    if (relativeTo == TS_Object)
        _om.setMatrix(_om * rot);
    else
        _om.setMatrix(rot * _om);

    needUpdate();
}
//-----------------------------------------------------------------------------
/*!
Adds a scale transform to the current object matrix
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
Rotates the object so that it's forward vector is pointing towards the 'target' 
point. Default forward is -Z. The 'relativeTo' parameter defines in what space
the 'target' parameter is to be interpreted in. 
*/
void SLNode::lookAt(const SLVec3f& target, const SLVec3f& up, 
                    SLTransformSpace relativeTo)
{
    SLVec3f pos = translation();
    SLVec3f dir;
    SLVec3f localUp = up;

    if (relativeTo == TS_World && _parent)
    {
        SLVec3f localTarget = _parent->updateAndGetWMI() * target;
        localUp = _parent->updateAndGetWMI().mat3() * up;
        dir = localTarget - translation();
    }
    else if (relativeTo == TS_Object)
        dir = _om * target - translation();
    else
        dir = target - translation();

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
Scales and translates the node so that its largest
dimension is maxDim and the center is in [0,0,0].
*/
void SLNode::scaleToCenter(SLfloat maxDim)
{  
    _aabb = updateAABBRec();
    SLVec3f size(_aabb.maxWS()-_aabb.minWS());
    SLVec3f center((_aabb.maxWS()+_aabb.minWS()) * 0.5f);
    SLfloat scaleFactor = maxDim / size.maxXYZ();
    if (fabs(scaleFactor) > FLT_EPSILON)
        scale(scaleFactor);
    else cout << "Node can't be scaled: " << name().c_str() << endl;
    translate(-center);
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
