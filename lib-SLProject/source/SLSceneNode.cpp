//#############################################################################
//  File:      SLSceneNode.cpp
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

#include <SLSceneNode.h>
#include <SLAnimation.h>
#include <SLSceneView.h>
#include <SLRay.h>
#include <SLCamera.h>
#include <SLLightSphere.h>

//-----------------------------------------------------------------------------
/*! 
Default constructor just setting the name. 
*/ 
SLSceneNode::SLSceneNode(SLstring name) : SLNode(name)
{  
    _stateGL = SLGLState::getInstance();
    _drawBits.allOff();
    _isAABBUpToDate = false;
}
//-----------------------------------------------------------------------------
/*! 
Constructor with a mesh pointer and name.
*/ 
SLSceneNode::SLSceneNode(SLMesh* mesh, SLstring name) : SLNode(name)
{  
    _stateGL = SLGLState::getInstance();
    _drawBits.allOff();
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
SLSceneNode::~SLSceneNode()
{  
    //SL_LOG("~SLSceneNode: %s\n", name().c_str());
}



//-----------------------------------------------------------------------------
/*! 
Simply adds a mesh to its mesh pointer vector of the node.
*/ 
void SLSceneNode::addMesh(SLMesh* mesh)
{
    if (!mesh)
        return;

    if (find(_meshes.begin(), _meshes.end(), mesh) != _meshes.end())
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
bool SLSceneNode::insertMesh(SLMesh* insertM, SLMesh* afterM)
{
    assert(insertM && afterM);
    assert(insertM != afterM);

    SLVMesh::iterator found = find(_meshes.begin(), _meshes.end(), afterM);
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
bool SLSceneNode::removeMesh()
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
bool SLSceneNode::removeMesh(SLMesh* mesh)
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
bool SLSceneNode::removeMesh(SLstring name)
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
SLMesh* SLSceneNode::findMesh(SLstring name)
{  
    assert(name!="");
    for (SLint i=0; i<_meshes.size(); ++i)
        if (_meshes[i]->name() == name) return _meshes[i];
    return 0;
}
//-----------------------------------------------------------------------------
/*!
Returns true if the node contains the provided mesh
*/
SLbool SLSceneNode::containsMesh(const SLMesh* mesh)
{
    for (SLint i = 0; i < _meshes.size(); ++i)
        if (_meshes[i] == mesh)
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
void SLSceneNode::drawMeshes(SLSceneView* sv)
{
    for (SLint i=0; i<_meshes.size(); ++i)
        _meshes[i]->draw(sv, this);
}

//-----------------------------------------------------------------------------
/*!
Searches for all nodes that contain the provided mesh
*/
vector<SLSceneNode*> SLSceneNode::findChildren(const SLMesh* mesh,
                                               SLbool findRecursive)
{
    vector<SLSceneNode*> list;
    findChildrenHelper(mesh, list, findRecursive);

    return list;
}
//-----------------------------------------------------------------------------
/*!
Helper function of findChildren for meshes
*/
void SLSceneNode::findChildrenHelper(const SLMesh* mesh,
                                     vector<SLSceneNode*>& list,
                                     SLbool findRecursive)
{
    for (SLint i = 0; i < _children.size(); ++i)
    {
        SLSceneNode* node = (SLSceneNode*)_children[i];
        if (node->containsMesh(mesh))
            list.push_back(node);
                    
        if (findRecursive)
            ((SLSceneNode*)_children[i])->findChildrenHelper(mesh,
                                                             list,
                                                             findRecursive);
    }
}

//-----------------------------------------------------------------------------
/*!
Applies an animation transform to the local matrix. If an
animation was done here or in one of the children node the function returns 
true.
*/
SLbool SLSceneNode::animateRec(SLfloat timeMS)
{  
    SLbool gotAnimated = false;

    if (!SLScene::current->_stopAnimations)
    {  
        //if (_animation && !_animation->isFinished())
        //{  _animation->animate(this, timeMS);
        //    gotAnimated = true;
        //}

        // animate children nodes for groups or group derived classes
        for (SLint i=0; i<_children.size(); ++i)
            if (((SLSceneNode*)_children[i])->animateRec(timeMS))
                gotAnimated = true;
    }
    return gotAnimated;
}
//-----------------------------------------------------------------------------
/*!
Does the view frustum culling by checking whether the AABB is 
inside the view frustum. The check is done in world space. If a AABB is visible
the nodes children are checked recursively.
If a node containes meshes with alpha blended materials it is added to the 
blendedNodes vector. If not it is added to the opaqueNodes vector.
*/
void SLSceneNode::cullRec(SLSceneView* sv)
{     
    // Do frustum culling for all shapes except cameras & lights
    if (sv->doFrustumCulling() &&
        typeid(*this)!=typeid(SLCamera) &&
        typeid(*this)!=typeid(SLLightSphere))
        sv->camera()->isInFrustum(&_aabb);
    else _aabb.isVisible(true);

    // Cull the group nodes recursively
    if (_aabb.isVisible())
    {  
        for (SLint i=0; i<_children.size(); ++i)
            ((SLSceneNode*)_children[i])->cullRec(sv);
      
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
void SLSceneNode::drawRec(SLSceneView* sv)
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
   
    for (SLint i=0; i<_children.size(); ++i)
        ((SLSceneNode*)_children[i])->drawRec(sv);

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

        // Draw the animation curve
        //if (_animation)
        //    _animation->drawWS();
         
        _stateGL->popModelViewMatrix(); 
    }
}
//-----------------------------------------------------------------------------
/*!
Updates the statistic numbers of the passed SLNodeStats struct
and calls recursively the same method for all children.
*/
void SLSceneNode::statsRec(SLNodeStats &stats)
{  
    stats.numBytes += sizeof(SLSceneNode);

    if (_children.size() == 0)
         stats.numLeafNodes++;
    else stats.numGroupNodes++;
     
    for (SLint i=0; i<_meshes.size(); ++i)
        _meshes[i]->addStats(stats);
   
    for (SLint i=0; i<_children.size(); ++i)
        ((SLSceneNode*)_children[i])->statsRec(stats);
}
//-----------------------------------------------------------------------------
/*!
Intersects the nodes meshes with the given ray. The intersection
test is only done if the AABB is intersected. The ray-mesh intersection is
done in the nodes object space. The rays origin and direction is therefore 
transformed into the object space.
*/
bool SLSceneNode::hitRec(SLRay* ray)
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
        for (SLint i=0; i<_meshes.size(); ++i)
        {
            if (_meshes[i]->hit(ray, this) && !wasHit) 
                wasHit = true;
            if (ray->isShaded()) 
                return true;
        }
    }

    // Test children nodes
    for (SLint i=0; i<_children.size(); ++i)
    {
        if (((SLSceneNode*)_children[i])->hitRec(ray) && !wasHit)
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
SLSceneNode* SLSceneNode::copyRec()
{
    SLSceneNode* copy = new SLSceneNode(name());
    copy->_om = _om;
    copy->_depth = _depth;
    copy->_isAABBUpToDate = _isAABBUpToDate;
    copy->_isAABBUpToDate = _isWMUpToDate;
    copy->_drawBits = _drawBits;
    copy->_aabb = _aabb;

    //if (_animation)
    //    copy->_animation = new SLAnimation(*_animation);
    //else
    //    copy->_animation = 0;

    for (SLint i=0; i<_meshes.size(); ++i)
        copy->addMesh(_meshes[i]);
   
    for (SLint i=0; i<_children.size(); ++i)
        copy->addChild(((SLSceneNode*)_children[i])->copyRec());
   
    return copy;
}

//-----------------------------------------------------------------------------
/*!
Flags this node's AABB for an update. If a node 
changed we need to update it's world space AABB. This needs to also be propagated
up the parent chain since the AABB of a node incorperates the AABB's of child
nodes.
*/
void SLSceneNode::needAABBUpdate()
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
Updates the axis aligned bounding box in world space. 
*/
SLAABBox& SLSceneNode::updateAABBRec()
{
    if (_isAABBUpToDate)
        return _aabb;

    // empty the AABB (= max negative AABB)
    if (_meshes.size() > 0 || _children.size() > 0)
    {   _aabb.minWS(SLVec3f( FLT_MAX, FLT_MAX, FLT_MAX));
        _aabb.maxWS(SLVec3f(-FLT_MAX,-FLT_MAX,-FLT_MAX));  
    }

    // Build or update AABB of meshes & merge them to the nodes aabb in WS
    for (SLint i=0; i<_meshes.size(); ++i)
    {   SLAABBox aabbMesh;
        _meshes[i]->buildAABB(aabbMesh, updateAndGetWM());
        _aabb.mergeWS(aabbMesh);
    }
    
    // Merge children in WS
    for (SLint i=0; i<_children.size(); ++i)
        if (typeid(*_children[i])!=typeid(SLCamera))
            _aabb.mergeWS(((SLSceneNode*)_children[i])->updateAABBRec());

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
void SLSceneNode::dumpRec()
{
    // dump node
    for (SLint i = 0; i < _depth; ++i) cout << "   ";
    cout << "Node: " << _name << endl;

    // dump meshes of node
    if (_meshes.size() > 0)
    {   for (SLint m = 0; m<_meshes.size(); ++m)
        {   for (SLint i = 0; i < _depth; ++i) cout << "   ";
            cout << "- Mesh: " << _meshes[m]->name();
            cout << ", " << _meshes[m]->numI*3 << " tri";
            if (_meshes[m]->mat)
            cout << ", Mat: " << _meshes[m]->mat->name();
            cout << endl;
        }
    }

    // dump children nodes
    for (SLint i=0; i<_children.size(); ++i)
        ((SLSceneNode*)_children[i])->dumpRec();
}
//-----------------------------------------------------------------------------
/*!
recursively sets the specified drawbit on or off.
See also SLDrawBits.
*/
void SLSceneNode::setDrawBitsRec(SLuint bit, SLbool state)
{
    _drawBits.set(bit, state);
    for (SLint i=0; i<_children.size(); ++i)
        ((SLSceneNode*)_children[i])->setDrawBitsRec(bit, state);
}
//-----------------------------------------------------------------------------
/*!
Scales and translates the node so that its largest
dimension is maxDim and the center is in [0,0,0].
*/
void SLSceneNode::scaleToCenter(SLfloat maxDim)
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

