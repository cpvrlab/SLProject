//#############################################################################
//   File:      SLNode.cpp
//   Date:      July 2014
//   Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//   Authors:   Marc Wacker, Marcus Hudritsch, Jan Dellsperger
//   License:   This software is provided under the GNU General Public License
//              Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLAnimation.h>
#include <SLKeyframeCamera.h>
#include <SLLightDirect.h>
#include <SLLightRect.h>
#include <SLLightSpot.h>
#include <SLNode.h>
#include <SLText.h>
#include <SLScene.h>
#include <SLEntities.h>
#include <SLSceneView.h>
#include <Profiler.h>

using std::cout;
using std::endl;

unsigned int SLNode::instanceIndex = 0;

//-----------------------------------------------------------------------------
// Static updateRec counter
SLuint SLNode::numWMUpdates = 0;
//-----------------------------------------------------------------------------
/*!
Default constructor just setting the name.
*/
SLNode::SLNode(const SLstring& name) : SLObject(name)
{
    _parent   = nullptr;
    _depth    = 1;
    _entityID = INT32_MIN;
    _om.identity();
    _wm.identity();
    _wmI.identity();
    _drawBits.allOff();
    _animation      = nullptr;
    _castsShadows   = true;
    _isWMUpToDate   = false;
    _isWMIUpToDate  = false;
    _isAABBUpToDate = false;
    _isSelected     = false;
    _mesh           = nullptr;
    _minLodCoverage = 0.0f;
    _levelForSM     = 0;
}
//-----------------------------------------------------------------------------
/*!
Constructor with a mesh pointer and name.
*/
SLNode::SLNode(SLMesh* mesh, const SLstring& name) : SLObject(name)
{
    assert(mesh && "No mesh passed");

    _parent   = nullptr;
    _depth    = 1;
    _entityID = INT32_MIN;
    _om.identity();
    _wm.identity();
    _wmI.identity();
    _drawBits.allOff();
    _animation      = nullptr;
    _castsShadows   = true;
    _isWMUpToDate   = false;
    _isWMIUpToDate  = false;
    _isAABBUpToDate = false;
    _isSelected     = false;
    _minLodCoverage = 0.0f;
    _levelForSM     = 0;
    _mesh           = nullptr;

    addMesh(mesh);
}
//-----------------------------------------------------------------------------
/*!
Constructor with a mesh pointer, translation vector and name.
*/
SLNode::SLNode(SLMesh*         mesh,
               const SLVec3f&  translation,
               const SLstring& name) : SLObject(name)
{
    assert(mesh && "No mesh passed");

    _parent   = nullptr;
    _depth    = 1;
    _entityID = INT32_MIN;
    _om.identity();
    _om.translate(translation);
    _wm.identity();
    _wmI.identity();
    _drawBits.allOff();
    _animation      = nullptr;
    _castsShadows   = true;
    _isWMUpToDate   = false;
    _isWMIUpToDate  = false;
    _isAABBUpToDate = false;
    _isSelected     = false;
    _minLodCoverage = 0.0f;
    _levelForSM     = 0;
    _mesh           = nullptr;

    addMesh(mesh);
}
//-----------------------------------------------------------------------------
/*!
Destructor deletes all children recursively and the animation.
The mesh is not deleted. Meshes get deleted at the end by the SLAssetManager
vector. The entire scenegraph is deleted by deleting the SLScene::_root3D node.
Nodes that are not in the scenegraph will not be deleted at scene destruction.
*/
SLNode::~SLNode()
{
#ifdef SL_USE_ENTITIES
    SLint entityID = SLScene::entities.getEntityID(this);
    SLint parentID = SLScene::entities.getParentID(this);
    if (entityID != INT32_MIN && parentID != INT32_MIN)
    {
        if (parentID == -1)
            SLScene::entities.clear();
        else
            SLScene::entities.deleteEntity(entityID);
    }
#endif

    for (auto* child : _children)
        delete child;
    _children.clear();

    delete _animation;
}
//-----------------------------------------------------------------------------
/*!
Simply adds a mesh to its mesh pointer vector of the node.
*/
void SLNode::addMesh(SLMesh* mesh)
{
    assert(mesh && "No mesh passed");

    // Take over mesh name if node name is default name
    if (_name == "Node" && mesh->name() != "Mesh")
        _name = mesh->name() + "-Node";

    _mesh = mesh;

    _isAABBUpToDate = false;
    mesh->init(this);
}
//-----------------------------------------------------------------------------
/*!
Inserts a mesh pointer in the mesh pointer vector after the
specified afterM pointer.
*/
//! Draws the single mesh
void SLNode::drawMesh(SLSceneView* sv)
{
    if (_mesh)
        _mesh->draw(sv, this);
}
//-----------------------------------------------------------------------------
//! Returns true if a mesh was assigned and set it to nullptr
bool SLNode::removeMesh()
{
    if (_mesh)
    {
        _mesh = nullptr;
        return true;
    }
    return false;
}
//-----------------------------------------------------------------------------
//! Returns true if the passed mesh was assigned and sets it to nullptr
bool SLNode::removeMesh(SLMesh* mesh)
{
    if (_mesh == mesh && mesh != nullptr)
    {
        _mesh = nullptr;
        return true;
    }
    return false;
}
//-----------------------------------------------------------------------------
/*!
Adds a child node to the children vector
*/
void SLNode::addChild(SLNode* child)
{
    assert(child && "The child pointer is null.");
    assert(this != child && "You can not add the node to itself.");
    assert(!child->parent() && "The child has already a parent.");

    _children.push_back(child);
    _isAABBUpToDate = false;
    child->parent(this);

#ifdef SL_USE_ENTITIES
    // Only add child to existing parents in entities
    if (_entityID != INT32_MIN)
    {
        SLScene::entities.addChildEntity(_entityID, SLEntity(child));
    }
#endif
}
//-----------------------------------------------------------------------------
/*!
Inserts a child node in the children vector after the afterC node.
*/
bool SLNode::insertChild(SLNode* insertC, SLNode* afterC)
{
    assert(insertC && afterC);
    assert(insertC != afterC);

    auto found = std::find(_children.begin(), _children.end(), afterC);
    if (found != _children.end())
    {
        _children.insert(found, insertC);
        insertC->parent(this);
        _isAABBUpToDate = false;
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
    for (auto& i : _children)
        delete i;
    _children.clear();

#ifdef SL_USE_ENTITIES
    SLint entityID = SLScene::entities.getEntityID(this);
    SLint parentID = SLScene::entities.getParentID(this);
    if (entityID != INT32_MIN && parentID != INT32_MIN)
    {
        SLScene::entities.deleteChildren(entityID);
    }
#endif
}
//-----------------------------------------------------------------------------
/*!
Deletes the last child in the child vector.
*/
bool SLNode::deleteChild()
{
    if (!_children.empty())
    {
        delete _children.back();
        _children.pop_back();
        _isAABBUpToDate = false;
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
    for (SLulong i = 0; i < _children.size(); ++i)
    {
        if (_children[i] == child)
        {
            _children.erase(_children.begin() + i);
            delete child;
            _isAABBUpToDate = false;
            return true;
        }
    }
    return false;
}
//-----------------------------------------------------------------------------
/*!
Searches for a child with the name 'name' and deletes it.
*/
bool SLNode::deleteChild(const SLstring& name)
{
    assert(!name.empty());
    SLNode* found = findChild<SLNode>(name);
    if (found) return deleteChild(found);
    return false;
}
//-----------------------------------------------------------------------------
/*!
Searches for all nodes that contain the provided mesh
*/
std::deque<SLNode*>
SLNode::findChildren(const SLMesh* mesh,
                     SLbool        findRecursive)
{
    std::deque<SLNode*> list;
    findChildrenHelper(mesh, list, findRecursive);

    return list;
}
//-----------------------------------------------------------------------------
/*!
Helper function of findChildren for the passed mesh pointer
*/
void SLNode::findChildrenHelper(const SLMesh*   mesh,
                                deque<SLNode*>& list,
                                SLbool          findRecursive)
{
    for (auto* child : _children)
    {
        if (child->_mesh == mesh)
            list.push_back(child);
        if (findRecursive)
            child->findChildrenHelper(mesh, list, findRecursive);
    }
}
//-----------------------------------------------------------------------------
/*!
Searches for all nodes that contain the provided mesh
*/
deque<SLNode*>
SLNode::findChildren(const SLuint drawbit,
                     SLbool       findRecursive)
{
    deque<SLNode*> list;
    findChildrenHelper(drawbit, list, findRecursive);
    return list;
}
//-----------------------------------------------------------------------------
/*!
Helper function of findChildren for passed drawing bit
*/
void SLNode::findChildrenHelper(const SLuint    drawbit,
                                deque<SLNode*>& list,
                                SLbool          findRecursive)
{
    for (auto* child : _children)
    {
        if (child->drawBits()->get(drawbit))
            list.push_back(child);
        if (findRecursive)
            child->findChildrenHelper(drawbit, list, findRecursive);
    }
}
//-----------------------------------------------------------------------------
//! remove child from vector of children. Removes false if not found, else true.
bool SLNode::removeChild(SLNode* child)
{
    assert(child);
    for (auto it = _children.begin(); it != _children.end(); ++it)
    {
        if (*it == child)
        {
            (*it)->parent(nullptr);
            _children.erase(it);
            return true;
        }
    }

    return false;
}
//-----------------------------------------------------------------------------
void SLNode::cullChildren3D(SLSceneView* sv)
{
    for (auto* child : _children)
        child->cull3DRec(sv);
}
//-----------------------------------------------------------------------------
/*!
Does the view frustum culling by checking whether the AABB is inside the 3D
cameras view frustum. The check is done in world space. If a AABB is visible
the nodes children are checked recursively.
If a node is visible its mesh material is added to the
SLSceneview::_visibleMaterials3D set and the node to the
SLMaterials::nodesVisible3D vector.
See also SLSceneView::draw3DGLAll for more details.
*/
void SLNode::cull3DRec(SLSceneView* sv)
{
    if (!this->drawBit(SL_DB_HIDDEN))
    {
        // Do frustum culling for all shapes except cameras & lights
        if (sv->doFrustumCulling() &&
            _parent != nullptr &&             // hsm4: do not frustum check the root node
            !dynamic_cast<SLCamera*>(this) && // Ghm1: Checking for typeid fails if someone adds a custom camera that inherits SLCamera
            typeid(*this) != typeid(SLLightRect) &&
            typeid(*this) != typeid(SLLightSpot) &&
            typeid(*this) != typeid(SLLightDirect))
        {
            sv->camera()->isInFrustum(&_aabb);
        }
        else
            _aabb.isVisible(true);

        // For particle system updating (Break, no update, setup to resume)
        SLParticleSystem* tempPS = dynamic_cast<SLParticleSystem*>(this->mesh());
        if (tempPS && !_aabb.isVisible())
            tempPS->setNotVisibleInFrustum();

        // Cull the group nodes recursively
        if (_aabb.isVisible())
        {
            cullChildren3D(sv);

            if (this->drawBit(SL_DB_OVERDRAW))
            {
                sv->nodesOverdrawn().push_back(this);
            }
            else
            {
                // All nodes with meshes get rendered sorted by their material
                if (this->mesh())
                {
                    sv->visibleMaterials3D().insert(this->mesh()->mat());
                    this->mesh()->mat()->nodesVisible3D().push_back(this);
                }

                // Add camera node without mesh to opaque vector for line drawing
                else if (dynamic_cast<SLCamera*>(this))
                    sv->nodesOpaque3D().push_back(this);

                // Add selected nodes without mesh to opaque vector for line drawing
                else if (this->_isSelected)
                    sv->nodesOpaque3D().push_back(this);

                // Add special text node to blended vector
                else if (typeid(*this) == typeid(SLText))
                    sv->nodesBlended3D().push_back(this);
            }
        }
    }
}
//-----------------------------------------------------------------------------
/*!
Does the 2D frustum culling. If a node is visible its mesh material is added
to the SLSceneview::_visibleMaterials2D set and the node to the
SLMaterials::nodesVisible2D vector.
See also SLSceneView::draw3DGLAll for more details.
*/
void SLNode::cull2DRec(SLSceneView* sv)
{
    // PROFILE_FUNCTION();

    _aabb.isVisible(true);

    // Cull the group nodes recursively
    for (auto* child : _children)
        child->cull2DRec(sv);

    // Add all nodes to the opaque 2D vector
    if (this->mesh())
    {
        sv->visibleMaterials2D().insert(this->mesh()->mat());
        this->mesh()->mat()->nodesVisible2D().push_back(this);
    }
    else if (typeid(*this) == typeid(SLText))
        sv->nodesBlended2D().push_back(this);
}
//-----------------------------------------------------------------------------
/*!
Updates the statistic numbers of the passed SLNodeStats struct
and calls recursively the same method for all children.
*/
void SLNode::statsRec(SLNodeStats& stats)
{
    // PROFILE_FUNCTION();

    stats.numBytes += sizeof(SLNode);
    stats.numNodes++;

    if (_children.empty())
        stats.numNodesLeaf++;
    else
        stats.numNodesGroup++;

    if (typeid(*this) == typeid(SLLightSpot)) stats.numLights++;
    if (typeid(*this) == typeid(SLLightRect)) stats.numLights++;
    if (typeid(*this) == typeid(SLLightDirect)) stats.numLights++;

    if (_mesh)
        _mesh->addStats(stats);

    for (auto* child : _children)
        child->statsRec(stats);
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
    assert(ray != nullptr);

    // Do not test hidden nodes
    if (_drawBits.get(SL_DB_HIDDEN))
        return false;

    // Do not test origin node for shadow rays
    // This restriction is not valid for objects that can shadow itself
    // if (this == ray->srcNode && ray->type == SHADOW)
    //    return false;

    // Check first AABB for intersection
    if (!_aabb.isHitInWS(ray))
        return false;

    SLbool meshWasHit = false;

    // Transform ray to object space for non-groups
    if (_mesh == nullptr)
    {
        // Special selection for cameras
        if (dynamic_cast<SLCamera*>(this) && ray->sv->camera() != this)
        {
            ray->hitNode = this;
            ray->hitMesh = nullptr;
            SLVec3f OC   = _aabb.centerWS() - ray->origin;
            ray->length  = OC.length();
            return true;
        }
    }
    else
    {
        // transform origin position to object space
        ray->originOS.set(updateAndGetWMI().multVec(ray->origin));

        // transform the direction only with the linear sub matrix
        ray->setDirOS(_wmI.mat3() * ray->dir);

        // test the mesh
        if (_mesh->hit(ray, this) && !meshWasHit)
            meshWasHit = true;

        if (ray->isShaded())
            return true;
    }

    // Test children nodes
    for (auto* child : _children)
    {
        if (child->hitRec(ray) && !meshWasHit)
            meshWasHit = true;
        if (ray->isShaded())
            return true;
    }

    return meshWasHit;
}
//-----------------------------------------------------------------------------
/*!
 Returns a deep copy of the node and its children recursively. The meshes do
 not get copied.
*/
SLNode* SLNode::copyRec()
{
    SLNode* copy = new SLNode(name());
    copy->_om.setMatrix(_om);
    copy->_depth          = _depth;
    copy->_isAABBUpToDate = _isAABBUpToDate;
    copy->_isAABBUpToDate = _isWMUpToDate;
    copy->_drawBits       = _drawBits;
    copy->_aabb           = _aabb;
    copy->_castsShadows   = _castsShadows;

    if (_animation)
        copy->_animation = new SLAnimation(*_animation);
    else
        copy->_animation = nullptr;

    if (_mesh)
        copy->addMesh(_mesh);

    for (auto* child : _children)
        copy->addChild(child->copyRec());

    return copy;
}
//-----------------------------------------------------------------------------
/*!
Sets the parent for this node and updates its depth
*/
void SLNode::parent(SLNode* p)
{
    _parent = p;

    if (_parent)
        _depth = _parent->depth() + 1;
    else
        _depth = 1;
}
//-----------------------------------------------------------------------------
/*!
Flags this node for an updateRec. This function is called
automatically if the local transform of the node or of its parent changed.
Nodes that are flagged for updating will recalculate their world transform
the next time it is requested by updateAndGetWM().
*/
void SLNode::needUpdate()
{
#ifdef SL_USE_ENTITIES
    if (_entityID != INT32_MIN)
        SLScene::entities.getEntity(_entityID)->om.setMatrix(_om);
#endif

    // stop if we reach a node that is already flagged.
    if (!_isWMUpToDate)
        return;

    _isWMUpToDate  = false;
    _isWMIUpToDate = false;

    // mark the WM of the children dirty since their parent just changed
    for (auto* child : _children)
        child->needUpdate();

    // flag AABB for an updateRec
    needAABBUpdate();
}
//-----------------------------------------------------------------------------
/*!
Flags this node for a wm updateRec. It is almost
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

    _isWMUpToDate  = false;
    _isWMIUpToDate = false;

    // mark the WM of the children dirty since their parent just changed
    for (auto* child : _children)
        child->needWMUpdate();
}
//-----------------------------------------------------------------------------
/*!
Flags this node's AABB for an updateRec. If a node
changed we need to updateRec it's world space AABB. This needs to also be propagated
up the parent chain since the AABB of a node incorporates the AABB's of child
nodes.
*/
void SLNode::needAABBUpdate()
{
    // stop if we reach a node that is already flagged.
    if (!_isAABBUpToDate)
        return;

    _isAABBUpToDate = false;

    // flag parent's for an AABB updateRec too since they need to
    // merge the child AABBs
    if (_parent)
        _parent->needAABBUpdate();
}
//-----------------------------------------------------------------------------
/*!
A helper function that updates the current _wm to reflect the local matrix.
recursively calls the updateAndGetWM of the node's parent.
@note
This function is const because it has to be called from inside the updateAndGetWM
function which has to be const. Since we only updateRec the private cache of this
class it is ok.
*/
void SLNode::updateWM() const
{
    // PROFILE_FUNCTION();

    if (_parent)
        _wm.setMatrix(_parent->updateAndGetWM() * _om);
    else
        _wm.setMatrix(_om);

    _isWMUpToDate = true;
    numWMUpdates++;
}
//-----------------------------------------------------------------------------
/*! Returns the current world matrix for this node. If the world matrix is out
 * of date it will updateRec it and return a current result.
 */
const SLMat4f& SLNode::updateAndGetWM() const
{
    if (!_isWMUpToDate)
        updateWM();

    return _wm;
}
//-----------------------------------------------------------------------------
/*! Returns the current world inverse matrix for this node. If the world matrix
 * is out of date it will updateRec it and return a current result.
 */
const SLMat4f& SLNode::updateAndGetWMI() const
{
    if (!_isWMUpToDate)
        updateWM();

    if (!_isWMIUpToDate)
    {
        _wmI.setMatrix(_wm);
        _wmI.invert();
        _isWMIUpToDate = true;
    }

    return _wmI;
}
//-----------------------------------------------------------------------------
/*! Updates the axis aligned bounding box in world space recursively.
 */
SLAABBox& SLNode::updateAABBRec(SLbool updateAlsoAABBinOS)
{
    if (_isAABBUpToDate)
        return _aabb;

    // empty the AABB (= max negative AABB)
    if (_mesh || !_children.empty())
    {
        _aabb.minWS(SLVec3f(FLT_MAX, FLT_MAX, FLT_MAX));
        _aabb.maxWS(SLVec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
    }

    // Update special case of camera because it has no mesh
    if (dynamic_cast<SLCamera*>(this))
        ((SLCamera*)this)->buildAABB(_aabb, updateAndGetWM());

    // Build or updateRec AABB of meshes & merge them to the nodes aabb in WS
    if (_mesh)
    {
        SLAABBox aabbMesh;
        _mesh->buildAABB(aabbMesh, updateAndGetWM());
        _aabb.mergeWS(aabbMesh);
    }

    // Merge children in WS recursively
    for (auto* child : _children)
        _aabb.mergeWS(child->updateAABBRec(updateAlsoAABBinOS));

    // We need min & max also in OS for the uniform grid intersection in OS
    // This is used for ray casts (picking) and raytracing.
    if (updateAlsoAABBinOS)
        _aabb.fromWStoOS(_aabb.minWS(), _aabb.maxWS(), updateAndGetWMI());

    _aabb.setCenterAndRadiusWS();

    // For visualizing the nodes' orientation we finally updateRec the axis in WS
    _aabb.updateAxisWS(updateAndGetWM());

    _isAABBUpToDate = true;
    return _aabb;
}
//-----------------------------------------------------------------------------
/*! Prints the node name with the names of the meshes recursively
 */
void SLNode::dumpRec()
{
    // dump node
    for (SLint i = 0; i < _depth; ++i)
        cout << "   ";
    cout << "Node: " << _name << endl;

    // dump meshes of node
    for (SLint i = 0; i < _depth; ++i)
        cout << "   ";
    cout << "- Mesh: " << _mesh->name();
    cout << ", " << _mesh->numI() * 3 << " tri";
    if (_mesh->mat())
        cout << ", Mat: " << _mesh->mat()->name();
    cout << endl;

    // dump children nodes
    for (auto* child : _children)
        child->dumpRec();
}
//-----------------------------------------------------------------------------
/*! Recursively sets the specified drawbit on or off. See also SLDrawBits.
 */
void SLNode::setDrawBitsRec(SLuint bit, SLbool state)
{
    _drawBits.set(bit, state);
    for (auto* child : _children)
        child->setDrawBitsRec(bit, state);
}
//-----------------------------------------------------------------------------
/*! Recursively sets the specified OpenGL primitive type.
 */
void SLNode::setPrimitiveTypeRec(SLGLPrimitiveType primitiveType)
{
    for (auto* child : _children)
        child->setPrimitiveTypeRec(primitiveType);

    _mesh->primitive(primitiveType);
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
    if (relativeTo == TS_world && _parent)
    { // transform position to local space
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
void SLNode::rotation(const SLQuat4f&  rot,
                      SLTransformSpace relativeTo)
{
    SLMat4f rotation = rot.toMat4();

    if (_parent && relativeTo == TS_world)
    {
        // get the inverse parent rotation to remove it from our current rotation
        // we want the input quaternion to absolutely set our new rotation relative
        // to the world axes
        SLMat4f parentRotInv = _parent->updateAndGetWMI();
        parentRotInv.translation(0, 0, 0);

        // set the om rotation to the inverse of the parents rotation to achieve a
        // 0, 0, 0 relative rotation in world space
        _om.rotation(0, 0, 0, 0);
        _om.setMatrix(_om * parentRotInv);
        rotate(rot, relativeTo);
    }
    else if (relativeTo == TS_parent)
    { // relative to parent, reset current rotation and just rotate again
        _om.rotation(0, 0, 0, 0);
        rotate(rot, relativeTo);
    }
    else
    {
        // in TS_Object everything is relative to our current orientation
        _om.rotation(0, 0, 0, 0);
        _om.setMatrix(_om * rotation);
    }
    needUpdate();
}
//-----------------------------------------------------------------------------
/*!
sets the rotation of this node. The axis parameter
will be transformed into 'relativeTo' space. So a passing in an axis
of (0, 1, 0) with TS_Object will rotate the node around its own up axis.
*/
void SLNode::rotation(SLfloat          angleDeg,
                      const SLVec3f&   axis,
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
        case TS_object:
            _om.translate(delta);
            break;

        case TS_world:
            if (_parent)
            {
                SLVec3f localVec = _parent->updateAndGetWMI().mat3() * delta;
                _om.translation(localVec + _om.translation());
            }
            else
                _om.translation(delta + _om.translation());
            break;

        case TS_parent:
            _om.translation(delta + _om.translation());
            break;
    }
    needUpdate();
}
//-----------------------------------------------------------------------------
/*!
Rotates the node around its local origin relative to the space expressed by 'relativeTo'.
*/
void SLNode::rotate(SLfloat          angleDeg,
                    const SLVec3f&   axis,
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
    SLMat4f rotation = rot.toMat4();

    if (relativeTo == TS_object)
    {
        _om.setMatrix(_om * rotation);
    }
    else if (_parent && relativeTo == TS_world)
    {
        SLMat4f rotWS;
        rotWS.translate(updateAndGetWM().translation());
        rotWS.multiply(rotation);
        rotWS.translate(-updateAndGetWM().translation());

        _om.setMatrix(_parent->_wm.inverted() * rotWS * updateAndGetWM());
    }
    else // relativeTo == TS_Parent || relativeTo == TS_World && !_parent
    {
        SLMat4f rotOS;
        rotOS.translate(translationOS());
        rotOS.multiply(rotation);
        rotOS.translate(-translationOS());

        _om.setMatrix(rotOS * _om);
    }

    needUpdate();
}
//-----------------------------------------------------------------------------
/*!
Rotates the node around an arbitrary point. The 'axis' and 'point' parameter
are relative to the space described by 'relativeTo'.
*/
void SLNode::rotateAround(const SLVec3f&   point,
                          SLVec3f&         axis,
                          SLfloat          angleDeg,
                          SLTransformSpace relativeTo)
{
    SLVec3f localPoint = point;
    SLVec3f localAxis  = axis;

    if (relativeTo == TS_world && _parent)
    {
        localPoint = _parent->updateAndGetWMI() * point;
        localAxis  = _parent->updateAndGetWMI().mat3() * axis;
    }

    SLMat4f rot;
    rot.translate(localPoint);
    rot.rotate(angleDeg, localAxis);
    rot.translate(-localPoint);

    if (relativeTo == TS_object)
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
isn't the possibility for shearing an object currently.
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
void SLNode::lookAt(const SLVec3f&   target,
                    const SLVec3f&   up,
                    SLTransformSpace relativeTo)
{
    SLVec3f pos = translationOS();
    SLVec3f dir;
    SLVec3f localUp = up;

    if (relativeTo == TS_world && _parent)
    {
        SLVec3f localTarget = _parent->updateAndGetWMI() * target;
        localUp             = _parent->updateAndGetWMI().mat3() * up;
        dir                 = localTarget - translationOS();
    }
    else if (relativeTo == TS_object)
        dir = _om * target - translationOS();
    else
        dir = target - translationOS();

    dir.normalize();

    SLfloat cosAngle = localUp.dot(dir);

    // dir and up are parallel and facing in the same direction
    // or facing in opposite directions.
    // in this case we just rotate the up vector by 90� around
    // our current right vector
    // @todo This check might make more sense to be in Mat3.posAtUp
    if (fabs(cosAngle - 1.0) <= FLT_EPSILON || fabs(cosAngle + 1.0) <= FLT_EPSILON)
    {
        SLMat3f rot;
        rot.rotation(-90.0f, rightOS());

        localUp = rot * localUp;
    }

    _om.posAtUp(pos, pos + dir, localUp);

    needUpdate();
}
//-----------------------------------------------------------------------------
/*!
Scales and translates the node so that its largest
dimension is maxDim and the center is in [0,0,0].
*/
void SLNode::scaleToCenter(SLfloat maxDim)
{
    _aabb = updateAABBRec(true);
    SLVec3f size(_aabb.maxWS() - _aabb.minWS());
    SLVec3f center((_aabb.maxWS() + _aabb.minWS()) * 0.5f);
    SLfloat scaleFactor = maxDim / size.maxXYZ();
    if (fabs(scaleFactor) > FLT_EPSILON)
        scale(scaleFactor);
    else
        cout << "Node can't be scaled: " << name().c_str() << endl;
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
Resets this object to its initial state
*/
void SLNode::resetToInitialState()
{
    _om.setMatrix(_initialOM);
    needUpdate();
}
//-----------------------------------------------------------------------------
//! Returns the first skeleton found in the meshes
const SLAnimSkeleton*
SLNode::skeleton()
{
    if (_mesh && _mesh->skeleton())
        return _mesh->skeleton();
    return nullptr;
}
//-----------------------------------------------------------------------------
void SLNode::updateRec()
{
    // if (_parent == nullptr) PROFILE_FUNCTION();

    // Call optional update callback of attached
    if (_onUpdateCB)
        _onUpdateCB();

    // Call doUpdate for inherited classes if implemented
    doUpdate();

    for (auto* child : _children)
        child->updateRec();
}
//-----------------------------------------------------------------------------
//! Update all skinned meshes recursively.
/*! Do software skinning on all changed skeletons && updateRec any out of date
 acceleration structure for RT or if they're being rendered.
*/
bool SLNode::updateMeshSkins(const std::function<void(SLMesh*)>& cbInformNodes)
{
    bool hasChanges = false;

    // Do software skinning on changed skeleton
    if (_mesh && _mesh->skeleton() && _mesh->skeleton()->changed())
    {
        _mesh->transformSkin(cbInformNodes);
        hasChanges = true;
    }

    for (auto* child : _children)
        hasChanges |= child->updateMeshSkins(cbInformNodes);

    return hasChanges;
}
//-----------------------------------------------------------------------------
void SLNode::updateMeshAccelStructs()
{
    PROFILE_FUNCTION();

    if (_mesh && _mesh->accelStructIsOutOfDate())
        _mesh->updateAccelStruct();

    for (auto* child : _children)
        child->updateMeshAccelStructs();
}
//-----------------------------------------------------------------------------
//! Updates the mesh material recursively with a material lambda
void SLNode::updateMeshMat(function<void(SLMaterial* m)> setMat, bool recursive)
{
    if (_mesh && _mesh->mat())
        setMat(_mesh->mat());

    if (recursive)
        for (auto* child : _children)
            child->updateMeshMat(setMat, recursive);
}
//-----------------------------------------------------------------------------
//! Set the mesh material recursively
void SLNode::setMeshMat(SLMaterial* mat, bool recursive)
{
    if (_mesh)
        _mesh->mat(mat);

    if (recursive)
        for (auto* child : _children)
            child->setMeshMat(mat, recursive);
}
//-----------------------------------------------------------------------------
#ifdef SL_HAS_OPTIX
void SLNode::createInstanceAccelerationStructureTree()
{
    vector<OptixInstance> instances;

    for (auto child : children())
    {
        if (!child->optixTraversableHandle())
        {
            child->createInstanceAccelerationStructureTree();
        }

        if (child->optixTraversableHandle())
        {
            OptixInstance instance;

            const SLMat4f mat4x4        = om();
            float         transform[12] = {mat4x4.m(0),
                                           mat4x4.m(4),
                                           mat4x4.m(8),
                                           mat4x4.m(12),
                                           mat4x4.m(1),
                                           mat4x4.m(5),
                                           mat4x4.m(9),
                                           mat4x4.m(13),
                                           mat4x4.m(2),
                                           mat4x4.m(6),
                                           mat4x4.m(10),
                                           mat4x4.m(14)};
            memcpy(instance.transform, transform, sizeof(float) * 12);

            instance.instanceId        = instanceIndex++;
            instance.visibilityMask    = 255;
            instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
            instance.traversableHandle = child->optixTraversableHandle();
            instance.sbtOffset         = 0;

            instances.push_back(instance);
        }
    }

    if (mesh())
    {
        _mesh->updateMeshAccelerationStructure();
        OptixInstance instance;

        const SLMat4f& mat4x4        = om();
        float          transform[12] = {mat4x4.m(0),
                                        mat4x4.m(4),
                                        mat4x4.m(8),
                                        mat4x4.m(12),
                                        mat4x4.m(1),
                                        mat4x4.m(5),
                                        mat4x4.m(9),
                                        mat4x4.m(13),
                                        mat4x4.m(2),
                                        mat4x4.m(6),
                                        mat4x4.m(10),
                                        mat4x4.m(14)};

        memcpy(instance.transform, transform, sizeof(float) * 12);

        instance.instanceId = instanceIndex++;
        if (_mesh->mat()->emissive().length() > 0)
        {
            instance.visibilityMask = 253;
        }
        else
        {
            instance.visibilityMask = 255;
        }
        instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle = _mesh->optixTraversableHandle();
        instance.sbtOffset         = _mesh->sbtIndex();

        instances.push_back(instance);
    }

    if (instances.empty())
    {
        return;
    }

    SLOptixCudaBuffer<OptixInstance> instanceBuffer = SLOptixCudaBuffer<OptixInstance>();
    instanceBuffer.alloc_and_upload(instances);

    _buildInput.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    _buildInput.instanceArray.instances    = instanceBuffer.devicePointer();
    _buildInput.instanceArray.numInstances = (SLuint)instances.size();

    buildAccelerationStructure();
}
//-----------------------------------------------------------------------------
void SLNode::createInstanceAccelerationStructureFlat()
{
    vector<OptixInstance> instances;

    createOptixInstances(instances);

    if (instances.empty())
    {
        return;
    }

    SLOptixCudaBuffer<OptixInstance> instanceBuffer = SLOptixCudaBuffer<OptixInstance>();
    instanceBuffer.alloc_and_upload(instances);

    _buildInput.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    _buildInput.instanceArray.instances    = instanceBuffer.devicePointer();
    _buildInput.instanceArray.numInstances = (SLuint)instances.size();

    buildAccelerationStructure();
}
//-----------------------------------------------------------------------------
void SLNode::createOptixInstances(vector<OptixInstance>& instances)
{
    for (auto child : children())
    {
        child->createOptixInstances(instances);
    }

    if (_mesh)
    {
        _mesh->updateMeshAccelerationStructure();
        OptixInstance instance;

        const SLMat4f& mat4x4        = updateAndGetWM();
        float          transform[12] = {mat4x4.m(0),
                                        mat4x4.m(4),
                                        mat4x4.m(8),
                                        mat4x4.m(12),
                                        mat4x4.m(1),
                                        mat4x4.m(5),
                                        mat4x4.m(9),
                                        mat4x4.m(13),
                                        mat4x4.m(2),
                                        mat4x4.m(6),
                                        mat4x4.m(10),
                                        mat4x4.m(14)};
        memcpy(instance.transform, transform, sizeof(float) * 12);

        instance.instanceId = instanceIndex++;
        if (_mesh->name().find("LightSpot") != -1 ||
            _mesh->name() == "line")
        {
            instance.visibilityMask = 252;
        }
        else if (_mesh->name().find("LightRect") != -1)
        {
            instance.visibilityMask = 254;
        }
        else
        {
            instance.visibilityMask = 255;
        }
        instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle = _mesh->optixTraversableHandle();
        instance.sbtOffset         = _mesh->sbtIndex();

        instances.push_back(instance);
    }
}
#endif
//-----------------------------------------------------------------------------
