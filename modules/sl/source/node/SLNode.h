//#############################################################################
//  File:      SLNode.h
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marc Wacker, Marcus Hudritsch, Jan Dellsperger
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLNODE_H
#define SLNODE_H

#include <SLDrawBits.h>
#include <SLEnums.h>
#include <SLEventHandler.h>
#include <SLMesh.h>
#include <SLQuat4.h>
#include <deque>

using std::deque;

class SLSceneView;
class SLRay;
class SLAABBox;
class SLNode;
class SLAnimation;

//-----------------------------------------------------------------------------
//! SLVNode typedef for a vector of SLNodes
typedef deque<SLNode*> SLVNode;
//-----------------------------------------------------------------------------
//! Struct for scene graph statistics
/*! The SLNodeStats struct holds some statistics that are set in the recursive
SLNode::statsRec method.
*/
struct SLNodeStats
{
    SLuint  numNodes;        //!< NO. of children nodes
    SLuint  numBytes;        //!< NO. of bytes allocated
    SLuint  numBytesAccel;   //!< NO. of bytes in accel. structs
    SLuint  numNodesGroup;   //!< NO. of group nodes
    SLuint  numNodesLeaf;    //!< NO. of leaf nodes
    SLuint  numNodesOpaque;  //!< NO. of visible opaque nodes
    SLuint  numNodesBlended; //!< NO. of visible blended nodes
    SLuint  numMeshes;       //!< NO. of meshes in node
    SLuint  numLights;       //!< NO. of lights in mesh
    SLuint  numTriangles;    //!< NO. of triangles in mesh
    SLuint  numLines;        //!< NO. of lines in mesh
    SLuint  numVoxels;       //!< NO. of voxels
    SLfloat numVoxEmpty;     //!< NO. of empty voxels
    SLuint  numVoxMaxTria;   //!< Max. no. of triangles per voxel
    SLuint  numAnimations;   //!< NO. of animations

    //! Resets all counters to zero
    void clear()
    {
        numNodes      = 0;
        numBytes      = 0;
        numBytesAccel = 0;
        numNodesGroup = 0;
        numNodesLeaf  = 0;
        numMeshes     = 0;
        numLights     = 0;
        numTriangles  = 0;
        numLines      = 0;
        numVoxels     = 0;
        numVoxEmpty   = 0.0f;
        numVoxMaxTria = 0;
        numAnimations = 0;
    }

    //! Prints all statistic informations on the std out stream.
    void print() const
    {
        SLfloat voxelsEmpty  = numVoxels ? (SLfloat)numVoxEmpty /
                                            (SLfloat)numVoxels * 100.0f
                                         : 0;
        SLfloat avgTriPerVox = numVoxels ? (SLfloat)numTriangles /
                                             (SLfloat)(numVoxels - numVoxEmpty)
                                         : 0;
        SL_LOG("Voxels         : %d", numVoxels);
        SL_LOG("Voxels empty   : %4.1f%%", voxelsEmpty);
        SL_LOG("Avg. Tria/Voxel: %4.1f", avgTriPerVox);
        SL_LOG("Max. Tria/Voxel: %d", numVoxMaxTria);
        SL_LOG("MB Meshes      : %f", (SLfloat)numBytes / 1000000.0f);
        SL_LOG("MB Accel.      : %f", (SLfloat)numBytesAccel / 1000000.0f);
        SL_LOG("Group Nodes    : %d", numNodesGroup);
        SL_LOG("Leaf Nodes     : %d", numNodesLeaf);
        SL_LOG("Meshes         : %d", numMeshes);
        SL_LOG("Triangles      : %d", numTriangles);
        SL_LOG("Lights         : %d\n", numLights);
    }
};
//-----------------------------------------------------------------------------
//! SLNode represents a node in a hierarchical scene graph.
/*!
 * SLNode is the most important building block of the scene graph.
 * A node can have 0-N children nodes in the vector _children. With child
 * nodes you can build hierarchical structures. A node without a mesh can act
 * as parent node to group its children. A node without children only makes
 * sense to hold a mesh for visualization. The pointer _parent points to the
 * parent of a child node. \n\n
 *
 * A node can point to a single SLMesh object for the rendering of triangles
 * lines or points meshes. Meshes are stored in the SLAssetManager::_meshes
 * vector. Multiple nodes can point to the same mesh object. The node is
 * therefore not the owner of the meshes and does not delete them. The mesh
 * is drawn by the methods SLNode::drawMesh and alternatively by
 * SLNode::drawRec.\n\n
 *
 * A node can be transformed and has therefore a object matrix (_om) for its
 * local transform. All other matrices such as the world matrix (_wm), the
 * inverse world matrix (_wmI) are derived from the object matrix and
 * automatically generated and updated.\n\n
 *
 * A node can be transformed by one of the various transform functions such
 * as translate(). Many of these functions take an additional parameter
 * 'relativeTo'. This parameter tells the transform function in what space
 * the transformation should be applied in. The available transform spaces
 * are:\n
 * - TS_World: Space relative to the global world coordinate system.
 * - TS_Parent: Space relative to our parent's transformation.
 * - TS_Object: Space relative to our current node's origin.
 *
 * A node can implement one of the event handlers defined in the inherited
 * SLEventHandler interface. There is special node called SLTransformNode
 * that acts as a visual gizmo for editing the transform. See the example
 * in the menu Edit of the SLProject demo app.\n\n
 *
 * The SLCamera is derived from the SLNode and implements a camera through
 * which the scene can be viewed (see also SLSceneView). The SLLightSpot
 * and SLLightRect are derived from SLNode and represent light sources in the
 * scene. Cameras and lights can be placed in the scene because of their
 * inheritance of SLNode.\n
 */
class SLNode
  : public SLObject
  , public SLEventHandler
#ifdef SL_HAS_OPTIX
  , public SLOptixAccelStruct
#endif
{
    friend class SLSceneView;

public:
    explicit SLNode(const SLstring& name = "Node");
    explicit SLNode(SLMesh* mesh, const SLstring& name = "Node");
    explicit SLNode(SLMesh* mesh, const SLVec3f& translation, const SLstring& name);
    ~SLNode() override;

    // Recursive scene traversal methods (see impl. for details)
    virtual void      cull3DRec(SLSceneView* sv);
    virtual void      cullChildren3D(SLSceneView* sv);
    virtual void      cull2DRec(SLSceneView* sv);
    virtual bool      hitRec(SLRay* ray);
    virtual void      statsRec(SLNodeStats& stats);
    virtual SLNode*   copyRec();
    virtual SLAABBox& updateAABBRec(SLbool updateAlsoAABBinOS);
    virtual void      dumpRec();
    void              setDrawBitsRec(SLuint bit, SLbool state);
    void              setPrimitiveTypeRec(SLGLPrimitiveType primitiveType);

    // Mesh methods (see impl. for details)
    virtual void addMesh(SLMesh* mesh);
    virtual void drawMesh(SLSceneView* sv);
    bool         removeMesh();
    bool         removeMesh(SLMesh* mesh);

    // Children methods (see impl. for details)
    SLint numChildren() { return (SLint)_children.size(); }
    void  addChild(SLNode* child);
    bool  insertChild(SLNode* insertC, SLNode* afterC);
    void  deleteChildren();
    bool  deleteChild();
    bool  deleteChild(SLNode* child);
    bool  deleteChild(const SLstring& name);
    bool  removeChild(SLNode* child);

    template<typename T>
    T* find(const SLstring& name          = "",
            SLbool          findRecursive = true);
    template<typename T>
    T* findChild(const SLstring& name          = "",
                 SLbool          findRecursive = true);
    template<typename T>
    deque<T*>      findChildren(const SLstring& name          = "",
                                SLbool          findRecursive = true,
                                SLbool          canContain    = false);
    deque<SLNode*> findChildren(const SLMesh* mesh,
                                SLbool        findRecursive = true);
    deque<SLNode*> findChildren(SLuint drawbit,
                                SLbool findRecursive = true);

    // local direction getter functions
    SLVec3f translationOS() const;
    SLVec3f forwardOS() const;
    SLVec3f rightOS() const;
    SLVec3f upOS() const;
    SLVec3f axisXOS() const;
    SLVec3f axisYOS() const;
    SLVec3f axisZOS() const;
    SLVec3f translationWS() const;
    SLVec3f forwardWS() const;
    SLVec3f rightWS() const;
    SLVec3f upWS() const;
    SLVec3f axisXWS() const;
    SLVec3f axisYWS() const;
    SLVec3f axisZWS() const;

    // transform setter methods
    void translation(const SLVec3f&   pos,
                     SLTransformSpace relativeTo = TS_parent);
    void translation(SLfloat          x,
                     SLfloat          y,
                     SLfloat          z,
                     SLTransformSpace relativeTo = TS_parent);
    void rotation(const SLQuat4f&  rot,
                  SLTransformSpace relativeTo = TS_parent);
    void rotation(SLfloat          angleDeg,
                  const SLVec3f&   axis,
                  SLTransformSpace relativeTo = TS_parent);
    void scaling(SLfloat s);
    void scaling(SLfloat x,
                 SLfloat y,
                 SLfloat z);
    void scaling(const SLVec3f& scaling);
    void lookAt(SLfloat          targetX,
                SLfloat          targetY,
                SLfloat          targetZ,
                SLfloat          upX        = 0,
                SLfloat          upY        = 1,
                SLfloat          upZ        = 0,
                SLTransformSpace relativeTo = TS_world);
    void lookAt(const SLVec3f&   target,
                const SLVec3f&   up         = SLVec3f::AXISY,
                SLTransformSpace relativeTo = TS_world);

    // transform modifier methods
    void translate(const SLVec3f&   vec,
                   SLTransformSpace relativeTo = TS_object);
    void translate(SLfloat          x,
                   SLfloat          y,
                   SLfloat          z,
                   SLTransformSpace relativeTo = TS_object);
    void rotate(const SLQuat4f&  rot,
                SLTransformSpace relativeTo = TS_object);
    void rotate(SLfloat          angleDeg,
                const SLVec3f&   axis,
                SLTransformSpace relativeTo = TS_object);
    void rotate(SLfloat          angleDeg,
                SLfloat          x,
                SLfloat          y,
                SLfloat          z,
                SLTransformSpace relativeTo = TS_object);
    void rotateAround(const SLVec3f&   point,
                      SLVec3f&         axis,
                      SLfloat          angleDeg,
                      SLTransformSpace relativeTo = TS_world);
    void scale(SLfloat s);
    void scale(SLfloat x, SLfloat y, SLfloat z);
    void scale(const SLVec3f& scale);

    // Misc.
    void scaleToCenter(SLfloat maxDim);
    void setInitialState();
    void resetToInitialState();

    // Setters (see also members)
    void parent(SLNode* p);
    void entityID(SLint entityID) { _entityID = entityID; }
    void om(const SLMat4f& mat)
    {
        _om.setMatrix(mat);
        needUpdate();
    }
    void         animation(SLAnimation* a) { _animation = a; }
    void         castsShadows(SLbool castsShadows) { _castsShadows = castsShadows; }
    virtual void needUpdate();
    void         needWMUpdate();
    void         needAABBUpdate();
    void         isSelected(bool isSelected) { _isSelected = isSelected; }
    void         minLodCoverage(SLfloat minLodCoverage) { _minLodCoverage = minLodCoverage; }
    void         levelForSM(SLubyte lfsm) { _levelForSM = lfsm; }
    void         onUpdateCB(function<void()> callbackFunc) { _onUpdateCB = callbackFunc; }

    // Getters (see also member)
    SLNode*               parent() { return _parent; }
    SLint                 depth() const { return _depth; }
    SLint                 entityID() const { return _entityID; }
    const SLMat4f&        om() { return _om; }
    const SLMat4f&        initialOM() { return _initialOM; }
    const SLMat4f&        updateAndGetWM() const;
    const SLMat4f&        updateAndGetWMI() const;
    SLDrawBits*           drawBits() { return &_drawBits; }
    SLbool                drawBit(SLuint bit) { return _drawBits.get(bit); }
    SLAABBox*             aabb() { return &_aabb; }
    SLAnimation*          animation() { return _animation; }
    SLbool                castsShadows() { return _castsShadows; }
    SLMesh*               mesh() { return _mesh; }
    SLVNode&              children() { return _children; }
    const SLAnimSkeleton* skeleton();
    void                  updateRec();
    virtual void          doUpdate() {}
    bool                  updateMeshSkins(const std::function<void(SLMesh*)>& cbInformNodes);
    void                  updateMeshAccelStructs();
    void                  updateMeshMat(std::function<void(SLMaterial* m)> setMat,
                                        bool                               recursive);
    void                  setMeshMat(SLMaterial* mat, bool recursive);
    bool                  isSelected() { return _isSelected; }
    SLfloat               minLodCoverage() { return _minLodCoverage; }
    SLubyte               levelForSM() { return _levelForSM; }

    static SLuint numWMUpdates;        //!< NO. of calls to updateWMRec per frame

    static unsigned int instanceIndex; //!< ???

#ifdef SL_HAS_OPTIX
    void createInstanceAccelerationStructureTree();
    void createInstanceAccelerationStructureFlat();
    void createOptixInstances(vector<OptixInstance>&);
#endif

private:
    void updateWM() const;
    template<typename T>
    void findChildrenHelper(const SLstring& name,
                            deque<T*>&      list,
                            SLbool          findRecursive,
                            SLbool          canContain = false);
    void findChildrenHelper(const SLMesh*   mesh,
                            deque<SLNode*>& list,
                            SLbool          findRecursive);
    void findChildrenHelper(SLuint          drawbit,
                            deque<SLNode*>& list,
                            SLbool          findRecursive);

protected:
    SLNode* _parent;                  //!< pointer to the parent node
    SLVNode _children;                //!< vector of children nodes
    SLMesh* _mesh;                    //!< pointer to a single mesh

    SLint            _depth;          //!< depth of the node in a scene tree
    SLint            _entityID;       //!< ID in the SLVEntity graph for Data Oriented Design
    SLMat4f          _om;             //!< object matrix for local transforms
    SLMat4f          _initialOM;      //!< the initial om state
    mutable SLMat4f  _wm;             //!< world matrix for world transform
    mutable SLMat4f  _wmI;            //!< inverse world matrix
    mutable SLbool   _isWMUpToDate;   //!< is the WM of this node still valid
    mutable SLbool   _isWMIUpToDate;  //!< is the inverse WM of this node still valid
    mutable SLbool   _isAABBUpToDate; //!< is the saved aabb still valid
    bool             _castsShadows;   //!< flag if meshes of node should cast shadows
    bool             _isSelected;     //!< flag if node and one or more of its meshes are selected
    SLDrawBits       _drawBits;       //!< node level drawing flags
    SLAABBox         _aabb;           //!< axis aligned bounding box
    SLAnimation*     _animation;      //!< animation of the node
    SLfloat          _minLodCoverage; //!< Min. LOD coverage for visibility (0.0 < _minLodCoverage < 1.0)
    SLubyte          _levelForSM;     //!< Level of LOD to use for shadow mapping (0 = the visible one will be drawn)
    function<void()> _onUpdateCB;     //!< Optional lambda callback once per update
};

////////////////////////
// TEMPLATE FUNCTIONS //
////////////////////////

//-----------------------------------------------------------------------------
/*!
SLNode::findChild<T> searches a node tree including the node by a given name.
*/
template<typename T>
T* SLNode::find(const SLstring& name, SLbool findRecursive)
{
    T* found = dynamic_cast<T*>(this);
    if (found && (name.empty() || name == _name))
        return found;
    return findChild<T>(name, findRecursive);
}
//-----------------------------------------------------------------------------
/*!
SLNode::findChild<T> finds the first child that is of type T or a subclass of T.
*/
template<typename T>
T* SLNode::findChild(const SLstring& name, SLbool findRecursive)
{
    for (auto node : _children)
    {
        T* found = dynamic_cast<T*>(node);
        if (found && (name.empty() || name == node->name()))
            return found;
    }
    if (findRecursive)
    {
        for (auto& i : _children)
        {
            T* found = i->findChild<T>(name, findRecursive);
            if (found)
                return found;
        }
    }

    return nullptr;
}
//-----------------------------------------------------------------------------

/*!
SLNode::findChildren<T> finds a list of all children that are of type T or
subclasses of T. If a name is specified only nodes with that name are included.
*/
template<typename T>
deque<T*>
SLNode::findChildren(const SLstring& name,
                     SLbool          findRecursive,
                     SLbool          canContain)
{
    deque<T*> list;
    findChildrenHelper<T>(name, list, findRecursive);
    return list;
}
//-----------------------------------------------------------------------------

/*!
SLNode::findChildrenHelper<T> is the helper function for findChildren<T>.
It appends all newly found children to 'list'.
*/
template<typename T>
void SLNode::findChildrenHelper(const SLstring& name,
                                deque<T*>&      list,
                                SLbool          findRecursive,
                                SLbool          canContain)
{
    for (auto& i : _children)
    {
        SLNode* node  = i;
        T*      found = dynamic_cast<T*>(node);

        if (canContain)
        {
            if (found && (name.empty() ||
                          Utils::containsString(node->name(), name)))
                list.push_back(found);
        }
        else
        {
            if (found && (name.empty() || name == node->name()))
                list.push_back(found);
        }

        if (findRecursive)
            i->findChildrenHelper<T>(name, list, findRecursive);
    }
}
//-----------------------------------------------------------------------------

//////////////////////
// INLINE FUNCTIONS //
//////////////////////

//-----------------------------------------------------------------------------
/*!
SLNode::position returns current local position
*/
inline SLVec3f
SLNode::translationOS() const
{
    return _om.translation();
}
//-----------------------------------------------------------------------------
/*!
Returns the local forward vector (= -z-axis) in a right-hand y-up system
*/
inline SLVec3f
SLNode::forwardOS() const
{
    return SLVec3f(-_om.m(8), -_om.m(9), -_om.m(10));
}
//-----------------------------------------------------------------------------
/*!
Returns the local right vector (= x-axis) in a right-hand y-up system
*/
inline SLVec3f
SLNode::rightOS() const
{
    return SLVec3f(_om.m(0), _om.m(1), _om.m(2));
}
//-----------------------------------------------------------------------------
/*!
Returns the local up vector in a right-hand y-up system
*/
inline SLVec3f
SLNode::upOS() const
{
    return SLVec3f(_om.m(4), _om.m(5), _om.m(6));
}
//-----------------------------------------------------------------------------
/*!
Returns the x-axis in object space
*/
inline SLVec3f
SLNode::axisXOS() const
{
    return _om.axisX();
}
//-----------------------------------------------------------------------------
/*!
Returns the y-axis in object space
*/
inline SLVec3f
SLNode::axisYOS() const
{
    return _om.axisY();
}
//-----------------------------------------------------------------------------
/*!
Returns the z-axis in object space
*/
inline SLVec3f
SLNode::axisZOS() const
{
    return _om.axisZ();
}
//-----------------------------------------------------------------------------
/*!
SLNode::position returns current world position
*/
inline SLVec3f
SLNode::translationWS() const
{
    updateAndGetWM();
    return _wm.translation();
}
//-----------------------------------------------------------------------------
/*!
SLNode::forward returns worlds forward vector
*/
inline SLVec3f
SLNode::forwardWS() const
{
    updateAndGetWM();
    return -_wm.axisZ();
}
//-----------------------------------------------------------------------------
/*!
SLNode::right returns worlds right vector
*/
inline SLVec3f
SLNode::rightWS() const
{
    updateAndGetWM();
    return _wm.axisX();
}
//-----------------------------------------------------------------------------
/*!
SLNode::up returns worlds up vector
*/
inline SLVec3f
SLNode::upWS() const
{
    updateAndGetWM();
    return _wm.axisY();
}
//-----------------------------------------------------------------------------
/*!
Returns the x-axis in world space
*/
inline SLVec3f
SLNode::axisXWS() const
{
    updateAndGetWM();
    return _wm.axisX();
}
//-----------------------------------------------------------------------------
/*!
Returns the y-axis in world space
*/
inline SLVec3f
SLNode::axisYWS() const
{
    updateAndGetWM();
    return _wm.axisY();
}
//-----------------------------------------------------------------------------
/*!
Returns the z-axis in world space
*/
inline SLVec3f
SLNode::axisZWS() const
{
    updateAndGetWM();
    return _wm.axisZ();
}
//-----------------------------------------------------------------------------
inline void
SLNode::translation(SLfloat          x,
                    SLfloat          y,
                    SLfloat          z,
                    SLTransformSpace relativeTo)
{
    translation(SLVec3f(x, y, z), relativeTo);
}
//-----------------------------------------------------------------------------
inline void
SLNode::scaling(SLfloat s)
{
    scaling(SLVec3f(s, s, s));
}
//-----------------------------------------------------------------------------
inline void
SLNode::scaling(SLfloat x,
                SLfloat y,
                SLfloat z)
{
    scaling(SLVec3f(x, y, z));
}
//-----------------------------------------------------------------------------
inline void
SLNode::translate(SLfloat          x,
                  SLfloat          y,
                  SLfloat          z,
                  SLTransformSpace relativeTo)
{
    translate(SLVec3f(x, y, z), relativeTo);
}
//-----------------------------------------------------------------------------
inline void
SLNode::rotate(SLfloat          angleDeg,
               SLfloat          x,
               SLfloat          y,
               SLfloat          z,
               SLTransformSpace relativeTo)
{
    rotate(angleDeg, SLVec3f(x, y, z), relativeTo);
}
//-----------------------------------------------------------------------------
inline void
SLNode::scale(SLfloat s)
{
    scale(SLVec3f(s, s, s));
}
//-----------------------------------------------------------------------------
inline void
SLNode::scale(SLfloat x, SLfloat y, SLfloat z)
{
    scale(SLVec3f(x, y, z));
}
//-----------------------------------------------------------------------------
inline void
SLNode::lookAt(SLfloat          targetX,
               SLfloat          targetY,
               SLfloat          targetZ,
               SLfloat          upX,
               SLfloat          upY,
               SLfloat          upZ,
               SLTransformSpace relativeTo)
{
    lookAt(SLVec3f(targetX,
                   targetY,
                   targetZ),
           SLVec3f(upX, upY, upZ),
           relativeTo);
}
//-----------------------------------------------------------------------------

#endif
