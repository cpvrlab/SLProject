//#############################################################################
//  File:      SLSceneNode.h
//  Author:    Marc Wacker, Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLSCENENODE_H
#define SLSCENENODE_H

#include <stdafx.h>
#include <SLEnums.h>
#include <SLNode.h>
#include <SLMesh.h>
#include <SLSceneNode.h>
#include <SLDrawBits.h>
#include <SLEventHandler.h>

class SLSceneView;
class SLRay;
class SLAABBox;
class SLSceneNode;
class SLAnimation;

//-----------------------------------------------------------------------------
//! SLVNode typdef for a vector of SLNodes
typedef std::vector<SLSceneNode*>  SLVSceneNode;
//-----------------------------------------------------------------------------
//! Struct for scene graph statistics
/*! The SLNodeStats struct holds some statistics that are set in the recursive
SLNode::statsRec method.
*/
struct SLNodeStats
{
    SLuint      numNodes;      //!< NO. of children nodes
    SLuint      numBytes;      //!< NO. of bytes allocated
    SLuint      numBytesAccel; //!< NO. of bytes in accel. structs
    SLuint      numGroupNodes; //!< NO. of group nodes
    SLuint      numLeafNodes;  //!< NO. of leaf nodes
    SLuint      numMeshes;     //!< NO. of visible shapes in node
    SLuint      numLights;     //!< NO. of lights in mesh
    SLuint      numTriangles;  //!< NO. of triangles in mesh
    SLuint      numLines;      //!< NO. of lines in mesh
    SLuint      numVoxels;     //!< NO. of voxels
    SLfloat     numVoxEmpty;   //!< NO. of empty voxels
    SLuint      numVoxMaxTria; //!< Max. no. of triangles per voxel
    SLuint      numAnimations; //!< NO. of animations

    //! Resets all counters to zero
    void clear()
    {
        numNodes       = 0;
        numBytes       = 0;
        numBytesAccel  = 0;
        numGroupNodes  = 0;
        numLeafNodes   = 0;
        numMeshes      = 0;
        numLights      = 0;
        numTriangles   = 0;
        numLines       = 0;
        numVoxels      = 0;
        numVoxEmpty    = 0.0f;
        numVoxMaxTria  = 0;
        numAnimations  = 0;
    }

    //! Prints all statistic informations on the std out stream.
    void print()
    {
        SLfloat voxelsEmpty  = numVoxels ? (SLfloat)numVoxEmpty / 
                                            (SLfloat)numVoxels*100.0f : 0;
        SLfloat avgTriPerVox = numVoxels ? (SLfloat)numTriangles / 
                                            (SLfloat)(numVoxels-numVoxEmpty) : 0;
        SL_LOG("Voxels         : %d\n", numVoxels);
        SL_LOG("Voxels empty   : %4.1f%%\n", voxelsEmpty); 
        SL_LOG("Avg. Tria/Voxel: %4.1f\n", avgTriPerVox);
        SL_LOG("Max. Tria/Voxel: %d\n", numVoxMaxTria);
        SL_LOG("MB Meshes      : %f\n", (SLfloat)numBytes / 1000000.0f);
        SL_LOG("MB Accel.      : %f\n", (SLfloat)numBytesAccel / 1000000.0f);
        SL_LOG("Group Nodes    : %d\n", numGroupNodes);
        SL_LOG("Leaf Nodes     : %d\n", numLeafNodes);
        SL_LOG("Meshes         : %d\n", numMeshes);
        SL_LOG("Triangles      : %d\n", numTriangles);
        SL_LOG("Lights         : %d\n", numLights);
        SL_LOG("\n");
    }
};
//-----------------------------------------------------------------------------
//! SLSceneNode represents a node in a hierarchical scene graph.
/*!
SLSceneNode is the most important building block of the scene graph.

A node can have 0-N children nodes in the vector _children. 
With child nodes you can build hierarchical structures. A node without meshes 
can act as parent node to group its children. A node without children only 
makes sense to hold one or more meshes for visualization. 
The pointer _parent points to the parent of a child node. 

A node can use 0-N mesh objects in the SLMesh vector _meshes for the rendering 
of triangled or lined meshes. Meshes are stored in the SLScene::_meshes vector.
Multiple nodes can point to the same mesh object.
The nodes meshes are drawn by the methods SLNode::drawMeshes and alternatively
by SLNode::drawRec.

A node can be transformed and has therefore a object matrix (_om) for its local
transform. All other matrices such as the world matrix (_wm), the inverse
world matrix (_wmI) and the normal world matrix (_wmN) are derived from the
object matrix and automatically generated and updated.

A node can be transformed by one of the various transform functions such
as translate(). Many of these functions take an additional parameter 
'relativeTo'. This parameter tells the transform function in what space
the transformation should be applied in. The available transform spaces
are:
   - TS_World: Space relative to the global world coordinate system.
   - TS_Parent: Space relative to our parent's transformation.
   - TS_Local: Space relative to our current node's origin.

A node can implement one of the eventhandlers defined in the inherited 
SLEventHandler interface.

The SLCamera is derived from the SLNode and implements a camera through which the
scene can be viewed (see also SLSceneView).
The SLLightSphere and SLLightRect are derived from SLNode and represent light
sources in the scene.
Cameras and lights can be placed in the scene because of their inheritance of 
SLNode.
*/
class SLSceneNode: public SLNode, public SLEventHandler
{
    friend class SLSceneView;

    public:
                            SLSceneNode         (SLstring name="Node");
                            SLSceneNode         (SLMesh* mesh, SLstring name="Node");
                            SLSceneNode         (const SLSceneNode& node);
    virtual                ~SLSceneNode         ();
         
            // Recursive scene traversal methods (see impl. for details)
    virtual void            cullRec             (SLSceneView* sv);
    virtual void            drawRec             (SLSceneView* sv);
    virtual bool            hitRec              (SLRay* ray);
    virtual void            statsRec            (SLNodeStats &stats);
    virtual SLbool          animateRec          (SLfloat timeMS);
    virtual SLSceneNode*    copyRec             ();
    virtual SLAABBox&       updateAABBRec       ();
    virtual void            dumpRec             ();
            void            setDrawBitsRec      (SLuint bit, SLbool state);

            // Mesh methods (see impl. for details)
            SLint           numMeshes           () {return (SLint)_meshes.size();}
            void            addMesh             (SLMesh* mesh);
            bool            insertMesh          (SLMesh* insertM, SLMesh* afterM);
            void            removeMeshes        () {_meshes.clear();}
            bool            removeMesh          ();
            bool            removeMesh          (SLMesh* mesh);
            bool            removeMesh          (SLstring name);
            SLMesh*         findMesh            (SLstring name);
            SLbool          containsMesh        (const SLMesh* mesh);
    virtual void            drawMeshes          (SLSceneView* sv);

            SLVSceneNode    findChildren        (const SLMesh* mesh,
                                                 SLbool findRecursive);
            void            findChildrenHelper  (const SLMesh* mesh,
                                                 vector<SLSceneNode*>& list,
                                                 SLbool findRecursive);

            void            scaleToCenter       (SLfloat maxDim);

            // Getters (see member)
            SLDrawBits*     drawBits            () {return &_drawBits;}
            SLbool          drawBit             (SLuint bit) {return _drawBits.get(bit);}
            SLAABBox*       aabb                () {return &_aabb;}
            SLVMesh&        meshes              () {return _meshes;}

    protected:
            SLGLState*   _stateGL;          //!< pointer to the global SLGLState instance
            SLVMesh      _meshes;           //!< vector of meshes of the node
    mutable SLbool       _isAABBUpToDate;   //!< is the saved aabb still valid
            SLDrawBits   _drawBits;         //!< node level drawing flags
            SLAABBox     _aabb;             //!< axis aligned bounding box
};
//-----------------------------------------------------------------------------

#endif // SLNODE_H
