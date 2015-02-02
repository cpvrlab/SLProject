//#############################################################################
//  File:      SLNode.h
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
#include <SLNode.h>

class SLSceneNode;

//-----------------------------------------------------------------------------
//! SLVNode typdef for a vector of SLNodes
typedef std::vector<SLSceneNode*>  SLVSceneNode;
//-----------------------------------------------------------------------------
//! Struct for scene graph statistics
/*! The SLNodeStats struct holds some statistics that are set in the recursive
SLNode::statsRec method.
*/
struct SLSceneNodeStats
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
/// @todo switch to this node for the scene graph and move the functionality that is specific to the scenenode
///         in here. A base SLNode doesn't need to hold meshes.
///         Advantages of this approach are: The concrete implementations like SLSceneNode and SLJoint
///         can provide concrete addChild and createChild implementations that use their type.

//-----------------------------------------------------------------------------
//! SLSceneNode represents a node in the scene graph and is a specialization of SLNode.
/*!

*/
class SLSceneNode : public SLNode
{
    public:
                            SLSceneNode              (SLstring name="Node");
                            SLSceneNode              (SLMesh* mesh, SLstring name="Node");
                            SLSceneNode              (const SLSceneNode& node);
    virtual                ~SLSceneNode              ();
         
};

#endif // SLNODE_H
