//#############################################################################
//  File:      SLSceneDOD.h
//  Date:      June 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLSCENEDOD_H
#define SLSCENEDOD_H

#include <SLMat4.h>
#include <SLMesh.h>

using namespace std;

//#define SL_TEST_SCENE_DOD

//-----------------------------------------------------------------------------
//! SLNodeDOD is the Data Oriented Design version of a SLNode
/* This struct is an entity for a tightly packed vector without pointers for
 * the parent-child relation.
 */
struct SLNodeDOD
{
    SLNodeDOD(SLNode* myNode = nullptr, SLMesh* myMesh = nullptr)
      : node(myNode),
        mesh(myMesh),
        parentID(0),
        childCount(0) {}

    SLint   parentID;   //!< ID of the parent node (-1 of no parent)
    SLuint  childCount; //!< Number of children
    SLMat4f om;         //!< Object matrix for local transforms
    SLMat4f wm;         //!< World matrix for world transform
    SLMat4f wmI;        //!< Inverse world matrix
    SLMesh* mesh;       //!< Pointer to the mesh if any
    SLNode* node;       //!< Pointer to the corresponding SLNode instance
};
//-----------------------------------------------------------------------------
//! SLVNode typedef for a vector of SLNodes
typedef vector<SLNodeDOD> SLVNodeDOD;
//-----------------------------------------------------------------------------
//! Scenegraph in Data Oriented Design with flat std::vector of SLNodeDOD
class SLSceneDOD
{
public:
    //! Adds a child into the vector nodes right after its parent
    SLint addChild(SLint     myParentID,
                  SLNodeDOD node);

    //! Deletes a node at index id with all its children
    void deleteNode(SLint id);

    //! Updates all world matrices
    void updateWM(SLint id, SLMat4f& parentWM);

    //! Returns the pointer to a node if id is valid else a nullptr
    SLNodeDOD* getNode(SLint id);

    //! Returns the pointer to the parent of a node if id is valid else a nullptr
    SLNodeDOD* getParent(SLint id);

    //! Dump scenegraph as a flat vector or as a tree
    void dump(SLbool doTreeDump);

    //! test operations on SLSceneDOD
    void test();

    SLuint size () { return _graph.size(); }

private:
    SLVNodeDOD _graph;  //!< Vector of SLNodeDOD of entire scenegraph
};
//-----------------------------------------------------------------------------
#endif // SLSCENEDOD_H
