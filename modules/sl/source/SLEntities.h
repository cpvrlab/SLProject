//#############################################################################
//  File:      SLEntities.h
//  Date:      June 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLENTITIES_H
#define SLENTITIES_H

#include <SLMat4.h>
#include <SLMesh.h>

using namespace std;

//#define SL_USE_ENTITIES
//#define SL_USE_ENTITIES_DEBUG

//-----------------------------------------------------------------------------
//! SLEntity is the Data Oriented Design version of a SLNode
/* This struct is an entity for a tightly packed vector without pointers for
 * the parent-child relation. This allows a scene traversal with much less
 * cache misses.
 */
struct SLEntity
{
    SLEntity(SLNode* myNode = nullptr)
      : node(myNode),
        parentID(0),
        childCount(0) {}

    SLint   parentID;   //!< ID of the parent node (-1 of no parent)
    SLuint  childCount; //!< Number of children
    SLMat4f om;         //!< Object matrix for local transforms
    SLMat4f wm;         //!< World matrix for world transform
    SLMat4f wmI;        //!< Inverse world matrix
    SLNode* node;       //!< Pointer to the corresponding SLNode instance
};
//-----------------------------------------------------------------------------
//! Vector of SLEntity
typedef vector<SLEntity> SLVEntity;
//-----------------------------------------------------------------------------
//! Scenegraph in Data Oriented Design with flat std::vector of SLEntity
class SLEntities
{
public:
    //! Adds a child into the vector nodes right after its parent
    void addChildEntity(SLint myParentID, SLEntity entity);

    //! Deletes a node at index id with all its children
    void deleteEntity(SLint id);

    //! Deletes all children of an entity with index id
    void deleteChildren(SLint id);

    //! Updates all world matrices and returns no. of updated
    SLint updateWMRec(SLint id, SLMat4f& parentWM);

    //! Returns the pointer to a node if id is valid else a nullptr
    SLEntity* getEntity(SLint id);

    //! Returns the ID of the entity with a SLNode pointer
    SLint getEntityID(SLNode* node);

    //! Returns the pointer to the parent of a node if id is valid else a nullptr
    SLEntity* getParent(SLint id);

    //! Returns the parentID of a SLNode pointer
    SLint getParentID(SLNode* node);

    //! Dump scenegraph as a flat vector or as a tree
    void dump(SLbool doTreeDump);

    //! Returns the size of the entity vector
    SLuint size() { return (SLuint)_graph.size(); }

    //! Clears the the entities vector
    void clear() { _graph.clear(); }

private:
    SLVEntity _graph; //!< Vector of SLEntity of entire scenegraph
};
//-----------------------------------------------------------------------------
#endif // SLENTITIES_H
