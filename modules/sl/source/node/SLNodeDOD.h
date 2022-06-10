//#############################################################################
//  File:      SLNodeDOD.h
//  Date:      June 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLNODEDOD_H
#define SLNODEDOD_H

#include <SLMat4.h>
#include <SLMesh.h>

using namespace std;

class SLNodeDOD;

//-----------------------------------------------------------------------------
//! SLVNode typedef for a vector of SLNodes
typedef vector<SLNodeDOD> SLVNodeDOD;
//-----------------------------------------------------------------------------
//! SLNodeDOD is the Data Oriented Design version of a SLNode
/* The intension of this struct is to be an entity for a tightly packed vector
 without pointers for the parent-child relation.
 */
struct SLNodeDOD
{
    SLNodeDOD(SLMesh* myMesh = nullptr)
      : mesh(myMesh),
        parentID(0),
        childCount(0) {}

    //! Adds a child into the vector nodes right after its parent
    static void addChild(SLVNodeDOD& scenegraph,
                         SLint       myParentID,
                         SLNodeDOD   node);

    //! Deletes a node at index id with all its children
    static void deleteNode(SLVNodeDOD& scenegraph, SLint id);

    //! Returns the pointer to a node if id is valid else a nullptr
    static SLNodeDOD* getNode(SLVNodeDOD& scenegraph, SLint id);

    //! Returns the pointer to the parent of a node if id is valid else a nullptr
    static SLNodeDOD* getParent(SLVNodeDOD& scenegraph, SLint id);

    //! Dump scenegraph as a flat vector or as a tree
    static void dump(SLVNodeDOD& scenegraph, SLbool doTreeDump);

    //! test operations on SLNodeDOD
    static void test();

    SLint   parentID;   //!< ID of the parent node (-1 of no parent)
    SLuint  childCount; //!< Number of children
    SLMat4f om;         //!< Object matrix for local transforms
    SLMat4f wm;         //!< World matrix for world transform
    SLMat4f wmI;        //!< Inverse world matrix
    SLMesh* mesh;       //!< Pointer to the mesh if any
};
//-----------------------------------------------------------------------------
/*! Returns the pointer to the node at id
 @param scenegraph The scenegraph as a SLNodeDOD vector
 @param id The index of the node to return as a pointer
 @return The pointer of the node at id
 */
SLNodeDOD* SLNodeDOD::getNode(SLVNodeDOD& scenegraph, SLint id)
{
    if (id >= 0 && id < scenegraph.size())
        return &scenegraph[id];
    else
        return nullptr;
}
//-----------------------------------------------------------------------------
/*! Returns the pointer to the parent of the node at id
 @param scenegraph The scenegraph as a SLNodeDOD vector
 @param id The index of the node in the scenegraph vector
 @return The pointer of the parent of the node at id
 */
SLNodeDOD* SLNodeDOD::getParent(SLVNodeDOD& scenegraph, SLint id)
{
    if (id >= 1 && id < scenegraph.size())
        return &scenegraph[scenegraph[id].parentID];
    else
        return nullptr;
}
//-----------------------------------------------------------------------------
/*! Prints the scenegraph vector flat or as hierarchical tree as follows:
 @param scenegraph The scenegraph as a node vector
 @param doTreeDump Flag if dump as a tree or flat

 Pattern: ID.Parent.childCount
 Example from SLNodeDOD::addChild:

 00.-1.04
 +--01.00.01
 |  +--02.01.00
 +--03.00.00
 +--04.00.02
 |  +--05.04.00
 |  +--06.04.00
 +--07.00.00
 */
void SLNodeDOD::dump(SLVNodeDOD& scenegraph, SLbool doTreeDump)
{
    if (doTreeDump)
    {
        SLuint depth        = 0;
        SLint  lastParentID = -1;

        for (SLuint i = 0; i < scenegraph.size(); ++i)
        {
            if (scenegraph[i].parentID > lastParentID)
                depth++;
            else if (scenegraph[i].parentID < lastParentID)
                depth--;

            string tabs;
            for (int d = 1; d < depth; ++d)
                tabs += "|  ";
            if (depth > 0)
                tabs += "+--";
            cout << tabs;

            printf("%02u.%02d.%02u\n",
                   i,
                   scenegraph[i].parentID,
                   scenegraph[i].childCount);

            lastParentID = scenegraph[i].parentID;
        }
        cout << endl;
    }
    else
    {
        for (SLuint i = 0; i < scenegraph.size(); ++i)
            printf("|  %02u  ", i);
        cout << "|" << endl;

        for (SLuint i = 0; i < scenegraph.size(); ++i)
            cout << "-------";
        cout << "-" << endl;

        for (SLuint i = 0; i < scenegraph.size(); ++i)
            if (scenegraph[i].parentID == -1)
                printf("|-1  %02u", scenegraph[i].childCount);
            else
                printf("|%02u  %02u", scenegraph[i].parentID, scenegraph[i].childCount);
        cout << "|" << endl;
    }
    cout << endl;
}
//-----------------------------------------------------------------------------
/*! addChild adds a child node by inserting an SLNodeDOD into a vector in
 Depth First Search order. The root node gets the parent ID -1.
 The child is inserted right after the parent node.
 @param scenegraph The scenegraph as a vector
 @param myParentID Index of the parent node
 @param node The node to add as child of the parent

 Legend: The * shows the updated fields
         The ------ shows the inserted node

                     |  00  |  01  |  02  |  03  |  04  |  05  |  06  |  07  |
addChild(sg,0,node)  |-1  00|
                      ------
addChild(sg,0,node)  |-1  01|00  00|
                           * ------
addChild(sg,0,node)  |-1  02|00  00|00  00|
                           * ------
addChild(sg,0,node)  |-1  03|00  00|00  00|00  00|
                           * ------
addChild(sg,2,node)  |-1  03|00  00|00  01|02  00|00  00|
                                         * ------
addChild(sg,2,node)  |-1  03|00  00|00  02|02  00|02  00|00  00|
                                         * ------
addChild(sg,0,node)  |-1  04|00  00|00  00|00  02|03  00|03  00|00  00|
                           * ------                *      *
addChild(sg,1,node)  |-1  04|00  01|01  00|00  00|00  02|04  00|04  00|00  00|
                                  * ------                *      *
*/
void SLNodeDOD::addChild(SLVNodeDOD& scenegraph,
                         SLint       myParentID,
                         SLNodeDOD   node)
{
    assert(myParentID <= scenegraph.size() &&
           myParentID >= -1 &&
           "Invalid parentID");

    if (scenegraph.empty())
    {
        // Root node and ignore myParentID
        scenegraph.push_back(node);
        scenegraph[0].parentID   = -1;
        scenegraph[0].childCount = 0;
    }
    else
    {
        node.parentID   = myParentID;
        node.childCount = 0;
        scenegraph.insert(scenegraph.begin() + myParentID + 1, node);
        scenegraph[myParentID].childCount++;

        // Increase parentIDs of following subtrees that are greater
        for (SLuint i = myParentID + 2; i < scenegraph.size(); i++)
            if (scenegraph[i].parentID > myParentID)
                scenegraph[i].parentID++;
    }
}
//-----------------------------------------------------------------------------
/*! Deletes a node at index id with all with all children
 @param scenegraph The scenegraph as a SLNodeDOD vector
 @param id Index of node to delete
 */
void SLNodeDOD::deleteNode(SLVNodeDOD& scenegraph, SLint id)
{
    assert(id <= scenegraph.size() &&
           id >= 0 &&
           "Invalid id");

    if (id == 0)
        scenegraph.clear();
    else
    {
        // Find the next child with the same parentID
        SLuint toID;
        SLint  myParentID = scenegraph[id].parentID;
        for (toID = id + 1; toID < scenegraph.size(); toID++)
            if (scenegraph[toID].parentID == myParentID)
                break;

        // Erase the elements in the vector
        scenegraph.erase(scenegraph.begin() + id, scenegraph.begin() + toID);
        scenegraph[myParentID].childCount--;

        // Decrease parentIDs of following subtrees that are greater
        SLuint numNodesToErase = toID - id;
        for (SLuint i = id; i < scenegraph.size(); i++)
            if (scenegraph[i].parentID > myParentID)
                scenegraph[i].parentID = scenegraph[i].parentID - numNodesToErase;
    }
}
//-----------------------------------------------------------------------------
void SLNodeDOD::test()
{
    SLVNodeDOD nodes;
    SLNodeDOD::addChild(nodes, 0, SLNodeDOD()); // Root node
    SLNodeDOD::addChild(nodes, 0, SLNodeDOD());
    SLNodeDOD::addChild(nodes, 0, SLNodeDOD());
    SLNodeDOD::addChild(nodes, 0, SLNodeDOD());
    SLNodeDOD::addChild(nodes, 2, SLNodeDOD());
    SLNodeDOD::addChild(nodes, 2, SLNodeDOD());
    SLNodeDOD::addChild(nodes, 0, SLNodeDOD());
    SLNodeDOD::addChild(nodes, 1, SLNodeDOD());
    SLNodeDOD::addChild(nodes, 5, SLNodeDOD());
    SLNodeDOD::dump(nodes, false);
    SLNodeDOD::dump(nodes, true);

    SLNodeDOD::deleteNode(nodes, 1);
    SLNodeDOD::dump(nodes, false);
    SLNodeDOD::dump(nodes, true);

    SLNodeDOD::deleteNode(nodes, 6);
    SLNodeDOD::dump(nodes, false);
    SLNodeDOD::dump(nodes, true);

    SLNodeDOD::deleteNode(nodes, 2);
    SLNodeDOD::dump(nodes, false);
    SLNodeDOD::dump(nodes, true);

    SLNodeDOD::deleteNode(nodes, 0);
    SLNodeDOD::dump(nodes, false);
    SLNodeDOD::dump(nodes, true);
}
//-----------------------------------------------------------------------------
#endif // SLNODEDOD_H
