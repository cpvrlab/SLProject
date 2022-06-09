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
 * without pointers for the parent-child relation.
 */
struct SLNodeDOD
{
    SLNodeDOD(SLMesh* myMesh = nullptr)
      : mesh(myMesh),
        parentID(0),
        childCount(0) {}

    //! Adds a child into the vector nodes right after its parent
    static void addChild(SLVNodeDOD& nodes,
                         SLint       myParentID,
                         SLNodeDOD   node);

    //! Dump scenegraph as a flat vector or as a tree
    static void dump(SLVNodeDOD& nodes, SLbool doTreeDump);

    SLint   parentID;   //!< ID of the parent node (-1 of no parent)
    SLuint  childCount; //!< Number of children
    SLMat4f om;         //!< Object matrix for local transforms
    SLMat4f wm;         //!< World matrix for world transform
    SLMat4f wmI;        //!< Inverse world matrix
    SLMesh* mesh;       //!< Pointer to the mesh if any

};
//-----------------------------------------------------------------------------
/*! Prints the scenegraph vector flat or as hierarchical tree as follows:
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
void SLNodeDOD::dump(SLVNodeDOD& nodes, SLbool doTreeDump)
{
    if (doTreeDump)
    {
        SLuint depth        = 0;
        SLint lastParentID = -1;

        for (SLuint i = 0; i < nodes.size(); ++i)
        {
            if (nodes[i].parentID > lastParentID)
                depth++;
            else if (nodes[i].parentID < lastParentID)
                depth--;

            string tabs;
            for (int d = 1; d < depth; ++d)
                tabs += "|  ";
            if (depth > 0)
                tabs += "+--";
            cout << tabs;

            printf("%02u.%02d.%02u\n",
                   i,
                   nodes[i].parentID,
                   nodes[i].childCount);

            lastParentID = nodes[i].parentID;
        }
    }
    else
    {
        for (SLuint i = 0; i < nodes.size(); ++i)
            printf("|  %02u  ", i);
        cout << "|" << endl;

        for (SLuint i = 0; i < nodes.size(); ++i)
            cout << "-------";
        cout << "-" << endl;

        for (SLuint i = 0; i < nodes.size(); ++i)
            if (nodes[i].parentID == -1)
                printf("|-1  %02u", nodes[i].childCount);
            else
                printf("|%02u  %02u", nodes[i].parentID, nodes[i].childCount);
        cout << "|" << endl;
    }
}
//-----------------------------------------------------------------------------
/*!
 addChild adds a child node by inserting an SLNodeDOD into a vector in
 Depth First Search order. The root node gets the parent ID -1.
 The child is inserted right after the parent node.
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

        // Increase parentIDs of following subtrees
        for (SLuint i = myParentID + 2; i < scenegraph.size(); i++)
            if (scenegraph[i].parentID > myParentID)
                scenegraph[i].parentID++;
    }
}
//-----------------------------------------------------------------------------
#endif // SLNODEDOD_H
