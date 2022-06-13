//#############################################################################
//  File:      SLSceneDOD.cpp
//  Date:      June 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLSceneDOD.h>
#include <SLNode.h>

//-----------------------------------------------------------------------------
/*! Returns the pointer to the node at id
 @param scenegraph The scenegraph as a SLSceneDOD vector
 @param id The index of the node to return as a pointer
 @return The pointer of the node at id
 */
SLNodeDOD* SLSceneDOD::getNode(SLint id)
{
    if (id >= 0 && id < _graph.size())
        return &_graph[id];
    else
        return nullptr;
}
//-----------------------------------------------------------------------------
/*! Returns the pointer to the parent of the node at id
 @param scenegraph The scenegraph as a SLSceneDOD vector
 @param id The index of the node in the scenegraph vector
 @return The pointer of the parent of the node at id
 */
SLNodeDOD* SLSceneDOD::getParent(SLint id)
{
    if (id >= 1 && id < _graph.size())
        return &_graph[_graph[id].parentID];
    else
        return nullptr;
}
//-----------------------------------------------------------------------------
/*! Updates the world matrix recursively
 * @param id Index of the current node to update
 * @param parentWM World transform matrix of the parent
 */
void SLSceneDOD::updateWM(SLint id, SLMat4f& parentWM)
{
    SLNodeDOD* nodeDOD = getNode(id);

    nodeDOD->om = nodeDOD->node->om();
    SLMat4f nodeWM = nodeDOD->node->updateAndGetWM();

    nodeDOD->wm.setMatrix(parentWM * nodeDOD->om);
    nodeDOD->wmI.setMatrix(nodeDOD->wm.inverted());

    if (!nodeDOD->wm.isEqual(nodeWM))
    {
        nodeDOD->wm.print("wmDOD:");
        nodeDOD->node->updateAndGetWM().print("nodeWM:");
    }

    for(SLint i = 0; i < nodeDOD->childCount; ++i)
    {
        SLint childID = id + i + 1;
        updateWM(childID, nodeDOD->wm);
    }
}
//-----------------------------------------------------------------------------
/*! Prints the scenegraph vector flat or as hierarchical tree as follows:
 @param scenegraph The scenegraph as a node vector
 @param doTreeDump Flag if dump as a tree or flat

Pattern: ID.Parent.childCount
Example from SLSceneDOD::addChild:
00.-1.04
+--01.00.01
|  +--02.01.00
+--03.00.00
+--04.00.02
|  +--05.04.00
|  +--06.04.00
+--07.00.00
*/
void SLSceneDOD::dump(SLbool doTreeDump)
{
    if (doTreeDump)
    {
        for (SLuint i = 0; i < _graph.size(); ++i)
        {
            // Calculate depth
            SLuint depth        = 0;
            SLint  myParentID = _graph[i].parentID;
            while(myParentID != -1)
            {
                depth++;
                myParentID = _graph[myParentID].parentID;
            }

            string tabs;
            for (int d = 1; d < depth; ++d)
                tabs += "|  ";
            if (depth > 0)
                tabs += "+--";
            cout << tabs;

            SLstring nodeStr = _graph[i].node ? _graph[i].node->name() : "";
            printf("%02u.%02d.%02u-%s\n",
                   i,
                   _graph[i].parentID,
                   _graph[i].childCount,
                   nodeStr.c_str());
        }
        cout << endl;
    }
    else
    {
        for (SLuint i = 0; i < _graph.size(); ++i)
            printf("|  %02u  ", i);
        cout << "|" << endl;

        for (SLuint i = 0; i < _graph.size(); ++i)
            cout << "-------";
        cout << "-" << endl;

        for (SLuint i = 0; i < _graph.size(); ++i)
            if (_graph[i].parentID == -1)
                printf("|-1  %02u", _graph[i].childCount);
            else
                printf("|%02u  %02u", _graph[i].parentID, _graph[i].childCount);
        cout << "|" << endl;
    }
    cout << endl;
}
//-----------------------------------------------------------------------------
/*! addChild adds a child node by inserting an SLSceneDOD into a vector in
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
SLint SLSceneDOD::addChild(SLint     myParentID,
                           SLNodeDOD nodeDOD)
{
    assert(myParentID <= _graph.size() &&
           myParentID >= -1 &&
           "Invalid parentID");

    if(nodeDOD.node)
        nodeDOD.om = nodeDOD.node->om();

    if (_graph.empty())
    {
        // Root node and ignore myParentID
        _graph.push_back(nodeDOD);
        _graph[0].parentID   = -1;
        _graph[0].childCount = 0;

        return 0;
    }
    else
    {
        nodeDOD.parentID   = myParentID;
        nodeDOD.childCount = 0;
        _graph.insert(_graph.begin() + myParentID + 1, nodeDOD);
        _graph[myParentID].childCount++;

        // Increase parentIDs of following subtrees that are greater
        for (SLuint i = myParentID + 2; i < _graph.size(); i++)
            if (_graph[i].parentID > myParentID)
                _graph[i].parentID++;

        return myParentID + 1;
    }
}
//-----------------------------------------------------------------------------
/*! Deletes a node at index id with all with all children
 @param scenegraph The scenegraph as a SLSceneDOD vector
 @param id Index of node to delete
 */
void SLSceneDOD::deleteNode(SLint id)
{
    assert(id <= _graph.size() &&
           id >= 0 &&
           "Invalid id");

    if (id == 0)
        _graph.clear();
    else
    {
        // Find the next child with the same parentID
        SLuint toID;
        SLint  myParentID = _graph[id].parentID;
        for (toID = id + 1; toID < _graph.size(); toID++)
            if (_graph[toID].parentID == myParentID)
                break;

        // Erase the elements in the vector
        _graph.erase(_graph.begin() + id, _graph.begin() + toID);
        _graph[myParentID].childCount--;

        // Decrease parentIDs of following subtrees that are greater
        SLuint numNodesToErase = toID - id;
        for (SLuint i = id; i < _graph.size(); i++)
            if (_graph[i].parentID > myParentID)
                _graph[i].parentID = _graph[i].parentID - numNodesToErase;
    }
}
//-----------------------------------------------------------------------------
void SLSceneDOD::test()
{
    addChild(0, SLNodeDOD()); // Root node
    addChild(0, SLNodeDOD());
    addChild(0, SLNodeDOD());
    addChild(0, SLNodeDOD());
    addChild(2, SLNodeDOD());
    addChild(2, SLNodeDOD());
    addChild(0, SLNodeDOD());
    addChild(1, SLNodeDOD());
    addChild(5, SLNodeDOD());
    dump(false);
    dump(true);

    deleteNode(1);
    dump(false);
    dump(true);

    deleteNode(6);
    dump(false);
    dump(true);

    deleteNode(2);
    dump(false);
    dump(true);

    deleteNode(0);
    dump(false);
    dump(true);
}
//-----------------------------------------------------------------------------