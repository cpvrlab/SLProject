//#############################################################################
//  File:      SLEntities.cpp
//  Date:      June 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLEntities.h>
#include <SLNode.h>

//-----------------------------------------------------------------------------
/*! addChild adds a child node by inserting an SLEntities into a vector in
 Depth First Search order. The root node gets the parent ID -1.
 The child is inserted right after the parent node.
 @param scenegraph The scenegraph as a vector
 @param myParentID Index of the parent node
 @param node The node to add as child of the parent
*/
void SLEntities::addChildEntity(SLint    myParentID,
                                SLEntity entity)
{
    if (myParentID > (int)_graph.size() || myParentID < -1)
        SL_EXIT_MSG("Invalid parent ID");

    SLint entityID; // ID and insert position of new entity

#ifdef SL_USE_ENTITIES_DEBUG
    this->dump(true);
#endif

    if (_graph.empty())
    {
        // Root node and ignore myParentID
        entityID = 0;
        _graph.push_back(entity);
        _graph[0].parentID   = -1;
        _graph[0].childCount = 0;
        _graph[0].om         = _graph[0].node->om();
        _graph[0].node->entityID(0);
        if (_graph[0].node->parent() != nullptr)
            SL_EXIT_MSG("Root node parent pointer must be null");
    }
    else
    {
        entity.parentID   = myParentID;
        entity.childCount = 0;
        entity.om         = entity.node->om();

        // Loop forward to the insert position
        SLint insertPos = myParentID + 1;
        while (insertPos < _graph.size() && _graph[insertPos].parentID >= myParentID)
            insertPos++;

        entityID = insertPos;
        if (entityID == _graph.size())
            _graph.push_back(entity); // faster than insert
        else
            _graph.insert(_graph.begin() + entityID, entity);

        _graph[myParentID].childCount++;

        // Increase parentIDs of following subtrees that are greater
        for (SLuint i = entityID + 1; i < _graph.size(); i++)
            if (_graph[i].parentID > myParentID)
                _graph[i].parentID++;

        // Correct all node->entityIDs
        for (SLint i = 0; i < _graph.size(); ++i)
            _graph[i].node->entityID(i);
    }

    // Recursively add children of the entity
    if (_graph[entityID].node->children().size())
    {
        for (SLint i = 0; i < _graph[entityID].node->children().size(); ++i)
        {
            SLNode* childNode = _graph[entityID].node->children()[i];
            this->addChildEntity(entityID, SLEntity(childNode));
        }
    }
}
//-----------------------------------------------------------------------------
/*! Returns the pointer to the node at id
 @param scenegraph The scenegraph as a SLEntities vector
 @param id The index of the node to return as a pointer
 @return The pointer of the node at id
 */
SLEntity* SLEntities::getEntity(SLint id)
{
    if (id >= 0 && id < _graph.size())
        return &_graph[id];
    else
        return nullptr;
}
//-----------------------------------------------------------------------------
/*! Returns the ID for the passed SLNode if found else INT32_MIN
 */
SLint SLEntities::getEntityID(SLNode* node)
{
    for (SLint i = 0; i < _graph.size(); ++i)
        if (_graph[i].node == node)
            return i;
    return INT32_MIN;
}
//-----------------------------------------------------------------------------
/*! Returns the pointer to the parent of the node at id
 @param scenegraph The scenegraph as a SLEntities vector
 @param id The index of the node in the scenegraph vector
 @return The pointer of the parent of the node at id
 */
SLEntity* SLEntities::getParent(SLint id)
{
    if (id >= 1 && id < _graph.size())
        return &_graph[_graph[id].parentID];
    else
        return nullptr;
}
//-----------------------------------------------------------------------------
/*! Returns the parentID for the passed SLNode if found else INT32_MIN
 */
SLint SLEntities::getParentID(SLNode* node)
{
    for (SLEntity& entity : _graph)
        if (entity.node == node)
            return entity.parentID;
    return INT32_MIN;
}
//-----------------------------------------------------------------------------
/*! Updates the world matrix recursively
 * @param id Index of the current node to update
 * @param parentWM World transform matrix of the parent
 * @return The no. of nodes updated
 */
SLint SLEntities::updateWMRec(SLint id, SLMat4f& parentWM)
{
    SLEntity* entity = getEntity(id);

#ifdef SL_USE_ENTITIES_DEBUG
    SLMat4f nodeOM = entity->node->om();
    if (!entity->om.isEqual(nodeOM))
    {
        string nodeName = "nodeOM: " + entity->node->name();
        entity->om.print("entityOM:");
        entity->node->om().print(nodeName.c_str());
        entity->om = entity->node->om();
    }
#endif

    entity->wm.setMatrix(parentWM * entity->om);
    entity->wmI.setMatrix(entity->wm.inverted());

#ifdef SL_USE_ENTITIES_DEBUG
    SLMat4f nodeWM = entity->node->updateAndGetWM();
    if (!entity->wm.isEqual(nodeWM))
    {
        string nodeName = "nodeWM: " + entity->node->name();
        entity->wm.print("entityWM:");
        nodeWM.print(nodeName.c_str());
    }
#endif

    SLuint handledChildren = 0;
    while (handledChildren < entity->childCount)
    {
        SLint childID = id + handledChildren + 1;
        handledChildren += this->updateWMRec(childID, entity->wm);
    }

    return handledChildren + 1;
}
//-----------------------------------------------------------------------------
/*! Prints the scenegraph vector flat or as hierarchical tree as follows:
 @param scenegraph The scenegraph as a node vector
 @param doTreeDump Flag if dump as a tree or flat

Pattern: ID.Parent.childCount
Example from SLEntities::addChild:
00.-1.04
+--01.00.01
|  +--02.01.00
+--03.00.00
+--04.00.02
|  +--05.04.00
|  +--06.04.01
|  |  +--07.06.01
|  |  |  +--08.07.00
+--09.00.00
*/
void SLEntities::dump(SLbool doTreeDump)
{
    if (doTreeDump)
    {
        for (SLuint i = 0; i < _graph.size(); ++i)
        {
            // Calculate depth
            SLuint depth      = 0;
            SLint  myParentID = _graph[i].parentID;
            while (myParentID != -1)
            {
                depth++;
                myParentID = _graph[myParentID].parentID;
            }

            string tabs;
            for (int d = 1; d < (int)depth; ++d)
                tabs += "|  ";
            if (depth > 0)
                tabs += "+--";
            cout << tabs;

            SLstring nodeStr = _graph[i].node ? _graph[i].node->name() : "";
            printf("%02u(%02d).%02d.%02u-%s\n",
                   i,
                   _graph[i].node->entityID(),
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
    cout << "----------------------------------------------------------" << endl;
}
//-----------------------------------------------------------------------------
/*! Deletes a node at index id with all with all children
 @param scenegraph The scenegraph as a SLEntities vector
 @param id Index of node to delete
 */
void SLEntities::deleteEntity(SLint id)
{
    assert(id <= _graph.size() &&
           id >= 0 &&
           "Invalid id");

    if (id == 0)
        _graph.clear();
    else
    {
        // Find the next child with the same parentID
        SLint toID;
        SLint myParentID = _graph[id].parentID;
        for (toID = id + 1; toID < _graph.size(); toID++)
            if (_graph[toID].parentID == myParentID)
                break;

        // Erase the elements in the vector
        _graph.erase(_graph.begin() + id, _graph.begin() + toID);
        _graph[myParentID].childCount--;

        // Decrease parentIDs of following subtrees that are greater
        SLint numNodesToErase = toID - id;
        for (SLuint i = id; i < _graph.size(); i++)
            if (_graph[i].parentID > myParentID)
                _graph[i].parentID = _graph[i].parentID - numNodesToErase;
    }
}
//-----------------------------------------------------------------------------
/*! Deletes all children of an entity with the index id. Also sub-children of
 * those children get deleted.
 * @param id Index of the parent entity
 */
void SLEntities::deleteChildren(SLint id)
{
    assert(id <= _graph.size() &&
           id >= 0 &&
           "Invalid id");

    // Find the next child with the same parentID
    SLuint toID;
    SLint  myParentID = _graph[id].parentID;
    for (toID = id + 1; toID < _graph.size(); toID++)
        if (_graph[toID].parentID == myParentID)
            break;

    // Erase the elements in the vector
    _graph.erase(_graph.begin() + id + 1, _graph.begin() + toID);
    _graph[id].childCount = 0;

    // Decrease parentIDs of following subtrees that are greater
    SLuint numNodesToErase = toID - id;
    for (SLuint i = id; i < _graph.size(); i++)
        if (_graph[i].parentID > myParentID)
            _graph[i].parentID = _graph[i].parentID - numNodesToErase;
}
//-----------------------------------------------------------------------------