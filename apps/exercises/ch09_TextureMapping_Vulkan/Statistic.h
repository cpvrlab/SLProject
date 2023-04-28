#ifndef STATISTIC_H
#define STATISTIC_H

#include "Node.h"

class Statistic
{
public:
    static NodeStats getSizeOf(vector<DrawingObject>& objList)
    {
        NodeStats nodeStats = NodeStats();

        for (int i = 0; i < objList.size(); i++)
        {
            vector<Node*> nodeList = objList[i].nodeList;
            for (int j = 0; j < nodeList.size(); j++)
            {
                Node& node = *nodeList[j];
                nodeStats.numNodes++;
                nodeStats.NodesBytes += sizeof(node);

                if (node.mesh())
                {
                    nodeStats.numMeshes++;
                    nodeStats.MeshBytes += sizeof(node.mesh());
                }
                else
                    nodeStats.numEmptyNodes++;
            }
        }

        return nodeStats;
    }
};

#endif
