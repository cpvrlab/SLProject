//#############################################################################
//  File:      SLCVMapNode.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCV_MAP_NODE_H
#define SLCV_MAP_NODE_H

#include <string>
#include <SLNode.h>

class SLCVMap;
class SLMaterial;
class SLPoints;

//-----------------------------------------------------------------------------
class SLCVMapNode : public SLNode
{
public:
    SLCVMapNode(SLCVMap* map, std::string name);
    ~SLCVMapNode();

    //!add objects in SLCVMap into SLCVMapNode
    void addMapObjects(SLCVMap& map);

    //!update map with SLPoints of map points for current frame
    //!If an empty vector is provided, the mesh is only removed
    void updateMapPoints(const std::vector<SLCVMapPoint*>& pts);
    void updateMapPointsLocal(const std::vector<SLCVMapPoint*>& pts);
    void updateMapPointsMatched(const std::vector<SLCVMapPoint*>& mapPointMatches);
    //!Remove map points
    void removeMapPoints();
    void removeMapPointsLocal();
    void removeMapPointsMatched();

private:
    //!convenience function: may be used to update all point clouds
    void doUpdateMapPoints(std::string name, const std::vector<SLCVMapPoint*>& pts,
        SLNode* node, SLMesh* mesh, SLMaterial* material);

    //Nodes:
    SLNode* _keyFrames = NULL;
    SLNode* _mapPC = NULL;
    SLNode* _mapMatchedPC = NULL;
    SLNode* _mapLocalPC = NULL;
    //Meshes:
    SLPoints* _mapMesh = NULL;
    SLPoints* _mapLocalMesh = NULL;
    SLPoints* _mapMatchesMesh = NULL;
    //Materials:
    SLMaterial* _pcMat = NULL;
    SLMaterial* _pcMatchedMat = NULL;
    SLMaterial* _pcLocalMat = NULL;
};

#endif //SLCV_MAP_NODE_H