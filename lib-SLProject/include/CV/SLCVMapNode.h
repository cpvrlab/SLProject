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
class SLCVKeyFrame;
class SLCVMapPoint;
class SLPolyline;

//-----------------------------------------------------------------------------
class SLCVMapNode : public SLNode
{
public:
    SLCVMapNode(std::string name);
    SLCVMapNode(std::string name, SLCVMap& map);

    //!set SLCVMapNode in SLCVMap and update SLCVMapNode
    void setMap(SLCVMap& map);

    void doUpdate() override;

    //!update map with SLPoints of map points for current frame
    //!If an empty vector is provided, the mesh is only removed
    void updateMapPoints(const std::vector<SLCVMapPoint*>& pts);
    void updateMapPointsLocal(const std::vector<SLCVMapPoint*>& pts);
    void updateMapPointsMatched(const std::vector<SLCVMapPoint*>& mapPointMatches);
    void updateKeyFrames(const std::vector<SLCVKeyFrame*>& kfs);
    void updateAll(SLCVMap& map); //todo: const SLCVMap

    //!Remove map points
    void removeMapPoints();
    void removeMapPointsLocal();
    void removeMapPointsMatched();
    void removeKeyFrames();
    void removeGraphs();

    //!used to remove all when map changes
    void clearAll();

    //! Set minimum number of covisible map points: directly updates the graph.
    //! It must be called from gui thread
    void updateMinNumOfCovisibles(int n);

    //!set hidden flags
    void setHideMapPoints(bool state);
    void setHideKeyFrames(bool state);
    void setHideCovisibilityGraph(bool state);
    void setHideSpanningTree(bool state);
    void setHideLoopEdges(bool state);

    //getters
    bool renderKfBackground() { return _renderKfBackground; }
    bool allowAsActiveCam() { return _allowAsActiveCam; }
    //setters
    void renderKfBackground(bool s) { _renderKfBackground = s; }
    void allowAsActiveCam(bool s) { _allowAsActiveCam = s; }
    int getMinNumOfCovisibles() { return _minNumOfCovisibles; }
private:
    //! add map nodes and instantiate materials
    void init();

    //!execute map points update. convenience function: may be used to update all point clouds
    void doUpdateMapPoints(std::string name, const std::vector<SLCVMapPoint*>& pts,
        SLNode*& node, SLPoints*& mesh, SLMaterial*& material);
    //!execute keyframe update
    void doUpdateKeyFrames(const std::vector<SLCVKeyFrame*>& kfs);
    //!execute update of spanning tree, covisibility graph and loop edges
    void doUpdateGraphs(const std::vector<SLCVKeyFrame*>& kfs);

    //Nodes:
    SLNode* _keyFrames = NULL;
    SLNode* _covisibilityGraph = NULL;
    SLNode* _spanningTree = NULL;
    SLNode* _loopEdges = NULL;
    SLNode* _mapPC = NULL;
    SLNode* _mapMatchedPC = NULL;
    SLNode* _mapLocalPC = NULL;
    //Meshes:
    SLPoints* _mapMesh = NULL;
    SLPoints* _mapLocalMesh = NULL;
    SLPoints* _mapMatchesMesh = NULL;
    SLPolyline* _covisibilityGraphMesh = NULL;
    SLPolyline* _spanningTreeMesh = NULL;
    SLPolyline* _loopEdgesMesh = NULL;
    //Materials:
    SLMaterial* _pcMat = NULL;
    SLMaterial* _pcMatchedMat = NULL;
    SLMaterial* _pcLocalMat = NULL;
    SLMaterial* _covisibilityGraphMat = NULL;
    SLMaterial* _spanningTreeMat = NULL;
    SLMaterial* _loopEdgesMat = NULL;

    //vector of keyframe textures: we store a pointer to correctly delete them during a keyframe update
    std::vector<SLGLTexture*> _kfTextures;

    //!mutex saved flags and vectors: only manipulate locking mutex
    bool _mapPtsChanged = false;
    bool _mapPtsLocalChanged = false;
    bool _mapPtsMatchedChanged = false;
    bool _keyFramesChanged = false;
    std::vector<SLCVMapPoint*> _mapPts;
    std::vector<SLCVMapPoint*> _mapPtsLocal;
    std::vector<SLCVMapPoint*> _mapPtsMatched;
    std::vector<SLCVKeyFrame*> _kfs;
    bool _removeMapPoints = false;
    bool _removeMapPointsLocal = false;
    bool _removeMapPointsMatched = false;
    bool _removeKeyFrames = false;
    bool _removeGraphs = false;

    //if backgound rendering is active kf images will be rendered on 
    //near clipping plane if kf is not the active camera
    bool _renderKfBackground = false;
    //allow SLCVCameras as active camera so that we can look through it
    bool _allowAsActiveCam = false;
    //! minimum number of covisibles for covisibility graph visualization
    int _minNumOfCovisibles = 50;

    std::mutex _mutex;
};

#endif //SLCV_MAP_NODE_H