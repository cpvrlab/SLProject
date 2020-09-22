#ifndef APP_WAI_SCENE_H
#define APP_WAI_SCENE_H

#include <SLScene.h>
#include <SLSceneView.h>
#include <SLPoints.h>
#include <SLPolyline.h>
#include <SLAssetManager.h>

#include <CVCalibration.h>
#include <WAIMapPoint.h>
#include <VideoBackgroundCamera.h>

class AppWAIScene : public SLScene
{
public:
    AppWAIScene(SLstring name, std::string dataDir, std::string erlebARDir);

    void resetMapNode();
    void updateCameraPose(const cv::Mat& pose);

    void renderMapPoints(const std::vector<WAIMapPoint*>& pts);
    void renderMarkerCornerMapPoints(const std::vector<WAIMapPoint*>& pts);
    void renderLocalMapPoints(const std::vector<WAIMapPoint*>& pts);
    void renderMatchedMapPoints(const std::vector<WAIMapPoint*>& pts, float opacity = 1.0f);
    void removeMapPoints();
    void removeMarkerCornerMapPoints();
    void removeLocalMapPoints();
    void removeMatchedMapPoints();

    void renderKeyframes(const std::vector<WAIKeyFrame*>& keyframes);
    void removeKeyframes();
    void renderGraphs(const std::vector<WAIKeyFrame*>& kfs,
                      const int&                       minNumOfCovisibles,
                      const bool                       showCovisibilityGraph,
                      const bool                       showSpanningTree,
                      const bool                       showLoopEdges);

    void removeGraphs();

    SLNode* augmentationRoot = nullptr;

    void rebuild(std::string location, std::string area);
    void adjustAugmentationTransparency(float kt);

    SLAssetManager assets;
    SLNode*        mapNode    = nullptr;

    VideoBackgroundCamera* camera = nullptr;
private:
    void renderMapPoints(std::string                      name,
                         const std::vector<WAIMapPoint*>& pts,
                         SLNode*&                         node,
                         SLPoints*&                       mesh,
                         SLMaterial*&                     material,
                         float                            opacity = 1.f);
    void removeMesh(SLNode* node, SLMesh* mesh);
    void hideNode(SLNode* node);

    void loadMesh(std::string path);

    SLNode* mapPC             = nullptr;
    SLNode* mapMatchedPC      = nullptr;
    SLNode* mapLocalPC        = nullptr;
    SLNode* mapMarkerCornerPC = nullptr;
    SLNode* keyFrameNode      = nullptr;
    SLNode* covisibilityGraph = nullptr;
    SLNode* spanningTree      = nullptr;
    SLNode* loopEdges         = nullptr;

    SLMaterial* redMat               = nullptr;
    SLMaterial* greenMat             = nullptr;
    SLMaterial* blueMat              = nullptr;
    SLMaterial* covisibilityGraphMat = nullptr;
    SLMaterial* spanningTreeMat      = nullptr;
    SLMaterial* loopEdgesMat         = nullptr;

    SLPoints*   mappointsMesh             = nullptr;
    SLPoints*   mappointsMatchedMesh      = nullptr;
    SLPoints*   mappointsLocalMesh        = nullptr;
    SLPoints*   mappointsMarkerCornerMesh = nullptr;
    SLPolyline* covisibilityGraphMesh     = nullptr;
    SLPolyline* spanningTreeMesh          = nullptr;
    SLPolyline* loopEdgesMesh             = nullptr;

    //path to data directory
    std::string _dataDir;
    std::string _erlebARDir;
};

#endif
