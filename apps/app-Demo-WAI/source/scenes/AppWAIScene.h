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
#include <ErlebAR.h>

class SLDeviceRotation;
class SLTexFont;

class AppWAIScene : public SLScene
{
public:
    AppWAIScene(SLstring name, std::string dataDir, std::string erlebARDir);
    ~AppWAIScene();

    void unInit() override;
    void initScene(ErlebAR::LocationId locationId, ErlebAR::AreaId areaId, SLDeviceRotation* devRot, SLDeviceLocation* devLoc, int svW, int svH);

    void resetMapNode();
    void updateCameraPose(const cv::Mat& cTw);

    void renderMapPoints(const std::vector<WAIMapPoint*>& pts);
    void renderMarkerCornerMapPoints(const std::vector<WAIMapPoint*>& pts);
    void renderLocalMapPoints(const std::vector<WAIMapPoint*>& pts);
    void renderMatchedMapPoints(const std::vector<WAIMapPoint*>& pts, float opacity = 1.0f);
    void removeMapPoints();
    void removeMarkerCornerMapPoints();
    void removeLocalMapPoints();
    void removeMatchedMapPoints();

    void renderKeyframes(const std::vector<WAIKeyFrame*>& keyframes, const std::vector<WAIKeyFrame*>& candidates);
    void removeKeyframes();
    void renderGraphs(const std::vector<WAIKeyFrame*>& kfs,
                      const int&                       minNumOfCovisibles,
                      const bool                       showCovisibilityGraph,
                      const bool                       showSpanningTree,
                      const bool                       showLoopEdges);
    void removeGraphs();

    void adjustAugmentationTransparency(float kt);

    SLAssetManager         assets;
    SLNode*                mapNode  = nullptr;
    VideoBackgroundCamera* camera   = nullptr;
    SLLightDirect*         sunLight = nullptr;

private:
    void renderMapPoints(std::string                      name,
                         const std::vector<WAIMapPoint*>& pts,
                         SLNode*&                         node,
                         SLPoints*&                       mesh,
                         SLMaterial*&                     material,
                         float                            opacity = 1.f);
    void removeMesh(SLNode* node, SLMesh* mesh);

    void initMapVisualization();
    void initAreaVisualization(ErlebAR::LocationId locationId, ErlebAR::AreaId areaId, SLDeviceRotation* devRot, SLDeviceLocation* devLoc, int svW, int svH);
    void initLocationAugst();
    void initAreaAvenchesAmphitheater();
    void initAreaAvenchesCigognier();
    void initAreaAvenchesTheatre();
    void initAreaEvilardOffice(SLDeviceRotation* devRot, SLDeviceLocation* devLoc, int svW, int svH);
    void initLocationBern();
    void initLocationBiel(SLDeviceRotation* devRot, SLDeviceLocation* devLoc);
    void initLocationDefault();
    void loadChristoffelBernBahnhofsplatz();
    void loadBielBFHRolex();
    void loadAugstTempelTheater();
    void loadAvenchesAmphitheater();
    void loadAvenchesCigognier();
    void loadAvenchesTheatre();

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

    SLTexFont* _font16 = nullptr;
};

#endif
