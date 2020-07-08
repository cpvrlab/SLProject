#ifndef APP_WAI_SCENE_H
#define APP_WAI_SCENE_H

#include <SLScene.h>
#include <SLSceneView.h>
#include <SLPoints.h>
#include <SLPolyline.h>
#include <SLAssetManager.h>

#include <CVCalibration.h>
#include <WAIMapPoint.h>

class AppWAIScene : public SLScene
{
public:
    AppWAIScene(SLstring name, std::string dataDir);

    void updateCameraIntrinsics(float cameraFovVDeg, cv::Mat cameraMatUndistorted)
    {
        cameraNode->fov(cameraFovVDeg);
        // Set camera intrinsics for scene camera frustum. (used in projection->intrinsics mode)
        //std::cout << "cameraMatUndistorted: " << cameraMatUndistorted << std::endl;
        cameraNode->intrinsics((float)cameraMatUndistorted.at<double>(0, 0),
                               (float)cameraMatUndistorted.at<double>(1, 1),
                               (float)cameraMatUndistorted.at<double>(0, 2),
                               (float)cameraMatUndistorted.at<double>(1, 2));

        //enable projection -> intrinsics mode
        //cameraNode->projection(P_monoIntrinsic);
        cameraNode->projection(P_monoPerspective);
    }

    void resetMapNode();
    void updateCameraPose(const cv::Mat& pose);
    void updateVideoImage(const cv::Mat& image);

    void renderMapPoints(const std::vector<WAIMapPoint*>& pts);
    void renderMarkerCornerMapPoints(const std::vector<WAIMapPoint*>& pts);
    void renderLocalMapPoints(const std::vector<WAIMapPoint*>& pts);
    void renderMatchedMapPoints(const std::vector<WAIMapPoint*>& pts);
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
    SLCamera*      cameraNode = nullptr;

private:
    void renderMapPoints(std::string                      name,
                         const std::vector<WAIMapPoint*>& pts,
                         SLNode*&                         node,
                         SLPoints*&                       mesh,
                         SLMaterial*&                     material);
    void removeMesh(SLNode* node, SLMesh* mesh);

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
    SLMaterial* yellowMat            = nullptr;
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

    SLGLTexture* _videoImage = nullptr;

    //path to data directory
    std::string _dataDir;
};

#endif
