#ifndef APP_WAI_SCENE
#define APP_WAI_SCENE

#include <SLScene.h>
#include <SLSceneView.h>
#include <SLPoints.h>
#include <SLPolyline.h>
#include <SLAssetManager.h>

#include <CVCalibration.h>

class AppWAIScene : public SLScene
{
public:
    AppWAIScene(SLstring name, SLInputManager& inputManager);
    SLNode*   rootNode          = nullptr;
    SLCamera* cameraNode        = nullptr;
    SLNode*   mapNode           = nullptr;
    SLNode*   mapPC             = nullptr;
    SLNode*   mapMatchedPC      = nullptr;
    SLNode*   mapLocalPC        = nullptr;
    SLNode*   mapMarkerCornerPC = nullptr;
    SLNode*   keyFrameNode      = nullptr;
    SLNode*   covisibilityGraph = nullptr;
    SLNode*   spanningTree      = nullptr;
    SLNode*   loopEdges         = nullptr;

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

    SLNode* augmentationRoot = nullptr;

    void rebuild(std::string location, std::string area);
    void adjustAugmentationTransparency(float kt);

    SLAssetManager assets;

private:
};

#endif
