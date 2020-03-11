#ifndef APP_WAI_SCENE_H
#define APP_WAI_SCENE_H

#include <SLSceneView.h>
#include <SLPoints.h>
#include <SLPolyline.h>

#include <CVCalibration.h>

class AppWAIScene
{
public:
    AppWAIScene();
    SLNode*   rootNode          = nullptr;
    SLNode*   root2DNode        = nullptr;
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
};

#endif
