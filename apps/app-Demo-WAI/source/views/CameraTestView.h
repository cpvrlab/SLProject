#ifndef CAMERA_TEST_VIEW_H
#define CAMERA_TEST_VIEW_H

#include <string>
#include <SLInputManager.h>
#include <SLSceneView.h>
#include <CameraTestGui.h>
#include <ErlebAR.h>
#include <sens/SENSCamera.h>
#include <scenes/CameraOnlyScene.h>

class CameraTestView : public SLSceneView
{
public:
    CameraTestView(sm::EventHandler&   eventHandler,
                   SLInputManager&     inputManager,
                   const ImGuiEngine&  imGuiEngine,
                   ErlebAR::Resources& resources,
                   SENSCamera*         sensCamera,
                   int                 screenWidth,
                   int                 screenHeight,
                   int                 dotsPerInch,
                   std::string         imguiIniPath);
    bool update();
    //call when view becomes visible
    void show() { _gui.onShow(); }

    void startCamera();
    void stopCamera();

private:
    CameraTestGui   _gui;
    CameraOnlyScene _scene;
    SENSCamera*     _sensCamera;
};

#endif //CAMERA_TEST_VIEW_H
