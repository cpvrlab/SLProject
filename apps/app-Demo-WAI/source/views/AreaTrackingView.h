#ifndef AREA_TRACKING_VIEW_H
#define AREA_TRACKING_VIEW_H

#include <string>
#include <SLInputManager.h>
#include <SLSceneView.h>
#include <AreaTrackingGui.h>
#include <ErlebAR.h>
#include <AreaTrackingScene.h>

class SENSCamera;

class AreaTrackingView : public SLSceneView
{
public:
    AreaTrackingView(sm::EventHandler&   eventHandler,
                     SLInputManager&     inputManager,
                     ErlebAR::Resources& resources,
                     SENSCamera*         camera,
                     int                 screenWidth,
                     int                 screenHeight,
                     int                 dotsPerInch,
                     std::string         fontPath,
                     std::string         imguiIniPath);
    bool update();
    //call when view becomes visible
    void show() { _gui.onShow(); }

    void initArea(ErlebAR::LocationId locId, ErlebAR::AreaId areaId);

    void resume();
    void hold();

private:
    void              startCamera();
    AreaTrackingGui   _gui;
    AreaTrackingScene _scene;

    std::map<ErlebAR::LocationId, ErlebAR::Location> _locations;

    SENSCamera* _camera = nullptr;
};

#endif //AREA_TRACKING_VIEW_H
