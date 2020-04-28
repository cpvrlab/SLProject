#ifndef AREA_INFO_VIEW_H
#define AREA_INFO_VIEW_H

#include <string>
#include <SLInputManager.h>
#include <SLSceneView.h>
#include <AreaInfoGui.h>
#include <ErlebAR.h>

class AreaInfoView : public SLSceneView
{
public:
    AreaInfoView(sm::EventHandler&   eventHandler,
                 SLInputManager&     inputManager,
                 ErlebAR::Resources& resources,
                 int                 screenWidth,
                 int                 screenHeight,
                 int                 dotsPerInch,
                 std::string         fontPath,
                 std::string         imguiIniPath);
    bool update();
    //call when view becomes visible
    void show() { _gui.onShow(); }

    void initArea(ErlebAR::LocationId locId, ErlebAR::AreaId areaId);

private:
    AreaInfoGui _gui;
};

#endif //AREA_INFO_VIEW_H
