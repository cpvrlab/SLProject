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
    AreaInfoView(sm::EventHandler&  eventHandler,
                 SLInputManager&    inputManager,
                 const ImGuiEngine& imGuiEngine,
                 ErlebAR::Config&   config,
                 const DeviceData&  deviceData);
    bool update();
    //call when view becomes visible
    void show() { _gui.onShow(); }

    void initArea(ErlebAR::LocationId locId, ErlebAR::AreaId areaId, bool hasData);

private:
    AreaInfoGui _gui;
};

#endif //AREA_INFO_VIEW_H
