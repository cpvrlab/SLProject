#ifndef LOCATION_MAP_VIEW_H
#define LOCATION_MAP_VIEW_H

#include <string>
#include <SLInputManager.h>
#include <SLSceneView.h>
#include <LocationMapGui.h>
#include <ErlebAR.h>
#include <sens/SENSGps.h>

class LocationMapView : public SLSceneView
{
public:
    LocationMapView(sm::EventHandler&   eventHandler,
                    SLInputManager&     inputManager,
                    const ImGuiEngine&  imGuiEngine,
                    ErlebAR::Resources& resources,
                    const DeviceData&   deviceData,
                    SENSGps*            gps);
    bool update();
    //call when view becomes visible
    void onShow() { _gui.onShow(); }
    void onHide() { _gui.onHide(); }
    void initLocation(ErlebAR::LocationId locId);

private:
    LocationMapGui _gui;
};

#endif //LOCATION_MAP_VIEW_H
