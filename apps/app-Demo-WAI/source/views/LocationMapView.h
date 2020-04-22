#ifndef LOCATION_MAP_VIEW_H
#define LOCATION_MAP_VIEW_H

#include <string>
#include <SLInputManager.h>
#include <SLSceneView.h>
#include <LocationMapGui.h>
#include <ErlebAR.h>

class LocationMapView : public SLSceneView
{
public:
    LocationMapView(sm::EventHandler&   eventHandler,
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
    void initLocation(ErlebAR::LocationId locId);

private:
    LocationMapGui _gui;

    std::map<ErlebAR::LocationId, ErlebAR::Location> _locations;
};

#endif //LOCATION_MAP_VIEW_H
