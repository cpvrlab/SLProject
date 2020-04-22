#include <views/LocationMapView.h>

LocationMapView::LocationMapView(sm::EventHandler&   eventHandler,
                                 SLInputManager&     inputManager,
                                 ErlebAR::Resources& resources,
                                 int                 screenWidth,
                                 int                 screenHeight,
                                 int                 dotsPerInch,
                                 std::string         fontPath,
                                 std::string         imguiIniPath)
  : SLSceneView(nullptr, dotsPerInch, inputManager),
    _gui(eventHandler, resources, dotsPerInch, screenWidth, screenHeight, fontPath)
{
    init("LocationMapView", screenWidth, screenHeight, nullptr, nullptr, &_gui, imguiIniPath);
    onInitialize();

    _locations = ErlebAR::defineLocations();
}

bool LocationMapView::update()
{
    return onPaint();
}

void LocationMapView::initLocation(ErlebAR::LocationId locId)
{
    if (_locations.find(locId) != _locations.end())
    {
        _gui.initLocation(_locations[locId]);
    }
    else
    {
        Utils::exitMsg("LocationMapView", "No location defined for location id!", __LINE__, __FILE__);
    }
}
