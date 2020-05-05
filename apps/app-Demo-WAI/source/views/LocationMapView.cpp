#include <views/LocationMapView.h>

LocationMapView::LocationMapView(sm::EventHandler&   eventHandler,
                                 SLInputManager&     inputManager,
                                 ErlebAR::Resources& resources,
                                 int                 screenWidth,
                                 int                 screenHeight,
                                 int                 dotsPerInch,
                                 std::string         imguiIniPath,
                                 std::string         erlebARDir)
  : SLSceneView(nullptr, dotsPerInch, inputManager),
    _gui(eventHandler, resources, dotsPerInch, screenWidth, screenHeight, erlebARDir)
{
    init("LocationMapView", screenWidth, screenHeight, nullptr, nullptr, &_gui, imguiIniPath);
    onInitialize();
}

bool LocationMapView::update()
{
    return onPaint();
}

void LocationMapView::initLocation(ErlebAR::LocationId locId)
{
    _gui.initLocation(locId);
}
