#include <views/AreaTrackingView.h>

AreaTrackingView::AreaTrackingView(sm::EventHandler&   eventHandler,
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
    init("AreaTrackingView", screenWidth, screenHeight, nullptr, nullptr, &_gui, imguiIniPath);
    onInitialize();
}

bool AreaTrackingView::update()
{
    return onPaint();
}

void AreaTrackingView::initArea(ErlebAR::AreaId id)
{
    _gui.initArea(id);
}
