#include <views/AreaInfoView.h>

AreaInfoView::AreaInfoView(sm::EventHandler&   eventHandler,
                           SLInputManager&     inputManager,
                           ErlebAR::Resources& resources,
                           int                 screenWidth,
                           int                 screenHeight,
                           int                 dotsPerInch,
                           std::string         imguiIniPath)
  : SLSceneView(nullptr, dotsPerInch, inputManager),
    _gui(eventHandler, resources, dotsPerInch, screenWidth, screenHeight)
{
    init("AreaInfoView", screenWidth, screenHeight, nullptr, nullptr, &_gui, imguiIniPath);
    onInitialize();
}

bool AreaInfoView::update()
{
    return onPaint();
}

void AreaInfoView::initArea(ErlebAR::LocationId locId, ErlebAR::AreaId areaId)
{
    _gui.initArea(locId, areaId);
}
