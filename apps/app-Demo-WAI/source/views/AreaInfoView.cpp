#include <views/AreaInfoView.h>

AreaInfoView::AreaInfoView(sm::EventHandler&   eventHandler,
                           SLInputManager&     inputManager,
                           const ImGuiEngine&  imGuiEngine,
                           ErlebAR::Resources& resources,
                           const DeviceData&   deviceData)
  : SLSceneView(nullptr, deviceData.dpi(), inputManager),
    _gui(imGuiEngine, eventHandler, resources, deviceData.dpi(), deviceData.scrWidth(), deviceData.scrHeight())
{
    init("AreaInfoView", deviceData.scrWidth(), deviceData.scrHeight(), nullptr, nullptr, &_gui, deviceData.writableDir());
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
