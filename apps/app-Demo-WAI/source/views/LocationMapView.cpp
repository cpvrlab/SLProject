#include <views/LocationMapView.h>

LocationMapView::LocationMapView(sm::EventHandler&   eventHandler,
                                 SLInputManager&     inputManager,
                                 const ImGuiEngine&  imGuiEngine,
                                 ErlebAR::Resources& resources,
                                 const DeviceData&   deviceData,
                                 SENSGps*            gps)
  : SLSceneView(nullptr, deviceData.dpi(), inputManager),
    _gui(imGuiEngine, eventHandler, resources, deviceData.dpi(), deviceData.scrWidth(), deviceData.scrHeight(), deviceData.erlebARDir(), gps)
{
    init("LocationMapView", deviceData.scrWidth(), deviceData.scrHeight(), nullptr, nullptr, &_gui, deviceData.writableDir());
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
