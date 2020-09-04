#include <views/SensorTestView.h>
#include <sens/SENSUtils.h>

SensorTestView::SensorTestView(sm::EventHandler&   eventHandler,
                               SLInputManager&     inputManager,
                               const ImGuiEngine&  imGuiEngine,
                               ErlebAR::Resources& resources,
                               SENSGps*            sensGps,
                               const DeviceData&   deviceData)
  : SLSceneView(nullptr, deviceData.dpi(), inputManager),
    _gui(imGuiEngine,
         eventHandler,
         resources,
         deviceData.dpi(),
         deviceData.scrWidth(),
         deviceData.scrHeight(),
         sensGps),
    _gps(sensGps),
    _deviceData(deviceData)
{
    init("CameraTestView", deviceData.scrWidth(), deviceData.scrHeight(), nullptr, nullptr, &_gui, deviceData.writableDir());
    onInitialize();
}

bool SensorTestView::update()
{
    return onPaint();
}

void SensorTestView::onHide()
{
    //stop all sensors
    if (_gps)
        _gps->stop();
}
