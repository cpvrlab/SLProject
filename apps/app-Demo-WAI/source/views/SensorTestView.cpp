#include <views/SensorTestView.h>
#include <sens/SENSUtils.h>

SensorTestView::SensorTestView(sm::EventHandler&   eventHandler,
                               SLInputManager&     inputManager,
                               const ImGuiEngine&  imGuiEngine,
                               ErlebAR::Resources& resources,
                               SENSGps*            sensGps,
                               SENSOrientation*    sensOrientation,
                               const DeviceData&   deviceData)
  : SLSceneView(nullptr, deviceData.dpi(), inputManager),
    _gui(imGuiEngine,
         eventHandler,
         resources,
         deviceData.dpi(),
         deviceData.scrWidth(),
         deviceData.scrHeight(),
         sensGps,
         sensOrientation),
    _gps(sensGps),
    _orientation(sensOrientation),
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
    if (_orientation)
        _orientation->stop();
}
