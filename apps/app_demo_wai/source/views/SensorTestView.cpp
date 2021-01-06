#include <views/SensorTestView.h>
#include <sens/SENSUtils.h>

SensorTestView::SensorTestView(sm::EventHandler&  eventHandler,
                               SLInputManager&    inputManager,
                               const ImGuiEngine& imGuiEngine,
                               ErlebAR::Config&   config,
                               SENSGps*           sensGps,
                               SENSOrientation*   sensOrientation,
                               SENSCamera*        sensCamera,
                               const DeviceData&  deviceData)
  : SLSceneView(nullptr, deviceData.dpi(), inputManager),
    _gui(imGuiEngine,
         eventHandler,
         config,
         deviceData,
         sensGps,
         sensOrientation,
         sensCamera),
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
    _gui.onHide();
    //stop all sensors
    if (_gps)
        _gps->stop();
    if (_orientation)
        _orientation->stop();
}
