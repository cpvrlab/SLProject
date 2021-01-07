#include <views/DownloadView.h>

DownloadView::DownloadView(sm::EventHandler&                    eventHandler,
                           SLInputManager&                      inputManager,
                           const ImGuiEngine&                   imGuiEngine,
                           ErlebAR::Config&                     config,
                           std::map<std::string, AsyncWorker*>& asyncWorkers,
                           const DeviceData&                    deviceData)
  : SLSceneView(nullptr, deviceData.dpi(), inputManager),
    _gui(imGuiEngine, eventHandler, config, deviceData.dataDir(), asyncWorkers, deviceData.dpi(), deviceData.scrWidth(), deviceData.scrHeight())
{
    init("DownloadView", deviceData.scrWidth(), deviceData.scrHeight(), nullptr, nullptr, &_gui, deviceData.writableDir());
    onInitialize();
}

bool DownloadView::update()
{
    return onPaint();
}

void DownloadView::initLocation(ErlebAR::LocationId locId)
{
    _gui.initLocation(locId);
}
