#include <views/SettingsView.h>

SettingsView::SettingsView(sm::EventHandler&   eventHandler,
                           SLInputManager&     inputManager,
                           const ImGuiEngine&  imGuiEngine,
                           ErlebAR::Resources& resources,
                           const DeviceData&   deviceData)
  : SLSceneView(nullptr, deviceData.dpi(), inputManager),
    _gui(imGuiEngine, eventHandler, resources, deviceData.dpi(), deviceData.scrWidth(), deviceData.scrHeight())
{
    init("SettingsView", deviceData.scrWidth(), deviceData.scrHeight(), nullptr, nullptr, &_gui, deviceData.writableDir());
    onInitialize();
}

bool SettingsView::update()
{
    return onPaint();
}
