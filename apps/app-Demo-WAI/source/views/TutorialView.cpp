#include <views/TutorialView.h>

TutorialView::TutorialView(sm::EventHandler&  eventHandler,
                           SLInputManager&    inputManager,
                           const ImGuiEngine& imGuiEngine,
                           ErlebAR::Config&   config,
                           const DeviceData&  deviceData)
  : SLSceneView(nullptr, deviceData.dpi(), inputManager),
    _gui(imGuiEngine, eventHandler, config, deviceData.dpi(), deviceData.scrWidth(), deviceData.scrHeight(), deviceData.fontDir(), deviceData.textureDir())
{
    init("TutorialView", deviceData.scrWidth(), deviceData.scrHeight(), nullptr, nullptr, &_gui, deviceData.writableDir());
    onInitialize();
}

bool TutorialView::update()
{
    return onPaint();
}
