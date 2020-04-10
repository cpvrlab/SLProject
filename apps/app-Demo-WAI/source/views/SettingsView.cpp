#include <views/SettingsView.h>

SettingsView::SettingsView(sm::EventHandler&   eventHandler,
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
    init("SettingsView", screenWidth, screenHeight, nullptr, nullptr, &_gui, imguiIniPath);
    onInitialize();
}

bool SettingsView::update()
{
    return onPaint();
}
