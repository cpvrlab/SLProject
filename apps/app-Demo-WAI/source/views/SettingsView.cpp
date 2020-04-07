#include <views/SettingsView.h>

SettingsView::SettingsView(SLInputManager& inputManager,
                           int             screenWidth,
                           int             screenHeight,
                           int             dotsPerInch,
                           std::string     fontPath,
                           std::string     imguiIniPath)
  : SLSceneView(nullptr, dotsPerInch, inputManager),
    _gui(dotsPerInch, screenWidth, screenHeight, fontPath)
{
    init("SettingsView", screenWidth, screenHeight, nullptr, nullptr, &_gui, imguiIniPath);
    onInitialize();
}

bool SettingsView::update()
{
    return onPaint();
}
