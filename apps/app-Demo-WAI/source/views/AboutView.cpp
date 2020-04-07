#include <views/AboutView.h>

AboutView::AboutView(sm::EventHandler& eventHandler,
                     SLInputManager&   inputManager,
                     int               screenWidth,
                     int               screenHeight,
                     int               dotsPerInch,
                     std::string       fontPath,
                     std::string       imguiIniPath)
  : SLSceneView(nullptr, dotsPerInch, inputManager),
    _gui(eventHandler, dotsPerInch, screenWidth, screenHeight, fontPath)
{
    init("AboutView", screenWidth, screenHeight, nullptr, nullptr, &_gui, imguiIniPath);
    onInitialize();
}

bool AboutView::update()
{
    return onPaint();
}
