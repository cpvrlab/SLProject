#include <views/AboutView.h>

AboutView::AboutView(sm::EventHandler&   eventHandler,
                     SLInputManager&     inputManager,
                     ErlebAR::Resources& resources,
                     int                 screenWidth,
                     int                 screenHeight,
                     int                 dotsPerInch,
                     std::string         imguiIniPath)
  : SLSceneView(nullptr, dotsPerInch, inputManager),
    _gui(eventHandler, resources, dotsPerInch, screenWidth, screenHeight)
{
    init("AboutView", screenWidth, screenHeight, nullptr, nullptr, &_gui, imguiIniPath);
    onInitialize();
}

bool AboutView::update()
{
    return onPaint();
}
