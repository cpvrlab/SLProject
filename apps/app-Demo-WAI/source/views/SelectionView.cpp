#include "SelectionView.h"

#include <SLInputManager.h>
#include <SelectionGui.h>

SelectionView::SelectionView(sm::EventHandler&   eventHandler,
                             SLInputManager&     inputManager,
                             ErlebAR::Resources& resources,
                             int                 screenWidth,
                             int                 screenHeight,
                             int                 dotsPerInch,
                             std::string         fontPath,
                             std::string         texturePath,
                             std::string         imguiIniPath)
  : SLSceneView(nullptr, dotsPerInch, inputManager),
    sm::EventSender(eventHandler),
    _gui(eventHandler, resources, dotsPerInch, screenWidth, screenHeight, fontPath, texturePath),
    _scene("SelectionScene", nullptr)
{
    scene(&_scene);
    init("SelectionSceneView", screenWidth, screenHeight, nullptr, nullptr, &_gui, imguiIniPath);
    _scene.init();
    onInitialize();
}

bool SelectionView::update()
{
    return onPaint();
}
