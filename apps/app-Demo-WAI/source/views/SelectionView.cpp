#include "SelectionView.h"

#include <SLInputManager.h>
#include <SelectionGui.h>

SelectionView::SelectionView(sm::EventHandler&   eventHandler,
                             SLInputManager&     inputManager,
                             const ImGuiEngine&  imGuiEngine,
                             ErlebAR::Resources& resources,
                             int                 screenWidth,
                             int                 screenHeight,
                             int                 dotsPerInch,
                             std::string         fontPath,
                             std::string         texturePath,
                             std::string         imguiIniPath)
  : SLSceneView(nullptr, dotsPerInch, inputManager),
    sm::EventSender(eventHandler),
    _gui(imGuiEngine, eventHandler, resources, dotsPerInch, screenWidth, screenHeight, fontPath, texturePath),
    _scene("SelectionScene", nullptr)
{
    scene(&_scene);
    init("SelectionSceneView", screenWidth, screenHeight, nullptr, nullptr, &_gui, imguiIniPath);
    _scene.init();
    onInitialize();
}

bool SelectionView::update()
{
    //the viewport may be wrong when returning from TestView, which may have another ratio.
    //setViewportFromRatio(SLVec2i(0, 0), VA_center, false);
    _viewportRect.set(0, 0, _scrW, _scrH);
    _gui.onResize(_viewportRect.width,
                  _viewportRect.height,
                  _scr2fbX,
                  _scr2fbY);
    return onPaint();
}
