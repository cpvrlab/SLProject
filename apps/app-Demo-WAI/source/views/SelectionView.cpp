#include "SelectionView.h"

#include <SLInputManager.h>
#include <SelectionGui.h>

SelectionView::SelectionView(sm::EventHandler&   eventHandler,
                             SLInputManager&     inputManager,
                             const ImGuiEngine&  imGuiEngine,
                             ErlebAR::Resources& resources,
                             const DeviceData&   deviceData)
  : SLSceneView(nullptr, deviceData.dpi(), inputManager),
    sm::EventSender(eventHandler),
    _gui(imGuiEngine,
         eventHandler,
         resources,
         deviceData.dpi(),
         deviceData.scrWidth(),
         deviceData.scrHeight(),
         deviceData.fontDir(),
         deviceData.textureDir()),
    _scene("SelectionScene", nullptr)
{
    scene(&_scene);
    init("SelectionSceneView",
         deviceData.scrWidth(),
         deviceData.scrHeight(),
         nullptr,
         nullptr,
         &_gui,
         deviceData.writableDir());
    _scene.init();
    onInitialize();
}

bool SelectionView::update()
{
    //the viewport may be wrong when returning from TestView, which may have another ratio.
    //setViewportFromRatio(SLVec2i(0, 0), VA_center, false);
    //todo: understand again and find another solution (without calling resize all the time)
    _viewportRect.set(0, 0, _scrW, _scrH);
    _gui.onResize(_viewportRect.width,
                  _viewportRect.height,
                  _scr2fbX,
                  _scr2fbY);
    return onPaint();
}
