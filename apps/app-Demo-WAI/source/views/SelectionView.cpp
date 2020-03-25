#include "SelectionView.h"
#include <SLSceneView.h>
#include <SLInputManager.h>

#include <SLGLTexture.h>
#include <SLGLProgram.h>
#include <SLLightSpot.h>
#include <SL/SLTexFont.h>
#include <SLSphere.h>
#include <SLText.h>
#include <SelectionGui.h>

SelectionView::SelectionView(sm::EventHandler& eventHandler,
                             SLInputManager&   inputManager,
                             int               screenWidth,
                             int               screenHeight,
                             int               dotsPerInch,
                             std::string       fontPath,
                             std::string       imguiIniPath)
  : sm::EventSender(eventHandler),
    _gui(eventHandler, dotsPerInch, screenWidth, screenHeight, fontPath),
    _s("SelectionScene", nullptr, inputManager),
    _sv(&_s, dotsPerInch)
{
    _sv.init("SelectionSceneView", screenWidth, screenHeight, nullptr, nullptr, &_gui, imguiIniPath);
    _s.init();

    _sv.onInitialize();
}

bool SelectionView::update()
{
    _s.onUpdate();
    return _sv.onPaint();
}
