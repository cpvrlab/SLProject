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
                             std::string       texturePath,
                             std::string       imguiIniPath)
  : SLSceneView(nullptr, dotsPerInch, inputManager),
    sm::EventSender(eventHandler),
    _gui(eventHandler, dotsPerInch, screenWidth, screenHeight, fontPath, texturePath),
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
