#ifndef SELECTION_VIEW_H
#define SELECTION_VIEW_H

#include <SelectionGui.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <ErlebAR.h>
#include <sm/EventSender.h>
#include <Resources.h>

class SLInputManager;

class SelectionView : public sm::EventSender
  , protected SLSceneView
{
public:
    SelectionView(sm::EventHandler&   eventHandler,
                  SLInputManager&     inputManager,
                  const ImGuiEngine&  imGuiEngine,
                  ErlebAR::Resources& resources,
                  int                 screenWidth,
                  int                 screenHeight,
                  int                 dotsPerInch,
                  std::string         fontPath,
                  std::string         texturePath,
                  std::string         imguiIniPath);
    SelectionView() = delete;
    bool update();

protected:
    SelectionGui _gui;
    SLScene      _scene;
};

#endif //SELECTION_VIEW_H
