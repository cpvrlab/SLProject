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
                  const DeviceData&   deviceData);
    SelectionView() = delete;
    bool update();

protected:
    SelectionGui _gui;
    SLScene      _scene;
};

#endif //SELECTION_VIEW_H
