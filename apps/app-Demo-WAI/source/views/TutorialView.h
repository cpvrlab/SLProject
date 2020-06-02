#ifndef TUTORIAL_VIEW_H
#define TUTORIAL_VIEW_H

#include <string>
#include <SLInputManager.h>
#include <SLSceneView.h>
#include <TutorialGui.h>
#include <ErlebAR.h>

class TutorialView : public SLSceneView
{
public:
    TutorialView(sm::EventHandler&   eventHandler,
                 SLInputManager&     inputManager,
                 const ImGuiEngine&  imGuiEngine,
                 ErlebAR::Resources& resources,
                 const DeviceData&   deviceData);
    bool update();
    //call when view becomes visible
    void show() { _gui.onShow(); }

private:
    TutorialGui _gui;
};

#endif //TUTORIAL_VIEW_H
