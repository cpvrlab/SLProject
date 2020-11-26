#ifndef ABOUT_VIEW_H
#define ABOUT_VIEW_H

#include <string>
#include <SLInputManager.h>
#include <SLSceneView.h>
#include <AboutGui.h>
#include <ErlebAR.h>

class AboutView : public SLSceneView
{
public:
    AboutView(sm::EventHandler&  eventHandler,
              SLInputManager&    inputManager,
              const ImGuiEngine& imGuiEngine,
              ErlebAR::Config&   config,
              const DeviceData&  deviceData);
    bool update();
    //call when view becomes visible
    void show() { _gui.onShow(); }

private:
    AboutGui _gui;
};

#endif //ABOUT_VIEW_H
