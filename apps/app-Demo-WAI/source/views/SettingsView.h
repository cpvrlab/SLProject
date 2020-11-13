#ifndef SETTINGS_VIEW_H
#define SETTINGS_VIEW_H

#include <string>
#include <SLInputManager.h>
#include <SLSceneView.h>
#include <SettingsGui.h>

class SettingsView : public SLSceneView
{
public:
    SettingsView(sm::EventHandler&  eventHandler,
                 SLInputManager&    inputManager,
                 const ImGuiEngine& imGuiEngine,
                 ErlebAR::Config&   config,
                 const DeviceData&  deviceData);
    bool update();
    //call when view becomes visible
    void show() { _gui.onShow(); }

private:
    SettingsGui _gui;
};

#endif //SETTINGS_VIEW_H
