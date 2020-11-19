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
    void onShow() { _gui.onShow(); }
    void onHide();

private:
    SettingsGui      _gui;
    ErlebAR::Config& _config;
};

#endif //SETTINGS_VIEW_H
