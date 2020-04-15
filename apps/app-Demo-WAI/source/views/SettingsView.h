#ifndef SETTINGS_VIEW_H
#define SETTINGS_VIEW_H

#include <string>
#include <SLInputManager.h>
#include <SLSceneView.h>
#include <SettingsGui.h>

class SettingsView : public SLSceneView
{
public:
    SettingsView(sm::EventHandler&   eventHandler,
                 SLInputManager&     inputManager,
                 ErlebAR::Resources& resources,
                 int                 screenWidth,
                 int                 screenHeight,
                 int                 dotsPerInch,
                 std::string         fontPath,
                 std::string         imguiIniPath);
    bool update();
    //call when view becomes visible
    void show() { _gui.onShow(); }

private:
    SettingsGui _gui;
};

#endif //SETTINGS_VIEW_H
