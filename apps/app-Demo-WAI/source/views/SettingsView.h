#ifndef SETTINGS_VIEW_H
#define SETTINGS_VIEW_H

#include <string>
#include <SLInputManager.h>
#include <SLSceneView.h>
#include <SettingsGui.h>

class SettingsView : public SLSceneView
{
public:
    SettingsView(SLInputManager& inputManager,
                 int             screenWidth,
                 int             screenHeight,
                 int             dotsPerInch,
                 std::string     fontPath,
                 std::string     imguiIniPath);
    bool update();

private:
    SettingsGui _gui;
};

#endif //SETTINGS_VIEW_H
