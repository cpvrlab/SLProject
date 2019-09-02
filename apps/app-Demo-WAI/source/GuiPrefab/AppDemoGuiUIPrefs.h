#ifndef SL_IMGUI_UI_PREFS_H
#define SL_IMGUI_UI_PREFS_H

#include <WAIModeOrbSlam2.h>
#include <AppDemoGuiInfosDialog.h>
#include <WAICalibration.h>
#include <SL.h>
#include <AppDemoGuiPrefs.h>

//-----------------------------------------------------------------------------
class AppDemoGuiUIPrefs : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiUIPrefs(std::string name, GUIPreferences* prefs, bool* activator);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

    private:
    GUIPreferences* _prefs;
};

#endif //SL_IMGUI_TRACKEDMAPPING_H
