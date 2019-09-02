#ifndef SL_IMGUI_INFOS_FRAMEWORKS_H
#define SL_IMGUI_INFOS_FRAMEWORKS_H

#include <WAIModeOrbSlam2.h>
#include <AppDemoGuiInfosDialog.h>
#include <WAICalibration.h>

//-----------------------------------------------------------------------------
class AppDemoGuiInfosFrameworks : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiInfosFrameworks(std::string name, bool* activator);

    void buildInfos(SLScene* s, SLSceneView* sv) override;
};

#endif //SL_IMGUI_TRACKEDMAPPING_H
