#ifndef SL_IMGUI_INFOS_FRAMEWORKS_H
#define SL_IMGUI_INFOS_FRAMEWORKS_H

#include <AppDemoGuiInfosDialog.h>
#include <WAICalibration.h>

//-----------------------------------------------------------------------------
class AppDemoGuiInfosFrameworks : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiInfosFrameworks(std::string name, bool* activator, ImFont* font);

    void buildInfos(SLScene* s, SLSceneView* sv) override;
};

#endif //SL_IMGUI_TRACKEDMAPPING_H
