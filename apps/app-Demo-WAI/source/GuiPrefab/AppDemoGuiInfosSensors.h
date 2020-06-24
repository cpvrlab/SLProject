#ifndef SL_IMGUI_INFOS_SENSORS_H
#define SL_IMGUI_INFOS_SENSORS_H

#include <AppDemoGuiInfosDialog.h>
#include <WAICalibration.h>

//-----------------------------------------------------------------------------
class AppDemoGuiInfosSensors : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiInfosSensors(std::string name, bool* activator, ImFont* font);

    void buildInfos(SLScene* s, SLSceneView* sv) override;
};

#endif //SL_IMGUI_TRACKEDMAPPING_H
