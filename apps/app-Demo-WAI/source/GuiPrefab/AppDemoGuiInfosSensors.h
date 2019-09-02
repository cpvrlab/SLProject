#ifndef SL_IMGUI_INFOS_SENSORS_H
#define SL_IMGUI_INFOS_SENSORS_H

#include <WAIModeOrbSlam2.h>
#include <AppDemoGuiInfosDialog.h>
#include <WAICalibration.h>

//-----------------------------------------------------------------------------
class AppDemoGuiInfosSensors : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiInfosSensors(std::string name, bool * activator);

    void buildInfos(SLScene* s, SLSceneView* sv) override;
};

#endif //SL_IMGUI_TRACKEDMAPPING_H
