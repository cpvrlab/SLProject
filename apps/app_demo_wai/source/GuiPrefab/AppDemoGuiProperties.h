#ifndef SL_IMGUI_PROPERTIES_H
#define SL_IMGUI_PROPERTIES_H

#include <SLScene.h>
#include <AppDemoGuiInfosDialog.h>
#include <WAICalibration.h>

//-----------------------------------------------------------------------------
class AppDemoGuiProperties : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiProperties(std::string name, bool* activator, ImFont* font);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

private:
};

#endif //SL_IMGUI_TRACKEDMAPPING_H
