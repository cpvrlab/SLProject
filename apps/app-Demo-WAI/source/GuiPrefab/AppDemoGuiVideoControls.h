#ifndef SL_IMGUI_VIDEOCONTROLS_H
#define SL_IMGUI_VIDEOCONTROLS_H

#include <AppDemoGuiInfosDialog.h>

class AppDemoGuiVideoControls : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiVideoControls(const std::string& name, bool* activator);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

    private:
};

#endif