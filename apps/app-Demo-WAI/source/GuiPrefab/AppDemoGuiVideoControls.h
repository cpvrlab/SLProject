#ifndef SL_IMGUI_VIDEOCONTROLS_H
#define SL_IMGUI_VIDEOCONTROLS_H

#include <AppDemoGuiInfosDialog.h>

class WAIApp;

class AppDemoGuiVideoControls : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiVideoControls(const std::string& name, bool* activator, WAIApp& waiApp);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

private:
    WAIApp& _waiApp;
};

#endif
