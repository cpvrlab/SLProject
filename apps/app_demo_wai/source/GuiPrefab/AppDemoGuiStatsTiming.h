#ifndef SL_IMGUI_STATS_TIMING_H
#define SL_IMGUI_STATS_TIMING_H

#include <AppDemoGuiInfosDialog.h>
#include <SL.h>
#include <SLSceneView.h>

//-----------------------------------------------------------------------------
class AppDemoGuiStatsTiming : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiStatsTiming(string name, bool* activator, ImFont* font);

    void buildInfos(SLScene* s, SLSceneView* sv) override;
};

#endif //SL_IMGUI_TRACKEDMAPPING_H