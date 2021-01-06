#ifndef SL_IMGUI_STATS_DEBUG_TIMING_H
#define SL_IMGUI_STATS_DEBUG_TIMING_H

#include <AppDemoGuiInfosDialog.h>
#include <SL.h>
#include <SLSceneView.h>

//-----------------------------------------------------------------------------
class AppDemoGuiStatsDebugTiming : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiStatsDebugTiming(string name, bool* activator, ImFont* font);

    void buildInfos(SLScene* s, SLSceneView* sv) override;
};

#endif //SL_IMGUI_TRACKEDMAPPING_H