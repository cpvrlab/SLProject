#ifndef SL_IMGUI_STATS_TIMING_H
#define SL_IMGUI_STATS_TIMING_H

#include <WAIModeOrbSlam2.h>
#include <AppDemoGuiInfosDialog.h>
#include <SL.h>
#include <SLSceneView.h>

//-----------------------------------------------------------------------------
class AppDemoGuiStatsTiming : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiStatsTiming(string name, bool* activator);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

};

#endif //SL_IMGUI_TRACKEDMAPPING_H
