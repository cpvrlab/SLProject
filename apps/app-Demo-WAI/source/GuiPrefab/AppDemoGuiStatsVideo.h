#ifndef SL_IMGUI_STATS_VIDEO_H
#define SL_IMGUI_STATS_VIDEO_H

#include <WAIModeOrbSlam2.h>
#include <AppDemoGuiInfosDialog.h>

class WAIApp;
//-----------------------------------------------------------------------------
class AppDemoGuiStatsVideo : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiStatsVideo(std::string name, bool* activator, WAIApp& waiApp);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

private:
    WAIApp& _waiApp;
};

#endif //SL_IMGUI_TRACKEDMAPPING_H
