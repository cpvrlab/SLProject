#ifndef SL_IMGUI_STATS_VIDEO_H
#define SL_IMGUI_STATS_VIDEO_H

#include <WAIModeOrbSlam2.h>
#include <AppDemoGuiInfosDialog.h>
#include <WAICalibration.h>

//-----------------------------------------------------------------------------
class AppDemoGuiStatsVideo : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiStatsVideo(std::string name, WAICalibration* wc, bool* activator);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

    private:

    WAICalibration* _wc;
};

#endif //SL_IMGUI_TRACKEDMAPPING_H
