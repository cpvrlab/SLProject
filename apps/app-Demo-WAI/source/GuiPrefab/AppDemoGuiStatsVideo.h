#ifndef SL_IMGUI_STATS_VIDEO_H
#define SL_IMGUI_STATS_VIDEO_H

#include <WAIModeOrbSlam2.h>
#include <AppDemoGuiInfosDialog.h>
#include <CVCalibration.h>

//-----------------------------------------------------------------------------
class AppDemoGuiStatsVideo : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiStatsVideo(std::string name, CVCalibration* calib, bool* activator);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

    private:
    CVCalibration* _calib;
};

#endif //SL_IMGUI_TRACKEDMAPPING_H
