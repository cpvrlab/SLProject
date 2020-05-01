#ifndef SL_IMGUI_STATS_VIDEO_H
#define SL_IMGUI_STATS_VIDEO_H

#include <AppDemoGuiInfosDialog.h>

class WAIApp;
class SENSCamera;
class CVCalibration;

//-----------------------------------------------------------------------------
class AppDemoGuiStatsVideo : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiStatsVideo(std::string                         name,
                         bool*                               activator,
                         ImFont*                             font,
                         std::function<SENSCamera*(void)>    getCameraCB,
                         std::function<CVCalibration*(void)> getCalibrationCB);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

private:
    std::function<SENSCamera*(void)>    _getCamera;
    std::function<CVCalibration*(void)> _getCalibration;
};

#endif //SL_IMGUI_TRACKEDMAPPING_H
