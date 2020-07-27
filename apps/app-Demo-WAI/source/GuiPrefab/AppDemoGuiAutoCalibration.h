#ifndef SL_IMGUI_AUTOCALIBRATION_H
#define SL_IMGUI_AUTOCALIBRATION_H

#include <AppDemoGuiInfosDialog.h>
#include <SLSceneView.h>
#include <SLScene.h>
#include <sens/SENSCalibration.h>
#include <sens/SENSCamera.h>

//-----------------------------------------------------------------------------
class AppDemoGuiAutoCalibration : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiAutoCalibration(string                                      name,
                              bool*                                       activator,
                              std::queue<WAIEvent*>*                      eventQueue,
                              std::function<SENSCamera*(void)>            getCameraCB,
                              std::function<const SENSCalibration*(void)> getCalibrationCB,
                              ImFont*                                     font);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

private:
    bool                                        _tryAutoCalibrate;
    std::queue<WAIEvent*>*                      _eventQueue;
    std::function<SENSCamera*(void)>            _getCamera;
    std::function<const SENSCalibration*(void)> _getCalibration;
};

#endif //SL_IMGUI_TRACKEDMAPPING_H
