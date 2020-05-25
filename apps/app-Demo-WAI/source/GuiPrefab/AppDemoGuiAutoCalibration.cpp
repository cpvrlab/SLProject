#include <imgui.h>
#include <imgui_internal.h>

#include <WAIEvent.h>
#include <AppDemoGuiAutoCalibration.h>
#include <SLGLImGui.h>
//-----------------------------------------------------------------------------
AppDemoGuiAutoCalibration::AppDemoGuiAutoCalibration(string                              name,
                                                     bool*                               activator,
                                                     std::queue<WAIEvent*>*              eventQueue,
                                                     std::function<SENSCamera*(void)>    getCameraCB,
                                                     std::function<CVCalibration*(void)> getCalibrationCB,
                                                     ImFont*                             font)
  : AppDemoGuiInfosDialog(name, activator, font)
{
    _eventQueue     = eventQueue;
    _getCamera      = getCameraCB;
    _getCalibration = getCalibrationCB;
}

//-----------------------------------------------------------------------------
void AppDemoGuiAutoCalibration::buildInfos(SLScene* s, SLSceneView* sv)
{
    SLchar m[2550];      // message character array
    m[0]            = 0; //
    SENSCamera* cam = _getCamera();
    // clang-format off
    if (cam)
        sprintf(m + strlen(m), "Capture size: %d x %d\n", cam->config().targetWidth, cam->config().targetHeight);
    else
        sprintf(m + strlen(m), "Camera invalid\n");
 
    const CVCalibration* calib = _getCalibration();
    if (calib)
    {
        sprintf(m + strlen(m), "Horiz. FOV (deg.): %4.1f\n", calib->cameraFovHDeg());
        sprintf(m + strlen(m), "fx,fy,cx,cy     : %4.1f,%4.1f,%4.1f,%4.1f\n", calib->fx(), calib->fy(), calib->cx(), calib->cy());
    }
    else
        sprintf(m + strlen(m), "Calibration invalid\n");

    ImGui::PushFont(_font);
    ImGui::Begin("AutoCalibration", _activator, 0);

    ImGui::TextUnformatted(m);

    WAIEventAutoCalibration* event = new WAIEventAutoCalibration();

    event->tryCalibrate               = false;
    event->restoreOriginalCalibration = false;
    event->useGuessCalibration        = false;

    if (ImGui::Button("Try to calibrate", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
        event->tryCalibrate            = true;

    if (ImGui::Button("Use guess", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
        event->useGuessCalibration     = true;

    if (ImGui::Button("Use loaded calib", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
        event->restoreOriginalCalibration = true;

    _eventQueue->push(event);

    ImGui::End();
    ImGui::PopFont();
}
