#include <imgui.h>
#include <imgui_internal.h>

#include <SLApplication.h>
#include <AppDemoGuiInfosDialog.h>
#include <AppDemoGuiStatsVideo.h>
#include <CVCapture.h>
//-----------------------------------------------------------------------------
AppDemoGuiStatsVideo::AppDemoGuiStatsVideo(std::string name, CVCalibration* calib, bool* activator)
  : AppDemoGuiInfosDialog(name, activator)
{
    _calib = calib;
}

//-----------------------------------------------------------------------------
void AppDemoGuiStatsVideo::buildInfos(SLScene* s, SLSceneView* sv)
{
    _calib = CVCapture::instance()->activeCalib;

    SLchar m[2550]; // message character array
    m[0] = 0;       // set zero length

    CVSize      capSize = CVCapture::instance()->captureSize;
    CVVideoType vt      = CVCapture::instance()->videoType();
    sprintf(m + strlen(m), "Video Type    : %s\n", vt == VT_NONE ? "None" : vt == VT_MAIN ? "Main Camera" : vt == VT_FILE ? "File" : "Secondary Camera");
    sprintf(m + strlen(m), "Display size  : %d x %d\n", CVCapture::instance()->lastFrame.cols, CVCapture::instance()->lastFrame.rows);
    sprintf(m + strlen(m), "Capture size  : %d x %d\n", capSize.width, capSize.height);

    if (_calib != nullptr)
    {
        sprintf(m + strlen(m), "Vert. FOV (deg.)    : %4.1f\n", _calib->cameraFovVDeg());
        sprintf(m + strlen(m), "fx,fy,cx,cy   : %4.1f,%4.1f,%4.1f,%4.1f\n", _calib->fx(), _calib->fy(), _calib->cx(), _calib->cy());
        sprintf(m + strlen(m), "k1,k2,p1,p2   : %4.2f,%4.2f,%4.2f,%4.2f\n", _calib->k1(), _calib->k2(), _calib->p1(), _calib->p2());
        sprintf(m + strlen(m), "Calib. file   : %s\n", (_calib->calibDir() + _calib->calibFileName()).c_str());
        sprintf(m + strlen(m), "Calib. state  : %s\n", _calib->stateStr().c_str());
    }

    // Switch to fixed font
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
    ImGui::Begin("Video", _activator, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::TextUnformatted(m);
    ImGui::End();
    ImGui::PopFont();
}
