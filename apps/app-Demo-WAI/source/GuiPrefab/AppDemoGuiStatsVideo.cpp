#include <AppDemoGuiStatsVideo.h>

#include <sens/SENSCamera.h>
#include <sens/SENSCalibration.h>

//-----------------------------------------------------------------------------
AppDemoGuiStatsVideo::AppDemoGuiStatsVideo(std::string                                 name,
                                           bool*                                       activator,
                                           ImFont*                                     font,
                                           std::function<SENSCvCamera*(void)>          getCameraCB,
                                           std::function<const SENSCalibration*(void)> getCalibrationCB)
  : AppDemoGuiInfosDialog(name, activator, font),
    _getCamera(getCameraCB),
    _getCalibration(getCalibrationCB)
{
}

//-----------------------------------------------------------------------------
void AppDemoGuiStatsVideo::buildInfos(SLScene* s, SLSceneView* sv)
{
    SLchar m[2550]; // message character array
    m[0] = 0;       // set zero length

    SENSCvCamera* cam = _getCamera();
    // clang-format off
    if (cam)
    {
        sprintf(m + strlen(m), "Capture size: %d x %d\n", cam->config().targetWidth, cam->config().targetHeight);
    }
    else
    {
        sprintf(m + strlen(m), "Camera invalid\n");
    }

    const SENSCalibration* calib = _getCalibration();
    if (calib)
    {
        sprintf(m + strlen(m),     "Vert. FOV (deg.): %4.1f\n", calib->cameraFovVDeg());
        sprintf(m + strlen(m),     "fx,fy,cx,cy     : %4.1f,%4.1f,%4.1f,%4.1f\n", calib->fx(), calib->fy(), calib->cx(), calib->cy());

        int distortionSize = calib->distortion().rows;
        sprintf(m + strlen(m), "distortion (*10e-2)\n");
        const float f = 100.f;
        sprintf(m + strlen(m),     "k1,k2      : %4.2f,%4.2f\n", calib->k1() * f, calib->k2() * f);
        sprintf(m + strlen(m),     "p1,p2      : %4.2f,%4.2f\n", calib->p1() * f, calib->p2() * f);
        if (distortionSize >= 8)
            sprintf(m + strlen(m), "k3,k4,k5,k6: %4.2f,%4.2f,%4.2f,%4.2f\n", calib->k3() * f, calib->k4() * f, calib->k5() * f, calib->k6() * f);
        else
            sprintf(m + strlen(m), "k3         : %4.2f\n", calib->k3() * f);

        if (distortionSize >= 12)
            sprintf(m + strlen(m), "s1,s2,s3,s4: %4.2f,%4.2f,%4.2f,%4.2f\n", calib->s1() * f, calib->s2() * f, calib->s3() * f, calib->s4() * f);
        if (distortionSize >= 14)
            sprintf(m + strlen(m), "tauX,tauY  : %4.2f,%4.2f\n", calib->tauX() * f, calib->tauY() * f);

        //sprintf(m + strlen(m), "calib-> file   : %s\n", (calib->calibDir() + calib->calibFileName()).c_str());
        sprintf(m + strlen(m), "calib time : %s\n", calib->calibrationTime().c_str());
        sprintf(m + strlen(m), "calib state: %s\n", calib->stateStr().c_str());
        sprintf(m + strlen(m), "Num. caps  : %d\n", calib->numCapturedImgs());
    }
    else
    {
        sprintf(m + strlen(m), "Calibration invalid\n");
    }

    // clang-format on

    // Switch to fixed font
    ImGui::PushFont(_font);

    ImGui::Begin("Video", _activator);
    ImGui::TextUnformatted(m);
    ImGui::End();
    ImGui::PopFont();
}
