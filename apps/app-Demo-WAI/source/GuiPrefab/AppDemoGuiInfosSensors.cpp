#include <imgui.h>
#include <imgui_internal.h>

#include <Utils.h>
#include <SLApplication.h>
#include <AppDemoGuiInfosDialog.h>
#include <AppDemoGuiInfosSensors.h>

//-----------------------------------------------------------------------------
AppDemoGuiInfosSensors::AppDemoGuiInfosSensors(std::string name, bool* activator)
  : AppDemoGuiInfosDialog(name, activator)
{
}

//-----------------------------------------------------------------------------
void AppDemoGuiInfosSensors::buildInfos(SLScene* s, SLSceneView* sv)
{
    SLchar m[1024];             // message character array
    m[0]                   = 0; // set zero length
    SLVec3d offsetToOrigin = SLApplication::devLoc.originENU() - SLApplication::devLoc.locENU();
    sprintf(m + strlen(m), "Uses Rotation       : %s\n", SLApplication::devRot.isUsed() ? "yes" : "no");
    sprintf(m + strlen(m), "Orientation Pitch   : %1.0f\n", SLApplication::devRot.pitchRAD() * Utils::RAD2DEG);
    sprintf(m + strlen(m), "Orientation Yaw     : %1.0f\n", SLApplication::devRot.yawRAD() * Utils::RAD2DEG);
    sprintf(m + strlen(m), "Orientation Roll    : %1.0f\n", SLApplication::devRot.rollRAD() * Utils::RAD2DEG);
    sprintf(m + strlen(m), "Zero Yaw at Start   : %s\n", SLApplication::devRot.zeroYawAtStart() ? "yes" : "no");
    sprintf(m + strlen(m), "Start Yaw           : %1.0f\n", SLApplication::devRot.startYawRAD() * Utils::RAD2DEG);
    sprintf(m + strlen(m), "---------------------\n");
    sprintf(m + strlen(m), "Uses Location       : %s\n", SLApplication::devLoc.isUsed() ? "yes" : "no");
    sprintf(m + strlen(m), "Latitude (deg)      : %11.6f\n", SLApplication::devLoc.locLLA().x);
    sprintf(m + strlen(m), "Longitude (deg)     : %11.6f\n", SLApplication::devLoc.locLLA().y);
    sprintf(m + strlen(m), "Altitude (m)        : %11.6f\n", SLApplication::devLoc.locLLA().z);
    sprintf(m + strlen(m), "Accuracy Radius (m) : %6.1f\n", SLApplication::devLoc.locAccuracyM());
    sprintf(m + strlen(m), "Dist. to Origin (m) : %6.1f\n", offsetToOrigin.length());
    sprintf(m + strlen(m), "Max. Dist. (m)      : %6.1f\n", SLApplication::devLoc.locMaxDistanceM());
    sprintf(m + strlen(m), "Origin improve time : %6.1f sec.\n", SLApplication::devLoc.improveTime());
    sprintf(m + strlen(m), "Sun Zenit (deg)     : %6.1f sec.\n", SLApplication::devLoc.originSolarZenit());
    sprintf(m + strlen(m), "Sun Azimut (deg)    : %6.1f sec.\n", SLApplication::devLoc.originSolarAzimut());
    sprintf(m + strlen(m), "---------------------\n");

    for (auto it = SLApplication::deviceParameter.begin(); it != SLApplication::deviceParameter.end(); ++it)
        sprintf(m + strlen(m), "%s : %s\n", it->first.c_str(), it->second.c_str());

    // Switch to fixed font
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
    ImGui::Begin("Sensor Informations", _activator, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::TextUnformatted(m);
    ImGui::End();
    ImGui::PopFont();
}
