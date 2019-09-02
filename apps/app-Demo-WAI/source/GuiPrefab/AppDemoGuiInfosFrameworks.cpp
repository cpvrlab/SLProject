#include <imgui.h>
#include <imgui_internal.h>

#include <SLApplication.h>
#include <AppDemoGuiInfosDialog.h>
#include <AppDemoGuiInfosFrameworks.h>
#include <CVCapture.h>
//-----------------------------------------------------------------------------
AppDemoGuiInfosFrameworks::AppDemoGuiInfosFrameworks(std::string name, bool* activator)
  : AppDemoGuiInfosDialog(name, activator)
{ }

//-----------------------------------------------------------------------------
void AppDemoGuiInfosFrameworks::buildInfos(SLScene* s, SLSceneView* sv)
{
    SLGLState* stateGL = SLGLState::instance();
    SLchar     m[2550]; // message character array
    m[0] = 0;           // set zero length

    sprintf(m + strlen(m), "SLProject Version: %s\n", SLApplication::version.c_str());
#ifdef _DEBUG
    sprintf(m + strlen(m), "Build Config.    : Debug\n");
#else
    sprintf(m + strlen(m), "Build Config.    : Release\n");
#endif
    sprintf(m + strlen(m), "OpenGL Version   : %s\n", stateGL->glVersionNO().c_str());
    sprintf(m + strlen(m), "OpenGL Vendor    : %s\n", stateGL->glVendor().c_str());
    sprintf(m + strlen(m), "OpenGL Renderer  : %s\n", stateGL->glRenderer().c_str());
    sprintf(m + strlen(m), "GLSL Version     : %s\n", stateGL->glSLVersionNO().c_str());
    sprintf(m + strlen(m), "OpenCV Version   : %d.%d.%d\n", CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_VERSION_REVISION);
    sprintf(m + strlen(m), "OpenCV has OpenCL: %s\n", cv::ocl::haveOpenCL() ? "yes" : "no");
    sprintf(m + strlen(m), "OpenCV has AVX   : %s\n", cv::checkHardwareSupport(CV_AVX) ? "yes" : "no");
    sprintf(m + strlen(m), "OpenCV has NEON  : %s\n", cv::checkHardwareSupport(CV_NEON) ? "yes" : "no");
    sprintf(m + strlen(m), "ImGui Version    : %s\n", ImGui::GetVersion());

    // Switch to fixed font
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
    ImGui::Begin("Framework Informations", _activator, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::TextUnformatted(m);
    ImGui::End();
    ImGui::PopFont();
}

