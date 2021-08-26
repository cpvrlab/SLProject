//#############################################################################
//  File:      AppDemoGui.cpp
//  Purpose:   UI with the ImGUI framework fully rendered in OpenGL 3+
//  Author:    Marcus Hudritsch
//  Date:      Summer 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <AppDemoGui.h>
#include <SLAnimPlayback.h>
#include <AppDemo.h>
#include <CVCapture.h>
#include <cv/CVImage.h>
#include <cv/CVTrackedFeatures.h>
#include <SLGLDepthBuffer.h>
#include <SLGLProgramManager.h>
#include <SLGLShader.h>
#include <SLGLTexture.h>
#include <SLInterface.h>
#include <SLDeviceLocation.h>
#include <SLDeviceRotation.h>
#include <SLLightDirect.h>
#include <SLLightRect.h>
#include <SLLightSpot.h>
#include <SLShadowMap.h>
#include <SLMaterial.h>
#include <SLMesh.h>
#include <SLNode.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLTexColorLUT.h>
#include <SLGLImGui.h>
#include <SLProjectScene.h>
#include <SLHorizonNode.h>
#include <AverageTiming.h>
#include <imgui.h>
#include <ftplib.h>
#include <HttpUtils.h>
#include <ZipUtils.h>
#include <Instrumentor.h>

#ifdef SL_BUILD_WAI
#    include <Eigen/Dense>
#endif

//-----------------------------------------------------------------------------
extern CVTracked*   tracker;      // Global pointer declared in AppDemoTracking
extern SLNode*      trackedNode;  // Global pointer declared in AppDemoTracking
extern SLGLTexture* gTexMRI3D;    // Global pointer declared in AppDemoLoad
extern SLNode*      gDragonModel; // Global pointer declared in AppDemoLoad

//#define IM_ARRAYSIZE(_ARR) ((int)(sizeof(_ARR) / sizeof(*_ARR)))

//-----------------------------------------------------------------------------
//! Vector getter callback for combo and listbox with std::vector<std::string>
static auto vectorGetter = [](void* vec, int idx, const char** out_text) {
    auto& vector = *(SLVstring*)vec;
    if (idx < 0 || idx >= (int)vector.size())
        return false;

    *out_text = vector.at((SLuint)idx).c_str();
    return true;
};

//-----------------------------------------------------------------------------
//! Combobox that allows to pass the items as a string vector
bool myComboBox(const char* label, int* currIndex, SLVstring& values)
{
    if (values.empty())
        return false;

    return ImGui::Combo(label,
                        currIndex,
                        vectorGetter,
                        (void*)&values,
                        (int)values.size());
}
//-----------------------------------------------------------------------------
//! Centers the next ImGui window in the parent
void centerNextWindow(SLSceneView* sv,
                      SLfloat      widthPC  = 0.9f,
                      SLfloat      heightPC = 0.9f)
{
    SLfloat width   = (SLfloat)sv->viewportW() * widthPC;
    SLfloat height  = (SLfloat)sv->viewportH() * heightPC;
    SLfloat offsetX = ((SLfloat)sv->viewportW() - width) * 0.5f;
    SLfloat offsetY = ((SLfloat)sv->viewportH() - height) * 0.5f;
    ImGui::SetNextWindowSize(ImVec2(width, height), ImGuiCond_Always);
    ImGui::SetNextWindowPos(ImVec2(offsetX, offsetY), ImGuiCond_Always);
}
//-----------------------------------------------------------------------------
// Init global static variables
SLstring    AppDemoGui::configTime          = "-";
SLbool      AppDemoGui::showDockSpace       = true;
SLbool      AppDemoGui::showProgress        = false;
SLbool      AppDemoGui::showAbout           = false;
SLbool      AppDemoGui::showHelp            = false;
SLbool      AppDemoGui::showHelpCalibration = false;
SLbool      AppDemoGui::showCredits         = false;
SLbool      AppDemoGui::showStatsTiming     = false;
SLbool      AppDemoGui::showStatsScene      = false;
SLbool      AppDemoGui::showStatsVideo      = false;
SLbool      AppDemoGui::showStatsWAI        = false;
SLbool      AppDemoGui::showImGuiMetrics    = false;
SLbool      AppDemoGui::showInfosScene      = false;
SLbool      AppDemoGui::showInfosSensors    = false;
SLbool      AppDemoGui::showInfosDevice     = false;
SLbool      AppDemoGui::showSceneGraph      = false;
SLbool      AppDemoGui::showProperties      = false;
SLbool      AppDemoGui::showErlebAR         = false;
SLbool      AppDemoGui::showUIPrefs         = false;
SLbool      AppDemoGui::showTransform       = false;
SLbool      AppDemoGui::showDateAndTime     = false;
std::time_t AppDemoGui::adjustedTime        = 0;
SLbool      AppDemoGui::_horizonVisuEnabled = false;
SLbool      AppDemoGui::hideUI              = false;

// Scene node for Christoffel objects
static SLNode* bern        = nullptr;
static SLNode* balda_stahl = nullptr;
static SLNode* balda_glas  = nullptr;
static SLNode* chrAlt      = nullptr;
static SLNode* chrNeu      = nullptr;

// Temp. transform node
static SLTransformNode* transformNode = nullptr;

SLstring AppDemoGui::infoAbout = R"(
Welcome to the SLProject demo app. It is developed at the Computer Science Department of the Bern University of Applied Sciences.
The app shows what you can learn in two semesters about 3D computer graphics in real time rendering and ray tracing. The framework is developed in C++ with OpenGL ES so that it can run also on mobile devices.
Ray tracing provides in addition high quality transparencies, reflections and soft shadows. Click to close and use the menu to choose different scenes and view settings.
For more information please visit: https://github.com/cpvrlab/SLProject
)";

SLstring AppDemoGui::infoCredits = R"(
Contributors since 2005 in alphabetic order:
Martin Christen, Jan Dellsperger, Manuel Frischknecht, Luc Girod, Michael Goettlicher, Michael Schertenleib, Stefan Thoeni, Timo Tschanz, Marc Wacker, Pascal Zingg

Credits for external libraries:
- assimp: assimp.sourceforge.net
- eigen: eigen.tuxfamily.org
- imgui: github.com/ocornut/imgui
- gl3w: https://github.com/skaslev/gl3w
- glfw: glfw.org
- g2o: github.com/RainerKuemmerle/g2o
- ktx: khronos.org/ktx
- libigl: libigl.github.io
- ORB-SLAM2: github.com/raulmur/ORB_SLAM2
- OpenCV: opencv.org
- OpenGL: opengl.org
- OpenSSL: openssl.org
- spa: midcdmz.nrel.gov/spa
- zlib: zlib.net
)";

SLstring AppDemoGui::infoHelp = R"(
Help for mouse or finger control:
- Use left mouse or your finger to rotate the scene
- Use mouse-wheel or pinch 2 fingers to go forward/backward
- Use middle-mouse or 2 fingers to move sidewards/up-down
- Double click or double tap to select object
- CTRL-mouse to select vertices of objects
- See keyboard shortcuts behind menu commands
- Check out the different test scenes under File > Load Test Scene
- You can open and dock additional windows from the menu Infos.
)";

SLstring AppDemoGui::infoCalibrate = R"(
The calibration process requires a chessboard image to be printed and glued on a flat board. You can find the PDF with the chessboard image on:
https://github.com/cpvrlab/SLProject/tree/master/data/calibrations/
For a calibration you have to take 20 images with detected inner chessboard corners. To take an image you have to click with the mouse
or tap with finger into the screen. View the chessboard from the side so that the inner corners cover the full image. Hold the camera or board really still
before taking the picture.
You can mirror the video image under Preferences > Video. You can check the distance to the chessboard in the dialog Stats. on Video.
After calibration the yellow wireframe cube should stick on the chessboard. Please close first this info dialog on the top-left.
)";

//-----------------------------------------------------------------------------
off64_t ftpXferSizeMax = 0;

//-----------------------------------------------------------------------------
// Callback routine for FTP file transfer to progress the progressbar
int ftpCallbackXfer(off64_t xfered, void* arg)
{
    if (ftpXferSizeMax)
    {
        int xferedPC = (int)((float)xfered / (float)ftpXferSizeMax * 100.0f);
        // cout << "Bytes transferred: " << xfered << " (" << xferedPC << ")" << endl;
        AppDemo::jobProgressNum(xferedPC);
    }
    else
        cout << "Bytes transferred: " << xfered << endl;
    return xfered ? 1 : 0;
}

//-----------------------------------------------------------------------------
void AppDemoGui::clear()
{
    _horizonVisuEnabled = false;
}
//-----------------------------------------------------------------------------
//! This is the main building function for the GUI of the Demo apps
/*! Is is passed to the AppDemoGui::build function in main of the app-Demo-SLProject
 app. This function will be called once per frame roughly at the end of
 SLSceneView::onPaint in SLSceneView::draw2DGL by calling ImGui::Render.\n
 See also the comments on SLGLImGui.
 */
void AppDemoGui::build(SLProjectScene* s, SLSceneView* sv)
{
    PROFILE_FUNCTION();

    if (AppDemoGui::hideUI ||
        (sv->camera() && sv->camera()->projection() == P_stereoSideBySideD))
    {
        // So far no UI in distorted stereo projection
        buildMenuContext(s, sv);
    }
    else
    {

        ///////////////////////////////////
        // Show modeless fullscreen dialogs
        ///////////////////////////////////

        // if parallel jobs are running show only the progress informations
        if (AppDemo::jobIsRunning)
        {
            centerNextWindow(sv, 0.9f, 0.5f);
            ImGui::Begin("Parallel Job in Progress",
                         &showProgress,
                         ImGuiWindowFlags_NoTitleBar);
            ImGui::Text("Parallel Job in Progress:");
            ImGui::Separator();
            ImGui::Text("%s", AppDemo::jobProgressMsg().c_str());
            if (AppDemo::jobProgressMax() > 0)
            {
                float num = (float)AppDemo::jobProgressNum();
                float max = (float)AppDemo::jobProgressMax();
                ImGui::ProgressBar(num / max);
            }
            else
            {
                ImGui::Text("Progress: %c", "|/-\\"[(int)(ImGui::GetTime() / 0.05f) & 3]);
            }

            ImGui::Separator();
            ImGui::Text("Parallel Jobs to follow: %lu",
                        AppDemo::jobsToBeThreaded.size());
            ImGui::Text("Sequential Jobs to follow: %lu",
                        AppDemo::jobsToFollowInMain.size());
            ImGui::End();
            return;
        }
        else
        {
            if (showDockSpace)
            {
                static bool               opt_fullscreen_persistant = true;
                bool                      opt_fullscreen            = opt_fullscreen_persistant;
                static ImGuiDockNodeFlags dockspace_flags           = ImGuiDockNodeFlags_PassthruCentralNode;

                // We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
                // because it would be confusing to have two docking targets within each others.
                ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;
                if (opt_fullscreen)
                {
                    ImGuiViewport* viewport = ImGui::GetMainViewport();
                    ImGui::SetNextWindowPos(viewport->GetWorkPos());
                    ImGui::SetNextWindowSize(viewport->GetWorkSize());
                    ImGui::SetNextWindowViewport(viewport->ID);
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
                    window_flags |= ImGuiWindowFlags_NoTitleBar |
                                    ImGuiWindowFlags_NoCollapse |
                                    ImGuiWindowFlags_NoResize |
                                    ImGuiWindowFlags_NoMove |
                                    ImGuiWindowFlags_NoBringToFrontOnFocus |
                                    ImGuiWindowFlags_NoNavFocus;
                }

                // When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background
                // and handle the pass-thru hole, so we ask Begin() to not render a background.
                if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
                    window_flags |= ImGuiWindowFlags_NoBackground;

                // Important: note that we proceed even if Begin() returns false (aka window is collapsed).
                // This is because we want to keep our DockSpace() active. If a DockSpace() is inactive,
                // all active windows docked into it will lose their parent and become undocked.
                // We cannot preserve the docking relationship between an active window and an inactive docking, otherwise
                // any change of dockspace/settings would lead to windows being stuck in limbo and never being visible.
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
                ImGui::Begin("DockSpace Demo", &showDockSpace, window_flags);
                ImGui::PopStyleVar();

                if (opt_fullscreen)
                    ImGui::PopStyleVar(2);

                // DockSpace
                ImGuiIO& io = ImGui::GetIO();
                if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
                {
                    ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
                    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
                }

                ImGui::End();
            }

            if (showAbout)
            {
                centerNextWindow(sv);
                ImGui::Begin("About SLProject", &showAbout, ImGuiWindowFlags_NoResize);
                ImGui::Text("Version: %s", AppDemo::version.c_str());
                ImGui::Text("Configuration: %s", AppDemo::configuration.c_str());
                ImGui::Separator();
                ImGui::Text("Git Branch: %s (Commit: %s)", AppDemo::gitBranch.c_str(), AppDemo::gitCommit.c_str());
                ImGui::Text("Git Date: %s", AppDemo::gitDate.c_str());
                ImGui::Separator();
                ImGui::TextWrapped("%s", infoAbout.c_str());
                ImGui::End();
                return;
            }

            if (showHelp)
            {
                centerNextWindow(sv);
                ImGui::Begin("Help on Interaction", &showHelp, ImGuiWindowFlags_NoResize);
                ImGui::TextWrapped("%s", infoHelp.c_str());
                ImGui::End();
                return;
            }

            if (showHelpCalibration)
            {
                centerNextWindow(sv);
                ImGui::Begin("Help on Camera Calibration", &showHelpCalibration, ImGuiWindowFlags_NoResize);
                ImGui::TextWrapped("%s", infoCalibrate.c_str());
                ImGui::End();
                return;
            }

            if (showCredits)
            {
                centerNextWindow(sv);
                ImGui::Begin("Credits for all Contributors and external Libraries", &showCredits, ImGuiWindowFlags_NoResize);
                ImGui::TextWrapped("%s", infoCredits.c_str());
                ImGui::End();
                return;
            }

            //////////////////
            // Show rest modal
            //////////////////

            buildMenuBar(s, sv);

            buildMenuContext(s, sv);

            if (showStatsTiming)
            {
                SLRenderType rType = sv->renderType();
                SLfloat      ft    = s->frameTimesMS().average();
                SLchar       m[2550]; // message character array
                m[0] = 0;             // set zero length

                if (rType == RT_gl)
                {
                    // Get averages from average variables (see Averaged)
                    SLfloat captureTime    = CVCapture::instance()->captureTimesMS().average();
                    SLfloat updateTime     = s->updateTimesMS().average();
                    SLfloat trackingTime   = CVTracked::trackingTimesMS.average();
                    SLfloat detectTime     = CVTracked::detectTimesMS.average();
                    SLfloat detect1Time    = CVTracked::detect1TimesMS.average();
                    SLfloat detect2Time    = CVTracked::detect2TimesMS.average();
                    SLfloat matchTime      = CVTracked::matchTimesMS.average();
                    SLfloat optFlowTime    = CVTracked::optFlowTimesMS.average();
                    SLfloat poseTime       = CVTracked::poseTimesMS.average();
                    SLfloat updateAnimTime = s->updateAnimTimesMS().average();
                    SLfloat updateAABBTime = s->updateAnimTimesMS().average();
                    SLfloat shadowMapTime  = sv->shadowMapTimeMS().average();
                    SLfloat cullTime       = sv->cullTimesMS().average();
                    SLfloat draw3DTime     = sv->draw3DTimesMS().average();
                    SLfloat draw2DTime     = sv->draw2DTimesMS().average();

                    // Calculate percentage from frame time
                    SLfloat captureTimePC    = Utils::clamp(captureTime / ft * 100.0f, 0.0f, 100.0f);
                    SLfloat updateTimePC     = Utils::clamp(updateTime / ft * 100.0f, 0.0f, 100.0f);
                    SLfloat trackingTimePC   = Utils::clamp(trackingTime / ft * 100.0f, 0.0f, 100.0f);
                    SLfloat detectTimePC     = Utils::clamp(detectTime / ft * 100.0f, 0.0f, 100.0f);
                    SLfloat matchTimePC      = Utils::clamp(matchTime / ft * 100.0f, 0.0f, 100.0f);
                    SLfloat optFlowTimePC    = Utils::clamp(optFlowTime / ft * 100.0f, 0.0f, 100.0f);
                    SLfloat poseTimePC       = Utils::clamp(poseTime / ft * 100.0f, 0.0f, 100.0f);
                    SLfloat updateAnimTimePC = Utils::clamp(updateAnimTime / ft * 100.0f, 0.0f, 100.0f);
                    SLfloat updateAABBTimePC = Utils::clamp(updateAABBTime / ft * 100.0f, 0.0f, 100.0f);
                    SLfloat shadowMapTimePC  = Utils::clamp(shadowMapTime / ft * 100.0f, 0.0f, 100.0f);
                    SLfloat draw3DTimePC     = Utils::clamp(draw3DTime / ft * 100.0f, 0.0f, 100.0f);
                    SLfloat draw2DTimePC     = Utils::clamp(draw2DTime / ft * 100.0f, 0.0f, 100.0f);
                    SLfloat cullTimePC       = Utils::clamp(cullTime / ft * 100.0f, 0.0f, 100.0f);

                    sprintf(m + strlen(m), "Renderer   : OpenGL\n");
                    sprintf(m + strlen(m), "Load time  : %5.1f ms\n", s->loadTimeMS());
                    sprintf(m + strlen(m), "Window size: %d x %d\n", sv->viewportW(), sv->viewportH());
                    sprintf(m + strlen(m), "Frambuffer : %d x %d\n", (int)(sv->viewportW() * sv->scr2fbY()), (int)(sv->viewportH() * sv->scr2fbX()));
                    sprintf(m + strlen(m), "Drawcalls  : %d\n", SLGLVertexArray::totalDrawCalls);
                    sprintf(m + strlen(m), "Primitives : %d\n", SLGLVertexArray::totalPrimitivesRendered);
                    sprintf(m + strlen(m), "FPS        : %5.1f\n", s->fps());
                    sprintf(m + strlen(m), "Frame time : %5.1f ms (100%%)\n", ft);
                    sprintf(m + strlen(m), " Capture   : %5.1f ms (%3d%%)\n", captureTime, (SLint)captureTimePC);
                    sprintf(m + strlen(m), " Update    : %5.1f ms (%3d%%)\n", updateTime, (SLint)updateTimePC);
                    sprintf(m + strlen(m), "  Anim.    : %5.1f ms (%3d%%)\n", updateAnimTime, (SLint)updateAnimTimePC);
                    sprintf(m + strlen(m), "  AABB     : %5.1f ms (%3d%%)\n", updateAABBTime, (SLint)updateAABBTimePC);
                    sprintf(m + strlen(m), "  Tracking : %5.1f ms (%3d%%)\n", trackingTime, (SLint)trackingTimePC);
                    sprintf(m + strlen(m), "   Detect  : %5.1f ms (%3d%%)\n", detectTime, (SLint)detectTimePC);
                    sprintf(m + strlen(m), "    Det1   : %5.1f ms\n", detect1Time);
                    sprintf(m + strlen(m), "    Det2   : %5.1f ms\n", detect2Time);
                    sprintf(m + strlen(m), "   Match   : %5.1f ms (%3d%%)\n", matchTime, (SLint)matchTimePC);
                    sprintf(m + strlen(m), "   OptFlow : %5.1f ms (%3d%%)\n", optFlowTime, (SLint)optFlowTimePC);
                    sprintf(m + strlen(m), "   Pose    : %5.1f ms (%3d%%)\n", poseTime, (SLint)poseTimePC);
                    sprintf(m + strlen(m), " Shadows   : %5.1f ms (%3d%%)\n", shadowMapTime, (SLint)shadowMapTimePC);
                    sprintf(m + strlen(m), " Culling   : %5.1f ms (%3d%%)\n", cullTime, (SLint)cullTimePC);
                    sprintf(m + strlen(m), " Drawing 3D: %5.1f ms (%3d%%)\n", draw3DTime, (SLint)draw3DTimePC);
                    sprintf(m + strlen(m), " Drawing 2D: %5.1f ms (%3d%%)\n", draw2DTime, (SLint)draw2DTimePC);
                }
                else if (rType == RT_rt)
                {
                    SLRaytracer* rt           = sv->raytracer();
                    SLint        rtWidth      = (SLint)(sv->viewportW() * rt->resolutionFactor());
                    SLint        rtHeight     = (SLint)(sv->viewportH() * rt->resolutionFactor());
                    SLuint       rayPrimaries = (SLuint)(rtWidth * rtHeight);
                    SLuint       rayTotal     = SLRay::totalNumRays();
                    SLfloat      renderSec    = rt->renderSec();
                    SLfloat      fps          = renderSec > 0.001f ? 1.0f / rt->renderSec() : 0.0f;

                    sprintf(m + strlen(m), "Renderer   :Ray Tracer\n");
                    sprintf(m + strlen(m), "Progress   :%3d%%\n", rt->progressPC());
                    sprintf(m + strlen(m), "Frame size :%d x %d\n", rtWidth, rtHeight);
                    sprintf(m + strlen(m), "FPS        :%0.2f\n", fps);
                    sprintf(m + strlen(m), "Frame Time :%0.3f sec.\n", renderSec);
                    sprintf(m + strlen(m), "Rays per ms:%0.0f\n", rt->raysPerMS());
                    sprintf(m + strlen(m), "AA Pixels  :%d (%d%%)\n", SLRay::subsampledPixels, (int)((float)SLRay::subsampledPixels / (float)rayPrimaries * 100.0f));
                    sprintf(m + strlen(m), "Threads    :%d\n", rt->numThreads());
                    sprintf(m + strlen(m), "---------------------------\n");
                    sprintf(m + strlen(m), "Total rays :%8d (%3d%%)\n", rayTotal, 100);
                    sprintf(m + strlen(m), "  Primary  :%8d (%3d%%)\n", rayPrimaries, (int)((float)rayPrimaries / (float)rayTotal * 100.0f));
                    sprintf(m + strlen(m), "  Reflected:%8d (%3d%%)\n", SLRay::reflectedRays, (int)((float)SLRay::reflectedRays / (float)rayTotal * 100.0f));
                    sprintf(m + strlen(m), "  Refracted:%8d (%3d%%)\n", SLRay::refractedRays, (int)((float)SLRay::refractedRays / (float)rayTotal * 100.0f));
                    sprintf(m + strlen(m), "  TIR      :%8d\n", SLRay::tirRays);
                    sprintf(m + strlen(m), "  Shadow   :%8d (%3d%%)\n", SLRay::shadowRays, (int)((float)SLRay::shadowRays / (float)rayTotal * 100.0f));
                    sprintf(m + strlen(m), "  AA       :%8d (%3d%%)\n", SLRay::subsampledRays, (int)((float)SLRay::subsampledRays / (float)rayTotal * 100.0f));
                    sprintf(m + strlen(m), "---------------------------\n");
                    sprintf(m + strlen(m), "Max. depth :%u\n", SLRay::maxDepthReached);
                    sprintf(m + strlen(m), "Avg. depth :%0.3f\n", SLRay::avgDepth / rayPrimaries);
                }
#if defined(SL_BUILD_WITH_OPTIX) && defined(SL_HAS_OPTIX)
                else if (rType == RT_optix_rt)
                {
                    SLOptixRaytracer* ort = sv->optixRaytracer();
                    sprintf(m + strlen(m), "Renderer   :OptiX Ray Tracer\n");
                    sprintf(m + strlen(m), "Frame size :%d x %d\n", sv->scrW(), sv->scrH());
                    sprintf(m + strlen(m), "FPS        :%5.1f\n", s->fps());
                    sprintf(m + strlen(m), "Frame Time :%0.3f sec.\n", 1.0f / s->fps());
                }
                else if (rType == RT_optix_pt)
                {
                    SLOptixPathtracer* opt = sv->optixPathtracer();
                    sprintf(m + strlen(m), "Renderer   :OptiX Ray Tracer\n");
                    sprintf(m + strlen(m), "Frame size :%d x %d\n", sv->scrW(), sv->scrH());
                    sprintf(m + strlen(m), "Frame Time :%0.2f sec.\n", opt->renderSec());
                    sprintf(m + strlen(m), "Denoiser Time :%0.0f ms.\n", opt->denoiserMS());
                }
#endif
                else if (rType == RT_pt)
                {
                    SLPathtracer* pt           = sv->pathtracer();
                    SLint         ptWidth      = (SLint)(sv->viewportW() * pt->resolutionFactor());
                    SLint         ptHeight     = (SLint)(sv->viewportH() * pt->resolutionFactor());
                    SLuint        rayPrimaries = (SLuint)(ptWidth * ptHeight);
                    SLuint        rayTotal     = SLRay::totalNumRays();

                    sprintf(m + strlen(m), "Renderer   :Path Tracer\n");
                    sprintf(m + strlen(m), "Progress   :%3d%%\n", pt->progressPC());
                    sprintf(m + strlen(m), "Frame size :%d x %d\n", ptWidth, ptHeight);
                    sprintf(m + strlen(m), "FPS        :%0.2f\n", 1.0f / pt->renderSec());
                    sprintf(m + strlen(m), "Frame Time :%0.2f sec.\n", pt->renderSec());
                    sprintf(m + strlen(m), "Rays per ms:%0.0f\n", pt->raysPerMS());
                    sprintf(m + strlen(m), "Samples/pix:%d\n", pt->aaSamples());
                    sprintf(m + strlen(m), "Threads    :%d\n", pt->numThreads());
                    sprintf(m + strlen(m), "---------------------------\n");
                    sprintf(m + strlen(m), "Total rays :%8d (%3d%%)\n", rayTotal, 100);
                    sprintf(m + strlen(m), "  Reflected:%8d (%3d%%)\n", SLRay::reflectedRays, (int)((float)SLRay::reflectedRays / (float)rayTotal * 100.0f));
                    sprintf(m + strlen(m), "  Refracted:%8d (%3d%%)\n", SLRay::refractedRays, (int)((float)SLRay::refractedRays / (float)rayTotal * 100.0f));
                    sprintf(m + strlen(m), "  TIR      :%8d\n", SLRay::tirRays);
                    sprintf(m + strlen(m), "  Shadow   :%8d (%3d%%)\n", SLRay::shadowRays, (int)((float)SLRay::shadowRays / (float)rayTotal * 100.0f));
                    sprintf(m + strlen(m), "---------------------------\n");
                }
                else if (rType == RT_ct)
                {
                    // Get averages from average variables (see Averaged)
                    SLfloat captureTime    = CVCapture::instance()->captureTimesMS().average();
                    SLfloat updateTime     = s->updateTimesMS().average();
                    SLfloat updateAnimTime = s->updateAnimTimesMS().average();
                    SLfloat updateAABBTime = s->updateAnimTimesMS().average();
                    SLfloat cullTime       = sv->cullTimesMS().average();
                    SLfloat draw3DTime     = sv->draw3DTimesMS().average();
                    SLfloat draw2DTime     = sv->draw2DTimesMS().average();

                    // Calculate percentage from frame time
                    SLfloat captureTimePC    = Utils::clamp(captureTime / ft * 100.0f, 0.0f, 100.0f);
                    SLfloat updateTimePC     = Utils::clamp(updateTime / ft * 100.0f, 0.0f, 100.0f);
                    SLfloat updateAnimTimePC = Utils::clamp(updateAnimTime / ft * 100.0f, 0.0f, 100.0f);
                    SLfloat updateAABBTimePC = Utils::clamp(updateAABBTime / ft * 100.0f, 0.0f, 100.0f);
                    SLfloat draw3DTimePC     = Utils::clamp(draw3DTime / ft * 100.0f, 0.0f, 100.0f);
                    SLfloat draw2DTimePC     = Utils::clamp(draw2DTime / ft * 100.0f, 0.0f, 100.0f);
                    SLfloat cullTimePC       = Utils::clamp(cullTime / ft * 100.0f, 0.0f, 100.0f);

                    sprintf(m + strlen(m), "Renderer   : Conetracing (OpenGL)\n");
                    sprintf(m + strlen(m), "Frame size : %d x %d\n", sv->viewportW(), sv->viewportH());
                    sprintf(m + strlen(m), "Drawcalls  : %d\n", SLGLVertexArray::totalDrawCalls);
                    sprintf(m + strlen(m), "FPS        :%5.1f\n", s->fps());
                    sprintf(m + strlen(m), "Frame time :%5.1f ms (100%%)\n", ft);
                    sprintf(m + strlen(m), " Capture   :%5.1f ms (%3d%%)\n", captureTime, (SLint)captureTimePC);
                    sprintf(m + strlen(m), " Update    :%5.1f ms (%3d%%)\n", updateTime, (SLint)updateTimePC);
                    sprintf(m + strlen(m), "  Anim.    :%5.1f ms (%3d%%)\n", updateAnimTime, (SLint)updateAnimTimePC);
                    sprintf(m + strlen(m), "  AABB     :%5.1f ms (%3d%%)\n", updateAABBTime, (SLint)updateAABBTimePC);
                    sprintf(m + strlen(m), " Culling   :%5.1f ms (%3d%%)\n", cullTime, (SLint)cullTimePC);
                    sprintf(m + strlen(m), " Drawing 3D:%5.1f ms (%3d%%)\n", draw3DTime, (SLint)draw3DTimePC);
                    sprintf(m + strlen(m), " Drawing 2D:%5.1f ms (%3d%%)\n", draw2DTime, (SLint)draw2DTimePC);
                }

                ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
                ImGui::Begin("Timing", &showStatsTiming, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
                ImGui::TextUnformatted(m);
                ImGui::End();
                ImGui::PopFont();
            }

            if (showStatsScene)
            {
                SLchar m[2550]; // message character array
                m[0] = 0;       // set zero length

                SLNodeStats& stats3D           = sv->stats3D();
                SLfloat      vox               = (SLfloat)stats3D.numVoxels;
                SLfloat      voxEmpty          = (SLfloat)stats3D.numVoxEmpty;
                SLfloat      voxelsEmpty       = vox > 0.0f ? voxEmpty / vox * 100.0f : 0.0f;
                SLfloat      numRTTria         = (SLfloat)stats3D.numTriangles;
                SLfloat      avgTriPerVox      = vox > 0.0f ? numRTTria / (vox - voxEmpty) : 0.0f;
                SLint        numOverdrawnNodes = (int)sv->nodesOverdrawn().size();
                SLint        numVisibleNodes   = stats3D.numNodesOpaque + stats3D.numNodesBlended + numOverdrawnNodes;
                SLint        numGroupPC        = (SLint)((SLfloat)stats3D.numNodesGroup / (SLfloat)stats3D.numNodes * 100.0f);
                SLint        numLeafPC         = (SLint)((SLfloat)stats3D.numNodesLeaf / (SLfloat)stats3D.numNodes * 100.0f);
                SLint        numLightsPC       = (SLint)((SLfloat)stats3D.numLights / (SLfloat)stats3D.numNodes * 100.0f);
                SLint        numOpaquePC       = (SLint)((SLfloat)stats3D.numNodesOpaque / (SLfloat)stats3D.numNodes * 100.0f);
                SLint        numBlendedPC      = (SLint)((SLfloat)stats3D.numNodesBlended / (SLfloat)stats3D.numNodes * 100.0f);
                SLint        numOverdrawnPC    = (SLint)((SLfloat)numOverdrawnNodes / (SLfloat)stats3D.numNodes * 100.0f);
                SLint        numVisiblePC      = (SLint)((SLfloat)numVisibleNodes / (SLfloat)stats3D.numNodes * 100.0f);

                // Calculate total size of texture bytes on CPU
                SLfloat cpuMBTexture = 0;
                for (auto* t : s->textures())
                    for (auto* i : t->images())
                        cpuMBTexture += i->bytesPerImage();
                cpuMBTexture = cpuMBTexture / 1E6f;

                SLfloat cpuMBMeshes    = (SLfloat)stats3D.numBytes / 1E6f;
                SLfloat cpuMBVoxels    = (SLfloat)stats3D.numBytesAccel / 1E6f;
                SLfloat cpuMBTotal     = cpuMBTexture + cpuMBMeshes + cpuMBVoxels;
                SLint   cpuMBTexturePC = (SLint)(cpuMBTexture / cpuMBTotal * 100.0f);
                SLint   cpuMBMeshesPC  = (SLint)(cpuMBMeshes / cpuMBTotal * 100.0f);
                SLint   cpuMBVoxelsPC  = (SLint)(cpuMBVoxels / cpuMBTotal * 100.0f);
                SLfloat gpuMBTexture   = (SLfloat)SLGLTexture::totalNumBytesOnGPU / 1E6f;
                SLfloat gpuMBVbo       = (SLfloat)SLGLVertexBuffer::totalBufferSize / 1E6f;
                SLfloat gpuMBTotal     = gpuMBTexture + gpuMBVbo;
                SLint   gpuMBTexturePC = (SLint)(gpuMBTexture / gpuMBTotal * 100.0f);
                SLint   gpuMBVboPC     = (SLint)(gpuMBVbo / gpuMBTotal * 100.0f);

                sprintf(m + strlen(m), "Name: %s\n", s->name().c_str());
                sprintf(m + strlen(m), "No. of Nodes  :%5d (100%%)\n", stats3D.numNodes);
                sprintf(m + strlen(m), "- Group Nodes :%5d (%3d%%)\n", stats3D.numNodesGroup, numGroupPC);
                sprintf(m + strlen(m), "- Leaf  Nodes :%5d (%3d%%)\n", stats3D.numNodesLeaf, numLeafPC);
                sprintf(m + strlen(m), "- Light Nodes :%5d (%3d%%)\n", stats3D.numLights, numLightsPC);
                sprintf(m + strlen(m), "- Opaque Nodes:%5d (%3d%%)\n", stats3D.numNodesOpaque, numOpaquePC);
                sprintf(m + strlen(m), "- Blend Nodes :%5d (%3d%%)\n", stats3D.numNodesBlended, numBlendedPC);
                sprintf(m + strlen(m), "- Overdrawn N.:%5d (%3d%%)\n", numOverdrawnNodes, numOverdrawnPC);
                sprintf(m + strlen(m), "- Vis. Nodes  :%5d (%3d%%)\n", numVisibleNodes, numVisiblePC);
                sprintf(m + strlen(m), "- WM Updates  :%5d\n", SLNode::numWMUpdates);
                sprintf(m + strlen(m), "No. of Meshes :%5u\n", stats3D.numMeshes);
                sprintf(m + strlen(m), "No. of Tri.   :%5u\n", stats3D.numTriangles);
                sprintf(m + strlen(m), "CPU MB Total  :%6.2f (100%%)\n", cpuMBTotal);
                sprintf(m + strlen(m), "-   MB Tex.   :%6.2f (%3d%%)\n", cpuMBTexture, cpuMBTexturePC);
                sprintf(m + strlen(m), "-   MB Meshes :%6.2f (%3d%%)\n", cpuMBMeshes, cpuMBMeshesPC);
                sprintf(m + strlen(m), "-   MB Voxels :%6.2f (%3d%%)\n", cpuMBVoxels, cpuMBVoxelsPC);
                sprintf(m + strlen(m), "GPU MB Total  :%6.2f (100%%)\n", gpuMBTotal);
                sprintf(m + strlen(m), "-   MB Tex.   :%6.2f (%3d%%)\n", gpuMBTexture, gpuMBTexturePC);
                sprintf(m + strlen(m), "-   MB VBO    :%6.2f (%3d%%)\n", gpuMBVbo, gpuMBVboPC);
                sprintf(m + strlen(m), "No. of Voxels :%d\n", stats3D.numVoxels);
                sprintf(m + strlen(m), "-empty Voxels :%4.1f%%\n", voxelsEmpty);
                sprintf(m + strlen(m), "Avg.Tri/Voxel :%4.1f\n", avgTriPerVox);
                sprintf(m + strlen(m), "Max.Tri/Voxel :%d\n", stats3D.numVoxMaxTria);

                // Switch to fixed font
                ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
                ImGui::Begin("Scene Statistics", &showStatsScene, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
                ImGui::TextUnformatted(m);

                ImGui::Separator();

                ImGui::Text("Global Resources:");

                if (s->meshes().size() && ImGui::TreeNode("Meshes"))
                {
                    for (SLuint i = 0; i < s->meshes().size(); ++i)
                        ImGui::Text("[%d] %s (%u v.)",
                                    i,
                                    s->meshes()[i]->name().c_str(),
                                    (SLuint)s->meshes()[i]->P.size());

                    ImGui::TreePop();
                }

                if (s->lights().size() && ImGui::TreeNode("Lights"))
                {
                    for (SLuint i = 0; i < s->lights().size(); ++i)
                    {
                        SLNode* light = dynamic_cast<SLNode*>(s->lights()[i]);
                        ImGui::Text("[%u] %s", i, light->name().c_str());
                    }

                    ImGui::TreePop();
                }

                if (sv->visibleMaterials3D().size() && ImGui::TreeNode("Materials"))
                {
                    for (auto* mat : sv->visibleMaterials3D())
                    {
                        SLVNode& matNodes = mat->nodesVisible3D();
                        sprintf(m,
                                "%s [%u n.]",
                                mat->name().c_str(),
                                (SLuint)matNodes.size());

                        if (matNodes.size())
                        {
                            if (ImGui::TreeNode(m))
                            {
                                for (auto* node : matNodes)
                                    ImGui::Text(node->name().c_str());

                                ImGui::TreePop();
                            }
                        }
                        else
                            ImGui::Text(m);
                    }

                    ImGui::TreePop();
                }

                if (s->textures().size() && ImGui::TreeNode("Textures"))
                {
                    for (SLuint i = 0; i < s->textures().size(); ++i)
                    {
                        if (s->textures()[i]->images().empty())
                            ImGui::Text("[%u] %s on GPU (%s)", i, s->textures()[i]->name().c_str(), s->textures()[i]->isTexture() ? "ok" : "not ok");
                        else
                            ImGui::Text("[%u] %s (%s)", i, s->textures()[i]->name().c_str(), s->textures()[i]->isTexture() ? "ok" : "not ok");
                    }

                    ImGui::TreePop();
                }

                if (s->programs().size() && ImGui::TreeNode("Programs (asset manager)"))
                {
                    for (SLuint i = 0; i < s->programs().size(); ++i)
                    {
                        SLGLProgram* p = s->programs()[i];
                        ImGui::Text("[%u] %s", i, p->name().c_str());
                    }
                    ImGui::TreePop();
                }

                if (ImGui::TreeNode("Programs (application)"))
                {
                    for (SLuint i = 0; i < SLGLProgramManager::size(); ++i)
                        ImGui::Text("[%u] %s", i, SLGLProgramManager::get((SLStdShaderProg)i)->name().c_str());

                    ImGui::TreePop();
                }

                ImGui::End();
                ImGui::PopFont();
            }

            if (showStatsVideo)
            {
                SLchar m[2550]; // message character array
                m[0] = 0;       // set zero length

                CVCamera*      ac       = CVCapture::instance()->activeCamera;
                CVCalibration* c        = &CVCapture::instance()->activeCamera->calibration;
                CVSize         capSize  = CVCapture::instance()->captureSize;
                CVVideoType    vt       = CVCapture::instance()->videoType();
                SLstring       mirrored = "None";
                if (c->isMirroredH() && c->isMirroredV())
                    mirrored = "horizontally & vertically";
                else if (c->isMirroredH())
                    mirrored = "horizontally";
                else if (c->isMirroredV())
                    mirrored = "vertically";

                sprintf(m + strlen(m), "Video Type   : %s\n", vt == VT_NONE ? "None" : vt == VT_MAIN ? "Main Camera"
                                                                                     : vt == VT_FILE ? "File"
                                                                                                     : "Secondary Camera");
                sprintf(m + strlen(m), "Display size : %d x %d\n", CVCapture::instance()->lastFrame.cols, CVCapture::instance()->lastFrame.rows);
                sprintf(m + strlen(m), "Capture size : %d x %d\n", capSize.width, capSize.height);
                sprintf(m + strlen(m), "Size Index   : %d\n", ac->camSizeIndex());
                sprintf(m + strlen(m), "Mirrored     : %s\n", mirrored.c_str());
                sprintf(m + strlen(m), "Chessboard   : %dx%d (%3.1fmm)\n", c->boardSize().width, c->boardSize().height, c->boardSquareMM());
                sprintf(m + strlen(m), "Undistorted  : %s\n", ac->showUndistorted() ? "Yes" : "No");
                sprintf(m + strlen(m), "Calibimg size: %d x %d\n", ac->calibration.imageSizeOriginal().width, ac->calibration.imageSizeOriginal().height);
                sprintf(m + strlen(m), "FOV H/V(deg.): %4.1f/%4.1f\n", c->cameraFovHDeg(), c->cameraFovVDeg());
                sprintf(m + strlen(m), "fx,fy        : %4.1f,%4.1f\n", c->fx(), c->fy());
                sprintf(m + strlen(m), "cx,cy        : %4.1f,%4.1f\n", c->cx(), c->cy());

                int         distortionSize = c->distortion().rows;
                const float f              = 100.f;
                sprintf(m + strlen(m), "dist.(*10e-2):\n");
                sprintf(m + strlen(m), "k1,k2        : %4.2f,%4.2f\n", c->k1() * f, c->k2() * f);
                sprintf(m + strlen(m), "p1,p2        : %4.2f,%4.2f\n", c->p1() * f, c->p2() * f);
                if (distortionSize >= 8)
                    sprintf(m + strlen(m), "k3,k4,k5,k6  : %4.2f,%4.2f,%4.2f,%4.2f\n", c->k3() * f, c->k4() * f, c->k5() * f, c->k6() * f);
                else
                    sprintf(m + strlen(m), "k3           : %4.2f\n", c->k3() * f);

                if (distortionSize >= 12)
                    sprintf(m + strlen(m), "s1,s2,s3,s4  : %4.2f,%4.2f,%4.2f,%4.2f\n", c->s1() * f, c->s2() * f, c->s3() * f, c->s4() * f);
                if (distortionSize >= 14)
                    sprintf(m + strlen(m), "tauX,tauY    : %4.2f,%4.2f\n", c->tauX() * f, c->tauY() * f);

                sprintf(m + strlen(m), "Calib. time  : %s\n", c->calibrationTime().c_str());
                sprintf(m + strlen(m), "Calib. state : %s\n", c->stateStr().c_str());
                sprintf(m + strlen(m), "Num. caps    : %d\n", c->numCapturedImgs());

                if (vt != VT_NONE && tracker != nullptr && trackedNode != nullptr)
                {
                    sprintf(m + strlen(m), "-------------:\n");
                    if (typeid(*trackedNode) == typeid(SLCamera))
                    {
                        SLVec3f cameraPos = trackedNode->updateAndGetWM().translation();
                        sprintf(m + strlen(m), "Dist. to zero: %4.2f\n", cameraPos.length());
                    }
                    else
                    {
                        SLVec3f cameraPos = ((SLNode*)sv->camera())->updateAndGetWM().translation();
                        SLVec3f objectPos = trackedNode->updateAndGetWM().translation();
                        SLVec3f camToObj  = objectPos - cameraPos;
                        sprintf(m + strlen(m), "Dist. to obj.: %4.2f\n", camToObj.length());
                    }
                }

                // Switch to fixed font
                ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
                ImGui::Begin("Video", &showStatsVideo, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
                ImGui::TextUnformatted(m);
                ImGui::End();
                ImGui::PopFont();
            }
#ifdef SL_BUILD_WAI
            if (showStatsWAI && AppDemo::sceneID == SID_VideoTrackWAI)
            {
                ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
                ImGui::Begin("WAI Statistics", &showStatsWAI, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);

                if (!AverageTiming::instance().empty())
                {
                    SLchar m[2550]; // message character array
                    m[0] = 0;       // set zero length

                    AverageTiming::getTimingMessage(m);

                    // define ui elements
                    ImGui::TextUnformatted(m);
                }

                ImGui::End();
                ImGui::PopFont();
            }
#endif
            if (showImGuiMetrics)
            {
                ImGui::ShowMetricsWindow();
            }

            if (showInfosScene)
            {
                // Calculate window position for dynamic status bar at the bottom of the main window
                ImGuiWindowFlags window_flags = 0;
                window_flags |= ImGuiWindowFlags_NoTitleBar;
                window_flags |= ImGuiWindowFlags_NoResize;
                window_flags |= ImGuiWindowFlags_NoScrollbar;
                SLfloat  w    = (SLfloat)sv->viewportW();
                ImVec2   size = ImGui::CalcTextSize(s->info().c_str(),
                                                    nullptr,
                                                    true,
                                                    w);
                SLfloat  h    = size.y + SLGLImGui::fontPropDots * 2.0f;
                SLstring info = "Scene Info: " + s->info();

                ImGui::SetNextWindowPos(ImVec2(0, sv->scrH() - h));
                ImGui::SetNextWindowSize(ImVec2(w, h));
                ImGui::Begin("Scene Information", &showInfosScene, window_flags);
                ImGui::SetCursorPosX((w - size.x) * 0.5f);
                ImGui::TextWrapped("%s", info.c_str());
                ImGui::End();
            }

            if (showTransform)
            {
                ImGuiWindowFlags window_flags = 0;
                window_flags |= ImGuiWindowFlags_AlwaysAutoResize;
                ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
                ImGui::Begin("Transform Selected Node", &showTransform, window_flags);

                if (s->singleNodeSelected())
                {
                    SLNode*                 selNode = s->singleNodeSelected();
                    static SLTransformSpace tSpace  = TS_object;
                    SLfloat                 t1 = 0.1f, t2 = 1.0f, t3 = 10.0f; // Delta translations
                    SLfloat                 r1 = 1.0f, r2 = 5.0f, r3 = 15.0f; // Delta rotations
                    SLfloat                 s1 = 1.01f, s2 = 1.1f, s3 = 1.5f; // Scale factors

                    // clang-format off
                    ImGui::Text("Space:");
                    ImGui::SameLine();
                    if (ImGui::RadioButton("World", (int *) &tSpace, 0)) tSpace = TS_world;
                    ImGui::SameLine();
                    if (ImGui::RadioButton("Parent", (int *) &tSpace, 1)) tSpace = TS_parent;
                    ImGui::SameLine();
                    if (ImGui::RadioButton("Object", (int *) &tSpace, 2)) tSpace = TS_object;
                    ImGui::Separator();

                    ImGui::Text("Transl. X :");
                    ImGui::SameLine();
                    if (ImGui::Button("<<<##Tx")) selNode->translate(-t3, 0, 0, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button("<<##Tx")) selNode->translate(-t2, 0, 0, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button("<##Tx")) selNode->translate(-t1, 0, 0, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button(">##Tx")) selNode->translate(t1, 0, 0, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button(">>##Tx")) selNode->translate(t2, 0, 0, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button(">>>##Tx")) selNode->translate(t3, 0, 0, tSpace);

                    ImGui::Text("Transl. Y :");
                    ImGui::SameLine();
                    if (ImGui::Button("<<<##Ty")) selNode->translate(0, -t3, 0, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button("<<##Ty")) selNode->translate(0, -t2, 0, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button("<##Ty")) selNode->translate(0, -t1, 0, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button(">##Ty")) selNode->translate(0, t1, 0, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button(">>##Ty")) selNode->translate(0, t2, 0, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button(">>>##Ty")) selNode->translate(0, t3, 0, tSpace);

                    ImGui::Text("Transl. Z :");
                    ImGui::SameLine();
                    if (ImGui::Button("<<<##Tz")) selNode->translate(0, 0, -t3, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button("<<##Tz")) selNode->translate(0, 0, -t2, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button("<##Tz")) selNode->translate(0, 0, -t1, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button(">##Tz")) selNode->translate(0, 0, t1, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button(">>##Tz")) selNode->translate(0, 0, t2, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button(">>>##Tz")) selNode->translate(0, 0, t3, tSpace);

                    ImGui::Text("Rotation X:");
                    ImGui::SameLine();
                    if (ImGui::Button("<<<##Rx")) selNode->rotate(r3, 1, 0, 0, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button("<<##Rx")) selNode->rotate(r2, 1, 0, 0, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button("<##Rx")) selNode->rotate(r1, 1, 0, 0, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button(">##Rx")) selNode->rotate(-r1, 1, 0, 0, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button(">>##Rx")) selNode->rotate(-r2, 1, 0, 0, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button(">>>##Rx")) selNode->rotate(-r3, 1, 0, 0, tSpace);

                    ImGui::Text("Rotation Y:");
                    ImGui::SameLine();
                    if (ImGui::Button("<<<##Ry")) selNode->rotate(r3, 0, 1, 0, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button("<<##Ry")) selNode->rotate(r2, 0, 1, 0, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button("<##Ry")) selNode->rotate(r1, 0, 1, 0, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button(">##Ry")) selNode->rotate(-r1, 0, 1, 0, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button(">>##Ry")) selNode->rotate(-r2, 0, 1, 0, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button(">>>##Ry")) selNode->rotate(-r3, 0, 1, 0, tSpace);

                    ImGui::Text("Rotation Z:");
                    ImGui::SameLine();
                    if (ImGui::Button("<<<##Rz")) selNode->rotate(r3, 0, 0, 1, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button("<<##Rz")) selNode->rotate(r2, 0, 0, 1, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button("<##Rz")) selNode->rotate(r1, 0, 0, 1, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button(">##Rz")) selNode->rotate(-r1, 0, 0, 1, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button(">>##Rz")) selNode->rotate(-r2, 0, 0, 1, tSpace);
                    ImGui::SameLine();
                    if (ImGui::Button(">>>##Rz")) selNode->rotate(-r3, 0, 0, 1, tSpace);

                    ImGui::Text("Scale     :");
                    ImGui::SameLine();
                    if (ImGui::Button("<<<##S")) selNode->scale(s3);
                    ImGui::SameLine();
                    if (ImGui::Button("<<##S")) selNode->scale(s2);
                    ImGui::SameLine();
                    if (ImGui::Button("<##S")) selNode->scale(s1);
                    ImGui::SameLine();
                    if (ImGui::Button(">##S")) selNode->scale(-s1);
                    ImGui::SameLine();
                    if (ImGui::Button(">>##S")) selNode->scale(-s2);
                    ImGui::SameLine();
                    if (ImGui::Button(">>>##S")) selNode->scale(-s3);
                    ImGui::Separator();
                    // clang-format on

                    if (ImGui::Button("Reset"))
                        selNode->om(selNode->initialOM());
                }
                else
                {
                    ImGui::Text("No node selected.");
                    ImGui::Text("Please select a node by double clicking it.");

                    if (transformNode)
                        removeTransformNode(s);
                }
                ImGui::End();
                ImGui::PopFont();
            }

            if (showInfosDevice)
            {
                SLGLState* stateGL = SLGLState::instance();
                SLchar     m[2550]; // message character array
                m[0] = 0;           // set zero length

                sprintf(m + strlen(m), "SLProject Version: %s\n", AppDemo::version.c_str());
#ifdef _DEBUG
                sprintf(m + strlen(m), "Build Config.    : Debug\n");
#else
                sprintf(m + strlen(m), "Build Config.    : Release\n");
#endif
                sprintf(m + strlen(m), "-----------------:\n");
                sprintf(m + strlen(m), "Computer User    : %s\n", Utils::ComputerInfos::user.c_str());
                sprintf(m + strlen(m), "Computer Name    : %s\n", Utils::ComputerInfos::name.c_str());
                sprintf(m + strlen(m), "Computer Brand   : %s\n", Utils::ComputerInfos::brand.c_str());
                sprintf(m + strlen(m), "Computer Model   : %s\n", Utils::ComputerInfos::model.c_str());
                sprintf(m + strlen(m), "Computer Arch.   : %s\n", Utils::ComputerInfos::arch.c_str());
                sprintf(m + strlen(m), "Computer OS      : %s\n", Utils::ComputerInfos::os.c_str());
                sprintf(m + strlen(m), "Computer OS Ver. : %s\n", Utils::ComputerInfos::osVer.c_str());
                sprintf(m + strlen(m), "-----------------:\n");
                sprintf(m + strlen(m), "OpenGL Version   : %s\n", stateGL->glVersionNO().c_str());
                sprintf(m + strlen(m), "OpenGL Vendor    : %s\n", stateGL->glVendor().c_str());
                sprintf(m + strlen(m), "OpenGL Renderer  : %s\n", stateGL->glRenderer().c_str());
                sprintf(m + strlen(m), "OpenGL GLSL Ver. : %s\n", stateGL->glSLVersionNO().c_str());
                sprintf(m + strlen(m), "-----------------:\n");
                sprintf(m + strlen(m), "OpenCV Version   : %d.%d.%d\n", CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_VERSION_REVISION);
                sprintf(m + strlen(m), "OpenCV has OpenCL: %s\n", cv::ocl::haveOpenCL() ? "yes" : "no");
                sprintf(m + strlen(m), "OpenCV has AVX   : %s\n", cv::checkHardwareSupport(CV_AVX) ? "yes" : "no");
                sprintf(m + strlen(m), "OpenCV has NEON  : %s\n", cv::checkHardwareSupport(CV_NEON) ? "yes" : "no");
                sprintf(m + strlen(m), "-----------------:\n");

#ifdef SL_BUILD_WAI
                sprintf(m + strlen(m), "Eigen Version    : %d.%d.%d\n", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
#    ifdef EIGEN_VECTORIZE
                sprintf(m + strlen(m), "Eigen vectorize  : yes\n");
#    else
                sprintf(m + strlen(m), "Eigen vectorize  : no\n");
#    endif
#endif
                sprintf(m + strlen(m), "-----------------:\n");
                sprintf(m + strlen(m), "ImGui Version    : %s\n", ImGui::GetVersion());

                // Switch to fixed font
                ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
                ImGui::Begin("Device Informations", &showInfosDevice, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
                ImGui::TextUnformatted(m);
                ImGui::End();
                ImGui::PopFont();
            }

            if (showInfosSensors)
            {
                SLchar m[1024];             // message character array
                m[0]                   = 0; // set zero length
                SLVec3d offsetToOrigin = AppDemo::devLoc.originENU() - AppDemo::devLoc.locENU();
                sprintf(m + strlen(m), "Uses IMU Senor   : %s\n", AppDemo::devRot.isUsed() ? "yes" : "no");
                sprintf(m + strlen(m), "Pitch (deg)      : %3.1f\n", AppDemo::devRot.pitchDEG());
                sprintf(m + strlen(m), "Yaw   (deg)      : %3.1f\n", AppDemo::devRot.yawDEG());
                sprintf(m + strlen(m), "Roll  (deg)      : %3.1f\n", AppDemo::devRot.rollDEG());
                sprintf(m + strlen(m), "No. averaged     : %d\n", AppDemo::devRot.numAveraged());
                // sprintf(m + strlen(m), "Pitch Offset(deg): %3.1f\n", AppDemo::devRot.pitchOffsetDEG());
                // sprintf(m + strlen(m), "Yaw   Offset(deg): %3.1f\n", AppDemo::devRot.yawOffsetDEG());
                sprintf(m + strlen(m), "Rot. Offset mode : %s\n", AppDemo::devRot.offsetModeStr().c_str());
                sprintf(m + strlen(m), "------------------\n");
                sprintf(m + strlen(m), "Uses GPS Sensor  : %s\n", AppDemo::devLoc.isUsed() ? "yes" : "no");
                sprintf(m + strlen(m), "Latitude (deg)   : %10.5f\n", AppDemo::devLoc.locLatLonAlt().lat);
                sprintf(m + strlen(m), "Longitude (deg)  : %10.5f\n", AppDemo::devLoc.locLatLonAlt().lon);
                sprintf(m + strlen(m), "Alt. used (m)    : %10.2f\n", AppDemo::devLoc.locLatLonAlt().alt);
                sprintf(m + strlen(m), "Alt. GPS (m)     : %10.2f\n", AppDemo::devLoc.altGpsM());
                sprintf(m + strlen(m), "Alt. DEM (m)     : %10.2f\n", AppDemo::devLoc.altDemM());
                sprintf(m + strlen(m), "Alt. origin (m)  : %10.2f\n", AppDemo::devLoc.altDemM());
                sprintf(m + strlen(m), "Accuracy Rad.(m) : %6.1f\n", AppDemo::devLoc.locAccuracyM());
                sprintf(m + strlen(m), "Dist. Origin (m) : %6.1f\n", offsetToOrigin.length());
                sprintf(m + strlen(m), "Origin improve(s): %6.1f sec.\n", AppDemo::devLoc.improveTime());
                sprintf(m + strlen(m), "Loc. Offset mode : %s\n", AppDemo::devLoc.offsetModeStr().c_str());
                sprintf(m + strlen(m), "Loc. Offset (m)  : %s\n", AppDemo::devLoc.offsetENU().toString(",", 1).c_str());

                // Switch to fixed font
                ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
                ImGui::Begin("Sensor Information", &showInfosSensors, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
                ImGui::TextUnformatted(m);
                ImGui::End();
                ImGui::PopFont();
            }

            if (showSceneGraph)
            {
                buildSceneGraph(s);
            }

            if (showProperties)
            {
                buildProperties(s, sv);
            }

            if (showUIPrefs)
            {
                ImGuiWindowFlags window_flags = 0;
                window_flags |= ImGuiWindowFlags_AlwaysAutoResize;
                ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
                ImGui::Begin("User Interface Preferences", &showUIPrefs, window_flags);
                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.66f);

                ImGui::SliderFloat("Prop. Font Size", &SLGLImGui::fontPropDots, 16.f, 70.f, "%0.0f");
                ImGui::SliderFloat("Fixed Font Size", &SLGLImGui::fontFixedDots, 13.f, 50.f, "%0.0f");
                ImGuiStyle& style = ImGui::GetStyle();
                if (ImGui::SliderFloat("Item Spacing X", &style.ItemSpacing.x, 0.0f, 20.0f, "%0.0f"))
                    style.WindowPadding.x = style.FramePadding.x = style.ItemInnerSpacing.x = style.ItemSpacing.x;
                if (ImGui::SliderFloat("Item Spacing Y", &style.ItemSpacing.y, 0.0f, 20.0f, "%0.0f"))
                {
                    style.FramePadding.y = style.ItemInnerSpacing.y = style.ItemSpacing.y;
                    style.WindowPadding.y                           = style.ItemSpacing.y * 3;
                }

                ImGui::Separator();

                ImGui::Checkbox("Dock-Space enabled", &showDockSpace);

                ImGui::Separator();

                SLchar reset[255];
                sprintf(reset, "Reset User Interface (DPI: %d)", sv->dpi());
                if (ImGui::MenuItem(reset))
                {
                    SLstring fullPathFilename = AppDemo::configPath + "DemoGui.yml";
                    Utils::deleteFile(fullPathFilename);
                    loadConfig(sv->dpi());
                }

                ImGui::PopItemWidth();
                ImGui::End();
                ImGui::PopFont();
            }

            if (showDateAndTime)
            {
                if (AppDemo::devLoc.originLatLonAlt() != SLVec3d::ZERO ||
                    AppDemo::devLoc.defaultLatLonAlt() != SLVec3d::ZERO)
                {
                    ImGuiWindowFlags window_flags = 0;
                    window_flags |= ImGuiWindowFlags_AlwaysAutoResize;
                    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
                    ImGui::Begin("Date and Time Settings", &showDateAndTime, window_flags);
                    ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.66f);

                    tm lt{};
                    if (adjustedTime)
                        memcpy(&lt, std::localtime(&adjustedTime), sizeof(tm));
                    else
                    {
                        std::time_t now = std::time(nullptr);
                        memcpy(&lt, std::localtime(&now), sizeof(tm));
                    }

                    SLint month = lt.tm_mon + 1;
                    if (ImGui::SliderInt("Month", &month, 1, 12))
                    {
                        lt.tm_mon    = month - 1;
                        adjustedTime = mktime(&lt);
                        AppDemo::devLoc.calculateSolarAngles(AppDemo::devLoc.originLatLonAlt(),
                                                             adjustedTime);
                    }

                    if (ImGui::SliderInt("Day", &lt.tm_mday, 1, 31))
                    {
                        adjustedTime = mktime(&lt);
                        AppDemo::devLoc.calculateSolarAngles(AppDemo::devLoc.originLatLonAlt(),
                                                             adjustedTime);
                    }

                    SLfloat SRh  = AppDemo::devLoc.originSolarSunrise();
                    SLfloat SSh  = AppDemo::devLoc.originSolarSunset();
                    SLfloat nowF = (SLfloat)lt.tm_hour + (float)lt.tm_min / 60.0f;
                    if (ImGui::SliderFloat("Hour", &nowF, SRh, SSh, "%.2f"))
                    {
                        lt.tm_hour   = (int)nowF;
                        lt.tm_min    = (int)((nowF - (int)nowF) * 60.0f);
                        adjustedTime = mktime(&lt);
                        AppDemo::devLoc.calculateSolarAngles(AppDemo::devLoc.originLatLonAlt(),
                                                             adjustedTime);
                    }

                    SLchar strTime[100];
                    sprintf(strTime, "Set now (%02d.%02d.%02d %02d:%02d)", lt.tm_mday, lt.tm_mon, lt.tm_year + 1900, lt.tm_hour, lt.tm_min);
                    if (ImGui::MenuItem(strTime))
                    {
                        adjustedTime    = 0;
                        std::time_t now = std::time(nullptr);
                        memcpy(&lt, std::localtime(&now), sizeof(tm));
                        AppDemo::devLoc.calculateSolarAngles(AppDemo::devLoc.originLatLonAlt(), now);
                    }

                    sprintf(strTime, "Set highest noon (21.06.%02d 12:00)", lt.tm_year - 100);
                    if (ImGui::MenuItem(strTime))
                    {
                        lt.tm_mon    = 6;
                        lt.tm_mday   = 21;
                        lt.tm_hour   = 12;
                        lt.tm_min    = 0;
                        lt.tm_sec    = 0;
                        adjustedTime = mktime(&lt);
                        AppDemo::devLoc.calculateSolarAngles(AppDemo::devLoc.originLatLonAlt(),
                                                             adjustedTime);
                    }

                    sprintf(strTime, "Set lowest noon (21.12.%02d 12:00)", lt.tm_year - 100);
                    if (ImGui::MenuItem(strTime))
                    {
                        lt.tm_mon    = 12;
                        lt.tm_mday   = 21;
                        lt.tm_hour   = 12;
                        lt.tm_min    = 0;
                        lt.tm_sec    = 0;
                        adjustedTime = mktime(&lt);
                        AppDemo::devLoc.calculateSolarAngles(AppDemo::devLoc.originLatLonAlt(),
                                                             adjustedTime);
                    }

                    SLNode* sunLightNode = AppDemo::devLoc.sunLightNode();
                    if (sunLightNode &&
                        typeid(*sunLightNode) == typeid(SLLightDirect) &&
                        ((SLLightDirect*)sunLightNode)->doSunPowerAdaptation())
                    {
                        SLLight* light        = (SLLight*)(SLLightDirect*)sunLightNode;
                        float    aP           = light->ambientPower();
                        float    dP           = light->diffusePower();
                        float    sum_aPdP     = aP + dP;
                        float    ambiFraction = aP / sum_aPdP;
                        ImGui::Separator();
                        if (ImGui::SliderFloat("Direct-Indirect", &ambiFraction, 0.0f, 1.0f, "%.2f"))
                        {
                            light->ambientPower(ambiFraction * sum_aPdP);
                            light->diffusePower((1.0f - ambiFraction) * sum_aPdP);
                        }
                    }

                    ImGui::PopItemWidth();
                    ImGui::End();
                    ImGui::PopFont();
                }
                else
                    showDateAndTime = false;
            }

            if (showErlebAR)
            {
                ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
                SLint   namedLocIndex = AppDemo::devLoc.activeNamedLocation();
                SLVec3f lookAtPoint   = SLVec3f::ZERO;

                if (AppDemo::sceneID == SID_ErlebARChristoffel)
                {
                    ImGui::Begin("Christoffel",
                                 &showErlebAR,
                                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);

                    // Get scene nodes once
                    if (!bern)
                    {
                        bern        = s->root3D()->findChild<SLNode>("bern-christoffel.gltf");
                        chrAlt      = bern->findChild<SLNode>("Chr-Alt", true);
                        chrNeu      = bern->findChild<SLNode>("Chr-Neu", true);
                        balda_stahl = bern->findChild<SLNode>("Baldachin-Stahl", true);
                        balda_glas  = bern->findChild<SLNode>("Baldachin-Glas", true);
                    }

                    SLbool chrAltIsOn = !chrAlt->drawBits()->get(SL_DB_HIDDEN);
                    if (ImGui::Checkbox("Christoffelturm 1500-1800", &chrAltIsOn))
                    {
                        chrAlt->drawBits()->set(SL_DB_HIDDEN, false);
                        chrNeu->drawBits()->set(SL_DB_HIDDEN, true);
                    }

                    SLbool chrNeuIsOn = !chrNeu->drawBits()->get(SL_DB_HIDDEN);
                    if (ImGui::Checkbox("Christoffelturm 1800-1865", &chrNeuIsOn))
                    {
                        chrAlt->drawBits()->set(SL_DB_HIDDEN, true);
                        chrNeu->drawBits()->set(SL_DB_HIDDEN, false);
                    }
                    SLbool baldachin = !balda_stahl->drawBits()->get(SL_DB_HIDDEN);
                    if (ImGui::Checkbox("Baldachin", &baldachin))
                    {
                        balda_stahl->drawBits()->set(SL_DB_HIDDEN, !baldachin);
                        balda_glas->drawBits()->set(SL_DB_HIDDEN, !baldachin);
                    }

                    ImGui::Separator();

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
                    bool devLocIsUsed = AppDemo::devLoc.isUsed();
                    if (ImGui::Checkbox("Use GPS Location", &devLocIsUsed))
                        AppDemo::devLoc.isUsed(true);
#endif
                    lookAtPoint.set(-21, 18, 6);
                    for (int i = 1; i < AppDemo::devLoc.nameLocations().size(); ++i)
                    {
                        bool namedLocIsActive = namedLocIndex == i;
                        if (ImGui::Checkbox(AppDemo::devLoc.nameLocations()[i].name.c_str(), &namedLocIsActive))
                            setActiveNamedLocation(i, sv, lookAtPoint);
                    }

                    ImGui::End();
                }
                else
                {
                    bern        = nullptr;
                    chrAlt      = nullptr;
                    chrNeu      = nullptr;
                    balda_stahl = nullptr;
                    balda_glas  = nullptr;
                }

                if (AppDemo::sceneID == SID_ErlebARAugustaRauricaTmpTht ||
                    AppDemo::sceneID == SID_ErlebARAugustaRauricaTht ||
                    AppDemo::sceneID == SID_ErlebARAugustaRauricaTmp)
                {
                    ImGui::Begin("Augst-Theatre-Temple",
                                 &showErlebAR,
                                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
                    bool devLocIsUsed = AppDemo::devLoc.isUsed();
                    if (ImGui::Checkbox("Use GPS Location", &devLocIsUsed))
                        AppDemo::devLoc.isUsed(true);
#endif
                    for (int i = 1; i < AppDemo::devLoc.nameLocations().size(); ++i)
                    {
                        bool namedLocIsActive = namedLocIndex == i;
                        if (ImGui::Checkbox(AppDemo::devLoc.nameLocations()[i].name.c_str(), &namedLocIsActive))
                            setActiveNamedLocation(i, sv);
                    }

                    ImGui::End();
                }

                if (AppDemo::sceneID == SID_ErlebARAventicumAmphiteatre)
                {
                    ImGui::Begin("Avenche-Amphitheatre",
                                 &showErlebAR,
                                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
                    bool devLocIsUsed = AppDemo::devLoc.isUsed();
                    if (ImGui::Checkbox("Use GPS Location", &devLocIsUsed))
                        AppDemo::devLoc.isUsed(true);
#endif
                    for (int i = 1; i < AppDemo::devLoc.nameLocations().size(); ++i)
                    {
                        bool namedLocIsActive = namedLocIndex == i;
                        if (ImGui::Checkbox(AppDemo::devLoc.nameLocations()[i].name.c_str(), &namedLocIsActive))
                            setActiveNamedLocation(i, sv);
                    }

                    ImGui::End();
                }

                if (AppDemo::sceneID == SID_ErlebARAventicumCigognier)
                {
                    ImGui::Begin("Avenche-Cigognier",
                                 &showErlebAR,
                                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
                    bool devLocIsUsed = AppDemo::devLoc.isUsed();
                    if (ImGui::Checkbox("Use GPS Location", &devLocIsUsed))
                        AppDemo::devLoc.isUsed(true);
#endif
                    for (int i = 1; i < AppDemo::devLoc.nameLocations().size(); ++i)
                    {
                        bool namedLocIsActive = namedLocIndex == i;
                        if (ImGui::Checkbox(AppDemo::devLoc.nameLocations()[i].name.c_str(), &namedLocIsActive))
                            setActiveNamedLocation(i, sv, lookAtPoint);
                    }
                    ImGui::End();
                }

                if (AppDemo::sceneID == SID_ErlebARAventicumTheatre)
                {
                    ImGui::Begin("Avenche-Theatre",
                                 &showErlebAR,
                                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
                    bool devLocIsUsed = AppDemo::devLoc.isUsed();
                    if (ImGui::Checkbox("Use GPS Location", &devLocIsUsed))
                        AppDemo::devLoc.isUsed(true);
#endif
                    for (int i = 1; i < AppDemo::devLoc.nameLocations().size(); ++i)
                    {
                        bool namedLocIsActive = namedLocIndex == i;
                        if (ImGui::Checkbox(AppDemo::devLoc.nameLocations()[i].name.c_str(), &namedLocIsActive))
                            setActiveNamedLocation(i, sv);
                    }

                    ImGui::End();
                }

                if (AppDemo::sceneID == SID_ErlebARSutzKirchrain18)
                {
                    ImGui::Begin("Sutz-Kirchrain18",
                                 &showErlebAR,
                                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);

#if defined(SL_OS_MACIOS) || defined(SL_OS_ANDROID)
                    bool devLocIsUsed = AppDemo::devLoc.isUsed();
                    if (ImGui::Checkbox("Use GPS Location", &devLocIsUsed))
                        AppDemo::devLoc.isUsed(true);
#endif
                    for (int i = 1; i < AppDemo::devLoc.nameLocations().size(); ++i)
                    {
                        bool namedLocIsActive = namedLocIndex == i;
                        if (ImGui::Checkbox(AppDemo::devLoc.nameLocations()[i].name.c_str(), &namedLocIsActive))
                            setActiveNamedLocation(i, sv);
                    }

                    ImGui::End();
                }

                ImGui::PopFont();
            }
        }
    }
}

//-----------------------------------------------------------------------------
CVCalibration guessCalibration(bool         mirroredH,
                               bool         mirroredV,
                               CVCameraType camType)
{
    // Try to read device lens and sensor information
    string strF = AppDemo::deviceParameter["DeviceLensFocalLength"];
    string strW = AppDemo::deviceParameter["DeviceSensorPhysicalSizeW"];
    string strH = AppDemo::deviceParameter["DeviceSensorPhysicalSizeH"];
    if (!strF.empty() && !strW.empty() && !strH.empty())
    {
        float devF = strF.empty() ? 0.0f : stof(strF);
        float devW = strW.empty() ? 0.0f : stof(strW);
        float devH = strH.empty() ? 0.0f : stof(strH);

        // Changes the state to CS_guessed
        return CVCalibration(devW,
                             devH,
                             devF,
                             cv::Size(CVCapture::instance()->lastFrame.cols,
                                      CVCapture::instance()->lastFrame.rows),
                             mirroredH,
                             mirroredV,
                             camType,
                             Utils::ComputerInfos::get());
    }
    else
    {
        // make a guess using frame size and a guessed field of view
        return CVCalibration(cv::Size(CVCapture::instance()->lastFrame.cols,
                                      CVCapture::instance()->lastFrame.rows),
                             60.0,
                             mirroredH,
                             mirroredV,
                             camType,
                             Utils::ComputerInfos::get());
    }
}

//-----------------------------------------------------------------------------
//! Builds the entire menu bar once per frame
void AppDemoGui::buildMenuBar(SLProjectScene* s, SLSceneView* sv)
{
    PROFILE_FUNCTION();

    SLSceneID    sid           = AppDemo::sceneID;
    SLGLState*   stateGL       = SLGLState::instance();
    CVCapture*   capture       = CVCapture::instance();
    SLRenderType rType         = sv->renderType();
    SLbool       hasAnimations = (!s->animManager().allAnimNames().empty());
    static SLint curAnimIx     = -1;
    if (!hasAnimations) curAnimIx = -1;

    // Remove transform node if no or the wrong one is selected
    if (transformNode && s->singleNodeSelected() != transformNode->targetNode())
        removeTransformNode(s);

    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::BeginMenu("Load Test Scene"))
            {
                if (ImGui::BeginMenu("General Scenes"))
                {
                    if (ImGui::MenuItem("Minimal Scene", nullptr, sid == SID_Minimal))
                        s->onLoad(s, sv, SID_Minimal);
                    if (ImGui::MenuItem("Figure Scene", nullptr, sid == SID_Figure))
                        s->onLoad(s, sv, SID_Figure);
                    if (ImGui::MenuItem("Mesh Loader", nullptr, sid == SID_MeshLoad))
                        s->onLoad(s, sv, SID_MeshLoad);
                    if (ImGui::MenuItem("Revolver Meshes", nullptr, sid == SID_Revolver))
                        s->onLoad(s, sv, SID_Revolver);
                    if (ImGui::MenuItem("Texture Blending", nullptr, sid == SID_TextureBlend))
                        s->onLoad(s, sv, SID_TextureBlend);
                    if (ImGui::MenuItem("Texture Filters", nullptr, sid == SID_TextureFilter))
                        s->onLoad(s, sv, SID_TextureFilter);
#ifdef SL_BUILD_WITH_KTX
                    if (ImGui::MenuItem("Texture Compression", nullptr, sid == SID_TextureCompression))
                        s->onLoad(s, sv, SID_TextureCompression);
#endif
                    if (ImGui::MenuItem("Frustum Culling", nullptr, sid == SID_FrustumCull))
                        s->onLoad(s, sv, SID_FrustumCull);
                    if (ImGui::MenuItem("2D and 3D Text", nullptr, sid == SID_2Dand3DText))
                        s->onLoad(s, sv, SID_2Dand3DText);
                    if (ImGui::MenuItem("Point Clouds", nullptr, sid == SID_PointClouds))
                        s->onLoad(s, sv, SID_PointClouds);
                    if (ImGui::MenuItem("OpenVR", nullptr, sid == SID_OpenVR))
                        s->onLoad(s, sv, SID_OpenVR);

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Shader"))
                {
                    if (ImGui::MenuItem("Per Vertex Blinn-Phong", nullptr, sid == SID_ShaderPerVertexBlinn))
                        s->onLoad(s, sv, SID_ShaderPerVertexBlinn);
                    if (ImGui::MenuItem("Per Pixel Blinn-Phong", nullptr, sid == SID_ShaderPerPixelBlinn))
                        s->onLoad(s, sv, SID_ShaderPerPixelBlinn);
                    if (ImGui::MenuItem("Per Pixel Cook-Torrance", nullptr, sid == SID_ShaderCook))
                        s->onLoad(s, sv, SID_ShaderCook);
                    if (ImGui::MenuItem("Image Based Lighting", nullptr, sid == SID_ShaderIBL))
                        s->onLoad(s, sv, SID_ShaderIBL);
                    if (ImGui::MenuItem("Per Vertex Wave", nullptr, sid == SID_ShaderPerVertexWave))
                        s->onLoad(s, sv, SID_ShaderPerVertexWave);
                    if (ImGui::MenuItem("Bump Mapping", nullptr, sid == SID_ShaderBumpNormal))
                        s->onLoad(s, sv, SID_ShaderBumpNormal);
                    if (ImGui::MenuItem("Parallax Mapping", nullptr, sid == SID_ShaderBumpParallax))
                        s->onLoad(s, sv, SID_ShaderBumpParallax);
                    if (ImGui::MenuItem("Skybox Shader", nullptr, sid == SID_ShaderSkyBox))
                        s->onLoad(s, sv, SID_ShaderSkyBox);
                    if (ImGui::MenuItem("Earth Shader", nullptr, sid == SID_ShaderEarth))
                        s->onLoad(s, sv, SID_ShaderEarth);
#if defined(GL_VERSION_4_4)
                    if (ImGui::MenuItem("Voxel Cone Tracing", nullptr, sid == SID_ShaderVoxelConeDemo))
                        s->onLoad(s, sv, SID_ShaderVoxelConeDemo);
#endif
                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Shadow Mapping"))
                {
                    if (ImGui::MenuItem("Basic Scene", nullptr, sid == SID_ShadowMappingBasicScene))
                        s->onLoad(s, sv, SID_ShadowMappingBasicScene);
                    if (ImGui::MenuItem("Light Types", nullptr, sid == SID_ShadowMappingLightTypes))
                        s->onLoad(s, sv, SID_ShadowMappingLightTypes);
                    if (ImGui::MenuItem("8 Spot Lights", nullptr, sid == SID_ShadowMappingSpotLights))
                        s->onLoad(s, sv, SID_ShadowMappingSpotLights);
                    if (ImGui::MenuItem("3 Point Lights", nullptr, sid == SID_ShadowMappingPointLights))
                        s->onLoad(s, sv, SID_ShadowMappingPointLights);
                    if (ImGui::MenuItem("RT Soft Shadows", nullptr, sid == SID_RTSoftShadows))
                        s->onLoad(s, sv, SID_RTSoftShadows);

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Suzanne Lighting"))
                {
                    if (ImGui::MenuItem("w. per Pixel Lighting (PL)", nullptr, sid == SID_SuzannePerPixBlinn))
                        s->onLoad(s, sv, SID_SuzannePerPixBlinn);
                    if (ImGui::MenuItem("w. PL and Texture Mapping (TM)", nullptr, sid == SID_SuzannePerPixBlinnTm))
                        s->onLoad(s, sv, SID_SuzannePerPixBlinnTm);
                    if (ImGui::MenuItem("w. PL and Normal Mapping (NM)", nullptr, sid == SID_SuzannePerPixBlinnNm))
                        s->onLoad(s, sv, SID_SuzannePerPixBlinnNm);
                    if (ImGui::MenuItem("w. PL and Ambient Occlusion (AO)", nullptr, sid == SID_SuzannePerPixBlinnAo))
                        s->onLoad(s, sv, SID_SuzannePerPixBlinnAo);
                    if (ImGui::MenuItem("w. PL and Shadow Mapping (SM)", nullptr, sid == SID_SuzannePerPixBlinnSm))
                        s->onLoad(s, sv, SID_SuzannePerPixBlinnSm);
                    if (ImGui::MenuItem("w. PL, TM, NM", nullptr, sid == SID_SuzannePerPixBlinnTmNm))
                        s->onLoad(s, sv, SID_SuzannePerPixBlinnTmNm);
                    if (ImGui::MenuItem("w. PL, TM, AO", nullptr, sid == SID_SuzannePerPixBlinnTmAo))
                        s->onLoad(s, sv, SID_SuzannePerPixBlinnTmAo);
                    if (ImGui::MenuItem("w. PL, NM, AO", nullptr, sid == SID_SuzannePerPixBlinnNmAo))
                        s->onLoad(s, sv, SID_SuzannePerPixBlinnNmAo);
                    if (ImGui::MenuItem("w. PL, NM, SM", nullptr, sid == SID_SuzannePerPixBlinnNmSm))
                        s->onLoad(s, sv, SID_SuzannePerPixBlinnNmSm);
                    if (ImGui::MenuItem("w. PL, TM, SM", nullptr, sid == SID_SuzannePerPixBlinnTmSm))
                        s->onLoad(s, sv, SID_SuzannePerPixBlinnTmSm);
                    if (ImGui::MenuItem("w. PL, AO, SM", nullptr, sid == SID_SuzannePerPixBlinnAoSm))
                        s->onLoad(s, sv, SID_SuzannePerPixBlinnAoSm);
                    if (ImGui::MenuItem("w. PL, TM, NM, AO", nullptr, sid == SID_SuzannePerPixBlinnTmNmAo))
                        s->onLoad(s, sv, SID_SuzannePerPixBlinnTmNmAo);
                    if (ImGui::MenuItem("w. PL, TM, NM, SM", nullptr, sid == SID_SuzannePerPixBlinnTmNmSm))
                        s->onLoad(s, sv, SID_SuzannePerPixBlinnTmNmSm);
                    if (ImGui::MenuItem("w. PL, TM, NM, AO, SM", nullptr, sid == SID_SuzannePerPixBlinnTmNmAoSm))
                        s->onLoad(s, sv, SID_SuzannePerPixBlinnTmNmAoSm);
                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Volume Rendering"))
                {
                    if (ImGui::MenuItem("Head MRI Ray Cast", nullptr, sid == SID_VolumeRayCast))
                        s->onLoad(s, sv, SID_VolumeRayCast);

                    if (ImGui::MenuItem("Head MRI Ray Cast Lighted", nullptr, sid == SID_VolumeRayCastLighted))
                    {
                        auto loadMRIImages = []() {
                            AppDemo::jobProgressMsg("Load MRI Images");
                            AppDemo::jobProgressMax(100);

                            // Load volume data into 3D texture
                            SLVstring mriImages;
                            for (SLint i = 0; i < 207; ++i)
                                mriImages.push_back(AppDemo::texturePath + Utils::formatString("i%04u_0000b.png", i));

                            gTexMRI3D             = new SLGLTexture(nullptr,
                                                                    mriImages,
                                                                    GL_LINEAR,
                                                                    GL_LINEAR,
                                                                    0x812D, // GL_CLAMP_TO_BORDER (GLSL 320)
                                                                    0x812D, // GL_CLAMP_TO_BORDER (GLSL 320)
                                                                    "mri_head_front_to_back",
                                                                    true);
                            AppDemo::jobIsRunning = false;
                        };

                        auto calculateGradients = []() {
                            AppDemo::jobProgressMsg("Calculate MRI Volume Gradients");
                            AppDemo::jobProgressMax(100);
                            gTexMRI3D->calc3DGradients(1,
                                                       [](int progress) { AppDemo::jobProgressNum(progress); });
                            AppDemo::jobIsRunning = false;
                        };

                        auto smoothGradients = []() {
                            AppDemo::jobProgressMsg("Smooth MRI Volume Gradients");
                            AppDemo::jobProgressMax(100);
                            gTexMRI3D->smooth3DGradients(1,
                                                         [](int progress) { AppDemo::jobProgressNum(progress); });
                            AppDemo::jobIsRunning = false;
                        };

                        auto followUpJob1 = [](SLScene* s, SLSceneView* sv) {
                            s->onLoad(s, sv, SID_VolumeRayCastLighted);
                        };
                        function<void(void)> onLoadScene = bind(followUpJob1, s, sv);

                        AppDemo::jobsToBeThreaded.emplace_back(loadMRIImages);
                        AppDemo::jobsToBeThreaded.emplace_back(calculateGradients);
                        // AppDemo::jobsToBeThreaded.emplace_back(smoothGradients);  // very slow
                        AppDemo::jobsToFollowInMain.push_back(onLoadScene);
                    }

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Animation"))
                {
                    if (ImGui::MenuItem("Node Animation", nullptr, sid == SID_AnimationNode))
                        s->onLoad(s, sv, SID_AnimationNode);
                    if (ImGui::MenuItem("Mass Animation", nullptr, sid == SID_AnimationMass))
                        s->onLoad(s, sv, SID_AnimationMass);
                    if (ImGui::MenuItem("Skeletal Animation", nullptr, sid == SID_AnimationSkeletal))
                        s->onLoad(s, sv, SID_AnimationSkeletal);
                    if (ImGui::MenuItem("AstroBoy Army", nullptr, sid == SID_AnimationAstroboyArmy))
                        s->onLoad(s, sv, SID_AnimationAstroboyArmy);

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Using Video"))
                {
                    if (ImGui::MenuItem("Texture from Video Live", nullptr, sid == SID_VideoTextureLive))
                        s->onLoad(s, sv, SID_VideoTextureLive);
                    if (ImGui::MenuItem("Texture from Video File", nullptr, sid == SID_VideoTextureFile))
                        s->onLoad(s, sv, SID_VideoTextureFile);
                    if (ImGui::MenuItem("Track ArUco Marker (Main)", nullptr, sid == SID_VideoTrackArucoMain))
                        s->onLoad(s, sv, SID_VideoTrackArucoMain);
                    if (ImGui::MenuItem("Track ArUco Marker (Scnd)", nullptr, sid == SID_VideoTrackArucoScnd, capture->hasSecondaryCamera))
                        s->onLoad(s, sv, SID_VideoTrackArucoScnd);
                    if (ImGui::MenuItem("Track Chessboard (Main)", nullptr, sid == SID_VideoTrackChessMain))
                        s->onLoad(s, sv, SID_VideoTrackChessMain);
                    if (ImGui::MenuItem("Track Chessboard (Scnd)", nullptr, sid == SID_VideoTrackChessScnd, capture->hasSecondaryCamera))
                        s->onLoad(s, sv, SID_VideoTrackChessScnd);
                    if (ImGui::MenuItem("Track Features (Main)", nullptr, sid == SID_VideoTrackFeature2DMain))
                        s->onLoad(s, sv, SID_VideoTrackFeature2DMain);
                    if (ImGui::MenuItem("Track Face (Main)", nullptr, sid == SID_VideoTrackFaceMain))
                        s->onLoad(s, sv, SID_VideoTrackFaceMain);
                    if (ImGui::MenuItem("Track Face (Scnd)", nullptr, sid == SID_VideoTrackFaceScnd, capture->hasSecondaryCamera))
                        s->onLoad(s, sv, SID_VideoTrackFaceScnd);
                    if (ImGui::MenuItem("Sensor AR (Main)", nullptr, sid == SID_VideoSensorAR))
                        s->onLoad(s, sv, SID_VideoSensorAR);
#ifdef SL_BUILD_WAI
                    if (ImGui::MenuItem("Track WAI (Main)", nullptr, sid == SID_VideoTrackWAI))
                        s->onLoad(s, sv, SID_VideoTrackWAI);
#endif
                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Ray tracing"))
                {
                    if (ImGui::MenuItem("Spheres", nullptr, sid == SID_RTSpheres))
                        s->onLoad(s, sv, SID_RTSpheres);
                    if (ImGui::MenuItem("Muttenzer Box", nullptr, sid == SID_RTMuttenzerBox))
                        s->onLoad(s, sv, SID_RTMuttenzerBox);
                    if (ImGui::MenuItem("Soft Shadows", nullptr, sid == SID_RTSoftShadows))
                        s->onLoad(s, sv, SID_RTSoftShadows);
                    if (ImGui::MenuItem("Depth of Field", nullptr, sid == SID_RTDoF))
                        s->onLoad(s, sv, SID_RTDoF);
                    if (ImGui::MenuItem("Lens Test", nullptr, sid == SID_RTLens))
                        s->onLoad(s, sv, SID_RTLens);
                    if (ImGui::MenuItem("RT Test", nullptr, sid == SID_RTTest))
                        s->onLoad(s, sv, SID_RTTest);

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Path tracing"))
                {
                    if (ImGui::MenuItem("Muttenzer Box", nullptr, sid == SID_RTMuttenzerBox))
                        s->onLoad(s, sv, SID_RTMuttenzerBox);

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Benchmarks"))
                {
                    if (ImGui::MenuItem("Large Model (via FTP)", nullptr, sid == SID_Benchmark1_LargeModel))
                    {
                        SLstring largeFile = AppDemo::configPath + "models/xyzrgb_dragon/xyzrgb_dragon.ply";
                        if (Utils::fileExists(largeFile))
                            s->onLoad(s, sv, SID_Benchmark1_LargeModel);
                        else
                        {
                            auto downloadJobFTP = []() {
                                AppDemo::jobProgressMsg("Downloading large dragon file via FTP:");
                                AppDemo::jobProgressMax(100);
                                ftplib ftp;
                                ftp.SetConnmode(ftplib::connmode::port); // enable active mode

                                if (ftp.Connect("pallas.ti.bfh.ch:21"))
                                {
                                    if (ftp.Login("guest", "g2Q7Z7OkDP4!"))
                                    {
                                        ftp.SetCallbackXferFunction(ftpCallbackXfer);
                                        ftp.SetCallbackBytes(1024000);
                                        if (ftp.Chdir("data/SLProject/models"))
                                        {
                                            int remoteSize = 0;
                                            ftp.Size("xyzrgb_dragon.zip",
                                                     &remoteSize,
                                                     ftplib::transfermode::image);
                                            ftpXferSizeMax  = remoteSize;
                                            SLstring dstDir = AppDemo::configPath;
                                            if (Utils::dirExists(dstDir))
                                            {
                                                SLstring outFile = AppDemo::configPath + "models/xyzrgb_dragon.zip";
                                                if (!ftp.Get(outFile.c_str(),
                                                             "xyzrgb_dragon.zip",
                                                             ftplib::transfermode::image))
                                                    SL_LOG("*** ERROR: ftp.Get failed. ***");
                                            }
                                            else
                                                SL_LOG("*** ERROR: Destination directory does not exist: %s ***", dstDir.c_str());
                                        }
                                        else
                                            SL_LOG("*** ERROR: ftp.Chdir failed. ***");
                                    }
                                    else
                                        SL_LOG("*** ERROR: ftp.Login failed. ***");
                                }
                                else
                                    SL_LOG("*** ERROR: ftp.Connect failed. ***");

                                ftp.Quit();
                                AppDemo::jobIsRunning = false;
                            };

                            auto unzipJob = [largeFile]() {
                                AppDemo::jobProgressMsg("Decompress dragon file:");
                                AppDemo::jobProgressMax(-1);
                                string zipFile = AppDemo::configPath + "models/xyzrgb_dragon.zip";
                                if (Utils::fileExists(zipFile))
                                {
                                    ZipUtils::unzip(zipFile, Utils::getPath(zipFile));
                                    Utils::deleteFile(zipFile);
                                }
                                AppDemo::jobIsRunning = false;
                            };

                            auto followUpJob1 = [s, sv, largeFile]() {
                                if (Utils::fileExists(largeFile))
                                    s->onLoad(s, sv, SID_Benchmark1_LargeModel);
                            };

                            AppDemo::jobsToBeThreaded.emplace_back(downloadJobFTP);
                            AppDemo::jobsToBeThreaded.emplace_back(unzipJob);
                            AppDemo::jobsToFollowInMain.emplace_back(followUpJob1);
                        }
                    }
                    if (ImGui::MenuItem("Large Model (via HTTPS)", nullptr, sid == SID_Benchmark1_LargeModel))
                    {
                        SLstring largeFile = AppDemo::configPath + "models/xyzrgb_dragon/xyzrgb_dragon.ply";
                        if (Utils::fileExists(largeFile))
                            s->onLoad(s, sv, SID_Benchmark1_LargeModel);
                        else
                        {
                            downloadModelAndLoadScene(s,
                                                      sv,
                                                      "xyzrgb_dragon.zip",
                                                      "https://pallas.ti.bfh.ch/data/SLProject/models/",
                                                      AppDemo::configPath + "models/",
                                                      "xyzrgb_dragon/xyzrgb_dragon.ply",
                                                      SID_Benchmark1_LargeModel);
                        }
                    }
                    if (ImGui::MenuItem("Massive Nodes", nullptr, sid == SID_Benchmark2_MassiveNodes))
                        s->onLoad(s, sv, SID_Benchmark2_MassiveNodes);
                    if (ImGui::MenuItem("Massive Node Animations", nullptr, sid == SID_Benchmark3_NodeAnimations))
                        s->onLoad(s, sv, SID_Benchmark3_NodeAnimations);
                    if (ImGui::MenuItem("Massive Skinned Animations", nullptr, sid == SID_Benchmark4_SkinnedAnimations))
                        s->onLoad(s, sv, SID_Benchmark4_SkinnedAnimations);

                    if (Utils::fileExists(AppDemo::configPath + "GLTF/DragonLOD/Dragon_LOD.glb"))
                        if (ImGui::MenuItem("Level of Detail", nullptr, sid == SID_Benchmark4_LOD))
                            s->onLoad(s, sv, SID_Benchmark4_LOD);

                    ImGui::EndMenu();
                }

                SLstring erlebarPath = AppDemo::dataPath + "erleb-AR/models/";
                SLstring modelBR2    = erlebarPath + "bern/bern-christoffel.gltf";
                SLstring modelBFH    = erlebarPath + "biel/Biel-BFH-Rolex.gltf";
                SLstring modelAR1    = erlebarPath + "augst/augst-thtL1-tmpL2.gltf";
                SLstring modelAR2    = erlebarPath + "augst/augst-thtL2-tmpL1.gltf";
                SLstring modelAR3    = erlebarPath + "augst/augst-thtL1L2-tmpL1L2.gltf";
                SLstring modelAV1_AO = erlebarPath + "avenches/avenches-amphitheater.gltf";
                SLstring modelAV2_AO = erlebarPath + "avenches/avenches-cigognier.gltf";
                SLstring modelAV3    = erlebarPath + "avenches/avenches-theater.gltf";
                SLstring modelSU1    = erlebarPath + "sutzKirchrain18/Sutz-Kirchrain18.gltf";
                SLstring modelEV1    = erlebarPath + "evilardCheminDuRoc2/EvilardCheminDuRoc2.gltf";

                if (Utils::fileExists(modelAR1) ||
                    Utils::fileExists(modelAR2) ||
                    Utils::fileExists(modelAR3) ||
                    Utils::fileExists(modelAV3) ||
                    Utils::fileExists(modelBR2) ||
                    Utils::fileExists(modelSU1) ||
                    Utils::fileExists(modelEV1))
                {
                    if (ImGui::BeginMenu("Erleb-AR"))
                    {
                        if (Utils::fileExists(modelBR2))
                            if (ImGui::MenuItem("Bern: Christoffel Tower", nullptr, sid == SID_ErlebARChristoffel))
                                s->onLoad(s, sv, SID_ErlebARChristoffel);

                        if (Utils::fileExists(modelBFH))
                            if (ImGui::MenuItem("Biel: BFH", nullptr, sid == SID_ErlebARBielBFH))
                                s->onLoad(s, sv, SID_ErlebARBielBFH);

                        if (Utils::fileExists(modelAR1))
                            if (ImGui::MenuItem("Augusta Raurica Temple", nullptr, sid == SID_ErlebARAugustaRauricaTmp))
                                s->onLoad(s, sv, SID_ErlebARAugustaRauricaTmp);

                        if (Utils::fileExists(modelAR2))
                            if (ImGui::MenuItem("Augusta Raurica Theater", nullptr, sid == SID_ErlebARAugustaRauricaTht))
                                s->onLoad(s, sv, SID_ErlebARAugustaRauricaTht);

                        if (Utils::fileExists(modelAR3))
                            if (ImGui::MenuItem("Augusta Raurica Temple & Theater", nullptr, sid == SID_ErlebARAugustaRauricaTmpTht))
                                s->onLoad(s, sv, SID_ErlebARAugustaRauricaTmpTht);

                        if (Utils::fileExists(modelAV1_AO))
                            if (ImGui::MenuItem("Aventicum: Amphitheatre", nullptr, sid == SID_ErlebARAventicumAmphiteatre))
                                s->onLoad(s, sv, SID_ErlebARAventicumAmphiteatre);

                        if (Utils::fileExists(modelAV2_AO))
                            if (ImGui::MenuItem("Aventicum: Cigognier", nullptr, sid == SID_ErlebARAventicumCigognier))
                                s->onLoad(s, sv, SID_ErlebARAventicumCigognier);

                        if (Utils::fileExists(modelAV3))
                            if (ImGui::MenuItem("Aventicum: Theatre", nullptr, sid == SID_ErlebARAventicumTheatre))
                                s->onLoad(s, sv, SID_ErlebARAventicumTheatre);

                        if (Utils::fileExists(modelSU1))
                            if (ImGui::MenuItem("Sutz: Kirchrain 18", nullptr, sid == SID_ErlebARSutzKirchrain18))
                                s->onLoad(s, sv, SID_ErlebARSutzKirchrain18);

                        if (Utils::fileExists(modelEV1))
                            if (ImGui::MenuItem("Evilard: Chemin du Roc 2", nullptr, sid == SID_ErlebAREvilardCheminDuRoc2))
                                s->onLoad(s, sv, SID_ErlebAREvilardCheminDuRoc2);

                        ImGui::EndMenu();
                    }
                }

                ImGui::EndMenu();
            }

            if (ImGui::MenuItem("Empty Scene", "Shift-Alt-0", sid == SID_Empty))
                s->onLoad(s, sv, SID_Empty);

            if (ImGui::MenuItem("Next Scene",
                                "Shift-Alt-CursorRight",
                                nullptr,
                                AppDemo::sceneID < SID_Maximal - 1))
                s->onLoad(s, sv, AppDemo::sceneID + 1);

            if (ImGui::MenuItem("Previous Scene",
                                "Shift-Alt-CursorLeft",
                                nullptr,
                                AppDemo::sceneID > SID_Empty))
                s->onLoad(s, sv, AppDemo::sceneID - 1);

            ImGui::Separator();

            if (ImGui::MenuItem("Multi-threaded Jobs"))
            {
                auto job1 = []() {
                    uint maxIter = 100000;
                    AppDemo::jobProgressMsg("Super long job 1");
                    AppDemo::jobProgressMax(100);
                    for (uint i = 0; i < maxIter; ++i)
                    {
                        SL_LOG("%u", i);
                        int progressPC = (int)((float)i / (float)maxIter * 100.0f);
                        AppDemo::jobProgressNum(progressPC);
                    }
                    AppDemo::jobIsRunning = false;
                };

                auto job2 = []() {
                    uint maxIter = 100000;
                    AppDemo::jobProgressMsg("Super long job 2");
                    AppDemo::jobProgressMax(100);
                    for (uint i = 0; i < maxIter; ++i)
                    {
                        SL_LOG("%u", i);
                        int progressPC = (int)((float)i / (float)maxIter * 100.0f);
                        AppDemo::jobProgressNum(progressPC);
                    }
                    AppDemo::jobIsRunning = false;
                };

                auto followUpJob1 = []() { SL_LOG("followUpJob1"); };
                auto jobToFollow2 = []() { SL_LOG("JobToFollow2"); };

                AppDemo::jobsToBeThreaded.emplace_back(job1);
                AppDemo::jobsToBeThreaded.emplace_back(job2);
                AppDemo::jobsToFollowInMain.emplace_back(followUpJob1);
                AppDemo::jobsToFollowInMain.emplace_back(jobToFollow2);
            }

#ifndef SL_OS_ANDROID
            ImGui::Separator();

            if (ImGui::MenuItem("Quit & Save"))
                slShouldClose(true);
#endif

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Preferences"))
        {
            if (ImGui::MenuItem("Do Wait on Idle", "I", sv->doWaitOnIdle()))
                sv->doWaitOnIdle(!sv->doWaitOnIdle());

            if (ImGui::MenuItem("Do Multi Sampling", "L", sv->doMultiSampling()))
                sv->doMultiSampling(!sv->doMultiSampling());

            if (ImGui::MenuItem("Do Frustum Culling", "F", sv->doFrustumCulling()))
                sv->doFrustumCulling(!sv->doFrustumCulling());

            if (ImGui::MenuItem("Do Alpha Sorting", "J", sv->doAlphaSorting()))
                sv->doAlphaSorting(!sv->doAlphaSorting());

            if (ImGui::MenuItem("Do Depth Test", "T", sv->doDepthTest()))
                sv->doDepthTest(!sv->doDepthTest());

            if (ImGui::MenuItem("Animation off", "Space", s->stopAnimations()))
                s->stopAnimations(!s->stopAnimations());

            ImGui::Separator();

            if (ImGui::BeginMenu("Viewport Aspect"))
            {

                SLVec2i videoAspect(0, 0);
                if (capture->videoType() != VT_NONE)
                {
                    videoAspect.x = capture->captureSize.width;
                    videoAspect.y = capture->captureSize.height;
                }
                SLchar strSameAsVideo[256];
                sprintf(strSameAsVideo, "Same as Video (%d:%d)", videoAspect.x, videoAspect.y);

                if (ImGui::MenuItem("Same as window", nullptr, sv->viewportRatio() == SLVec2i::ZERO))
                    sv->setViewportFromRatio(SLVec2i(0, 0), sv->viewportAlign(), false);
                if (ImGui::MenuItem(strSameAsVideo, nullptr, sv->viewportSameAsVideo()))
                    sv->setViewportFromRatio(videoAspect, sv->viewportAlign(), true);
                if (ImGui::MenuItem("16:9", nullptr, sv->viewportRatio() == SLVec2i(16, 9)))
                    sv->setViewportFromRatio(SLVec2i(16, 9), sv->viewportAlign(), false);
                if (ImGui::MenuItem("4:3", nullptr, sv->viewportRatio() == SLVec2i(4, 3)))
                    sv->setViewportFromRatio(SLVec2i(4, 3), sv->viewportAlign(), false);
                if (ImGui::MenuItem("2:1", nullptr, sv->viewportRatio() == SLVec2i(2, 1)))
                    sv->setViewportFromRatio(SLVec2i(2, 1), sv->viewportAlign(), false);
                if (ImGui::MenuItem("1:1", nullptr, sv->viewportRatio() == SLVec2i(1, 1)))
                    sv->setViewportFromRatio(SLVec2i(1, 1), sv->viewportAlign(), false);

                if (ImGui::BeginMenu("Alignment", sv->viewportRatio() != SLVec2i::ZERO))
                {
                    if (ImGui::MenuItem("Center", nullptr, sv->viewportAlign() == VA_center))
                        sv->setViewportFromRatio(sv->viewportRatio(), VA_center, sv->viewportSameAsVideo());
                    if (ImGui::MenuItem("Left or top", nullptr, sv->viewportAlign() == VA_leftOrTop))
                        sv->setViewportFromRatio(sv->viewportRatio(), VA_leftOrTop, sv->viewportSameAsVideo());
                    if (ImGui::MenuItem("Right or bottom", nullptr, sv->viewportAlign() == VA_rightOrBottom))
                        sv->setViewportFromRatio(sv->viewportRatio(), VA_rightOrBottom, sv->viewportSameAsVideo());

                    ImGui::EndMenu();
                }
                ImGui::EndMenu();
            }

            ImGui::Separator();

            // Rotation and Location Sensor
#if defined(SL_OS_ANDROID) || defined(SL_OS_MACIOS)
            if (ImGui::BeginMenu("Rotation Sensor"))
            {
                SLDeviceRotation& devRot = AppDemo::devRot;

                if (ImGui::MenuItem("Use Device Rotation (IMU)", nullptr, devRot.isUsed()))
                    devRot.isUsed(!AppDemo::devRot.isUsed());

                if (devRot.isUsed())
                {
                    SLint numAveraged = devRot.numAveraged();
                    if (ImGui::SliderInt("Average length", &numAveraged, 1, 10))
                        devRot.numAveraged(numAveraged);

                    if (ImGui::BeginMenu("Offset Mode"))
                    {
                        SLRotOffsetMode om = devRot.offsetMode();
                        if (ImGui::MenuItem("None", nullptr, om == ROM_none))
                            devRot.offsetMode(ROM_none);
                        if (ImGui::MenuItem("Finger rot. X", nullptr, om == ROM_oneFingerX))
                            devRot.offsetMode(ROM_oneFingerX);
                        if (ImGui::MenuItem("Finger rot. X and Y", nullptr, om == ROM_oneFingerXY))
                            devRot.offsetMode(ROM_oneFingerXY);

                        ImGui::EndMenu();
                    }

                    if (ImGui::MenuItem("Zero Yaw at Start", nullptr, devRot.zeroYawAtStart()))
                        devRot.zeroYawAtStart(!devRot.zeroYawAtStart());

                    if (ImGui::MenuItem("Reset Zero Yaw"))
                        devRot.hasStarted(true);

                    if (ImGui::MenuItem("Show Horizon", nullptr, _horizonVisuEnabled))
                    {
                        if (_horizonVisuEnabled)
                            hideHorizon(s);
                        else
                            showHorizon(s, sv);
                    }
                }

                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Location Sensor"))
            {
                SLDeviceLocation& devLoc = AppDemo::devLoc;

                if (ImGui::MenuItem("Use Device Location (GPS)", nullptr, AppDemo::devLoc.isUsed()))
                    AppDemo::devLoc.isUsed(!AppDemo::devLoc.isUsed());

                if (!AppDemo::devLoc.geoTiffIsAvailableAndValid())
                    if (ImGui::MenuItem("Use Origin Altitude", nullptr, AppDemo::devLoc.useOriginAltitude()))
                        AppDemo::devLoc.useOriginAltitude(!AppDemo::devLoc.useOriginAltitude());

                if (ImGui::MenuItem("Reset Origin to here"))
                    AppDemo::devLoc.hasOrigin(false);

                if (ImGui::BeginMenu("Offset Mode"))
                {
                    SLLocOffsetMode om = devLoc.offsetMode();
                    if (ImGui::MenuItem("None", nullptr, om == LOM_none))
                        devLoc.offsetMode(LOM_none);
                    if (ImGui::MenuItem("Two Finger Y", nullptr, om == LOM_twoFingerY))
                        devLoc.offsetMode(LOM_twoFingerY);

                    ImGui::EndMenu();
                }

                ImGui::EndMenu();
            }
#endif

            if (ImGui::BeginMenu("Video Sensor"))
            {
                CVCamera* ac = capture->activeCamera;
                if (ImGui::BeginMenu("Mirror Camera"))
                {
                    if (ImGui::MenuItem("Horizontally", nullptr, ac->mirrorH()))
                    {
                        ac->toggleMirrorH();
                        // make a guessed calibration, if there was a calibrated camera it is not valid anymore
                        ac->calibration = guessCalibration(ac->mirrorH(), ac->mirrorV(), ac->type());
                    }

                    if (ImGui::MenuItem("Vertically", nullptr, ac->mirrorV()))
                    {
                        ac->toggleMirrorV();
                        // make a guessed calibration, if there was a calibrated camera it is not valid anymore
                        ac->calibration = guessCalibration(ac->mirrorH(), ac->mirrorV(), ac->type());
                    }

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Resolution",
                                     (capture->videoType() == VT_MAIN ||
                                      capture->videoType() == VT_SCND)))
                {
                    for (int i = 0; i < (int)capture->camSizes.size(); ++i)
                    {
                        SLchar menuStr[256];
                        sprintf(menuStr,
                                "%d x %d",
                                capture->camSizes[(uint)i].width,
                                capture->camSizes[(uint)i].height);
                        if (ImGui::MenuItem(menuStr, nullptr, i == capture->activeCamSizeIndex))
                            if (i != capture->activeCamSizeIndex)
                                ac->camSizeIndex(i);
                    }
                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Calibration"))
                {
                    if (ImGui::MenuItem("Start Calibration (Main Camera)"))
                    {
                        s->onLoad(s, sv, SID_VideoCalibrateMain);
                        showHelpCalibration = false;
                        showInfosScene      = true;
                    }

                    if (ImGui::MenuItem("Start Calibration (Scnd. Camera)", nullptr, false, capture->hasSecondaryCamera))
                    {
                        s->onLoad(s, sv, SID_VideoCalibrateScnd);
                        showHelpCalibration = false;
                        showInfosScene      = true;
                    }

                    if (ImGui::MenuItem("Undistort Image", nullptr, ac->showUndistorted(), ac->calibration.state() == CS_calibrated))
                        ac->showUndistorted(!ac->showUndistorted());

                    if (ImGui::MenuItem("No Tangent Distortion", nullptr, AppDemo::calibrationEstimatorParams.zeroTangentDistortion))
                        AppDemo::calibrationEstimatorParams.toggleZeroTangentDist();

                    if (ImGui::MenuItem("Fix Aspect Ratio", nullptr, AppDemo::calibrationEstimatorParams.fixAspectRatio))
                        AppDemo::calibrationEstimatorParams.toggleFixAspectRatio();

                    if (ImGui::MenuItem("Fix Principal Point", nullptr, AppDemo::calibrationEstimatorParams.fixPrincipalPoint))
                        AppDemo::calibrationEstimatorParams.toggleFixPrincipalPoint();

                    if (ImGui::MenuItem("Use rational model", nullptr, AppDemo::calibrationEstimatorParams.calibRationalModel))
                        AppDemo::calibrationEstimatorParams.toggleRationalModel();

                    if (ImGui::MenuItem("Use tilted model", nullptr, AppDemo::calibrationEstimatorParams.calibTiltedModel))
                        AppDemo::calibrationEstimatorParams.toggleTiltedModel();

                    if (ImGui::MenuItem("Use thin prism model", nullptr, AppDemo::calibrationEstimatorParams.calibThinPrismModel))
                        AppDemo::calibrationEstimatorParams.toggleThinPrismModel();

                    ImGui::EndMenu();
                }

                CVTrackedFeatures* featureTracker = nullptr;
                if (tracker != nullptr && typeid(*tracker) == typeid(CVTrackedFeatures))
                    featureTracker = (CVTrackedFeatures*)tracker;

                if (tracker != nullptr)
                    if (ImGui::MenuItem("Draw Detection", nullptr, tracker->drawDetection()))
                        tracker->drawDetection(!tracker->drawDetection());

                if (ImGui::BeginMenu("Feature Tracking", featureTracker != nullptr))
                {
                    if (ImGui::MenuItem("Force Relocation", nullptr, featureTracker->forceRelocation()))
                        featureTracker->forceRelocation(!featureTracker->forceRelocation());

                    if (ImGui::BeginMenu("Detector/Descriptor", featureTracker != nullptr))
                    {
                        CVDetectDescribeType type = featureTracker->type();

                        if (ImGui::MenuItem("RAUL/RAUL", nullptr, type == DDT_RAUL_RAUL))
                            featureTracker->type(DDT_RAUL_RAUL);
                        if (ImGui::MenuItem("ORB/ORB", nullptr, type == DDT_ORB_ORB))
                            featureTracker->type(DDT_ORB_ORB);
                        if (ImGui::MenuItem("FAST/BRIEF", nullptr, type == DDT_FAST_BRIEF))
                            featureTracker->type(DDT_FAST_BRIEF);
                        if (ImGui::MenuItem("SURF/SURF", nullptr, type == DDT_SURF_SURF))
                            featureTracker->type(DDT_SURF_SURF);
                        if (ImGui::MenuItem("SIFT/SIFT", nullptr, type == DDT_SIFT_SIFT))
                            featureTracker->type(DDT_SIFT_SIFT);

                        ImGui::EndMenu();
                    }

                    ImGui::EndMenu();
                }

                ImGui::EndMenu();
            }

            ImGui::Separator();

            ImGui::MenuItem("UI Preferences", nullptr, &showUIPrefs);

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Edit", s->singleNodeSelected() != nullptr || !sv->camera()->selectRect().isZero()))
        {
            if (s->singleNodeSelected())
            {
                buildMenuEdit(s, sv);
            }
            else
            {
                if (ImGui::MenuItem("Clear selection"))
                {
                    sv->camera()->selectRect().setZero();
                    sv->camera()->deselectRect().setZero();
                }
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Renderer"))
        {
            if (ImGui::MenuItem("OpenGL", "ESC", rType == RT_gl))
                sv->renderType(RT_gl);

            if (ImGui::MenuItem("Ray Tracing", "R", rType == RT_rt))
                sv->startRaytracing(5);

            if (ImGui::MenuItem("Path Tracing", "P", rType == RT_pt))
                sv->startPathtracing(5, 10);

#ifdef SL_HAS_OPTIX
            if (ImGui::MenuItem("Ray Tracing with OptiX", "Shift-R", rType == RT_optix_rt))
                sv->startOptixRaytracing(5);

            if (ImGui::MenuItem("Path Tracing with OptiX", "Shift-P", rType == RT_optix_pt))
                sv->startOptixPathtracing(5, 10);
#else
            ImGui::MenuItem("Ray Tracing with OptiX", nullptr, false, false);
            ImGui::MenuItem("Path Tracing with OptiX", nullptr, false, false);
#endif
#ifdef GL_VERSION_4_4
            if (gl3wIsSupported(4, 4))
            {
                if (ImGui::MenuItem("Cone Tracing (CT)", "C", rType == RT_ct))
                    sv->startConetracing();
            }
            else
#endif
                ImGui::MenuItem("Cone Tracing (GL>4.4)", nullptr, false, false);

            ImGui::EndMenu();
        }

        if (rType == RT_gl)
        {
            if (ImGui::BeginMenu("GL"))
            {
                if (ImGui::MenuItem("Mesh Wired", "M", sv->drawBits()->get(SL_DB_MESHWIRED)))
                    sv->drawBits()->toggle(SL_DB_MESHWIRED);

                if (ImGui::MenuItem("Width hard edges", "H", sv->drawBits()->get(SL_DB_WITHEDGES)))
                    sv->drawBits()->toggle(SL_DB_WITHEDGES);

                if (ImGui::MenuItem("Only hard edges", "O", sv->drawBits()->get(SL_DB_ONLYEDGES)))
                    sv->drawBits()->toggle(SL_DB_ONLYEDGES);

                if (ImGui::MenuItem("Normals", "N", sv->drawBits()->get(SL_DB_NORMALS)))
                    sv->drawBits()->toggle(SL_DB_NORMALS);

                if (ImGui::MenuItem("Bounding Boxes", "B", sv->drawBits()->get(SL_DB_BBOX)))
                    sv->drawBits()->toggle(SL_DB_BBOX);

                if (ImGui::MenuItem("Voxels", "V", sv->drawBits()->get(SL_DB_VOXELS)))
                    sv->drawBits()->toggle(SL_DB_VOXELS);

                if (ImGui::MenuItem("Axis", "X", sv->drawBits()->get(SL_DB_AXIS)))
                    sv->drawBits()->toggle(SL_DB_AXIS);

                if (ImGui::MenuItem("Back Faces", "C", sv->drawBits()->get(SL_DB_CULLOFF)))
                    sv->drawBits()->toggle(SL_DB_CULLOFF);

                if (ImGui::MenuItem("Skeleton", "K", sv->drawBits()->get(SL_DB_SKELETON)))
                    sv->drawBits()->toggle(SL_DB_SKELETON);

                if (ImGui::MenuItem("All off"))
                    sv->drawBits()->allOff();

                if (ImGui::MenuItem("All on"))
                {
                    sv->drawBits()->on(SL_DB_MESHWIRED);
                    sv->drawBits()->on(SL_DB_WITHEDGES);
                    sv->drawBits()->on(SL_DB_ONLYEDGES);
                    sv->drawBits()->on(SL_DB_NORMALS);
                    sv->drawBits()->on(SL_DB_VOXELS);
                    sv->drawBits()->on(SL_DB_AXIS);
                    sv->drawBits()->on(SL_DB_BBOX);
                    sv->drawBits()->on(SL_DB_SKELETON);
                    sv->drawBits()->on(SL_DB_CULLOFF);
                }

                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.65f);
                SLfloat gamma = SLLight::gamma;
                if (ImGui::SliderFloat("Gamma", &gamma, 0.1f, 3.0f, "%.1f"))
                    SLLight::gamma = gamma;
                ImGui::PopItemWidth();

                ImGui::EndMenu();
            }
        }
        else if (rType == RT_rt)
        {
            if (ImGui::BeginMenu("RT"))
            {
                SLRaytracer* rt = sv->raytracer();

                if (ImGui::BeginMenu("Resolution Factor"))
                {
                    if (ImGui::MenuItem("1.00", nullptr, rt->resolutionFactorPC() == 100))
                    {
                        rt->resolutionFactor(1.0f);
                        sv->startRaytracing(rt->maxDepth());
                    }
                    if (ImGui::MenuItem("0.50", nullptr, rt->resolutionFactorPC() == 50))
                    {
                        rt->resolutionFactor(0.5f);
                        sv->startRaytracing(rt->maxDepth());
                    }
                    if (ImGui::MenuItem("0.25", nullptr, rt->resolutionFactorPC() == 25))
                    {
                        rt->resolutionFactor(0.25f);
                        sv->startRaytracing(rt->maxDepth());
                    }

                    ImGui::EndMenu();
                }

                if (ImGui::MenuItem("Parallel distributed", nullptr, rt->doDistributed()))
                {
                    rt->doDistributed(!rt->doDistributed());
                    sv->startRaytracing(rt->maxDepth());
                }

                if (ImGui::MenuItem("Continuously", nullptr, rt->doContinuous()))
                {
                    rt->doContinuous(!rt->doContinuous());
                    sv->doWaitOnIdle(!rt->doContinuous());
                }

                if (ImGui::MenuItem("Fresnel Reflection", nullptr, rt->doFresnel()))
                {
                    rt->doFresnel(!rt->doFresnel());
                    sv->startRaytracing(rt->maxDepth());
                }

                if (ImGui::BeginMenu("Max. Depth"))
                {
                    if (ImGui::MenuItem("1", nullptr, rt->maxDepth() == 1)) sv->startRaytracing(1);
                    if (ImGui::MenuItem("2", nullptr, rt->maxDepth() == 2)) sv->startRaytracing(2);
                    if (ImGui::MenuItem("3", nullptr, rt->maxDepth() == 3)) sv->startRaytracing(3);
                    if (ImGui::MenuItem("5", nullptr, rt->maxDepth() == 5)) sv->startRaytracing(5);
                    if (ImGui::MenuItem("Max. Contribution", nullptr, rt->maxDepth() == 0)) sv->startRaytracing(0);

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Anti-Aliasing Samples"))
                {
                    if (ImGui::MenuItem("Off", nullptr, rt->aaSamples() == 1)) rt->aaSamples(1);
                    if (ImGui::MenuItem("3x3", nullptr, rt->aaSamples() == 3)) rt->aaSamples(3);
                    if (ImGui::MenuItem("5x5", nullptr, rt->aaSamples() == 5)) rt->aaSamples(5);
                    if (ImGui::MenuItem("7x7", nullptr, rt->aaSamples() == 7)) rt->aaSamples(7);
                    if (ImGui::MenuItem("9x9", nullptr, rt->aaSamples() == 9)) rt->aaSamples(9);

                    ImGui::EndMenu();
                }

                if (ImGui::MenuItem("Save Rendered Image"))
                    rt->saveImage();

                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.65f);
                SLfloat gamma = rt->gamma();
                if (ImGui::SliderFloat("Gamma", &gamma, 0.1f, 3.0f, "%.1f"))
                {
                    rt->gamma(gamma);
                    sv->startRaytracing(5);
                }
                ImGui::PopItemWidth();

                ImGui::EndMenu();
            }
        }

#ifdef SL_HAS_OPTIX
        else if (rType == RT_optix_rt)
        {
            if (ImGui::BeginMenu("RT"))
            {
                SLOptixRaytracer* rt_optix = sv->optixRaytracer();

                if (ImGui::MenuItem("Parallel distributed", nullptr, rt_optix->doDistributed()))
                {
                    rt_optix->doDistributed(!rt_optix->doDistributed());
                    sv->startOptixRaytracing(rt_optix->maxDepth());
                }

                //                if (ImGui::MenuItem("Fresnel Reflection", nullptr, rt->doFresnel()))
                //                {
                //                    rt->doFresnel(!rt->doFresnel());
                //                    sv->startRaytracing(rt->maxDepth());
                //                }

                if (ImGui::BeginMenu("Max. Depth"))
                {
                    if (ImGui::MenuItem("1", nullptr, rt_optix->maxDepth() == 1))
                        sv->startOptixRaytracing(1);
                    if (ImGui::MenuItem("2", nullptr, rt_optix->maxDepth() == 2))
                        sv->startOptixRaytracing(2);
                    if (ImGui::MenuItem("3", nullptr, rt_optix->maxDepth() == 3))
                        sv->startOptixRaytracing(3);
                    if (ImGui::MenuItem("5", nullptr, rt_optix->maxDepth() == 5))
                        sv->startOptixRaytracing(5);
                    if (ImGui::MenuItem("Max. Contribution", nullptr, rt_optix->maxDepth() == 0))
                        sv->startOptixRaytracing(0);

                    ImGui::EndMenu();
                }

                //                if (ImGui::BeginMenu("Anti-Aliasing Samples"))
                //                {
                //                    if (ImGui::MenuItem("Off", nullptr, rt->aaSamples() == 1))
                //                        rt->aaSamples(1);
                //                    if (ImGui::MenuItem("3x3", nullptr, rt->aaSamples() == 3))
                //                        rt->aaSamples(3);
                //                    if (ImGui::MenuItem("5x5", nullptr, rt->aaSamples() == 5))
                //                        rt->aaSamples(5);
                //                    if (ImGui::MenuItem("7x7", nullptr, rt->aaSamples() == 7))
                //                        rt->aaSamples(7);
                //                    if (ImGui::MenuItem("9x9", nullptr, rt->aaSamples() == 9))
                //                        rt->aaSamples(9);
                //
                //                    ImGui::EndMenu();
                //                }

                if (ImGui::MenuItem("Save Rendered Image"))
                    rt_optix->saveImage();

                ImGui::EndMenu();
            }
        }
#endif
        else if (rType == RT_pt)
        {
            if (ImGui::BeginMenu("PT"))
            {
                SLPathtracer* pt = sv->pathtracer();

                if (ImGui::BeginMenu("Resolution Factor"))
                {
                    if (ImGui::MenuItem("1.00", nullptr, pt->resolutionFactorPC() == 100))
                    {
                        pt->resolutionFactor(1.0f);
                        sv->startPathtracing(5, pt->aaSamples());
                    }
                    if (ImGui::MenuItem("0.50", nullptr, pt->resolutionFactorPC() == 50))
                    {
                        pt->resolutionFactor(0.5f);
                        sv->startPathtracing(5, pt->aaSamples());
                    }
                    if (ImGui::MenuItem("0.25", nullptr, pt->resolutionFactorPC() == 25))
                    {
                        pt->resolutionFactor(0.25f);
                        sv->startPathtracing(5, pt->aaSamples());
                    }

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("NO. of Samples"))
                {
                    if (ImGui::MenuItem("1", nullptr, pt->aaSamples() == 1)) sv->startPathtracing(5, 1);
                    if (ImGui::MenuItem("10", nullptr, pt->aaSamples() == 10)) sv->startPathtracing(5, 10);
                    if (ImGui::MenuItem("100", nullptr, pt->aaSamples() == 100)) sv->startPathtracing(5, 100);
                    if (ImGui::MenuItem("1000", nullptr, pt->aaSamples() == 1000)) sv->startPathtracing(5, 1000);
                    if (ImGui::MenuItem("10000", nullptr, pt->aaSamples() == 10000)) sv->startPathtracing(5, 10000);

                    ImGui::EndMenu();
                }

                if (ImGui::MenuItem("Direct illumination", nullptr, pt->calcDirect()))
                {
                    pt->calcDirect(!pt->calcDirect());
                    sv->startPathtracing(5, 10);
                }

                if (ImGui::MenuItem("Indirect illumination", nullptr, pt->calcIndirect()))
                {
                    pt->calcIndirect(!pt->calcIndirect());
                    sv->startPathtracing(5, 10);
                }

                if (ImGui::MenuItem("Save Rendered Image"))
                    pt->saveImage();

                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.65f);
                SLfloat gamma = pt->gamma();
                if (ImGui::SliderFloat("Gamma", &gamma, 0.1f, 3.0f, "%.1f"))
                {
                    pt->gamma(gamma);
                    sv->startPathtracing(5, 1);
                }
                ImGui::PopItemWidth();

                ImGui::EndMenu();
            }
        }
        else if (rType == RT_ct)
        {
            if (ImGui::BeginMenu("CT-Setting"))
            {
                if (ImGui::MenuItem("Show Voxelization", nullptr, sv->conetracer()->showVoxels()))
                    sv->conetracer()->toggleVoxels();

                if (ImGui::MenuItem("Direct illumination", nullptr, sv->conetracer()->doDirectIllum()))
                    sv->conetracer()->toggleDirectIllum();

                if (ImGui::MenuItem("Diffuse indirect illumination", nullptr, sv->conetracer()->doDiffuseIllum()))
                    sv->conetracer()->toggleDiffuseIllum();

                if (ImGui::MenuItem("Specular indirect illumination", nullptr, sv->conetracer()->doSpecularIllum()))
                    sv->conetracer()->toggleSpecIllumination();

                if (ImGui::MenuItem("Shadows", nullptr, sv->conetracer()->shadows()))
                    sv->conetracer()->toggleShadows();

                SLfloat angle = sv->conetracer()->diffuseConeAngle();
                if (ImGui::SliderFloat("Diffuse cone angle (rad)", &angle, 0.f, 1.5f))
                    sv->conetracer()->diffuseConeAngle(angle);

                SLfloat specAngle = sv->conetracer()->specularConeAngle();
                if (ImGui::SliderFloat("Specular cone angle (rad)", &specAngle, 0.004f, 0.5f))
                    sv->conetracer()->specularConeAngle(specAngle);

                SLfloat shadowAngle = sv->conetracer()->shadowConeAngle();
                if (ImGui::SliderFloat("Shadow cone angle (rad)", &shadowAngle, 0.f, 1.5f))
                    sv->conetracer()->shadowConeAngle(shadowAngle);

                SLfloat lightSize = sv->conetracer()->lightMeshSize();
                if (ImGui::SliderFloat("Max. size of a lightsource mesh", &lightSize, 0.0f, 100.0f))
                    sv->conetracer()->lightMeshSize(lightSize);

                SLfloat gamma = sv->conetracer()->gamma();
                if (ImGui::SliderFloat("Gamma", &gamma, 1.0f, 3.0f))
                    sv->conetracer()->gamma(gamma);

                ImGui::EndMenu();
            }
        }

#ifdef SL_HAS_OPTIX
        else if (rType == RT_optix_pt)
        {
            if (ImGui::BeginMenu("PT"))
            {
                SLOptixPathtracer* pt = sv->optixPathtracer();

                if (ImGui::BeginMenu("NO. of Samples"))
                {
                    if (ImGui::MenuItem("1", nullptr, pt->samples() == 1))
                        sv->startOptixPathtracing(5, 1);
                    if (ImGui::MenuItem("10", nullptr, pt->samples() == 10))
                        sv->startOptixPathtracing(5, 10);
                    if (ImGui::MenuItem("100", nullptr, pt->samples() == 100))
                        sv->startOptixPathtracing(5, 100);
                    if (ImGui::MenuItem("1000", nullptr, pt->samples() == 1000))
                        sv->startOptixPathtracing(5, 1000);
                    if (ImGui::MenuItem("10000", nullptr, pt->samples() == 10000))
                        sv->startOptixPathtracing(5, 10000);

                    ImGui::EndMenu();
                }

                if (ImGui::MenuItem("Denoiser", nullptr, pt->getDenoiserEnabled()))
                {
                    pt->setDenoiserEnabled(!pt->getDenoiserEnabled());
                    sv->startOptixPathtracing(5, pt->samples());
                }

                if (ImGui::MenuItem("Save Rendered Image"))
                    pt->saveImage();

                ImGui::EndMenu();
            }
        }
#endif

        if (ImGui::BeginMenu("Camera"))
        {
            SLCamera*    cam  = sv->camera();
            SLProjection proj = cam->projection();

            if (ImGui::MenuItem("Reset"))
                cam->resetToInitialState();

            if (ImGui::BeginMenu("Look from"))
            {
                if (ImGui::MenuItem("Left (+X)", "3")) cam->lookFrom(SLVec3f::AXISX);
                if (ImGui::MenuItem("Right (-X)", "CTRL-3")) cam->lookFrom(-SLVec3f::AXISX);
                if (ImGui::MenuItem("Top (+Y)", "7")) cam->lookFrom(SLVec3f::AXISY, -SLVec3f::AXISZ);
                if (ImGui::MenuItem("Bottom (-Y)", "CTRL-7")) cam->lookFrom(-SLVec3f::AXISY, SLVec3f::AXISZ);
                if (ImGui::MenuItem("Front (+Z)", "1")) cam->lookFrom(SLVec3f::AXISZ);
                if (ImGui::MenuItem("Back (-Z)", "CTRL-1")) cam->lookFrom(-SLVec3f::AXISZ);

                if (s->numSceneCameras())
                {
                    if (ImGui::MenuItem("Next camera in Scene", "TAB"))
                        sv->switchToNextCameraInScene();

                    if (ImGui::MenuItem("Sceneview Camera", "TAB"))
                        sv->switchToSceneViewCamera();
                }

                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Projection"))
            {
                static SLfloat clipN     = cam->clipNear();
                static SLfloat clipF     = cam->clipFar();
                static SLfloat focalDist = cam->focalDist();
                static SLfloat fov       = cam->fovV();

                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.66f);

                if (ImGui::MenuItem("Perspective", "5", proj == P_monoPerspective))
                {
                    cam->projection(P_monoPerspective);
                    if (sv->renderType() == RT_rt && !sv->raytracer()->doContinuous() &&
                        sv->raytracer()->state() == rtFinished)
                        sv->raytracer()->state(rtReady);
                }

                if (ImGui::MenuItem("Orthographic", "5", proj == P_monoOrthographic))
                {
                    cam->projection(P_monoOrthographic);
                    if (sv->renderType() == RT_rt && !sv->raytracer()->doContinuous() &&
                        sv->raytracer()->state() == rtFinished)
                        sv->raytracer()->state(rtReady);
                }

                if (ImGui::BeginMenu("Stereo"))
                {
                    for (SLint p = P_stereoSideBySide; p <= P_stereoOpenVR; ++p)
                    {
                        SLstring pStr = SLCamera::projectionToStr((SLProjection)p);
                        if (ImGui::MenuItem(pStr.c_str(), nullptr, proj == (SLProjection)p))
                            cam->projection((SLProjection)p);
                    }

                    if (proj >= P_stereoSideBySide)
                    {
                        ImGui::Separator();
                        static SLfloat eyeSepar = cam->stereoEyeSeparation();
                        if (ImGui::SliderFloat("Eye Sep.", &eyeSepar, 0.0f, focalDist / 10.f))
                            cam->stereoEyeSeparation(eyeSepar);
                    }

                    ImGui::EndMenu();
                }

                ImGui::Separator();

                if (ImGui::SliderFloat("FOV", &fov, 1.f, 179.f))
                    cam->fov(fov);

                if (ImGui::SliderFloat("Near Clip", &clipN, 0.001f, 10.f))
                    cam->clipNear(clipN);

                if (ImGui::SliderFloat("Focal Dist.", &focalDist, clipN, clipF))
                    cam->focalDist(focalDist);

                if (ImGui::SliderFloat("Far Clip", &clipF, clipN, std::min(clipF * 1.1f, 1000000.f)))
                    cam->clipFar(clipF);

                ImGui::PopItemWidth();
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Animation"))
            {
                SLCamAnim ca = cam->camAnim();

                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.66f);

                if (ImGui::MenuItem("Turntable Y up", nullptr, ca == CA_turntableYUp))
                    sv->camera()->camAnim(CA_turntableYUp);

                if (ImGui::MenuItem("Turntable Z up", nullptr, ca == CA_turntableZUp))
                    sv->camera()->camAnim(CA_turntableZUp);

                if (ImGui::MenuItem("Trackball", nullptr, ca == CA_trackball))
                    sv->camera()->camAnim(CA_trackball);

                if (ImGui::MenuItem("Walk Y up", nullptr, ca == CA_walkingYUp))
                    sv->camera()->camAnim(CA_walkingYUp);

                if (ImGui::MenuItem("Walk Z up", nullptr, ca == CA_walkingZUp))
                    sv->camera()->camAnim(CA_walkingZUp);

                float mouseRotFactor = sv->camera()->mouseRotationFactor();
                if (ImGui::SliderFloat("Mouse Sensibility", &mouseRotFactor, 0.1f, 2.0f, "%2.1f"))
                    sv->camera()->mouseRotationFactor(mouseRotFactor);

                ImGui::Separator();

                if (ImGui::MenuItem("IMU rotated", nullptr, ca == CA_deviceRotYUp))
                    sv->camera()->camAnim(CA_deviceRotYUp);

                if (ImGui::MenuItem("IMU rotated & GPS located", nullptr, ca == CA_deviceRotLocYUp))
                    sv->camera()->camAnim(CA_deviceRotLocYUp);

                if (ca == CA_walkingZUp || ca == CA_walkingYUp || ca == CA_deviceRotYUp)
                {
                    static SLfloat ms = cam->maxSpeed();
                    if (ImGui::SliderFloat("Walk Speed", &ms, 0.01f, std::min(ms * 1.1f, 10000.f)))
                        cam->maxSpeed(ms);
                }

                ImGui::PopItemWidth();
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Fog"))
            {
                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.66f);

                if (ImGui::MenuItem("Fog is on", nullptr, cam->fogIsOn()))
                    cam->fogIsOn(!cam->fogIsOn());

                if (ImGui::BeginMenu("Mode"))
                {
                    if (ImGui::MenuItem("linear", nullptr, cam->fogMode() == FM_linear))
                        cam->fogMode(FM_linear);
                    if (ImGui::MenuItem("exp", nullptr, cam->fogMode() == FM_exp))
                        cam->fogMode(FM_exp);
                    if (ImGui::MenuItem("exp2", nullptr, cam->fogMode() == FM_exp2))
                        cam->fogMode(FM_exp2);
                    ImGui::EndMenu();
                }

                if (cam->fogMode() == FM_exp || cam->fogMode() == FM_exp2)
                {
                    static SLfloat fogDensity = cam->fogDensity();
                    if (ImGui::SliderFloat("Density", &fogDensity, 0.0f, 1.0f))
                        cam->fogDensity(fogDensity);
                }

                ImGui::PopItemWidth();
                ImGui::EndMenu();
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Animation", hasAnimations))
        {

            if (ImGui::MenuItem("Stop all", "Space", s->stopAnimations()))
                s->stopAnimations(!s->stopAnimations());

            ImGui::Separator();

            SLVstring animations = s->animManager().allAnimNames();
            if (curAnimIx == -1) curAnimIx = 0;
            SLAnimPlayback* anim = s->animManager().allAnimPlayback((SLuint)curAnimIx);

            ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.8f);
            if (myComboBox("", &curAnimIx, animations))
                anim = s->animManager().allAnimPlayback((SLuint)curAnimIx);
            ImGui::PopItemWidth();

            if (ImGui::MenuItem("Play forward", nullptr, anim->isPlayingForward()))
                anim->playForward();

            if (ImGui::MenuItem("Play backward", nullptr, anim->isPlayingBackward()))
                anim->playBackward();

            if (ImGui::MenuItem("Pause", nullptr, anim->isPaused()))
                anim->pause();

            if (ImGui::MenuItem("Stop", nullptr, anim->isStopped()))
                anim->enabled(false);

            if (ImGui::MenuItem("Skip to next keyfr.", nullptr, false))
                anim->skipToNextKeyframe();

            if (ImGui::MenuItem("Skip to prev. keyfr.", nullptr, false))
                anim->skipToPrevKeyframe();

            if (ImGui::MenuItem("Skip to start", nullptr, false))
                anim->skipToStart();

            if (ImGui::MenuItem("Skip to end", nullptr, false))
                anim->skipToEnd();

            ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.6f);

            SLfloat speed = anim->playbackRate();
            if (ImGui::SliderFloat("Speed", &speed, 0.f, 4.f))
                anim->playbackRate(speed);

            SLfloat lenSec       = anim->parentAnimation()->lengthSec();
            SLfloat localTimeSec = anim->localTime();
            if (ImGui::SliderFloat("Time", &localTimeSec, 0.f, lenSec))
                anim->localTime(localTimeSec);

            SLint       curEasing = (SLint)anim->easing();
            const char* easings[] = {"linear",
                                     "in quad",
                                     "out quad",
                                     "in out quad",
                                     "out in quad",
                                     "in cubic",
                                     "out cubic",
                                     "in out cubic",
                                     "out in cubic",
                                     "in quart",
                                     "out quart",
                                     "in out quart",
                                     "out in quart",
                                     "in quint",
                                     "out quint",
                                     "in out quint",
                                     "out in quint",
                                     "in sine",
                                     "out sine",
                                     "in out sine",
                                     "out in sine"};
            if (ImGui::Combo("Easing", &curEasing, easings, IM_ARRAYSIZE(easings)))
                anim->easing((SLEasingCurve)curEasing);

            ImGui::PopItemWidth();
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Infos"))
        {
            ImGui::MenuItem("Infos on Scene", nullptr, &showInfosScene);

            if (ImGui::BeginMenu("Statistics"))
            {
                ImGui::MenuItem("Stats on Timing", nullptr, &showStatsTiming);
                ImGui::MenuItem("Stats on Scene", nullptr, &showStatsScene);
                ImGui::MenuItem("Stats on Video", nullptr, &showStatsVideo);
#ifdef SL_BUILD_WAI
                if (AppDemo::sceneID == SID_VideoTrackWAI)
                    ImGui::MenuItem("Stats on WAI", nullptr, &showStatsWAI);
#endif
                ImGui::MenuItem("Stats on ImGui", nullptr, &showImGuiMetrics);
                ImGui::EndMenu();
            }

            ImGui::MenuItem("Scenegraph", nullptr, &showSceneGraph);
            ImGui::MenuItem("Properties", nullptr, &showProperties);
            ImGui::MenuItem("Transform", nullptr, &showTransform);
            if (AppDemo::devLoc.originLatLonAlt() != SLVec3d::ZERO ||
                AppDemo::devLoc.defaultLatLonAlt() != SLVec3d::ZERO)
                ImGui::MenuItem("Date-Time", nullptr, &showDateAndTime);
            ImGui::MenuItem("UI-Preferences", nullptr, &showUIPrefs);
            ImGui::Separator();
            ImGui::MenuItem("Infos on Device", nullptr, &showInfosDevice);
            ImGui::MenuItem("Infos on Sensors", nullptr, &showInfosSensors);
            if (AppDemo::sceneID >= SID_ErlebARBielBFH &&
                AppDemo::sceneID <= SID_ErlebAREvilardCheminDuRoc2)
            {
                ImGui::Separator();
                ImGui::MenuItem("ErlebAR Settings", nullptr, &showErlebAR);
            }
            ImGui::Separator();
            ImGui::MenuItem("Help on Interaction", nullptr, &showHelp);
            ImGui::MenuItem("Help on Calibration", nullptr, &showHelpCalibration);
            ImGui::Separator();
            ImGui::MenuItem("Credits", nullptr, &showCredits);
            ImGui::MenuItem("About SLProject", nullptr, &showAbout);

            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
}
//-----------------------------------------------------------------------------
//! Builds the edit menu that can be in the menu bar and the context menu
void AppDemoGui::buildMenuEdit(SLProjectScene* s, SLSceneView* sv)
{
    if (ImGui::MenuItem("Deselect Node", "ESC"))
        s->deselectAllNodesAndMeshes();

    ImGui::Separator();

    if (ImGui::MenuItem("Translate Node", nullptr, transformNode && transformNode->editMode() == NodeEditMode_Translate))
    {
        if (transformNode && transformNode->editMode() == NodeEditMode_Translate)
            removeTransformNode(s);
        else
            setTransformEditMode(s, sv, NodeEditMode_Translate);
    }
    if (ImGui::MenuItem("Rotate Node", nullptr, transformNode && transformNode->editMode() == NodeEditMode_Rotate))
    {
        if (transformNode && transformNode->editMode() == NodeEditMode_Rotate)
            removeTransformNode(s);
        else
            setTransformEditMode(s, sv, NodeEditMode_Rotate);
    }
    if (ImGui::MenuItem("Scale Node", nullptr, transformNode && transformNode->editMode() == NodeEditMode_Scale))
    {
        if (transformNode && transformNode->editMode() == NodeEditMode_Scale)
            removeTransformNode(s);
        else
            setTransformEditMode(s, sv, NodeEditMode_Scale);
    }

    ImGui::Separator();

    if (ImGui::BeginMenu("Node Flags"))
    {
        SLNode* selN = s->singleNodeSelected();

        if (ImGui::MenuItem("Wired Mesh", nullptr, selN->drawBits()->get(SL_DB_MESHWIRED)))
            selN->drawBits()->toggle(SL_DB_MESHWIRED);

        if (ImGui::MenuItem("With hard edges", nullptr, selN->drawBits()->get(SL_DB_WITHEDGES)))
            selN->drawBits()->toggle(SL_DB_WITHEDGES);

        if (ImGui::MenuItem("Only hard edges", nullptr, selN->drawBits()->get(SL_DB_ONLYEDGES)))
            selN->drawBits()->toggle(SL_DB_ONLYEDGES);

        if (ImGui::MenuItem("Normals", nullptr, selN->drawBits()->get(SL_DB_NORMALS)))
            selN->drawBits()->toggle(SL_DB_NORMALS);

        if (ImGui::MenuItem("Bounding Boxes", nullptr, selN->drawBits()->get(SL_DB_BBOX)))
            selN->drawBits()->toggle(SL_DB_BBOX);

        if (ImGui::MenuItem("Voxels", nullptr, selN->drawBits()->get(SL_DB_VOXELS)))
            selN->drawBits()->toggle(SL_DB_VOXELS);

        if (ImGui::MenuItem("Axis", nullptr, selN->drawBits()->get(SL_DB_AXIS)))
            selN->drawBits()->toggle(SL_DB_AXIS);

        if (ImGui::MenuItem("Back Faces", nullptr, selN->drawBits()->get(SL_DB_CULLOFF)))
            selN->drawBits()->toggle(SL_DB_CULLOFF);

        if (ImGui::MenuItem("Skeleton", nullptr, selN->drawBits()->get(SL_DB_SKELETON)))
            selN->drawBits()->toggle(SL_DB_SKELETON);

        if (ImGui::MenuItem("All off"))
            selN->drawBits()->allOff();

        ImGui::EndMenu();
    }
}
//-----------------------------------------------------------------------------
//! Builds context menu if right mouse click is over non-imgui area
void AppDemoGui::buildMenuContext(SLProjectScene* s, SLSceneView* sv)
{
    if (!ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) &&
        ImGui::IsMouseReleased(1))
    {
        ImGui::OpenPopup("Context Menu");
    }

    if (ImGui::BeginPopup("Context Menu"))
    {
        if (s->singleNodeSelected() != nullptr || !sv->camera()->selectRect().isZero())
        {
            if (s->singleNodeSelected())
            {
                buildMenuEdit(s, sv);
                ImGui::Separator();

                if (!showProperties)
                    if (ImGui::MenuItem("Show Properties"))
                        showProperties = true;
            }
        }

        if (AppDemoGui::hideUI)
            if (ImGui::MenuItem("Show user interface"))
                AppDemoGui::hideUI = false;

        if (!AppDemoGui::hideUI)
            if (ImGui::MenuItem("Hide user interface"))
                AppDemoGui::hideUI = true;

        if (s->root3D()->drawBits()->get(SL_DB_HIDDEN))
            if (ImGui::MenuItem("Show root node"))
                s->root3D()->drawBits()->toggle(SL_DB_HIDDEN);

        if (!s->root3D()->drawBits()->get(SL_DB_HIDDEN))
            if (ImGui::MenuItem("Hide root node"))
                s->root3D()->drawBits()->toggle(SL_DB_HIDDEN);

        if (ImGui::MenuItem("Capture Screen"))
            sv->screenCaptureIsRequested(true);

        ImGui::EndPopup();
    }
}
//-----------------------------------------------------------------------------
//! Builds the scenegraph dialog once per frame
void AppDemoGui::buildSceneGraph(SLScene* s)
{
    PROFILE_FUNCTION();

    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
    ImGui::Begin("Scenegraph", &showSceneGraph);

    if (s->root3D())
        addSceneGraphNode(s, s->root3D());

    if (s->root2D())
        addSceneGraphNode(s, s->root2D());

    ImGui::End();
    ImGui::PopFont();
}
//-----------------------------------------------------------------------------
//! Builds the node information once per frame
void AppDemoGui::addSceneGraphNode(SLScene* s, SLNode* node)
{
    PROFILE_FUNCTION();

    SLbool isSelectedNode = s->singleNodeSelected() == node;
    SLbool isLeafNode     = node->children().empty() && !node->mesh();

    ImGuiTreeNodeFlags nodeFlags = 0;
    if (isLeafNode)
        nodeFlags |= ImGuiTreeNodeFlags_Leaf;
    else
        nodeFlags |= ImGuiTreeNodeFlags_OpenOnArrow;

    if (isSelectedNode)
        nodeFlags |= ImGuiTreeNodeFlags_Selected;

    bool nodeIsOpen = ImGui::TreeNodeEx(node->name().c_str(), nodeFlags);

    if (ImGui::IsItemClicked())
    {
        s->deselectAllNodesAndMeshes();
        s->selectNodeMesh(node, nullptr);
    }

    if (nodeIsOpen)
    {
        if (node->mesh())
        {
            SLMesh* mesh = node->mesh();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));

            ImGuiTreeNodeFlags meshFlags = ImGuiTreeNodeFlags_Leaf;
            if (s->singleMeshFullSelected() == mesh)
                meshFlags |= ImGuiTreeNodeFlags_Selected;

            ImGui::TreeNodeEx(mesh, meshFlags, "%s", mesh->name().c_str());

            if (ImGui::IsItemClicked())
            {
                s->deselectAllNodesAndMeshes();
                s->selectNodeMesh(node, mesh);
            }

            ImGui::TreePop();
            ImGui::PopStyleColor();
        }

        for (auto* child : node->children())
            addSceneGraphNode(s, child);

        ImGui::TreePop();
    }
}
//-----------------------------------------------------------------------------
//! Builds the properties dialog once per frame
void AppDemoGui::buildProperties(SLScene* s, SLSceneView* sv)
{
    PROFILE_FUNCTION();

    SLNode* singleNode       = s->singleNodeSelected();
    SLMesh* singleFullMesh   = s->singleMeshFullSelected();
    bool    partialSelection = !s->selectedMeshes().empty() && !s->selectedMeshes()[0]->IS32.empty();

    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);

    if (sv->renderType() != RT_gl)
    {
        ImGui::Begin("Properties of Selection", &showProperties);
        ImGui::Text("Node selection and the");
        ImGui::Text("properties of it can only");
        ImGui::Text("be shown in the OpenGL");
        ImGui::Text("renderer.");
        ImGui::End();
    }
    else
    {
        // Only single node and no partial mesh selection
        if (singleNode && !partialSelection)
        {
            ImGui::Begin("Properties of Selection", &showProperties);

            if (ImGui::TreeNode("Single Node Properties"))
            {
                if (singleNode)
                {
                    SLuint c = (SLuint)singleNode->children().size();
                    SLuint m = singleNode->mesh() ? 1 : 0;
                    ImGui::Text("Node name  : %s", singleNode->name().c_str());
                    ImGui::Text("# children : %u", c);
                    ImGui::Text("# meshes   : %u", m);
                    if (ImGui::TreeNode("Drawing flags"))
                    {
                        SLbool db = singleNode->drawBit(SL_DB_HIDDEN);
                        if (ImGui::Checkbox("Hide", &db))
                            singleNode->drawBits()->set(SL_DB_HIDDEN, db);

                        db = singleNode->drawBit(SL_DB_NOTSELECTABLE);
                        if (ImGui::Checkbox("Not selectable", &db))
                            singleNode->drawBits()->set(SL_DB_NOTSELECTABLE, db);

                        db = singleNode->drawBit(SL_DB_MESHWIRED);
                        if (ImGui::Checkbox("Show wireframe", &db))
                            singleNode->drawBits()->set(SL_DB_MESHWIRED, db);

                        db = singleNode->drawBit(SL_DB_WITHEDGES);
                        if (ImGui::Checkbox("Show with hard edges", &db))
                            singleNode->drawBits()->set(SL_DB_WITHEDGES, db);

                        db = singleNode->drawBit(SL_DB_ONLYEDGES);
                        if (ImGui::Checkbox("Show only hard edges", &db))
                            singleNode->drawBits()->set(SL_DB_ONLYEDGES, db);

                        db = singleNode->drawBit(SL_DB_NORMALS);
                        if (ImGui::Checkbox("Show normals", &db))
                            singleNode->drawBits()->set(SL_DB_NORMALS, db);

                        db = singleNode->drawBit(SL_DB_VOXELS);
                        if (ImGui::Checkbox("Show voxels", &db))
                            singleNode->drawBits()->set(SL_DB_VOXELS, db);

                        db = singleNode->drawBit(SL_DB_BBOX);
                        if (ImGui::Checkbox("Show bounding boxes", &db))
                            singleNode->drawBits()->set(SL_DB_BBOX, db);

                        db = singleNode->drawBit(SL_DB_AXIS);
                        if (ImGui::Checkbox("Show axis", &db))
                            singleNode->drawBits()->set(SL_DB_AXIS, db);

                        db = singleNode->drawBit(SL_DB_CULLOFF);
                        if (ImGui::Checkbox("Show back faces", &db))
                            singleNode->drawBits()->set(SL_DB_CULLOFF, db);

                        ImGui::TreePop();
                    }

                    if (ImGui::TreeNode("Local transform"))
                    {
                        SLMat4f om(singleNode->om());
                        SLVec3f trn, rot, scl;
                        om.decompose(trn, rot, scl);
                        rot *= Utils::RAD2DEG;

                        ImGui::Text("Translation  : %s", trn.toString().c_str());
                        ImGui::Text("Rotation     : %s", rot.toString().c_str());
                        ImGui::Text("Scaling      : %s", scl.toString().c_str());
                        ImGui::TreePop();
                    }

                    // Properties related to shadow mapping
                    if (ImGui::TreeNode("Shadow mapping"))
                    {
                        SLbool castsShadows = singleNode->castsShadows();
                        if (ImGui::Checkbox("Casts shadows", &castsShadows))
                            singleNode->castsShadows(castsShadows);

                        if (auto* light = dynamic_cast<SLLight*>(singleNode))
                        {
                            SLbool createsShadows = light->createsShadows();
                            if (ImGui::Checkbox("Creates shadows", &createsShadows))
                                light->createsShadows(createsShadows);

                            if (createsShadows)
                            {
                                SLShadowMap* shadowMap = light->shadowMap();

                                if (shadowMap != nullptr)
                                {
                                    if (shadowMap->projection() == P_monoPerspective &&
                                        light->spotCutOffDEG() < 90.0f)
                                    {
                                        SLbool useCubemap = shadowMap->useCubemap();
                                        if (ImGui::Checkbox("Uses Cubemap", &useCubemap))
                                            shadowMap->useCubemap(useCubemap);
                                    }

                                    SLfloat clipNear = shadowMap->clipNear();
                                    SLfloat clipFar  = shadowMap->clipFar();

                                    if (ImGui::SliderFloat("Near clipping plane", &clipNear, 0.01f, clipFar))
                                        shadowMap->clipNear(clipNear);

                                    if (ImGui::SliderFloat("Far clipping plane", &clipFar, clipNear, 200.0f))
                                        shadowMap->clipFar(clipFar);

                                    SLVec2i texSize = shadowMap->textureSize();
                                    if (ImGui::SliderInt2("Texture resolution", (int*)&texSize, 32, 4096))
                                        shadowMap->textureSize(
                                          SLVec2i((int)Utils::closestPowerOf2((unsigned)texSize.x),
                                                  (int)Utils::closestPowerOf2((unsigned)texSize.y)));

                                    SLfloat shadowMinBias = light->shadowMinBias();
                                    SLfloat shadowMaxBias = light->shadowMaxBias();
                                    if (ImGui::SliderFloat("Min. shadow bias", &shadowMinBias, 0.0f, shadowMaxBias, "%.03f"))
                                        light->shadowMinBias(shadowMinBias);
                                    if (ImGui::SliderFloat("Max. shadow bias", &shadowMaxBias, shadowMinBias, 0.02f, "%.03f"))
                                        light->shadowMaxBias(shadowMaxBias);

                                    if (typeid(*singleNode) == typeid(SLLightDirect))
                                    {
                                        SLVec2f size = shadowMap->size();
                                        if (ImGui::InputFloat2("Size", (float*)&size))
                                            shadowMap->size(size);
                                    }

                                    if (!shadowMap->useCubemap())
                                    {
                                        SLbool doesSmoothShadows = light->doSoftShadows();
                                        if (ImGui::Checkbox("Smooth shadows enabled", &doesSmoothShadows))
                                            light->doSmoothShadows(doesSmoothShadows);

                                        SLuint pcfLevel = light->softShadowLevel();
                                        if (ImGui::SliderInt("Smoothing level", (SLint*)&pcfLevel, 1, 3))
                                            light->smoothShadowLevel(pcfLevel);
                                    }

#ifndef SL_GLES
                                    SLVec2i rayCount = shadowMap->rayCount();
                                    if (ImGui::InputInt2("Visualization rays", (int*)&rayCount))
                                        shadowMap->rayCount(rayCount);
#endif

                                    if (ImGui::TreeNode(shadowMap->useCubemap()
                                                          ? "Light space matrices"
                                                          : "Light space matrix"))
                                    {
                                        if (shadowMap->useCubemap())
                                            for (SLint i = 0; i < 6; ++i)
                                                ImGui::Text("Matrix %i:\n%s", i + 1, shadowMap->mvp()[i].toString().c_str());
                                        else
                                            ImGui::Text(shadowMap->mvp()[0].toString().c_str());

                                        ImGui::TreePop();
                                    }

                                    if (!shadowMap->useCubemap())
                                    {
                                        ImGui::Text("Depth Buffer:");
                                        ImGui::Image((void*)(intptr_t)shadowMap->depthBuffer()->texID(),
                                                     ImVec2(200, 200));
                                    }
                                }
                            }
                        }

                        ImGui::TreePop();
                    }

                    // Show special camera properties
                    if (typeid(*singleNode) == typeid(SLCamera))
                    {
                        auto* cam = (SLCamera*)singleNode;

                        if (ImGui::TreeNode("Camera"))
                        {
                            SLfloat clipN     = cam->clipNear();
                            SLfloat clipF     = cam->clipFar();
                            SLfloat focalDist = cam->focalDist();
                            SLfloat fov       = cam->fovV();

                            const char* projections[] = {"Mono Perspective",
                                                         "Mono Intrinsic Calibrated",
                                                         "Mono Orthographic",
                                                         "Stereo Side By Side",
                                                         "Stereo Side By Side Prop.",
                                                         "Stereo Side By Side Dist.",
                                                         "Stereo Line By Line",
                                                         "Stereo Column By Column",
                                                         "Stereo Pixel By Pixel",
                                                         "Stereo Color Red-Cyan",
                                                         "Stereo Color Red-Green",
                                                         "Stereo Color Red-Blue",
                                                         "Stereo Color Yellow-Blue",
                                                         "Stereo OpenVR"};

                            int proj = cam->projection();
                            if (ImGui::Combo("Projection", &proj, projections, IM_ARRAYSIZE(projections)))
                                cam->projection((SLProjection)proj);

                            if (cam->projection() > P_monoOrthographic)
                            {
                                SLfloat eyeSepar = cam->stereoEyeSeparation();
                                if (ImGui::SliderFloat("Eye Sep.", &eyeSepar, 0.0f, focalDist / 10.f))
                                    cam->stereoEyeSeparation(eyeSepar);
                            }

                            if (ImGui::SliderFloat("FOV", &fov, 1.f, 179.f))
                                cam->fov(fov);

                            if (ImGui::SliderFloat("Near Clip", &clipN, 0.001f, 10.f))
                                cam->clipNear(clipN);

                            if (ImGui::SliderFloat("Far Clip", &clipF, clipN, std::min(clipF * 1.1f, 1000000.f)))
                                cam->clipFar(clipF);

                            if (ImGui::SliderFloat("Focal Dist.", &focalDist, clipN, clipF))
                                cam->focalDist(focalDist);

                            ImGui::TreePop();
                        }
                    }

                    // Show special light properties
                    if (typeid(*singleNode) == typeid(SLLightSpot) ||
                        typeid(*singleNode) == typeid(SLLightRect) ||
                        typeid(*singleNode) == typeid(SLLightDirect))
                    {
                        SLLight* light = nullptr;
                        SLstring typeName;
                        SLbool   doSunPowerAdaptation = false;
                        if (typeid(*singleNode) == typeid(SLLightSpot))
                        {
                            light    = (SLLight*)(SLLightSpot*)singleNode;
                            typeName = "Light (spot):";
                        }
                        if (typeid(*singleNode) == typeid(SLLightRect))
                        {
                            light    = (SLLight*)(SLLightRect*)singleNode;
                            typeName = "Light (rectangular):";
                        }
                        if (typeid(*singleNode) == typeid(SLLightDirect))
                        {
                            light                = (SLLight*)(SLLightDirect*)singleNode;
                            typeName             = "Light (directional):";
                            doSunPowerAdaptation = ((SLLightDirect*)singleNode)->doSunPowerAdaptation();
                        }

                        if (light && ImGui::TreeNode(typeName.c_str()))
                        {
                            SLbool on = light->isOn();
                            if (ImGui::Checkbox("Is on", &on))
                                light->isOn(on);

                            ImGuiColorEditFlags cef = ImGuiColorEditFlags_NoInputs;
                            SLCol4f             aC  = light->ambientColor();
                            if (ImGui::ColorEdit3("Ambient color", (float*)&aC, cef))
                                light->ambientColor(aC);

                            float aP = light->ambientPower();
                            float dP = light->diffusePower();
                            if (doSunPowerAdaptation)
                            {
                                float sum_aPdP     = aP + dP;
                                float ambiFraction = aP / sum_aPdP;
                                if (ImGui::SliderFloat("Diffuse-Ambient-Mix", &ambiFraction, 0.0f, 1.0f, "%.2f"))
                                {
                                    light->ambientPower(ambiFraction * sum_aPdP);
                                    light->diffusePower((1.0f - ambiFraction) * sum_aPdP);
                                }
                            }
                            else
                            {
                                SLCol4f dC = light->diffuseColor();
                                if (ImGui::ColorEdit3("Diffuse color", (float*)&dC, cef))
                                    light->diffuseColor(dC);

                                SLCol4f sC = light->specularColor();
                                if (ImGui::ColorEdit3("Specular color", (float*)&sC, cef))
                                    light->specularColor(sC);
                            }

                            if (ImGui::SliderFloat("Ambient power", &aP, 0.0f, 10.0f, "%.2f"))
                                light->ambientPower(aP);

                            if (ImGui::SliderFloat("Diffuse power", &dP, 0.0f, 10.0f, "%.2f"))
                                light->diffusePower(dP);

                            float sP = light->specularPower();
                            if (ImGui::SliderFloat("Specular power", &sP, 0.0f, 10.0f, "%.2f"))
                                light->specularPower(sP);

                            float cutoff = light->spotCutOffDEG();
                            if (ImGui::SliderFloat("Spot cut off angle", &cutoff, 0.0f, 180.0f, "%.2f"))
                                light->spotCutOffDEG(cutoff);

                            float spotExp = light->spotExponent();
                            if (ImGui::SliderFloat("Spot attenuation", &spotExp, 0.0f, 128.0f, "%.2f"))
                                light->spotExponent(spotExp);

                            float kc = light->kc();
                            if (ImGui::SliderFloat("Constant attenuation", &kc, 0.0f, 1.0f, "%.2f"))
                                light->kc(kc);

                            float kl = light->kl();
                            if (ImGui::SliderFloat("Linear attenuation", &kl, 0.0f, 1.0f, "%.2f"))
                                light->kl(kl);

                            float kq = light->kq();
                            if (ImGui::SliderFloat("Quadratic attenuation", &kq, 0.0f, 1.0f, "%.2f"))
                                light->kq(kq);

                            if (typeid(*singleNode) == typeid(SLLightDirect))
                            {
                                SLLightDirect* dirLight = (SLLightDirect*)singleNode;
                                if (ImGui::Checkbox("Do Sun Power Adaptation", &doSunPowerAdaptation))
                                    dirLight->doSunPowerAdaptation(doSunPowerAdaptation);

                                if (doSunPowerAdaptation)
                                {
                                    SLTexColorLUT* lut = dirLight->sunLightColorLUT();
                                    if (ImGui::TreeNode("Sun Color LUT"))
                                    {
                                        showLUTColors(lut);
                                        ImGui::TreePop();
                                    }

                                    lut->bindActive(); // This texture is not an scenegraph texture
                                    SLfloat texW =
                                      ImGui::GetWindowWidth() - 4 * ImGui::GetTreeNodeToLabelSpacing() - 10;
                                    void* tid = (ImTextureID)(uintptr_t)lut->texID();
                                    ImGui::Image(tid,
                                                 ImVec2(texW, texW * 0.15f),
                                                 ImVec2(0, 1),
                                                 ImVec2(1, 0),
                                                 ImVec4(1, 1, 1, 1),
                                                 ImVec4(1, 1, 1, 1));
                                }
                            }

                            ImGui::TreePop();
                        }
                    }
                }
                else
                {
                    ImGui::Text("No single node selected.");
                }
                ImGui::TreePop();
            }

            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
            ImGui::Separator();

            if (singleFullMesh)
            {
                // See also SLScene::selectNodeMesh
                if (ImGui::TreeNode("Single Mesh Properties"))
                {
                    SLuint      v = (SLuint)singleFullMesh->P.size();
                    SLuint      t = (SLuint)(!singleFullMesh->I16.empty() ? singleFullMesh->I16.size() / 3 : singleFullMesh->I32.size() / 3);
                    SLuint      e = (SLuint)(!singleFullMesh->IE16.empty() ? singleFullMesh->IE16.size() / 2 : singleFullMesh->IE32.size() / 2);
                    SLMaterial* m = singleFullMesh->mat();
                    ImGui::Text("Mesh name    : %s", singleFullMesh->name().c_str());
                    ImGui::Text("# vertices   : %u", v);
                    ImGui::Text("# triangles  : %u", t);
                    ImGui::Text("# hard edges : %u", e);
                    ImGui::Text("Material Name: %s", m->name().c_str());

                    if (m->lightModel() == LM_BlinnPhong)
                    {
                        if (ImGui::TreeNode("Light Model: Blinn-Phong"))
                        {
                            ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.5f);

                            ImGuiColorEditFlags cef = ImGuiColorEditFlags_NoInputs;
                            SLCol4f             ac  = m->ambient();
                            if (ImGui::ColorEdit3("Ambient color", (float*)&ac, cef))
                                m->ambient(ac);

                            SLCol4f dc = m->diffuse();
                            if (ImGui::ColorEdit3("Diffuse color", (float*)&dc, cef))
                                m->diffuse(dc);

                            SLCol4f sc = m->specular();
                            if (ImGui::ColorEdit3("Specular color", (float*)&sc, cef))
                                m->specular(sc);

                            SLCol4f ec = m->emissive();
                            if (ImGui::ColorEdit3("Emissive color", (float*)&ec, cef))
                                m->emissive(ec);

                            SLfloat shine = m->shininess();
                            if (ImGui::SliderFloat("Shininess", &shine, 0.0f, 1000.0f))
                                m->shininess(shine);

                            SLfloat kr = m->kr();
                            if (ImGui::SliderFloat("kr", &kr, 0.0f, 1.0f))
                                m->kr(kr);

                            SLfloat kt = m->kt();
                            if (ImGui::SliderFloat("kt", &kt, 0.0f, 1.0f))
                                m->kt(kt);

                            SLfloat kn = m->kn();
                            if (ImGui::SliderFloat("kn", &kn, 1.0f, 2.5f))
                                m->kn(kn);

                            SLbool receivesShadows = m->getsShadows();
                            if (ImGui::Checkbox("Receives shadows", &receivesShadows))
                                m->getsShadows(receivesShadows);

                            ImGui::PopItemWidth();
                            ImGui::TreePop();
                        }
                    }
                    else if (m->lightModel() == LM_CookTorrance)
                    {
                        if (ImGui::TreeNode("Light Model: Cook-Torrance"))
                        {
                            ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.5f);

                            ImGuiColorEditFlags cef = ImGuiColorEditFlags_NoInputs;
                            SLCol4f             dc  = m->diffuse();
                            if (ImGui::ColorEdit3("Diffuse color", (float*)&dc, cef))
                                m->diffuse(dc);

                            SLfloat rough = m->roughness();
                            if (ImGui::SliderFloat("Roughness", &rough, 0.0f, 1.0f))
                                m->roughness(rough);

                            SLfloat metal = m->metalness();
                            if (ImGui::SliderFloat("Metalness", &metal, 0.0f, 1.0f))
                                m->metalness(metal);

                            ImGui::PopItemWidth();
                            ImGui::TreePop();
                        }
                    }
                    else
                    {
                    }

                    if (!m->textures().empty() &&
                        ImGui::TreeNode("Tex", "Textures (%lu)", m->textures().size()))
                    {
                        // SLfloat lineH = ImGui::GetTextLineHeightWithSpacing();
                        SLfloat texW = ImGui::GetWindowWidth() - 4 * ImGui::GetTreeNodeToLabelSpacing() - 10;

                        for (auto& i : m->textures())
                        {
                            SLGLTexture* tex    = i;
                            void*        tid    = (ImTextureID)(intptr_t)tex->texID();
                            SLfloat      w      = (SLfloat)tex->width();
                            SLfloat      h      = (SLfloat)tex->height();
                            SLfloat      h_to_w = h / w;

                            if (ImGui::TreeNode(tex->name().c_str()))
                            {
                                float mbCPU = 0.0f;
                                for (auto img : tex->images())
                                    mbCPU += (float)img->bytesPerImage();
                                float mbGPU = (float)tex->bytesOnGPU();
                                float mbDSK = (float)tex->bytesInFile();

                                mbDSK /= 1E6f;
                                mbCPU /= 1E6f;
                                mbGPU /= 1E6f;

                                ImGui::Text("Size(PX): %dx%dx%d", tex->width(), tex->height(), tex->depth());
                                ImGui::Text("Size(MB): GPU:%4.2f, CPU:%4.2f, DSK:%4.2f", mbGPU, mbCPU, mbDSK);
                                ImGui::Text("TexID   : %u (%s)", tex->texID(), tex->isTexture() ? "ok" : "not ok");
                                ImGui::Text("Type    : %s", tex->typeName().c_str());
#ifdef SL_BUILD_WITH_KTX
                                ImGui::Text("Compr.  : %s", tex->compressionFormatStr(tex->compressionFormat()).c_str());
#endif
                                ImGui::Text("Min.Flt : %s", tex->minificationFilterName().c_str());
                                ImGui::Text("Mag.Flt : %s", tex->magnificationFilterName().c_str());

                                if (tex->target() == GL_TEXTURE_2D)
                                {
                                    if (typeid(*tex) == typeid(SLTexColorLUT))
                                    {
                                        SLTexColorLUT* lut = (SLTexColorLUT*)i;
                                        if (ImGui::TreeNode("Color Points in Transfer Function"))
                                        {
                                            showLUTColors(lut);
                                            ImGui::TreePop();
                                        }

                                        if (ImGui::TreeNode("Alpha Points in Transfer Function"))
                                        {
                                            for (SLulong a = 0; a < lut->alphas().size(); ++a)
                                            {
                                                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.25f);
                                                SLfloat alpha = lut->alphas()[a].alpha;
                                                SLchar  label[20];
                                                sprintf(label, "Alpha %lu", a);
                                                if (ImGui::SliderFloat(label, &alpha, 0.0f, 1.0f, "%3.2f"))
                                                {
                                                    lut->alphas()[a].alpha = alpha;
                                                    lut->generateTexture();
                                                }
                                                ImGui::SameLine();
                                                sprintf(label, "Pos. %lu", a);
                                                SLfloat pos = lut->alphas()[a].pos;
                                                if (a > 0 && a < lut->alphas().size() - 1)
                                                {
                                                    SLfloat min = lut->alphas()[a - 1].pos +
                                                                  2.0f / (SLfloat)lut->length();
                                                    SLfloat max = lut->alphas()[a + 1].pos -
                                                                  2.0f / (SLfloat)lut->length();
                                                    if (ImGui::SliderFloat(label, &pos, min, max, "%3.2f"))
                                                    {
                                                        lut->alphas()[a].pos = pos;
                                                        lut->generateTexture();
                                                    }
                                                }
                                                else
                                                    ImGui::Text("%3.2f Pos. %lu", pos, a);

                                                ImGui::PopItemWidth();
                                            }

                                            ImGui::TreePop();
                                        }

                                        ImGui::Image(tid,
                                                     ImVec2(texW, texW * 0.15f),
                                                     ImVec2(0, 1),
                                                     ImVec2(1, 0),
                                                     ImVec4(1, 1, 1, 1),
                                                     ImVec4(1, 1, 1, 1));

                                        SLVfloat allAlpha = lut->allAlphas();
                                        ImGui::PlotLines("",
                                                         allAlpha.data(),
                                                         (SLint)allAlpha.size(),
                                                         0,
                                                         nullptr,
                                                         0.0f,
                                                         1.0f,
                                                         ImVec2(texW, texW * 0.25f));
                                    }
                                    else
                                    {
                                        ImGui::Image(tid,
                                                     ImVec2(texW, texW * h_to_w),
                                                     ImVec2(0, 1),
                                                     ImVec2(1, 0),
                                                     ImVec4(1, 1, 1, 1),
                                                     ImVec4(1, 1, 1, 1));
                                    }
                                }
                                else
                                {
                                    if (tex->target() == GL_TEXTURE_CUBE_MAP)
                                        ImGui::Text("Cube maps can not be displayed.");
                                    else if (tex->target() == GL_TEXTURE_3D)
                                        ImGui::Text("3D textures can not be displayed.");
                                }

                                ImGui::TreePop();
                            }
                        }

                        ImGui::TreePop();
                    }

                    for (auto* shd : m->program()->shaders())
                    {
                        SLfloat lineH = ImGui::GetTextLineHeight();

                        if (ImGui::TreeNode(shd->name().c_str()))
                        {
                            SLchar* text = new char[shd->code().length() + 1];
                            strcpy(text, shd->code().c_str());
                            ImGui::InputTextMultiline(shd->name().c_str(),
                                                      text,
                                                      shd->code().length() + 1,
                                                      ImVec2(-1.0f, -1.0f));
                            ImGui::TreePop();
                            delete[] text;
                        }
                    }

                    ImGui::TreePop();
                }
            }
            else
            {
                ImGui::Text("No single single mesh selected.");
            }

            ImGui::PopStyleColor();
            ImGui::End();
        }
        else if (!singleFullMesh && !s->selectedMeshes().empty())
        {
            // See also SLMesh::handleRectangleSelection
            ImGui::Begin("Properties of Selection", &showProperties);

            for (auto* selectedNode : s->selectedNodes())
            {
                if (selectedNode->mesh())
                {
                    ImGui::Text("Node: %s", selectedNode->name().c_str());
                    SLMesh* selectedMesh = selectedNode->mesh();

                    if (!selectedMesh->IS32.empty())
                    {
                        ImGui::Text("   Mesh: %s {%u v.}",
                                    selectedMesh->name().c_str(),
                                    (SLuint)selectedMesh->IS32.size());
                        ImGui::SameLine();
                        SLstring delBtn = "DEL##" + selectedMesh->name();
                        if (ImGui::Button(delBtn.c_str()))
                        {
                            selectedMesh->deleteSelected(selectedNode);
                        }
                    }
                }
            }

            ImGui::End();
        }
        else
        {
            // Nothing is selected
            ImGui::Begin("Properties of Selection", &showProperties);
            ImGui::Text("There is nothing selected.");
            ImGui::Text("");
            ImGui::Text("Select a single node by");
            ImGui::Text("double-clicking it or");
            ImGui::Text("select multiple nodes by");
            ImGui::Text("SHIFT-double-clicking them.");
            ImGui::Text("");
            ImGui::Text("Select partial meshes by");
            ImGui::Text("CTRL-LMB rectangle drawing.");
            ImGui::Text("");
            ImGui::Text("Press ESC to deselect all.");
            ImGui::Text("");
            ImGui::Text("Be aware that a node may be");
            ImGui::Text("flagged as not selectable.");
            ImGui::End();
        }
    }

    ImGui::PopFont();
}
//-----------------------------------------------------------------------------
//! Loads the UI configuration
void AppDemoGui::loadConfig(SLint dotsPerInch)
{
    ImGuiStyle& style               = ImGui::GetStyle();
    SLstring    fullPathAndFilename = AppDemo::configPath +
                                   AppDemo::name + ".yml";

    if (!Utils::fileExists(fullPathAndFilename))
    {
        SL_LOG("No config file %s: ", fullPathAndFilename.c_str());

        // Scale for proportional and fixed size fonts
        SLfloat dpiScaleProp  = dotsPerInch / 120.0f;
        SLfloat dpiScaleFixed = dotsPerInch / 142.0f;

        // Default settings for the first time
        SLGLImGui::fontPropDots  = std::max(16.0f * dpiScaleProp, 16.0f);
        SLGLImGui::fontFixedDots = std::max(13.0f * dpiScaleFixed, 13.0f);

        // Store dialog show states
        AppDemoGui::showAbout        = true;
        AppDemoGui::showInfosScene   = true;
        AppDemoGui::showStatsTiming  = false;
        AppDemoGui::showStatsScene   = false;
        AppDemoGui::showStatsVideo   = false;
        AppDemoGui::showInfosDevice  = false;
        AppDemoGui::showInfosSensors = false;
        AppDemoGui::showSceneGraph   = false;
        AppDemoGui::showProperties   = false;
        AppDemoGui::showDateAndTime  = false;
        AppDemoGui::showDockSpace    = true;

        // Adjust UI padding on DPI
        style.WindowPadding.x = style.FramePadding.x = style.ItemInnerSpacing.x = std::max(8.0f * dpiScaleFixed, 8.0f);
        style.FramePadding.y = style.ItemInnerSpacing.y = std::max(4.0f * dpiScaleFixed, 4.0f);
        style.WindowPadding.y                           = style.ItemSpacing.y * 3;
        style.ScrollbarSize                             = std::max(16.0f * dpiScaleFixed, 16.0f);

        // HSM4: Bugfix in some unknown cases ScrollbarSize gets INT::MIN
        if (style.ScrollbarSize < 0.0f)
            style.ScrollbarSize = 16.0f;

        style.ScrollbarRounding = std::floor(style.ScrollbarSize / 2);

        return;
    }

    CVFileStorage fs;
    try
    {
        fs.open(fullPathAndFilename, CVFileStorage::READ);
        if (fs.isOpened())
        {
            // clang-format off
            SLint i = 0;
            SLbool b = false;
            fs["configTime"] >> AppDemoGui::configTime;
            fs["fontPropDots"] >> i;        SLGLImGui::fontPropDots = (SLfloat) i;
            fs["fontFixedDots"] >> i;       SLGLImGui::fontFixedDots = (SLfloat) i;
            fs["ItemSpacingX"] >> i;        style.ItemSpacing.x = (SLfloat) i;
            fs["ItemSpacingY"] >> i;        style.ItemSpacing.y = (SLfloat) i;
                                            style.WindowPadding.x = style.FramePadding.x = style.ItemInnerSpacing.x = style.ItemSpacing.x;
                                            style.FramePadding.y = style.ItemInnerSpacing.y = style.ItemSpacing.y;
                                            style.WindowPadding.y = style.ItemSpacing.y * 3;
            fs["ScrollbarSize"] >> i;       style.ScrollbarSize = (SLfloat) i;
            // HSM4: Bugfix in some unknown cases ScrollbarSize gets INT::MIN
            if (style.ScrollbarSize < 0.0f)
                style.ScrollbarSize = 16.0f;

            fs["ScrollbarRounding"] >> i;   style.ScrollbarRounding = (SLfloat) i;
            fs["sceneID"] >> i;             AppDemo::sceneID = (SLSceneID) i;
            fs["showInfosScene"] >> b;      AppDemoGui::showInfosScene = b;
            fs["showStatsTiming"] >> b;     AppDemoGui::showStatsTiming = b;
            fs["showStatsMemory"] >> b;     AppDemoGui::showStatsScene = b;
            fs["showStatsVideo"] >> b;      AppDemoGui::showStatsVideo = b;
            fs["showStatsWAI"] >> b;        AppDemoGui::showStatsWAI = b;
            fs["showInfosFrameworks"] >> b; AppDemoGui::showInfosDevice = b;
            fs["showInfosSensors"] >> b;    AppDemoGui::showInfosSensors = b;
            fs["showSceneGraph"] >> b;      AppDemoGui::showSceneGraph = b;
            fs["showProperties"] >> b;      AppDemoGui::showProperties = b;
            fs["showErlebAR"] >> b;         AppDemoGui::showErlebAR = b;
            fs["showTransform"] >> b;       AppDemoGui::showTransform = b;
            fs["showUIPrefs"] >> b;         AppDemoGui::showUIPrefs = b;
            fs["showDateAndTime"] >> b;     AppDemoGui::showDateAndTime = b;
            fs["showDockSpace"] >> b;       AppDemoGui::showDockSpace = b;
            // clang-format on

            fs.release();
            SL_LOG("Config. loaded   : %s", fullPathAndFilename.c_str());
            SL_LOG("Config. date     : %s", AppDemoGui::configTime.c_str());
            SL_LOG("fontPropDots     : %f", SLGLImGui::fontPropDots);
            SL_LOG("fontFixedDots    : %f", SLGLImGui::fontFixedDots);
        }
        else
        {
            SL_LOG("****** Failed to open file for reading: %s", fullPathAndFilename.c_str());
        }
    }
    catch (...)
    {
        SL_LOG("****** Parsing of file failed: %s", fullPathAndFilename.c_str());
    }

    // check font sizes for HDPI displays
    if (dotsPerInch > 300)
    {
        if (SLGLImGui::fontPropDots < 16.1f &&
            SLGLImGui::fontFixedDots < 13.1)
        {
            // Scale for proportional and fixed size fonts
            SLfloat dpiScaleProp  = dotsPerInch / 120.0f;
            SLfloat dpiScaleFixed = dotsPerInch / 142.0f;

            // Default settings for the first time
            SLGLImGui::fontPropDots  = std::max(16.0f * dpiScaleProp, 16.0f);
            SLGLImGui::fontFixedDots = std::max(13.0f * dpiScaleFixed, 13.0f);
        }
    }
}
//-----------------------------------------------------------------------------
//! Stores the UI configuration
void AppDemoGui::saveConfig()
{
    ImGuiStyle& style               = ImGui::GetStyle();
    SLstring    fullPathAndFilename = AppDemo::configPath +
                                   AppDemo::name + ".yml";

    if (!Utils::fileExists(fullPathAndFilename))
        SL_LOG("New config file will be written: %s",
               fullPathAndFilename.c_str());

    CVFileStorage fs(fullPathAndFilename, CVFileStorage::WRITE);

    if (!fs.isOpened())
    {
        SL_LOG("Failed to open file for writing: %s",
               fullPathAndFilename.c_str());
        SL_EXIT_MSG("Exit in AppDemoGui::saveConfig");
    }

    fs << "configTime" << Utils::getLocalTimeString();
    fs << "fontPropDots" << (SLint)SLGLImGui::fontPropDots;
    fs << "fontFixedDots" << (SLint)SLGLImGui::fontFixedDots;
    if (AppDemo::sceneID == SID_VolumeRayCastLighted ||
        AppDemo::sceneID == SID_VolumeRayCast)
        fs << "sceneID" << (SLint)SID_Minimal;
    else
        fs << "sceneID" << (SLint)AppDemo::sceneID;
    fs << "ItemSpacingX" << (SLint)style.ItemSpacing.x;
    fs << "ItemSpacingY" << (SLint)style.ItemSpacing.y;
    fs << "ScrollbarSize" << (SLfloat)style.ScrollbarSize;
    fs << "ScrollbarRounding" << (SLfloat)style.ScrollbarRounding;
    fs << "showStatsTiming" << AppDemoGui::showStatsTiming;
    fs << "showStatsMemory" << AppDemoGui::showStatsScene;
    fs << "showStatsVideo" << AppDemoGui::showStatsVideo;
    fs << "showStatsWAI" << AppDemoGui::showStatsWAI;
    fs << "showInfosFrameworks" << AppDemoGui::showInfosDevice;
    fs << "showInfosScene" << AppDemoGui::showInfosScene;
    fs << "showInfosSensors" << AppDemoGui::showInfosSensors;
    fs << "showSceneGraph" << AppDemoGui::showSceneGraph;
    fs << "showProperties" << AppDemoGui::showProperties;
    fs << "showErlebAR" << AppDemoGui::showErlebAR;
    fs << "showTransform" << AppDemoGui::showTransform;
    fs << "showUIPrefs" << AppDemoGui::showUIPrefs;
    fs << "showDateAndTime" << AppDemoGui::showDateAndTime;
    fs << "showDockSpace" << AppDemoGui::showDockSpace;

    fs.release();
    SL_LOG("Config. saved   : %s", fullPathAndFilename.c_str());
}
//-----------------------------------------------------------------------------
//! Adds a transform node for the selected node and toggles the edit mode
void AppDemoGui::setTransformEditMode(SLProjectScene* s,
                                      SLSceneView*    sv,
                                      SLNodeEditMode  editMode)
{
    SLTransformNode* tN = s->root3D()->findChild<SLTransformNode>("Edit Gizmos");

    if (!tN)
    {
        tN = new SLTransformNode(sv, s->singleNodeSelected(), AppDemo::shaderPath);
        s->root3D()->addChild(tN);
    }

    tN->editMode(editMode);
    transformNode = tN;
}
//-----------------------------------------------------------------------------
//! Searches and removes the transform node
void AppDemoGui::removeTransformNode(SLProjectScene* s)
{
    SLTransformNode* tN = s->root3D()->findChild<SLTransformNode>("Edit Gizmos");
    if (tN)
    {
        auto it = find(s->eventHandlers().begin(), s->eventHandlers().end(), tN);
        if (it != s->eventHandlers().end())
            s->eventHandlers().erase(it);

        s->root3D()->deleteChild(tN);
    }
    transformNode = nullptr;
}
//-----------------------------------------------------------------------------
//! Enables calculation and visualization of horizon line (using rotation sensors)
void AppDemoGui::showHorizon(SLProjectScene* s, SLSceneView* sv)
{
    // todo: why is root2D not always valid?
    if (!s->root2D())
    {
        SLNode* scene2D = new SLNode("root2D");
        s->root2D(scene2D);
    }

    SLstring       horizonName = "Horizon";
    SLHorizonNode* horizonNode = s->root2D()->findChild<SLHorizonNode>(horizonName);

    if (!horizonNode)
    {
        horizonNode = new SLHorizonNode(horizonName,
                                        &AppDemo::devRot,
                                        s->font16,
                                        AppDemo::shaderPath,
                                        sv->scrW(),
                                        sv->scrH());
        s->root2D()->addChild(horizonNode);
        _horizonVisuEnabled = true;
    }
}
//-----------------------------------------------------------------------------
//! Disables calculation and visualization of horizon line
void AppDemoGui::hideHorizon(SLProjectScene* s)
{
    if (s->root2D())
    {
        SLstring       horizonName = "Horizon";
        SLHorizonNode* horizonNode = s->root2D()->findChild<SLHorizonNode>(horizonName);
        if (horizonNode)
        {
            s->root2D()->deleteChild(horizonNode);
        }
    }
    _horizonVisuEnabled = false;
}
//-----------------------------------------------------------------------------
//! Displays a editable color lookup table wit ImGui widgets
void AppDemoGui::showLUTColors(SLTexColorLUT* lut)
{
    ImGuiColorEditFlags cef = ImGuiColorEditFlags_NoInputs;
    for (SLulong c = 0; c < lut->colors().size(); ++c)
    {
        SLCol3f color = lut->colors()[c].color;
        SLchar  label[20];
        sprintf(label, "Color %lu", c);
        if (ImGui::ColorEdit3(label, (float*)&color, cef))
        {
            lut->colors()[c].color = color;
            lut->generateTexture();
        }
        ImGui::SameLine();
        ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.5f);
        sprintf(label, "Pos. %lu", c);
        SLfloat pos = lut->colors()[c].pos;
        if (c > 0 && c < lut->colors().size() - 1)
        {
            SLfloat min = lut->colors()[c - 1].pos + 2.0f / (SLfloat)lut->length();
            SLfloat max = lut->colors()[c + 1].pos - 2.0f / (SLfloat)lut->length();
            if (ImGui::SliderFloat(label, &pos, min, max, "%3.2f"))
            {
                lut->colors()[c].pos = pos;
                lut->generateTexture();
            }
        }
        else
            ImGui::Text("%3.2f Pos. %lu", pos, c);
        ImGui::PopItemWidth();
    }
}
//-----------------------------------------------------------------------------
//! Parallel HTTP download, unzip and load scene job scheduling
void AppDemoGui::downloadModelAndLoadScene(SLScene*     s,
                                           SLSceneView* sv,
                                           string       downloadFilename,
                                           string       urlFolder,
                                           string       dstFolder,
                                           string       filenameToLoad,
                                           SLSceneID    sceneIDToLoad)
{
    auto progressCallback = [](size_t curr, size_t filesize) {
        if (filesize > 0)
        {
            int transferredPC = (int)((float)curr / (float)filesize * 100.0f);
            AppDemo::jobProgressNum(transferredPC);
        }
        else
            cout << "Bytes transferred: " << curr << endl;

        return 0; // Return Non-Zero to cancel
    };

    auto downloadJobHTTP = [=]() {
        PROFILE_FUNCTION();
        string jobMsg = "Downloading file via HTTPS: " + downloadFilename;
        AppDemo::jobProgressMsg(jobMsg);
        AppDemo::jobProgressMax(100);
        string fileToDownload = urlFolder + downloadFilename;
        if (!HttpUtils::download(fileToDownload, dstFolder, progressCallback))
            SL_LOG("*** Nothing downloaded from: %s ***", fileToDownload.c_str());
        AppDemo::jobIsRunning = false;
    };

    auto unzipJob = [=]() {
        string jobMsg = "Decompressing file: " + downloadFilename;
        AppDemo::jobProgressMsg(jobMsg);
        AppDemo::jobProgressMax(-1);
        string zipFile = dstFolder + downloadFilename;
        if (Utils::fileExists(zipFile))
        {
            string extension = Utils::getFileExt(zipFile);
            if (extension == "zip")
            {
                ZipUtils::unzip(zipFile, Utils::getPath(zipFile));
                Utils::deleteFile(zipFile);
            }
        }
        else
            SL_LOG("*** File do decompress doesn't exist: %s ***",
                   zipFile.c_str());
        AppDemo::jobIsRunning = false;
    };

    auto followUpJob1 = [=]() {
        string modelFile = dstFolder + filenameToLoad;
        if (Utils::fileExists(modelFile))
            s->onLoad(s, sv, sceneIDToLoad);
        else
            SL_LOG("*** File do load doesn't exist: %s ***",
                   modelFile.c_str());
    };

    AppDemo::jobsToBeThreaded.emplace_back(downloadJobHTTP);
    AppDemo::jobsToBeThreaded.emplace_back(unzipJob);
    AppDemo::jobsToFollowInMain.push_back(followUpJob1);
}
//-----------------------------------------------------------------------------
//! Set the a new active named location from SLDeviceLocation
void AppDemoGui::setActiveNamedLocation(int          locIndex,
                                        SLSceneView* sv,
                                        SLVec3f      lookAtPoint)
{
    AppDemo::devLoc.activeNamedLocation(locIndex);

#if !defined(SL_OS_MACIOS) && !defined(SL_OS_ANDROID)
    SLVec3d   pos_d = AppDemo::devLoc.defaultENU() - AppDemo::devLoc.originENU();
    SLVec3f   pos_f((SLfloat)pos_d.x, (SLfloat)pos_d.y, (SLfloat)pos_d.z);
    SLCamera* cam = sv->camera();
    cam->translation(pos_f);
    SLVec3f camToLookAt = pos_f - lookAtPoint;
    cam->focalDist(camToLookAt.length());
    cam->lookAt(lookAtPoint);
    cam->camAnim(SLCamAnim::CA_turntableYUp);
#endif
}
//-----------------------------------------------------------------------------
