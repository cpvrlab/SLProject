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

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

#include <AppDemoGui.h>
#include <SLAnimPlayback.h>
#include <SLApplication.h>
#include <CVCapture.h>
#include <CVImage.h>
#include <CVTrackedFeatures.h>
#include <SLGLShader.h>
#include <SLGLTexture.h>
#include <SLImporter.h>
#include <SLInterface.h>
#include <SLLightDirect.h>
#include <SLLightRect.h>
#include <SLLightSpot.h>
#include <SLMaterial.h>
#include <SLMesh.h>
#include <SLNode.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLTransferFunction.h>
#include <SLGLImGui.h>
#include <imgui.h>
#include <ftplib.h>

//-----------------------------------------------------------------------------
// Global pointers declared in AppDemoTracking
extern CVTracked* tracker;
extern SLNode*    trackedNode;

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
//! Listbox that allows to pass the items as a string vector
bool myListBox(const char* label, int* currIndex, SLVstring& values)
{
    if (values.empty())
        return false;

    return ImGui::ListBox(label,
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
    SLfloat width  = (SLfloat)sv->viewportW() * widthPC;
    SLfloat height = (SLfloat)sv->viewportH() * heightPC;
    ImGui::SetNextWindowSize(ImVec2(width, height), ImGuiSetCond_Always);
    ImGui::SetNextWindowPosCenter(ImGuiSetCond_Always);
}
//-----------------------------------------------------------------------------
// Init global static variables
SLstring AppDemoGui::configTime          = "-";
SLbool   AppDemoGui::showProgress        = false;
SLbool   AppDemoGui::showAbout           = false;
SLbool   AppDemoGui::showHelp            = false;
SLbool   AppDemoGui::showHelpCalibration = false;
SLbool   AppDemoGui::showCredits         = false;
SLbool   AppDemoGui::showStatsTiming     = false;
SLbool   AppDemoGui::showStatsScene      = false;
SLbool   AppDemoGui::showStatsVideo      = false;
SLbool   AppDemoGui::showInfosScene      = false;
SLbool   AppDemoGui::showInfosSensors    = false;
SLbool   AppDemoGui::showInfosDevice     = false;
SLbool   AppDemoGui::showSceneGraph      = false;
SLbool   AppDemoGui::showProperties      = false;
SLbool   AppDemoGui::showChristoffel     = false;
SLbool   AppDemoGui::showUIPrefs         = false;
SLbool   AppDemoGui::showTransform       = false;

// Scene node for Christoffel objects
static SLNode* bern          = nullptr;
static SLNode* umgeb_dach    = nullptr;
static SLNode* umgeb_fass    = nullptr;
static SLNode* boden         = nullptr;
static SLNode* balda_stahl   = nullptr;
static SLNode* balda_glas    = nullptr;
static SLNode* mauer_wand    = nullptr;
static SLNode* mauer_dach    = nullptr;
static SLNode* mauer_turm    = nullptr;
static SLNode* mauer_weg     = nullptr;
static SLNode* grab_mauern   = nullptr;
static SLNode* grab_brueck   = nullptr;
static SLNode* grab_grass    = nullptr;
static SLNode* grab_t_dach   = nullptr;
static SLNode* grab_t_fahn   = nullptr;
static SLNode* grab_t_stein  = nullptr;
static SLNode* christ_aussen = nullptr;
static SLNode* christ_innen  = nullptr;

SLstring AppDemoGui::infoAbout =
  "Welcome to the SLProject demo app. It is developed at the \
Computer Science Department of the Bern University of Applied Sciences. \
The app shows what you can learn in two semesters about 3D computer graphics \
in real time rendering and ray tracing. The framework is developed \
in C++ with OpenGL ES so that it can run also on mobile devices. \
Ray tracing provides in addition high quality transparencies, reflections and soft shadows. \
Click to close and use the menu to choose different scenes and view settings. \
For more information please visit: https://github.com/cpvrlab/SLProject\n\
";

SLstring AppDemoGui::infoCredits =
  "Contributors since 2005 in alphabetic order: \
Martin Christen, Jan Dellsperger, Manuel Frischknecht, Luc Girod, \
Michael Goettlicher, Stefan Thoeni, Timo Tschanz, Marc Wacker, Pascal Zingg \n\n\
Credits for external libraries:\n\
- assimp: assimp.sourceforge.net\n\
- imgui: github.com/ocornut/imgui\n\
- glew: glew.sourceforge.net\n\
- glfw: glfw.org\n\
- OpenCV: opencv.org\n\
- OpenGL: opengl.org\n\
- spa: Solar Position Algorithm\n\
- zlib: zlib.net\n\
";

SLstring AppDemoGui::infoHelp =
  "Help for mouse or finger control:\n\
- Use left mouse or your finger to rotate the scene\n\
- Use mouse-wheel or pinch 2 fingers to go forward/backward\n\
- Use middle-mouse or 2 fingers to move sidewards/up-down\n\
- Double click or double tap to select object\n\
- CTRL-mouse to select vertices of objects\n\
- On desktop see shortcuts behind menu commands\n\
- Check out the different test scenes under File > Load Test Scene\n\
";

SLstring AppDemoGui::infoCalibrate =
  "The calibration process requires a chessboard image to be printed \
and glued on a flat board. You can find the PDF with the chessboard image on: \n\
https://github.com/cpvrlab/SLProject/tree/master/data/calibrations/ \n\
For a calibration you have to take 20 images with detected inner \
chessboard corners. To take an image you have to click with the mouse \
or tap with finger into the screen. View the chessboard from the side so that \
the inner corners cover the full image. Hold the camera or board really still \
before taking the picture.\n \
You can mirror the video image under Preferences > Video. You can check the \
distance to the chessboard in the dialog Stats. on Video.\n \
After calibration the yellow wireframe cube should stick on the chessboard.\n\n\
Please close first this info dialog on the top-left.\n\
";

//-----------------------------------------------------------------------------
off64_t ftpXferSizeMax = 0;
//-----------------------------------------------------------------------------
int ftpCallbackXfer(off64_t xfered, void* arg)
{
    if (ftpXferSizeMax)
    {
        int xferedPC = (int)((float)xfered / (float)ftpXferSizeMax * 100.0f);
        cout << "Bytes transfered: " << xfered << " (" << xferedPC << ")" << endl;
        SLApplication::jobProgressNum(xferedPC);
    }
    else
    {
        cout << "Bytes transfered: " << xfered << endl;
    }
    return xfered ? 1 : 0;
}
//-----------------------------------------------------------------------------
//! This is the main building function for the GUI of the Demo apps
/*! Is is passed to the AppDemoGui::build function in main of the app-Demo-SLProject
 app. This function will be called once per frame roughly at the end of
 SLSceneView::onPaint in SLSceneView::draw2DGL by calling ImGui::Render.\n
 See also the comments on SLGLImGui.
 */
void AppDemoGui::build(SLScene* s, SLSceneView* sv)
{
    ///////////////////////////////////
    // Show modeless fullscreen dialogs
    ///////////////////////////////////

    // if parallel jobs are running show only the progress informations
    if (SLApplication::jobIsRunning)
    {
        centerNextWindow(sv, 0.9f, 0.5f);
        ImGui::Begin("Parallel Job in Progress",
                     &showProgress,
                     ImGuiWindowFlags_NoTitleBar);
        ImGui::Text("Parallel Job in Progress:");
        ImGui::Separator();
        ImGui::Text("%s", SLApplication::jobProgressMsg().c_str());
        if (SLApplication::jobProgressMax() > 0)
        {
            float num = (float)SLApplication::jobProgressNum();
            float max = (float)SLApplication::jobProgressMax();
            ImGui::ProgressBar(num / max);
        }
        else
            ImGui::Text("Progress: %d%%", SLApplication::jobProgressNum());

        ImGui::Separator();
        ImGui::Text("Parallel Jobs to follow: %lu",
                    SLApplication::jobsToBeThreaded.size());
        ImGui::Text("Sequential Jobs to follow: %lu",
                    SLApplication::jobsToFollowInMain.size());
        ImGui::End();
        return;
    }
    else
    {
        if (showAbout)
        {
            centerNextWindow(sv);
            ImGui::Begin("About SLProject", &showAbout, ImGuiWindowFlags_NoResize);
            ImGui::Text("Version: %s", SLApplication::version.c_str());
            ImGui::Text("Configuration: %s", SLApplication::configuration.c_str());
            ImGui::Separator();
            ImGui::Text("Git Branch: %s (Commit: %s)", SLApplication::gitBranch.c_str(), SLApplication::gitCommit.c_str());
            ImGui::Text("Git Date: %s", SLApplication::gitDate.c_str());
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
                SLfloat cullTime       = s->cullTimesMS().average();
                SLfloat draw3DTime     = s->draw3DTimesMS().average();
                SLfloat draw2DTime     = s->draw2DTimesMS().average();

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
                SLfloat draw3DTimePC     = Utils::clamp(draw3DTime / ft * 100.0f, 0.0f, 100.0f);
                SLfloat draw2DTimePC     = Utils::clamp(draw2DTime / ft * 100.0f, 0.0f, 100.0f);
                SLfloat cullTimePC       = Utils::clamp(cullTime / ft * 100.0f, 0.0f, 100.0f);

                sprintf(m + strlen(m), "Renderer   : OpenGL\n");
                sprintf(m + strlen(m), "Frame size : %d x %d\n", sv->viewportW(), sv->viewportH());
                sprintf(m + strlen(m), "Drawcalls  : %d\n", SLGLVertexArray::totalDrawCalls);
                sprintf(m + strlen(m), "FPS        :%5.1f\n", s->fps());
                sprintf(m + strlen(m), "Frame time :%5.1f ms (100%%)\n", ft);
                sprintf(m + strlen(m), " Capture   :%5.1f ms (%3d%%)\n", captureTime, (SLint)captureTimePC);
                sprintf(m + strlen(m), " Update    :%5.1f ms (%3d%%)\n", updateTime, (SLint)updateTimePC);
                sprintf(m + strlen(m), "  Anim.    :%5.1f ms (%3d%%)\n", updateAnimTime, (SLint)updateAnimTimePC);
                sprintf(m + strlen(m), "  AABB     :%5.1f ms (%3d%%)\n", updateAABBTime, (SLint)updateAABBTimePC);
                sprintf(m + strlen(m), "  Tracking :%5.1f ms (%3d%%)\n", trackingTime, (SLint)trackingTimePC);
                sprintf(m + strlen(m), "   Detect  :%5.1f ms (%3d%%)\n", detectTime, (SLint)detectTimePC);
                sprintf(m + strlen(m), "    Det1   :%5.1f ms\n", detect1Time);
                sprintf(m + strlen(m), "    Det2   :%5.1f ms\n", detect2Time);
                sprintf(m + strlen(m), "   Match   :%5.1f ms (%3d%%)\n", matchTime, (SLint)matchTimePC);
                sprintf(m + strlen(m), "   OptFlow :%5.1f ms (%3d%%)\n", optFlowTime, (SLint)optFlowTimePC);
                sprintf(m + strlen(m), "   Pose    :%5.1f ms (%3d%%)\n", poseTime, (SLint)poseTimePC);
                sprintf(m + strlen(m), " Culling   :%5.1f ms (%3d%%)\n", cullTime, (SLint)cullTimePC);
                sprintf(m + strlen(m), " Drawing 3D:%5.1f ms (%3d%%)\n", draw3DTime, (SLint)draw3DTimePC);
                sprintf(m + strlen(m), " Drawing 2D:%5.1f ms (%3d%%)\n", draw2DTime, (SLint)draw2DTimePC);
            }
            else if (rType == RT_ct)
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
                SLfloat cullTime       = s->cullTimesMS().average();
                SLfloat draw3DTime     = s->draw3DTimesMS().average();
                SLfloat draw2DTime     = s->draw2DTimesMS().average();

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
            else if (rType == RT_rt)
            {
                SLRaytracer* rt           = sv->raytracer();
                SLuint       rayPrimaries = (SLuint)(sv->viewportW() * sv->viewportH());
                SLuint       rayTotal     = rayPrimaries + SLRay::reflectedRays + SLRay::subsampledRays + SLRay::refractedRays + SLRay::shadowRays;
                SLfloat      renderSec    = rt->renderSec();
                SLfloat      rpms         = renderSec > 0.001f ? rayTotal / renderSec / 1000.0f : 0.0f;
                SLfloat      fps          = renderSec > 0.001f ? 1.0f / rt->renderSec() : 0.0f;

                sprintf(m + strlen(m), "Renderer   :Ray Tracer\n");
                sprintf(m + strlen(m), "Frame size :%d x %d\n", sv->viewportW(), sv->viewportH());
                sprintf(m + strlen(m), "FPS        :%0.2f\n", fps);
                sprintf(m + strlen(m), "Frame Time :%0.2f sec.\n", renderSec);
                sprintf(m + strlen(m), "Rays per ms:%0.0f\n", rpms);
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
            else if (rType == RT_pt)
            {
                SLPathtracer* pt           = sv->pathtracer();
                SLuint        rayPrimaries = (SLuint)(sv->viewportW() * sv->viewportH());
                SLuint        rayTotal     = rayPrimaries + SLRay::reflectedRays + SLRay::subsampledRays + SLRay::refractedRays + SLRay::shadowRays;
                SLfloat       rpms         = pt->renderSec() > 0.0f ? rayTotal / pt->renderSec() / 1000.0f : 0.0f;

                sprintf(m + strlen(m), "Renderer   :Path Tracer\n");
                sprintf(m + strlen(m), "Frame size :%d x %d\n", sv->viewportW(), sv->viewportH());
                sprintf(m + strlen(m), "FPS        :%0.2f\n", 1.0f / pt->renderSec());
                sprintf(m + strlen(m), "Frame Time :%0.2f sec.\n", pt->renderSec());
                sprintf(m + strlen(m), "Rays per ms:%0.0f\n", rpms);
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

            SLNodeStats& stats3D         = sv->stats3D();
            SLfloat      vox             = (SLfloat)stats3D.numVoxels;
            SLfloat      voxEmpty        = (SLfloat)stats3D.numVoxEmpty;
            SLfloat      voxelsEmpty     = vox > 0.0f ? voxEmpty / vox * 100.0f : 0.0f;
            SLfloat      numRTTria       = (SLfloat)stats3D.numTriangles;
            SLfloat      avgTriPerVox    = vox > 0.0f ? numRTTria / (vox - voxEmpty) : 0.0f;
            SLint        numOpaqueNodes  = (int)sv->nodesVisible()->size();
            SLint        numBlendedNodes = (int)sv->nodesBlended()->size();
            SLint        numVisibleNodes = numOpaqueNodes + numBlendedNodes;
            SLint        numGroupPC      = (SLint)((SLfloat)stats3D.numGroupNodes / (SLfloat)stats3D.numNodes * 100.0f);
            SLint        numLeafPC       = (SLint)((SLfloat)stats3D.numLeafNodes / (SLfloat)stats3D.numNodes * 100.0f);
            SLint        numLightsPC     = (SLint)((SLfloat)stats3D.numLights / (SLfloat)stats3D.numNodes * 100.0f);
            SLint        numOpaquePC     = (SLint)((SLfloat)numOpaqueNodes / (SLfloat)stats3D.numNodes * 100.0f);
            SLint        numBlendedPC    = (SLint)((SLfloat)numBlendedNodes / (SLfloat)stats3D.numNodes * 100.0f);
            SLint        numVisiblePC    = (SLint)((SLfloat)numVisibleNodes / (SLfloat)stats3D.numNodes * 100.0f);

            // Calculate total size of texture bytes on CPU
            SLfloat cpuMBTexture = 0;
            for (auto t : s->textures())
                for (auto i : t->images())
                    cpuMBTexture += i->bytesPerImage();
            cpuMBTexture = cpuMBTexture / 1E6f;

            SLfloat cpuMBMeshes    = (SLfloat)stats3D.numBytes / 1E6f;
            SLfloat cpuMBVoxels    = (SLfloat)stats3D.numBytesAccel / 1E6f;
            SLfloat cpuMBTotal     = cpuMBTexture + cpuMBMeshes + cpuMBVoxels;
            SLint   cpuMBTexturePC = (SLint)(cpuMBTexture / cpuMBTotal * 100.0f);
            SLint   cpuMBMeshesPC  = (SLint)(cpuMBMeshes / cpuMBTotal * 100.0f);
            SLint   cpuMBVoxelsPC  = (SLint)(cpuMBVoxels / cpuMBTotal * 100.0f);
            SLfloat gpuMBTexture   = (SLfloat)SLGLTexture::numBytesInTextures / 1E6f;
            SLfloat gpuMBVbo       = (SLfloat)SLGLVertexBuffer::totalBufferSize / 1E6f;
            SLfloat gpuMBTotal     = gpuMBTexture + gpuMBVbo;
            SLint   gpuMBTexturePC = (SLint)(gpuMBTexture / gpuMBTotal * 100.0f);
            SLint   gpuMBVboPC     = (SLint)(gpuMBVbo / gpuMBTotal * 100.0f);

            sprintf(m + strlen(m), "Name: %s\n", s->name().c_str());
            sprintf(m + strlen(m), "No. of Nodes  :%5d (100%%)\n", stats3D.numNodes);
            sprintf(m + strlen(m), "- Group Nodes :%5d (%3d%%)\n", stats3D.numGroupNodes, numGroupPC);
            sprintf(m + strlen(m), "- Leaf  Nodes :%5d (%3d%%)\n", stats3D.numLeafNodes, numLeafPC);
            sprintf(m + strlen(m), "- Light Nodes :%5d (%3d%%)\n", stats3D.numLights, numLightsPC);
            sprintf(m + strlen(m), "- Opaque Nodes:%5d (%3d%%)\n", numOpaqueNodes, numOpaquePC);
            sprintf(m + strlen(m), "- Blend Nodes :%5d (%3d%%)\n", numBlendedNodes, numBlendedPC);
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

            sprintf(m + strlen(m), "Video Type   : %s\n", vt == VT_NONE ? "None" : vt == VT_MAIN ? "Main Camera" : vt == VT_FILE ? "File" : "Secondary Camera");
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

            int distortionSize = c->distortion().rows;
            sprintf(m + strlen(m), "distortion (*10e-2):\n");
            const float f = 100.f;
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

        if (showInfosScene)
        {
            // Calculate window position for dynamic status bar at the bottom of the main window
            ImGuiWindowFlags window_flags = 0;
            window_flags |= ImGuiWindowFlags_NoTitleBar;
            window_flags |= ImGuiWindowFlags_NoResize;
            SLfloat  w    = (SLfloat)sv->viewportW();
            ImVec2   size = ImGui::CalcTextSize(s->info().c_str(), nullptr, true, w);
            SLfloat  h    = size.y + SLGLImGui::fontPropDots * 1.2f;
            SLstring info = "Scene Info: " + s->info();

            ImGui::SetNextWindowPos(ImVec2(0, sv->scrH() - h));
            ImGui::SetNextWindowSize(ImVec2(w, h));
            ImGui::Begin("Scene Information", &showInfosScene, window_flags);
            ImGui::TextWrapped("%s", info.c_str());
            ImGui::End();
        }

        if (showTransform)
        {
            ImGuiWindowFlags window_flags = 0;
            window_flags |= ImGuiWindowFlags_AlwaysAutoResize;
            ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
            ImGui::Begin("Transform Selected Node", &showTransform, window_flags);

            if (s->selectedNode())
            {
                SLNode*                 node   = s->selectedNode();
                static SLTransformSpace tSpace = TS_object;
                SLfloat                 t1 = 0.1f, t2 = 1.0f, t3 = 10.0f; // Delta translations
                SLfloat                 r1 = 1.0f, r2 = 5.0f, r3 = 15.0f; // Delta rotations
                SLfloat                 s1 = 1.01f, s2 = 1.1f, s3 = 1.5f; // Scale factors

                // clang-format off
                ImGui::Text("Transf. Space:"); ImGui::SameLine();
                if (ImGui::RadioButton("Object", (int*)&tSpace, 0)) tSpace = TS_object; ImGui::SameLine();
                if (ImGui::RadioButton("World",  (int*)&tSpace, 1)) tSpace = TS_world; ImGui::SameLine();
                if (ImGui::RadioButton("Parent", (int*)&tSpace, 2)) tSpace = TS_parent;
                ImGui::Separator();

                ImGui::Text("Translation X:"); ImGui::SameLine();
                if (ImGui::Button("<<<##Tx")) node->translate(-t3, 0, 0, tSpace); ImGui::SameLine();
                if (ImGui::Button("<<##Tx"))  node->translate(-t2, 0, 0, tSpace); ImGui::SameLine();
                if (ImGui::Button("<##Tx"))   node->translate(-t1, 0, 0, tSpace); ImGui::SameLine();
                if (ImGui::Button(">##Tx"))   node->translate( t1, 0, 0, tSpace); ImGui::SameLine();
                if (ImGui::Button(">>##Tx"))  node->translate( t2, 0, 0, tSpace); ImGui::SameLine();
                if (ImGui::Button(">>>##Tx")) node->translate( t3, 0, 0, tSpace);

                ImGui::Text("Translation Y:"); ImGui::SameLine();
                if (ImGui::Button("<<<##Ty")) node->translate(0, -t3, 0, tSpace); ImGui::SameLine();
                if (ImGui::Button("<<##Ty"))  node->translate(0, -t2, 0, tSpace); ImGui::SameLine();
                if (ImGui::Button("<##Ty"))   node->translate(0, -t1, 0, tSpace); ImGui::SameLine();
                if (ImGui::Button(">##Ty"))   node->translate(0,  t1, 0, tSpace); ImGui::SameLine();
                if (ImGui::Button(">>##Ty"))  node->translate(0,  t2, 0, tSpace); ImGui::SameLine();
                if (ImGui::Button(">>>##Ty")) node->translate(0,  t3, 0, tSpace);

                ImGui::Text("Translation Z:"); ImGui::SameLine();
                if (ImGui::Button("<<<##Tz")) node->translate(0, 0, -t3, tSpace); ImGui::SameLine();
                if (ImGui::Button("<<##Tz"))  node->translate(0, 0, -t2, tSpace); ImGui::SameLine();
                if (ImGui::Button("<##Tz"))   node->translate(0, 0, -t1, tSpace); ImGui::SameLine();
                if (ImGui::Button(">##Tz"))   node->translate(0, 0,  t1, tSpace); ImGui::SameLine();
                if (ImGui::Button(">>##Tz"))  node->translate(0, 0,  t2, tSpace); ImGui::SameLine();
                if (ImGui::Button(">>>##Tz")) node->translate(0, 0,  t3, tSpace);

                ImGui::Text("Rotation X   :"); ImGui::SameLine();
                if (ImGui::Button("<<<##Rx")) node->rotate( r3, 1, 0, 0, tSpace); ImGui::SameLine();
                if (ImGui::Button("<<##Rx"))  node->rotate( r2, 1, 0, 0, tSpace); ImGui::SameLine();
                if (ImGui::Button("<##Rx"))   node->rotate( r1, 1, 0, 0, tSpace); ImGui::SameLine();
                if (ImGui::Button(">##Rx"))   node->rotate(-r1, 1, 0, 0, tSpace); ImGui::SameLine();
                if (ImGui::Button(">>##Rx"))  node->rotate(-r2, 1, 0, 0, tSpace); ImGui::SameLine();
                if (ImGui::Button(">>>##Rx")) node->rotate(-r3, 1, 0, 0, tSpace);

                ImGui::Text("Rotation Y   :"); ImGui::SameLine();
                if (ImGui::Button("<<<##Ry")) node->rotate( r3, 0, 1, 0, tSpace); ImGui::SameLine();
                if (ImGui::Button("<<##Ry"))  node->rotate( r2, 0, 1, 0, tSpace); ImGui::SameLine();
                if (ImGui::Button("<##Ry"))   node->rotate( r1, 0, 1, 0, tSpace); ImGui::SameLine();
                if (ImGui::Button(">##Ry"))   node->rotate(-r1, 0, 1, 0, tSpace); ImGui::SameLine();
                if (ImGui::Button(">>##Ry"))  node->rotate(-r2, 0, 1, 0, tSpace); ImGui::SameLine();
                if (ImGui::Button(">>>##Ry")) node->rotate(-r3, 0, 1, 0, tSpace);

                ImGui::Text("Rotation Z   :"); ImGui::SameLine();
                if (ImGui::Button("<<<##Rz")) node->rotate( r3, 0, 0, 1, tSpace); ImGui::SameLine();
                if (ImGui::Button("<<##Rz"))  node->rotate( r2, 0, 0, 1, tSpace); ImGui::SameLine();
                if (ImGui::Button("<##Rz"))   node->rotate( r1, 0, 0, 1, tSpace); ImGui::SameLine();
                if (ImGui::Button(">##Rz"))   node->rotate(-r1, 0, 0, 1, tSpace); ImGui::SameLine();
                if (ImGui::Button(">>##Rz"))  node->rotate(-r2, 0, 0, 1, tSpace); ImGui::SameLine();
                if (ImGui::Button(">>>##Rz")) node->rotate(-r3, 0, 0, 1, tSpace);

                ImGui::Text("Scale        :"); ImGui::SameLine();
                if (ImGui::Button("<<<##S")) node->scale( s3); ImGui::SameLine();
                if (ImGui::Button("<<##S"))  node->scale( s2); ImGui::SameLine();
                if (ImGui::Button("<##S"))   node->scale( s1); ImGui::SameLine();
                if (ImGui::Button(">##S"))   node->scale(-s1); ImGui::SameLine();
                if (ImGui::Button(">>##S"))  node->scale(-s2); ImGui::SameLine();
                if (ImGui::Button(">>>##S")) node->scale(-s3);
                ImGui::Separator();
                if (ImGui::Button("Reset")) node->om(node->initialOM());

                // clang-format on
            }
            else
            {
                ImGui::Text("No node selected.");
                ImGui::Text("Please select a node by double clicking it.");
            }
            ImGui::End();
            ImGui::PopFont();
        }

        if (showInfosDevice)
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
            sprintf(m + strlen(m), "Computer User    : %s\n", SLApplication::computerUser.c_str());
            sprintf(m + strlen(m), "Computer Name    : %s\n", SLApplication::computerName.c_str());
            sprintf(m + strlen(m), "Computer Brand   : %s\n", SLApplication::computerBrand.c_str());
            sprintf(m + strlen(m), "Computer Model   : %s\n", SLApplication::computerModel.c_str());
            sprintf(m + strlen(m), "Computer Arch.   : %s\n", SLApplication::computerArch.c_str());
            sprintf(m + strlen(m), "Computer OS      : %s\n", SLApplication::computerOS.c_str());
            sprintf(m + strlen(m), "Computer OS Ver. : %s\n", SLApplication::computerOSVer.c_str());
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
            ImGui::Begin("Device Informations", &showInfosDevice, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
            ImGui::TextUnformatted(m);
            ImGui::End();
            ImGui::PopFont();
        }

        if (showInfosSensors)
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
            sprintf(m + strlen(m), "Altitude GPS (m)    : %11.6f\n", SLApplication::devLoc.altGpsM());
            sprintf(m + strlen(m), "Altitude DEM (m)    : %11.6f\n", SLApplication::devLoc.altGpsM());
            sprintf(m + strlen(m), "Accuracy Radius (m) : %6.1f\n", SLApplication::devLoc.locAccuracyM());
            sprintf(m + strlen(m), "Dist. to Origin (m) : %6.1f\n", offsetToOrigin.length());
            sprintf(m + strlen(m), "Max. Dist. (m)      : %6.1f\n", SLApplication::devLoc.locMaxDistanceM());
            sprintf(m + strlen(m), "Origin improve time : %6.1f sec.\n", SLApplication::devLoc.improveTime());
            sprintf(m + strlen(m), "Sun Zenit (deg)     : %6.1f sec.\n", SLApplication::devLoc.originSolarZenit());
            sprintf(m + strlen(m), "Sun Azimut (deg)    : %6.1f sec.\n", SLApplication::devLoc.originSolarAzimut());

            // Switch to fixed font
            ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
            ImGui::Begin("Sensor Informations", &showInfosSensors, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
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
            buildProperties(s);
        }

        if (showUIPrefs)
        {
            ImGuiWindowFlags window_flags = 0;
            window_flags |= ImGuiWindowFlags_AlwaysAutoResize;
            ImGui::Begin("User Interface Preferences", &showUIPrefs, window_flags);

            ImGui::SliderFloat("Prop. Font Size", &SLGLImGui::fontPropDots, 16.f, 70.f, "%0.0f");
            ImGui::SliderFloat("Fixed Font Size", &SLGLImGui::fontFixedDots, 13.f, 50.f, "%0.0f");
            ImGuiStyle& style = ImGui::GetStyle();
            if (ImGui::SliderFloat("Item Spacing X", &style.ItemSpacing.x, 0.0f, 20.0f, "%0.0f"))
                style.WindowPadding.x = style.FramePadding.x = style.ItemInnerSpacing.x = style.ItemSpacing.x;
            if (ImGui::SliderFloat("Item Spacing Y", &style.ItemSpacing.y, 0.0f, 10.0f, "%0.0f"))
                style.WindowPadding.y = style.FramePadding.y = style.ItemInnerSpacing.y = style.ItemSpacing.y;

            ImGui::Separator();

            SLchar reset[255];

            ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
            sprintf(reset, "Reset User Interface (DPI: %d)", SLApplication::dpi);
            if (ImGui::MenuItem(reset))
            {
                SLstring fullPathFilename = SLApplication::configPath + "DemoGui.yml";
                Utils::deleteFile(fullPathFilename);
                loadConfig(SLApplication::dpi);
            }
            ImGui::PopFont();

            ImGui::End();
        }

        if (showChristoffel && SLApplication::sceneID == SID_VideoChristoffel)
        {

            ImGui::Begin("Christoffel",
                         &showChristoffel,
                         ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);

            // Get scene nodes once
            if (!bern)
            {
                bern          = s->root3D()->findChild<SLNode>("Bern-Bahnhofsplatz.fbx");
                boden         = bern->findChild<SLNode>("Boden", true);
                balda_stahl   = bern->findChild<SLNode>("Baldachin-Stahl", true);
                balda_glas    = bern->findChild<SLNode>("Baldachin-Glas", true);
                umgeb_dach    = bern->findChild<SLNode>("Umgebung-Daecher", true);
                umgeb_fass    = bern->findChild<SLNode>("Umgebung-Fassaden", true);
                mauer_wand    = bern->findChild<SLNode>("Mauer-Wand", true);
                mauer_dach    = bern->findChild<SLNode>("Mauer-Dach", true);
                mauer_turm    = bern->findChild<SLNode>("Mauer-Turm", true);
                mauer_weg     = bern->findChild<SLNode>("Mauer-Weg", true);
                grab_mauern   = bern->findChild<SLNode>("Graben-Mauern", true);
                grab_brueck   = bern->findChild<SLNode>("Graben-Bruecken", true);
                grab_grass    = bern->findChild<SLNode>("Graben-Grass", true);
                grab_t_dach   = bern->findChild<SLNode>("Graben-Turm-Dach", true);
                grab_t_fahn   = bern->findChild<SLNode>("Graben-Turm-Fahne", true);
                grab_t_stein  = bern->findChild<SLNode>("Graben-Turm-Stein", true);
                christ_aussen = bern->findChild<SLNode>("Christoffel-Aussen", true);
                christ_innen  = bern->findChild<SLNode>("Christoffel-Innen", true);
            }

            ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.66f);

            SLbool umgebung = !umgeb_fass->drawBits()->get(SL_DB_HIDDEN);
            if (ImGui::Checkbox("Umgebung", &umgebung))
            {
                umgeb_fass->drawBits()->set(SL_DB_HIDDEN, !umgebung);
                umgeb_dach->drawBits()->set(SL_DB_HIDDEN, !umgebung);
            }

            SLbool bodenBool = !boden->drawBits()->get(SL_DB_HIDDEN);
            if (ImGui::Checkbox("Boden", &bodenBool))
            {
                boden->drawBits()->set(SL_DB_HIDDEN, !bodenBool);
            }

            SLbool baldachin = !balda_stahl->drawBits()->get(SL_DB_HIDDEN);
            if (ImGui::Checkbox("Baldachin", &baldachin))
            {
                balda_stahl->drawBits()->set(SL_DB_HIDDEN, !baldachin);
                balda_glas->drawBits()->set(SL_DB_HIDDEN, !baldachin);
            }

            SLbool mauer = !mauer_wand->drawBits()->get(SL_DB_HIDDEN);
            if (ImGui::Checkbox("Mauer", &mauer))
            {
                mauer_wand->drawBits()->set(SL_DB_HIDDEN, !mauer);
                mauer_dach->drawBits()->set(SL_DB_HIDDEN, !mauer);
                mauer_turm->drawBits()->set(SL_DB_HIDDEN, !mauer);
                mauer_weg->drawBits()->set(SL_DB_HIDDEN, !mauer);
            }

            SLbool graben = !grab_mauern->drawBits()->get(SL_DB_HIDDEN);
            if (ImGui::Checkbox("Graben", &graben))
            {
                grab_mauern->drawBits()->set(SL_DB_HIDDEN, !graben);
                grab_brueck->drawBits()->set(SL_DB_HIDDEN, !graben);
                grab_grass->drawBits()->set(SL_DB_HIDDEN, !graben);
                grab_t_dach->drawBits()->set(SL_DB_HIDDEN, !graben);
                grab_t_fahn->drawBits()->set(SL_DB_HIDDEN, !graben);
                grab_t_stein->drawBits()->set(SL_DB_HIDDEN, !graben);
            }

            static SLfloat christTransp = 0.0f;
            if (ImGui::SliderFloat("Transparency", &christTransp, 0.0f, 1.0f, "%0.2f"))
            {
                for (auto mesh : christ_aussen->meshes())
                {
                    mesh->mat()->kt(christTransp);
                    mesh->mat()->ambient(SLCol4f(0.3f, 0.3f, 0.3f));
                    mesh->init(christ_aussen);
                }

                // Hide inner parts if transparency is on
                christ_innen->drawBits()->set(SL_DB_HIDDEN, christTransp > 0.01f);
            }
            ImGui::PopItemWidth();
            ImGui::End();
        }
        else
        {
            bern         = nullptr;
            boden        = nullptr;
            balda_stahl  = nullptr;
            balda_glas   = nullptr;
            umgeb_dach   = nullptr;
            umgeb_fass   = nullptr;
            mauer_wand   = nullptr;
            mauer_dach   = nullptr;
            mauer_turm   = nullptr;
            mauer_weg    = nullptr;
            grab_mauern  = nullptr;
            grab_brueck  = nullptr;
            grab_grass   = nullptr;
            grab_t_dach  = nullptr;
            grab_t_fahn  = nullptr;
            grab_t_stein = nullptr;
        }
    }
}
//-----------------------------------------------------------------------------
CVCalibration guessCalibration(bool         mirroredH,
                               bool         mirroredV,
                               CVCameraType camType)
{
    // Try to read device lens and sensor information
    string strF = SLApplication::deviceParameter["DeviceLensFocalLength"];
    string strW = SLApplication::deviceParameter["DeviceSensorPhysicalSizeW"];
    string strH = SLApplication::deviceParameter["DeviceSensorPhysicalSizeH"];
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
                             SLApplication::getComputerInfos());
    }
    else
    {
        //make a guess using frame size and a guessed field of view
        return CVCalibration(cv::Size(CVCapture::instance()->lastFrame.cols,
                                      CVCapture::instance()->lastFrame.rows),
                             60.0,
                             mirroredH,
                             mirroredV,
                             camType,
                             SLApplication::getComputerInfos());
    }
}
//-----------------------------------------------------------------------------
//! Builds the entire menu bar once per frame
void AppDemoGui::buildMenuBar(SLScene* s, SLSceneView* sv)
{
    SLSceneID    sid           = SLApplication::sceneID;
    SLGLState*   stateGL       = SLGLState::instance();
    CVCapture*   capture       = CVCapture::instance();
    SLRenderType rType         = sv->renderType();
    SLbool       hasAnimations = (!s->animManager().allAnimNames().empty());
    static SLint curAnimIx     = -1;
    if (!hasAnimations) curAnimIx = -1;

    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::BeginMenu("Load Test Scene"))
            {
                if (ImGui::BeginMenu("General Scenes"))
                {
                    if (ImGui::MenuItem("Minimal Scene", nullptr, sid == SID_Minimal))
                    {
                        s->onLoad(s, sv, SID_Minimal);
                    }
                    if (ImGui::MenuItem("Figure Scene", nullptr, sid == SID_Figure))
                        s->onLoad(s, sv, SID_Figure);
#if not defined(SL_OS_ANDROID) and not defined(SL_OS_IOS)
                    if (ImGui::MenuItem("Large Model", nullptr, sid == SID_LargeModel))
                    {
                        SLstring largeFile = SLImporter::defaultPath + "PLY/xyzrgb_dragon.ply";
                        if (Utils::fileExists(largeFile))
                            s->onLoad(s, sv, SID_LargeModel);
                        else
                        {
                            auto downloadJob = []() {
                                SLApplication::jobProgressMsg("Downloading large Dragon file from pallas.bfh.ch");
                                SLApplication::jobProgressMax(100);
                                ftplib ftp;
                                if (ftp.Connect("pallas.bfh.ch:21"))
                                {
                                    if (ftp.Login("upload", "FaAdbD3F2a"))
                                    {
                                        ftp.SetCallbackXferFunction(ftpCallbackXfer);
                                        ftp.SetCallbackBytes(1024000);
                                        if (ftp.Chdir("test"))
                                        {
                                            int remoteSize = 0;
                                            ftp.Size("xyzrgb_dragon.ply",
                                                     &remoteSize,
                                                     ftplib::transfermode::image);
                                            ftpXferSizeMax  = remoteSize;
                                            SLstring plyDir = SLImporter::defaultPath + "PLY";
                                            if (!Utils::dirExists(plyDir))
                                                Utils::makeDir(plyDir);
                                            if (Utils::dirExists(plyDir))
                                            {
                                                SLstring outFile = SLImporter::defaultPath + "PLY/xyzrgb_dragon.ply";
                                                if (!ftp.Get(outFile.c_str(),
                                                             "xyzrgb_dragon.ply",
                                                             ftplib::transfermode::image))
                                                    SL_LOG("*** ERROR: ftp.Get failed. ***");
                                            }
                                            else
                                                SL_LOG("*** ERROR: Utils::makeDir %s failed. ***", plyDir.c_str());
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
                                SLApplication::jobIsRunning = false;
                            };

                            auto jobToFollow1 = [](SLScene* s, SLSceneView* sv) {
                                SLstring largeFile = SLImporter::defaultPath + "PLY/xyzrgb_dragon.ply";
                                if (Utils::fileExists(largeFile))
                                    s->onLoad(s, sv, SID_LargeModel);
                            };

                            function<void(void)> jobNoArgs = bind(jobToFollow1, s, sv);

                            SLApplication::jobsToBeThreaded.emplace_back(downloadJob);
                            SLApplication::jobsToFollowInMain.push_back(jobNoArgs);
                        }
                    }
#endif
                    if (ImGui::MenuItem("Mesh Loader", nullptr, sid == SID_MeshLoad))
                        s->onLoad(s, sv, SID_MeshLoad);
                    if (ImGui::MenuItem("Revolver Meshes", nullptr, sid == SID_Revolver))
                        s->onLoad(s, sv, SID_Revolver);
                    if (ImGui::MenuItem("Texture Blending", nullptr, sid == SID_TextureBlend))
                        s->onLoad(s, sv, SID_TextureBlend);
                    if (ImGui::MenuItem("Texture Filters", nullptr, sid == SID_TextureFilter))
                        s->onLoad(s, sv, SID_TextureFilter);
                    if (ImGui::MenuItem("Frustum Culling", nullptr, sid == SID_FrustumCull))
                        s->onLoad(s, sv, SID_FrustumCull);
                    if (ImGui::MenuItem("Massive Data Scene", nullptr, sid == SID_MassiveData))
                        s->onLoad(s, sv, SID_MassiveData);
                    if (ImGui::MenuItem("2D and 3D Text", nullptr, sid == SID_2Dand3DText))
                        s->onLoad(s, sv, SID_2Dand3DText);
                    if (ImGui::MenuItem("Point Clouds", nullptr, sid == SID_PointClouds))
                        s->onLoad(s, sv, SID_PointClouds);

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Shader"))
                {
                    if (ImGui::MenuItem("Per Vertex Blinn-Phong", nullptr, sid == SID_ShaderPerVertexBlinn))
                        s->onLoad(s, sv, SID_ShaderPerVertexBlinn);
                    if (ImGui::MenuItem("Per Pixel Blinn-Phing", nullptr, sid == SID_ShaderPerPixelBlinn))
                        s->onLoad(s, sv, SID_ShaderPerPixelBlinn);
                    if (ImGui::MenuItem("Per Pixel Cook-Torrance", nullptr, sid == SID_ShaderCookTorrance))
                        s->onLoad(s, sv, SID_ShaderCookTorrance);
                    if (ImGui::MenuItem("Per Vertex Wave", nullptr, sid == SID_ShaderPerVertexWave))
                        s->onLoad(s, sv, SID_ShaderPerVertexWave);
                    if (ImGui::MenuItem("Water", nullptr, sid == SID_ShaderWater))
                        s->onLoad(s, sv, SID_ShaderWater);
                    if (ImGui::MenuItem("Bump Mapping", nullptr, sid == SID_ShaderBumpNormal))
                        s->onLoad(s, sv, SID_ShaderBumpNormal);
                    if (ImGui::MenuItem("Parallax Mapping", nullptr, sid == SID_ShaderBumpParallax))
                        s->onLoad(s, sv, SID_ShaderBumpParallax);
                    if (ImGui::MenuItem("Skybox Shader", nullptr, sid == SID_ShaderSkyBox))
                        s->onLoad(s, sv, SID_ShaderSkyBox);
                    if (ImGui::MenuItem("Earth Shader", nullptr, sid == SID_ShaderEarth))
                        s->onLoad(s, sv, SID_ShaderEarth);
                    if (ImGui::MenuItem("Voxel Cone Tracing Shader", nullptr, sid == SID_ShaderVoxelConeDemo))
                        s->onLoad(s, sv, SID_ShaderVoxelConeDemo);

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Animation"))
                {
                    if (ImGui::MenuItem("Node Animation", nullptr, sid == SID_AnimationNode))
                        s->onLoad(s, sv, SID_AnimationNode);
                    if (ImGui::MenuItem("Mass Animation", nullptr, sid == SID_AnimationMass))
                        s->onLoad(s, sv, SID_AnimationMass);
                    if (ImGui::MenuItem("Astroboy Army", nullptr, sid == SID_AnimationArmy))
                        s->onLoad(s, sv, SID_AnimationArmy);
                    if (ImGui::MenuItem("Skeletal Animation", nullptr, sid == SID_AnimationSkeletal))
                        s->onLoad(s, sv, SID_AnimationSkeletal);

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

                if (ImGui::BeginMenu("Erleb-AR"))
                {
                    if (ImGui::MenuItem("Christoffel Tower AR (Main)", nullptr, sid == SID_VideoChristoffel))
                        s->onLoad(s, sv, SID_VideoChristoffel);

                    SLstring modelAR1 = SLImporter::defaultPath + "Tempel-Theater-02.gltf"; // Android
                    SLstring modelAR2 = SLImporter::defaultPath + "GLTF/AugustaRaurica/Tempel-Theater-02.gltf";

                    if (Utils::fileExists(modelAR1) || Utils::fileExists(modelAR2))
                        if (ImGui::MenuItem("Augusta Raurica AR (Main)", nullptr, sid == SID_VideoAugustaRaurica))
                            s->onLoad(s, sv, SID_VideoAugustaRaurica);

                    SLstring modelAV11 = SLImporter::defaultPath + "Aventicum-Amphitheater1.gltf"; // Android
                    SLstring modelAV12 = SLImporter::defaultPath + "GLTF/Aventicum/Aventicum-Amphitheater1.gltf";
                    if (Utils::fileExists(modelAV11) || Utils::fileExists(modelAV12))
                        if (ImGui::MenuItem("Aventicum Amphitheatre AR (Main)", nullptr, sid == SID_VideoAventicumAmphi))
                            s->onLoad(s, sv, SID_VideoAventicumAmphi);

                    SLstring modelAV21 = SLImporter::defaultPath + "Aventicum-Theater1.gltf"; // Android
                    SLstring modelAV22 = SLImporter::defaultPath + "GLTF/Aventicum/Aventicum-Theater1.gltf";
                    if (Utils::fileExists(modelAV21) || Utils::fileExists(modelAV22))
                        if (ImGui::MenuItem("Aventicum Theatre AR (Main)", nullptr, sid == SID_VideoAventicumTheatre))
                            s->onLoad(s, sv, SID_VideoAventicumTheatre);

                    SLstring modelAV31 = SLImporter::defaultPath + "Aventicum-Cigognier1.gltf"; // Android
                    SLstring modelAV32 = SLImporter::defaultPath + "GLTF/Aventicum/Aventicum-Cigonier1.gltf";
                    if (Utils::fileExists(modelAV31) || Utils::fileExists(modelAV32))
                        if (ImGui::MenuItem("Aventicum Cigognier AR (Main)", nullptr, sid == SID_VideoAventicumCigonier))
                            s->onLoad(s, sv, SID_VideoAventicumCigonier);

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Volume Rendering"))
                {
                    if (ImGui::MenuItem("Head MRI Ray Cast", nullptr, sid == SID_VolumeRayCast))
                        s->onLoad(s, sv, SID_VolumeRayCast);
#ifndef SL_GLES
                    if (ImGui::MenuItem("Head MRI Ray Cast Lighted", nullptr, sid == SID_VolumeRayCastLighted))
                        s->onLoad(s, sv, SID_VolumeRayCastLighted);
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

                ImGui::EndMenu();
            }

            if (ImGui::MenuItem("Empty Scene", nullptr, sid == SID_Empty))
                s->onLoad(s, sv, SID_Empty);

            if (ImGui::MenuItem("Multithreaded Job demo"))
            {
                auto job1 = []() {
                    uint maxIter = 1000000;
                    SLApplication::jobProgressMsg("Super long job 1");
                    SLApplication::jobProgressMax(100);
                    for (uint i = 0; i < maxIter; ++i)
                    {
                        SL_LOG("%u", i);
                        int progressPC = (int)((float)i / (float)maxIter * 100.0f);
                        SLApplication::jobProgressNum(progressPC);
                    }
                    SLApplication::jobIsRunning = false;
                };

                auto job2 = []() {
                    uint maxIter = 100000;
                    SLApplication::jobProgressMsg("Super long job 2");
                    SLApplication::jobProgressMax(100);
                    for (uint i = 0; i < maxIter; ++i)
                    {
                        SL_LOG("%u", i);
                        int progressPC = (int)((float)i / (float)maxIter * 100.0f);
                        SLApplication::jobProgressNum(progressPC);
                    }
                    SLApplication::jobIsRunning = false;
                };

                auto jobToFollow1 = []() { SL_LOG("JobToFollow1"); };
                auto jobToFollow2 = []() { SL_LOG("JobToFollow2"); };

                SLApplication::jobsToBeThreaded.emplace_back(job1);
                SLApplication::jobsToBeThreaded.emplace_back(job2);
                SLApplication::jobsToFollowInMain.emplace_back(jobToFollow1);
                SLApplication::jobsToFollowInMain.emplace_back(jobToFollow2);
            }

#ifndef SL_OS_ANDROID
            ImGui::Separator();

            if (ImGui::MenuItem("Quit & Save", "ESC"))
                slShouldClose(true);
#endif

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Preferences"))
        {
            if (ImGui::MenuItem("Do Wait on Idle", "I", sv->doWaitOnIdle()))
                sv->doWaitOnIdle(!sv->doWaitOnIdle());

            if (ImGui::MenuItem("Do Multi Sampling", "M", sv->doMultiSampling()))
                sv->doMultiSampling(!sv->doMultiSampling());

            if (ImGui::MenuItem("Do Frustum Culling", "F", sv->doFrustumCulling()))
                sv->doFrustumCulling(!sv->doFrustumCulling());

            if (ImGui::MenuItem("Do Depth Test", "T", sv->doDepthTest()))
                sv->doDepthTest(!sv->doDepthTest());

            if (ImGui::MenuItem("Animation off", "O", s->stopAnimations()))
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
#if defined(SL_OS_ANDROID) || defined(SL_OS_IOS)
            if (ImGui::BeginMenu("Rotation Sensor"))
            {
                if (ImGui::MenuItem("Use Device Rotation (IMU)", nullptr, SLApplication::devRot.isUsed()))
                    SLApplication::devRot.isUsed(!SLApplication::devRot.isUsed());

                if (ImGui::MenuItem("Zero Yaw at Start", nullptr, SLApplication::devRot.zeroYawAtStart()))
                    SLApplication::devRot.zeroYawAtStart(!SLApplication::devRot.zeroYawAtStart());

                if (ImGui::MenuItem("Reset Zero Yaw"))
                    SLApplication::devRot.hasStarted(true);

                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Location Sensor"))
            {
                if (ImGui::MenuItem("Use Device Location (GPS)", nullptr, SLApplication::devLoc.isUsed()))
                    SLApplication::devLoc.isUsed(!SLApplication::devLoc.isUsed());

                if (ImGui::MenuItem("Use Origin Altitude", nullptr, SLApplication::devLoc.useOriginAltitude()))
                    SLApplication::devLoc.useOriginAltitude(!SLApplication::devLoc.useOriginAltitude());

                if (ImGui::MenuItem("Reset Origin to here"))
                    SLApplication::devLoc.hasOrigin(false);

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
                        //make a guessed calibration, if there was a calibrated camera it is not valid anymore
                        ac->calibration = guessCalibration(ac->mirrorH(), ac->mirrorV(), ac->type());
                    }

                    if (ImGui::MenuItem("Vertically", nullptr, ac->mirrorV()))
                    {
                        ac->toggleMirrorV();
                        //make a guessed calibration, if there was a calibrated camera it is not valid anymore
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

                    if (ImGui::MenuItem("No Tangent Distortion", nullptr, SLApplication::calibrationEstimatorParams.zeroTangentDistortion))
                        SLApplication::calibrationEstimatorParams.toggleZeroTangentDist();

                    if (ImGui::MenuItem("Fix Aspect Ratio", nullptr, SLApplication::calibrationEstimatorParams.fixAspectRatio))
                        SLApplication::calibrationEstimatorParams.toggleFixAspectRatio();

                    if (ImGui::MenuItem("Fix Principal Point", nullptr, SLApplication::calibrationEstimatorParams.fixPrincipalPoint))
                        SLApplication::calibrationEstimatorParams.toggleFixPrincipalPoint();

                    if (ImGui::MenuItem("Use rational model", nullptr, SLApplication::calibrationEstimatorParams.calibRationalModel))
                        SLApplication::calibrationEstimatorParams.toggleRationalModel();

                    if (ImGui::MenuItem("Use tilted model", nullptr, SLApplication::calibrationEstimatorParams.calibTiltedModel))
                        SLApplication::calibrationEstimatorParams.toggleTiltedModel();

                    if (ImGui::MenuItem("Use thin prism model", nullptr, SLApplication::calibrationEstimatorParams.calibThinPrismModel))
                        SLApplication::calibrationEstimatorParams.toggleThinPrismModel();

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

        if (ImGui::BeginMenu("Renderer"))
        {
            if (ImGui::MenuItem("OpenGL (GL)", "G", rType == RT_gl))
                sv->renderType(RT_gl);

            if (ImGui::MenuItem("Ray Tracing (RT)", "R", rType == RT_rt))
                sv->startRaytracing(5);

            if (ImGui::MenuItem("Path Tracing (PT)", nullptr, rType == RT_pt))
                sv->startPathtracing(5, 10);

#if defined(GL_VERSION_4_4)
            if (glewIsSupported("GL_ARB_clear_texture GL_ARB_shader_image_load_store GL_ARB_texture_storage"))
            {
                if (ImGui::MenuItem("Cone Tracing (CT)", "C", rType == RT_ct))
                    sv->startConetracing();
            }
            else
#endif
            {
                if (ImGui::MenuItem("Cone Tracing (CT) (GL 4.4 or higher)", nullptr, rType == RT_ct, false))
                    sv->startConetracing();
            }

            ImGui::EndMenu();
        }

        if (rType == RT_gl)
        {
            if (ImGui::BeginMenu("GL"))
            {
                if (ImGui::MenuItem("Wired Mesh", "P", sv->drawBits()->get(SL_DB_WIREMESH)))
                    sv->drawBits()->toggle(SL_DB_WIREMESH);

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
                    sv->drawBits()->on(SL_DB_WIREMESH);
                    sv->drawBits()->on(SL_DB_NORMALS);
                    sv->drawBits()->on(SL_DB_VOXELS);
                    sv->drawBits()->on(SL_DB_AXIS);
                    sv->drawBits()->on(SL_DB_BBOX);
                    sv->drawBits()->on(SL_DB_SKELETON);
                    sv->drawBits()->on(SL_DB_CULLOFF);
                    sv->drawBits()->on(SL_DB_TEXOFF);
                }

                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.65f);
                SLfloat gamma = stateGL->gamma();
                if (ImGui::SliderFloat("Gamma", &gamma, 0.1f, 3.0f, "%.1f"))
                    stateGL->gamma(gamma);
                ImGui::PopItemWidth();

                ImGui::EndMenu();
            }
        }
        else if (rType == RT_rt)
        {
            if (ImGui::BeginMenu("RT"))
            {
                SLRaytracer* rt = sv->raytracer();

                if (ImGui::MenuItem("Parallel distributed", nullptr, rt->doDistributed()))
                {
                    rt->doDistributed(!rt->doDistributed());
                    sv->startRaytracing(rt->maxDepth());
                }

                if (ImGui::MenuItem("Continuously", nullptr, rt->doContinuous()))
                    rt->doContinuous(!rt->doContinuous());

                if (ImGui::MenuItem("Fresnel Reflection", nullptr, rt->doFresnel()))
                {
                    rt->doFresnel(!rt->doFresnel());
                    sv->startRaytracing(rt->maxDepth());
                }

                if (ImGui::BeginMenu("Max. Depth"))
                {
                    if (ImGui::MenuItem("1", nullptr, rt->maxDepth() == 1))
                        sv->startRaytracing(1);
                    if (ImGui::MenuItem("2", nullptr, rt->maxDepth() == 2))
                        sv->startRaytracing(2);
                    if (ImGui::MenuItem("3", nullptr, rt->maxDepth() == 3))
                        sv->startRaytracing(3);
                    if (ImGui::MenuItem("5", nullptr, rt->maxDepth() == 5))
                        sv->startRaytracing(5);
                    if (ImGui::MenuItem("Max. Contribution", nullptr, rt->maxDepth() == 0))
                        sv->startRaytracing(0);

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Anti-Aliasing Samples"))
                {
                    if (ImGui::MenuItem("Off", nullptr, rt->aaSamples() == 1))
                        rt->aaSamples(1);
                    if (ImGui::MenuItem("3x3", nullptr, rt->aaSamples() == 3))
                        rt->aaSamples(3);
                    if (ImGui::MenuItem("5x5", nullptr, rt->aaSamples() == 5))
                        rt->aaSamples(5);
                    if (ImGui::MenuItem("7x7", nullptr, rt->aaSamples() == 7))
                        rt->aaSamples(7);
                    if (ImGui::MenuItem("9x9", nullptr, rt->aaSamples() == 9))
                        rt->aaSamples(9);

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
        else if (rType == RT_pt)
        {
            if (ImGui::BeginMenu("PT"))
            {
                SLPathtracer* pt = sv->pathtracer();

                if (ImGui::BeginMenu("NO. of Samples"))
                {
                    if (ImGui::MenuItem("1", nullptr, pt->aaSamples() == 1))
                        sv->startPathtracing(5, 1);
                    if (ImGui::MenuItem("10", nullptr, pt->aaSamples() == 10))
                        sv->startPathtracing(5, 10);
                    if (ImGui::MenuItem("100", nullptr, pt->aaSamples() == 100))
                        sv->startPathtracing(5, 100);
                    if (ImGui::MenuItem("1000", nullptr, pt->aaSamples() == 1000))
                        sv->startPathtracing(5, 1000);
                    if (ImGui::MenuItem("10000", nullptr, pt->aaSamples() == 10000))
                        sv->startPathtracing(5, 10000);

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
                static SLfloat fov       = cam->fov();

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
                    for (SLint p = P_stereoSideBySide; p <= P_stereoColorYB; ++p)
                    {
                        SLstring pStr = SLCamera::projectionToStr((SLProjection)p);
                        if (ImGui::MenuItem(pStr.c_str(), nullptr, proj == (SLProjection)p))
                            cam->projection((SLProjection)p);
                    }

                    if (proj >= P_stereoSideBySide)
                    {
                        ImGui::Separator();
                        static SLfloat eyeSepar = cam->eyeSeparation();
                        if (ImGui::SliderFloat("Eye Sep.", &eyeSepar, 0.0f, focalDist / 10.f))
                            cam->eyeSeparation(eyeSepar);
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

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Animation", hasAnimations))
        {
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
            ImGui::MenuItem("Stats on Timing", nullptr, &showStatsTiming);
            ImGui::MenuItem("Stats on Scene", nullptr, &showStatsScene);
            ImGui::MenuItem("Stats on Video", nullptr, &showStatsVideo);
            ImGui::Separator();
            ImGui::MenuItem("Show Scenegraph", nullptr, &showSceneGraph);
            ImGui::MenuItem("Show Properties", nullptr, &showProperties);
            ImGui::MenuItem("Show Transform", nullptr, &showTransform);
            ImGui::Separator();
            ImGui::MenuItem("Infos on Device", nullptr, &showInfosDevice);
            ImGui::MenuItem("Infos on Sensors", nullptr, &showInfosSensors);
            if (SLApplication::sceneID == SID_VideoChristoffel)
            {
                ImGui::Separator();
                ImGui::MenuItem("Infos on Christoffel", nullptr, &showChristoffel);
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
//! Builds the scenegraph dialog once per frame
void AppDemoGui::buildSceneGraph(SLScene* s)
{
    ImGui::Begin("Scenegraph", &showSceneGraph);

    if (s->root3D())
        addSceneGraphNode(s, s->root3D());

    ImGui::End();
}
//-----------------------------------------------------------------------------
//! Builds the node information once per frame
void AppDemoGui::addSceneGraphNode(SLScene* s, SLNode* node)
{
    SLbool isSelectedNode = s->selectedNode() == node;
    SLbool isLeafNode     = node->children().empty() && node->meshes().empty();

    ImGuiTreeNodeFlags nodeFlags = 0;
    if (isLeafNode)
        nodeFlags |= ImGuiTreeNodeFlags_Leaf;
    else
        nodeFlags |= ImGuiTreeNodeFlags_OpenOnArrow;

    if (isSelectedNode)
        nodeFlags |= ImGuiTreeNodeFlags_Selected;

    bool nodeIsOpen = ImGui::TreeNodeEx(node->name().c_str(), nodeFlags);

    if (ImGui::IsItemClicked())
        s->selectNodeMesh(node, nullptr);

    if (nodeIsOpen)
    {
        for (auto mesh : node->meshes())
        {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));

            ImGuiTreeNodeFlags meshFlags = ImGuiTreeNodeFlags_Leaf;
            if (s->selectedMesh() == mesh)
                meshFlags |= ImGuiTreeNodeFlags_Selected;

            ImGui::TreeNodeEx(mesh, meshFlags, "%s", mesh->name().c_str());

            if (ImGui::IsItemClicked())
                s->selectNodeMesh(node, mesh);

            ImGui::TreePop();
            ImGui::PopStyleColor();
        }

        for (auto child : node->children())
            addSceneGraphNode(s, child);

        ImGui::TreePop();
    }
}
//-----------------------------------------------------------------------------
//! Builds the properties dialog once per frame
void AppDemoGui::buildProperties(SLScene* s)
{
    SLNode* node = s->selectedNode();
    SLMesh* mesh = s->selectedMesh();

    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);

    if (node && s->selectedRect().isEmpty())
    {

        ImGui::Begin("Properties of Selection", &showProperties);

        if (ImGui::TreeNode("Single Node Properties"))
        {
            if (node)
            {
                SLuint c = (SLuint)node->children().size();
                SLuint m = (SLuint)node->meshes().size();

                ImGui::Text("Node Name       : %s", node->name().c_str());
                ImGui::Text("No. of children : %u", c);
                ImGui::Text("No. of meshes   : %u", m);
                if (ImGui::TreeNode("Drawing Flags"))
                {
                    SLbool db;
                    db = node->drawBit(SL_DB_HIDDEN);
                    if (ImGui::Checkbox("Hide", &db))
                        node->drawBits()->set(SL_DB_HIDDEN, db);

                    db = node->drawBit(SL_DB_WIREMESH);
                    if (ImGui::Checkbox("Show wireframe", &db))
                        node->drawBits()->set(SL_DB_WIREMESH, db);

                    db = node->drawBit(SL_DB_NORMALS);
                    if (ImGui::Checkbox("Show normals", &db))
                        node->drawBits()->set(SL_DB_NORMALS, db);

                    db = node->drawBit(SL_DB_VOXELS);
                    if (ImGui::Checkbox("Show voxels", &db))
                        node->drawBits()->set(SL_DB_VOXELS, db);

                    db = node->drawBit(SL_DB_BBOX);
                    if (ImGui::Checkbox("Show bounding boxes", &db))
                        node->drawBits()->set(SL_DB_BBOX, db);

                    db = node->drawBit(SL_DB_AXIS);
                    if (ImGui::Checkbox("Show axis", &db))
                        node->drawBits()->set(SL_DB_AXIS, db);

                    db = node->drawBit(SL_DB_CULLOFF);
                    if (ImGui::Checkbox("Show back faces", &db))
                        node->drawBits()->set(SL_DB_CULLOFF, db);

                    db = node->drawBit(SL_DB_TEXOFF);
                    if (ImGui::Checkbox("No textures", &db))
                        node->drawBits()->set(SL_DB_TEXOFF, db);

                    ImGui::TreePop();
                }

                if (ImGui::TreeNode("Local Transform"))
                {
                    SLMat4f om(node->om());
                    SLVec3f t, r, s;
                    om.decompose(t, r, s);
                    r *= Utils::RAD2DEG;

                    ImGui::Text("Translation  : %s", t.toString().c_str());
                    ImGui::Text("Rotation     : %s", r.toString().c_str());
                    ImGui::Text("Scaling      : %s", s.toString().c_str());
                    ImGui::TreePop();
                }

                // Show special camera properties
                if (typeid(*node) == typeid(SLCamera))
                {
                    SLCamera* cam = (SLCamera*)node;

                    if (ImGui::TreeNode("Camera"))
                    {
                        SLfloat clipN     = cam->clipNear();
                        SLfloat clipF     = cam->clipFar();
                        SLfloat focalDist = cam->focalDist();
                        SLfloat fov       = cam->fov();

                        const char* projections[] = {"Mono Perspective",
                                                     "Mono Orthographic",
                                                     "Stereo Side By Side",
                                                     "Stereo Side By Side Prop.",
                                                     "Stereo Side By Side Dist.",
                                                     "Stereo Line By Line",
                                                     "Stereo Column By Column",
                                                     "Stereo Pixel By Pixel",
                                                     "Stereo Color Red Cyan",
                                                     "Stereo Color Red Green",
                                                     "Stereo Color Red Blue",
                                                     "Stereo Color Yelle Blue"};

                        int proj = cam->projection();
                        if (ImGui::Combo("Projection", &proj, projections, IM_ARRAYSIZE(projections)))
                            cam->projection((SLProjection)proj);

                        if (cam->projection() > P_monoOrthographic)
                        {
                            SLfloat eyeSepar = cam->eyeSeparation();
                            if (ImGui::SliderFloat("Eye Sep.", &eyeSepar, 0.0f, focalDist / 10.f))
                                cam->eyeSeparation(eyeSepar);
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
                if (typeid(*node) == typeid(SLLightSpot) ||
                    typeid(*node) == typeid(SLLightRect) ||
                    typeid(*node) == typeid(SLLightDirect))
                {
                    SLLight* light = nullptr;
                    SLstring typeName;
                    if (typeid(*node) == typeid(SLLightSpot))
                    {
                        light    = (SLLight*)(SLLightSpot*)node;
                        typeName = "Light (spot):";
                    }
                    if (typeid(*node) == typeid(SLLightRect))
                    {
                        light    = (SLLight*)(SLLightRect*)node;
                        typeName = "Light (rectangular):";
                    }
                    if (typeid(*node) == typeid(SLLightDirect))
                    {
                        light    = (SLLight*)(SLLightDirect*)node;
                        typeName = "Light (directional):";
                    }

                    if (light && ImGui::TreeNode(typeName.c_str()))
                    {
                        SLbool on = light->isOn();
                        if (ImGui::Checkbox("Is on", &on))
                            light->isOn(on);

                        ImGuiInputTextFlags flags = ImGuiInputTextFlags_EnterReturnsTrue;
                        SLCol4f             a     = light->ambient();
                        if (ImGui::InputFloat3("Ambient", (float*)&a, 1, flags))
                            light->ambient(a);

                        SLCol4f d = light->diffuse();
                        if (ImGui::InputFloat3("Diffuse", (float*)&d, 1, flags))
                            light->diffuse(d);

                        SLCol4f s = light->specular();
                        if (ImGui::InputFloat3("Specular", (float*)&s, 1, flags))
                            light->specular(s);

                        float cutoff = light->spotCutOffDEG();
                        if (ImGui::SliderFloat("Spot cut off angle", &cutoff, 0.0f, 180.0f))
                            light->spotCutOffDEG(cutoff);

                        float kc = light->kc();
                        if (ImGui::SliderFloat("Constant attenutation", &kc, 0.0f, 1.0f))
                            light->kc(kc);

                        float kl = light->kl();
                        if (ImGui::SliderFloat("Linear attenutation", &kl, 0.0f, 1.0f))
                            light->kl(kl);

                        float kq = light->kq();
                        if (ImGui::SliderFloat("Quadradic attenutation", &kq, 0.0f, 1.0f))
                            light->kq(kq);

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
        if (ImGui::TreeNode("Single Mesh Properties"))
        {
            if (mesh)
            {
                SLuint      v = (SLuint)mesh->P.size();
                SLuint      t = (SLuint)(!mesh->I16.empty() ? mesh->I16.size() / 3 : mesh->I32.size() / 3);
                SLMaterial* m = mesh->mat();
                ImGui::Text("Mesh Name       : %s", mesh->name().c_str());
                ImGui::Text("No. of Vertices : %u", v);
                ImGui::Text("No. of Triangles: %u", t);

                if (m && ImGui::TreeNode("Material"))
                {
                    ImGui::Text("Material Name: %s", m->name().c_str());

                    if (ImGui::TreeNode("Reflection colors"))
                    {
                        SLCol4f ac = m->ambient();
                        if (ImGui::ColorEdit3("Ambient color", (float*)&ac))
                            m->ambient(ac);

                        SLCol4f dc = m->diffuse();
                        if (ImGui::ColorEdit3("Diffuse color", (float*)&dc))
                            m->diffuse(dc);

                        SLCol4f sc = m->specular();
                        if (ImGui::ColorEdit3("Specular color", (float*)&sc))
                            m->specular(sc);

                        SLCol4f ec = m->emissive();
                        if (ImGui::ColorEdit3("Emissive color", (float*)&ec))
                            m->emissive(ec);

                        ImGui::TreePop();
                    }

                    if (ImGui::TreeNode("Other variables"))
                    {
                        ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.5f);

                        SLfloat shine = m->shininess();
                        if (ImGui::SliderFloat("Shininess", &shine, 0.0f, 1000.0f))
                            m->shininess(shine);

                        SLfloat rough = m->roughness();
                        if (ImGui::SliderFloat("Roughness", &rough, 0.0f, 1.0f))
                            m->roughness(rough);

                        SLfloat metal = m->metalness();
                        if (ImGui::SliderFloat("Metalness", &metal, 0.0f, 1.0f))
                            m->metalness(metal);

                        SLfloat kr = m->kr();
                        if (ImGui::SliderFloat("kr", &kr, 0.0f, 1.0f))
                            m->kr(kr);

                        SLfloat kt = m->kt();
                        if (ImGui::SliderFloat("kt", &kt, 0.0f, 1.0f))
                            m->kt(kt);

                        SLfloat kn = m->kn();
                        if (ImGui::SliderFloat("kn", &kn, 1.0f, 2.5f))
                            m->kn(kn);

                        ImGui::PopItemWidth();
                        ImGui::TreePop();
                    }

                    if (!m->textures().empty() && ImGui::TreeNode("Textures"))
                    {
                        ImGui::Text("No. of textures: %lu", m->textures().size());

                        //SLfloat lineH = ImGui::GetTextLineHeightWithSpacing();
                        SLfloat texW = ImGui::GetWindowWidth() - 4 * ImGui::GetTreeNodeToLabelSpacing() - 10;

                        for (SLulong i = 0; i < m->textures().size(); ++i)
                        {
                            SLGLTexture* t      = m->textures()[i];
                            void*        tid    = (ImTextureID)(intptr_t)t->texID();
                            SLfloat      w      = (SLfloat)t->width();
                            SLfloat      h      = (SLfloat)t->height();
                            SLfloat      h_to_w = h / w;

                            if (ImGui::TreeNode(t->name().c_str()))
                            {
                                ImGui::Text("Size    : %d x %d x %d", t->width(), t->height(), t->depth());
                                ImGui::Text("Type    : %s", t->typeName().c_str());

                                if (t->depth() > 1)
                                {
                                    if (t->target() == GL_TEXTURE_CUBE_MAP)
                                        ImGui::Text("Cube maps can not be displayed.");
                                    else if (t->target() == GL_TEXTURE_3D)
                                        ImGui::Text("3D textures can not be displayed.");
                                }
                                else
                                {
                                    if (typeid(*t) == typeid(SLTransferFunction))
                                    {
                                        SLTransferFunction* tf = (SLTransferFunction*)m->textures()[i];
                                        if (ImGui::TreeNode("Color Points in Transfer Function"))
                                        {
                                            for (SLulong c = 0; c < tf->colors().size(); ++c)
                                            {
                                                SLCol3f color = tf->colors()[c].color;
                                                SLchar  label[20];
                                                sprintf(label, "Color %lu", c);
                                                if (ImGui::ColorEdit3(label, (float*)&color))
                                                {
                                                    tf->colors()[c].color = color;
                                                    tf->generateTexture();
                                                }
                                                ImGui::SameLine();
                                                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.5f);
                                                sprintf(label, "Pos. %lu", c);
                                                SLfloat pos = tf->colors()[c].pos;
                                                if (c > 0 && c < tf->colors().size() - 1)
                                                {
                                                    SLfloat min = tf->colors()[c - 1].pos + 2.0f / (SLfloat)tf->length();
                                                    SLfloat max = tf->colors()[c + 1].pos - 2.0f / (SLfloat)tf->length();
                                                    if (ImGui::SliderFloat(label, &pos, min, max, "%3.2f"))
                                                    {
                                                        tf->colors()[c].pos = pos;
                                                        tf->generateTexture();
                                                    }
                                                }
                                                else
                                                    ImGui::Text("%3.2f Pos. %lu", pos, c);
                                                ImGui::PopItemWidth();
                                            }

                                            ImGui::TreePop();
                                        }

                                        if (ImGui::TreeNode("Alpha Points in Transfer Function"))
                                        {
                                            for (SLulong a = 0; a < tf->alphas().size(); ++a)
                                            {
                                                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.25f);
                                                SLfloat alpha = tf->alphas()[a].alpha;
                                                SLchar  label[20];
                                                sprintf(label, "Alpha %lu", a);
                                                if (ImGui::SliderFloat(label, &alpha, 0.0f, 1.0f, "%3.2f"))
                                                {
                                                    tf->alphas()[a].alpha = alpha;
                                                    tf->generateTexture();
                                                }
                                                ImGui::SameLine();
                                                sprintf(label, "Pos. %lu", a);
                                                SLfloat pos = tf->alphas()[a].pos;
                                                if (a > 0 && a < tf->alphas().size() - 1)
                                                {
                                                    SLfloat min = tf->alphas()[a - 1].pos + 2.0f / (SLfloat)tf->length();
                                                    SLfloat max = tf->alphas()[a + 1].pos - 2.0f / (SLfloat)tf->length();
                                                    if (ImGui::SliderFloat(label, &pos, min, max, "%3.2f"))
                                                    {
                                                        tf->alphas()[a].pos = pos;
                                                        tf->generateTexture();
                                                    }
                                                }
                                                else
                                                    ImGui::Text("%3.2f Pos. %lu", pos, a);

                                                ImGui::PopItemWidth();
                                            }

                                            ImGui::TreePop();
                                        }

                                        ImGui::Image(tid, ImVec2(texW, texW * 0.25f), ImVec2(0, 1), ImVec2(1, 0), ImVec4(1, 1, 1, 1), ImVec4(1, 1, 1, 1));

                                        SLVfloat allAlpha = tf->allAlphas();
                                        ImGui::PlotLines("", allAlpha.data(), (SLint)allAlpha.size(), 0, nullptr, 0.0f, 1.0f, ImVec2(texW, texW * 0.25f));
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

                                ImGui::TreePop();
                            }
                        }

                        ImGui::TreePop();
                    }

                    if (ImGui::TreeNode("GLSL Program"))
                    {
                        for (SLulong i = 0; i < m->program()->shaders().size(); ++i)
                        {
                            SLGLShader* s     = m->program()->shaders()[i];
                            SLfloat     lineH = ImGui::GetTextLineHeight();

                            if (ImGui::TreeNode(s->name().c_str()))
                            {
                                SLchar text[1024 * 16];
                                strcpy(text, s->code().c_str());
                                ImGui::InputTextMultiline(s->name().c_str(), text, IM_ARRAYSIZE(text), ImVec2(-1.0f, lineH * 16));
                                ImGui::TreePop();
                            }
                        }

                        ImGui::TreePop();
                    }

                    ImGui::TreePop();
                }
            }
            else
            {
                ImGui::Text("No single mesh selected.");
            }

            ImGui::TreePop();
        }

        ImGui::PopStyleColor();
        ImGui::End();
    }
    else if (!node && !s->selectedRect().isEmpty())
    {
        /* The selection rectangle is defined in SLScene::selectRect and gets set and
        drawn in SLCamera::onMouseDown and SLCamera::onMouseMove. If the selectRect is
        not empty the SLScene::selectedNode is null. All vertices that are within the
        selectRect are listed in SLMesh::IS32. The selection evaluation is done during
        drawing in SLMesh::draw and is only valid for the current frame.
        All nodes that have selected vertice have their drawbit SL_DB_SELECTED set. */

        vector<SLNode*> selectedNodes = s->root3D()->findChildren(SL_DB_SELECTED);

        ImGui::Begin("Properties of Selection", &showProperties);

        for (auto selectedNode : selectedNodes)
        {
            if (!selectedNode->meshes().empty())
            {
                ImGui::Text("Node: %s", selectedNode->name().c_str());
                for (auto selectedMesh : selectedNode->meshes())
                {
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
        }

        ImGui::End();
    }
    else
    { // Nothing is selected
        ImGui::Begin("Properties of Selection", &showProperties);
        ImGui::Text("There is nothing selected.");
        ImGui::Text("Please select a single node");
        ImGui::Text("by double-clicking or");
        ImGui::Text("select multiple nodes by");
        ImGui::Text("CTRL-LMB rectangle selection.");
        ImGui::End();
    }
    ImGui::PopFont();
}
//-----------------------------------------------------------------------------
//! Loads the UI configuration
void AppDemoGui::loadConfig(SLint dotsPerInch)
{
    ImGuiStyle& style               = ImGui::GetStyle();
    SLstring    fullPathAndFilename = SLApplication::configPath +
                                   SLApplication::name + ".yml";

    if (!Utils::fileExists(fullPathAndFilename))
    {
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

        // Adjust UI paddings on DPI
        style.FramePadding.x     = std::max(8.0f * dpiScaleFixed, 8.0f);
        style.WindowPadding.x    = style.FramePadding.x;
        style.FramePadding.y     = std::max(3.0f * dpiScaleFixed, 3.0f);
        style.ItemSpacing.x      = std::max(8.0f * dpiScaleFixed, 8.0f);
        style.ItemSpacing.y      = std::max(3.0f * dpiScaleFixed, 3.0f);
        style.ItemInnerSpacing.x = style.ItemSpacing.y;
        style.ScrollbarSize      = std::max(16.0f * dpiScaleFixed, 16.0f);
        style.ScrollbarRounding  = std::floor(style.ScrollbarSize / 2);

        return;
    }

    CVFileStorage fs;
    try
    {
        fs.open(fullPathAndFilename, CVFileStorage::READ);
        if (fs.isOpened())
        {
            // clang-format off
            SLint  i;
            SLbool b;
            fs["configTime"] >>             AppDemoGui::configTime;
            fs["fontPropDots"] >> i;        SLGLImGui::fontPropDots = (SLfloat)i;
            fs["fontFixedDots"] >> i;       SLGLImGui::fontFixedDots = (SLfloat)i;
            fs["ItemSpacingX"] >> i;        style.ItemSpacing.x = (SLfloat)i;
            fs["ItemSpacingY"] >> i;        style.ItemSpacing.y = (SLfloat)i;
            style.WindowPadding.x = style.FramePadding.x = style.ItemInnerSpacing.x = style.ItemSpacing.x;
            style.WindowPadding.y = style.FramePadding.y = style.ItemInnerSpacing.y = style.ItemSpacing.y;
            fs["ScrollbarSize"] >> i;
            style.ScrollbarSize = (SLfloat)i;
            fs["ScrollbarRounding"] >> i;
            style.ScrollbarRounding = (SLfloat)i;
            fs["sceneID"] >> i;             SLApplication::sceneID = (SLSceneID)i;
            fs["showInfosScene"] >> b;      AppDemoGui::showInfosScene = b;
            fs["showStatsTiming"] >> b;     AppDemoGui::showStatsTiming = b;
            fs["showStatsMemory"] >> b;     AppDemoGui::showStatsScene = b;
            fs["showStatsVideo"] >> b;      AppDemoGui::showStatsVideo = b;
            fs["showInfosFrameworks"] >> b; AppDemoGui::showInfosDevice = b;
            fs["showInfosSensors"] >> b;    AppDemoGui::showInfosSensors = b;
            fs["showSceneGraph"] >> b;      AppDemoGui::showSceneGraph = b;
            fs["showProperties"] >> b;      AppDemoGui::showProperties = b;
            fs["showChristoffel"] >> b;     AppDemoGui::showChristoffel = b;
            fs["showUIPrefs"] >> b;         AppDemoGui::showUIPrefs = b;
            // clang-format on

            fs.release();
            SL_LOG("Config. loaded  : %s", fullPathAndFilename.c_str());
            SL_LOG("Config. date    : %s", AppDemoGui::configTime.c_str());
            SL_LOG("fontPropDots    : %f", SLGLImGui::fontPropDots);
            SL_LOG("fontFixedDots   : %f", SLGLImGui::fontFixedDots);
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
    SLstring    fullPathAndFilename = SLApplication::configPath +
                                   SLApplication::name + ".yml";

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
    fs << "sceneID" << (SLint)SLApplication::sceneID;
    fs << "ItemSpacingX" << (SLint)style.ItemSpacing.x;
    fs << "ItemSpacingY" << (SLint)style.ItemSpacing.y;
    fs << "ScrollbarSize" << (SLfloat)style.ScrollbarSize;
    fs << "ScrollbarRounding" << (SLfloat)style.ScrollbarRounding;
    fs << "showStatsTiming" << AppDemoGui::showStatsTiming;
    fs << "showStatsMemory" << AppDemoGui::showStatsScene;
    fs << "showStatsVideo" << AppDemoGui::showStatsVideo;
    fs << "showInfosFrameworks" << AppDemoGui::showInfosDevice;
    fs << "showInfosScene" << AppDemoGui::showInfosScene;
    fs << "showInfosSensors" << AppDemoGui::showInfosSensors;
    fs << "showSceneGraph" << AppDemoGui::showSceneGraph;
    fs << "showProperties" << AppDemoGui::showProperties;
    fs << "showChristoffel" << AppDemoGui::showChristoffel;
    fs << "showUIPrefs" << AppDemoGui::showUIPrefs;

    fs.release();
    SL_LOG("Config. saved   : %s", fullPathAndFilename.c_str());
}
//-----------------------------------------------------------------------------
