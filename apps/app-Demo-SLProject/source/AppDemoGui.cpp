//#############################################################################
//  File:      AppDemoGui.cpp
//  Purpose:   UI with the ImGUI framework fully rendered in OpenGL 3+
//  Author:    Marcus Hudritsch
//  Date:      Summer 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
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
#include <SLAverageTiming.h>
#include <SLCVCapture.h>
#include <SLCVImage.h>
#include <SLCVTrackedFeatures.h>
#include <SLGLProgram.h>
#include <SLGLShader.h>
#include <SLGLTexture.h>
#include <SLImGuiInfosDialog.h>
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

#include <imgui.h>
#include <imgui_internal.h>

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
    SLfloat width  = (SLfloat)sv->scrW() * widthPC;
    SLfloat height = (SLfloat)sv->scrH() * heightPC;
    ImGui::SetNextWindowSize(ImVec2(width, height), ImGuiSetCond_Always);
    ImGui::SetNextWindowPosCenter(ImGuiSetCond_Always);
}
//-----------------------------------------------------------------------------
// Init global static variables
SLGLTexture* AppDemoGui::cpvrLogo             = nullptr;
SLstring     AppDemoGui::configTime           = "-";
SLbool       AppDemoGui::showAbout            = false;
SLbool       AppDemoGui::showHelp             = false;
SLbool       AppDemoGui::showHelpCalibration  = false;
SLbool       AppDemoGui::showCredits          = false;
SLbool       AppDemoGui::showStatsTiming      = false;
SLbool       AppDemoGui::showStatsScene       = false;
SLbool       AppDemoGui::showStatsVideo       = false;
SLbool       AppDemoGui::showInfosFrameworks  = false;
SLbool       AppDemoGui::showInfosScene       = false;
SLbool       AppDemoGui::showInfosSensors     = false;
SLbool       AppDemoGui::showSceneGraph       = false;
SLbool       AppDemoGui::showProperties       = false;
SLbool       AppDemoGui::showStatsDebugTiming = false;
SLbool       AppDemoGui::showChristoffel      = false;
SLbool       AppDemoGui::showUIPrefs          = false;
SLbool       AppDemoGui::showTransform        = false;

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

map<string, std::shared_ptr<SLImGuiInfosDialog>> AppDemoGui::_infoDialogs;

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
  "Contributors since 2005 in alphabetic order: Martin Christen, Jan Dellsperger, \
Manuel Frischknecht, Michael Goettlicher, Timo Tschanz, Marc Wacker, Pascal Zingg \n\n\
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

    if (showAbout)
    {
        if (cpvrLogo == nullptr)
        {
            // The texture resources get deleted by the SLScene destructor
            cpvrLogo = new SLGLTexture("LogoCPVR_256L.png");
            if (cpvrLogo != nullptr)
                cpvrLogo->bindActive();
        }
        else
            cpvrLogo->bindActive();

        SLfloat iconSize = sv->scrW() * 0.15f;

        centerNextWindow(sv);
        ImGui::Begin("About SLProject", &showAbout, ImGuiWindowFlags_NoResize);
        ImGui::Image((ImTextureID)(intptr_t)cpvrLogo->texName(), ImVec2(iconSize, iconSize), ImVec2(0, 1), ImVec2(1, 0));
        ImGui::SameLine();
        ImGui::Text("Version: %s", SLApplication::version.c_str());
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
            // Get averages from average variables (see SLAverage)
            SLfloat captureTime  = s->captureTimesMS().average();
            SLfloat updateTime   = s->updateTimesMS().average();
            SLfloat trackingTime = s->trackingTimesMS().average();
            SLfloat detectTime   = s->detectTimesMS().average();
            SLfloat detect1Time  = s->detect1TimesMS().average();
            SLfloat detect2Time  = s->detect2TimesMS().average();
            SLfloat matchTime    = s->matchTimesMS().average();
            SLfloat optFlowTime  = s->optFlowTimesMS().average();
            SLfloat poseTime     = s->poseTimesMS().average();
            SLfloat draw3DTime   = s->draw3DTimesMS().average();
            SLfloat draw2DTime   = s->draw2DTimesMS().average();
            SLfloat cullTime     = s->cullTimesMS().average();

            // Calculate percentage from frame time
            SLfloat captureTimePC  = SL_clamp(captureTime / ft * 100.0f, 0.0f, 100.0f);
            SLfloat updateTimePC   = SL_clamp(updateTime / ft * 100.0f, 0.0f, 100.0f);
            SLfloat trackingTimePC = SL_clamp(trackingTime / ft * 100.0f, 0.0f, 100.0f);
            SLfloat detectTimePC   = SL_clamp(detectTime / ft * 100.0f, 0.0f, 100.0f);
            SLfloat matchTimePC    = SL_clamp(matchTime / ft * 100.0f, 0.0f, 100.0f);
            SLfloat optFlowTimePC  = SL_clamp(optFlowTime / ft * 100.0f, 0.0f, 100.0f);
            SLfloat poseTimePC     = SL_clamp(poseTime / ft * 100.0f, 0.0f, 100.0f);
            SLfloat draw3DTimePC   = SL_clamp(draw3DTime / ft * 100.0f, 0.0f, 100.0f);
            SLfloat draw2DTimePC   = SL_clamp(draw2DTime / ft * 100.0f, 0.0f, 100.0f);
            SLfloat cullTimePC     = SL_clamp(cullTime / ft * 100.0f, 0.0f, 100.0f);

            sprintf(m + strlen(m), "Renderer      : OpenGL\n");
            sprintf(m + strlen(m), "Frame size    : %d x %d\n", sv->scrW(), sv->scrH());
            sprintf(m + strlen(m), "NO. drawcalls : %d\n", SLGLVertexArray::totalDrawCalls);
            sprintf(m + strlen(m), "Frames per s. : %4.1f\n", s->fps());
            sprintf(m + strlen(m), "Frame time    : %4.1f ms (100%%)\n", ft);
            sprintf(m + strlen(m), "  Capture     : %4.1f ms (%3d%%)\n", captureTime, (SLint)captureTimePC);
            sprintf(m + strlen(m), "  Update      : %4.1f ms (%3d%%)\n", updateTime, (SLint)updateTimePC);
            sprintf(m + strlen(m), "    Tracking  : %4.1f ms (%3d%%)\n", trackingTime, (SLint)trackingTimePC);
            sprintf(m + strlen(m), "      Detect  : %4.1f ms (%3d%%)\n", detectTime, (SLint)detectTimePC);
            sprintf(m + strlen(m), "        Det1  : %4.1f ms\n", detect1Time);
            sprintf(m + strlen(m), "        Det2  : %4.1f ms\n", detect2Time);
            sprintf(m + strlen(m), "      Match   : %4.1f ms (%3d%%)\n", matchTime, (SLint)matchTimePC);
            sprintf(m + strlen(m), "      Opt.Flow: %4.1f ms (%3d%%)\n", optFlowTime, (SLint)optFlowTimePC);
            sprintf(m + strlen(m), "      Pose    : %4.1f ms (%3d%%)\n", poseTime, (SLint)poseTimePC);
            sprintf(m + strlen(m), "  Culling     : %4.1f ms (%3d%%)\n", cullTime, (SLint)cullTimePC);
            sprintf(m + strlen(m), "  Drawing 3D  : %4.1f ms (%3d%%)\n", draw3DTime, (SLint)draw3DTimePC);
            sprintf(m + strlen(m), "  Drawing 2D  : %4.1f ms (%3d%%)\n", draw2DTime, (SLint)draw2DTimePC);
        }
        else if (rType == RT_rt)
        {
            SLRaytracer* rt           = sv->raytracer();
            SLuint       rayPrimaries = (SLuint)(sv->scrW() * sv->scrH());
            SLuint       rayTotal     = rayPrimaries + SLRay::reflectedRays + SLRay::subsampledRays + SLRay::refractedRays + SLRay::shadowRays;
            SLfloat      rpms         = rt->renderSec() > 0.0f ? rayTotal / rt->renderSec() / 1000.0f : 0.0f;

            sprintf(m + strlen(m), "Renderer      : Ray Tracer\n");
            sprintf(m + strlen(m), "Frame size    : %d x %d\n", sv->scrW(), sv->scrH());
            sprintf(m + strlen(m), "Frames per s. : %0.2f\n", 1.0f / rt->renderSec());
            sprintf(m + strlen(m), "Frame Time    : %0.2f sec.\n", rt->renderSec());
            sprintf(m + strlen(m), "Rays per ms   : %0.0f\n", rpms);
            sprintf(m + strlen(m), "AA Pixels     : %d (%d%%)\n", SLRay::subsampledPixels, (int)((float)SLRay::subsampledPixels / (float)rayPrimaries * 100.0f));
            sprintf(m + strlen(m), "Threads       : %d\n", rt->numThreads());
            sprintf(m + strlen(m), "-------------------------------\n");
            sprintf(m + strlen(m), "Primary rays  : %8d (%3d%%)\n", rayPrimaries, (int)((float)rayPrimaries / (float)rayTotal * 100.0f));
            sprintf(m + strlen(m), "Reflected rays: %8d (%3d%%)\n", SLRay::reflectedRays, (int)((float)SLRay::reflectedRays / (float)rayTotal * 100.0f));
            sprintf(m + strlen(m), "Refracted rays: %8d (%3d%%)\n", SLRay::refractedRays, (int)((float)SLRay::refractedRays / (float)rayTotal * 100.0f));
            sprintf(m + strlen(m), "TIR rays      : %8d\n", SLRay::tirRays);
            sprintf(m + strlen(m), "Shadow rays   : %8d (%3d%%)\n", SLRay::shadowRays, (int)((float)SLRay::shadowRays / (float)rayTotal * 100.0f));
            sprintf(m + strlen(m), "AA rays       : %8d (%3d%%)\n", SLRay::subsampledRays, (int)((float)SLRay::subsampledRays / (float)rayTotal * 100.0f));
            sprintf(m + strlen(m), "Total rays    : %8d (%3d%%)\n", rayTotal, 100);
            sprintf(m + strlen(m), "-------------------------------\n");
            sprintf(m + strlen(m), "Maximum depth : %u\n", SLRay::maxDepthReached);
            sprintf(m + strlen(m), "Average depth : %0.3f\n", SLRay::avgDepth / rayPrimaries);
        }
        else if (rType == RT_rt)
        {
            SLRaytracer* rt           = sv->raytracer();
            SLuint       rayPrimaries = sv->scrW() * sv->scrH();
            SLuint       rayTotal     = rayPrimaries + SLRay::reflectedRays + SLRay::subsampledRays + SLRay::refractedRays + SLRay::shadowRays;
            SLfloat      rpms         = rt->renderSec() ? rayTotal / rt->renderSec() / 1000.0f : 0.0f;

            sprintf(m + strlen(m), "Renderer      : Ray Tracer\n");
            sprintf(m + strlen(m), "Frame size    : %d x %d\n", sv->scrW(), sv->scrH());
            sprintf(m + strlen(m), "Frames per s. : %0.2f\n", 1.0f / rt->renderSec());
            sprintf(m + strlen(m), "Frame Time    : %0.2f sec.\n", rt->renderSec());
            sprintf(m + strlen(m), "Rays per ms   : %0.0f\n", rpms);
            sprintf(m + strlen(m), "Threads       : %d\n", rt->numThreads());
            sprintf(m + strlen(m), "-------------------------------\n");
            sprintf(m + strlen(m), "Primary rays  : %8d (%3d%%)\n", rayPrimaries, (int)((float)rayPrimaries / (float)rayTotal * 100.0f));
            sprintf(m + strlen(m), "Reflected rays: %8d (%3d%%)\n", SLRay::reflectedRays, (int)((float)SLRay::reflectedRays / (float)rayTotal * 100.0f));
            sprintf(m + strlen(m), "Refracted rays: %8d (%3d%%)\n", SLRay::refractedRays, (int)((float)SLRay::refractedRays / (float)rayTotal * 100.0f));
            sprintf(m + strlen(m), "TIR rays      : %8d\n", SLRay::tirRays);
            sprintf(m + strlen(m), "Shadow rays   : %8d (%3d%%)\n", SLRay::shadowRays, (int)((float)SLRay::shadowRays / (float)rayTotal * 100.0f));
            sprintf(m + strlen(m), "AA rays       : %8d (%3d%%)\n", SLRay::subsampledRays, (int)((float)SLRay::subsampledRays / (float)rayTotal * 100.0f));
            sprintf(m + strlen(m), "Total rays    : %8d (%3d%%)\n", rayTotal, 100);
            sprintf(m + strlen(m), "-------------------------------\n");
            sprintf(m + strlen(m), "Maximum depth : %u\n", SLRay::maxDepthReached);
            sprintf(m + strlen(m), "Average depth : %0.3f\n", SLRay::avgDepth / rayPrimaries);
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
        SLint        numOpaqueNodes  = (int)sv->visibleNodes()->size();
        SLint        numBlendedNodes = (int)sv->blendNodes()->size();
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
        sprintf(m + strlen(m), "No. of Nodes    : %5d (100%%)\n", stats3D.numNodes);
        sprintf(m + strlen(m), "- Group Nodes   : %5d (%3d%%)\n", stats3D.numGroupNodes, numGroupPC);
        sprintf(m + strlen(m), "- Leaf  Nodes   : %5d (%3d%%)\n", stats3D.numLeafNodes, numLeafPC);
        sprintf(m + strlen(m), "- Light Nodes   : %5d (%3d%%)\n", stats3D.numLights, numLightsPC);
        sprintf(m + strlen(m), "- Opaque Nodes  : %5d (%3d%%)\n", numOpaqueNodes, numOpaquePC);
        sprintf(m + strlen(m), "- Blended Nodes : %5d (%3d%%)\n", numBlendedNodes, numBlendedPC);
        sprintf(m + strlen(m), "- Visible Nodes : %5d (%3d%%)\n", numVisibleNodes, numVisiblePC);
        sprintf(m + strlen(m), "- WM Updates    : %5d\n", SLNode::numWMUpdates);
        sprintf(m + strlen(m), "No. of Meshes   : %5u\n", stats3D.numMeshes);
        sprintf(m + strlen(m), "No. of Triangles: %5u\n", stats3D.numTriangles);
        sprintf(m + strlen(m), "CPU MB in Total : %6.2f (100%%)\n", cpuMBTotal);
        sprintf(m + strlen(m), "-   MB in Tex.  : %6.2f (%3d%%)\n", cpuMBTexture, cpuMBTexturePC);
        sprintf(m + strlen(m), "-   MB in Meshes: %6.2f (%3d%%)\n", cpuMBMeshes, cpuMBMeshesPC);
        sprintf(m + strlen(m), "-   MB in Voxels: %6.2f (%3d%%)\n", cpuMBVoxels, cpuMBVoxelsPC);
        sprintf(m + strlen(m), "GPU MB in Total : %6.2f (100%%)\n", gpuMBTotal);
        sprintf(m + strlen(m), "-   MB in Tex.  : %6.2f (%3d%%)\n", gpuMBTexture, gpuMBTexturePC);
        sprintf(m + strlen(m), "-   MB in VBO   : %6.2f (%3d%%)\n", gpuMBVbo, gpuMBVboPC);

        sprintf(m + strlen(m), "No. of Voxels   : %d\n", stats3D.numVoxels);
        sprintf(m + strlen(m), "- empty Voxels  : %4.1f%%\n", voxelsEmpty);
        sprintf(m + strlen(m), "Avg. Tria/Voxel : %4.1f\n", avgTriPerVox);
        sprintf(m + strlen(m), "Max. Tria/Voxel : %d\n", stats3D.numVoxMaxTria);

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

        SLCVCalibration* c        = SLApplication::activeCalib;
        SLCVSize         capSize  = SLCVCapture::captureSize;
        SLVideoType      vt       = s->videoType();
        SLstring         mirrored = "None";
        if (c->isMirroredH() && c->isMirroredV())
            mirrored = "horizontally & vertically";
        else if (c->isMirroredH())
            mirrored = "horizontally";
        else if (c->isMirroredV())
            mirrored = "vertically";

        sprintf(m + strlen(m), "Video Type    : %s\n", vt == VT_NONE ? "None" : vt == VT_MAIN ? "Main Camera" : vt == VT_FILE ? "File" : "Secondary Camera");
        sprintf(m + strlen(m), "Display size  : %d x %d\n", SLCVCapture::lastFrame.cols, SLCVCapture::lastFrame.rows);
        sprintf(m + strlen(m), "Capture size  : %d x %d\n", capSize.width, capSize.height);
        sprintf(m + strlen(m), "Requested size: %d\n", SLCVCapture::requestedSizeIndex);
        sprintf(m + strlen(m), "Mirrored      : %s\n", mirrored.c_str());
        sprintf(m + strlen(m), "Undistorted   : %s\n", c->showUndistorted() && c->state() == CS_calibrated ? "Yes" : "No");
        sprintf(m + strlen(m), "FOV (deg.)    : %4.1f\n", c->cameraFovDeg());
        sprintf(m + strlen(m), "fx,fy,cx,cy   : %4.1f,%4.1f,%4.1f,%4.1f\n", c->fx(), c->fy(), c->cx(), c->cy());
        sprintf(m + strlen(m), "k1,k2,p1,p2   : %4.2f,%4.2f,%4.2f,%4.2f\n", c->k1(), c->k2(), c->p1(), c->p2());
        sprintf(m + strlen(m), "Calib. time   : %s\n", c->calibrationTime().c_str());
        sprintf(m + strlen(m), "Calib. file   : %s\n", c->calibFileName().c_str());
        sprintf(m + strlen(m), "Calib. state  : %s\n", c->stateStr().c_str());

        if (vt != VT_NONE && s->trackers().size() > 0)
        {
            sprintf(m + strlen(m), "--------------:\n");
            for (auto tracker : s->trackers())
            {
                SLNode* node = tracker->node();
                if (node)
                {
                    if (typeid(*node) == typeid(SLCamera))
                    {
                        SLVec3f cameraPos = tracker->node()->updateAndGetWM().translation();
                        sprintf(m + strlen(m), "Dist. to zero : %4.2f (%s)\n", cameraPos.length(), node->name().c_str());
                    }
                    else
                    {
                        SLVec3f cameraPos = ((SLNode*)sv->camera())->updateAndGetWM().translation();
                        SLVec3f objectPos = tracker->node()->updateAndGetWM().translation();
                        SLVec3f camToObj  = objectPos - cameraPos;
                        sprintf(m + strlen(m), "Dist. to obj. : %4.2f (%s)\n", camToObj.length(), node->name().c_str());
                    }
                }
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
        SLfloat  w    = (SLfloat)sv->scrW();
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
            SLfloat                 s1 = 1.1f, s2 = 2.f, s3 = 10.f;   // Scale factors

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
            if (ImGui::Button("<<<##S")) 
                node->scale( s3); ImGui::SameLine();
            if (ImGui::Button("<<##S"))  
                node->scale( s2); ImGui::SameLine();
            if (ImGui::Button("<##S"))   
                node->scale( s1); ImGui::SameLine();
            if (ImGui::Button(">##S"))   
                node->scale(1/s1); ImGui::SameLine();
            if (ImGui::Button(">>##S"))  
                node->scale(1/s2); ImGui::SameLine();
            if (ImGui::Button(">>>##S")) 
                node->scale(1/s3);
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

    if (showInfosFrameworks)
    {
        SLGLState* stateGL = SLGLState::getInstance();
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
        ImGui::Begin("Framework Informations", &showInfosFrameworks, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
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
        sprintf(m + strlen(m), "Orientation Pitch   : %1.0f\n", SLApplication::devRot.pitchRAD() * SL_RAD2DEG);
        sprintf(m + strlen(m), "Orientation Yaw     : %1.0f\n", SLApplication::devRot.yawRAD() * SL_RAD2DEG);
        sprintf(m + strlen(m), "Orientation Roll    : %1.0f\n", SLApplication::devRot.rollRAD() * SL_RAD2DEG);
        sprintf(m + strlen(m), "Zero Yaw at Start   : %s\n", SLApplication::devRot.zeroYawAtStart() ? "yes" : "no");
        sprintf(m + strlen(m), "Start Yaw           : %1.0f\n", SLApplication::devRot.startYawRAD() * SL_RAD2DEG);
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
            SLFileSystem::deleteFile(fullPathFilename);
            loadConfig(SLApplication::dpi);
        }
        ImGui::PopFont();

        ImGui::End();
    }

    if (showStatsDebugTiming)
    {
        buildStatsDebugTiming(s, sv);
    }

    //add scene specifc info dialogs
    buildInfosDialogs();
}
//-----------------------------------------------------------------------------
void AppDemoGui::addInfoDialog(const shared_ptr<SLImGuiInfosDialog>& dialog)
{
    string name = string(dialog->getName());
    if (_infoDialogs.find(name) == _infoDialogs.end())
    {
        _infoDialogs[name] = dialog;
    }
}
//-----------------------------------------------------------------------------
void AppDemoGui::clearInfoDialogs()
{
    _infoDialogs.clear();
}
//-----------------------------------------------------------------------------
void AppDemoGui::buildMenuBar(SLScene* s, SLSceneView* sv)
{
    SLSceneID    sid           = SLApplication::sceneID;
    SLRenderType rType         = sv->renderType();
    SLbool       hasAnimations = (s->animManager().allAnimNames().size() > 0);
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
                    SLstring large1          = SLImporter::defaultPath + "PLY/xyzrgb_dragon.ply";
                    SLstring large2          = SLImporter::defaultPath + "PLY/mesh_zermatt.ply";
                    SLstring large3          = SLImporter::defaultPath + "PLY/switzerland.ply";
                    SLbool   largeFileExists = SLFileSystem::fileExists(large1) ||
                                             SLFileSystem::fileExists(large2) ||
                                             SLFileSystem::fileExists(large3);

                    if (ImGui::MenuItem("Minimal Scene", nullptr, sid == SID_Minimal))
                        s->onLoad(s, sv, SID_Minimal);
                    if (ImGui::MenuItem("Figure Scene", nullptr, sid == SID_Figure))
                        s->onLoad(s, sv, SID_Figure);
                    if (ImGui::MenuItem("Large Model", nullptr, sid == SID_LargeModel, largeFileExists))
                        s->onLoad(s, sv, SID_LargeModel);
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
                    if (ImGui::MenuItem("Track ArUco Marker (Scnd)", nullptr, sid == SID_VideoTrackArucoScnd, SLCVCapture::hasSecondaryCamera))
                        s->onLoad(s, sv, SID_VideoTrackArucoScnd);
                    if (ImGui::MenuItem("Track Chessboard (Main)", nullptr, sid == SID_VideoTrackChessMain))
                        s->onLoad(s, sv, SID_VideoTrackChessMain);
                    if (ImGui::MenuItem("Track Chessboard (Scnd)", nullptr, sid == SID_VideoTrackChessScnd, SLCVCapture::hasSecondaryCamera))
                        s->onLoad(s, sv, SID_VideoTrackChessScnd);
                    if (ImGui::MenuItem("Track Features (Main)", nullptr, sid == SID_VideoTrackFeature2DMain))
                        s->onLoad(s, sv, SID_VideoTrackFeature2DMain);
                    if (ImGui::MenuItem("Track Face (Main)", nullptr, sid == SID_VideoTrackFaceMain))
                        s->onLoad(s, sv, SID_VideoTrackFaceMain);
                    if (ImGui::MenuItem("Track Face (Scnd)", nullptr, sid == SID_VideoTrackFaceScnd, SLCVCapture::hasSecondaryCamera))
                        s->onLoad(s, sv, SID_VideoTrackFaceScnd);
                    if (ImGui::MenuItem("Sensor AR (Main)", nullptr, sid == SID_VideoSensorAR))
                        s->onLoad(s, sv, SID_VideoSensorAR);
                    if (ImGui::MenuItem("Christoffel Tower AR (Main)", nullptr, sid == SID_VideoChristoffel))
                        s->onLoad(s, sv, SID_VideoChristoffel);
                    if (ImGui::MenuItem("Christoffel Tower AR ORB-SLAM", 0, sid == SID_VideoChristoffelOrbSlam))
                        s->onLoad(s, sv, SID_VideoChristoffelOrbSlam);
                    if (ImGui::MenuItem("Orb SLAM video file mapping example", 0, sid == sid == SID_VideoFilesMapping))
                        s->onLoad(s, sv, SID_VideoFilesMapping);
                    if (ImGui::MenuItem("Orb SLAM mapping Burgplatz Biel", 0, sid == sid == SID_VideoMappingBurgplatz))
                        s->onLoad(s, sv, SID_VideoMappingBurgplatz);
                    if (ImGui::MenuItem("Orb SLAM tracking Burgplatz Biel", 0, sid == sid == SID_VideoTrackingBurgplatz))
                        s->onLoad(s, sv, SID_VideoTrackingBurgplatz);
                    if (ImGui::MenuItem("Orb SLAM mapping example", 0, sid == sid == SID_VideoMapping))
                        s->onLoad(s, sv, SID_VideoMapping);
                    if (ImGui::MenuItem("Track Features from Keyframes", 0, sid == sid == SID_VideoTrackKeyFrames))
                        s->onLoad(s, sv, SID_VideoTrackKeyFrames);
                    if (ImGui::MenuItem("Track Features from Keyframes in Video Files", 0, sid == sid == SID_VideoFilesTrackKeyFrames))
                        s->onLoad(s, sv, SID_VideoFilesTrackKeyFrames);

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

            ImGui::Separator();

            if (ImGui::BeginMenu("Video"))
            {
                SLCVCalibration* ac = SLApplication::activeCalib;
                SLCVCalibration* mc = &SLApplication::calibMainCam;
                SLCVCalibration* sc = &SLApplication::calibScndCam;

                SLCVTrackedFeatures* featureTracker = nullptr;
                for (auto tracker : s->trackers())
                {
                    if (typeid(*tracker) == typeid(SLCVTrackedFeatures))
                    {
                        featureTracker = (SLCVTrackedFeatures*)tracker;
                        break;
                    }
                }

                if (ImGui::BeginMenu("Mirror Main Camera"))
                {
                    if (ImGui::MenuItem("Horizontally", nullptr, mc->isMirroredH()))
                        mc->toggleMirrorH();

                    if (ImGui::MenuItem("Vertically", nullptr, mc->isMirroredV()))
                        mc->toggleMirrorV();

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Mirror Scnd. Camera", SLCVCapture::hasSecondaryCamera))
                {
                    if (ImGui::MenuItem("Horizontally", nullptr, sc->isMirroredH()))
                        sc->toggleMirrorH();

                    if (ImGui::MenuItem("Vertically", nullptr, sc->isMirroredV()))
                        sc->toggleMirrorV();

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Calibration"))
                {
                    if (ImGui::MenuItem("Start Calibration on Main Camera"))
                    {
                        s->onLoad(s, sv, SID_VideoCalibrateMain);
                        showHelpCalibration = true;
                        showInfosScene      = true;
                    }

                    if (ImGui::MenuItem("Start Calibration on Scnd. Camera", nullptr, false, SLCVCapture::hasSecondaryCamera))
                    {
                        s->onLoad(s, sv, SID_VideoCalibrateScnd);
                        showHelpCalibration = true;
                        showInfosScene      = true;
                    }

                    if (ImGui::MenuItem("Undistort Image", nullptr, ac->showUndistorted(), ac->state() == CS_calibrated))
                        ac->showUndistorted(!ac->showUndistorted());

                    if (ImGui::MenuItem("Zero Tangent Distortion", nullptr, ac->calibZeroTangentDist()))
                        ac->toggleZeroTangentDist();

                    if (ImGui::MenuItem("Fix Aspect Ratio", nullptr, ac->calibFixAspectRatio()))
                        ac->toggleFixAspectRatio();

                    if (ImGui::MenuItem("Fix Principal Point", nullptr, ac->calibFixPrincipalPoint()))
                        ac->toggleFixPrincipalPoint();

                    ImGui::EndMenu();
                }

                if (ImGui::MenuItem("Show Tracking Detection", nullptr, s->showDetection()))
                    s->showDetection(!s->showDetection());

                if (ImGui::BeginMenu("Feature Tracking", featureTracker != nullptr))
                {
                    if (ImGui::MenuItem("Force Relocation", nullptr, featureTracker->forceRelocation()))
                        featureTracker->forceRelocation(!featureTracker->forceRelocation());

                    if (ImGui::BeginMenu("Detector/Descriptor", featureTracker != nullptr))
                    {
                        SLCVDetectDescribeType type = featureTracker->type();

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

            ImGui::EndMenu();
        }

        if (rType == RT_gl)
        {
            if (ImGui::BeginMenu("GL-Setting"))
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

                ImGui::EndMenu();
            }
        }
        else if (rType == RT_rt)
        {
            if (ImGui::BeginMenu("RT-Settings"))
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
                    sv->raytracer()->saveImage();

                ImGui::EndMenu();
            }
        }
        else if (rType == RT_pt)
        {
            if (ImGui::BeginMenu("PT-Settings"))
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

                if (ImGui::MenuItem("Apply Gamma Corr.", nullptr, pt->applyGamma()))
                {
                    pt->applyGamma(!pt->applyGamma());
                    sv->startPathtracing(5, 10);
                }

                if (ImGui::MenuItem("Save Rendered Image"))
                    sv->pathtracer()->saveImage();

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

                if (ImGui::SliderFloat("Far Clip", &clipF, clipN, SL_min(clipF * 1.1f, 1000000.f)))
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
                    if (ImGui::SliderFloat("Walk Speed", &ms, 0.01f, SL_min(ms * 1.1f, 10000.f)))
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

            if (SLAverageTiming::instance().size())
            {
                ImGui::MenuItem("Stats on Debug Time", nullptr, &showStatsDebugTiming);
            }
            ImGui::MenuItem("Stats on Scene", nullptr, &showStatsScene);
            ImGui::MenuItem("Stats on Video", nullptr, &showStatsVideo);
            ImGui::Separator();
            ImGui::MenuItem("Show Scenegraph", nullptr, &showSceneGraph);
            ImGui::MenuItem("Show Properties", nullptr, &showProperties);
            ImGui::MenuItem("Show Transform", nullptr, &showTransform);
            ImGui::Separator();
            ImGui::MenuItem("Infos on Sensors", nullptr, &showInfosSensors);
            ImGui::MenuItem("Infos on Frameworks", nullptr, &showInfosFrameworks);

            for (auto dialog : _infoDialogs)
            {
                ImGui::Separator();
                ImGui::MenuItem(dialog.second->getName(), nullptr, &dialog.second->show);
            }

            ImGui::Separator();
            ImGui::MenuItem("Help on Interaction", nullptr, &showHelp);
            ImGui::MenuItem("Help on Calibration", nullptr, &showHelpCalibration);
            ImGui::Separator();
            ImGui::MenuItem("UI Pfreferences", nullptr, &showUIPrefs);
            ImGui::MenuItem("Credits", nullptr, &showCredits);
            ImGui::MenuItem("About SLProject", nullptr, &showAbout);

            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
}
//-----------------------------------------------------------------------------
void AppDemoGui::buildSceneGraph(SLScene* s)
{
    ImGui::Begin("Scenegraph", &showSceneGraph);

    if (s->root3D())
        addSceneGraphNode(s, s->root3D());

    ImGui::End();
}
//-----------------------------------------------------------------------------
void AppDemoGui::addSceneGraphNode(SLScene* s, SLNode* node)
{
    SLbool isSelectedNode = s->selectedNode() == node;
    SLbool isLeafNode     = node->children().size() == 0 && node->meshes().size() == 0;

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
                    r *= SL_RAD2DEG;

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

                        if (ImGui::SliderFloat("Far Clip", &clipF, clipN, SL_min(clipF * 1.1f, 1000000.f)))
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
                SLuint      t = (SLuint)(mesh->I16.size() ? mesh->I16.size() : mesh->I32.size());
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

                    if (m->textures().size() && ImGui::TreeNode("Textures"))
                    {
                        ImGui::Text("No. of textures: %lu", m->textures().size());

                        //SLfloat lineH = ImGui::GetTextLineHeightWithSpacing();
                        SLfloat texW = ImGui::GetWindowWidth() - 4 * ImGui::GetTreeNodeToLabelSpacing() - 10;

                        for (SLuint i = 0; i < m->textures().size(); ++i)
                        {
                            SLGLTexture* t      = m->textures()[i];
                            void*        tid    = (ImTextureID)(intptr_t)t->texName();
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
                                            for (SLuint c = 0; c < tf->colors().size(); ++c)
                                            {
                                                SLCol3f color = tf->colors()[c].color;
                                                SLchar  label[20];
                                                sprintf(label, "Color %u", c);
                                                if (ImGui::ColorEdit3(label, (float*)&color))
                                                {
                                                    tf->colors()[c].color = color;
                                                    tf->generateTexture();
                                                }
                                                ImGui::SameLine();
                                                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.5f);
                                                sprintf(label, "Pos. %u", c);
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
                                                    ImGui::Text("%3.2f Pos. %u", pos, c);
                                                ImGui::PopItemWidth();
                                            }

                                            ImGui::TreePop();
                                        }

                                        if (ImGui::TreeNode("Alpha Points in Transfer Function"))
                                        {
                                            for (SLuint a = 0; a < tf->alphas().size(); ++a)
                                            {
                                                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.25f);
                                                SLfloat alpha = tf->alphas()[a].alpha;
                                                SLchar  label[20];
                                                sprintf(label, "Alpha %u", a);
                                                if (ImGui::SliderFloat(label, &alpha, 0.0f, 1.0f, "%3.2f"))
                                                {
                                                    tf->alphas()[a].alpha = alpha;
                                                    tf->generateTexture();
                                                }
                                                ImGui::SameLine();
                                                sprintf(label, "Pos. %u", a);
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
                                                    ImGui::Text("%3.2f Pos. %u", pos, a);

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
                                        ImGui::Image(tid, ImVec2(texW, texW * h_to_w), ImVec2(0, 1), ImVec2(1, 0), ImVec4(1, 1, 1, 1), ImVec4(1, 1, 1, 1));
                                    }
                                }

                                ImGui::TreePop();
                            }
                        }

                        ImGui::TreePop();
                    }

                    if (ImGui::TreeNode("GLSL Program"))
                    {
                        for (SLuint i = 0; i < m->program()->shaders().size(); ++i)
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
            if (selectedNode->meshes().size() > 0)
            {
                ImGui::Text("Node: %s", selectedNode->name().c_str());
                for (auto selectedMesh : selectedNode->meshes())
                {
                    if ((SLuint)selectedMesh->IS32.size() > 0)
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
void AppDemoGui::buildStatsDebugTiming(SLScene* s, SLSceneView* sv)
{
    if (!SLAverageTiming::instance().size())
        return;

    SLchar m[2550]; // message character array
    m[0] = 0;       // set zero length

    //sort vertically
    std::vector<SLAverageTimingBlock*> blocks;
    for (auto& block : SLAverageTiming::instance())
    {
        blocks.push_back(block.second);
    }
    std::sort(blocks.begin(), blocks.end(), [](SLAverageTimingBlock* lhs, SLAverageTimingBlock* rhs) -> bool {
        return lhs->posV < rhs->posV;
    });

    //find reference time
    SLfloat refTime = 1.0f;
    if (blocks.size())
    {
        refTime = (*blocks.begin())->val.average();
        //insert number of measurment calls
        sprintf(m + strlen(m), "Num. calls: %i\n", (SLint)(*blocks.begin())->nCalls);
    }

    //insert time measurements
    for (auto* block : blocks)
    {
        SLfloat      val   = block->val.average();
        SLfloat      valPC = SL_clamp(val / refTime * 100.0f, 0.0f, 100.0f);
        string       name  = block->name;
        stringstream ss;
        for (int i = 0; i < block->posH; ++i)
            ss << " ";
        ss << "%s: %4.1f ms (%3d%%)\n";
        sprintf(m + strlen(m), ss.str().c_str(), name.c_str(), val, (SLint)valPC);
    }

    //define ui elements
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
    ImGui::Begin("Tracking Timing", &showStatsDebugTiming, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::TextUnformatted(m);
    ImGui::End();
    ImGui::PopFont();
}
//-----------------------------------------------------------------------------
//void AppDemoGui::buildInfosTracking(SLScene* s, SLSceneView* sv)
//{
//    //ImGui::Begin("Tracking Infos", &showInfosTracking, ImVec2(300, 0), -1.f, ImGuiWindowFlags_NoCollapse);
//
//    ////try to find SLCVTrackedRaulMur instance
//    //if (!raulMurTracker)
//    //{
//    //    for (SLCVTracked* tracker : s->trackers()) {
//    //        if (raulMurTracker = dynamic_cast<SLCVTrackedRaulMur*>(tracker))
//    //            break;
//    //    }
//
//    //    if (!raulMurTracker)
//    //        return;
//    //}
//
//    //if (!keyFrames)
//    //    keyFrames = s->root3D()->findChild<SLNode>("KeyFrames", true);
//    //if (!mapPoints)
//    //    mapPoints = s->root3D()->findChild<SLNode>("MapPoints", true);
//
//    ////-------------------------------------------------------------------------
//    ////numbers
//    ////add tracking state
//    //ImGui::Text("Tracking State : %s ", raulMurTracker->getPrintableState().c_str());
//    ////mean reprojection error
//    //ImGui::Text("Mean Reproj. Error : %f ", raulMurTracker->meanReprojectionError());
//    ////add number of matches map points in current frame
//    //ImGui::Text("Num Map Matches : %d ", raulMurTracker->getNMapMatches());
//    ////L2 norm of the difference between the last and the current camera pose
//    //ImGui::Text("Pose Difference : %f ", raulMurTracker->poseDifference());
//    //ImGui::Separator();
//
//    //SLbool b;
//    ////-------------------------------------------------------------------------
//    ////keypoints infos
//    //if (ImGui::CollapsingHeader("KeyPoints"))
//    //{
//    //    //ImGui::Text("KeyPoints");
//
//    //    //show 2D key points in video image
//    //    b = raulMurTracker->showKeyPoints();
//    //    if (ImGui::Checkbox("KeyPts", &b))
//    //    {
//    //        raulMurTracker->showKeyPoints(b);
//    //    }
//
//    //    //show matched 2D key points in video image
//    //    b = raulMurTracker->showKeyPointsMatched();
//    //    if (ImGui::Checkbox("KeyPts Matched", &b))
//    //    {
//    //        raulMurTracker->showKeyPointsMatched(b);
//    //    }
//
//    //    //undistort image
//    //    SLCVCalibration* ac = SLApplication::activeCalib;
//    //    b = (ac->showUndistorted() && ac->state() == CS_calibrated);
//    //    if (ImGui::Checkbox("Undistort Image", &b))
//    //    {
//    //        ac->showUndistorted(b);
//    //    }
//    //    //ImGui::Separator();
//    //}
//
//    ////-------------------------------------------------------------------------
//    ////mappoints infos
//    //if (ImGui::CollapsingHeader("MapPoints"))
//    //{
//    //    //ImGui::Text("MapPoints");
//    //    //number of map points
//    //    ImGui::Text("Count : %d ", raulMurTracker->mapPointsCount());
//    //    //show mappoints scene objects
//    //    if (mapPoints)
//    //    {
//    //        b = !mapPoints->drawBits()->get(SL_DB_HIDDEN);
//    //        if (ImGui::Checkbox("Show All", &b))
//    //        {
//    //            mapPoints->drawBits()->set(SL_DB_HIDDEN, !b);
//    //        }
//    //    }
//    //    //show and update matches to mappoints
//    //    b = raulMurTracker->showMatchesPC();
//    //    if (ImGui::Checkbox("Show Matches to Map", &b))
//    //    {
//    //        raulMurTracker->showMatchesPC(b);
//    //    }
//    //    //show and update local map points
//    //    b = raulMurTracker->showLocalMapPC();
//    //    if (ImGui::Checkbox("Show Local Map", &b))
//    //    {
//    //        raulMurTracker->showLocalMapPC(b);
//    //    }
//
//    //    //ImGui::Separator();
//    //}
//    ////-------------------------------------------------------------------------
//    ////keyframe infos
//    //if (ImGui::CollapsingHeader("KeyFrames"))
//    //{
//    //    //ImGui::Text("KeyFrames");
//    //    //add number of keyframes
//    //    if (keyFrames) {
//    //        ImGui::Text("Number of Keyframes : %d ", keyFrames->children().size());
//    //    }
//    //    //show keyframe scene objects
//    //    if (keyFrames)
//    //    {
//    //        b = !keyFrames->drawBits()->get(SL_DB_HIDDEN);
//    //        if (ImGui::Checkbox("Show", &b))
//    //        {
//    //            keyFrames->drawBits()->set(SL_DB_HIDDEN, !b);
//    //            for (SLNode* child : keyFrames->children()) {
//    //                if (child)
//    //                    child->drawBits()->set(SL_DB_HIDDEN, !b);
//    //            }
//    //        }
//    //    }
//
//    //    //get keyframe database
//    //    if (SLCVKeyFrameDB* kfDB = raulMurTracker->getKfDB())
//    //    {
//    //        //if backgound rendering is active kf images will be rendered on
//    //        //near clipping plane if kf is not the active camera
//    //        b = kfDB->renderKfBackground();
//    //        if (ImGui::Checkbox("Show Image", &b))
//    //        {
//    //            kfDB->renderKfBackground(b);
//    //        }
//
//    //        //allow SLCVCameras as active camera so that we can look through it
//    //        b = kfDB->allowAsActiveCam();
//    //        if (ImGui::Checkbox("Allow as Active Cam", &b))
//    //        {
//    //            kfDB->allowAsActiveCam(b);
//    //        }
//    //    }
//    //    //ImGui::Separator();
//    //}
//
//    ////-------------------------------------------------------------------------
//    //if (ImGui::CollapsingHeader("Alignment"))
//    //{
//    //    //ImGui::Text("Alignment");
//    //    //slider to adjust transformation value
//    //    //ImGui::SliderFloat("Value", &SLGLImGui::transformationValue, -10.f, 10.f, "%5.2f");
//
//    //    //rotation
//    //    ImGui::InputFloat("Rot. Value", &SLGLImGui::transformationRotValue, 0.1f);
//    //    SLGLImGui::transformationRotValue = ImClamp(SLGLImGui::transformationRotValue, -360.0f, 360.0f);
//
//    //    static SLfloat sp = 3; //spacing
//    //    SLfloat bW = (ImGui::GetContentRegionAvailWidth() - 2 * sp) / 3;
//    //    if (ImGui::Button("RotX", ImVec2(bW, 0.0f))) {
//    //        raulMurTracker->applyTransformation(SLGLImGui::transformationRotValue, SLCVTrackedRaulMur::ROT_X);
//    //    } ImGui::SameLine(0.0, sp);
//    //    if (ImGui::Button("RotY", ImVec2(bW, 0.0f))) {
//    //        raulMurTracker->applyTransformation(SLGLImGui::transformationRotValue, SLCVTrackedRaulMur::ROT_Y);
//    //    } ImGui::SameLine(0.0, sp);
//    //    if (ImGui::Button("RotZ", ImVec2(bW, 0.0f))) {
//    //        raulMurTracker->applyTransformation(SLGLImGui::transformationRotValue, SLCVTrackedRaulMur::ROT_Z);
//    //    }
//    //    ImGui::Separator();
//
//    //    //translation
//    //    ImGui::InputFloat("Transl. Value", &SLGLImGui::transformationTransValue, 0.1f);
//
//    //    if (ImGui::Button("TransX", ImVec2(bW, 0.0f))) {
//    //        raulMurTracker->applyTransformation(SLGLImGui::transformationTransValue, SLCVTrackedRaulMur::TRANS_X);
//    //    } ImGui::SameLine(0.0, sp);
//    //    if (ImGui::Button("TransY", ImVec2(bW, 0.0f))) {
//    //        raulMurTracker->applyTransformation(SLGLImGui::transformationTransValue, SLCVTrackedRaulMur::TRANS_Y);
//    //    } ImGui::SameLine(0.0, sp);
//    //    if (ImGui::Button("TransZ", ImVec2(bW, 0.0f))) {
//    //        raulMurTracker->applyTransformation(SLGLImGui::transformationTransValue, SLCVTrackedRaulMur::TRANS_Z);
//    //    }
//    //    ImGui::Separator();
//
//    //    //scale
//    //    ImGui::InputFloat("Scale Value", &SLGLImGui::transformationScaleValue, 0.1f);
//    //    SLGLImGui::transformationScaleValue = ImClamp(SLGLImGui::transformationScaleValue, 0.0f, 1000.0f);
//
//    //    if (ImGui::Button("Scale", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f))) {
//    //        raulMurTracker->applyTransformation(SLGLImGui::transformationScaleValue, SLCVTrackedRaulMur::SCALE);
//    //    }
//    //    ImGui::Separator();
//
//    //    if (ImGui::Button("Save State", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f))) {
//    //        raulMurTracker->saveState();
//    //    }
//    //}
//
//    //ImGui::End();
//}
//-----------------------------------------------------------------------------
void AppDemoGui::buildInfosDialogs()
{
    for (auto dialog : _infoDialogs)
    {
        if (dialog.second->show)
        {
            ImGui::Begin(dialog.second->getName(), &dialog.second->show, ImVec2(300, 0), -1.f, ImGuiWindowFlags_NoCollapse);

            dialog.second->buildInfos();

            ImGui::End();
        }
    }
}
//-----------------------------------------------------------------------------
void AppDemoGui::loadConfig(SLint dotsPerInch)
{
    ImGuiStyle& style               = ImGui::GetStyle();
    SLstring    fullPathAndFilename = SLApplication::configPath +
                                   SLApplication::name + ".yml";

    if (!SLFileSystem::fileExists(fullPathAndFilename))
    {
        // Scale for proportional and fixed size fonts
        SLfloat dpiScaleProp  = dotsPerInch / 120.0f;
        SLfloat dpiScaleFixed = dotsPerInch / 142.0f;

        // Default settings for the first time
        SLGLImGui::fontPropDots  = SL_max(16.0f * dpiScaleProp, 16.0f);
        SLGLImGui::fontFixedDots = SL_max(13.0f * dpiScaleFixed, 13.0f);

        // Store dialog show states
        AppDemoGui::showAbout           = true;
        AppDemoGui::showInfosScene      = true;
        AppDemoGui::showStatsTiming     = false;
        AppDemoGui::showStatsScene      = false;
        AppDemoGui::showStatsVideo      = false;
        AppDemoGui::showInfosFrameworks = false;
        AppDemoGui::showInfosSensors    = false;
        AppDemoGui::showSceneGraph      = false;
        AppDemoGui::showProperties      = false;

        // Adjust UI paddings on DPI
        style.FramePadding.x     = SL_max(8.0f * dpiScaleFixed, 8.0f);
        style.WindowPadding.x    = style.FramePadding.x;
        style.FramePadding.y     = SL_max(3.0f * dpiScaleFixed, 3.0f);
        style.ItemSpacing.x      = SL_max(8.0f * dpiScaleFixed, 8.0f);
        style.ItemSpacing.y      = SL_max(3.0f * dpiScaleFixed, 3.0f);
        style.ItemInnerSpacing.x = style.ItemSpacing.y;
        style.ScrollbarSize      = SL_max(16.0f * dpiScaleFixed, 16.0f);
        style.ScrollbarRounding  = std::floor(style.ScrollbarSize / 2);

        return;
    }

    SLCVFileStorage fs;
    try
    {
        fs.open(fullPathAndFilename, SLCVFileStorage::READ);
        if (fs.isOpened())
        {
            // clang-format off
            SLint  i;
            SLbool b;
            fs["configTime"] >> AppDemoGui::configTime;
            fs["fontPropDots"] >> i;        SLGLImGui::fontPropDots = (SLfloat)i;
            fs["fontFixedDots"] >> i;       SLGLImGui::fontFixedDots = (SLfloat)i;
            fs["ItemSpacingX"] >> i;        style.ItemSpacing.x = (SLfloat)i;
            fs["ItemSpacingY"] >> i;        style.ItemSpacing.y = (SLfloat)i;
            style.WindowPadding.x = style.FramePadding.x = style.ItemInnerSpacing.x = style.ItemSpacing.x;
            style.WindowPadding.y = style.FramePadding.y = style.ItemInnerSpacing.y = style.ItemSpacing.y;
            fs["ScrollbarSize"] >> i; style.ScrollbarSize = (SLfloat)i;
            style.ScrollbarRounding = std::floor(style.ScrollbarSize / 2);
            fs["sceneID"] >> i;             SLApplication::sceneID = (SLSceneID)i;
            fs["showInfosScene"] >> b;      AppDemoGui::showInfosScene = b;
            fs["showStatsTiming"] >> b;     AppDemoGui::showStatsTiming = b;
            fs["showStatsMemory"] >> b;     AppDemoGui::showStatsScene = b;
            fs["showStatsVideo"] >> b;      AppDemoGui::showStatsVideo = b;
            fs["showInfosFrameworks"] >> b; AppDemoGui::showInfosFrameworks = b;
            fs["showInfosSensors"] >> b;    AppDemoGui::showInfosSensors = b;
            fs["showSceneGraph"] >> b;      AppDemoGui::showSceneGraph = b;
            fs["showProperties"] >> b;      AppDemoGui::showProperties = b;
            fs["showUIPrefs"] >> b;         AppDemoGui::showUIPrefs = b;
            fs["showDetection"] >> b;       SLApplication::scene->showDetection(b);
            // clang-format on

            fs.release();
            SL_LOG("Config. loaded  : %s\n", fullPathAndFilename.c_str());
            SL_LOG("Config. date    : %s\n", AppDemoGui::configTime.c_str());
            SL_LOG("fontPropDots    : %f\n", SLGLImGui::fontPropDots);
            SL_LOG("fontFixedDots   : %f\n", SLGLImGui::fontFixedDots);
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
            SLGLImGui::fontPropDots  = SL_max(16.0f * dpiScaleProp, 16.0f);
            SLGLImGui::fontFixedDots = SL_max(13.0f * dpiScaleFixed, 13.0f);
        }
    }
}
//-----------------------------------------------------------------------------
void AppDemoGui::saveConfig()
{
    ImGuiStyle& style               = ImGui::GetStyle();
    SLstring    fullPathAndFilename = SLApplication::configPath +
                                   SLApplication::name + ".yml";
    SLCVFileStorage fs(fullPathAndFilename, SLCVFileStorage::WRITE);

    if (!fs.isOpened())
    {
        SL_LOG("Failed to open file for writing: %s", fullPathAndFilename.c_str());
        SL_EXIT_MSG("Exit in AppDemoGui::saveConfig");
        return;
    }

    fs << "configTime" << SLUtils::getLocalTimeString();
    fs << "fontPropDots" << (SLint)SLGLImGui::fontPropDots;
    fs << "fontFixedDots" << (SLint)SLGLImGui::fontFixedDots;
    fs << "sceneID" << (SLint)SLApplication::sceneID;
    fs << "ItemSpacingX" << (SLint)style.ItemSpacing.x;
    fs << "ItemSpacingY" << (SLint)style.ItemSpacing.y;
    fs << "ScrollbarSize" << (SLint)style.ScrollbarSize;
    fs << "showStatsTiming" << AppDemoGui::showStatsTiming;
    fs << "showStatsMemory" << AppDemoGui::showStatsScene;
    fs << "showStatsVideo" << AppDemoGui::showStatsVideo;
    fs << "showInfosFrameworks" << AppDemoGui::showInfosFrameworks;
    fs << "showInfosScene" << AppDemoGui::showInfosScene;
    fs << "showInfosSensors" << AppDemoGui::showInfosSensors;
    fs << "showSceneGraph" << AppDemoGui::showSceneGraph;
    fs << "showProperties" << AppDemoGui::showProperties;
    fs << "showUIPrefs" << AppDemoGui::showUIPrefs;
    fs << "showDetection" << SLApplication::scene->showDetection();

    fs.release();
    SL_LOG("Config. saved   : %s\n", fullPathAndFilename.c_str());
}
//-----------------------------------------------------------------------------
