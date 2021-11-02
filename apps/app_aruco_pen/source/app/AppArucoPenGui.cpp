//#############################################################################
//  File:      AppArucoPenGui.cpp
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch, Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <app/AppArucoPenGui.h>
#include <app/AppArucoPen.h>
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
#include <SLLightSpot.h>
#include <SLLightRect.h>
#include <SLLightSpot.h>
#include <SLShadowMap.h>
#include <SLMaterial.h>
#include <SLMesh.h>
#include <SLNode.h>
#include <SLScene.h>
#include <SLGLImGui.h>
#include <SLTexColorLUT.h>
#include <SLGLImGui.h>
#include <SLProjectScene.h>
#include <AverageTiming.h>
#include <imgui.h>
#include <ftplib.h>
#include <HttpUtils.h>
#include <ZipUtils.h>
#include <Instrumentor.h>
#include <app/AppArucoPen.h>

#ifdef SL_BUILD_WAI
#    include <Eigen/Dense>
#endif

//-----------------------------------------------------------------------------
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
SLstring AppArucoPenGui::configTime        = "-";
SLbool   AppArucoPenGui::showDockSpace     = true;
SLbool   AppArucoPenGui::showStatsTiming   = false;
SLbool   AppArucoPenGui::showStatsScene    = false;
SLbool   AppArucoPenGui::showStatsVideo    = false;
SLbool   AppArucoPenGui::showImGuiMetrics  = false;
SLbool   AppArucoPenGui::showInfosSensors  = false;
SLbool   AppArucoPenGui::showInfosDevice   = false;
SLbool   AppArucoPenGui::showInfosTracking = false;
SLbool   AppArucoPenGui::hideUI            = false;

//-----------------------------------------------------------------------------
void AppArucoPenGui::clear()
{
}
//-----------------------------------------------------------------------------
//! This is the main building function for the GUI of the Demo apps
/*! Is is passed to the AppArucoPenGui::build function in main of the app-Demo-SLProject
 app. This function will be called once per frame roughly at the end of
 SLSceneView::onPaint in SLSceneView::draw2DGL by calling ImGui::Render.\n
 See also the comments on SLGLImGui.
 */
void AppArucoPenGui::build(SLProjectScene* s, SLSceneView* sv)
{
    PROFILE_FUNCTION();

    if (AppArucoPenGui::hideUI ||
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
                    ImGui::SetNextWindowPos(viewport->WorkPos);
                    ImGui::SetNextWindowSize(viewport->WorkSize);
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
                    sprintf(m + strlen(m), "Drawcalls  : %d\n", SLGLVertexArray::totalDrawCalls);
                    sprintf(m + strlen(m), " Shadow    : %d\n", SLShadowMap::drawCalls);
                    sprintf(m + strlen(m), " Render    : %d\n", SLGLVertexArray::totalDrawCalls - SLShadowMap::drawCalls);
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

                if (vt != VT_NONE && AppArucoPen::instance().trackedNode != nullptr)
                {
                    sprintf(m + strlen(m), "-------------:\n");

                    SLNode* trackedNode = AppArucoPen::instance().trackedNode;
                    if (typeid(*trackedNode) == typeid(SLCamera))
                    {
                        SLVec3f cameraPos = AppArucoPen::instance().trackedNode->updateAndGetWM().translation();
                        sprintf(m + strlen(m), "Dist. to zero: %4.2f\n", cameraPos.length());
                    }
                    else
                    {
                        SLVec3f cameraPos = ((SLNode*)sv->camera())->updateAndGetWM().translation();
                        SLVec3f objectPos = AppArucoPen::instance().trackedNode->updateAndGetWM().translation();
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

            if (showImGuiMetrics)
            {
                ImGui::ShowMetricsWindow();
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

            if (showInfosTracking)
            {
                SLchar m[1024]; // message character array
                m[0] = 0;       // set zero length

                SLArucoPen& pen    = AppArucoPen::instance().arucoPen();
                SLVec3f     tipPos = pen.tipPosition();
                sprintf(m + strlen(m), "Tip position             : %s\n", tipPos.toString(", ", 2).c_str());
                sprintf(m + strlen(m), "Measured Distance (Live) : %.2f cm\n", pen.liveDistance() * 100.0f);
                sprintf(m + strlen(m), "Measured Distance (Last) : %.2f cm\n", pen.lastDistance() * 100.0f);

                // Switch to fixed font
                ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
                ImGui::Begin("Tracking Information", &showInfosTracking, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
                ImGui::TextUnformatted(m);
                ImGui::End();
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
void AppArucoPenGui::buildMenuBar(SLProjectScene* s, SLSceneView* sv)
{
    PROFILE_FUNCTION();

    SLSceneID    sid           = AppDemo::sceneID;
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
            if (ImGui::BeginMenu("Load Scene"))
            {
                if (ImGui::MenuItem("Track Aruco Cube", nullptr, sid == SID_VideoTrackArucoCubeMain))
                    s->onLoad(s, sv, SID_VideoTrackArucoCubeMain);
                if (ImGui::MenuItem("Track Aruco Cube (Virtual)", nullptr, sid == SID_VirtualArucoPen))
                    s->onLoad(s, sv, SID_VirtualArucoPen);
                if (ImGui::MenuItem("Show Aruco Pen Trail", nullptr, sid == SID_ArucoPenTrail))
                    s->onLoad(s, sv, SID_ArucoPenTrail);
                if (ImGui::MenuItem("Track Chessboard (Main)", nullptr, sid == SID_VideoTrackChessMain))
                    s->onLoad(s, sv, SID_VideoTrackChessMain);

                ImGui::EndMenu();
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

            if (ImGui::BeginMenu("Video Sensor"))
            {
                if (ImGui::BeginMenu("Capture Provider"))
                {
                    AppArucoPen& app = AppArucoPen::instance();

                    for (CVCaptureProvider* provider : app.captureProviders())
                    {
                        if (ImGui::MenuItem(provider->name().c_str(), nullptr, app.currentCaptureProvider() == provider))
                            app.currentCaptureProvider(provider);
                    }

                    ImGui::EndMenu();
                }

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

                if (ImGui::BeginMenu("Resolution", capture->videoType() == VT_MAIN))
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
                    if (ImGui::MenuItem("Start Calibration (Current Camera)"))
                        s->onLoad(s, sv, SID_VideoCalibrateMain);

                    if (ImGui::MenuItem("Calculate Extrinsic (Current Camera)"))
                        AppArucoPenCalibrator::calcExtrinsicParams(AppArucoPen::instance().currentCaptureProvider());

                    if (ImGui::MenuItem("Calculate Extrinsic (All Cameras)"))
                    {
                        for (CVCaptureProvider* provider : AppArucoPen::instance().captureProviders())
                        {
                            AppArucoPenCalibrator::calcExtrinsicParams(provider);
                        }
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

//                if (!AppArucoPen::instance().trackers().empty())
//                    if (ImGui::MenuItem("Draw Detection", nullptr, AppArucoPen::instance().tracker->drawDetection()))
//                        AppArucoPen::instance().tracker->drawDetection(!AppArucoPen::instance().tracker->drawDetection());

                ImGui::EndMenu();
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Camera"))
        {
            SLCamera*    cam  = sv->camera();
            SLProjection proj = cam->projection();

            if (ImGui::MenuItem("Reset"))
            {
                cam->resetToInitialState();
                cam->focalDist(cam->translationOS().length());
            }

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

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Infos"))
        {
            if (ImGui::BeginMenu("Statistics"))
            {
                ImGui::MenuItem("Stats on Timing", nullptr, &showStatsTiming);
                ImGui::MenuItem("Stats on Scene", nullptr, &showStatsScene);
                ImGui::MenuItem("Stats on Video", nullptr, &showStatsVideo);
                ImGui::MenuItem("Stats on ImGui", nullptr, &showImGuiMetrics);
                ImGui::EndMenu();
            }

            ImGui::Separator();
            ImGui::MenuItem("Infos on Device", nullptr, &showInfosDevice);
            ImGui::MenuItem("Infos on Sensors", nullptr, &showInfosSensors);
            ImGui::MenuItem("Infos on Tracking", nullptr, &showInfosTracking);

            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
}
//-----------------------------------------------------------------------------
//! Builds context menu if right mouse click is over non-imgui area
void AppArucoPenGui::buildMenuContext(SLProjectScene* s, SLSceneView* sv)
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
                ImGui::Separator();
            }
        }

        if (AppArucoPenGui::hideUI)
            if (ImGui::MenuItem("Show user interface"))
                AppArucoPenGui::hideUI = false;

        if (!AppArucoPenGui::hideUI)
            if (ImGui::MenuItem("Hide user interface"))
                AppArucoPenGui::hideUI = true;

        ImGui::EndPopup();
    }
}
//-----------------------------------------------------------------------------
//! Loads the UI configuration
void AppArucoPenGui::loadConfig(SLint dotsPerInch)
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
        AppArucoPenGui::showStatsTiming  = false;
        AppArucoPenGui::showStatsScene   = false;
        AppArucoPenGui::showStatsVideo   = false;
        AppArucoPenGui::showInfosDevice  = false;
        AppArucoPenGui::showInfosSensors = false;
        AppArucoPenGui::showDockSpace    = true;

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
            fs["configTime"] >> AppArucoPenGui::configTime;
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
            fs["showStatsTiming"] >> b;     AppArucoPenGui::showStatsTiming = b;
            fs["showStatsMemory"] >> b;     AppArucoPenGui::showStatsScene = b;
            fs["showStatsVideo"] >> b;      AppArucoPenGui::showStatsVideo = b;
            fs["showInfosFrameworks"] >> b; AppArucoPenGui::showInfosDevice = b;
            fs["showInfosSensors"] >> b;    AppArucoPenGui::showInfosSensors = b;
            fs["showInfosTracking"] >> b;   AppArucoPenGui::showInfosTracking = b;
            fs["showDockSpace"] >> b;       AppArucoPenGui::showDockSpace = b;
            // clang-format on

            fs.release();
            SL_LOG("Config. loaded   : %s", fullPathAndFilename.c_str());
            SL_LOG("Config. date     : %s", AppArucoPenGui::configTime.c_str());
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
void AppArucoPenGui::saveConfig()
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
        SL_EXIT_MSG("Exit in AppArucoPenGui::saveConfig");
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
    fs << "showStatsTiming" << AppArucoPenGui::showStatsTiming;
    fs << "showStatsMemory" << AppArucoPenGui::showStatsScene;
    fs << "showStatsVideo" << AppArucoPenGui::showStatsVideo;
    fs << "showInfosFrameworks" << AppArucoPenGui::showInfosDevice;
    fs << "showInfosSensors" << AppArucoPenGui::showInfosSensors;
    fs << "showInfosTracking" << AppArucoPenGui::showInfosTracking;
    fs << "showDockSpace" << AppArucoPenGui::showDockSpace;

    fs.release();
    SL_LOG("Config. saved   : %s", fullPathAndFilename.c_str());
}
//-----------------------------------------------------------------------------
