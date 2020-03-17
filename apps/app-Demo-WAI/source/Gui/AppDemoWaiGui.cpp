//#############################################################################
//  File:      AppDemoWaiGui.cpp
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

#include <AppDemoWaiGui.h>
#include <SLAnimPlayback.h>
#include <AverageTiming.h>
#include <CVImage.h>
#include <CVTrackedFeatures.h>
#include <SLGLProgram.h>
#include <SLGLShader.h>
#include <SLGLTexture.h>
#include <AppDemoGuiInfosDialog.h>
#include <AppDemoGuiTrackedMapping.h>
#include <SLImporter.h>
#include <SLMaterial.h>
#include <SLMesh.h>
#include <SLNode.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLGLImGui.h>
#include <imgui.h>
#include <imgui_internal.h>

//map<string, AppDemoGuiInfosDialog*> AppDemoWaiGui::_infoDialogs;

//-----------------------------------------------------------------------------
AppDemoWaiGui::AppDemoWaiGui(std::string     appName,
                             std::string     configDir,
                             int             dotsPerInch,
                             std::string     fontPath,
                             int             windowWidthPix,
                             int             windowHeightPix,
                             ButtonPressedCB backButtonCB)
{
    //load preferences
    uiPrefs        = std::make_unique<GUIPreferences>(dotsPerInch);
    _prefsFileName = Utils::unifySlashes(configDir) + appName + ".yml";
    uiPrefs->load(_prefsFileName, ImGui::GetStyle());
    //load fonts
    loadFonts(uiPrefs->fontPropDots, uiPrefs->fontFixedDots, fontPath);

    _backButton = BackButton(dotsPerInch,
                             windowWidthPix,
                             windowHeightPix,
                             GuiAlignment::BOTTOM_RIGHT,
                             5.f,
                             5.f,
                             {10.f, 7.f},
                             backButtonCB);
}
//-----------------------------------------------------------------------------
AppDemoWaiGui::~AppDemoWaiGui()
{
    //save preferences
    //todo: destructor not callled on android as expected (too late)
    uiPrefs->save(_prefsFileName, ImGui::GetStyle());
}
//-----------------------------------------------------------------------------
void AppDemoWaiGui::addInfoDialog(std::shared_ptr<AppDemoGuiInfosDialog> dialog)
{
    string name = string(dialog->getName());
    if (_infoDialogs.find(name) == _infoDialogs.end())
    {
        _infoDialogs[name] = dialog;
    }
}
//-----------------------------------------------------------------------------
void AppDemoWaiGui::clearInfoDialogs()
{
    _infoDialogs.clear();
}
//-----------------------------------------------------------------------------
void AppDemoWaiGui::build(SLScene* s, SLSceneView* sv)
{
    _backButton.render();

    buildInfosDialogs(s, sv);
    buildMenu(s, sv);
}
//-----------------------------------------------------------------------------
void AppDemoWaiGui::buildInfosDialogs(SLScene* s, SLSceneView* sv)
{
    for (auto dialog : _infoDialogs)
    {
        if (dialog.second->show())
        {
            dialog.second->buildInfos(s, sv);
        }
    }
}
//-----------------------------------------------------------------------------
void AppDemoWaiGui::buildMenu(SLScene* s, SLSceneView* sv)
{
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("Slam"))
        {
            ImGui::MenuItem("Start", nullptr, &uiPrefs->showSlamLoad);
            ImGui::MenuItem("Tracked Mapping", nullptr, &uiPrefs->showTrackedMapping);

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Video/GPS"))
        {
            ImGui::MenuItem("Video/GPS Storage", nullptr, &uiPrefs->showVideoStorage);
            ImGui::MenuItem("Video controls", nullptr, &uiPrefs->showVideoControls);

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Map"))
        {
            ImGui::MenuItem("Infos Map Node Transform", nullptr, &uiPrefs->showInfosMapNodeTransform);
            ImGui::EndMenu();
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

                if (ImGui::MenuItem("Intrinsic", "5", proj == P_monoIntrinsic))
                {
                    cam->projection(P_monoIntrinsic);
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

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Infos"))
        {
            ImGui::MenuItem("Infos on Scene", nullptr, &uiPrefs->showInfosScene);
            ImGui::MenuItem("Stats on Timing", nullptr, &uiPrefs->showStatsTiming);

            ImGui::MenuItem("Stats on Debug Time", nullptr, &uiPrefs->showStatsDebugTiming);

            ImGui::MenuItem("Stats on Video", nullptr, &uiPrefs->showStatsVideo);
            ImGui::Separator();
            ImGui::MenuItem("Show Scenegraph", nullptr, &uiPrefs->showSceneGraph);
            ImGui::MenuItem("Show Properties", nullptr, &uiPrefs->showProperties);
            ImGui::MenuItem("Show Transform", nullptr, &uiPrefs->showTransform);
            ImGui::Separator();
            ImGui::MenuItem("Infos on Sensors", nullptr, &uiPrefs->showInfosSensors);
            ImGui::MenuItem("Infos on Frameworks", nullptr, &uiPrefs->showInfosFrameworks);
            ImGui::MenuItem("Infos on Tracking", nullptr, &uiPrefs->showInfosTracking);
            ImGui::MenuItem("UI Preferences", nullptr, &uiPrefs->showUIPrefs);

            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
}
//-----------------------------------------------------------------------------
//! Loads the proportional and fixed size font depending on the passed DPI
void AppDemoWaiGui::loadFonts(SLfloat fontPropDots, SLfloat fontFixedDots, std::string fontPath)
{
    ImGuiIO& io = ImGui::GetIO();
    //io.Fonts->Clear();

    //// Load proportional font for menue and text displays
    //SLstring DroidSans = fontPath + "DroidSans.ttf";
    //if (Utils::fileExists(DroidSans))
    //{
    //    io.Fonts->AddFontFromFileTTF(DroidSans.c_str(), fontPropDots);
    //    SL_LOG("ImGuiWrapper::loadFonts: %f", fontPropDots);
    //}
    //else
    //    SL_LOG("\n*** Error ***: \nFont doesn't exist: %s\n", DroidSans.c_str());

    // Load fixed size font for statistics windows
    SLstring ProggyClean = fontPath + "ProggyClean.ttf";
    if (Utils::fileExists(ProggyClean))
    {
        io.Fonts->AddFontFromFileTTF(ProggyClean.c_str(), fontFixedDots);
        SL_LOG("ImGuiWrapper::loadFonts: %f", fontFixedDots);
    }
    else
        SL_LOG("\n*** Error ***: \nFont doesn't exist: %s\n", ProggyClean.c_str());
}
