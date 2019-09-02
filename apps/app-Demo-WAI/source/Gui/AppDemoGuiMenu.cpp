#include <SL.h>
#include <SLInterface.h>
#include <SLApplication.h>
#include <CVCapture.h>
#include <SLCamera.h>
#include <SLGLImGui.h>
#include <SLAverageTiming.h>
#include <AppDemoGuiMenu.h>
#include <Utils.h>


void AppDemoGuiMenu::build(GUIPreferences * prefs, SLScene* s, SLSceneView* sv)
{
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Quit & Save", "ESC"))
                slShouldClose(true);
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Preferences"))
        {
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
                CVCalibration* ac = CVCapture::instance()->activeCalib;
                CVCalibration* mc = &CVCapture::instance()->calibMainCam;
                CVCalibration* sc = &CVCapture::instance()->calibScndCam;

                if (ImGui::BeginMenu("Mirror Main Camera"))
                {
                    if (ImGui::MenuItem("Horizontally", nullptr, mc->isMirroredH()))
                        mc->toggleMirrorH();

                    if (ImGui::MenuItem("Vertically", nullptr, mc->isMirroredV()))
                        mc->toggleMirrorV();

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Mirror Scnd. Camera", CVCapture::instance()->hasSecondaryCamera))
                {
                    if (ImGui::MenuItem("Horizontally", nullptr, sc->isMirroredH()))
                        sc->toggleMirrorH();

                    if (ImGui::MenuItem("Vertically", nullptr, sc->isMirroredV()))
                        sc->toggleMirrorV();

                    ImGui::EndMenu();
                }

                ImGui::EndMenu();
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Camera"))
        {
            SLCamera*     cam = sv->camera();
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

                if (ImGui::SliderFloat("Far Clip", &clipF, clipN, Utils::min(clipF * 1.1f, 1000000.f)))
                    cam->clipFar(clipF);

                ImGui::PopItemWidth();
                ImGui::EndMenu();
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Infos"))
        {
            ImGui::MenuItem("Infos on Scene", nullptr, &prefs->showInfosScene);
            ImGui::MenuItem("Stats on Timing", nullptr, &prefs->showStatsTiming);

            if (SLAverageTiming::instance().size())
            {
                ImGui::MenuItem("Stats on Debug Time", nullptr, &prefs->showStatsDebugTiming);
            }

            ImGui::MenuItem("Stats on Scene", nullptr, &prefs->showStatsScene);
            ImGui::MenuItem("Infos Map Node Transform", nullptr, &prefs->showInfosMapNodeTransform);
            ImGui::MenuItem("Stats on Video", nullptr, &prefs->showStatsVideo);
            ImGui::Separator();
            ImGui::MenuItem("Show Scenegraph", nullptr, &prefs->showSceneGraph);
            ImGui::MenuItem("Show Properties", nullptr, &prefs->showProperties);
            ImGui::MenuItem("Show Transform", nullptr, &prefs->showTransform);
            ImGui::Separator();
            ImGui::MenuItem("Infos on Sensors", nullptr, &prefs->showInfosSensors);
            ImGui::MenuItem("Infos on Frameworks", nullptr, &prefs->showInfosFrameworks);
            ImGui::MenuItem("Infos on Tracking", nullptr, &prefs->showInfosTracking);
            ImGui::Separator();
            ImGui::MenuItem("Help on Interaction", nullptr, &prefs->showHelp);
            ImGui::MenuItem("Help on Calibration", nullptr, &prefs->showHelpCalibration);
            ImGui::MenuItem("Map storage", nullptr, &prefs->showMapStorage);
            ImGui::MenuItem("Tracked Mapping", nullptr, &prefs->showTrackedMapping);
            ImGui::MenuItem("Video Storage", nullptr, &prefs->showVideoStorage);
            ImGui::Separator();
            ImGui::MenuItem("UI Preferences", nullptr, &prefs->showUIPrefs);
            ImGui::MenuItem("Credits", nullptr, &prefs->showCredits);
            ImGui::MenuItem("About WAI-Demo", nullptr, &prefs->showAbout);

            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
}
