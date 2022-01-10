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
#include <app/AppArucoPenEvaluator.h>
#include <AppDemo.h>
#include <cv/CVImage.h>
#include <SLGLProgramManager.h>
#include <SLGLShader.h>
#include <SLInterface.h>
#include <SLDeviceRotation.h>
#include <SLGLImGui.h>
#include <SLProjectScene.h>
#include <imgui.h>
#include <CVCaptureProviderIDSPeak.h>
#include <TrackingSystemArucoCube.h>
#include <TrackingSystemSpryTrack.h>
#include <Instrumentor.h>

//-----------------------------------------------------------------------------
SLbool AppArucoPenGui::showDockSpace     = true;
SLbool AppArucoPenGui::showInfosTracking = false;
SLbool AppArucoPenGui::hideUI            = false;
//-----------------------------------------------------------------------------
void AppArucoPenGui::build(SLProjectScene* s, SLSceneView* sv)
{
    PROFILE_FUNCTION();

    if (AppArucoPenGui::hideUI)
    {
        buildMenuContext(s, sv);
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

        buildMenuBar(s, sv);

        buildMenuContext(s, sv);

        if (showInfosTracking)
        {
            SLchar m[1024]; // message character array
            m[0] = 0;       // set zero length

            ArucoPen& pen    = AppArucoPen::instance().arucoPen();
            SLVec3f   tipPos = pen.tipPosition();
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
//-----------------------------------------------------------------------------
void AppArucoPenGui::buildMenuBar(SLProjectScene* s, SLSceneView* sv)
{
    PROFILE_FUNCTION();

    SLSceneID sid = AppDemo::sceneID;

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

                ImGui::EndMenu();
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Quit & Save"))
                slShouldClose(true);

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Pen Tracking"))
        {
            AppArucoPen& app = AppArucoPen::instance();

            if (ImGui::BeginMenu("Tracking Mode"))
            {
                TrackingSystem* ts = app.arucoPen().trackingSystem();

                if (ImGui::MenuItem("ArUco Cube", nullptr, typeid(*ts) == typeid(TrackingSystemArucoCube)))
                    app.arucoPen().trackingSystem(new TrackingSystemArucoCube());

                if (ImGui::MenuItem("SpryTrack", nullptr, typeid(*ts) == typeid(TrackingSystemSpryTrack)))
                    app.arucoPen().trackingSystem(new TrackingSystemSpryTrack());

                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Capture Provider"))
            {
                for (CVCaptureProvider* provider : app.captureProviders())
                {
                    if (app.arucoPen().trackingSystem()->isAcceptedProvider(provider))
                    {
                        if (ImGui::MenuItem(provider->name().c_str(), nullptr, app.currentCaptureProvider() == provider))
                            app.currentCaptureProvider(provider);
                    }
                }

                ImGui::EndMenu();
            }

            vector<CVCaptureProviderIDSPeak*> providersIDSPeak;
            for (CVCaptureProvider* provider : AppArucoPen::instance().captureProviders())
            {
                if (app.arucoPen().trackingSystem()->isAcceptedProvider(provider) &&
                    typeid(*provider) == typeid(CVCaptureProviderIDSPeak))
                {
                    providersIDSPeak.push_back((CVCaptureProviderIDSPeak*)provider);
                }
            }

            if (!providersIDSPeak.empty())
            {
                auto gain  = (float)providersIDSPeak[0]->gain();
                auto gamma = (float)providersIDSPeak[0]->gamma();

                if (ImGui::SliderFloat("Gain", &gain, 1.0f, 3.0f, "%.2f"))
                    for (auto providerIDSPeak : providersIDSPeak)
                        providerIDSPeak->gain(gain);
                if (ImGui::SliderFloat("Gamma", &gamma, 0.3f, 3.0f, "%.2f"))
                    for (auto providerIDSPeak : providersIDSPeak)
                        providerIDSPeak->gamma(gamma);
            }

            if (ImGui::MenuItem("Calibrate Full"))
            {
                for (CVCaptureProvider* provider : app.captureProviders())
                {
                    if (app.arucoPen().trackingSystem()->isAcceptedProvider(provider))
                        app.arucoPen().trackingSystem()->calibrate(provider);
                }
            }

            if (dynamic_cast<TrackingSystemArucoCube*>(app.arucoPen().trackingSystem()))
            {
                if (ImGui::MenuItem("Calibrate Intrinsic (Current Camera)"))
                    s->onLoad(s, sv, SID_VideoCalibrateMain);

                if (ImGui::MenuItem("Calibrate Extrinsic (Current Camera)"))
                    AppArucoPenCalibrator::calcExtrinsicParams(AppArucoPen::instance().currentCaptureProvider());

                if (ImGui::MenuItem("Calibrate Extrinsic (All Cameras)"))
                {
                    for (CVCaptureProvider* provider : AppArucoPen::instance().captureProviders())
                    {
                        if (app.arucoPen().trackingSystem()->isAcceptedProvider(provider))
                        {
                            AppArucoPenCalibrator::calcExtrinsicParams(provider);
                        }
                    }
                }

                if (ImGui::MenuItem("Multi Tracking", nullptr, AppArucoPen::instance().doMultiTracking()))
                {
                    AppArucoPen::instance().doMultiTracking(!AppArucoPen::instance().doMultiTracking());
                }
            }

            if (ImGui::MenuItem("Start Evaluation"))
            {
                AppArucoPenEvaluator::instance().start(0.06f);
            }

            ImGui::MenuItem("Infos on Tracking", nullptr, &showInfosTracking);

            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
}
//-----------------------------------------------------------------------------
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
        AppArucoPenGui::showDockSpace = true;

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
            fs["showInfosTracking"] >> b;   AppArucoPenGui::showInfosTracking = b;
            fs["showDockSpace"] >> b;       AppArucoPenGui::showDockSpace = b;
            // clang-format on

            fs.release();
            SL_LOG("Config. loaded   : %s", fullPathAndFilename.c_str());
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
    fs << "showInfosTracking" << AppArucoPenGui::showInfosTracking;
    fs << "showDockSpace" << AppArucoPenGui::showDockSpace;

    fs.release();
    SL_LOG("Config. saved   : %s", fullPathAndFilename.c_str());
}
//-----------------------------------------------------------------------------
