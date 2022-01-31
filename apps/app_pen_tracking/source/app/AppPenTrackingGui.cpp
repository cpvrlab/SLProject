//#############################################################################
//  File:      AppPenTrackingGui.cpp
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch, Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <app/AppPenTrackingGui.h>
#include <app/AppPenTracking.h>
#include <app/AppPenTrackingEvaluator.h>
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
SLbool   AppPenTrackingGui::showInfosTracking = false;
SLbool   AppPenTrackingGui::hideUI            = false;
SLbool   AppPenTrackingGui::showError         = false;
SLstring AppPenTrackingGui::errorString       = "";
//-----------------------------------------------------------------------------
void AppPenTrackingGui::build(SLProjectScene* s, SLSceneView* sv)
{
    PROFILE_FUNCTION();

    if (AppPenTrackingGui::hideUI)
    {
        buildMenuContext(s, sv);
    }
    else
    {
        buildMenuBar(s, sv);
        buildMenuContext(s, sv);

        if (showInfosTracking)
        {
            std::stringstream ss;
            TrackedPen&       pen = AppPenTracking::instance().arucoPen();
            ss << "Tip position             : " << pen.tipPosition().toString(", ", 2) << "\n";
            ss << "Head position            : " << pen.headTransform().translation().toString(", ", 2) << "\n";
            ss << "X Axis                   : " << pen.headTransform().axisX().toString(", ", 2) << "\n";
            ss << "Y Axis                   : " << pen.headTransform().axisY().toString(", ", 2) << "\n";
            ss << "Z Axis                   : " << pen.headTransform().axisZ().toString(", ", 2) << "\n";
            ss << "Measured Distance (Live) : " << Utils::toString(pen.liveDistance() * 100.0f, 2) << "cm\n";
            ss << "Measured Distance (Last) : " << Utils::toString(pen.lastDistance() * 100.0f, 2) << "cm\n";

            auto* trackingSystem = pen.trackingSystem();
            if (typeid(*trackingSystem) == typeid(TrackingSystemSpryTrack))
            {
                auto* providerSpryTrack = dynamic_cast<CVCaptureProviderSpryTrack*>(AppPenTracking::instance().currentCaptureProvider());
                if (providerSpryTrack)
                {
                    auto* trackingSystemSpryTrack = (TrackingSystemSpryTrack*)trackingSystem;

                    auto& device = providerSpryTrack->device();
                    for (SpryTrackMarker* marker : device.markers())
                    {
                        CVMatx44f objectViewMat = marker->objectViewMat();
                        CVMatx44f worldMat      = trackingSystemSpryTrack->extrinsicMat().inv() * objectViewMat * trackingSystemSpryTrack->markerMat();
                        SLVec3f   position(worldMat.val[3], worldMat.val[7], worldMat.val[11]);
                        SLVec3f   xAxis(worldMat.val[0], worldMat.val[4], worldMat.val[8]);
                        SLVec3f   yAxis(worldMat.val[1], worldMat.val[5], worldMat.val[9]);
                        SLVec3f   zAxis(worldMat.val[2], worldMat.val[6], worldMat.val[10]);

                        ss << "Marker " << marker->id() << "\n";
                        ss << "    Visible  : " << (marker->visible() ? "yes" : "no") << "\n";
                        ss << "    Position : " << position.toString(", ", 3) << "\n";
                        ss << "    X Axis   : " << xAxis.toString(", ", 3) << "\n";
                        ss << "    Y Axis   : " << yAxis.toString(", ", 3) << "\n";
                        ss << "    Z Axis   : " << zAxis.toString(", ", 3) << "\n";
                        ss << "    Error    : " << Utils::toString(marker->errorMM(), 3) << "\n";
                    }
                }
            }

            // Switch to fixed font
            ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
            ImGui::Begin("Tracking Information", &showInfosTracking, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
            ImGui::TextUnformatted(ss.str().c_str());
            ImGui::End();
            ImGui::PopFont();
        }

        if (showError)
        {
            SLfloat width   = (SLfloat)sv->viewportW() * 0.5f;
            SLfloat height  = (SLfloat)sv->viewportH() * 0.2f;
            SLfloat offsetX = ((SLfloat)sv->viewportW() - width) * 0.5f;
            SLfloat offsetY = ((SLfloat)sv->viewportH() - height) * 0.5f;
            ImGui::SetNextWindowSize(ImVec2(width, height), ImGuiCond_Always);
            ImGui::SetNextWindowPos(ImVec2(offsetX, offsetY), ImGuiCond_Always);

            ImGui::Begin("Error", &showError, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove);
            ImGui::SetCursorPosX(ImGui::GetWindowSize().x / 2 - ImGui::CalcTextSize(errorString.c_str()).x / 2);
            ImGui::SetCursorPosY(ImGui::GetWindowSize().y / 2 - ImGui::CalcTextSize(errorString.c_str()).y / 2);
            ImGui::TextUnformatted(errorString.c_str());
            ImGui::End();
        }
    }
}
//-----------------------------------------------------------------------------
void AppPenTrackingGui::buildMenuBar(SLProjectScene* s, SLSceneView* sv)
{
    PROFILE_FUNCTION();

    SLSceneID sid = AppDemo::sceneID;

    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::BeginMenu("View"))
            {
                if (ImGui::MenuItem("Live Camera", nullptr, sid == SID_VideoTrackArucoCubeMain))
                    s->onLoad(s, sv, SID_VideoTrackArucoCubeMain);
                if (ImGui::MenuItem("Virtual Render", nullptr, sid == SID_VirtualTrackedPen))
                    s->onLoad(s, sv, SID_VirtualTrackedPen);

                ImGui::EndMenu();
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Quit & Save"))
                slShouldClose(true);

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Pen Tracking"))
        {
            AppPenTracking& app = AppPenTracking::instance();

            if (ImGui::BeginMenu("Tracking Mode"))
            {
                TrackingSystem* ts = app.arucoPen().trackingSystem();

                if (ImGui::MenuItem("ArUco Cube", nullptr, typeid(*ts) == typeid(TrackingSystemArucoCube)))
                    runOrReportError([]
                                     { AppPenTracking::instance().arucoPen().trackingSystem(new TrackingSystemArucoCube()); });

                if (ImGui::MenuItem("SpryTrack", nullptr, typeid(*ts) == typeid(TrackingSystemSpryTrack)))
                    runOrReportError([]
                                     { AppPenTracking::instance().arucoPen().trackingSystem(new TrackingSystemSpryTrack()); });

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
            for (CVCaptureProvider* provider : AppPenTracking::instance().captureProviders())
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
                    {
                        runOrReportError([provider]
                                         { AppPenTracking::instance().arucoPen().trackingSystem()->calibrate(provider); });
                    }
                }
            }

            if (dynamic_cast<TrackingSystemArucoCube*>(app.arucoPen().trackingSystem()))
            {
                if (ImGui::MenuItem("Calibrate Intrinsic (Current Camera)"))
                    s->onLoad(s, sv, SID_VideoCalibrateMain);

                if (ImGui::MenuItem("Calibrate Extrinsic (Current Camera)"))
                    AppPenTrackingCalibrator::calcExtrinsicParams(AppPenTracking::instance().currentCaptureProvider());

                if (ImGui::MenuItem("Calibrate Extrinsic (All Cameras)"))
                {
                    for (CVCaptureProvider* provider : AppPenTracking::instance().captureProviders())
                    {
                        if (app.arucoPen().trackingSystem()->isAcceptedProvider(provider))
                        {
                            AppPenTrackingCalibrator::calcExtrinsicParams(provider);
                        }
                    }
                }

                if (ImGui::MenuItem("Multi Tracking", nullptr, AppPenTracking::instance().doMultiTracking()))
                {
                    AppPenTracking::instance().doMultiTracking(!AppPenTracking::instance().doMultiTracking());
                }
            }

            if (ImGui::MenuItem("Start Evaluation"))
            {
                AppPenTrackingEvaluator::instance().start();
            }

            ImGui::MenuItem("Infos on Tracking", nullptr, &showInfosTracking);

            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
}
//-----------------------------------------------------------------------------
void AppPenTrackingGui::buildMenuContext(SLProjectScene* s, SLSceneView* sv)
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

        if (AppPenTrackingGui::hideUI)
            if (ImGui::MenuItem("Show user interface"))
                AppPenTrackingGui::hideUI = false;

        if (!AppPenTrackingGui::hideUI)
            if (ImGui::MenuItem("Hide user interface"))
                AppPenTrackingGui::hideUI = true;

        ImGui::EndPopup();
    }
}
//-----------------------------------------------------------------------------
void AppPenTrackingGui::loadConfig(SLint dotsPerInch)
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
            fs["showInfosTracking"] >> b;   AppPenTrackingGui::showInfosTracking = b;
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
void AppPenTrackingGui::saveConfig()
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
        SL_EXIT_MSG("Exit in AppPenTrackingGui::saveConfig");
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
    fs << "showInfosTracking" << AppPenTrackingGui::showInfosTracking;

    fs.release();
    SL_LOG("Config. saved   : %s", fullPathAndFilename.c_str());
}
//-----------------------------------------------------------------------------
void AppPenTrackingGui::runOrReportError(const std::function<void()>& func)
{
    try
    {
        func();
    }
    catch (std::exception& e)
    {
        showError   = true;
        errorString = e.what();
    }
}