#include <SL.h>

#include <SLApplication.h>
#include <SLGLImGui.h>
#include <AppDemoGuiPrefs.h>
#include <opencv2/core/core.hpp>
#include <Utils.h>

GUIPreferences::GUIPreferences()
{
    dpi = 200;
    reset();
};

void GUIPreferences::reset()
{
    showAbout                 = true;
    showChristoffel           = false;
    showCredits               = false;
    showHelp                  = false;
    showHelpCalibration       = false;
    showInfosFrameworks       = false;
    showInfosMapNodeTransform = false;
    showInfosScene            = false;
    showInfosSensors          = false;
    showProperties            = false;
    showSceneGraph            = false;
    showStatsDebugTiming      = false;
    showStatsScene            = false;
    showStatsTiming           = false;
    showStatsVideo            = false;
    showTransform             = false;
    showTrackedMapping        = false;
    showUIPrefs               = false;
    showMapStorage            = false;
    showVideoStorage          = false;
    showVideoLoad             = false;
    showTestSettings          = false;
    showTestWriter            = false;
    showSlamParam             = false;
};

void GUIPreferences::setDPI(int dotsPerInch)
{
    dpi = dotsPerInch;
}

void GUIPreferences::load()
{
    ImGuiStyle& style               = ImGui::GetStyle();
    SLstring    fullPathAndFilename = SLApplication::configPath +
                                   SLApplication::name + ".yml";

    if (!Utils::fileExists(fullPathAndFilename))
    {
        // Scale for proportional and fixed size fonts
        SLfloat dpiScaleProp  = dpi / 120.0f;
        SLfloat dpiScaleFixed = dpi / 142.0f;

        // Default settings for the first time
        SLGLImGui::fontPropDots  = std::max(16.0f * dpiScaleProp, 16.0f);
        SLGLImGui::fontFixedDots = std::max(13.0f * dpiScaleFixed, 13.0f);

        // Store dialog show states
        reset();

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

    cv::FileStorage fs;
    try
    {
        fs.open(fullPathAndFilename, cv::FileStorage::READ);
        if (fs.isOpened())
        {
            // clang-format off
            SLint  i;
            SLbool b;
            fs["configTime"] >>             configTime;
            fs["fontPropDots"] >> i;        SLGLImGui::fontPropDots = (SLfloat)i;
            fs["fontFixedDots"] >> i;       SLGLImGui::fontFixedDots = (SLfloat)i;
            fs["ItemSpacingX"] >> i;        style.ItemSpacing.x = (SLfloat)i;
            fs["ItemSpacingY"] >> i;        style.ItemSpacing.y = (SLfloat)i;
            style.WindowPadding.x = style.FramePadding.x = style.ItemInnerSpacing.x = style.ItemSpacing.x;
            style.WindowPadding.y = style.FramePadding.y = style.ItemInnerSpacing.y = style.ItemSpacing.y;
            fs["ScrollbarSize"] >> i; style.ScrollbarSize = (SLfloat)i;
            style.ScrollbarRounding = std::floor(style.ScrollbarSize / 2);
            fs["sceneID"] >> i;             SLApplication::sceneID = (SLSceneID)i;
            fs["showInfosScene"] >> b;      showInfosScene = b;
            fs["showStatsTiming"] >> b;     showStatsTiming = b;
            fs["showStatsMemory"] >> b;     showStatsScene = b;
            fs["showStatsVideo"] >> b;      showStatsVideo = b;
            fs["showInfosFrameworks"] >> b; showInfosFrameworks = b;
            fs["showInfosSensors"] >> b;    showInfosSensors = b;
            fs["showSceneGraph"] >> b;      showSceneGraph = b;
            fs["showProperties"] >> b;      showProperties = b;
            fs["showUIPrefs"] >> b;         showUIPrefs = b;
            fs["showTestSettings"] >> b;    showTestSettings = b;
            fs["showTestWriter"] >> b;      showTestWriter = b;
            // clang-format on

            fs.release();
            SL_LOG("Config. loaded  : %s\n", fullPathAndFilename.c_str());
            SL_LOG("Config. date    : %s\n", configTime.c_str());
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
    if (dpi > 300)
    {
        if (SLGLImGui::fontPropDots < 16.1f &&
            SLGLImGui::fontFixedDots < 13.1)
        {
            // Scale for proportional and fixed size fonts
            SLfloat dpiScaleProp  = dpi / 120.0f;
            SLfloat dpiScaleFixed = dpi / 142.0f;

            // Default settings for the first time
            SLGLImGui::fontPropDots  = std::max(16.0f * dpiScaleProp, 16.0f);
            SLGLImGui::fontFixedDots = std::max(13.0f * dpiScaleFixed, 13.0f);
        }
    }
}

void GUIPreferences::save()
{
    ImGuiStyle& style               = ImGui::GetStyle();
    SLstring    fullPathAndFilename = SLApplication::configPath +
                                   SLApplication::name + ".yml";
    cv::FileStorage fs(fullPathAndFilename, cv::FileStorage::WRITE);

    if (!fs.isOpened())
    {
        SL_LOG("Failed to open file for writing: %s", fullPathAndFilename.c_str());
        SL_EXIT_MSG("Exit in AppDemoGui::saveConfig");
        return;
    }

    fs << "configTime" << Utils::getLocalTimeString();
    fs << "fontPropDots" << (SLint)SLGLImGui::fontPropDots;
    fs << "fontFixedDots" << (SLint)SLGLImGui::fontFixedDots;
    fs << "sceneID" << (SLint)SLApplication::sceneID;
    fs << "ItemSpacingX" << (SLint)style.ItemSpacing.x;
    fs << "ItemSpacingY" << (SLint)style.ItemSpacing.y;
    fs << "ScrollbarSize" << (SLint)style.ScrollbarSize;
    fs << "showStatsTiming" << showStatsTiming;
    fs << "showStatsMemory" << showStatsScene;
    fs << "showStatsVideo" << showStatsVideo;
    fs << "showInfosFrameworks" << showInfosFrameworks;
    fs << "showInfosScene" << showInfosScene;
    fs << "showInfosSensors" << showInfosSensors;
    fs << "showSceneGraph" << showSceneGraph;
    fs << "showProperties" << showProperties;
    fs << "showUIPrefs" << showUIPrefs;
    fs << "showTestSettings" << showTestSettings;
    fs << "showTestWriter" << showTestWriter;

    fs.release();
    SL_LOG("Config. saved   : %s\n", fullPathAndFilename.c_str());
}
