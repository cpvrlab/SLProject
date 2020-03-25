#include <SL.h>

#include <SLApplication.h>
#include <GUIPreferences.h>
#include <opencv2/core/core.hpp>
#include <Utils.h>
#include <imgui.h>

GUIPreferences::GUIPreferences(int dotsPerInch)
  : _dpi(dotsPerInch)
{
}

void GUIPreferences::load(std::string fileName, ImGuiStyle& style)
{
    //ImGuiStyle& style = ImGui::GetStyle();
    //SLstring    fullPathAndFilename = SLApplication::configPath +
    //                               SLApplication::name + ".yml";

    if (!Utils::fileExists(fileName))
    {
        // Scale for proportional and fixed size fonts
        SLfloat dpiScaleProp  = _dpi / 120.0f;
        SLfloat dpiScaleFixed = _dpi / 142.0f;

        // Default settings for the first time
        fontPropDots  = std::max(16.0f * dpiScaleProp, 16.0f);
        fontFixedDots = std::max(13.0f * dpiScaleFixed, 13.0f);

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
        fs.open(fileName, cv::FileStorage::READ);
        if (fs.isOpened())
        {
            SLint  i;
            SLbool b;

            if (!fs["configTime"].empty())
                fs["configTime"] >> configTime;

            if (!fs["fontPropDots"].empty())
                fs["fontPropDots"] >> i;
            fontPropDots = (SLfloat)i;

            if (!fs["fontFixedDots"].empty())
                fs["fontFixedDots"] >> i;
            fontFixedDots = (SLfloat)i;

            if (!fs["ItemSpacingX"].empty())
                fs["ItemSpacingX"] >> i;
            style.ItemSpacing.x = (SLfloat)i;

            if (!fs["ItemSpacingY"].empty())
                fs["ItemSpacingY"] >> i;
            style.ItemSpacing.y   = (SLfloat)i;
            style.WindowPadding.x = style.FramePadding.x = style.ItemInnerSpacing.x = style.ItemSpacing.x;
            style.WindowPadding.y = style.FramePadding.y = style.ItemInnerSpacing.y = style.ItemSpacing.y;

            if (!fs["ScrollbarSize"].empty())
                fs["ScrollbarSize"] >> i;
            style.ScrollbarSize     = (SLfloat)i;
            style.ScrollbarRounding = std::floor(style.ScrollbarSize / 2);

            //slam menu
            if (!fs["showSlamParam"].empty())
                fs["showSlamParam"] >> showSlamParam;
            if (!fs["showTrackedMapping"].empty())
                fs["showTrackedMapping"] >> showTrackedMapping;

            //video menu
            if (!fs["showVideoControls"].empty())
                fs["showVideoControls"] >> showVideoControls;
            if (!fs["showVideoStorage"].empty())
                fs["showVideoStorage"] >> showVideoStorage;

            //map menu
            if (!fs["showInfosMapNodeTransform"].empty())
                fs["showInfosMapNodeTransform"] >> showInfosMapNodeTransform;

            //experiments menu
            if (!fs["showSlamLoad"].empty())
                fs["showSlamLoad"] >> showSlamLoad;
            if (!fs["showTestSettings"].empty())
                fs["showTestSettings"] >> showTestSettings;
            if (!fs["showTestWriter"].empty())
                fs["showTestWriter"] >> showTestWriter;

            //infos menu
            if (!fs["showInfosScene"].empty())
                fs["showInfosScene"] >> showInfosScene;
            if (!fs["showStatsTiming"].empty())
                fs["showStatsTiming"] >> showStatsTiming;
            if (!fs["showStatsDebugTiming"].empty())
                fs["showStatsDebugTiming"] >> showStatsDebugTiming;
            if (!fs["showStatsVideo"].empty())
                fs["showStatsVideo"] >> showStatsVideo;
            if (!fs["showSceneGraph"].empty())
                fs["showSceneGraph"] >> showSceneGraph;
            if (!fs["showProperties"].empty())
                fs["showProperties"] >> showProperties;
            if (!fs["showTransform"].empty())
                fs["showTransform"] >> showTransform;
            if (!fs["showInfosSensors"].empty())
                fs["showInfosSensors"] >> showInfosSensors;
            if (!fs["showInfosFrameworks"].empty())
                fs["showInfosFrameworks"] >> showInfosFrameworks;
            if (!fs["showInfosTracking"].empty())
                fs["showInfosTracking"] >> showInfosTracking;
            if (!fs["showUIPrefs"].empty())
                fs["showUIPrefs"] >> showUIPrefs;
            if (!fs["showAbout"].empty())
                fs["showAbout"] >> showAbout;

            //dialogue AppDemoGuiInfosTracking
            if (!fs["minNumOfCovisibles"].empty())
                fs["minNumOfCovisibles"] >> minNumOfCovisibles;
            if (!fs["showKeyPoints"].empty())
                fs["showKeyPoints"] >> showKeyPoints;
            if (!fs["showKeyPointsMatched"].empty())
                fs["showKeyPointsMatched"] >> showKeyPointsMatched;
            if (!fs["showMapPC"].empty())
                fs["showMapPC"] >> showMapPC;
            if (!fs["showLocalMapPC"].empty())
                fs["showLocalMapPC"] >> showLocalMapPC;
            if (!fs["showMatchesPC"].empty())
                fs["showMatchesPC"] >> showMatchesPC;
            if (!fs["showKeyFrames"].empty())
                fs["showKeyFrames"] >> showKeyFrames;
            if (!fs["renderKfBackground"].empty())
                fs["renderKfBackground"] >> renderKfBackground;
            if (!fs["allowKfsAsActiveCam"].empty())
                fs["allowKfsAsActiveCam"] >> allowKfsAsActiveCam;
            if (!fs["showCovisibilityGraph"].empty())
                fs["showCovisibilityGraph"] >> showCovisibilityGraph;
            if (!fs["showSpanningTree"].empty())
                fs["showSpanningTree"] >> showSpanningTree;
            if (!fs["showLoopEdges"].empty())
                fs["showLoopEdges"] >> showLoopEdges;

            fs.release();
            SL_LOG("Config. loaded  : %s\n", fileName.c_str());
            SL_LOG("Config. date    : %s\n", configTime.c_str());
            SL_LOG("fontPropDots    : %f\n", fontPropDots);
            SL_LOG("fontFixedDots   : %f\n", fontFixedDots);
        }
        else
        {
            SL_LOG("****** Failed to open file for reading: %s", fileName.c_str());
        }
    }
    catch (...)
    {
        SL_LOG("****** Parsing of file failed: %s", fileName.c_str());
    }

    // check font sizes for HDPI displays
    if (_dpi > 300)
    {
        if (fontPropDots < 16.1f &&
            fontFixedDots < 13.1)
        {
            // Scale for proportional and fixed size fonts
            SLfloat dpiScaleProp  = _dpi / 120.0f;
            SLfloat dpiScaleFixed = _dpi / 142.0f;

            // Default settings for the first time
            fontPropDots  = std::max(16.0f * dpiScaleProp, 16.0f);
            fontFixedDots = std::max(13.0f * dpiScaleFixed, 13.0f);
        }
    }
}

void GUIPreferences::save(std::string fileName, ImGuiStyle& style)
{
    //ImGuiStyle& style = ImGui::GetStyle();
    //SLstring    fullPathAndFilename = SLApplication::configPath +
    //                               SLApplication::name + ".yml";
    cv::FileStorage fs(fileName, cv::FileStorage::WRITE);

    if (!fs.isOpened())
    {
        SL_LOG("Failed to open file for writing: %s", fileName.c_str());
        SL_EXIT_MSG("Exit in AppDemoGui::saveConfig");
        return;
    }

    fs << "configTime" << Utils::getLocalTimeString();
    fs << "fontPropDots" << (SLint)fontPropDots;
    fs << "fontFixedDots" << (SLint)fontFixedDots;
    fs << "sceneID" << (SLint)SLApplication::sceneID;
    fs << "ItemSpacingX" << (SLint)style.ItemSpacing.x;
    fs << "ItemSpacingY" << (SLint)style.ItemSpacing.y;
    fs << "ScrollbarSize" << (SLint)style.ScrollbarSize;

    //slam menu
    fs << "showSlamParam" << showSlamParam;
    fs << "showTrackedMapping" << showTrackedMapping;

    //video menu
    fs << "showVideoControls" << showVideoControls;
    fs << "showVideoStorage" << showVideoStorage;

    //map menu
    fs << "showInfosMapNodeTransform" << showInfosMapNodeTransform;

    //experiments menu
    fs << "showSlamLoad" << showSlamLoad;
    fs << "showTestSettings" << showTestSettings;
    fs << "showTestWriter" << showTestWriter;

    //infos menu
    fs << "showInfosScene" << showInfosScene;
    fs << "showStatsTiming" << showStatsTiming;
    fs << "showStatsDebugTiming" << showStatsDebugTiming;
    fs << "showStatsVideo" << showStatsVideo;
    fs << "showSceneGraph" << showSceneGraph;
    fs << "showProperties" << showProperties;
    fs << "showTransform" << showTransform;
    fs << "showInfosSensors" << showInfosSensors;
    fs << "showInfosFrameworks" << showInfosFrameworks;
    fs << "showInfosTracking" << showInfosTracking;
    fs << "showUIPrefs" << showUIPrefs;
    fs << "showAbout" << showAbout;

    //dialogue AppDemoGuiInfosTracking
    fs << "minNumOfCovisibles" << minNumOfCovisibles;
    fs << "showKeyPoints" << showKeyPoints;
    fs << "showKeyPointsMatched" << showKeyPointsMatched;
    fs << "showMapPC" << showMapPC;
    fs << "showLocalMapPC" << showLocalMapPC;
    fs << "showMatchesPC" << showMatchesPC;
    fs << "showKeyFrames" << showKeyFrames;
    fs << "renderKfBackground" << renderKfBackground;
    fs << "allowKfsAsActiveCam" << allowKfsAsActiveCam;
    fs << "showCovisibilityGraph" << showCovisibilityGraph;
    fs << "showSpanningTree" << showSpanningTree;
    fs << "showLoopEdges" << showLoopEdges;

    fs.release();
    SL_LOG("Config. saved   : %s\n", fileName.c_str());
}
