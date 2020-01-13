#ifndef GUI_PREFERENCES_H
#define GUI_PREFERENCES_H

class ImGuiStyle;

class GUIPreferences
{
public:
    GUIPreferences(int dotsPerInch);

    void load(std::string fileName, ImGuiStyle& style);
    void save(std::string fileName, ImGuiStyle& style);

    //slam menu
    bool showSlamParam      = false;
    bool showTrackedMapping = false;
    bool showSlamLoad       = false;

    //video menu
    bool showVideoControls = false;
    bool showVideoStorage  = false;

    //map menu
    bool showInfosMapNodeTransform = false;

    //infos menu
    bool showInfosScene       = false; //!< Flag if scene info should be shown
    bool showStatsTiming      = false; //!< Flag if timing info should be shown
    bool showStatsDebugTiming = false; //!< Flag if tracking info should be shown
    bool showStatsVideo       = false; //!< Flag if video info should be shown
    bool showSceneGraph       = false; //!< Flag if scene graph should be shown
    bool showProperties       = false; //!< Flag if properties should be shown
    bool showTransform        = false; //!< Flag if tranform dialog should be shown
    bool showInfosSensors     = false; //!< Flag if device sensors info should be shown
    bool showInfosFrameworks  = false; //!< Flag if frameworks info should be shown
    bool showInfosTracking    = false; //!< Flag if frameworks info should be shown
    bool showUIPrefs          = false; //!< Flag if UI preferences
    bool showAbout            = true;  //!< Flag if about info should be shown

    //dialogue AppDemoGuiInfosTracking
    int  minNumOfCovisibles    = 50;
    bool showKeyPoints         = true;
    bool showKeyPointsMatched  = true;
    bool showMapPC             = true;
    bool showLocalMapPC        = false;
    bool showMatchesPC         = true;
    bool showKeyFrames         = true;
    bool renderKfBackground    = false;
    bool allowKfsAsActiveCam   = false;
    bool showCovisibilityGraph = false;
    bool showSpanningTree      = true;
    bool showLoopEdges         = true;

    std::string configTime;

    //error dialog
    bool showError = false;

    float fontPropDots  = 16.0f; //!< Default font size of proportional font
    float fontFixedDots = 13.0f; //!< Default font size of fixed size font

private:
    int _dpi = 200;
};

#endif
