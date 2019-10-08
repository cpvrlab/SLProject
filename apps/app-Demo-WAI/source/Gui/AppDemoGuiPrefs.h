#ifndef GUI_PREFERENCES_H
#define GUI_PREFERENCES_H
#include <SL.h>

class GUIPreferences
{
    public:
    GUIPreferences();
    void setDPI(int dotsPerInch);

    void load();
    void save();

    //slam menu
    bool showSlamParam      = false;
    bool showTrackedMapping = false;
    bool showMarker         = false;

    //video menu
    bool showVideoControls = false;
    bool showVideoStorage  = false;

    //map menu
    bool showMapStorage            = false;
    bool showInfosMapNodeTransform = false;

    //experiments menu
    bool showSlamLoad     = false;
    bool showTestSettings = false;
    bool showTestWriter   = false;

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

    SLstring configTime;

    //error dialog
    bool showError = false;

    private:
    int dpi = 200;
};

#endif
