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
    SLbool showSlamParam      = false;
    SLbool showTrackedMapping = false;

    //video menu
    SLbool showVideoControls = false;
    SLbool showVideoStorage  = false;

    //map menu
    SLbool showMapStorage            = false;
    SLbool showInfosMapNodeTransform = false;

    //experiments menu
    SLbool showSlamLoad     = false;
    SLbool showTestSettings = false;
    SLbool showTestWriter   = false;

    //infos menu
    SLbool showInfosScene       = false; //!< Flag if scene info should be shown
    SLbool showStatsTiming      = false; //!< Flag if timing info should be shown
    SLbool showStatsDebugTiming = false; //!< Flag if tracking info should be shown
    SLbool showStatsVideo       = false; //!< Flag if video info should be shown
    SLbool showSceneGraph       = false; //!< Flag if scene graph should be shown
    SLbool showProperties       = false; //!< Flag if properties should be shown
    SLbool showTransform        = false; //!< Flag if tranform dialog should be shown
    SLbool showInfosSensors     = false; //!< Flag if device sensors info should be shown
    SLbool showInfosFrameworks  = false; //!< Flag if frameworks info should be shown
    SLbool showInfosTracking    = false; //!< Flag if frameworks info should be shown
    SLbool showUIPrefs          = false; //!< Flag if UI preferences
    SLbool showAbout            = true;  //!< Flag if about info should be shown

    SLstring configTime;

    //error dialog
    SLbool showError = false;

    private:
    int dpi = 200;
};

#endif
