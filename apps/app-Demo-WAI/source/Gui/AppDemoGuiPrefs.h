#ifndef GUI_PREFERENCES_H
#define GUI_PREFERENCES_H
#include <SL.h>

class GUIPreferences
{
    public:
    GUIPreferences();
    void setDPI(int dotsPerInch);
    void reset();

    void load();
    void save();

    int      dpi;
    SLbool   showAbout;            //!< Flag if about info should be shown
    SLbool   showChristoffel;      //!< Flag if Christoffel infos should be shown
    SLbool   showCredits;          //!< Flag if credits info should be shown
    SLbool   showHelp;             //!< Flag if help info should be shown
    SLbool   showHelpCalibration;  //!< Flag if calibration info should be shown
    SLbool   showInfosFrameworks;  //!< Flag if frameworks info should be shown
    SLbool   showInfosMapNodeTransform;
    SLbool   showInfosScene;       //!< Flag if scene info should be shown
    SLbool   showInfosSensors;     //!< Flag if device sensors info should be shown
    SLbool   showInfosTracking;  //!< Flag if frameworks info should be shown
    SLbool   showProperties;       //!< Flag if properties should be shown
    SLbool   showSceneGraph;       //!< Flag if scene graph should be shown
    SLbool   showStatsDebugTiming; //!< Flag if tracking info should be shown
    SLbool   showStatsScene;       //!< Flag if scene info should be shown
    SLbool   showStatsTiming;      //!< Flag if timing info should be shown
    SLbool   showStatsVideo;       //!< Flag if video info should be shown
    SLbool   showTransform;        //!< Flag if tranform dialog should be shown
    SLbool   showTrackedMapping;
    SLbool   showUIPrefs;          //!< Flag if UI preferences
    SLbool   showMapStorage;
    SLbool   showVideoStorage;
    SLstring configTime;
};

#endif
