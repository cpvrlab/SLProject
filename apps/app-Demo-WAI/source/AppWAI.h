//#############################################################################
//  File:      WAISceneView.h
//  Purpose:   Node transform test application that demonstrates all transform
//             possibilities of SLNode
//  Author:    Marc Wacker
//  Date:      July 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef APP_WAI_SCENE_VIEW
#define APP_WAI_SCENE_VIEW

#include "AppWAIScene.h"
#include <SLSceneView.h>
#include <SLPoints.h>
#include <SLPolyline.h>

#include <CVCalibration.h>
#include <WAIAutoCalibration.h>
#include <AppDirectories.h>
#include <AppDemoGuiPrefs.h>
#include <AppDemoGuiAbout.h>

#include <WAI.h>

#define LIVE_VIDEO 1

//-----------------------------------------------------------------------------
class WAIApp
{
    public:
    static int load(int width, int height, float scr2fbX, float scr2fbY, int dpi, AppWAIDirectories* dirs);

    static void onLoadWAISceneView(SLScene* s, SLSceneView* sv, SLSceneID sid);
    static bool update();
    static void updateMinNumOfCovisibles(int n);

    static void updateTrackingVisualization(const bool iKnowWhereIAm);

    static void renderMapPoints(std::string                      name,
                                const std::vector<WAIMapPoint*>& pts,
                                SLNode*&                         node,
                                SLPoints*&                       mesh,
                                SLMaterial*&                     material);

    static void renderKeyframes();
    static void renderGraphs();
    static void refreshTexture(cv::Mat *image);

    static void setupGUI();
    static void buildGUI(SLScene* s, SLSceneView* sv);
    static void openTest(std::string path);

    //! minimum number of covisibles for covisibility graph visualization
    static AppDemoGuiAbout*   aboutDial;
    static GUIPreferences     uiPrefs;
    static AppWAIDirectories* dirs;
    static WAI::WAI*          wai;
    static WAICalibration*    wc;
    static int                scrWidth;
    static int                scrHeight;
    static cv::VideoWriter*   videoWriter;
    static cv::VideoWriter*   videoWriterInfo;
    static WAI::ModeOrbSlam2* mode;
    static AppWAIScene*       waiScene;
    static bool               loaded;
    static SLGLTexture*       cpvrLogo;
    static SLGLTexture*       videoImage;

    static int   minNumOfCovisibles;
    static float meanReprojectionError;
    static bool  showKeyPoints;
    static bool  showKeyPointsMatched;
    static bool  showMapPC;
    static bool  showLocalMapPC;
    static bool  showMatchesPC;
    static bool  showKeyFrames;
    static bool  renderKfBackground;
    static bool  allowKfsAsActiveCam;
    static bool  showCovisibilityGraph;
    static bool  showSpanningTree;
    static bool  showLoopEdges;
};

#endif
