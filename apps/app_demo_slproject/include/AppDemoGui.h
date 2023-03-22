//#############################################################################
//  File:      AppDemoGui.h
//  Date:      Summer 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGUIDEMO_H
#define SLGUIDEMO_H

#include <SL.h>
#include <SLTransformNode.h>
#include <assimp/ProgressHandler.hpp>

class SLScene;
class SLSceneView;
class SLNode;
class SLGLTexture;
class SLTexColorLUT;

//-----------------------------------------------------------------------------
//! ImGui UI class for the UI of the demo applications
/* The UI is completely built within this class by calling build function
AppDemoGui::build. This build function is passed in the slCreateSceneView and
it is called in SLSceneView::onPaint in every frame.<br>
The entire UI is configured and built on every frame. That is why it is called
"Im" for immediate. See also the SLGLImGui class to see how it is minimal
integrated in the SLProject.<br>
*/
class AppDemoGui
{
public:
    static void clear();
    static void build(SLScene* s, SLSceneView* sv);
    static void buildMenuBar(SLScene* s, SLSceneView* sv);
    static void buildMenuEdit(SLScene* s, SLSceneView* sv);
    static void buildMenuContext(SLScene* s, SLSceneView* sv);
    static void buildSceneGraph(SLScene* s);
    static void addSceneGraphNode(SLScene* s, SLNode* node);
    static void buildProperties(SLScene* s, SLSceneView* sv);
    static void showTexInfos(SLGLTexture* tex);
    static void loadConfig(SLint dotsPerInch);
    static void saveConfig();
    static void showLUTColors(SLTexColorLUT* lut);
    static void setActiveNamedLocation(int          locIndex,
                                       SLSceneView* sv,
                                       SLVec3f      lookAtPoint = SLVec3f::ZERO);

    static SLstring    configTime;          //!< Time of stored configuration
    static SLstring    infoAbout;           //!< About info string
    static SLstring    infoCredits;         //!< Credits info string
    static SLstring    infoHelp;            //!< Help info string
    static SLstring    infoCalibrate;       //!< Calibration info string
    static SLbool      hideUI;              //!< Flag if menubar should be shown
    static SLbool      showProgress;        //!< Flag if about info should be shown
    static SLbool      showDockSpace;       //!< Flag if dock space should be enabled
    static SLbool      showAbout;           //!< Flag if about info should be shown
    static SLbool      showHelp;            //!< Flag if help info should be shown
    static SLbool      showHelpCalibration; //!< Flag if calibration info should be shown
    static SLbool      showCredits;         //!< Flag if credits info should be shown
    static SLbool      showStatsTiming;     //!< Flag if timing info should be shown
    static SLbool      showStatsScene;      //!< Flag if scene info should be shown
    static SLbool      showStatsVideo;      //!< Flag if video info should be shown
    static SLbool      showStatsWAI;        //!< Flag if WAI info should be shown
    static SLbool      showImGuiMetrics;    //!< Flag if imgui metrics infor should be shown
    static SLbool      showInfosSensors;    //!< Flag if device sensors info should be shown
    static SLbool      showInfosDevice;     //!< Flag if device info should be shown
    static SLbool      showInfosScene;      //!< Flag if scene info should be shown
    static SLbool      showSceneGraph;      //!< Flag if scene graph should be shown
    static SLbool      showProperties;      //!< Flag if properties should be shown
    static SLbool      showErlebAR;         //!< Flag if Christoffel infos should be shown
    static SLbool      showUIPrefs;         //!< Flag if UI preferences
    static SLbool      showTransform;       //!< Flag if transform dialog should be shown
    static SLbool      showDateAndTime;     //!< Flag if date-time dialog should be shown
    static std::time_t adjustedTime;        //!< Adjusted GUI time for sun setting (default 0)

private:
    static void   setTransformEditMode(SLScene*       s,
                                       SLSceneView*   sv,
                                       SLNodeEditMode editMode);
    static void   removeTransformNode(SLScene* s);
    static void   showHorizon(SLScene*     s,
                              SLSceneView* sv);
    static void   hideHorizon(SLScene* s);
    static SLbool _horizonVisuEnabled;

    static void loadSceneWithLargeModel(SLScene*     s,
                                        SLSceneView* sv,
                                        string       downloadFilename,
                                        string       filenameToLoad,
                                        SLSceneID    sceneIDToLoad);
    static void downloadModelAndLoadScene(SLScene*     s,
                                          SLSceneView* sv,
                                          string       downloadFilename,
                                          string       urlFolder,
                                          string       dstFolder,
                                          string       filenameToLoad,
                                          SLSceneID    sceneIDToLoad);
};
//-----------------------------------------------------------------------------
#endif
