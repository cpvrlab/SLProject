//#############################################################################
//  File:      sl/SLInterface.h
//  Purpose:   Declaration of the main Scene Library C-Interface.
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLINTERFACE_H
#define SLINTERFACE_H

#include <SLEnums.h>
#include <SLGLEnums.h>

class SLAssetManager;
class SLScene;
class SLSceneView;
class SLInputManager;

//! \file SLInterface.h SLProject C-functions interface declaration.
/*! \file SLInterface.h
The SLInterface.h has all declarations of the SLProject C-Interface.
Only these functions should called by the OS-dependent GUI applications.
These functions can be called from any C, C++ or ObjectiveC GUI framework or
by a native API such as Java Native Interface (JNI).
See the implementation for more information.<br>
 <br>
 See examples usages in:
 - app_demo_slproject/glfw:    in AppDemoMainGLFW.cpp
 - app-Demo-SLProject/android: in AppDemoAndroidJNI.cpp
 - app_demo_slproject/ios:     in ViewController.m
 <br>
*/
//-----------------------------------------------------------------------------
void slCreateAppAndScene(SLVstring&      cmdLineArgs,
                         const SLstring& dataPath,
                         const SLstring& shaderPath,
                         const SLstring& modelPath,
                         const SLstring& texturePath,
                         const SLstring& fontPath,
                         const SLstring& videoPath,
                         const SLstring& configPath,
                         const SLstring& applicationName,
                         void*           onSceneLoadCallback = nullptr);

SLint slCreateSceneView(SLAssetManager* am,
                        SLScene*        scene,
                        int             screenWidth,
                        int             screenHeight,
                        int             dotsPerInch,
                        SLSceneID       initScene,
                        void*           onWndUpdateCallback,
                        void*           onSelectNodeMeshCallback = nullptr,
                        void*           onNewSceneViewCallback   = nullptr,
                        void*           onImGuiBuild             = nullptr,
                        void*           onImGuiLoadConfig        = nullptr,
                        void*           onImGuiSaveConfig        = nullptr);

SLSceneView* slNewSceneView(SLScene* s, int dotsPerInch, SLInputManager& inputManager);
bool         slShouldClose();
void         slShouldClose(bool val);
void         slTerminate();
void         slResize(int sceneViewIndex, int width, int height);
bool         slUpdateParallelJob();
bool         slPaintAllViews();

void slMouseDown(int sceneViewIndex, SLMouseButton button, int x, int y, SLKey modifier);
void slMouseMove(int sceneViewIndex, int x, int y);
void slMouseUp(int sceneViewIndex, SLMouseButton button, int x, int y, SLKey modifier);
void slDoubleClick(int sceneViewIndex, SLMouseButton button, int x, int y, SLKey modifier);
void slTouch2Down(int sceneViewIndex, int x1, int y1, int x2, int y2);
void slTouch2Move(int sceneViewIndex, int x1, int y1, int x2, int y2);
void slTouch2Up(int sceneViewIndex, int x1, int y1, int x2, int y2);
void slTouch3Down(int sceneViewIndex, int x, int y);
void slTouch3Move(int sceneViewIndex, int x, int y);
void slTouch3Up(int sceneViewIndex, int x, int y);
void slMouseWheel(int sceneViewIndex, int pos, SLKey modifier);
void slKeyPress(int sceneViewIndex, SLKey key, SLKey modifier);
void slKeyRelease(int sceneViewIndex, SLKey key, SLKey modifier);
void slCharInput(int sceneViewIndex, unsigned int character);

bool   slUsesRotation();
void   slRotationQUAT(float quatX, float quatY, float quatZ, float quatW);
bool   slUsesLocation();
void   slLocationLatLonAlt(double latitudeDEG, double longitudeDEG, double altitudeM, float accuracyM);
string slGetWindowTitle(int sceneViewIndex);
void   slSetupExternalDir(const SLstring& externalDirPath);
void   slSetDeviceParameter(const SLstring& parameter, SLstring value);
//-----------------------------------------------------------------------------
#endif // SLINTERFACE_H
