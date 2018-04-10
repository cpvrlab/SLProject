//#############################################################################
//  File:      SL/SLInterface.h
//  Purpose:   Declaration of the main Scene Library C-Interface.
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLINTERFACE_H
#define SLINTERFACE_H

#include <SLEnums.h>
#include <SLGLEnums.h>

//! \file SLInterface.h SLProject C-functions interface declaration.
/*! \file SLInterface.h
The SLInterface.h has all declarations of the SLProject C-Interface.
Only these functions should called by the OS-dependent GUI applications.
These functions can be called from any C, C++ or ObjectiveC GUI framework or
by a native API such as Java Native Interface (JNI).
See the implementation for more information.<br>
 <br>
 See examples usages in:
 - app-Demo-GLFW:    in AppDemoMainGLFW.cpp
 - app-Demo-Android: in AppDemoAndroidJNI.cpp
 - app-Demo-iOS:     in ViewController.m
 <br>
*/
//-----------------------------------------------------------------------------
void    slCreateAppAndScene     (SLVstring& cmdLineArgs,
                                 SLstring shaderPath,
                                 SLstring modelPath,
                                 SLstring texturePath,
                                 SLstring videoPath,
                                 SLstring fontPath,
                                 SLstring calibrationPath,
                                 SLstring configPath,
                                 SLstring applicationName,
                                 void*    onSceneLoadCallback = 0);

int     slCreateSceneView       (int screenWidth,
                                 int screenHeight,
                                 int dotsPerInch,
                                 SLSceneID initScene,
                                 void* onWndUpdateCallback,
                                 void* onSelectNodeMeshCallback = 0,
                                 void* onNewSceneViewCallback = 0,
                                 void* onImGuiBuild = 0);

int     slNewSceneView          ();
bool    slShouldClose           ();
void    slShouldClose           (bool val);
void    slTerminate             ();
void    slResize                (int sceneViewIndex, int width, int height);
bool    slUpdateAndPaint        (int sceneViewIndex);
void    slMouseDown             (int sceneViewIndex, SLMouseButton button, int x, int y, SLKey modifier);
void    slMouseMove             (int sceneViewIndex, int x, int y);
void    slMouseUp               (int sceneViewIndex, SLMouseButton button, int x, int y, SLKey modifier);
void    slDoubleClick           (int sceneViewIndex, SLMouseButton button, int x, int y, SLKey modifier);
void    slLongTouch             (int sceneViewIndex, int x, int y);
void    slTouch2Down            (int sceneViewIndex, int x1, int y1, int x2, int y2);
void    slTouch2Move            (int sceneViewIndex, int x1, int y1, int x2, int y2);
void    slTouch2Up              (int sceneViewIndex, int x1, int y1, int x2, int y2);
void    slMouseWheel            (int sceneViewIndex, int pos, SLKey modifier);
void    slKeyPress              (int sceneViewIndex, SLKey key, SLKey modifier);
void    slKeyRelease            (int sceneViewIndex, SLKey key, SLKey modifier);
void    slCharInput             (int sceneViewIndex, unsigned int character);
bool    slUsesRotation          ();
void    slRotationQUAT          (float quatX, float quatY, float quatZ, float quatW);
bool    slUsesLocation          ();
void    slLocationLLA           (double latitudeDEG, double longitudeDEG, double altitudeM, float accuracyM);
string  slGetWindowTitle        (int sceneViewIndex);
int     slGetVideoType          ();
int     slGetVideoSizeIndex     ();
void    slGrabVideoFileFrame    ();
void    slCopyVideoImage        (int srcW, int srcH, SLPixelFormat glFormat, SLuchar* data, bool isContinuous);
void    slCopyVideoYUVPlanes    (int srcW, int srcH,
                                 SLuchar* y, int ySize, int yPixStride, int yLineStride,
                                 SLuchar* u, int uSize, int uPixStride, int uLineStride,
                                 SLuchar* v, int vSize, int vPixStride, int vLineStride);
//-----------------------------------------------------------------------------
#endif // SLINTERFACE_H
