//#############################################################################
//  File:      SL/SLInterface.h
//  Purpose:   Delcaration of the main Scene Library C-Interface. Only these 
//             functions should called by the OS-dependend GUI applications. 
//             These functions can be called from any C, C++ or ObjectiveC GUI 
//             framework or by a native API such as Java Native Interface 
//             (JNI). See the implementation for more information.
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLINTERFACE_H
#define SLINTERFACE_H

#include <stdafx.h>
#include <SLEnums.h>

//-----------------------------------------------------------------------------
void    slCreateScene       (SLstring shaderPath,
                             SLstring modelPath,
                             SLstring texturePath);
int     slCreateSceneView   (int screenWidth,
                             int screenHeight,
                             int dotsPerInch,
                             SLCmd initScene,  
                             SLVstring& cmdLineArgs,
                             void* onWndUpdateCallback,
                             void* onSelectNodeMeshCallback = 0,
                             void* onNewSceneViewCallback = 0,
                             void* onShowSystemCursorCallback = 0);
int     slNewSceneView      ();
bool    slShouldClose       ();
void    slShouldClose       (bool val);
void    slTerminate         ();
void    slResize            (int sceneViewIndex, int width, int height);
bool    slPaint             (int sceneViewIndex);
bool    slMouseDown         (int sceneViewIndex, SLMouseButton button, int x, int y, SLKey modifier);
bool    slMouseMove         (int sceneViewIndex, int x, int y);
bool    slMouseUp           (int sceneViewIndex, SLMouseButton button, int x, int y, SLKey modifier);
bool    slDoubleClick       (int sceneViewIndex, SLMouseButton button, int x, int y, SLKey modifier);
bool    slTouch2Down        (int sceneViewIndex, int x1, int y1, int x2, int y2);
bool    slTouch2Move        (int sceneViewIndex, int x1, int y1, int x2, int y2);
bool    slTouch2Up          (int sceneViewIndex, int x1, int y1, int x2, int y2);
bool    slMouseWheel        (int sceneViewIndex, int pos, SLKey modifier);
bool    slKeyPress          (int sceneViewIndex, SLKey key, SLKey modifier);
bool    slKeyRelease        (int sceneViewIndex, SLKey key, SLKey modifier);
bool    slCommand           (int sceneViewIndex, SLCmd command);
void    slRotationPYR       (int sceneViewIndex, float pitchRAD, float yawRAD, float rollRAD);
void    slRotationQUAT      (int sceneViewIndex, float angleRAD, float axisX, float axisY, float axisZ);
string  slGetWindowTitle    (int sceneViewIndex);
//-----------------------------------------------------------------------------
#endif // SLINTERFACE_H
