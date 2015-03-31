//#############################################################################
//  File:      SL/SLInterface.h
//  Purpose:   Delcaration of the main Scene Library C-Interface.
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLINTERFACE_H
#define SLINTERFACE_H

#include <stdafx.h>
#include <SLEnums.h>

//! \file SLInterface.h SLProject C-functions interface declaration.
/*! \file SLInterface.h
The SLInterface.h has all delcarations of the SLProject C-Interface.
Only these functions should called by the OS-dependend GUI applications.
These functions can be called from any C, C++ or ObjectiveC GUI framework or
by a native API such as Java Native Interface (JNI).
See the implementation for more information.
*/
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
bool    slUpdateAndPaint    (int sceneViewIndex);
void    slMouseDown         (int sceneViewIndex, SLMouseButton button, int x, int y, SLKey modifier);
void    slMouseMove         (int sceneViewIndex, int x, int y);
void    slMouseUp           (int sceneViewIndex, SLMouseButton button, int x, int y, SLKey modifier);
void    slDoubleClick       (int sceneViewIndex, SLMouseButton button, int x, int y, SLKey modifier);
void    slLongTouch         (int sceneViewIndex, int x, int y);
void    slTouch2Down        (int sceneViewIndex, int x1, int y1, int x2, int y2);
void    slTouch2Move        (int sceneViewIndex, int x1, int y1, int x2, int y2);
void    slTouch2Up          (int sceneViewIndex, int x1, int y1, int x2, int y2);
void    slMouseWheel        (int sceneViewIndex, int pos, SLKey modifier);
void    slKeyPress          (int sceneViewIndex, SLKey key, SLKey modifier);
void    slKeyRelease        (int sceneViewIndex, SLKey key, SLKey modifier);
void    slCharInput         (int sceneViewIndex, unsigned int character);
void    slCommand           (int sceneViewIndex, SLCmd command);
void    slRotationPYR       (int sceneViewIndex, float pitchRAD, float yawRAD, float rollRAD);
void    slRotationQUAT      (int sceneViewIndex, float angleRAD, float axisX, float axisY, float axisZ);
string  slGetWindowTitle    (int sceneViewIndex);
//-----------------------------------------------------------------------------
#endif // SLINTERFACE_H
