//#############################################################################
//  File:      SL/SLInterface.cpp
//  Purpose:   Implementation of the main Scene Library C-Interface. Only these 
//             functions should called by the OS-dependend GUI applications. 
//             These functions can be called from any C, C++ or ObjectiveC GUI 
//             framework or by a native API such as Java Native Interface (JNI)
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLInterface.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLAssImp.h>


//-----------------------------------------------------------------------------
 //! global flag that determines if the application should be closed
bool gShouldClose = false; 

//-----------------------------------------------------------------------------
/*! Global creation function for a SLScene instance. This function should be
called only once per application. The SLScene constructor call is delayed until
the first SLSceneView is created to guarantee, that the OpenGL context is
present.
*/
void slCreateScene(SLstring shaderPath,
                   SLstring modelPath,
                   SLstring texturePath)
{
    assert(SLScene::current==0 && "SLScene is already created!");
   
    SLGLShaderProg::defaultPath = shaderPath;
    SLGLTexture::defaultPath    = texturePath;
    SLAssImp::defaultPath       = modelPath;
    SLGLState* stateGL          = SLGLState::getInstance();
    
    SL_LOG("Path to Models  : %s\n", modelPath.c_str());
    SL_LOG("Path to Shaders : %s\n", shaderPath.c_str());
    SL_LOG("Path to Textures: %s\n", texturePath.c_str());
    SL_LOG("OpenGL Version  : %s\n", stateGL->glVersion().c_str());
    SL_LOG("Vendor          : %s\n", stateGL->glVendor().c_str());
    SL_LOG("Renderer        : %s\n", stateGL->glRenderer().c_str());
    SL_LOG("GLSL Version    : %s\n", stateGL->glGLSLVersion().c_str());

    SLScene::current = new SLScene("");
}
//-----------------------------------------------------------------------------
/*! Global creation function for a SLSceneview instance returning the index of 
the sceneview. It creates the new SLSceneView instance by calling the callback
function slNewSceneView. If you have a custom SLSceneView inherited class you 
have to provide a similar function and pass it function pointer to 
slCreateSceneView. You can create multiple sceneview per application.
*/
int slCreateSceneView(int screenWidth,
                      int screenHeight,
                      int dotsPerInch,
                      SLCmd initScene,  
                      SLVstring& cmdLineArgs,
                      void* onWndUpdateCallback,
                      void* onSelectNodeMeshCallback,
                      void* onNewSceneViewCallback,
                      void* onShowSystemCursorCallback)
{
    assert(SLScene::current && "No SLScene::current!");

    // Use our own sceneview creator callback or the the passed one.
    cbOnNewSceneView newSVCallback;
    if (onNewSceneViewCallback==0)
         newSVCallback = &slNewSceneView;
    else newSVCallback = (cbOnNewSceneView)onNewSceneViewCallback;

    // Create the sceneview & get the pointer with the sceneview index
    SLuint index = newSVCallback();
    SLSceneView* sv = SLScene::current->sv(index);

    sv->init("SceneView", 
             screenWidth, 
             screenHeight, 
             dotsPerInch, 
             cmdLineArgs,
             onWndUpdateCallback,
             onSelectNodeMeshCallback,
             onShowSystemCursorCallback);

    // Set active sceneview and load scene. This is done for the first sceneview
    if (!SLScene::current->root3D())
    {   SLScene::current->onLoad(sv, initScene);
    } else
        sv->onInitialize();
   
    // return the identifier index
    return sv->index();
}
//-----------------------------------------------------------------------------
/*! Global sceneview construction function returning the index of the created
sceneview instance. If you have a custom SLSceneView inherited class you 
have to provide a similar function and pass it function pointer to 
slCreateSceneView.
*/
int slNewSceneView()
{
    SLSceneView* sv = new SLSceneView();
    return sv->index();
}
//-----------------------------------------------------------------------------
/*! Global closing function that deallocates the sceneview and scene instances.
All the scenegraph deallocation is started from here and has to be done before
the GUI app terminates.
*/
bool slShouldClose()
{
    return gShouldClose;
}
//-----------------------------------------------------------------------------
/*! Global closing function that sets our global running flag. This lets
the windowing system know that we want to terminate.
*/
void slShouldClose(bool val)
{
    gShouldClose = val;
}
//-----------------------------------------------------------------------------
/*! Global closing function that deallocates the sceneview and scene instances.
All the scenegraph deallocation is started from here and has to be done before
the GUI app terminates.
*/
void slTerminate()
{    
    // Deletes all remaining sceneviews the current scene instance  
    delete SLScene::current;
    SLScene::current = 0;
}
//-----------------------------------------------------------------------------
/*! Global rendering function that simply calls the sceneview's onPaint method.
This function must be called for each frame. After the frame is generated the
OS must swap the OpenGL's backbuffer to the visible front buffer.
*/
bool slPaint(int sceneViewIndex)
{  
    SLSceneView* sv = SLScene::current->sv(sceneViewIndex);
    return sv->onPaint();
}
//-----------------------------------------------------------------------------
/*! Global resize function that must be called whenever the OpenGL frame
changes it's size.
*/
void slResize(int sceneViewIndex, int width, int height)
{
    SLSceneView* sv = SLScene::current->sv(sceneViewIndex);
    sv->onResize(width, height);
}
//-----------------------------------------------------------------------------
/*! Global event handler for mouse button down events. 
*/
bool slMouseDown(int sceneViewIndex, SLMouseButton button, 
                 int xpos, int ypos, SLKey modifier) 
{  
    SLSceneView* sv = SLScene::current->sv(sceneViewIndex);
    return sv->onMouseDown(button, xpos, ypos, modifier);
}
//-----------------------------------------------------------------------------
/*! Global event handler for mouse move events.
*/
bool slMouseMove(int sceneViewIndex, int x, int y)
{  
    SLSceneView* sv = SLScene::current->sv(sceneViewIndex);
    return sv->onMouseMove(x, y);
}
//-----------------------------------------------------------------------------
/*! Global event handler for mouse button up events.
*/
bool slMouseUp(int sceneViewIndex, SLMouseButton button, 
               int xpos, int ypos, SLKey modifier) 
{  
    SLSceneView* sv = SLScene::current->sv(sceneViewIndex);
    return sv->onMouseUp(button, xpos, ypos, modifier);
}
//-----------------------------------------------------------------------------
/*! Global event handler for double click events.
*/
bool slDoubleClick(int sceneViewIndex, SLMouseButton button, 
                   int xpos, int ypos, SLKey modifier) 
{  
    SLSceneView* sv = SLScene::current->sv(sceneViewIndex);
    return sv->onDoubleClick(ButtonLeft, xpos, ypos, modifier);
}
//-----------------------------------------------------------------------------
/*! Global event handler for the two finger touch down events of touchscreen 
devices.
*/
bool slTouch2Down(int sceneViewIndex, int xpos1, int ypos1, int xpos2, int ypos2) 
{  
    SLSceneView* sv = SLScene::current->sv(sceneViewIndex);
    return sv->onTouch2Down(xpos1, ypos1, xpos2, ypos2);
}
//-----------------------------------------------------------------------------
/*! Global event handler for the two finger move events of touchscreen devices. 
*/
bool slTouch2Move(int sceneViewIndex, int xpos1, int ypos1, int xpos2, int ypos2) 
{  
    SLSceneView* sv = SLScene::current->sv(sceneViewIndex);
    return sv->onTouch2Move(xpos1, ypos1, xpos2, ypos2);
}
//-----------------------------------------------------------------------------
/*! Global event handler for the two finger touch up events of touchscreen 
devices. 
*/
bool slTouch2Up(int sceneViewIndex, int xpos1, int ypos1, int xpos2, int ypos2) 
{  
    SLSceneView* sv = SLScene::current->sv(sceneViewIndex);
    return sv->onTouch2Up(xpos1, ypos1, xpos2, ypos2);
}
//-----------------------------------------------------------------------------
/*! Global event handler for mouse wheel events. 
*/
bool slMouseWheel(int sceneViewIndex, int pos, SLKey modifier)
{  
    SLSceneView* sv = SLScene::current->sv(sceneViewIndex);
    return sv->onMouseWheel(pos, modifier);
}
//-----------------------------------------------------------------------------
/*! Global event handler for keyboard key press events. 
*/
bool slKeyPress(int sceneViewIndex, SLKey key, SLKey modifier) 
{  
    SLSceneView* sv = SLScene::current->sv(sceneViewIndex);
    return sv->onKeyPress(key, modifier);
}
//-----------------------------------------------------------------------------
/*! Global event handler for keyboard key release events. 
*/
bool slKeyRelease(int sceneViewIndex, SLKey key, SLKey modifier) 
{  
    SLSceneView* sv = SLScene::current->sv(sceneViewIndex);
    return sv->onKeyRelease(key, modifier);
}
//-----------------------------------------------------------------------------
/*! Global event handler for keyboard key release events. 
*/
bool slCommand(int sceneViewIndex, SLCmd command) 
{  
    SLSceneView* sv = SLScene::current->sv(sceneViewIndex);
    return sv->onCommand(command);
}
//-----------------------------------------------------------------------------
/*! Global event handler for device rotation change with Euler angles pitch
yaw and roll. With the parameter zeroYawAfterSec sets the time in seconds after
which the yaw angle is set to zero by subtracting the average yaw in this time.
*/
void slRotationPYR(int sceneViewIndex, 
                   float pitchRAD, float yawRAD, float rollRAD)
{
    SLSceneView* sv = SLScene::current->sv(sceneViewIndex);
    sv->onRotationPYR(pitchRAD, yawRAD, rollRAD, 3.0f);
}
//-----------------------------------------------------------------------------
/*! Global event handler for device rotation change with angle & and axis. 
*/
void slRotationQUAT(int sceneViewIndex, 
                    float quatX, float quatY, float quatZ, float quatW)
{
    SLSceneView* sv = SLScene::current->sv(sceneViewIndex);
    sv->onRotationQUAT(quatX, quatY, quatZ, quatW);
}
//-----------------------------------------------------------------------------
/*! Global function to retrieve a window title text generated by the scene
library. 
*/
string slGetWindowTitle(int sceneViewIndex) 
{  
    SLSceneView* sv = SLScene::current->sv(sceneViewIndex);
    return sv->windowTitle();
}
//-----------------------------------------------------------------------------
