//#############################################################################
//  File:      AppNodeMainGLFW.cpp
//  Purpose:   Implementation of the GUI with the GLFW3 (http://www.glfw.org/)
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#include <GLFW/glfw3.h>
#include <SLEnums.h>
#include <SLInterface.h>
#include <SLSceneView.h>
#include <opencv2/opencv.hpp>
#include <SLApplication.h>

#include "AppNodeGui.h"
#include "AppNodeSceneView.h"

extern void appNodeLoadScene(SLProjectScene* s, SLSceneView* sv, SLSceneID sid);

//-----------------------------------------------------------------------------
// GLobal application variables
GLFWwindow* window;                     //!< The global GLFW window handle.
SLint       svIndex;                    //!< SceneView index
SLint       scrWidth;                   //!< Window width at start up
SLint       scrHeight;                  //!< Window height at start up
SLfloat     scr2fbX;                    //!< Factor from screen to framebuffer coords
SLfloat     scr2fbY;                    //!< Factor from screen to framebuffer coords
SLint       mouseX;                     //!< Last mouse position x in pixels
SLint       mouseY;                     //!< Last mouse position y in pixels
SLint       touchX2;                    //!< Last finger touch 2 position x in pixels
SLint       touchY2;                    //!< Last finger touch 2 position y in pixels
SLint       touchDeltaX;                //!< Delta between two fingers in x
SLint       touchDeltaY;                //!< Delta between two fingers in <
SLint       lastWidth;                  //!< Last window width in pixels
SLint       lastHeight;                 //!< Last window height in pixels
SLint       lastMouseWheelPos;          //!< Last mouse wheel position
SLfloat     lastMouseDownTime = 0.0f;   //!< Last mouse press time
SLKey       modifiers         = K_none; //!< last modifier keys
SLbool      fullscreen        = false;  //!< flag if window is in fullscreen mode

//-----------------------------------------------------------------------------
/*! 
onClose event handler for deallocation of the scene & sceneview. onClose is
called glfwPollEvents, glfwWaitEvents or glfwSwapBuffers.
*/
void onClose(GLFWwindow* window)
{
    slShouldClose(true);
}

//-----------------------------------------------------------------------------
/*!
onPaint: Paint event handler that passes the event to the slPaint function. 
For accurate frame rate measurement we have to take the time after the OpenGL 
frame buffer swapping. The FPS calculation is done in slGetWindowTitle.
*/
SLbool onPaint()
{
    /////////////////////////////////////////////
    bool jobIsRunning      = slUpdateParallelJob();
    bool viewsNeedsRepaint = slPaintAllViews();
    /////////////////////////////////////////////

    return jobIsRunning || viewsNeedsRepaint;
}
//-----------------------------------------------------------------------------
//! Maps the GLFW key codes to the SLKey codes
SLKey mapKeyToSLKey(SLint key)
{
    switch (key)
    {
        case GLFW_KEY_SPACE: return K_space;
        case GLFW_KEY_ESCAPE: return K_esc;
        case GLFW_KEY_F1: return K_F1;
        case GLFW_KEY_F2: return K_F2;
        case GLFW_KEY_F3: return K_F3;
        case GLFW_KEY_F4: return K_F4;
        case GLFW_KEY_F5: return K_F5;
        case GLFW_KEY_F6: return K_F6;
        case GLFW_KEY_F7: return K_F7;
        case GLFW_KEY_F8: return K_F8;
        case GLFW_KEY_F9: return K_F9;
        case GLFW_KEY_F10: return K_F10;
        case GLFW_KEY_F11: return K_F11;
        case GLFW_KEY_F12: return K_F12;
        case GLFW_KEY_UP: return K_up;
        case GLFW_KEY_DOWN: return K_down;
        case GLFW_KEY_LEFT: return K_left;
        case GLFW_KEY_RIGHT: return K_right;
        case GLFW_KEY_LEFT_SHIFT: return K_shift;
        case GLFW_KEY_RIGHT_SHIFT: return K_shift;
        case GLFW_KEY_LEFT_CONTROL: return K_ctrl;
        case GLFW_KEY_RIGHT_CONTROL: return K_ctrl;
        case GLFW_KEY_LEFT_ALT: return K_alt;
        case GLFW_KEY_RIGHT_ALT: return K_alt;
        case GLFW_KEY_LEFT_SUPER: return K_super;  // Apple command key
        case GLFW_KEY_RIGHT_SUPER: return K_super; // Apple command key
        case GLFW_KEY_TAB: return K_tab;
        case GLFW_KEY_ENTER: return K_enter;
        case GLFW_KEY_BACKSPACE: return K_backspace;
        case GLFW_KEY_INSERT: return K_insert;
        case GLFW_KEY_DELETE: return K_delete;
        case GLFW_KEY_PAGE_UP: return K_pageUp;
        case GLFW_KEY_PAGE_DOWN: return K_pageDown;
        case GLFW_KEY_HOME: return K_home;
        case GLFW_KEY_END: return K_end;
        case GLFW_KEY_KP_0: return K_NP0;
        case GLFW_KEY_KP_1: return K_NP1;
        case GLFW_KEY_KP_2: return K_NP2;
        case GLFW_KEY_KP_3: return K_NP3;
        case GLFW_KEY_KP_4: return K_NP4;
        case GLFW_KEY_KP_5: return K_NP5;
        case GLFW_KEY_KP_6: return K_NP6;
        case GLFW_KEY_KP_7: return K_NP7;
        case GLFW_KEY_KP_8: return K_NP8;
        case GLFW_KEY_KP_9: return K_NP9;
        case GLFW_KEY_KP_DIVIDE: return K_NPDivide;
        case GLFW_KEY_KP_MULTIPLY: return K_NPMultiply;
        case GLFW_KEY_KP_SUBTRACT: return K_NPSubtract;
        case GLFW_KEY_KP_ADD: return K_NPAdd;
        case GLFW_KEY_KP_DECIMAL: return K_NPDecimal;
        default: break;
    }
    return (SLKey)key;
}
//-----------------------------------------------------------------------------
/*!
onResize: Event handler called on the resize event of the window. This event
should called once before the onPaint event.
*/
static void onResize(GLFWwindow* window,
                     int         width,
                     int         height)
{
    lastWidth  = width;
    lastHeight = height;

    // width & height are in screen coords.
    // We need to scale them to framebuffer coords.
    slResize(svIndex, (int)((float)width * scr2fbX), (int)((float)height * scr2fbY));
}
//-----------------------------------------------------------------------------
/*!
Mouse button event handler forwards the events to the slMouseDown or slMouseUp.
Two finger touches of touch devices are simulated with ALT & CTRL modifiers.
*/
static void onMouseButton(GLFWwindow* window,
                          int         button,
                          int         action,
                          int         mods)
{
    SLint x = mouseX;
    SLint y = mouseY;

    // Translate modifiers
    modifiers = K_none;
    if (mods & GLFW_MOD_SHIFT) modifiers = (SLKey)(modifiers | K_shift);
    if (mods & GLFW_MOD_CONTROL) modifiers = (SLKey)(modifiers | K_ctrl);
    if (mods & GLFW_MOD_ALT) modifiers = (SLKey)(modifiers | K_alt);

    if (action == GLFW_PRESS)
    {
        // simulate double touch from touch devices
        if (modifiers & K_alt)
        {
            // Do parallel double finger move
            if (modifiers & K_shift)
                slTouch2Down(svIndex, x, y, x - touchDeltaX, y - touchDeltaY);
            else // Do concentric double finger pinch
                slTouch2Down(svIndex, x, y, touchX2, touchY2);
        }
        else // Do standard mouse down
        {
            SLfloat mouseDeltaTime = (SLfloat)glfwGetTime() - lastMouseDownTime;
            lastMouseDownTime      = (SLfloat)glfwGetTime();

            // handle double click
            if (mouseDeltaTime < 0.3f)
            {
                switch (button)
                {
                    case GLFW_MOUSE_BUTTON_LEFT:
                        slDoubleClick(svIndex, MB_left, x, y, modifiers);
                        break;
                    case GLFW_MOUSE_BUTTON_RIGHT:
                        slDoubleClick(svIndex, MB_right, x, y, modifiers);
                        break;
                    case GLFW_MOUSE_BUTTON_MIDDLE:
                        slDoubleClick(svIndex, MB_middle, x, y, modifiers);
                        break;
                    default: break;
                }
            }
            else // normal mouse clicks
            {
                switch (button)
                {
                    case GLFW_MOUSE_BUTTON_LEFT:
                        slMouseDown(svIndex, MB_left, x, y, modifiers);
                        break;
                    case GLFW_MOUSE_BUTTON_RIGHT:
                        slMouseDown(svIndex, MB_right, x, y, modifiers);
                        break;
                    case GLFW_MOUSE_BUTTON_MIDDLE:
                        slMouseDown(svIndex, MB_middle, x, y, modifiers);
                        break;
                    default: break;
                }
            }
        }
    }
    else
    {
        // simulate double touch from touch devices
        if (modifiers & K_alt)
        {
            // Do parallel double finger move
            if (modifiers & K_shift)
            {
                slTouch2Up(svIndex, x, y, x - (touchX2 - x), y - (touchY2 - y));
            }
            else // Do concentric double finger pinch
            {
                slTouch2Up(svIndex, x, y, touchX2, touchY2);
            }
        }
        else // Do standard mouse down
        {
            switch (button)
            {
                case GLFW_MOUSE_BUTTON_LEFT:
                    slMouseUp(svIndex, MB_left, x, y, modifiers);
                    break;
                case GLFW_MOUSE_BUTTON_RIGHT:
                    slMouseUp(svIndex, MB_right, x, y, modifiers);
                    break;
                case GLFW_MOUSE_BUTTON_MIDDLE:
                    slMouseUp(svIndex, MB_middle, x, y, modifiers);
                    break;
                default: break;
            }
        }
    }
}
//-----------------------------------------------------------------------------
/*!
Mouse move event handler forwards the events to slMouseMove or slTouch2Move.
*/
static void onMouseMove(GLFWwindow* window,
                        double      x,
                        double      y)
{
    // x & y are in screen coords.
    // We need to scale them to framebuffer coords
    x *= scr2fbX;
    y *= scr2fbY;
    mouseX = (int)x;
    mouseY = (int)y;

    // Offset of 2nd. finger for two finger simulation

    // Simulate double finger touches
    if (modifiers & K_alt)
    {
        // Do parallel double finger move
        if (modifiers & K_shift)
        {
            slTouch2Move(svIndex,
                         (int)x,
                         (int)y,
                         (int)x - touchDeltaX,
                         (int)y - touchDeltaY);
        }
        else // Do concentric double finger pinch
        {
            int scrW2   = lastWidth / 2;
            int scrH2   = lastHeight / 2;
            touchX2     = scrW2 - ((int)x - scrW2);
            touchY2     = scrH2 - ((int)y - scrH2);
            touchDeltaX = (int)x - touchX2;
            touchDeltaY = (int)y - touchY2;
            slTouch2Move(svIndex, (int)x, (int)y, touchX2, touchY2);
        }
    }
    else // Do normal mouse move
        slMouseMove(svIndex, (int)x, (int)y);
}
//-----------------------------------------------------------------------------
/*!
Mouse wheel event handler forwards the events to slMouseWheel
*/
static void onMouseWheel(GLFWwindow* window,
                         double      xscroll,
                         double      yscroll)
{
    // make sure the delta is at least one integer
    int dY = (int)yscroll;
    if (dY == 0) dY = (int)(Utils::sign(yscroll));

    slMouseWheel(svIndex, dY, modifiers);
}
//-----------------------------------------------------------------------------
/*!
Key action event handler sets the modifier key state & forwards the event to
the slKeyPress function.
*/
static void onKeyAction(GLFWwindow* window,
                        int         GLFWKey,
                        int         scancode,
                        int         action,
                        int         mods)
{
    SLKey key = mapKeyToSLKey(GLFWKey);

    if (action == GLFW_PRESS)
    {
        switch (key)
        {
            case K_ctrl: modifiers = (SLKey)(modifiers | K_ctrl); return;
            case K_alt: modifiers = (SLKey)(modifiers | K_alt); return;
            case K_shift: modifiers = (SLKey)(modifiers | K_shift); return;
        }
    }
    else if (action == GLFW_RELEASE)
    {
        switch (key)
        {
            case K_ctrl: modifiers = (SLKey)(modifiers ^ K_ctrl); return;
            case K_alt: modifiers = (SLKey)(modifiers ^ K_alt); return;
            case K_shift: modifiers = (SLKey)(modifiers ^ K_shift); return;
        }
    }

    // Special treatment for ESC key
    if (key == K_esc && action == GLFW_RELEASE)
    {
        if (fullscreen)
        {
            fullscreen = !fullscreen;
            glfwSetWindowSize(window, scrWidth, scrHeight);
            glfwSetWindowPos(window, 10, 30);
        }
        else
        {
            slKeyPress(svIndex, key, modifiers); // ESC during RT stops it and returns false
            onClose(window);
            glfwSetWindowShouldClose(window, GL_TRUE);
        }
    }
    else

      // Toggle fullscreen mode
      if (key == K_F9 && action == GLFW_PRESS)
    {
        fullscreen = !fullscreen;

        if (fullscreen)
        {
            GLFWmonitor*       primary = glfwGetPrimaryMonitor();
            const GLFWvidmode* mode    = glfwGetVideoMode(primary);
            glfwSetWindowSize(window, mode->width, mode->height);
            glfwSetWindowPos(window, 0, 0);
        }
        else
        {
            glfwSetWindowSize(window, scrWidth, scrHeight);
            glfwSetWindowPos(window, 10, 30);
        }
    }
    else
    {
        if (action == GLFW_PRESS)
            slKeyPress(svIndex, key, modifiers);
        else if (action == GLFW_RELEASE)
            slKeyRelease(svIndex, key, modifiers);
    }
}
//-----------------------------------------------------------------------------
//! Error callback handler for GLFW.
void onGLFWError(int error, const char* description)
{
    fputs(description, stderr);
}
//-----------------------------------------------------------------------------
//! Alternative SceneView creation C-function passed by slCreateSceneView
SLSceneView* createAppNodeSceneView(SLProjectScene* scene, int dpi, SLInputManager& inputManager)
{
    return new AppNodeSceneView(scene, dpi, inputManager);
}
//-----------------------------------------------------------------------------
/*!
The C main procedure running the GLFW GUI application.
*/
int main(int argc, char* argv[])
{
    // set command line arguments
    SLVstring cmdLineArgs;
    for (int i = 0; i < argc; i++)
        cmdLineArgs.push_back(argv[i]);

    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
        exit(EXIT_FAILURE);
    }

    glfwSetErrorCallback(onGLFWError);

    // Enable fullscreen anti aliasing with 4 samples
    glfwWindowHint(GLFW_SAMPLES, 4);

    //You can enable or restrict newer OpenGL context here (read the GLFW documentation)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    scrWidth  = 640;
    scrHeight = 480;

    window = glfwCreateWindow(scrWidth, scrHeight, "My Title", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // Get the current GL context. After this you can call GL
    glfwMakeContextCurrent(window);

    // On some systems screen & framebuffer size are different
    // All commands in GLFW are in screen coords but rendering in GL is
    // in framebuffer coords
    SLint fbWidth, fbHeight;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    scr2fbX = (float)fbWidth / (float)scrWidth;
    scr2fbY = (float)fbHeight / (float)scrHeight;

    /* Include OpenGL via GLEW
    The goal of the OpenGL Extension Wrangler Library (GLEW) is to assist C/C++ 
    OpenGL developers with two tedious tasks: initializing and using extensions 
    and writing portable applications. GLEW provides an efficient run-time 
    mechanism to determine whether a certain extension is supported by the 
    driver or not. OpenGL core and extension functionality is exposed via a 
    single header file. Download GLEW at: http://glew.sourceforge.net/
    */
    glewExperimental = GL_TRUE;
    GLenum err       = glewInit();
    if (GLEW_OK != err)
    {
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    glfwSetWindowTitle(window, "SLProject Test Application");
    glfwSetWindowPos(window, 10, 30);

    // Set number of monitor refreshes between 2 buffer swaps
    glfwSwapInterval(1);

    // Set your own physical screen dpi
    int dpi = (int)(142 * scr2fbX);
    cout << "------------------------------------------------------------------" << endl;
    cout << "GUI             : GLFW (Version: " << GLFW_VERSION_MAJOR << "." << GLFW_VERSION_MINOR << ")" << endl;
    cout << "DPI             : " << dpi << endl;

    // get executable path
    SLstring projectRoot = SLstring(SL_PROJECT_ROOT);
    SLstring configPath  = Utils::getAppsWritableDir();

    //////////////////////////////////////////////////////////
    slCreateAppAndScene(cmdLineArgs,
                        projectRoot + "/data/shaders/",
                        projectRoot + "/data/models/",
                        projectRoot + "/data/images/textures/",
                        projectRoot + "/data/images/fonts/",
                        configPath,
                        "AppNode_GLFW",
                        (void*)appNodeLoadScene);
    //////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////
    svIndex = slCreateSceneView(SLApplication::scene,
                                (int)(scrWidth * scr2fbX),
                                (int)(scrHeight * scr2fbY),
                                dpi,
                                (SLSceneID)0,
                                (void*)&onPaint,
                                nullptr,
                                (void*)createAppNodeSceneView,
                                (void*)AppNodeGui::build);
    //////////////////////////////////////////////////////////

    // Set GLFW callback functions
    glfwSetKeyCallback(window, onKeyAction);
    glfwSetWindowSizeCallback(window, onResize);
    glfwSetMouseButtonCallback(window, onMouseButton);
    glfwSetCursorPosCallback(window, onMouseMove);
    glfwSetScrollCallback(window, onMouseWheel);
    glfwSetWindowCloseCallback(window, onClose);

    // Event loop
    while (!slShouldClose())
    {
        /////////////////////////////
        SLbool doRepaint = onPaint();
        /////////////////////////////

        // Fast copy the back buffer to the front buffer. This is OS dependent.
        glfwSwapBuffers(window);

        // Show the title generated by the scene library (FPS etc.)
        glfwSetWindowTitle(window, slGetWindowTitle(svIndex).c_str());

        // if no updated occurred wait for the next event (power saving)
        if (!doRepaint)
            glfwWaitEvents();
        else
            glfwPollEvents();
    }

    slTerminate();
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(0);
}
//-----------------------------------------------------------------------------
