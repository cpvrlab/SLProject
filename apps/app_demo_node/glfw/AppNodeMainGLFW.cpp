//#############################################################################
//  File:      AppNodeMainGLFW.cpp
//  Purpose:   Implementation of the GUI with the GLFW3 (http://www.glfw.org/)
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLGLState.h>
#include <SLEnums.h>
#include <SLInterface.h>
#include <SLSceneView.h>
#include <opencv2/opencv.hpp>
#include <AppDemo.h>
#include "AppNodeGui.h"
#include "AppNodeSceneView.h"
#include <GLFW/glfw3.h>

#include <SLGLFWInterface.h>

extern void appNodeLoadScene(SLProjectScene* s, SLSceneView* sv, SLSceneID sid);

//-----------------------------------------------------------------------------
// GLobal application variables
GLFWwindow* window;                     //!< The global GLFW window handle.
SLint       svIndex;                    //!< SceneView index
SLint       scrWidth;                   //!< Window width at start up
SLint       scrHeight;                  //!< Window height at start up
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
SLint       dpi               = 142;    //!< Dot per inch resolution of screen

//-------------------------------------------------------------------------
/*! 
onClose event handler for deallocation of the scene & sceneview. onClose is
called glfwPollEvents, glfwWaitEvents or glfwSwapBuffers.
*/
void onClose(GLFWwindow* myWindow)
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
    if (AppDemo::sceneViews.empty())
        return false;
    SLSceneView* sv = AppDemo::sceneViews[svIndex];

    ///////////////////////////////////////////////
    bool jobIsRunning      = slUpdateParallelJob();
    bool viewsNeedsRepaint = slPaintAllViews();
    ///////////////////////////////////////////////

    return jobIsRunning || viewsNeedsRepaint;
}
//-----------------------------------------------------------------------------
/*!
onResize: Event handler called on the resize event of the window. This event
should called once before the onPaint event.
*/
static void onResize(GLFWwindow* myWindow,
                     int         width,
                     int         height)
{
    if (AppDemo::sceneViews.empty()) return;
    SLSceneView* sv = AppDemo::sceneViews[svIndex];

    lastWidth  = width;
    lastHeight = height;

    // width & height are in screen coords.
    // We need to scale them to framebuffer coords.
    slResize(svIndex, width, height);
}
//-----------------------------------------------------------------------------
/*!
Mouse button event handler forwards the events to the slMouseDown or slMouseUp.
Two finger touches of touch devices are simulated with ALT & CTRL modifiers.
*/
static void onMouseButton(GLFWwindow* myWindow,
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
static void onMouseMove(GLFWwindow* myWindow,
                        double      x,
                        double      y)
{
    // x & y are in screen coords.
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
static void onMouseWheel(GLFWwindow* myWindow,
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
static void onKeyAction(GLFWwindow* myWindow,
                        int         GLFWKey,
                        int         scancode,
                        int         action,
                        int         mods)
{
    SLKey key = SLGLFWInterface::mapKeyToSLKey(GLFWKey);

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
            glfwSetWindowSize(myWindow, scrWidth, scrHeight);
            glfwSetWindowPos(myWindow, 10, 30);
        }
        else
        {
            slKeyPress(svIndex, key, modifiers); // ESC during RT stops it and returns false
            onClose(myWindow);
            glfwSetWindowShouldClose(myWindow, GL_TRUE);
        }
    }
    else if (key == K_F9 && action == GLFW_PRESS) // Toggle fullscreen mode
    {
        fullscreen = !fullscreen;

        if (fullscreen)
        {
            GLFWmonitor*       primary = glfwGetPrimaryMonitor();
            const GLFWvidmode* mode    = glfwGetVideoMode(primary);
            glfwSetWindowSize(myWindow, mode->width, mode->height);
            glfwSetWindowPos(myWindow, 0, 0);
        }
        else
        {
            glfwSetWindowSize(myWindow, scrWidth, scrHeight);
            glfwSetWindowPos(myWindow, 10, 30);
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
//! Alternative SceneView creation C-function passed by slCreateSceneView
SLSceneView* createAppNodeSceneView(SLProjectScene* scene,
                                    int             myDPI,
                                    SLInputManager& inputManager)
{
    return new AppNodeSceneView(scene, myDPI, inputManager);
}
//-----------------------------------------------------------------------------
//! Initialises all GLFW and GL3W stuff
void initGLFW(int screenWidth, int screenHeight)
{
    SLGLFWInterface::initialize();

    // Enable fullscreen anti aliasing with 4 samples
    glfwWindowHint(GLFW_SAMPLES, 4);

#ifdef __APPLE__
    //You can enable or restrict newer OpenGL context here (read the GLFW documentation)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GL_FALSE);
#endif

    window = glfwCreateWindow(screenWidth, screenHeight, "My Title", nullptr, nullptr);

    //get real window size
    glfwGetWindowSize(window, &scrWidth, &scrHeight);

    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // Get the current GL context. After this you can call GL
    glfwMakeContextCurrent(window);

    // Init OpenGL access library gl3w
    if (gl3wInit() != 0)
    {
        cerr << "Failed to initialize OpenGL" << endl;
        exit(-1);
    }

    glfwSetWindowTitle(window, "SLProject Test Application");
    glfwSetWindowPos(window, 10, 30);

    // With GLFW ImGui draws the cursor
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

    // Set number of monitor refreshes between 2 buffer swaps
    glfwSwapInterval(2);

    // Get GL errors that occurred before our framework is involved
    GET_GL_ERROR;

    // Set your own physical screen dpi
    Utils::log("SLProject", "------------------------------------------------------------------");
    Utils::log("SLProject",
               "GUI-Framwork     : GLFW (Version: %d.%d.%d",
               GLFW_VERSION_MAJOR,
               GLFW_VERSION_MINOR,
               GLFW_VERSION_REVISION);
    Utils::log("SLProject",
               "Resolution (DPI) : %d",
               dpi);

    // Set GLFW callback functions
    glfwSetKeyCallback(window, onKeyAction);
    glfwSetWindowSizeCallback(window, onResize);
    glfwSetMouseButtonCallback(window, onMouseButton);
    glfwSetCursorPosCallback(window, onMouseMove);
    glfwSetScrollCallback(window, onMouseWheel);
    glfwSetWindowCloseCallback(window, onClose);
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

    initGLFW(640, 480);

    // get executable path
    SLstring projectRoot = SLstring(SL_PROJECT_ROOT);
    SLstring configPath  = Utils::getAppsWritableDir();

    //////////////////////////////////////////////////////////
    slCreateAppAndScene(cmdLineArgs,
                        projectRoot + "/data/",
                        projectRoot + "/data/shaders/",
                        projectRoot + "/data/models/",
                        projectRoot + "/data/images/textures/",
                        projectRoot + "/data/images/fonts/",
                        projectRoot + "/data/videos/",
                        configPath,
                        "AppNode_GLFW",
                        (void*)appNodeLoadScene);
    //////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////
    svIndex = slCreateSceneView(AppDemo::scene,
                                scrWidth,
                                scrHeight,
                                dpi,
                                (SLSceneID)0,
                                (void*)&onPaint,
                                nullptr,
                                (void*)createAppNodeSceneView,
                                (void*)AppNodeGui::build);
    //////////////////////////////////////////////////////////

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
