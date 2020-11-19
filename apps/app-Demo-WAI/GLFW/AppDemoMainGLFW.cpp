//#############################################################################
//  File:      AppDemoMainGLFW.cpp
//  Purpose:   The demo application demonstrates most features of libWAI
//  Author:    Jan Dellsperger, Luc Girod
//  Date:      November 2018
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#include <SLGLState.h>
#include <ErlebARApp.h>
#include <Utils.h>
#include <HighResTimer.h>
#include <GLFW/SENSWebCamera.h>
#include <sens/SENSGps.h>
#include <DeviceData.h>
#include <GLFW/glfw3.h>
#include <sens/SENSSimulator.h>

static ErlebARApp app;

//-----------------------------------------------------------------------------
// GLobal application variables
static GLFWwindow* window;                                         //!< The global glfw window handle
static SLint       svIndex;                                        //!< SceneView index
static SLint       scrWidth  = 1920;                               //!< Window width at start up
static SLint       scrHeight = 1080;                               //!< Window height at start up
static SLfloat     scrWdivH  = (float)scrWidth / (float)scrHeight; //!< aspect ratio screen width divided by height
static SLint       startX;                                         //!< start position x in pixels
static SLint       startY;                                         //!< start position y in pixels
static SLint       mouseX;                                         //!< Last mouse position x in pixels
static SLint       mouseY;                                         //!< Last mouse position y in pixels
static SLVec2i     touch2;                                         //!< Last finger touch 2 position in pixels
static SLVec2i     touchDelta;                                     //!< Delta between two fingers in x
static SLint       lastWidth;                                      //!< Last window width in pixels
static SLint       lastHeight;                                     //!< Last window height in pixels
static SLfloat     lastMouseDownTime = 0.0f;                       //!< Last mouse press time
static SLKey       modifiers         = K_none;                     //!< last modifier keys
static SLbool      fullscreen        = false;                      //!< flag if window is in fullscreen mode
static int         dpi;
static bool        appShouldClose = false;
static int         longTouchMS    = 500;

//-----------------------------------------------------------------------------
int getDpi()
{
    const float  inchPerMM = 0.0393701;
    float        scaleX = 0, scaleY = 0;
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    glfwGetMonitorContentScale(monitor, &scaleX, &scaleY);
    int widthMM, heightMM;
    glfwGetMonitorPhysicalSize(monitor, &widthMM, &heightMM);
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    float widthDpi = (float)mode->width / (float)widthMM / inchPerMM * scaleY;

    return (int)widthDpi;
}
//-----------------------------------------------------------------------------
/*!
onClose event handler for deallocation of the scene & sceneview. onClose is
called glfwPollEvents, glfwWaitEvents or glfwSwapBuffers.
*/
void onClose(GLFWwindow* myWindow)
{
    app.destroy();
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
        case GLFW_KEY_UNKNOWN: return K_none;
    }
    return (SLKey)key;
}
//-----------------------------------------------------------------------------
/*!
onResize: Event handler called on the resize event of the window. This event
should called once before the onPaint event.
*/
void onResize(GLFWwindow* myWindow, int width, int height)
{
    //on windows minimizing gives callback with (0,0)
    if (width == 0 && height == 0)
    {
        return;
    }

    lastWidth  = width;
    lastHeight = height;

    // width & height are in screen coords.
    // We need to scale them to framebuffer coords.
    app.resize(svIndex, width, height);
}
//-----------------------------------------------------------------------------
/*!
onLongTouch gets called from a 500ms timer after a mouse down event.
*/
void onLongTouch()
{
    // forward the long touch only if the mouse or touch hasn't moved.
    if (Utils::abs(mouseX - startX) < 2 && Utils::abs(mouseY - startY) < 2)
    {
        app.longTouch(svIndex, mouseX, mouseY);
    }
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
    startX  = x;
    startY  = y;

    // Translate modifiers
    modifiers = K_none;
    if (mods & GLFW_MOD_SHIFT) modifiers = (SLKey)(modifiers | K_shift);
    if (mods & GLFW_MOD_CONTROL) modifiers = (SLKey)(modifiers | K_ctrl);
    if (mods & GLFW_MOD_ALT) modifiers = (SLKey)(modifiers | K_alt);

    if (action == GLFW_PRESS)
    {
        // simulate double touch from touch devices
        /*
        if (modifiers & K_alt)
        {
            // init for first touch
            if (touch2.x < 0)
            {
                int scrW2 = lastWidth / 2;
                int scrH2 = lastHeight / 2;
                touch2.set(scrW2 - (x - scrW2), scrH2 - (y - scrH2));
                touchDelta.set(x - touch2.x, y - touch2.y);
            }

            // Do parallel double finger move
            if (modifiers & K_shift)
            {
                app.touch2Down(svIndex, x, y, x - touchDelta.x, y - touchDelta.y);
            }
            else // Do concentric double finger pinch
            {
                app.touch2Down(svIndex, x, y, touch2.x, touch2.y);
            }
        }
        */
        {
            SLfloat mouseDeltaTime = (SLfloat)glfwGetTime() - lastMouseDownTime;
            lastMouseDownTime      = (SLfloat)glfwGetTime();

            // handle double click
            if (mouseDeltaTime < 0.3f)
            {
                switch (button)
                {
                    case GLFW_MOUSE_BUTTON_LEFT:
                        app.doubleClick(svIndex, MB_left, x, y, modifiers);
                        break;
                    case GLFW_MOUSE_BUTTON_RIGHT:
                        app.doubleClick(svIndex, MB_right, x, y, modifiers);
                        break;
                    case GLFW_MOUSE_BUTTON_MIDDLE:
                        app.doubleClick(svIndex, MB_middle, x, y, modifiers);
                        break;
                }
            }
            else // normal mouse clicks
            {
                // Start timer for the long touch detection
                HighResTimer::callAfterSleep(longTouchMS, onLongTouch);

                switch (button)
                {
                    case GLFW_MOUSE_BUTTON_LEFT:
                        app.mouseDown(svIndex, MB_left, x, y, modifiers);
                        break;
                    case GLFW_MOUSE_BUTTON_RIGHT:
                        app.mouseDown(svIndex, MB_right, x, y, modifiers);
                        break;
                    case GLFW_MOUSE_BUTTON_MIDDLE:
                        app.mouseDown(svIndex, MB_middle, x, y, modifiers);
                        break;
                }
            }
        }
    }
    else
    { // flag end of mouse click for long touches
        startX = -1;
        startY = -1;

        // simulate double touch from touch devices
        /*
        if (modifiers & K_alt)
        {
            // Do parallel double finger move
            if (modifiers & K_shift)
            {
                app.touch2Up(svIndex, x, y, x - (touch2.x - x), y - (touch2.y - y));
            }
            else // Do concentric double finger pinch
            {
                app.touch2Up(svIndex, x, y, touch2.x, touch2.y);
            }
        }
        else // Do standard mouse down
        */
        {
            switch (button)
            {
                case GLFW_MOUSE_BUTTON_LEFT:
                    app.mouseUp(svIndex, MB_left, x, y, modifiers);
                    break;
                case GLFW_MOUSE_BUTTON_RIGHT:
                    app.mouseUp(svIndex, MB_right, x, y, modifiers);
                    break;
                case GLFW_MOUSE_BUTTON_MIDDLE:
                    app.mouseUp(svIndex, MB_middle, x, y, modifiers);
                    break;
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
    /*
    if (modifiers & K_alt)
    {
        // Do parallel double finger move
        if (modifiers & K_shift)
        {
            app.touch2Move(svIndex, (int)x, (int)y, (int)x - touchDelta.x, (int)y - touchDelta.y);
        }
        else // Do concentric double finger pinch
        {
            int scrW2    = lastWidth / 2;
            int scrH2    = lastHeight / 2;
            touch2.x     = scrW2 - ((int)x - scrW2);
            touch2.y     = scrH2 - ((int)y - scrH2);
            touchDelta.x = (int)x - touch2.x;
            touchDelta.y = (int)y - touch2.y;
            app.touch2Move(svIndex, (int)x, (int)y, touch2.x, touch2.y);
        }
    }
    else // Do normal mouse move
    */
    {
        app.mouseMove(svIndex, (int)x, (int)y);
    }
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

    app.mouseWheel(svIndex, dY, modifiers);
}
//-----------------------------------------------------------------------------
/*!
Key event handler sets the modifier key state & forwards the event to
the slKeyPress function.
*/
static void onKeyPress(GLFWwindow* myWindow,
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
            default: break;
        }
    }
    else if (action == GLFW_RELEASE)
    {
        switch (key)
        {
            case K_ctrl: modifiers = (SLKey)(modifiers ^ K_ctrl); return;
            case K_alt: modifiers = (SLKey)(modifiers ^ K_alt); return;
            case K_shift: modifiers = (SLKey)(modifiers ^ K_shift); return;
            default: break;
        }
    }

    // Special treatment for ESC key
    /*
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
            app.keyPress(svIndex, key, modifiers);
            onClose(myWindow);
            glfwSetWindowShouldClose(myWindow, GL_TRUE);
        }
    }
    else */
    if (key == K_F9 && action == GLFW_PRESS) // Toggle fullscreen mode
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
    else if (key == K_space && action == GLFW_PRESS) //go back
    {
        app.goBack();
    }
    else if (GLFWKey == GLFW_KEY_H && action == GLFW_PRESS) //hold
    {
        app.hold();
    }
    else if (GLFWKey == GLFW_KEY_R && action == GLFW_PRESS) //resume
    {
        app.resume();
    }
    else
    {
        if (action == GLFW_PRESS)
        {
            app.keyPress(svIndex, key, modifiers);
        }
        else if (action == GLFW_RELEASE)
            app.keyRelease(svIndex, key, modifiers);
    }
}
//-----------------------------------------------------------------------------
//! Event handler for GLFW character input
void onCharInput(GLFWwindow*, SLuint c)
{
    app.charInput(svIndex, c);
}
//-----------------------------------------------------------------------------
/*!
Error callback handler for GLFW.
*/
void onGLFWError(int error, const char* description)
{
    fputs(description, stderr);
}
//-----------------------------------------------------------------------------
//! Initialises all GLFW and GL3W stuff
void GLFWInit()
{
    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
        exit(EXIT_FAILURE);
    }

    glfwSetErrorCallback(onGLFWError);

    // Enable fullscreen anti aliasing with 4 samples
    glfwWindowHint(GLFW_SAMPLES, 4);

#ifdef __APPLE__
    //You can enable or restrict newer OpenGL context here (read the GLFW documentation)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif

    touch2.set(-1, -1);
    touchDelta.set(-1, -1);

    window = glfwCreateWindow(scrWidth, scrHeight, "WAI Demo", nullptr, nullptr);

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
    glfwSetWindowTitle(window, "libWAI Test Application");
    glfwSetWindowPos(window, 10, 30);

    // Set number of monitor refreshes between 2 buffer swaps
    glfwSwapInterval(2);

    // Get GL errors that occurred before our framework is involved
    GET_GL_ERROR;

    // Set your own physical screen dpi
    dpi = getDpi();

    std::cout << "------------------------------------------------------------------" << endl;
    std::cout << "GUI             : GLFW (Version: " << GLFW_VERSION_MAJOR << "." << GLFW_VERSION_MINOR << "." << GLFW_VERSION_REVISION << ")" << endl;
    std::cout << "DPI             : " << dpi << endl;

    // Set GLFW callback functions
    glfwSetKeyCallback(window, onKeyPress);
    glfwSetCharCallback(window, onCharInput);
    glfwSetWindowSizeCallback(window, onResize);
    glfwSetMouseButtonCallback(window, onMouseButton);
    glfwSetCursorPosCallback(window, onMouseMove);
    glfwSetScrollCallback(window, onMouseWheel);
    glfwSetWindowCloseCallback(window, onClose);
}

//-----------------------------------------------------------------------------
void closeAppCallback()
{
    Utils::log("Engine", "closeAppCallback");
    appShouldClose = true;
}
//-----------------------------------------------------------------------------
/*!
The C main procedure running the GLFW GUI application.
*/
int main(int argc, char* argv[])
{
    GLFWInit();

    bool simulateSensors = true;
    bool useDummyGps     = false;
    try
    {
        std::unique_ptr<SENSSimulator> sensSim;
        std::unique_ptr<SENSWebCamera> webCamera;
        std::unique_ptr<SENSDummyGps>  dummyGps;
        std::unique_ptr<SENSARCore>    arcore;
        arcore = std::make_unique<SENSARCore>();

        SENSOrientation* orientation = nullptr;
        SENSGps*         gps         = nullptr;
        SENSCamera*      camera      = nullptr;
        if (simulateSensors)
        {
            //std::string simDir = Utils::getAppsWritableDir() + "SENSSimData/20201106-232556_SENSRecorder";
            //std::string simDir = Utils::getAppsWritableDir() + "SENSSimData/20201106-232621_SENSRecorder";
            //std::string simDir = Utils::getAppsWritableDir() + "SENSSimData/20201106-232643_SENSRecorder";

            std::string simDir = Utils::getAppsWritableDir() + "SENSSimData/20201113-131407_SENSRecorder";
            //std::string simDir = Utils::getAppsWritableDir() + "SENSSimData/20201113-131543_SENSRecorder";
            sensSim     = std::make_unique<SENSSimulator>(simDir);
            gps         = sensSim->getGpsSensorPtr();
            orientation = sensSim->getOrientationSensorPtr();
            camera      = sensSim->getCameraSensorPtr();
        }
        else
        {
            webCamera = std::make_unique<SENSWebCamera>();
            camera    = webCamera.get();

            if (useDummyGps)
            {
                dummyGps             = std::make_unique<SENSDummyGps>();
                SENSGps::Location tl = {47.14290, 7.24225, 506.3, 10.0f};
                SENSGps::Location br = {47.14060, 7.24693, 434.3, 1.0f};

                //interpolate n values
                int    n    = 10;
                double latD = (br.latitudeDEG - tl.latitudeDEG) / n;
                double lonD = (br.longitudeDEG - tl.longitudeDEG) / n;
                double altD = (br.altitudeM - tl.altitudeM) / n;
                double accD = (br.accuracyM - tl.accuracyM) / n;

                dummyGps->addDummyPos(tl);
                for (int i = 1; i < n; ++i)
                {
                    SENSGps::Location loc;
                    loc.latitudeDEG  = tl.latitudeDEG + i * latD;
                    loc.longitudeDEG = tl.longitudeDEG + i * lonD;
                    loc.altitudeM    = tl.altitudeM + i * altD;
                    loc.accuracyM    = tl.accuracyM + i * accD;
                    dummyGps->addDummyPos(loc);
                }
                dummyGps->addDummyPos(br);
                gps = dummyGps.get();
            }
        }

        app.init(scrWidth,
                 scrHeight,
                 dpi,
                 std::string(SL_PROJECT_ROOT) + "/data/",
                 Utils::getAppsWritableDir(),
                 camera,
                 gps,
                 orientation,
                 arcore.get());
        app.setCloseAppCallback(closeAppCallback);

        glfwSetWindowTitle(window, "ErlebAR");

        // Event loop
        while (!appShouldClose)
        {
            // Calculate screen to framebuffer ratio for high-DPI monitors
            /* This ratio can be different per monitor. We can not retrieve the
               correct framebuffer size until the first paint event is done. So
               we have to do it in here on every frame because we can move the window
               to another monitor. */
            int fbWidth = 0, fbHeight = 0, wndWidth = 0, wndHeight = 0;
            glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
            glfwGetWindowSize(window, &wndWidth, &wndHeight);
            float scr2fbX = (float)fbWidth / (float)wndWidth;
            float scr2fbY = (float)fbHeight / (float)wndHeight;
            app.updateScr2fb(svIndex, scr2fbX, scr2fbY);

            bool doRepaint = app.update();

            glfwSwapBuffers(window);

            // if no updated occurred wait for the next event (power saving)
            if (!doRepaint)
                glfwWaitEvents();
            else
                glfwPollEvents();
        }
    }
    catch (std::exception& e)
    {
        std::cout << "main: std exception catched: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cout << "main: Unknown exception catched!" << std::endl;
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
//-----------------------------------------------------------------------------
