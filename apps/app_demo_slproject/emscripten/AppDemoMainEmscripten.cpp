// #############################################################################
//   File:      AppDemoMainEmscripten.cpp
//   Purpose:   Application that demonstrates most features of the SLProject
//              framework with WebGL, WebAssembly and Emscripten in a web
//              browser. Implementation of the GUI is done with the emscripten
//              framework.
//   Date:      October 2022
//   Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//   Authors:   Marino von Wattenwyl
//   License:   This software is provided under the GNU General Public License
//              Please visit: http://opensource.org/licenses/GPL-3.0
// #############################################################################

#include <SLInterface.h>
#include <SLScene.h>
#include <AppDemo.h>
#include <AppDemoSceneView.h>
#include <AppDemoGui.h>
#include <CVCapture.h>

#include <GLFW/glfw3.h>
#include <GLES3/gl3.h>

#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/val.h>

static int canvasWidth;
static int canvasHeight;

static int    lastTouchDownX;
static int    lastTouchDownY;
static double lastTouchDownTimeMS;

static GLFWwindow* window;             //!< The global glfw window handle
static SLint       svIndex;            //!< SceneView index
static SLint       scrWidth;           //!< Window width at start up
static SLint       scrHeight;          //!< Window height at start up
static SLbool      fixAspectRatio;     //!< Flag if wnd aspect ratio should be fixed
static SLfloat     scrWdivH;           //!< aspect ratio screen width divided by height
static SLint       dpi = 142;          //!< Dot per inch resolution of screen
static SLint       startX;             //!< start position x in pixels
static SLint       startY;             //!< start position y in pixels
static SLint       mouseX;             //!< Last mouse position x in pixels
static SLint       mouseY;             //!< Last mouse position y in pixels
static SLVec2i     touch2;             //!< Last finger touch 2 position in pixels
static SLVec2i     touchDelta;         //!< Delta between two fingers in x
static SLint       lastWidth;          //!< Last window width in pixels
static SLint       lastHeight;         //!< Last window height in pixels
static SLbool      fullscreen = false; //!< flag if window is in fullscreen mode

//-----------------------------------------------------------------------------
extern void appDemoLoadScene(SLAssetManager* am,
                             SLScene*        s,
                             SLSceneView*    sv,
                             SLSceneID       sceneID);
extern bool onUpdateVideo();
//-----------------------------------------------------------------------------
void updateCanvas()
{
    // clang-format off
    EM_ASM({
        let canvas = Module['canvas'];
        canvas.width = $0;
        canvas.height = $1;
    }, canvasWidth, canvasHeight);
    // clang-format on
}
//-----------------------------------------------------------------------------
SLKey mapKeyToSLKey(unsigned long key)
{
    switch (key)
    {
        case 8: return K_backspace;
        case 9: return K_tab;
        case 13: return K_enter;
        case 16: return K_shift;
        case 17: return K_ctrl;
        case 18: return K_alt;
        case 27: return K_esc;
        case 32: return K_space;
        case 33: return K_pageUp;
        case 34: return K_pageDown;
        case 35: return K_end;
        case 36: return K_home;
        case 37: return K_left;
        case 38: return K_up;
        case 39: return K_right;
        case 40: return K_down;
        case 45: return K_insert;
        case 46: return K_delete;
        case 96: return K_NP0;
        case 97: return K_NP1;
        case 98: return K_NP2;
        case 99: return K_NP3;
        case 100: return K_NP4;
        case 101: return K_NP5;
        case 102: return K_NP6;
        case 103: return K_NP7;
        case 104: return K_NP8;
        case 105: return K_NP9;
        case 106: return K_NPMultiply;
        case 107: return K_NPAdd;
        case 109: return K_NPSubtract;
        case 110: return K_NPDecimal;
        case 111: return K_NPDivide;
        case 112: return K_F1;
        case 113: return K_F2;
        case 114: return K_F3;
        case 115: return K_F4;
        case 116: return K_F5;
        case 117: return K_F6;
        case 118: return K_F7;
        case 119: return K_F8;
        case 120: return K_F9;
        case 121: return K_F10;
        case 122: return K_F11;
        case 123: return K_F12;
        default: return (SLKey)key;
    }
}
//-----------------------------------------------------------------------------
SLKey mapModifiersToSLModifiers(bool shiftDown, bool ctrlDown, bool altDown)
{
    int modifiers = 0;
    if (shiftDown) modifiers |= K_shift;
    if (ctrlDown) modifiers |= K_ctrl;
    if (altDown) modifiers |= K_alt;
    return (SLKey)modifiers;
}
//-----------------------------------------------------------------------------
SLKey mapModifiersToSLModifiers(const EmscriptenMouseEvent* mouseEvent)
{
    return mapModifiersToSLModifiers(mouseEvent->shiftKey,
                                     mouseEvent->ctrlKey,
                                     mouseEvent->altKey);
}
//-----------------------------------------------------------------------------
SLKey mapModifiersToSLModifiers(const EmscriptenKeyboardEvent* keyEvent)
{
    return mapModifiersToSLModifiers(keyEvent->shiftKey,
                                     keyEvent->ctrlKey,
                                     keyEvent->altKey);
}
//-----------------------------------------------------------------------------
EMSCRIPTEN_RESULT emOnMousePressed(int                         eventType,
                                   const EmscriptenMouseEvent* mouseEvent,
                                   void*                       userData)
{
    SLint x         = mouseX;
    SLint y         = mouseY;
    SLKey modifiers = mapModifiersToSLModifiers(mouseEvent);

    startX = x;
    startY = y;

    switch (mouseEvent->button)
    {
        case 0:
            if (modifiers & K_alt && modifiers & K_ctrl)
                slTouch2Down(svIndex, x - 20, y, x + 20, y);
            else
                slMouseDown(svIndex,
                            MB_left,
                            x,
                            y,
                            modifiers);
            break;
        case 1:
            slMouseDown(svIndex,
                        MB_middle,
                        x,
                        y,
                        modifiers);
            break;
        case 2:
            slMouseDown(svIndex,
                        MB_right,
                        x,
                        y,
                        modifiers);
            break;
        default: break;
    }

    return EM_TRUE;
}
//-----------------------------------------------------------------------------
EM_BOOL emOnMouseReleased(int                         eventType,
                          const EmscriptenMouseEvent* mouseEvent,
                          void*                       userData)
{
    SLint x         = mouseX;
    SLint y         = mouseY;
    SLKey modifiers = mapModifiersToSLModifiers(mouseEvent);

    startX = -1;
    startY = -1;

    switch (mouseEvent->button)
    {
        case 0:
            slMouseUp(svIndex,
                      MB_left,
                      x,
                      y,
                      modifiers);
            break;
        case 1:
            slMouseUp(svIndex,
                      MB_middle,
                      x,
                      y,
                      modifiers);
            break;
        case 2:
            slMouseUp(svIndex,
                      MB_right,
                      x,
                      y,
                      modifiers);
            break;
        default: break;
    }

    return EM_TRUE;
}
//-----------------------------------------------------------------------------
EM_BOOL emOnMouseDoubleClicked(int                         eventType,
                               const EmscriptenMouseEvent* mouseEvent,
                               void*                       userData)
{
    SLint x         = mouseX;
    SLint y         = mouseY;
    SLKey modifiers = mapModifiersToSLModifiers(mouseEvent);

    switch (mouseEvent->button)
    {
        case GLFW_MOUSE_BUTTON_LEFT:
            slDoubleClick(svIndex,
                          MB_left,
                          x,
                          y,
                          modifiers);
            break;
        case GLFW_MOUSE_BUTTON_RIGHT:
            slDoubleClick(svIndex,
                          MB_right,
                          x,
                          y,
                          modifiers);
            break;
        case GLFW_MOUSE_BUTTON_MIDDLE:
            slDoubleClick(svIndex,
                          MB_middle,
                          x,
                          y,
                          modifiers);
            break;
        default: break;
    }

    return EM_TRUE;
}
//-----------------------------------------------------------------------------
EM_BOOL emOnMouseMove(int                         eventType,
                      const EmscriptenMouseEvent* mouseEvent,
                      void*                       userData)
{
    mouseX = (int)mouseEvent->targetX;
    mouseY = (int)mouseEvent->targetY;

    if (mouseEvent->altKey && mouseEvent->ctrlKey)
        slTouch2Move(svIndex,
                     mouseX - 20,
                     mouseY,
                     mouseX + 20,
                     mouseY);
    else
        slMouseMove(svIndex,
                    mouseX,
                    mouseY);

    return EM_TRUE;
}
//-----------------------------------------------------------------------------
EM_BOOL emOnMouseWheel(int                         eventType,
                       const EmscriptenWheelEvent* wheelEvent,
                       void*                       userData)
{
    // Invert the sign because the scroll value is inverted
    double deltaY = -wheelEvent->deltaY;

    // Make sure the delta is at least one integer
    if (std::abs(deltaY) < 1) deltaY = Utils::sign(wheelEvent->deltaY);

    SLKey modifiers = mapModifiersToSLModifiers(&wheelEvent->mouse);
    slMouseWheel(svIndex, (int)deltaY, modifiers);

    return EM_TRUE;
}
//-----------------------------------------------------------------------------
EM_BOOL emOnKeyPressed(int                            eventType,
                       const EmscriptenKeyboardEvent* keyEvent,
                       void*                          userData)
{
    if (keyEvent->repeat)
        return EM_TRUE;

    SLKey key       = mapKeyToSLKey(keyEvent->keyCode);
    SLKey modifiers = mapModifiersToSLModifiers(keyEvent);

    if (modifiers & K_alt && modifiers & K_shift)
    {
        SLSceneView* sv = AppDemo::sceneViews[0];

        if (key == '0' && sv)
        {
            appDemoLoadScene(AppDemo::assetManager,
                             AppDemo::scene,
                             sv,
                             SID_Empty);
            SL_LOG("Loading SceneID: %d", AppDemo::sceneID);
        }
        else if (key == K_left && sv && AppDemo::sceneID > 0)
        {
            appDemoLoadScene(AppDemo::assetManager,
                             AppDemo::scene,
                             sv,
                             (SLSceneID)(AppDemo::sceneID - 1));
            SL_LOG("Loading SceneID: %d", AppDemo::sceneID);
        }
        else if (key == K_right && sv && AppDemo::sceneID < SID_Maximal - 1)
        {
            appDemoLoadScene(AppDemo::assetManager,
                             AppDemo::scene,
                             sv,
                             (SLSceneID)(AppDemo::sceneID + 1));
            SL_LOG("Loading SceneID: %d", AppDemo::sceneID);
        }
    }

    slKeyPress(svIndex, key, modifiers);

    return EM_TRUE;
}
//-----------------------------------------------------------------------------
EM_BOOL emOnKeyReleased(int                            eventType,
                        const EmscriptenKeyboardEvent* keyEvent,
                        void*                          userData)
{
    SLKey key       = mapKeyToSLKey(keyEvent->keyCode);
    SLKey modifiers = mapModifiersToSLModifiers(keyEvent);
    slKeyRelease(svIndex, key, modifiers);

    return EM_TRUE;
}
//-----------------------------------------------------------------------------
EM_BOOL emOnTouchStart(int                         eventType,
                       const EmscriptenTouchEvent* touchEvent,
                       void*                       userData)
{
    if (touchEvent->numTouches == 1)
    {
        mouseX = (int)touchEvent->touches[0].clientX;
        mouseY = (int)touchEvent->touches[0].clientY;
        slMouseDown(svIndex,
                    MB_left,
                    mouseX,
                    mouseY,
                    K_none);
        lastTouchDownTimeMS = touchEvent->timestamp;
    }
    else if (touchEvent->numTouches == 2)
    {
        int x0 = (int)touchEvent->touches[0].clientX;
        int y0 = (int)touchEvent->touches[0].clientY;
        int x1 = (int)touchEvent->touches[1].clientX;
        int y1 = (int)touchEvent->touches[1].clientY;
        slTouch2Down(svIndex, x0, y0, x1, y1);
    }

    lastTouchDownX = mouseX;
    lastTouchDownY = mouseY;
    return EM_TRUE;
}
//-----------------------------------------------------------------------------
EM_BOOL emOnTouchEnd(int                         eventType,
                     const EmscriptenTouchEvent* touchEvent,
                     void*                       userData)
{
    if (touchEvent->numTouches == 1)
    {
        mouseX = (int)touchEvent->touches[0].clientX;
        mouseY = (int)touchEvent->touches[0].clientY;
        slMouseUp(svIndex,
                  MB_left,
                  mouseX,
                  mouseY,
                  K_none);

        int    dx = std::abs(mouseX - lastTouchDownX);
        int    dy = std::abs(mouseY - lastTouchDownY);
        double dt = touchEvent->timestamp - lastTouchDownTimeMS;

        if (dt > 800 && dx < 15 && dy < 15)
        {
            slMouseDown(svIndex,
                        MB_right,
                        lastTouchDownX,
                        lastTouchDownY,
                        K_none);
            slMouseUp(svIndex,
                      MB_right,
                      lastTouchDownX,
                      lastTouchDownY,
                      K_none);
        }
    }
    else if (touchEvent->numTouches == 2)
    {
        int x0 = (int)touchEvent->touches[0].clientX;
        int y0 = (int)touchEvent->touches[0].clientY;
        int x1 = (int)touchEvent->touches[1].clientX;
        int y1 = (int)touchEvent->touches[1].clientY;
        slTouch2Up(svIndex, x0, y0, x1, y1);
    }

    return EM_TRUE;
}
//-----------------------------------------------------------------------------
EM_BOOL emOnTouchMove(int                         eventType,
                      const EmscriptenTouchEvent* touchEvent,
                      void*                       userData)
{
    if (touchEvent->numTouches == 1)
    {
        mouseX = (int)touchEvent->touches[0].clientX;
        mouseY = (int)touchEvent->touches[0].clientY;
        slMouseMove(svIndex, mouseX, mouseY);
    }
    else if (touchEvent->numTouches == 2)
    {
        int x0 = (int)touchEvent->touches[0].clientX;
        int y0 = (int)touchEvent->touches[0].clientY;
        int x1 = (int)touchEvent->touches[1].clientX;
        int y1 = (int)touchEvent->touches[1].clientY;
        slTouch2Move(svIndex, x0, y0, x1, y1);
    }

    return EM_TRUE;
}
//-----------------------------------------------------------------------------
const char* emOnUnload(int         eventType,
                       const void* reserved,
                       void*       userData)
{
    slTerminate();
    return nullptr;
}
//-----------------------------------------------------------------------------
SLSceneView* createAppDemoSceneView(SLScene*        scene,
                                    int             curDPI,
                                    SLInputManager& inputManager)
{
    return (SLSceneView*)new AppDemoSceneView(scene,
                                              curDPI,
                                              inputManager);
}
//-----------------------------------------------------------------------------
bool onPaint()
{
    if (AppDemo::sceneViews.empty())
        return false;
    SLSceneView* sv = AppDemo::sceneViews[svIndex];

    int newCanvasWidth  = MAIN_THREAD_EM_ASM_INT(return window.innerWidth;);
    int newCanvasHeight = MAIN_THREAD_EM_ASM_INT(return window.innerHeight;);

    if (newCanvasWidth != canvasWidth || newCanvasHeight != canvasHeight)
    {
        canvasWidth  = newCanvasWidth;
        canvasHeight = newCanvasHeight;
        updateCanvas();

        if (!AppDemo::sceneViews.empty())
            slResize(svIndex,
                     canvasWidth,
                     canvasHeight);
    }

    // If live video image is requested grab it and copy it
    if (CVCapture::instance()->videoType() != VT_NONE)
    {
        float viewportWdivH = sv->viewportWdivH();
        CVCapture::instance()->grabAndAdjustForSL(viewportWdivH);
    }

    ///////////////////////////////////////////////
    onUpdateVideo();
    bool jobIsRunning      = slUpdateParallelJob();
    bool viewsNeedsRepaint = slPaintAllViews();
    ///////////////////////////////////////////////

    return jobIsRunning || viewsNeedsRepaint;
}
//-----------------------------------------------------------------------------
void onLoop()
{
    onPaint();
}
//-----------------------------------------------------------------------------
int main(void)
{
    canvasWidth  = MAIN_THREAD_EM_ASM_INT(return window.innerWidth);
    canvasHeight = MAIN_THREAD_EM_ASM_INT(return window.innerHeight);
    updateCanvas();

    EmscriptenWebGLContextAttributes attributes;
    emscripten_webgl_init_context_attributes(&attributes);
    attributes.enableExtensionsByDefault = true;
    attributes.antialias                 = false;
    attributes.depth                     = true;
    attributes.stencil                   = true;
    attributes.alpha                     = true;
    attributes.majorVersion              = 2;
    attributes.minorVersion              = 0;
    attributes.preserveDrawingBuffer     = true;

    auto context = emscripten_webgl_create_context("#canvas", &attributes);
    if (context > 0)
        SL_LOG("WebGL context created.");
    else
        SL_EXIT_MSG("Failed to create WebGL context.");

    EMSCRIPTEN_RESULT result = emscripten_webgl_make_context_current(context);
    if (result == EMSCRIPTEN_RESULT_SUCCESS)
        SL_LOG("WebGL context made current.");
    else
        SL_EXIT_MSG("Failed to make WebGL context current.");

    emscripten_set_mousedown_callback("#canvas", nullptr, true, emOnMousePressed);
    emscripten_set_mouseup_callback("#canvas", nullptr, true, emOnMouseReleased);
    emscripten_set_dblclick_callback("#canvas", nullptr, true, emOnMouseDoubleClicked);
    emscripten_set_mousemove_callback("#canvas", nullptr, true, emOnMouseMove);
    emscripten_set_wheel_callback("#canvas", nullptr, true, emOnMouseWheel);
    emscripten_set_keydown_callback("#canvas", nullptr, true, emOnKeyPressed);
    emscripten_set_keyup_callback("#canvas", nullptr, true, emOnKeyReleased);
    emscripten_set_touchstart_callback("#canvas", nullptr, true, emOnTouchStart);
    emscripten_set_touchend_callback("#canvas", nullptr, true, emOnTouchEnd);
    emscripten_set_touchmove_callback("#canvas", nullptr, true, emOnTouchMove);
    emscripten_set_beforeunload_callback(nullptr, emOnUnload);

    AppDemo::calibIniPath = "data/calibrations/";

    SLVstring args;
    slCreateAppAndScene(
      args,
      "data/",
      "data/shaders/",
      "data/models/",
      "data/images/textures/",
      "data/images/fonts/",
      "data/videos/",
      "data/config/",
      "AppDemoEmscripten",
      (void*)appDemoLoadScene);

    slCreateSceneView(
      AppDemo::assetManager,
      AppDemo::scene,
      canvasWidth,
      canvasHeight,
      dpi,
      (SLSceneID)SID_Minimal,
      (void*)&onPaint,
      nullptr,
      (void*)createAppDemoSceneView,
      (void*)AppDemoGui::build,
      (void*)AppDemoGui::loadConfig,
      (void*)AppDemoGui::saveConfig);

    // We cannot loop ourselves because that would block the page,
    // but we can register an update function to be called in every iteration
    // of the JavaScript event loop.
    emscripten_set_main_loop(onLoop, 0, true);

    return 0;
}
//-----------------------------------------------------------------------------