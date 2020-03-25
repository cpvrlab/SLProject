//#include <stdafx.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <Utils.h>

#include <SLApplication.h>
#include <SLInterface.h>
#include <SLGLProgram.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLBox.h>
#include <SLMaterial.h>
#include <SLLightSpot.h>
#include <SLPoints.h>
#include <SLKeyframeCamera.h>

#include <WAIHelper.h>
#include <WAIMode.h>
#include <WAISensorCamera.h>
#include <WAIKeyFrameDB.h>
#include <WAIMap.h>
#include <WAIOrbVocabulary.h>
#include <WAIFrame.h>

#include <OrbSlam/LocalMapping.h>
#include <OrbSlam/LoopClosing.h>
#include <OrbSlam/Initializer.h>
#include <OrbSlam/ORBmatcher.h>
#include <OrbSlam/Optimizer.h>
#include <OrbSlam/PnPsolver.h>
#include <OrbSlam/Optimizer.h>

static SLint   scrWidth;                   //!< Window width at start up
static SLint   scrHeight;                  //!< Window height at start up
static SLbool  fixAspectRatio;             //!< Flag if aspect ratio should be fixed
static SLfloat scrWdivH;                   //!< aspect ratio screen width divided by height
static SLfloat scr2fbX;                    //!< Factor from screen to framebuffer coords
static SLfloat scr2fbY;                    //!< Factor from screen to framebuffer coords
static SLint   startX;                     //!< start position x in pixels
static SLint   startY;                     //!< start position y in pixels
static SLint   mouseX;                     //!< Last mouse position x in pixels
static SLint   mouseY;                     //!< Last mouse position y in pixels
static SLVec2i touch2;                     //!< Last finger touch 2 position in pixels
static SLVec2i touchDelta;                 //!< Delta between two fingers in x
static SLint   lastWidth;                  //!< Last window width in pixels
static SLint   lastHeight;                 //!< Last window height in pixels
static SLfloat lastMouseDownTime = 0.0f;   //!< Last mouse press time
static SLKey   modifiers         = K_none; //!< last modifier keys
static SLbool  fullscreen        = false;  //!< flag if window is in fullscreen mode
GLFWwindow*    window;
int            svIndex;

void onClose(GLFWwindow* window)
{
    slShouldClose(true);
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
    }
    return (SLKey)key;
}
//-----------------------------------------------------------------------------
/*!
onResize: Event handler called on the resize event of the window. This event
should called once before the onPaint event.
*/
static void onResize(GLFWwindow* window, int width, int height)
{
    if (fixAspectRatio)
    {
        //correct target width and height
        if (height * scrWdivH <= width)
        {
            width  = (int)(height * scrWdivH);
            height = (int)(width / scrWdivH);
        }
        else
        {
            height = (int)(width / scrWdivH);
            width  = (int)(height * scrWdivH);
        }
    }

    lastWidth  = width;
    lastHeight = height;

    // width & height are in screen coords.
    // We need to scale them to framebuffer coords.
    slResize(svIndex, (int)(width * scr2fbX), (int)(height * scr2fbY));

    //update glfw window with new size
    glfwSetWindowSize(window, width, height);
}
//-----------------------------------------------------------------------------
/*!
onLongTouch gets called from a 500ms timer after a mouse down event.
*/
void onLongTouch()
{
    // forward the long touch only if the mouse or touch hasn't moved.
    if (Utils::abs(mouseX - startX) < 2 && Utils::abs(mouseY - startY) < 2)
        slLongTouch(svIndex, mouseX, mouseY);
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
                slTouch2Down(svIndex, x, y, x - touchDelta.x, y - touchDelta.y);
            }
            else // Do concentric double finger pinch
            {
                slTouch2Down(svIndex, x, y, touch2.x, touch2.y);
            }
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
                }
            }
            else // normal mouse clicks
            {
                // Start timer for the long touch detection
                HighResTimer::callAfterSleep(SLSceneView::LONGTOUCH_MS, onLongTouch);

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
                }
            }
        }
    }
    else
    { // flag end of mouse click for long touches
        startX = -1;
        startY = -1;

        // simulate double touch from touch devices
        if (modifiers & K_alt)
        {
            // Do parallel double finger move
            if (modifiers & K_shift)
            {
                slTouch2Up(svIndex, x, y, x - (touch2.x - x), y - (touch2.y - y));
            }
            else // Do concentric double finger pinch
            {
                slTouch2Up(svIndex, x, y, touch2.x, touch2.y);
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
            slTouch2Move(svIndex, (int)x, (int)y, (int)x - touchDelta.x, (int)y - touchDelta.y);
        }
        else // Do concentric double finger pinch
        {
            int scrW2    = lastWidth / 2;
            int scrH2    = lastHeight / 2;
            touch2.x     = scrW2 - ((int)x - scrW2);
            touch2.y     = scrH2 - ((int)y - scrH2);
            touchDelta.x = (int)x - touch2.x;
            touchDelta.y = (int)y - touch2.y;
            slTouch2Move(svIndex, (int)x, (int)y, touch2.x, touch2.y);
        }
    }
    else // Do normal mouse move
    {
        slMouseMove(svIndex, (int)x, (int)y);
    }
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
Key event handler sets the modifier key state & forwards the event to
the slKeyPress function.
*/
static void onKeyPress(GLFWwindow* window,
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
            slKeyPress(svIndex, key, modifiers);
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
//! Event handler for GLFW character input
void onCharInput(GLFWwindow*, SLuint c)
{
    slCharInput(svIndex, c);
}

void onGLFWError(int error, const char* description)
{
    fputs(description, stderr);
}

bool update()
{
    slUpdateScene();
    bool viewNeedsRepaint = slPaintAllViews();

    glfwSwapBuffers(window);

    return viewNeedsRepaint;
}

void renderMapPoints(std::string                      name,
                     const std::vector<WAIMapPoint*>& pts,
                     SLNode*&                         node,
                     SLPoints*&                       mesh,
                     SLMaterial*&                     material)
{
    //remove old mesh, if it exists
    if (mesh)
        node->deleteMesh(mesh);

    //instantiate and add new mesh
    if (pts.size())
    {
        //get points as Vec3f
        std::vector<SLVec3f> points, normals;
        for (WAIMapPoint* mapPt : pts)
        {
            WAI::V3 wP = mapPt->worldPosVec();
            WAI::V3 wN = mapPt->normalVec();
            points.push_back(SLVec3f(wP.x, wP.y, wP.z));
            normals.push_back(SLVec3f(wN.x, wN.y, wN.z));
        }

        mesh = new SLPoints(points, normals, name, material);
        node->addMesh(mesh);
        node->updateAABBRec();
    }
}

void renderKeyframes(const std::vector<WAIKeyFrame*>& keyframes,
                     SLNode*&                         node)
{
    node->deleteChildren();

    for (WAIKeyFrame* kf : keyframes)
    {
        if (kf->isBad())
            continue;

        SLKeyframeCamera* cam = new SLKeyframeCamera("KeyFrame " + std::to_string(kf->mnId));

        cv::Mat Twc = kf->getObjectMatrix();
        SLMat4f om;
        om.setMatrix(Twc.at<float>(0, 0),
                     -Twc.at<float>(0, 1),
                     -Twc.at<float>(0, 2),
                     Twc.at<float>(0, 3),
                     Twc.at<float>(1, 0),
                     -Twc.at<float>(1, 1),
                     -Twc.at<float>(1, 2),
                     Twc.at<float>(1, 3),
                     Twc.at<float>(2, 0),
                     -Twc.at<float>(2, 1),
                     -Twc.at<float>(2, 2),
                     Twc.at<float>(2, 3),
                     Twc.at<float>(3, 0),
                     -Twc.at<float>(3, 1),
                     -Twc.at<float>(3, 2),
                     Twc.at<float>(3, 3));

        cam->om(om);

        //calculate vertical field of view
        SLfloat fy     = (SLfloat)kf->fy;
        SLfloat cy     = (SLfloat)kf->cy;
        SLfloat fovDeg = 2 * (SLfloat)atan2(cy, fy) * Utils::RAD2DEG;
        cam->fov(fovDeg);
        cam->focalDist(0.11f);
        cam->clipNear(0.1f);
        cam->clipFar(1000.0f);

        node->addChild(cam);
    }
}

int main()
{
    std::string     dataRoot = std::string(SL_PROJECT_ROOT) + "/experimental/Initialization/testdata/";
    cv::FileStorage fs(dataRoot + "cam_calibration_main.xml", cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        return false;
    }

    cv::Mat cameraMat, distortionMat;

    fs["cameraMat"] >> cameraMat;
    fs["distortion"] >> distortionMat;

    // GLFW initialization
    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
        exit(EXIT_FAILURE);
    }

    glfwSetErrorCallback(onGLFWError);

    glfwWindowHint(GLFW_SAMPLES, 4);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    int scrWidth  = 640;
    int scrHeight = 480;

    window = glfwCreateWindow(scrWidth, scrHeight, "Initialization Test", nullptr, nullptr);
    glfwGetWindowSize(window, &scrWidth, &scrHeight);

    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);

    SLint fbWidth, fbHeight;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    scr2fbX = (float)fbWidth / (float)scrWidth;
    scr2fbY = (float)fbHeight / (float)scrHeight;

    glewExperimental = GL_TRUE; // avoids a crash
    GLenum err       = glewInit();
    if (GLEW_OK != err)
    {
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    glfwSetWindowTitle(window, "Initialization Test");
    glfwSetWindowPos(window, 10, 30);

    glfwSetKeyCallback(window, onKeyPress);
    glfwSetCharCallback(window, onCharInput);
    glfwSetWindowSizeCallback(window, onResize);
    glfwSetMouseButtonCallback(window, onMouseButton);
    glfwSetCursorPosCallback(window, onMouseMove);
    glfwSetScrollCallback(window, onMouseWheel);
    glfwSetWindowCloseCallback(window, onClose);

    GET_GL_ERROR;

    // SLProject initialization
    SLstring slRoot          = SLstring(SL_PROJECT_ROOT);
    SLGLProgram::defaultPath = slRoot + "/data/shaders/";

    SLScene* scene = new SLScene("Initialization Test Scene", nullptr);
    //scene->init();

    SLSceneView* sv = new SLSceneView();

    SLApplication::scene = scene;

    sv->init("SceneView", scrWidth, scrHeight, (void*)&update, nullptr, nullptr);
    scene->init();

    SLLightSpot* lightNode = new SLLightSpot(1, 1, 1, 0.3f);
    lightNode->ambient(SLCol4f(0.2f, 0.2f, 0.2f));
    lightNode->diffuse(SLCol4f(0.8f, 0.8f, 0.8f));
    lightNode->specular(SLCol4f(1, 1, 1));
    lightNode->attenuation(1, 0, 0);

    SLCamera* cameraNode = new SLCamera("Default Camera");
    cameraNode->translation(2.0f, 2.0f, 0.0f);
    cameraNode->lookAt(0, 0, 0);

    //calculate vertical field of view
    float fy     = cameraMat.at<double>(1, 1);
    float cy     = cameraMat.at<double>(1, 2);
    float fovRad = 2.0 * atan2(cy, fy);
    float fov    = fovRad * 180.0 / M_PI;

    //for tracking we have to use the field of view from calibration
    cameraNode->fov(fov);
    cameraNode->clipNear(0.001f);
    cameraNode->clipFar(1000000.0f); // Increase to infinity?
    cameraNode->setInitialState();

    sv->doWaitOnIdle(false);
    sv->camera(cameraNode);

    svIndex = sv->index();

    SLMaterial* yellow = new SLMaterial("mY", SLCol4f(1, 1, 0, 0.5f));
    SLfloat     l = 0.042f, b = 0.042f, h = 0.042f;
    SLBox*      box1    = new SLBox(0.0f, 0.0f, 0.0f, l, h, b, "Box 1", yellow);
    SLNode*     boxNode = new SLNode(box1, "boxNode");

    SLNode*   mapPCNode    = new SLNode("MapPC");
    SLNode*   keyFrameNode = new SLNode("KeyFrames");
    SLPoints* mapPointMesh = nullptr;

    SLMaterial* redMat = new SLMaterial(SLCol4f::RED, "Red");
    redMat->program(new SLGLGenericProgram("ColorUniformPoint.vert", "Color.frag"));
    redMat->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 3.0f));

    SLNode* mapNode = new SLNode("Map");
    mapNode->addChild(mapPCNode);
    mapNode->addChild(keyFrameNode);

    SLNode* rootNode = new SLNode("scene");
    rootNode->addChild(lightNode);
    rootNode->addChild(boxNode);
    rootNode->addChild(cameraNode);
    rootNode->addChild(mapNode);

    scene->root3D(rootNode);

    sv->onInitialize();
    //scene->onAfterLoad();

    // WAI initialization
    std::string orbVocFile   = std::string(SL_PROJECT_ROOT) + "/data/calibrations/ORBvoc.bin";
    int         nFeatures    = 1000;
    float       fScaleFactor = 1.2;
    int         nLevels      = 1;
    int         fIniThFAST   = 20;
    int         fMinThFAST   = 7;

    WAIOrbVocabulary::initialize(orbVocFile);
    ORB_SLAM2::ORBVocabulary* orbVoc = WAIOrbVocabulary::get();

    ORB_SLAM2::ORBextractor extractor = ORB_SLAM2::ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    int from_to[] = {0, 0};

    cv::Mat img1     = cv::imread(dataRoot + "chessboard_logitech_steep_01.jpg");
    cv::Mat img1Gray = cv::Mat(img1.rows,
                               img1.cols,
                               CV_8UC1);
    cv::mixChannels(&img1, 1, &img1Gray, 1, from_to, 1);

    cv::Mat img2     = cv::imread(dataRoot + "chessboard_logitech_steep_02.jpg");
    cv::Mat img2Gray = cv::Mat(img2.rows,
                               img2.cols,
                               CV_8UC1);
    cv::mixChannels(&img2, 1, &img2Gray, 1, from_to, 1);

    int flags =
      //CALIB_CB_ADAPTIVE_THRESH |
      //CALIB_CB_NORMALIZE_IMAGE |
      cv::CALIB_CB_FAST_CHECK;
    cv::Size chessboardSize(8, 5);

    std::vector<cv::Point2f> p2D1;
    bool                     found1 = cv::findChessboardCorners(img1Gray,
                                            chessboardSize,
                                            p2D1,
                                            flags);

    if (found1)
    {
        cv::drawChessboardCorners(img1, chessboardSize, p2D1, found1);
    }

    std::vector<cv::Point2f> p2D2;
    bool                     found2 = cv::findChessboardCorners(img2Gray,
                                            chessboardSize,
                                            p2D2,
                                            flags);

    if (found2)
    {
        cv::drawChessboardCorners(img2, chessboardSize, p2D2, found2);
    }

    WAIMap* map = new WAIMap("Map");

    if (found1 && found2)
    {
        std::vector<cv::Point3f> p3Dw;

        float chessboardWidthMM = 0.042f;
        for (int y = 0; y < chessboardSize.height; y++)
        {
            for (int x = 0; x < chessboardSize.width; x++)
            {
                p3Dw.push_back(cv::Point3f(y * chessboardWidthMM, x * chessboardWidthMM, 0.0f));
            }
        }

        cv::Mat r1, t1, r2, t2;
        bool    pose1Found = cv::solvePnP(p3Dw,
                                       p2D1,
                                       cameraMat,
                                       distortionMat,
                                       r1,
                                       t1,
                                       false,
                                       cv::SOLVEPNP_ITERATIVE);
        bool    pose2Found = cv::solvePnP(p3Dw,
                                       p2D2,
                                       cameraMat,
                                       distortionMat,
                                       r2,
                                       t2,
                                       false,
                                       cv::SOLVEPNP_ITERATIVE);

        if (pose1Found && pose2Found)
        {
            std::vector<int> matches;

            for (int i = 0; i < p2D1.size(); i++)
            {
                matches.push_back(i);
            }

            std::vector<cv::KeyPoint> kp1, kp2;
            for (int i = 0; i < p2D1.size(); i++)
            {
                kp1.push_back(cv::KeyPoint(p2D1[i], 1.0f));
            }

            for (int i = 0; i < p2D2.size(); i++)
            {
                kp2.push_back(cv::KeyPoint(p2D2[i], 1.0f));
            }

            WAIFrame frame1 = WAIFrame(img1Gray, &extractor, cameraMat, distortionMat, kp1, orbVoc);
            WAIFrame frame2 = WAIFrame(img2Gray, &extractor, cameraMat, distortionMat, kp2, orbVoc);

            cv::Mat rotMat1, rotMat2;
            cv::Rodrigues(r1, rotMat1);

            cv::Mat pose1 = cv::Mat::eye(4, 4, CV_32F);
            rotMat1.copyTo(pose1.rowRange(0, 3).colRange(0, 3));
            t1.copyTo(pose1.rowRange(0, 3).col(3));

            frame1.SetPose(pose1);

            std::cout << frame1.mTcw << std::endl;

            cv::Rodrigues(r2, rotMat2);

            cv::Mat pose2 = cv::Mat::eye(4, 4, CV_32F);
            rotMat2.copyTo(pose2.rowRange(0, 3).colRange(0, 3));
            t2.copyTo(pose2.rowRange(0, 3).col(3));

            frame2.SetPose(pose2);

            std::cout << frame2.mTcw << std::endl;

            cv::Mat                  r21, t21;
            std::vector<cv::Point3f> vP3De;
            std::vector<bool>        triangulated;
            ORB_SLAM2::Initializer   initializer(frame1, 1.0f, 200);
            bool                     initializationSuccess = initializer.InitializeWithKnownPose(frame1, frame2, matches, r21, t21, vP3De, triangulated);

            WAIKeyFrameDB* kfDB = new WAIKeyFrameDB(*orbVoc);

            WAIKeyFrame* pKFini = new WAIKeyFrame(frame1, map, kfDB);
            WAIKeyFrame* pKFcur = new WAIKeyFrame(frame2, map, kfDB);

            map->AddKeyFrame(pKFini);
            map->AddKeyFrame(pKFcur);

            int triangulatedCount = 0;
            for (int i = 0; i < vP3De.size(); i++)
            {
                std::cout << vP3De[i] << " : " << p3Dw[i] << " : " << cv::norm(vP3De[i] - p3Dw[i]) << std::endl;

                if (!triangulated[i])
                {
                    continue;
                }

                triangulatedCount++;

                //Create MapPoint.
                cv::Mat worldPos(vP3De[i]);

                WAIMapPoint* pMP = new WAIMapPoint(worldPos, pKFcur);

                pKFini->AddMapPoint(pMP, i);
                pKFcur->AddMapPoint(pMP, matches[i]);

                pMP->AddObservation(pKFini, i);
                pMP->AddObservation(pKFcur, matches[i]);

                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();

                //Add to Map
                map->AddMapPoint(pMP);
            }

            pKFini->UpdateConnections();
            pKFcur->UpdateConnections();

            Optimizer::GlobalBundleAdjustemnt(map, 20);

            printf("triangulated %i points (of %i)\n", triangulatedCount, matches.size());
        }
    }

    while (!slShouldClose())
    {
        renderMapPoints("MapPoints",
                        map->GetAllMapPoints(),
                        mapPCNode,
                        mapPointMesh,
                        redMat);
        renderKeyframes(map->GetAllKeyFrames(), keyFrameNode);

        update();

        glfwPollEvents();
    }

    slTerminate();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
