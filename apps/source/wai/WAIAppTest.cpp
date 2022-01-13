#include "WAIAppTest.h"

#include <AppDemo.h>
#include <SLGLTexture.h>
#include <SLGLProgram.h>
#include <SLAssimpImporter.h>
#include <SLSceneView.h>
#include <SLLightSpot.h>
#include <SLTexFont.h>
#include <SLSphere.h>
#include <SLText.h>
#include <SENSCamera.h>

#define WAIAPP_DEBUG(...) Utils::log("WAIApp", __VA_ARGS__)
#define WAIAPP_INFO(...) Utils::log("WAIApp", __VA_ARGS__)
#define WAIAPP_WARN(...) Utils::log("WAIApp", __VA_ARGS__)

//#define WAIAPP_DEBUG(...) // nothing
//#define WAIAPP_INFO(...)  // nothing
//#define WAIAPP_WARN(...)  // nothing

#define WAIAPPSTATE_DEBUG(...) Utils::log("WAIAppStateHandler", __VA_ARGS__)
#define WAIAPPSTATE_INFO(...) Utils::log("WAIAppStateHandler", __VA_ARGS__)
#define WAIAPPSTATE_WARN(...) Utils::log("WAIAppStateHandler", __VA_ARGS__)

//#define WAIAPPSTATE_DEBUG(...) // nothing
//#define WAIAPPSTATE_INFO(...)  // nothing
//#define WAIAPPSTATE_WARN(...)  // nothing

WAIApp::WAIApp()
  : SLInputEventInterface(AppDemo::inputManager)
{
}

void WAIApp::init(int screenWidth, int screenHeight, int screenDpi, AppDirectories directories)
{
    WAIAPPSTATE_DEBUG("init");
    // Utils::initFileLog(directories.logFileDir, true);

    if (!_initSceneGraphDone)
    {
        initDirectories(directories);
        initSceneGraph(screenWidth, screenHeight, screenDpi);
        _initSceneGraphDone = true;
    }
    if (!_initIntroSceneDone)
    {
        initIntroScene();
        _initIntroSceneDone = true;
    }
}

void WAIApp::initCloseAppCallback(CloseAppCallback cb)
{
    _closeAppCallback = cb;
}

void WAIApp::initCamera(SENSCamera* camera)
{
    _camera = camera;

    /*
    if (_sv)
        _sv->setViewportFromRatio(SLVec2i(_camera->getFrameSize().width, _camera->getFrameSize().height), SLViewportAlign::VA_center, true);

    if (_sceneCamera)
    {
        //make sure scene camera is initialized
        initSceneCamera();
    }

    if (!_videoImage)
    {
        _videoImage = new SLGLTexture("LiveVideoError.png", GL_LINEAR, GL_LINEAR);
        _sceneCamera->background().texture(_videoImage);
    }
     */
}

void WAIApp::initDirectories(AppDirectories directories)
{
    WAIAPP_DEBUG("initDirectories");
    _dirs = directories;
    // Default paths for all loaded resources
    SLGLProgram::defaultPath      = _dirs.slDataRoot + "/shaders/";
    SLGLTexture::defaultPath      = _dirs.slDataRoot + "/images/textures/";
    SLGLTexture::defaultPathFonts = _dirs.slDataRoot + "/images/fonts/";
    CVImage::defaultPath          = _dirs.slDataRoot + "/images/textures/";
    SLAssimpImporter::defaultPath = _dirs.slDataRoot + "/models/";
    AppDemo::configPath           = _dirs.writableDir;
}

void WAIApp::initSceneGraph(int scrWidth, int scrHeight, int dpi)
{
    WAIAPP_DEBUG("initSceneGraph");
    if (!AppDemo::scene)
    {
        AppDemo::name  = "WAI Demo App";
        AppDemo::scene = new SLScene("WAI Demo App", nullptr);

        int screenWidth  = scrWidth;
        int screenHeight = scrHeight;

        // setupGUI(AppDemo::name, AppDemo::configPath, dpi);
        //  Set default font sizes depending on the dpi no matter if ImGui is used
        // todo: is this still needed?
        if (!AppDemo::dpi)
            AppDemo::dpi = dpi;

        _sv = new SLSceneView();
        _sv->init("SceneView",
                  screenWidth,
                  screenHeight,
                  nullptr,
                  nullptr,
                  nullptr);
    }
    else
    {
        WAIAPP_WARN("initSceneGraph: SLScene already loaded!");
    }
}

bool WAIApp::update()
{
    if (_goBackRequested && _closeAppCallback)
    {
        _closeAppCallback();
        _goBackRequested = false;
    }

    if (!_initSceneGraphDone)
        return false;

    // WAIAPP_DEBUG("render");
    if (_initIntroSceneDone)
    {
        if (_camera)
        {
            // if (_camera->started())
            //{

            //}

            // copy video image to background
            SENSFramePtr frame = _camera->getLatestFrame();
            if (frame)
            {
                if (!_videoImage)
                {
                    if (!_sceneCamera)
                    {
                        // make sure scene camera is initialized
                        initSceneCamera();
                    }

                    if (_sv)
                        _sv->setViewportFromRatio(SLVec2i(_camera->getFrameSize().width,
                                                          _camera->getFrameSize().height),
                                                  SLViewportAlign::VA_center,
                                                  true);

                    _videoImage = new SLGLTexture("LiveVideoError.png", GL_LINEAR, GL_LINEAR);
                    _sceneCamera->background().texture(_videoImage);
                }

                WAIAPP_DEBUG("valid frame");
                cv::Mat& img = frame->imgRGB;
                _videoImage->copyVideoImage(img.cols,
                                            img.rows,
                                            CVImage::cv2glPixelFormat(img.type()),
                                            img.data,
                                            img.isContinuous(),
                                            true);
            }
        }

        // update scene
        AppDemo::scene->onUpdate();

        // update sceneviews
        bool needUpdate = false;
        for (auto sv : AppDemo::scene->sceneViews())
            if (sv->onPaint() && !needUpdate)
                needUpdate = true;

        return needUpdate;
    }
    else
    {
        WAIAPP_WARN("render: SLScene not initialized!");
        return false;
    }
}

void WAIApp::close()
{
    deleteSceneGraph();
    _initSceneGraphDone = false;
    _initIntroSceneDone = false;
}

void WAIApp::goBack()
{
    WAIAPP_DEBUG("goBack");
    // todo: enqueue event
    _goBackRequested = true;
}

void WAIApp::deleteSceneGraph()
{
    // Deletes all remaining sceneviews the current scene instance
    if (AppDemo::scene)
    {
        delete AppDemo::scene;
        AppDemo::scene = nullptr;
    }
}

void WAIApp::initSceneCamera()
{
    if (!_sceneCamera)
    {
        _sceneCamera = new SLCamera("Camera 1");
        _sceneCamera->clipNear(0.1f);
        _sceneCamera->clipFar(100);
        _sceneCamera->translation(0, 0, 5);
        _sceneCamera->lookAt(0, 0, 0);
        _sceneCamera->focalDist(5);
        _sceneCamera->background().colors(SLCol4f(0.7f, 0.7f, 0.7f),
                                          SLCol4f(0.2f, 0.2f, 0.2f));
        _sceneCamera->setInitialState();
    }
}

void WAIApp::initIntroScene()
{
    WAIAPP_DEBUG("initIntroScene");
    SLScene* s = AppDemo::scene;
    // clear old scene content
    s->init();

    s->name("Loading scene");
    s->info("Scene shown while starting application");

    SLMaterial* m1 = new SLMaterial("m1", SLCol4f::RED);

    initSceneCamera(); //_sceneCamera is valid now

    SLLightSpot* light1 = new SLLightSpot(10, 10, 10, 0.3f);
    light1->ambient(SLCol4f(0.2f, 0.2f, 0.2f));
    light1->diffuse(SLCol4f(0.8f, 0.8f, 0.8f));
    light1->specular(SLCol4f(1, 1, 1));
    light1->attenuation(1, 0, 0);

    // Because all text objects get their sizes in pixels we have to scale them down
    SLfloat  scale = 0.01f;
    SLstring txt   = "This is text in 3D with font07";
    SLVec2f  size  = SLTexFont::font07->calcTextSize(txt);
    SLNode*  t07   = new SLText(txt, SLTexFont::font07);
    t07->translate(-size.x * 0.5f * scale, 1.0f, 0);
    t07->scale(scale);

    txt         = "This is text in 3D with font22";
    size        = SLTexFont::font22->calcTextSize(txt);
    SLNode* t22 = new SLText(txt, SLTexFont::font22);
    t22->translate(-size.x * 0.5f * scale, -1.2f, 0);
    t22->scale(scale);

    // Assemble 3D scene as usual with camera and light
    SLNode* scene3D = new SLNode("root3D");
    scene3D->addChild(_sceneCamera);
    scene3D->addChild(light1);
    scene3D->addChild(new SLNode(new SLSphere(0.5f, 32, 32, "Sphere", m1)));
    scene3D->addChild(t07);
    scene3D->addChild(t22);

    _sv->camera(_sceneCamera);
    _sv->doWaitOnIdle(false);

    s->root3D(scene3D);

    for (auto sceneView : s->sceneViews())
        if (sceneView != nullptr)
            sceneView->onInitialize();

    /*
    WAIAPP_DEBUG("initIntroScene");
    SLScene* s = AppDemo::scene;
    //clear old scene content
    s->init();

    s->name("Loading scene");
    s->info("Scene shown while starting application");

    SLMaterial* m1 = new SLMaterial("m1", SLCol4f::RED);

    SLCamera* cam1 = new SLCamera("Camera 1");
    cam1->clipNear(0.1f);
    cam1->clipFar(100);
    cam1->translation(0, 0, 5);
    cam1->lookAt(0, 0, 0);
    cam1->focalDist(5);
    cam1->background().colors(SLCol4f(0.1f, 0.1f, 0.1f));
    cam1->setInitialState();

    SLLightSpot* light1 = new SLLightSpot(10, 10, 10, 0.3f);
    light1->ambient(SLCol4f(0.2f, 0.2f, 0.2f));
    light1->diffuse(SLCol4f(0.8f, 0.8f, 0.8f));
    light1->specular(SLCol4f(1, 1, 1));
    light1->attenuation(1, 0, 0);

    // Because all text objects get their sizes in pixels we have to scale them down
    SLfloat  scale = 0.01f;
    SLstring txt   = "This is text in 3D with font07";
    SLVec2f  size  = SLTexFont::font07->calcTextSize(txt);
    SLNode*  t07   = new SLText(txt, SLTexFont::font07);
    t07->translate(-size.x * 0.5f * scale, 1.0f, 0);
    t07->scale(scale);

    txt         = "This is text in 3D with font09";
    size        = SLTexFont::font09->calcTextSize(txt);
    SLNode* t09 = new SLText(txt, SLTexFont::font09);
    t09->translate(-size.x * 0.5f * scale, 0.8f, 0);
    t09->scale(scale);

    txt         = "This is text in 3D with font12";
    size        = SLTexFont::font12->calcTextSize(txt);
    SLNode* t12 = new SLText(txt, SLTexFont::font12);
    t12->translate(-size.x * 0.5f * scale, 0.6f, 0);
    t12->scale(scale);

    txt         = "This is text in 3D with font20";
    size        = SLTexFont::font20->calcTextSize(txt);
    SLNode* t20 = new SLText(txt, SLTexFont::font20);
    t20->translate(-size.x * 0.5f * scale, -0.8f, 0);
    t20->scale(scale);

    txt         = "This is text in 3D with font22";
    size        = SLTexFont::font22->calcTextSize(txt);
    SLNode* t22 = new SLText(txt, SLTexFont::font22);
    t22->translate(-size.x * 0.5f * scale, -1.2f, 0);
    t22->scale(scale);

    // Now create 2D text but don't scale it (all sizes in pixels)
    txt           = "This is text in 2D with font16";
    size          = SLTexFont::font16->calcTextSize(txt);
    SLNode* t2D16 = new SLText(txt, SLTexFont::font16);
    t2D16->translate(-size.x * 0.5f, 0, 0);

    // Assemble 3D scene as usual with camera and light
    SLNode* scene3D = new SLNode("root3D");
    scene3D->addChild(cam1);
    scene3D->addChild(light1);
    scene3D->addChild(new SLNode(new SLSphere(0.5f, 32, 32, "Sphere", m1)));
    scene3D->addChild(t07);
    scene3D->addChild(t09);
    scene3D->addChild(t12);
    scene3D->addChild(t20);
    scene3D->addChild(t22);

    // Assemble 2D scene
    SLNode* scene2D = new SLNode("root2D");
    scene2D->addChild(t2D16);

    _sv->camera(cam1);
    _sv->doWaitOnIdle(true);

    s->root3D(scene3D);
    s->root2D(scene2D);
    */
}

/*
void WAIApp::enableSceneGraph()
{
    SLScene* s = AppDemo::scene;
    for (auto sceneView : s->sceneViews())
        if (sceneView != nullptr)
            sceneView->onInitialize();

}
*/

/*
WAIAppStateHandler::WAIAppStateHandler(CloseAppCallback cb)
  : SLInputEventInterface(AppDemo::inputManager),
    _closeAppCallback(cb)
{
    WAIAPPSTATE_DEBUG("constructor");
    _waiApp = std::make_unique<WAIApp>();
}

bool WAIAppStateHandler::update()
{
    //WAIAPPSTATE_DEBUG("update");

    if (_goBackRequested && _closeAppCallback)
    {
        _closeAppCallback();
        _goBackRequested = false;
    }

    if (_initIntroSceneDone)
    {
        if (_initIntroSceneDone)
        {
            _waiApp->render();
        }
    }

    return true;
    //checkStateTransition();
    //return processState();
}
 */

/*
void WAIAppStateHandler::checkStateTransition()
{
    WAIAPPSTATE_DEBUG("checkStateTransition");
    switch (_state)
    {
        case State::STARTUP:
        {
            if (_initIntroSceneDone)
            {
                WAIAPPSTATE_DEBUG("checkStateTransition: transition to state INTROSCENE");
                _state = State::INTROSCENE;
            }
            break;
        }
        case State::INTROSCENE:
        {

            break;
        }
        case State::START_SLAM_SCENE:
        {
            break;
        }
        case State::SLAM_SCENE:
        {
            break;
        }
    };
}

bool WAIAppStateHandler::processState()
{
    WAIAPPSTATE_DEBUG("processState");
    bool updateScreen = false;
    switch (_state)
    {
        case State::STARTUP:
        {
            break;
        }
        case State::INTROSCENE:
        {
            //_waiApp->updateIntroScene();
            updateScreen = _waiApp->render();
            break;
        }
        case State::START_SLAM_SCENE:
        {
            break;
        }
        case State::SLAM_SCENE:
        {
            break;
        }
    };

    return updateScreen;
}
 */

/*

void WAIAppStateHandler::init(int screenWidth, int screenHeight, int screenDpi, AppDirectories directories)
{
    WAIAPPSTATE_DEBUG("init");
    //Utils::initFileLog(directories.logFileDir, true);

    if (!_initSceneGraphDone)
    {
        _waiApp->initDirectories(directories);
        _waiApp->initSceneGraph(screenWidth, screenHeight, screenDpi);
        _initSceneGraphDone = true;
    }
    if (!_initIntroSceneDone)
    {
        _waiApp->initIntroScene();
        _initIntroSceneDone = true;
    }
}
/*

/*
void WAIAppStateHandler::show()
{
    WAIAPPSTATE_DEBUG("show");
    //_waiApp->enableSceneGraph();
    //_waiApp->initIntroScene();
}

void WAIAppStateHandler::hide()
{
    WAIAPPSTATE_DEBUG("hide");
}
 */

/*
void WAIAppStateHandler::close()
{
    WAIAPPSTATE_DEBUG("close");
    _waiApp->deleteSceneGraph();
    _initSceneGraphDone = false;
    _initIntroSceneDone = false;
}

//back button was pressed
void WAIAppStateHandler::goBack()
{
    WAIAPPSTATE_DEBUG("goBack");
    //todo: enqueue event
    _goBackRequested = true;
}
*/