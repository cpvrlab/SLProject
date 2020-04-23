#include <views/AreaTrackingView.h>
#include <sens/SENSCamera.h>

AreaTrackingView::AreaTrackingView(sm::EventHandler&   eventHandler,
                                   SLInputManager&     inputManager,
                                   ErlebAR::Resources& resources,
                                   SENSCamera*         camera,
                                   int                 screenWidth,
                                   int                 screenHeight,
                                   int                 dotsPerInch,
                                   std::string         fontPath,
                                   std::string         imguiIniPath)
  : SLSceneView(nullptr, dotsPerInch, inputManager),
    _gui(eventHandler,
         resources,
         dotsPerInch,
         screenWidth,
         screenHeight,
         std::bind(&AreaTrackingScene::modelTransparencyChanged, &_scene, std::placeholders::_1),
         fontPath),
    _camera(camera)
{
    scene(&_scene);
    init("AreaTrackingView", screenWidth, screenHeight, nullptr, nullptr, &_gui, imguiIniPath);
    //todo: ->moved to constructor of AreaTrackingScene: can this lead to any problems?
    //_scene.init();
    //_scene.build();
    onInitialize();
    //set scene camera into sceneview
    this->camera(_scene.cameraNode);

    _locations = resources.locations();
}

bool AreaTrackingView::update()
{
    SENSFramePtr frame;
    if (_camera)
        frame = _camera->getLatestFrame();

    if (frame)
    {
        _scene.updateVideoImage(frame->imgRGB);
    }

    return onPaint();
}

void AreaTrackingView::initArea(ErlebAR::LocationId locId, ErlebAR::AreaId areaId)
{
    _gui.initArea(_locations[locId].areas[areaId]);
    //start camera
    startCamera();

    //start wai with map for this area (as non-blocking as possible)

    //load model into scene graph
    //todo: separate loading from opengl calls (task in projectplan)
}

void AreaTrackingView::startCamera()
{
    if (_camera && !_camera->started())
    {
        //start camera
        SENSCamera::Config config;
        config.targetWidth   = 640;
        config.targetHeight  = 360;
        config.convertToGray = true;

        _camera->init(SENSCamera::Facing::BACK);
        _camera->start(config);
    }
}
