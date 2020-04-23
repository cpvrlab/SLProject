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
    _gui(eventHandler, resources, dotsPerInch, screenWidth, screenHeight, fontPath),
    _camera(camera)
{
    scene(&_scene);
    init("AreaTrackingView", screenWidth, screenHeight, nullptr, nullptr, &_gui, imguiIniPath);
    //todo: cant we move the init and the build into constructor or AreaTrackingScene? What is init good for if the SLScene is empty anyway?
    _scene.init();
    _scene.build(); //->moved to constructor of AreaTrackingScene
    onInitialize();
    //set scene camera into sceneview
    this->camera(_scene.cameraNode);

    _locations = ErlebAR::defineLocations();
}

bool AreaTrackingView::update()
{
    //update video
    SENSFramePtr frame;
    if (_camera)
        frame = _camera->getLatestFrame();

    if (frame)
    {
        _scene.updateVideoImage(frame->imgRGB);
        cv::imwrite("karvuras.jpg", frame->imgRGB);
    }

    return onPaint();
}

void AreaTrackingView::initArea(ErlebAR::LocationId locId, ErlebAR::AreaId areaId)
{
    _gui.initArea(_locations[locId].areas[areaId]);
    //start camera
    startCamera();

    //start wai with map for this area
    //load model into scene graph
}

void AreaTrackingView::startCamera()
{
    if (_camera)
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
