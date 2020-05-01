#include <views/CameraTestView.h>
#include <sens/SENSUtils.h>

CameraTestView::CameraTestView(sm::EventHandler&   eventHandler,
                               SLInputManager&     inputManager,
                               ErlebAR::Resources& resources,
                               SENSCamera*         sensCamera,
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
         fontPath,
         sensCamera),
    _scene("CameraTestScene"),
    _sensCamera(sensCamera)
{
    scene(&_scene);
    init("CameraTestView", screenWidth, screenHeight, nullptr, nullptr, &_gui, imguiIniPath);
    onInitialize();
    //set the cameraNode as active camera so that we see the video image
    camera(_scene.cameraNode);
}

bool CameraTestView::update()
{
    if (_gui.camera() && _gui.camera()->started())
    {
        SENSFramePtr frame = _gui.camera()->getLatestFrame();
        if (frame)
        {
            //add bars to image instead of viewport adjustment (we update the mat in the buffer)
            //todo: the matrices in the buffer have different sizes.. problem? no! no!
            int addW, addH;
            SENS::extendWithBars(frame->imgRGB, this->viewportWdivH(), cv::BORDER_CONSTANT, addW, addH);
            _scene.updateVideoImage(frame->imgRGB);
        }
    }

    return onPaint();
}

void CameraTestView::startCamera()
{
    if (!_camera)
        return;

    //SENSCamera::Config config;
    //config.targetWidth   = 640;
    //config.targetHeight  = 360;
    //config.convertToGray = true;

    //_camera->init(SENSCameraFacing::BACK);
    //_camera->start(config);
}

void CameraTestView::stopCamera()
{
    if (!_gui.camera())
        return;

    _gui.camera()->stop();
}
