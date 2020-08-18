#include <views/CameraTestView.h>
#include <sens/SENSUtils.h>

CameraTestView::CameraTestView(sm::EventHandler&   eventHandler,
                               SLInputManager&     inputManager,
                               const ImGuiEngine&  imGuiEngine,
                               ErlebAR::Resources& resources,
                               SENSCamera*         sensCamera,
                               const DeviceData&   deviceData)
  : SLSceneView(nullptr, deviceData.dpi(), inputManager),
    _gui(imGuiEngine,
         eventHandler,
         resources,
         deviceData.dpi(),
         deviceData.scrWidth(),
         deviceData.scrHeight(),
         sensCamera),
    _scene("CameraTestScene", deviceData.dataDir()),
    _sensCamera(sensCamera),
    _deviceData(deviceData)
{
    scene(&_scene);
    init("CameraTestView", deviceData.scrWidth(), deviceData.scrHeight(), nullptr, nullptr, &_gui, deviceData.writableDir());
    onInitialize();
    //set the cameraNode as active camera so that we see the video image
    camera(_scene.cameraNode);
}

bool CameraTestView::update()
{
    if (_gui.camera() && _gui.camera()->started())
    {
        SENSFramePtr frame = _gui.camera()->latestFrame();

        if (frame)
        {
            //std::string imgFileName = _deviceData.writableDir() + "imgTest.jpg";
            //Utils::log("CameraTestView", "saving image to %s", imgFileName.c_str());
            //cv::imwrite(imgFileName, frame->imgRGB);

            //add bars to image instead of viewport adjustment (we update the mat in the buffer)
            //todo: the matrices in the buffer have different sizes.. problem? no! no!
            //if(!frame->extended)
            //{
            //Utils::log("SENSiOSCamera", "update before: w %d w %d", frame->imgRGB.size().width, frame->imgRGB.size().height);
            //SENS::extendWithBars(frame->imgRGB, this->viewportWdivH());
            //Utils::log("SENSiOSCamera", "update after: w %d w %d", frame->imgRGB.size().width, frame->imgRGB.size().height);
            //frame->extended = true;
            //}s
            SENS::extendWithBars(frame->imgRGB, this->viewportWdivH());
            _scene.updateVideoImage(frame->imgRGB);
        }
    }

    return onPaint();
}

void CameraTestView::startCamera()
{
    if (!_camera)
        return;

    //SENSCameraConfig config;
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
