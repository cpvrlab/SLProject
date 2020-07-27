#include "CameraOnlyScene.h"

CameraOnlyScene::CameraOnlyScene(std::string name, std::string dataDir)
  : SLScene(name, nullptr),
    _dataDir(dataDir)
{
    init();
    build();
}

void CameraOnlyScene::build()
{
    _root3D    = new SLNode("scene");
    cameraNode = new SLCamera("Camera 1");
    _mapNode   = new SLNode("map");

    _videoImage = new SLGLTexture(&assets, _dataDir + "images/textures/LiveVideoError.png", GL_LINEAR, GL_LINEAR);
    cameraNode->background().texture(_videoImage, true);

    _root3D->addChild(_mapNode);
}

void CameraOnlyScene::updateVideoImage(const cv::Mat& image)
{
    float newImgWdivH = (float)image.cols / (float)image.rows;
    float oldImgWdivH = (float)cameraNode->background().texture()->width() / (float)cameraNode->background().texture()->height();
    if (std::abs(newImgWdivH - oldImgWdivH) > 0.001f)
    {
        cameraNode->background().texture(_videoImage, true);
    }
    _videoImage->copyVideoImage(image.cols,
                                image.rows,
                                CVImage::cv2glPixelFormat(image.type()),
                                image.data,
                                image.isContinuous(),
                                true);
}

void CameraOnlyScene::modelTransparencyChanged(float newValue)
{
    Utils::log("AreaTrackingScene", "modelTransparencyChanged, new value: %f", newValue);
}
