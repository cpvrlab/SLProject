#include "CameraOnlyScene.h"

CameraOnlyScene::CameraOnlyScene(std::string name)
  : SLScene(name, nullptr)
{
    init();
    build();
}

void CameraOnlyScene::build()
{
    _root3D    = new SLNode("scene");
    cameraNode = new SLCamera("Camera 1");
    _mapNode   = new SLNode("map");

    _videoImage = new SLGLTexture(&assets, "LiveVideoError.png", GL_LINEAR, GL_LINEAR);
    cameraNode->background().texture(_videoImage);

    _root3D->addChild(_mapNode);
}

void CameraOnlyScene::updateVideoImage(const cv::Mat& image)
{
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
