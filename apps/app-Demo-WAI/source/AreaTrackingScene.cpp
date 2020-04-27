#include "AreaTrackingScene.h"

AreaTrackingScene::AreaTrackingScene()
  : SLScene("AreaTrackingScene", nullptr)
{
    init();
    build();
}

void AreaTrackingScene::build()
{
    _root3D    = new SLNode("scene");
    cameraNode = new SLCamera("Camera 1");
    _mapNode   = new SLNode("map");

    _videoImage = new SLGLTexture(&assets, "LiveVideoError.png", GL_LINEAR, GL_LINEAR);
    cameraNode->background().texture(_videoImage);

    _root3D->addChild(_mapNode);
}

void AreaTrackingScene::updateVideoImage(const cv::Mat& image)
{
    _videoImage->copyVideoImage(image.cols,
                                image.rows,
                                CVImage::cv2glPixelFormat(image.type()),
                                image.data,
                                image.isContinuous(),
                                true);
}

void AreaTrackingScene::modelTransparencyChanged(float newValue)
{
    Utils::log("AreaTrackingScene", "modelTransparencyChanged, new value: %f", newValue);
}
