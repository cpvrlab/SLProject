#ifndef CAMERA_ONLY_SCENE_H
#define CAMERA_ONLY_SCENE_H

#include <SLScene.h>
#include <SLAssetManager.h>

class CameraOnlyScene : public SLScene
{
public:
    CameraOnlyScene(std::string name);

    void updateVideoImage(const cv::Mat& image);
    void build();

    SLCamera* cameraNode = nullptr;
    void      modelTransparencyChanged(float newValue);

private:
    SLAssetManager assets;

    SLNode*      _mapNode    = nullptr;
    SLGLTexture* _videoImage = nullptr;
};

#endif // !AREA_TRACKING_SCENE_H
