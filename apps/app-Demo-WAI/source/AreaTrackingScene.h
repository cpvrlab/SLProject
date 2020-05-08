#ifndef AREA_TRACKING_SCENE_H
#define AREA_TRACKING_SCENE_H

#include <SLScene.h>
#include <SLAssetManager.h>

class AreaTrackingScene : public SLScene
{
public:
    AreaTrackingScene();

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
