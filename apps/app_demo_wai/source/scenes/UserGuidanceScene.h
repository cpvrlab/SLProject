#ifndef USER_GUIDANNCE_SCENE_H
#define USER_GUIDANNCE_SCENE_H

#include <SLScene.h>
#include <SLAssetManager.h>
#include <VideoBackgroundCamera.h>

class DirectionArrow : public SLNode
{
public:
    DirectionArrow(SLAssetManager& assets, std::string name, SLCamera* cameraNode);
};

class UserGuidanceScene : public SLScene
{
public:
    UserGuidanceScene(std::string dataDir);

    void updateArrowRot(SLMat3f camRarrow);

    void hideDirArrow();
    void showDirArrow();
    
    VideoBackgroundCamera* camera = nullptr;
private:
    std::string _dataDir;

    DirectionArrow* _dirArrow;

    SLAssetManager _assets;
};

#endif
