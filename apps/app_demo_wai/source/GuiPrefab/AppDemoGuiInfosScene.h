#ifndef SL_IMGUI_INFO_SCENE_H
#define SL_IMGUI_INFO_SCENE_H

#include <AppDemoGuiInfosDialog.h>
#include <SL.h>
#include <SLSceneView.h>
#include <SLScene.h>

//-----------------------------------------------------------------------------
class AppDemoGuiInfosScene : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiInfosScene(string name, bool* activator, ImFont* font);

    void buildInfos(SLScene* s, SLSceneView* sv) override;
};

#endif //SL_IMGUI_TRACKEDMAPPING_H
