#ifndef SL_IMGUI_INFO_SCENE_H
#define SL_IMGUI_INFO_SCENE_H

#include <WAIModeOrbSlam2.h>
#include <AppDemoGuiInfosDialog.h>
#include <SL.h>
#include <SLSceneView.h>
#include <SLScene.h>

//-----------------------------------------------------------------------------
class AppDemoGuiInfosScene : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiInfosScene(string name, bool* activator);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

    private:
};

#endif //SL_IMGUI_TRACKEDMAPPING_H
