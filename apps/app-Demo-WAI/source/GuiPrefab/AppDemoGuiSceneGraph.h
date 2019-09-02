#ifndef SL_IMGUI_SCENE_GRAPH_H
#define SL_IMGUI_SCENE_GRAPH_H

#include <WAIModeOrbSlam2.h>
#include <AppDemoGuiInfosDialog.h>
#include <WAICalibration.h>
#include <SLScene.h>

//-----------------------------------------------------------------------------
class AppDemoGuiSceneGraph : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiSceneGraph(std::string name, bool* activator);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

    void addSceneGraphNode(SLScene* s, SLNode* node);

};

#endif //SL_IMGUI_TRACKEDMAPPING_H
