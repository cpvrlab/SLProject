#ifndef SL_IMGUI_SCENE_GRAPH_H
#define SL_IMGUI_SCENE_GRAPH_H

#include <AppDemoGuiInfosDialog.h>
#include <WAICalibration.h>
#include <SLScene.h>

//-----------------------------------------------------------------------------
class AppDemoGuiSceneGraph : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiSceneGraph(std::string name, bool* activator, ImFont* font);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

    void addSceneGraphNode(SLScene* s, SLNode* node);
};

#endif //SL_IMGUI_TRACKEDMAPPING_H
