#ifndef SL_IMGUI_TRANSFORM_H
#define SL_IMGUI_TRANSFORM_H

#include <AppDemoGuiInfosDialog.h>
#include <SL.h>
#include <SLGLTexture.h>
#include <SLSceneView.h>

//-----------------------------------------------------------------------------
class AppDemoGuiTransform : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiTransform(std::string name, bool* activator, ImFont* font);

    void buildInfos(SLScene* s, SLSceneView* sv) override;
};

#endif //SL_IMGUI_TRACKEDMAPPING_H