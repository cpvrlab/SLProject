#ifndef APP_DEMO_GUI_ABOUT_H
#define APP_DEMO_GUI_ABOUT_H

#include <WAIModeOrbSlam2.h>
#include <AppDemoGuiInfosDialog.h>
#include <SL.h>
#include <SLGLTexture.h>
#include <SLSceneView.h>

//-----------------------------------------------------------------------------
class AppDemoGuiAbout : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiAbout(std::string name, SLGLTexture* cpvrLogo, bool* activator);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

    private:
    void centerNextWindow(SLSceneView* sv, SLfloat widthPC = 0.9f, SLfloat heightPC = 0.9f);

    SLstring     _infoAbout;
    //SLGLTexture* _cpvrLogo;
};

#endif //SL_IMGUI_TRACKEDMAPPING_H
