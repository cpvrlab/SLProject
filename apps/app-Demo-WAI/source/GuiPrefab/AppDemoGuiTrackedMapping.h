#ifndef APP_DEMO_GUI_TRACKED_MAPPING
#define APP_DEMO_GUI_TRACKED_MAPPING

#include <WAIModeOrbSlam2.h>

#include <AppDemoGuiInfosDialog.h>

//-----------------------------------------------------------------------------
class AppDemoGuiTrackedMapping : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiTrackedMapping(std::string name, bool* activator);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

    private:
    //!currently selected combobox item
    static const char* _currItem;
    //!currently selected combobox index
    static int _currN;
};

#endif //SL_IMGUI_TRACKEDMAPPING_H
