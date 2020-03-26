#ifndef APP_DEMO_GUI_TRACKED_MAPPING
#define APP_DEMO_GUI_TRACKED_MAPPING

#include <AppDemoGuiInfosDialog.h>

class WAIApp;
class WAISlam;

//-----------------------------------------------------------------------------
class AppDemoGuiTrackedMapping : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiTrackedMapping(std::string                   name,
                             bool*                         activator,
                             ImFont*                       font,
                             std::function<WAISlam*(void)> getModeCB);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

private:
    std::function<WAISlam*(void)> _getMode;

    //!currently selected combobox item
    static const char* _currItem;
    //!currently selected combobox index
    static int _currN;
};

#endif //SL_IMGUI_TRACKEDMAPPING_H
