#ifndef SL_IMGUI_ERROR_H
#define SL_IMGUI_ERROR_H

#include <AppDemoGuiInfosDialog.h>

//-----------------------------------------------------------------------------
class AppDemoGuiError : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiError(string name, bool* activator);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

    void setErrorMsg(std::string msg) { _errorMsg = msg; }

    private:
    std::string _errorMsg = "";
};

#endif
