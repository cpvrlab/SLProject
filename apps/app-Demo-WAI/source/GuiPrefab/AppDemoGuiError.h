#ifndef SL_IMGUI_ERROR_H
#define SL_IMGUI_ERROR_H

#include <AppDemoGuiInfosDialog.h>

//-----------------------------------------------------------------------------
class AppDemoGuiError : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiError(string name, bool* activator, ImFont* font);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

    void setErrorMsg(std::string msg) { _errorMsg = msg; }

private:
    ImFont*     _font     = nullptr;
    std::string _errorMsg = "";
};

#endif
