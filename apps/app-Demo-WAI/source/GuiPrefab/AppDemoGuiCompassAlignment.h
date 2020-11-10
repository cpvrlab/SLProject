#ifndef APP_DEMO_GUI_COMPASS_ALIGNMENT_H
#define APP_DEMO_GUI_COMPASS_ALIGNMENT_H

#include <AppDemoGuiInfosDialog.h>

struct WAIEvent;

class AppDemoGuiCompassAlignment : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiCompassAlignment(const std::string&               name,
                               std ::queue<WAIEvent*>*          eventQueue,
                               ImFont*                          font,
                               bool*                            activator,
                               std::function<void(std::string)> errorMsgCB);
    void buildInfos(SLScene* s, SLSceneView* sv) override;

private:
    std::queue<WAIEvent*>*           _eventQueue;
    std::function<void(std::string)> _errorMsgCB = nullptr;
};

#endif