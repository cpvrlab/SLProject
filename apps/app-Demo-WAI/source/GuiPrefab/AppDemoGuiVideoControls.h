#ifndef SL_IMGUI_VIDEOCONTROLS_H
#define SL_IMGUI_VIDEOCONTROLS_H

#include <AppDemoGuiInfosDialog.h>
#include <WAIApp.h>

class WAIApp;

class AppDemoGuiVideoControls : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiVideoControls(const std::string& name, bool* activator, std::queue<WAIEvent*>* eventQueue, WAIApp& waiApp);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

private:
    bool                   _pauseVideo;
    std::queue<WAIEvent*>* _eventQueue;
    const WAIApp&          _waiApp;
};

#endif
