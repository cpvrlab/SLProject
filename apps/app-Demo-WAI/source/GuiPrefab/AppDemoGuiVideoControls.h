#ifndef SL_IMGUI_VIDEOCONTROLS_H
#define SL_IMGUI_VIDEOCONTROLS_H

#include <AppDemoGuiInfosDialog.h>

class WAIEvent;
class SENSVideoStream;

class AppDemoGuiVideoControls : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiVideoControls(const std::string&                    name,
                            bool*                                 activator,
                            std::queue<WAIEvent*>*                eventQueue,
                            ImFont*                               font,
                            std::function<SENSVideoStream*(void)> getVideoFileStreamCB);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

private:
    bool                   _pauseVideo;
    std::queue<WAIEvent*>* _eventQueue;

    std::function<SENSVideoStream*(void)> _getVideoFileStream;
};

#endif
