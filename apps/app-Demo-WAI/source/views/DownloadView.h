#ifndef DOWNLOAD_VIEW_H
#define DOWNLOAD_VIEW_H

#include <string>
#include <SLInputManager.h>
#include <SLSceneView.h>
#include <DownloadGui.h>
#include <ErlebAR.h>
#include <AsyncWorker.h>

class DownloadView : public SLSceneView
{
public:
    DownloadView(sm::EventHandler&                    eventHandler,
                 SLInputManager&                      inputManager,
                 const ImGuiEngine&                   imGuiEngine,
                 ErlebAR::Config&                     config,
                 std::map<std::string, AsyncWorker*>& asyncWorkers,
                 const DeviceData&                    deviceData);
    bool update();
    //call when view becomes visible
    void show() { _gui.onShow(); }
    void initLocation(ErlebAR::LocationId locId);

private:
    DownloadGui _gui;
};

#endif //ABOUT_VIEW_H
