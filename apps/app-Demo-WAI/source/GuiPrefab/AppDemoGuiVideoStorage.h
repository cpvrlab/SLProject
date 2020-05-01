//#############################################################################
//  File:      AppDemoGuiVideoStorage.h
//  Author:    Luc Girod
//  Date:      April 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_VIDEOSTORAGE_H
#define SL_IMGUI_VIDEOSTORAGE_H

#include <opencv2/core.hpp>
#include <AppDemoGuiInfosDialog.h>

struct WAIEvent;
class SENSCameraInterface;

//-----------------------------------------------------------------------------
class AppDemoGuiVideoStorage : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiVideoStorage(const std::string&                        name,
                           bool*                                     activator,
                           std::queue<WAIEvent*>*                    eventQueue,
                           ImFont*                                   font,
                           std::function<SENSCameraInterface*(void)> getCameraCB);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

private:
    bool                   _recording = false;
    std::queue<WAIEvent*>* _eventQueue;

    std::function<SENSCameraInterface*(void)> _getCamera;
};

#endif //SL_IMGUI_VIDEOSTORAGE_H
