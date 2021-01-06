//#############################################################################
//  File:      AppDemoGuiVideoStorage.cpp
//  Author:    Luc Girod
//  Date:      April 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <AppDemoGuiVideoStorage.h>
#include <stdio.h>
#include <Utils.h>
#include <WAIEvent.h>
#include <sens/SENSCvCamera.h>

//-----------------------------------------------------------------------------

AppDemoGuiVideoStorage::AppDemoGuiVideoStorage(const std::string&                 name,
                                               bool*                              activator,
                                               std::queue<WAIEvent*>*             eventQueue,
                                               ImFont*                            font,
                                               std::function<SENSCvCamera*(void)> getCameraCB)
  : AppDemoGuiInfosDialog(name, activator, font),
    _eventQueue(eventQueue),
    _getCamera(getCameraCB)
{
}
//-----------------------------------------------------------------------------
void AppDemoGuiVideoStorage::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::PushFont(_font);
    ImGui::Begin("Video/GPS storage", _activator, 0);
    ImGui::Separator();
    if (ImGui::Button(_recording ? "Stop recording" : "Start recording", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        WAIEventVideoRecording* event = new WAIEventVideoRecording();

        if (!_recording)
        {
            SENSCvCamera* cam = _getCamera();
            if (cam && cam->isConfigured())
            {
                cv::Size    size(cam->config()->targetWidth, cam->config()->targetHeight);
                std::string filename = Utils::getDateTime2String() + "_" +
                                       Utils::ComputerInfos::get() + "_" +
                                       std::to_string(size.width) + "x" + std::to_string(size.height) + ".avi";

                event->filename = filename;
            }
        }

        _eventQueue->push(event);

        _recording = !_recording;
    }

    ImGui::End();
    ImGui::PopFont();
}
