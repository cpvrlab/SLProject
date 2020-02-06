//#############################################################################
//  File:      AppDemoGuiVideoStorage.cpp
//  Author:    Luc Girod
//  Date:      April 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <imgui.h>
#include <imgui_internal.h>
#include <stdio.h>

#include <WAIApp.h>
#include <Utils.h>
#include <AppDemoGuiVideoStorage.h>
#include <SLApplication.h>
#include <WAIApp.h>

//-----------------------------------------------------------------------------

AppDemoGuiVideoStorage::AppDemoGuiVideoStorage(const std::string&     name,
                                               bool*                  activator,
                                               std::queue<WAIEvent*>* eventQueue,
                                               WAIApp&                waiApp)
  : AppDemoGuiInfosDialog(name, activator),
    _eventQueue(eventQueue),
    _waiApp(waiApp)
{
}
//-----------------------------------------------------------------------------
void AppDemoGuiVideoStorage::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::Begin("Video/GPS storage", _activator, 0);
    ImGui::Separator();
    if (ImGui::Button(_recording ? "Stop recording" : "Start recording", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        WAIEventVideoRecording* event = new WAIEventVideoRecording();

        if (!_recording)
        {
            const cv::Size& size     = _waiApp.getFrameSize();
            std::string     filename = Utils::getDateTime2String() + "_" +
                                   SLApplication::getComputerInfos() + "_" +
                                   std::to_string(size.width) + "x" + std::to_string(size.height) + ".avi";

            event->filename = filename;
        }

        _eventQueue->push(event);

        _recording = !_recording;
    }

    ImGui::End();
}
