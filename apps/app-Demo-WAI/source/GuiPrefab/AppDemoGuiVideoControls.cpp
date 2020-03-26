#include <AppDemoGuiVideoControls.h>
#include <WAIEvent.h>
#include <SENSVideoStream.h>

AppDemoGuiVideoControls::AppDemoGuiVideoControls(const std::string&                    name,
                                                 bool*                                 activator,
                                                 std ::queue<WAIEvent*>*               eventQueue,
                                                 ImFont*                               font,
                                                 std::function<SENSVideoStream*(void)> getVideoFileStreamCB)
  : AppDemoGuiInfosDialog(name, activator, font),
    _eventQueue(eventQueue),
    _pauseVideo(false),
    _getVideoFileStream(getVideoFileStreamCB)
{
}

void AppDemoGuiVideoControls::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::PushFont(_font);
    ImGui::Begin("Video controls", _activator, 0);
    const SENSVideoStream* vs = _getVideoFileStream();
    if (vs)
    {

        std::string videoFilename = Utils::getFileName(vs->videoFilename());
        ImGui::Text("Current video file: %s", videoFilename.c_str());
        ImGui::Text("Current frame: %d", vs->nextFrameIndex() - 1);
        ImGui::Text("Number of frames: %d", vs->frameCount());
        ImGui::Text("Frames per second: %f", vs->fps());

        ImGui::Separator();

        bool eventOccured         = false;
        int  videoCursorMoveIndex = 0;

        if (ImGui::Button((_pauseVideo ? "Play" : "Pause"),
                          ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
        {
            _pauseVideo = !_pauseVideo;

            eventOccured = true;
        }

        ImGui::Text("Frame control");
        if (ImGui::Button("<", ImVec2(0, 0)))
        {
            videoCursorMoveIndex += -1;

            eventOccured = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("<<", ImVec2(0, 0)))
        {
            videoCursorMoveIndex += -10;

            eventOccured = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("<<<", ImVec2(0, 0)))
        {
            videoCursorMoveIndex += -100;

            eventOccured = true;
        }
        ImGui::SameLine();
        if (ImGui::Button(">>>", ImVec2(0, 0)))
        {
            videoCursorMoveIndex += 100;

            eventOccured = true;
        }
        ImGui::SameLine();
        if (ImGui::Button(">>", ImVec2(0, 0)))
        {
            videoCursorMoveIndex += 10;

            eventOccured = true;
        }
        ImGui::SameLine();
        if (ImGui::Button(">", ImVec2(0, 0)))
        {
            videoCursorMoveIndex += 1;

            eventOccured = true;
        }

        if (eventOccured)
        {
            WAIEventVideoControl* event = new WAIEventVideoControl();
            event->videoCursorMoveIndex = videoCursorMoveIndex;
            event->pauseVideo           = _pauseVideo;

            _eventQueue->push(event);
        }
    }
    else
    {
        ImGui::Text("No video running");
    }

    ImGui::End();
    ImGui::PopFont();
}
