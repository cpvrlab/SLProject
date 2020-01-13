#include <AppDemoGuiVideoControls.h>
#include <WAIApp.h>
#include <CVCapture.h>

AppDemoGuiVideoControls::AppDemoGuiVideoControls(const std::string&      name,
                                                 bool*                   activator,
                                                 std ::queue<WAIEvent*>* eventQueue)
  : AppDemoGuiInfosDialog(name, activator),
    _eventQueue(eventQueue),
    _pauseVideo(false)
{
}

void AppDemoGuiVideoControls::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::Begin("Video controls", _activator, 0);

    std::string videoFilename = Utils::getFileName(CVCapture::instance()->videoFilename);
    ImGui::Text("Current video file: %s", videoFilename.c_str());
    ImGui::Text("Current frame: %d", CVCapture::instance()->nextFrameIndex() - 1);
    ImGui::Text("Number of frames: %d", CVCapture::instance()->frameCount);
    ImGui::Text("Frames per second: %d", CVCapture::instance()->fps);

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

    ImGui::End();

    if (eventOccured)
    {
        WAIEventVideoControl* event = new WAIEventVideoControl();
        event->videoCursorMoveIndex = videoCursorMoveIndex;
        event->pauseVideo           = _pauseVideo;

        _eventQueue->push(event);
    }
}
