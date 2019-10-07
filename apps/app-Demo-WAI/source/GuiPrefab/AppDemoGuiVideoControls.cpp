#include <AppDemoGuiVideoControls.h>
#include <AppWAI.h>
#include <CVCapture.h>

AppDemoGuiVideoControls::AppDemoGuiVideoControls(const std::string& name, bool* activator)
  : AppDemoGuiInfosDialog(name, activator)
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

    if (ImGui::Button((WAIApp::pauseVideo ? "Play" : "Pause"),
                      ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        WAIApp::pauseVideo = !WAIApp::pauseVideo;
    }

    ImGui::Text("Frame control");
    if (ImGui::Button("<", ImVec2(0, 0)))
    {
        WAIApp::videoCursorMoveIndex = -1;
    }
    ImGui::SameLine();
    if (ImGui::Button("<<", ImVec2(0, 0)))
    {
        WAIApp::videoCursorMoveIndex = -10;
    }
    ImGui::SameLine();
    if (ImGui::Button("<<<", ImVec2(0, 0)))
    {
        WAIApp::videoCursorMoveIndex = -100;
    }
    ImGui::SameLine();
    if (ImGui::Button(">>>", ImVec2(0, 0)))
    {
        WAIApp::videoCursorMoveIndex = 100;
    }
    ImGui::SameLine();
    if (ImGui::Button(">>", ImVec2(0, 0)))
    {
        WAIApp::videoCursorMoveIndex = 10;
    }
    ImGui::SameLine();
    if (ImGui::Button(">", ImVec2(0, 0)))
    {
        WAIApp::videoCursorMoveIndex = 1;
    }

#if 0
    if (ImGui::Button("Save current frame with candidates", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        if (WAIApp::mode != nullptr && WAIApp::mode->retainImage())
        {
            WAIFrame     frame = WAIApp::mode->getCurrentFrame();
            WAIKeyFrame* ref   = frame.mpReferenceKF;

            // TODO(dgj1): Save image from frame and its reference keyframe
        }
    }
#endif

    ImGui::End();
}
