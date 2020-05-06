#include <TestRunnerGui.h>
#include <ErlebAREvents.h>
#include <views/TestRunnerView.h>
#include <imgui.h>

TestRunnerGui::TestRunnerGui(sm::EventHandler& eventHandler,
                             int               dotsPerInch,
                             std::string       fontPath)
  : sm::EventSender(eventHandler),
    _dpi(dotsPerInch)
{
    ImGuiIO& io = ImGui::GetIO();

    // Load proportional font for menue and text displays
    SLstring DroidSans = fontPath + "DroidSans.ttf";
    if (Utils::fileExists(DroidSans))
    {
        _fontPropDots = io.Fonts->AddFontFromFileTTF(DroidSans.c_str(), std::max(16.0f * (dotsPerInch / 120.0f), 16.0f));
    }
}

void TestRunnerGui::build(SLScene* s, SLSceneView* sv)
{
    ImGui::PushFont(_fontPropDots);

    TestRunnerView* view = (TestRunnerView*)sv;

    ImGui::Begin("Slam Load", 0, ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::Text("Video: %s", view->videoName().c_str());
    ImGui::Text("Frame: %i / %i", view->currentFrameIndex(), view->frameIndex());

    if (ImGui::Button("Begin test"))
    {
        view->start(TestRunnerView::TestMode_Tracking,
                    ExtractorType_GLSL);
    }

    if (ImGui::Button("Back"))
    {
        sendEvent(new GoBackEvent());
    }

    ImGui::End();

    ImGui::PopFont();
}
