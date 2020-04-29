#include <TestRunnerGui.h>
#include <ErlebAREvents.h>
#include <views/TestRunnerView.h>

TestRunnerGui::TestRunnerGui(sm::EventHandler& eventHandler)
  : sm::EventSender(eventHandler)
{
}

void TestRunnerGui::build(SLScene* s, SLSceneView* sv)
{
    TestRunnerView* view = (TestRunnerView*)sv;

    ImGui::Text("Video: %s", view->videoName().c_str());
    ImGui::Text("Frame: %i / %i", view->currentFrameIndex(), view->frameIndex());

    if (ImGui::Button("Begin test"))
    {
        view->start(TestRunnerView::TestMode_Relocalization,
                    ExtractorType_GLSL);
    }

    if (ImGui::Button("Back"))
    {
        sendEvent(new GoBackEvent());
    }
}
