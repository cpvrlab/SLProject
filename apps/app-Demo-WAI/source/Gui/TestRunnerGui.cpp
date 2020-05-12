#include <TestRunnerGui.h>
#include <ErlebAREvents.h>
#include <imgui.h>
#include <views/TestRunnerView.h>

TestRunnerGui::TestRunnerGui(sm::EventHandler& eventHandler,
                             int               dotsPerInch,
                             std::string       fontPath)
  : sm::EventSender(eventHandler),
    _dpi(dotsPerInch),
    _selectedMode(TestRunnerView::TestMode_None)
{
    ImGuiIO& io = ImGui::GetIO();

    // Load proportional font for menue and text displays
    SLstring DroidSans = fontPath + "DroidSans.ttf";
    if (Utils::fileExists(DroidSans))
    {
        _fontPropDots = io.Fonts->AddFontFromFileTTF(DroidSans.c_str(), std::max(16.0f * (dotsPerInch / 120.0f), 16.0f));
    }

    _availableModes[TestRunnerView::TestMode_None]           = "None";
    _availableModes[TestRunnerView::TestMode_Relocalization] = "Relocalization";
    _availableModes[TestRunnerView::TestMode_Tracking]       = "Tracking";
}

void TestRunnerGui::build(SLScene* s, SLSceneView* sv)
{
    ImGui::PushFont(_fontPropDots);

    TestRunnerView* view = (TestRunnerView*)sv;

    ImGui::Begin("Testrunner", 0, ImGuiWindowFlags_AlwaysAutoResize);

    if (view->testsRunning())
    {
        ImGui::Text("Mode: %s", _availableModes[_selectedMode].c_str());
        ImGui::Text("Test: %i / %i", view->testIndex(), view->testCount());
        ImGui::Text("Location: %s", view->location().c_str());
        ImGui::Text("Area: %s", view->area().c_str());
        ImGui::Text("Video: %s", view->video().c_str());
        ImGui::Text("Frame: %i / %i", view->currentFrameIndex(), view->frameIndex());
    }
    else
    {
        if (ImGui::BeginCombo("Mode", _availableModes[_selectedMode].c_str()))
        {
            for (auto modeIt = _availableModes.begin();
                 modeIt != _availableModes.end();
                 modeIt++)
            {
                if (ImGui::Selectable(modeIt->second.c_str(), _selectedMode == modeIt->first))
                {
                    _selectedMode = modeIt->first;
                }
            }

            ImGui::EndCombo();
        }

        if (ImGui::Button("Begin test"))
        {
            if (_selectedMode)
            {
                view->start((TestRunnerView::TestMode)_selectedMode);
            }
        }
    }

    if (ImGui::Button("Back"))
    {
        sendEvent(new GoBackEvent());
    }

    ImGui::End();

    ImGui::PopFont();
}
