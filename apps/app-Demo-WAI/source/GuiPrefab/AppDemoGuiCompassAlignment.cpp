#include <AppDemoGuiCompassAlignment.h>
#include <WAIEvent.h>

AppDemoGuiCompassAlignment::AppDemoGuiCompassAlignment(const std::string&               name,
                                                       std ::queue<WAIEvent*>*          eventQueue,
                                                       ImFont*                          font,
                                                       bool*                            activator,
                                                       std::function<void(std::string)> errorMsgCB)
  : AppDemoGuiInfosDialog(name, activator, font),
    _eventQueue(eventQueue),
    _errorMsgCB(errorMsgCB)
{
}

void AppDemoGuiCompassAlignment::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::PushFont(_font);
    ImGui::Begin("Compass Alignment", _activator, ImGuiWindowFlags_AlwaysAutoResize);

    if (ImGui::Button("Start", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        _eventQueue->push(new WAIEventStartCompassAlignment());
    }

    ImGui::End();
    ImGui::PopFont();
}
