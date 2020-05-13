#ifndef TEST_RUNNER_GUI_H
#define TEST_RUNNER_GUI_H

#include <ImGuiWrapper.h>
#include <sm/EventSender.h>
#include <Resources.h>

class TestRunnerGui : public ImGuiWrapper
  , private sm::EventSender
{
public:
    TestRunnerGui(const ImGuiEngine&  imGuiEngine,
                  sm::EventHandler&   eventHandler,
                  ErlebAR::Resources& resources,
                  int                 dotsPerInch,
                  std::string         fontPath);

    void build(SLScene* s, SLSceneView* sv) override;

private:
    ErlebAR::Resources& _resources;
    int                 _dpi;

    std::map<int, std::string> _availableModes;
    int                        _selectedMode;
    std::string                _selectedConfigFile;
};

#endif