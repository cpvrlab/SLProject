#ifndef TEST_RUNNER_GUI_H
#define TEST_RUNNER_GUI_H

#include <ImGuiWrapper.h>
#include <sm/EventSender.h>

class TestRunnerGui : public ImGuiWrapper
  , private sm::EventSender
{
public:
    TestRunnerGui(sm::EventHandler& eventHandler,
                  int               dotsPerInch,
                  std::string       fontPath);

    void build(SLScene* s, SLSceneView* sv) override;

private:
    ImFont* _fontPropDots = nullptr;
    int     _dpi;
};

#endif