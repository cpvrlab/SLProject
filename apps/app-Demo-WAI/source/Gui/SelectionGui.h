#ifndef SELECTION_GUI_H
#define SELECTION_GUI_H

#include <string>
#include <map>
#include <memory>

#include <ErlebAR.h>
#include <ImGuiWrapper.h>

class SLScene;
class SLSceneView;

class SelectionGui : public ImGuiWrapper
{
public:
    SelectionGui(int dotsPerInch, std::string fontPath);

    void build(SLScene* s, SLSceneView* sv) override;

    AppMode getSelection() { return _selection; }

private:
    float   _pixPerMM;
    AppMode _selection = AppMode::NONE;
};

#endif //SELECTION_GUI_H
