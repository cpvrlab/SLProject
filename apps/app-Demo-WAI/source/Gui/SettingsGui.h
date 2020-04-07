#ifndef SETTINGS_GUI_H
#define SETTINGS_GUI_H

#include <string>

#include <ErlebAR.h>
#include <ImGuiWrapper.h>
#include <CVImage.h>

class SLScene;
class SLSceneView;

class SettingsGui : public ImGuiWrapper
{
public:
    SettingsGui(int         dotsPerInch,
                int         screenWidthPix,
                int         screenHeightPix,
                std::string fontPath);
    ~SettingsGui();

    void build(SLScene* s, SLSceneView* sv) override;
    void onResize(SLint scrW, SLint scrH) override;

private:
    void resize(int scrW, int scrH);
};

#endif //SETTINGS_GUI_H
