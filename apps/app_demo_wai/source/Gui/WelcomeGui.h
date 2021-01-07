#ifndef WELCOME_GUI_H
#define WELCOME_GUI_H

#include <string>
#include <ImGuiWrapper.h>
#include <ErlebAR.h>
#include <Resources.h>

class SLScene;
class SLSceneView;

class WelcomeGui : public ImGuiWrapper
{
public:
    WelcomeGui(const ImGuiEngine&  imGuiEngine,
               ErlebAR::Resources& resources,
               int                 dotsPerInch,
               int                 screenWidthPix,
               int                 screenHeightPix,
               std::string         fontPath,
               std::string         texturePath,
               std::string         version);
    ~WelcomeGui();

    void build(SLScene* s, SLSceneView* sv) override;
    void onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY) override;

private:
    void resize(int scrW, int scrH);

    void pushStyle();
    void popStyle();

    const std::string _versionStr;

    GLuint       _launchImgTexId = 0;
    unsigned int _textureLaunchImgW  = 0;
    unsigned int _textureLaunchImgH  = 0;
    const float  _scaleToSmallerLen  = 0.4f;

    float _screenWPix;
    float _screenHPix;

    ErlebAR::Resources& _resources;
};

#endif //WELCOME_GUI_H
