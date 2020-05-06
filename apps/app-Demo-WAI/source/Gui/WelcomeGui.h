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

    GLuint       _logoBFHTexId = 0;
    unsigned int _textureBFHW  = 0;
    unsigned int _textureBFHH  = 0;

    GLuint       _logoAdminCHTexId = 0;
    unsigned int _textureAdminCHW  = 0;
    unsigned int _textureAdminCHH  = 0;

    float _screenWPix;
    float _screenHPix;
    float _smallFontShift;

    float _textFrameTPix;
    float _textFrameLRPix;
    float _bigTextHPix;
    float _smallTextHPix;

    float _bfhLogoHPix;
    float _bfhLogoWPix;
    float _adminLogoHPix;
    float _adminLogoWPix;
    float _logoFrameBPix;
    float _logoFrameLRPix;

    //ImFont* _fontBig   = nullptr;
    //ImFont* _fontSmall = nullptr;

    ErlebAR::Resources& _resources;
};

#endif //WELCOME_GUI_H
