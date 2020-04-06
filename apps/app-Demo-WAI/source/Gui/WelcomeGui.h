#ifndef WELCOME_GUI_H
#define WELCOME_GUI_H

#include <string>

#include <ErlebAR.h>
#include <ImGuiWrapper.h>
#include <CVImage.h>

class SLScene;
class SLSceneView;

class WelcomeGui : public ImGuiWrapper
{
public:
    WelcomeGui(int         dotsPerInch,
               int         screenWidthPix,
               int         screenHeightPix,
               std::string fontPath,
               std::string version);

    void build(SLScene* s, SLSceneView* sv) override;
    //void onResize(SLint scrW, SLint scrH) override;

private:
    void pushStyle();
    void popStyle();

    const float _pixPerMM;
    ImFont*     _fontBig   = nullptr;
    ImFont*     _fontSmall = nullptr;
    const float _screenWPix;
    const float _screenHPix;

    const std::string _versionStr;

    GLuint       _logoBFHTexId;
    unsigned int _textureBFHW;
    unsigned int _textureBFHH;

    GLuint       _logoAdminCHTexId;
    unsigned int _textureAdminCHW;
    unsigned int _textureAdminCHH;

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
};

#endif //WELCOME_GUI_H
