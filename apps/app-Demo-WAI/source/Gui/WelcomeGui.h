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

private:
    void pushStyle();
    void popStyle();

    const float _pixPerMM;
    ImFont*     _fontBig   = nullptr;
    ImFont*     _fontSmall = nullptr;
    const int   _screenWidthPix;
    const int   _screenHeightPix;

    const std::string _versionStr;

    GLuint       _logoBFHTexId;
    unsigned int _logoBFHWidth;
    unsigned int _logoBFHHeight;

    GLuint       _logoAdminCHTexId;
    unsigned int _logoAdminCHWidth;
    unsigned int _logoAdminCHHeight;

    int   _fontHeightBigDots;
    float _smallFontShift;
};

#endif //WELCOME_GUI_H
