#ifndef LOCATON_MAP_GUI_H
#define LOCATON_MAP_GUI_H

#include <string>

#include <ImGuiWrapper.h>
#include <sm/EventSender.h>
#include <ErlebAR.h>
#include <Resources.h>

class SLScene;
class SLSceneView;
class ImFont;

class LocationMapGui : public ImGuiWrapper
  , private sm::EventSender
{
public:
    LocationMapGui(sm::EventHandler&   eventHandler,
                   ErlebAR::Resources& resources,
                   int                 dotsPerInch,
                   int                 screenWidthPix,
                   int                 screenHeightPix,
                   std::string         fontPath);
    ~LocationMapGui();

    void build(SLScene* s, SLSceneView* sv) override;
    void onResize(SLint scrW, SLint scrH) override;
    void onShow(); //call when gui becomes visible

    void initLocation(ErlebAR::Location loc);

private:
    void resize(int scrW, int scrH);
    void loadLocationMapTexture(std::string fileName);

    float _screenW;
    float _screenH;
    float _headerBarH;
    float _contentH;
    float _contentStartY;
    float _spacingBackButtonToText;
    float _buttonRounding;
    float _textWrapW;
    float _windowPaddingContent;
    float _itemSpacingContent;

    ImFont*             _fontBig = nullptr;
    ErlebAR::Resources& _resources;

    GLuint _locMapTexId = 0;

    ErlebAR::Location _loc;
    int               _locImgCropW;
    int               _locImgCropH;

    std::string _locationMapImgDir = "C:/Users/ghm1/Development/SLProject/data/erlebAR/";
};

#endif //LOCATON_MAP_GUI_H
