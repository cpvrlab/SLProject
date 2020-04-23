#ifndef AREA_INFO_GUI_H
#define AREA_INFO_GUI_H

#include <string>

#include <ImGuiWrapper.h>
#include <sm/EventSender.h>
#include <ErlebAR.h>
#include <Resources.h>

class SLScene;
class SLSceneView;
class ImFont;

class AreaInfoGui : public ImGuiWrapper
  , private sm::EventSender
{
public:
    AreaInfoGui(sm::EventHandler&   eventHandler,
                ErlebAR::Resources& resources,
                int                 dotsPerInch,
                int                 screenWidthPix,
                int                 screenHeightPix,
                std::string         fontPath);
    ~AreaInfoGui();

    void build(SLScene* s, SLSceneView* sv) override;
    void onResize(SLint scrW, SLint scrH) override;
    void onShow(); //call when gui becomes visible

    void initArea(ErlebAR::LocationId locId, ErlebAR::AreaId areaId);

private:
    void resize(int scrW, int scrH);

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
    float _buttonBoardH;

    ImFont* _fontBig      = nullptr;
    ImFont* _fontSmall    = nullptr;
    ImFont* _fontStandard = nullptr;

    ErlebAR::Resources& _resources;
    ErlebAR::Area       _area;
};

#endif //AREA_INFO_GUI_H
