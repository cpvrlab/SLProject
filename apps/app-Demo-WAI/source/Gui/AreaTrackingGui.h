#ifndef AREA_TRACKING_GUI_H
#define AREA_TRACKING_GUI_H

#include <string>

#include <ImGuiWrapper.h>
#include <sm/EventSender.h>
#include <ErlebAR.h>
#include <Resources.h>

class SLScene;
class SLSceneView;
class ImFont;

class AreaTrackingGui : public ImGuiWrapper
  , private sm::EventSender
{
public:
    AreaTrackingGui(sm::EventHandler&   eventHandler,
                    ErlebAR::Resources& resources,
                    int                 dotsPerInch,
                    int                 screenWidthPix,
                    int                 screenHeightPix,
                    std::string         fontPath);
    ~AreaTrackingGui();

    void build(SLScene* s, SLSceneView* sv) override;
    void onResize(SLint scrW, SLint scrH) override;
    void onShow(); //call when gui becomes visible

    void initArea(ErlebAR::Area area);

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

    ImFont*       _fontBig = nullptr;
    ErlebAR::Area _area;

    ErlebAR::Resources& _resources;
};

#endif //AREA_TRACKING_GUI_H
