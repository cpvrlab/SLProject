#ifndef AREA_TRACKING_GUI_H
#define AREA_TRACKING_GUI_H

#include <string>

#include <ImGuiWrapper.h>
#include <sm/EventSender.h>
#include <ErlebAR.h>
#include <Resources.h>

class SLScene;
class SLSceneView;
struct ImFont;

class AreaTrackingGui : public ImGuiWrapper
  , private sm::EventSender
{
public:
    AreaTrackingGui(const ImGuiEngine&         imGuiEngine,
                    sm::EventHandler&          eventHandler,
                    ErlebAR::Resources&        resources,
                    int                        dotsPerInch,
                    int                        screenWidthPix,
                    int                        screenHeightPix,
                    std::function<void(float)> transparencyChangedCB);
    ~AreaTrackingGui();

    void build(SLScene* s, SLSceneView* sv) override;
    void onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY) override;
    void onShow(); //call when gui becomes visible

    void initArea(ErlebAR::Area area);

    void showLoading() { _isLoading = true; }
    void hideLoading() { _isLoading = false; }

    void showErrorMsg(const std::string& msg) { _errorMsg = msg; }

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

    float                      _sliderValue = 0.f;
    ErlebAR::Area              _area;
    std::function<void(float)> _transparencyChangedCB;

    //indicates that area information is loading
    bool        _isLoading = false;
    std::string _errorMsg;

    ErlebAR::Resources& _resources;
};

#endif //AREA_TRACKING_GUI_H
