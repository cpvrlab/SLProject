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

class OpacityController
{
public:
    float opacity() const { return _opacity; }

    void update()
    {
        float elapsedTimeS = _timer.elapsedTimeInSec();
        //if visible time is over, start dimming
        if (elapsedTimeS > _visibleTimeS &&
            _opacity > 0.0001f)
        {
            _opacity = 1.f - (elapsedTimeS - _visibleTimeS) / _dimTimeS;
        }
    }

    void reset()
    {
        _timer.start();
        _opacity         = 1.f;
        _manualSwitchOff = false;
    }

    void mouseDown()
    {
        if (_timer.elapsedTimeInSec() < _visibleTimeS && !_manualSwitchOff)
        {
            _manualSwitchOff = true;
            _opacity         = 0.f;
        }
        else
            reset();
    }

private:
    HighResTimer _timer;

    const float _visibleTimeS = 5.f;
    const float _dimTimeS     = 1.f;
    float       _opacity      = 1.f;
    //user tapped to switch off visibility
    bool _manualSwitchOff = false;
};

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

    void mouseDown(bool doNotDispatch);
    void mouseMove();

private:
    void resize(int scrW, int scrH);
    //void onMouseDown(SLMouseButton button, SLint x, SLint y) override;
    //void onMouseMove(SLint xPos, SLint yPos) override;

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

    OpacityController _opacityController;

    ErlebAR::Resources& _resources;
};

#endif //AREA_TRACKING_GUI_H
