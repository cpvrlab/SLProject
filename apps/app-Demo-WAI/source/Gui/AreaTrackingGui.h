#ifndef AREA_TRACKING_GUI_H
#define AREA_TRACKING_GUI_H

#include <string>

#include <ImGuiWrapper.h>
#include <sm/EventSender.h>
#include <ErlebAR.h>
#include <Resources.h>
#include <SimHelperGui.h>

class SLScene;
class SLSceneView;
class SENSSimHelper;
struct ImFont;

/*! OpacityController to estimate UI opacity depending on user interaction.
- If the user does nothing, opacity will fade out after some time.
- If the user tabs the screen in a region where there is no ui element and opacity is non-zero, the opacity is set to zero
- If the user tabs the screen in a region where there is no ui element and opacity is zero, the opacity is set to visible
- If the user makes ui interaction (e.g. slider movement) the opacity is reset to visible
*/
class OpacityController
{
public:
    float opacity() const { return _opacity; }

    void update()
    {
        float elapsedTimeS = _timer.elapsedTimeInSec();
        //if visible time is over, start dimming
        if (elapsedTimeS > _visibleTimeS && isVisible())
        {
            _opacity = 1.f - (elapsedTimeS - _visibleTimeS) / _dimTimeS;
        }
    }

    void resetVisible()
    {
        _timer.start();
        _opacity         = 1.f;
        _manualSwitchOff = false;
    }

    void mouseDown(bool uiInteraction)
    {
        if (uiInteraction)
            resetVisible();
        else if (isVisible()) //prepare for lights off in mouse up if still visible
        {
            Utils::log("mouseDown", "isvisible");
            _manualSwitchOff = true;
        }
    }

    void mouseUp(bool uiInteraction)
    {
        Utils::log("mouseUp", "_manualSwitchOff: %s", _manualSwitchOff ? "true" : "false");
        if (_manualSwitchOff)
        {
            _manualSwitchOff = false;
            _opacity         = 0.f;
        }
        else
            resetVisible();
    }

    void mouseMove()
    {
        //check if there is significant movement
        _manualSwitchOff = false;
        resetVisible();

        //todo: (problem: when tabbing on mobile device, there may always be some movement)
    }

private:
    bool isVisible()
    {
        return _opacity > 0.0001f;
    }
    HighResTimer _timer;

    const float _visibleTimeS = 3.f;
    const float _dimTimeS     = 0.5f;
    float       _opacity      = 1.f;
    //user tapped to switch off visibility
    bool _manualSwitchOff = false;
};

class AreaTrackingGui : public ImGuiWrapper
  , private sm::EventSender
{
public:
    AreaTrackingGui(const ImGuiEngine&                  imGuiEngine,
                    sm::EventHandler&                   eventHandler,
                    ErlebAR::Config&                    config,
                    int                                 dotsPerInch,
                    int                                 screenWidthPix,
                    int                                 screenHeightPix,
                    std::function<void(float)>          transparencyChangedCB,
                    std::string                         erlebARDir,
                    std::function<SENSSimHelper*(void)> getSimHelperCB);
    ~AreaTrackingGui();

    void build(SLScene* s, SLSceneView* sv) override;
    void onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY) override;
    void onShow(); //call when gui becomes visible

    void initArea(const ErlebAR::Area& area);

    void showInfoText(const std::string& str);
    void showImageAlignTexture(float alpha);
    void showLoading() { _isLoading = true; }
    void hideLoading() { _isLoading = false; }

    void showErrorMsg(const std::string& msg) { _errorMsg = msg; }

    void  mouseDown(SLMouseButton button, bool doNotDispatch);
    void  mouseMove(bool doNotDispatch);
    void  mouseUp(SLMouseButton button, bool doNotDispatch);
    float opacity() const { return _opacityController.opacity(); }

private:
    void resize(int scrW, int scrH);
    void renderSimInfos(SENSSimHelper* simHelper, ImFont* fontText, ImFont* fontHeading, const char* title);

    float _screenW;
    float _screenH;
    float _headerBarH;
    float _contentH;
    float _contentStartY;
    float _spacingBackButtonToText;
    float _buttonRounding;
    float _textWrapW;
    //float _windowPaddingContent;
    //float _itemSpacingContent;
    std::string                         _erlebARDir;
    float                               _sliderValue = 0.f;
    ErlebAR::Area                       _area;
    std::function<void(float)>          _transparencyChangedCB;
    std::function<SENSSimHelper*(void)> _getSimHelper;
    //indicates that area information is loading
    bool        _isLoading = false;
    std::string _errorMsg;

    OpacityController _opacityController;

    ErlebAR::Config&    _config;
    ErlebAR::Resources& _resources;

    std::string _infoText;
    GLuint      _areaAlignTexture         = 0;
    float       _areaAlighTextureBlending = 1.0f;
    bool        _showAlignImage           = false;

    std::unique_ptr<SimHelperGui> _simHelperGui;
};

#endif //AREA_TRACKING_GUI_H
