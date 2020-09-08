#ifndef SENSOR_TEST_GUI_H
#define SENSOR_TEST_GUI_H

#include <string>

#include <ImGuiWrapper.h>
#include <sm/EventSender.h>
#include <ErlebAR.h>
#include <Resources.h>
#include <sens/SENSGps.h>
#include <sens/SENSOrientation.h>

class SLScene;
class SLSceneView;
struct ImFont;

class SensorTestGui : public ImGuiWrapper
  , private sm::EventSender
{
public:
    SensorTestGui(const ImGuiEngine&  imGuiEngine,
                  sm::EventHandler&   eventHandler,
                  ErlebAR::Resources& resources,
                  int                 dotsPerInch,
                  int                 screenWidthPix,
                  int                 screenHeightPix,
                  SENSGps*            gps,
                  SENSOrientation*    orientation);
    ~SensorTestGui();

    void build(SLScene* s, SLSceneView* sv) override;
    void onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY) override;
    void onShow(); //call when gui becomes visible

private:
    void resize(int scrW, int scrH);
    void updateGpsSensor();
    void updateOrientationSensor();

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

    ErlebAR::Resources& _resources;

    bool        _hasException = false;
    std::string _exceptionText;

    SENSGps*         _gps         = nullptr;
    SENSOrientation* _orientation = nullptr;
};

#endif //SENSOR_TEST_GUI_H
