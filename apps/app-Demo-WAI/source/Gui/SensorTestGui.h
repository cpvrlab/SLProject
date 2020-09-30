#ifndef SENSOR_TEST_GUI_H
#define SENSOR_TEST_GUI_H

#include <string>

#include <ImGuiWrapper.h>
#include <sm/EventSender.h>
#include <ErlebAR.h>
#include <Resources.h>
#include <sens/SENSGps.h>
#include <sens/SENSOrientation.h>
#include <sens/SENSRecorder.h>
#include <sens/SENSCamera.h>

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
                  const DeviceData&   deviceData,
                  SENSGps*            gps,
                  SENSOrientation*    orientation);
    ~SensorTestGui();

    void build(SLScene* s, SLSceneView* sv) override;
    void onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY) override;
    void onShow(); //call when gui becomes visible
    void onHide();

private:
    void resize(int scrW, int scrH);
    void updateGpsSensor();
    void updateOrientationSensor();
    void updateSensorRecording();

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
    SENSCamera*      _camera      = nullptr;
    //std::unique_ptr<SENSOrientationRecorder> _orientationRecorder;
    std::unique_ptr<SENSRecorder> _sensorRecorder;

    bool _recordGps         = false;
    bool _recordOrientation = false;
    bool _recordCamera      = false;
    
    std::string recordButtonText = "Start recording";
};

#endif //SENSOR_TEST_GUI_H
