#ifndef SENSOR_TEST_GUI_H
#define SENSOR_TEST_GUI_H

#include <string>

#include <ImGuiWrapper.h>
#include <sm/EventSender.h>
#include <ErlebAR.h>
#include <Resources.h>
#include <SENSGps.h>
#include <SENSOrientation.h>
#include <SENSRecorder.h>
#include <SENSCamera.h>
#include <SENSSimulator.h>

#include <SENSSimHelper.h>
#include <Gui/SimHelperGui.h>

class SLScene;
class SLSceneView;
struct ImFont;

class SensorTestGui : public ImGuiWrapper
  , private sm::EventSender
{
public:
    SensorTestGui(const ImGuiEngine& imGuiEngine,
                  sm::EventHandler&  eventHandler,
                  ErlebAR::Config&   config,
                  const DeviceData&  deviceData,
                  SENSGps*           gps,
                  SENSOrientation*   orientation,
                  SENSCamera*        camera);
    ~SensorTestGui();

    void build(SLScene* s, SLSceneView* sv) override;
    void onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY) override;
    void onShow(); //call when gui becomes visible
    void onHide();

private:
    void resize(int scrW, int scrH);

    void updateGpsSensor();
    void updateOrientationSensor();
    void updateCameraSensor();

    void updateCameraParameter();

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

    ErlebAR::Config&    _config;
    ErlebAR::Resources& _resources;

    bool        _hasException = false;
    std::string _exceptionText;

    //active sensors
    SENSGps*         _gps         = nullptr;
    SENSOrientation* _orientation = nullptr;
    SENSCamera*      _camera      = nullptr;

    std::unique_ptr<SENSSimHelper> _simHelper;
    std::unique_ptr<SimHelperGui>  _simHelperGui;

    //camera stuff:
    SENSCaptureProperties                           _camCharacs;
    std::map<std::string, std::vector<std::string>> _sizesStrings;
    //selection values
    const SENSCameraDeviceProperties* _currCamProps{nullptr};
    int                               _currSizeIndex{0};
    const std::string*                _currSizeStr{nullptr};

    std::string recordButtonText = "Start recording";

    GLuint   _videoTextureId = 0;
    cv::Size _videoTextureSize;

    std::string _simDataDir;
    ImVec2      _imgViewSize = {640, 480};
    cv::Mat     _currentImgRGB;
};

#endif //SENSOR_TEST_GUI_H
