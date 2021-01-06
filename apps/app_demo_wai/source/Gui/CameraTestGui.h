#ifndef CAMERA_TEST_GUI_H
#define CAMERA_TEST_GUI_H

#include <string>

#include <ImGuiWrapper.h>
#include <sm/EventSender.h>
#include <ErlebAR.h>
#include <Resources.h>
#include <sens/SENSCamera.h>

class SLScene;
class SLSceneView;
struct ImFont;

class CameraTestGui : public ImGuiWrapper
  , private sm::EventSender
{
public:
    CameraTestGui(const ImGuiEngine& imGuiEngine,
                  sm::EventHandler&  eventHandler,
                  ErlebAR::Config&   config,
                  int                dotsPerInch,
                  int                screenWidthPix,
                  int                screenHeightPix,
                  SENSCamera*        camera);
    ~CameraTestGui();

    void build(SLScene* s, SLSceneView* sv) override;
    void onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY) override;
    void onShow(); //call when gui becomes visible

    SENSCamera* camera() { return _camera; }

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

    ErlebAR::Config&    _config;
    ErlebAR::Resources& _resources;

    SENSCamera* _camera;

    SENSCaptureProperties _camCharacs;

    std::map<std::string, std::vector<std::string>> _sizesStrings;
    //callbacks
    //std::function<void(void)> _startCameraCB;
    //std::function<void(void)> _stopCameraCB;

    bool        _hasException = false;
    std::string _exceptionText;

    //selection values
    const SENSCameraDeviceProperties* _currCamProps{nullptr};
    int                               _currSizeIndex{0};
    const std::string*                _currSizeStr{nullptr};
};

#endif //CAMERA_TEST_GUI_H
