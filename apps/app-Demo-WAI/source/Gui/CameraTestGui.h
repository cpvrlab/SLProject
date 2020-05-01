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
class ImFont;

class CameraTestGui : public ImGuiWrapper
  , private sm::EventSender
{
public:
    CameraTestGui(sm::EventHandler&   eventHandler,
                  ErlebAR::Resources& resources,
                  int                 dotsPerInch,
                  int                 screenWidthPix,
                  int                 screenHeightPix,
                  std::string         fontPath,
                  SENSCamera*         camera);
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

    ImFont* _fontBig = nullptr;

    ErlebAR::Resources& _resources;

    SENSCamera*                                     _camera;
    SENSCamera::Config                              _cameraConfig;

    std::vector<SENSCameraCharacteristics>          _camCharacs;
    std::map<std::string, std::vector<std::string>> _sizesStrings;
    //callbacks
    //std::function<void(void)> _startCameraCB;
    //std::function<void(void)> _stopCameraCB;

    bool        _hasException = false;
    std::string _exceptionText;
};

#endif //CAMERA_TEST_GUI_H
