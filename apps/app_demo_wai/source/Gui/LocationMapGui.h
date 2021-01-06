#ifndef LOCATON_MAP_GUI_H
#define LOCATON_MAP_GUI_H

#include <string>

#include <ImGuiWrapper.h>
#include <sm/EventSender.h>
#include <ErlebAR.h>
#include <Resources.h>
#include <GPSMapper2D.h>
#include <sens/SENSGps.h>
#include <sens/SENSOrientation.h>

class SLScene;
class SLSceneView;
struct ImFont;

class LocationMapGui : public ImGuiWrapper
  , private sm::EventSender
{
public:
    LocationMapGui(const ImGuiEngine& imGuiEngine,
                   sm::EventHandler&  eventHandler,
                   ErlebAR::Config&   config,
                   int                dotsPerInch,
                   int                screenWidthPix,
                   int                screenHeightPix,
                   std::string        erlebARDir,
                   SENSGps*           gps,
                   SENSOrientation*   orientation);
    ~LocationMapGui();

    void build(SLScene* s, SLSceneView* sv) override;
    void onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY) override;
    void onShow(); //call when gui becomes visible
    void onHide();

    void initLocation(ErlebAR::LocationId locId);

    void onMouseDown(SLMouseButton button, SLint x, SLint y) override;
    void onMouseUp(SLMouseButton button, SLint x, SLint y) override;
    void onMouseMove(SLint xPos, SLint yPos) override;

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

    GLuint _locMapTexId = 0;

    std::string       _erlebARDir;
    ErlebAR::Location _loc;
    int               _locImgCropW = 0;
    int               _locImgCropH = 0;
    int               _locTextureW = 0;
    int               _locTextureH = 0;
    float             _x           = 0;
    float             _y           = 0;
    int               _lastPosX    = 0;
    int               _lastPosY    = 0;
    bool              _move;
    float             _fracW;
    float             _fracH;
    float             _dspPixWidth;
    float             _dspPixHeight;

    //area map positions in pixel
    std::map<ErlebAR::AreaId, SLVec2i> _areaMapPosPix;

    SENSGps*                     _gps         = nullptr;
    SENSOrientation*             _orientation = nullptr;
    std::unique_ptr<GPSMapper2D> _gpsMapper;
};

#endif //LOCATON_MAP_GUI_H
