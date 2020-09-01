#include <LocationMapGui.h>
#include <imgui_internal.h>
#include <GuiUtils.h>
#include <ErlebAREvents.h>

using namespace ErlebAR;

LocationMapGui::LocationMapGui(const ImGuiEngine&  imGuiEngine,
                               sm::EventHandler&   eventHandler,
                               ErlebAR::Resources& resources,
                               int                 dotsPerInch,
                               int                 screenWidthPix,
                               int                 screenHeightPix,
                               std::string         erlebARDir)
  : ImGuiWrapper(imGuiEngine.context(), imGuiEngine.renderer()),
    sm::EventSender(eventHandler),
    _resources(resources),
    _erlebARDir(erlebARDir)
{
    resize(screenWidthPix, screenHeightPix);
}

LocationMapGui::~LocationMapGui()
{
}

void LocationMapGui::onShow()
{
    _panScroll.enable();
}

void LocationMapGui::onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY)
{
    resize(scrW, scrH);
    ImGuiWrapper::onResize(scrW, scrH, scr2fbX, scr2fbY);
}

void LocationMapGui::resize(int scrW, int scrH)
{
    _screenW = (float)scrW;
    _screenH = (float)scrH;

    _headerBarH              = _resources.style().headerBarPercH * _screenH;
    _contentH                = _screenH - _headerBarH;
    _contentStartY           = _headerBarH;
    _spacingBackButtonToText = _resources.style().headerBarSpacingBB2Text * _headerBarH;
    _buttonRounding          = _resources.style().buttonRounding * _screenH;
    _textWrapW               = 0.9f * _screenW;
    _windowPaddingContent    = _resources.style().windowPaddingContent * _screenH;
    _itemSpacingContent      = _resources.style().itemSpacingContent * _screenH;
}

void LocationMapGui::onMouseDown(SLMouseButton button, SLint x, SLint y)
{
    ImGuiWrapper::onMouseDown(button, x, y);
    _move     = true;
    _lastPosY = y;
    _lastPosX = x;
}

void LocationMapGui::onMouseUp(SLMouseButton button, SLint x, SLint y)
{
    ImGuiWrapper::onMouseUp(button, x, y);
    _move     = false;
    _lastPosY = y;
    _lastPosX = x;
}

void LocationMapGui::onMouseMove(SLint xPos, SLint yPos)
{
    if (_move)
    {
        //[0, 1] on the texture
        float dx = (float)(xPos - _lastPosX) * _fracW / _screenW;
        float dy = (float)(yPos - _lastPosY) * _fracH / _screenH;
        _x -= dx;
        _y -= dy;

        if (_x + _fracW > 1.0)
            _x = 1.0f - _fracW;
        if (_x < 0.0)
            _x = 0.0;
        if (_y + _fracH > 1.0)
            _y = 1.0f - _fracH;
        if (_y < 0.0)
            _y = 0.0;
    }

    _lastPosY = yPos;
    _lastPosX = xPos;
}

void LocationMapGui::build(SLScene* s, SLSceneView* sv)
{
    //background texture
    if (_locMapTexId != 0)
    {
        ErlebAR::renderPanningBackgroundTexture(_x, _y, _fracW, _fracH, _screenW, _screenH, _locMapTexId);
    }

    //header bar
    float buttonSize = _resources.style().headerBarButtonH * _headerBarH;

    ErlebAR::renderHeaderBar("LocationMapGui",
                             _screenW,
                             _headerBarH,
                             _resources.style().headerBarBackgroundTranspColor,
                             _resources.style().headerBarTextColor,
                             _resources.style().headerBarBackButtonTranspColor,
                             _resources.style().headerBarBackButtonPressedTranspColor,
                             _resources.fonts().headerBar,
                             _buttonRounding,
                             buttonSize,
                             _resources.textures.texIdBackArrow,
                             _spacingBackButtonToText,
                             _loc.name.c_str(),
                             [&]() { sendEvent(new GoBackEvent("LocationMapGui")); });

    //content
    if (_loc.id != LocationId::NONE)
    {
        ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoTitleBar |
                                       ImGuiWindowFlags_NoMove |
                                       ImGuiWindowFlags_AlwaysAutoResize |
                                       ImGuiWindowFlags_NoBackground |
                                       ImGuiWindowFlags_NoScrollbar |
                                       ImGuiWindowFlags_NoScrollWithMouse;

        ImGui::SetNextWindowPos(ImVec2(0, _contentStartY), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(_screenW, _contentH), ImGuiCond_Always);

        ImGui::PushStyleColor(ImGuiCol_WindowBg, _resources.style().transparentColor);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, _buttonRounding);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(_windowPaddingContent, _windowPaddingContent));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(_itemSpacingContent, _itemSpacingContent));

        ImGui::PushStyleColor(ImGuiCol_Button, _resources.style().areaPoseButtonColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, _resources.style().areaPoseButtonColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, _resources.style().areaPoseButtonColorPressed);

        ImGui::Begin("LocationMapGui_content", nullptr, windowFlags);

        float triangleWidth  = _resources.style().areaPoseButtonViewTriangleWidth * _screenH;
        float triangleLength = _resources.style().areaPoseButtonViewTriangleLength * triangleWidth;
        float circleRadius   = _resources.style().areaPoseButtonCircleRadius * triangleWidth;

        int   i          = 0;
        float buttonSize = 0.1f * _screenH;
        for (const auto& it : _loc.areas)
        {
            const Area& area = it.second;

            //[0, 1] on the texture
            float x = (float)area.xPosPix / (float)_locTextureW - _x;
            float y = (float)area.yPosPix / (float)_locTextureH - _y;

            //[0, texSize] to screen coordinate
            ImGui::SetCursorPosX(x * (float)_locTextureW * (float)_screenW / (float)(_dspPixWidth));
            ImGui::SetCursorPosY(y * (float)_locTextureH * (float)_screenH / (float)(_dspPixHeight)-_headerBarH);

            //ImGui::PushID(i);
            if (ErlebAR::poseShapeButton(area.name.c_str(),
                                         ImVec2(buttonSize, buttonSize),
                                         circleRadius,
                                         triangleLength,
                                         triangleWidth,
                                         area.viewAngleDeg,
                                         _resources.style().areaPoseButtonShapeColor,
                                         _resources.style().areaPoseButtonShapeColorPressed))
            {
                sendEvent(new AreaSelectedEvent("LocationMapGui", _loc.id, it.first));
            }

            //ImGui::PopID();
            i++;
        }

        ImGui::End();

        ImGui::PopStyleColor(4);
        ImGui::PopStyleVar(6);
    }
    //ImGui::ShowMetricsWindow();

    //debug: draw log window
    _resources.logWinDraw();
}

void LocationMapGui::initLocation(ErlebAR::LocationId locId)
{
    const auto& locations = _resources.locations();
    auto        locIt     = locations.find(locId);
    if (locIt != locations.end())
    {
        _loc = locIt->second;

        //reload texture
        ErlebAR::deleteTexture(_locMapTexId);
        _locMapTexId = ErlebAR::loadTexture(_erlebARDir + _loc.areaMapImageFileName,
                                            false,
                                            true,
                                            _screenW / _screenH,
                                            _locTextureW,
                                            _locTextureH);

        float screenRatio = _screenH / _screenW;
        _dspPixWidth      = (float)_loc.dspPixWidth;
        _dspPixHeight     = (float)(_loc.dspPixWidth * screenRatio);

        _fracW = _dspPixWidth / (float)_locTextureW; //Should never be bigger than 1
        _fracH = _dspPixHeight / (float)_locTextureH;

        if (_fracW > 1.0)
        {
            _fracW        = 1.0;
            _fracH        = _screenH / _screenW;
            _dspPixWidth  = (float)_locTextureW;
            _dspPixHeight = (float)(_dspPixWidth * _screenH) / (float)(_screenW);
        }
        /*if (_fracH > 1.0)
        {
            _fracH        = 1.0;
            _fracW        = _screenW / _screenH;
            _dspPixHeight = (float)_locTextureH;
            _dspPixWidth  = (float)(_dspPixHeight * _screenW) / (float)(_screenH);
        }*/
    }
    else
        Utils::exitMsg("LocationMapGui", "No location defined for location id!", __LINE__, __FILE__);
}
