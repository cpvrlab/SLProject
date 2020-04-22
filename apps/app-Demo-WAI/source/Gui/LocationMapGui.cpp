#include <LocationMapGui.h>
#include <imgui_internal.h>
#include <GuiUtils.h>
#include <ErlebAREvents.h>

using namespace ErlebAR;

LocationMapGui::LocationMapGui(sm::EventHandler&   eventHandler,
                               ErlebAR::Resources& resources,
                               int                 dotsPerInch,
                               int                 screenWidthPix,
                               int                 screenHeightPix,
                               std::string         fontPath)
  : sm::EventSender(eventHandler),
    _resources(resources)
{
    resize(screenWidthPix, screenHeightPix);
    float bigTextH = _resources.style().headerBarTextH * (float)_headerBarH;
    //load fonts for big ErlebAR text and verions text
    SLstring ttf = fontPath + "Roboto-Medium.ttf";

    if (Utils::fileExists(ttf))
    {
        _fontBig = _context->IO.Fonts->AddFontFromFileTTF(ttf.c_str(), bigTextH);
    }
    else
        Utils::warnMsg("LocationMapGui", "font does not exist!", __LINE__, __FILE__);
}

LocationMapGui::~LocationMapGui()
{
}

void LocationMapGui::onShow()
{
    _panScroll.enable();
}

void LocationMapGui::onResize(SLint scrW, SLint scrH)
{
    resize(scrW, scrH);
    ImGuiWrapper::onResize(scrW, scrH);
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

void LocationMapGui::build(SLScene* s, SLSceneView* sv)
{
    //background texture
    if (_locMapTexId != 0)
        ErlebAR::renderBackgroundTexture(_screenW, _screenH, _locMapTexId);

    //header bar
    float buttonSize = _resources.style().headerBarButtonH * _headerBarH;

    ErlebAR::renderHeaderBar("LocationMapGui",
                             _screenW,
                             _headerBarH,
                             _resources.style().headerBarBackgroundTranspColor,
                             _resources.style().headerBarTextColor,
                             _resources.style().headerBarBackButtonTranspColor,
                             _resources.style().headerBarBackButtonPressedTranspColor,
                             _fontBig,
                             _buttonRounding,
                             buttonSize,
                             _resources.textures.texIdBackArrow,
                             _spacingBackButtonToText,
                             _loc.name,
                             [&]() { sendEvent(new GoBackEvent()); });

    //render area place buttons
    //if (_loc.id != LocationId::NONE)
    //    ErlebAR::renderAreaPlaceButtons(_loc.areas);

    //content
    if (_loc.id != LocationId::NONE)
    {
        ImGuiWindowFlags childWindowFlags = ImGuiWindowFlags_NoTitleBar |
                                            ImGuiWindowFlags_NoMove |
                                            ImGuiWindowFlags_AlwaysAutoResize |
                                            ImGuiWindowFlags_NoBringToFrontOnFocus |
                                            ImGuiWindowFlags_NoScrollbar;
        ImGuiWindowFlags windowFlags = childWindowFlags |
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

        ImGui::Begin("LocationMap_content", nullptr, windowFlags);
        //todo: offset of cropping (x, y) and header bar (y)
        for (const auto& it : _loc.areas)
        {
            const Area& area = it.second;
            ImGui::SetCursorPosX(_locImgCropW + area.xPosPix);
            ImGui::SetCursorPosY(_locImgCropH + _headerBarH + area.yPosPix);
            //ImGui::BeginChild("LocationMap_content_child", ImVec2(0, 0), false, childWindowFlags);

            //ImGui::EndChild();
        }

        ImGui::End();

        ImGui::PopStyleColor(1);
        ImGui::PopStyleVar(6);
    }

    //ImGui::ShowMetricsWindow();
}

void LocationMapGui::loadLocationMapTexture(std::string fileName)
{
    ErlebAR::deleteTexture(_locMapTexId);
    int cropW, cropH;
    _locMapTexId = ErlebAR::loadTexture(fileName, false, false, (float)_screenW / (float)_screenW, cropW, cropH);
}

void LocationMapGui::initLocation(ErlebAR::Location loc)
{
    _loc = loc;

    //reload texture
    ErlebAR::deleteTexture(_locMapTexId);
    _locMapTexId = ErlebAR::loadTexture(_locationMapImgDir + _loc.areaMapImageFileName,
                                        false,
                                        true,
                                        (float)_screenW / (float)_screenH,
                                        _locImgCropW,
                                        _locImgCropH);
}
