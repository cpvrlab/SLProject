#include <AreaTrackingGui.h>
#include <imgui_internal.h>
#include <GuiUtils.h>
#include <ErlebAREvents.h>

using namespace ErlebAR;

AreaTrackingGui::AreaTrackingGui(sm::EventHandler&   eventHandler,
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
        Utils::warnMsg("AreaTrackingGui", "font does not exist!", __LINE__, __FILE__);
}

AreaTrackingGui::~AreaTrackingGui()
{
}

void AreaTrackingGui::onShow()
{
    _panScroll.enable();
}

void AreaTrackingGui::onResize(SLint scrW, SLint scrH)
{
    resize(scrW, scrH);
    ImGuiWrapper::onResize(scrW, scrH);
}

void AreaTrackingGui::resize(int scrW, int scrH)
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

void AreaTrackingGui::build(SLScene* s, SLSceneView* sv)
{
    //header bar
    float buttonSize = _resources.style().headerBarButtonH * _headerBarH;

    ErlebAR::renderHeaderBar("AreaTrackingGui",
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
                             _area.name,
                             [&]() { sendEvent(new GoBackEvent()); });

    //ImGui::ShowMetricsWindow();
}

void AreaTrackingGui::initArea(ErlebAR::Area area)
{
    _area = area;
    //start wai with map for this area
    //load model into scene graph
}
