#include <WelcomeGui.h>
#include <ErlebAR.h>
#include <imgui_internal.h>
#include <CVImage.h>

using namespace ErlebAR;

WelcomeGui::WelcomeGui(const ImGuiEngine&  imGuiEngine,
                       ErlebAR::Resources& resources,
                       int                 dotsPerInch,
                       int                 screenWidthPix,
                       int                 screenHeightPix,
                       std::string         fontPath,
                       std::string         texturePath,
                       std::string         version)
  : ImGuiWrapper(imGuiEngine.context(), imGuiEngine.renderer()),
    _versionStr(version),
    _resources(resources)
{
    //load bfh logo texture
    std::string logoBFHPath = texturePath + "logo_bfh.png";
    if (Utils::fileExists(logoBFHPath))
    {
        // load texture image
        CVImage logoBFH(logoBFHPath);
        logoBFH.flipY();
        _textureBFHW = logoBFH.width();
        _textureBFHH = logoBFH.height();

        // Create a OpenGL texture identifier
        glGenTextures(1, &_logoBFHTexId);
        glBindTexture(GL_TEXTURE_2D, _logoBFHTexId);

        // Setup filtering parameters for display
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Upload pixels into texture
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     logoBFH.format(),
                     (GLsizei)logoBFH.width(),
                     (GLsizei)logoBFH.height(),
                     0,
                     logoBFH.format(),
                     GL_UNSIGNED_BYTE,
                     (GLvoid*)logoBFH.data());
    }
    else
        Utils::warnMsg("WelcomeGui", "logoBFHPath does not exist!", __LINE__, __FILE__);

    //load admin ch logo
    std::string logoAdminCHPath = texturePath + "logo_admin_ch.png";
    if (Utils::fileExists(logoAdminCHPath))
    {
        // load texture image
        CVImage logoAdminCH(logoAdminCHPath);
        logoAdminCH.flipY();
        _textureAdminCHW = logoAdminCH.width();
        _textureAdminCHH = logoAdminCH.height();

        // Create a OpenGL texture identifier
        glGenTextures(1, &_logoAdminCHTexId);
        glBindTexture(GL_TEXTURE_2D, _logoAdminCHTexId);

        // Setup filtering parameters for display
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Upload pixels into texture
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     logoAdminCH.format(),
                     (GLsizei)logoAdminCH.width(),
                     (GLsizei)logoAdminCH.height(),
                     0,
                     logoAdminCH.format(),
                     GL_UNSIGNED_BYTE,
                     (GLvoid*)logoAdminCH.data());
    }
    else
        Utils::warnMsg("WelcomeGui", "logoAdminCHPath does not exist!", __LINE__, __FILE__);

    resize(screenWidthPix, screenHeightPix);
}

WelcomeGui::~WelcomeGui()
{
    if (_logoBFHTexId)
    {
        glDeleteTextures(1, &_logoBFHTexId);
        _logoBFHTexId = 0;
    }
    if (_logoAdminCHTexId)
    {
        glDeleteTextures(1, &_logoAdminCHTexId);
        _logoAdminCHTexId = 0;
    }
}

void WelcomeGui::onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY)
{
    resize(scrW, scrH);
    ImGuiWrapper::onResize(scrW, scrH, scr2fbX, scr2fbY);
}

void WelcomeGui::resize(int scrW, int scrH)
{
    _screenWPix = (float)scrW;
    _screenHPix = (float)scrH;

    _textFrameLRPix = _screenWPix * 0.15;
    _textFrameTPix  = _screenHPix * 0.2;
    _logoFrameBPix  = 0.07f * _screenHPix;
    _logoFrameLRPix = _logoFrameBPix;

    _bfhLogoHPix   = 0.2f * _screenHPix;
    _bfhLogoWPix   = _bfhLogoHPix / (float)_textureBFHH * (float)_textureBFHW;
    _adminLogoHPix = 0.15f * _screenHPix;
    _adminLogoWPix = _adminLogoHPix / (float)_textureAdminCHH * (float)_textureAdminCHW;
}

void WelcomeGui::build(SLScene* s, SLSceneView* sv)
{
    //push styles at first
    pushStyle();

    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(_screenWPix, _screenHPix), ImGuiCond_Always);
    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoScrollbar;

    ImGui::Begin("WelcomeGui", nullptr, windowFlags);

    //big text
    {
        ImGui::PushFont(_resources.fonts().big);
        ImGui::SetNextWindowPos(ImVec2(_textFrameLRPix, _textFrameTPix), ImGuiCond_Always);
        ImGui::BeginChild("ChildBigText", ImVec2(0, 0), false, windowFlags);
        ImGui::Text("ErlebAR");
        ImGui::EndChild();
        ImGui::PopFont();
    }
    //small text
    {
        ImGui::PushFont(_resources.fonts().tiny);
        ImGui::SetNextWindowPos(ImVec2(_textFrameLRPix, _textFrameTPix + _resources.fonts().big->FontSize * _resources.fonts().big->Scale), ImGuiCond_Always);
        ImGui::BeginChild("ChildSmallText", ImVec2(0, 0), false, windowFlags);
        ImGui::Text(_versionStr.c_str());
        ImGui::EndChild();
        ImGui::PopFont();
    }

    //bfh logo texture
    {
        ImGui::SetNextWindowPos(ImVec2(_textFrameLRPix, _screenHPix - _logoFrameBPix - _bfhLogoHPix), ImGuiCond_Always);
        ImGui::BeginChild("BFHLogo", ImVec2(_bfhLogoWPix, _bfhLogoHPix));
        ImGui::Image((void*)(intptr_t)_logoBFHTexId, ImVec2(_bfhLogoWPix, _bfhLogoHPix));
        ImGui::EndChild();
    }

    //admin ch logo texture
    {
        ImGui::SetNextWindowPos(ImVec2(_screenWPix - _adminLogoWPix - _textFrameLRPix, _screenHPix - _logoFrameBPix - _bfhLogoHPix), ImGuiCond_Always);
        ImGui::BeginChild("AdminCHLogo", ImVec2(_adminLogoWPix + 10, _adminLogoHPix + 10));
        ImGui::Image((void*)(intptr_t)_logoAdminCHTexId, ImVec2(_adminLogoWPix, _adminLogoHPix));
        ImGui::EndChild();
    }

    ImGui::End();

    popStyle();

    //ImGui::ShowMetricsWindow();
}

void WelcomeGui::pushStyle()
{
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.f, 0.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));

    ImGui::PushStyleColor(ImGuiCol_WindowBg, _resources.style().backgroundColorPrimary);
}

void WelcomeGui::popStyle()
{
    ImGui::PopStyleVar(7);
    ImGui::PopStyleColor(1);
}
