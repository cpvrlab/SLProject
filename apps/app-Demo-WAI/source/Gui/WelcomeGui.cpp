#include <WelcomeGui.h>
#include <ErlebAR.h>
#include <imgui_internal.h>

WelcomeGui::WelcomeGui(int         dotsPerInch,
                       int         screenWidthPix,
                       int         screenHeightPix,
                       std::string fontPath,
                       std::string version)
  : _pixPerMM((float)dotsPerInch / 25.4f),
    _screenWidthPix(screenWidthPix),
    _screenHeightPix(screenHeightPix),
    _versionStr(version)
{
    //load fonts for big ErlebAR text and verions text
    SLstring ttf = fontPath + "Roboto-Medium.ttf";

    if (Utils::fileExists(ttf))
    {
        _fontHeightBigDots = 0.3f * (float)screenHeightPix;
        _smallFontShift    = (float)_fontHeightBigDots / 135.f * 7.f;
        _fontBig           = _context->IO.Fonts->AddFontFromFileTTF(ttf.c_str(), _fontHeightBigDots);

        int fontHeightSmallDots = 0.05f * (float)screenHeightPix;
        _fontSmall              = _context->IO.Fonts->AddFontFromFileTTF(ttf.c_str(), fontHeightSmallDots);
    }
    else
        Utils::warnMsg("WelcomeGui", "font does not exist!", __LINE__, __FILE__);

    /*
    //load bfh logo texture
    std::string logoBFHPath = fontPath + "../textures/logo_bfh.png";
    if (Utils::fileExists(logoBFHPath))
    {
        // load texture image
        CVImage logoBFH(logoBFHPath);
        logoBFH.flipY();
        _logoBFHWidth  = logoBFH.width();
        _logoBFHHeight = logoBFH.height();

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
    std::string logoAdminCHPath = fontPath + "../textures/logo_admin_ch.png";
    if (Utils::fileExists(logoAdminCHPath))
    {
        // load texture image
        CVImage logoAdminCH(logoAdminCHPath);
        logoAdminCH.flipY();
        _logoAdminCHWidth  = logoAdminCH.width();
        _logoAdminCHHeight = logoAdminCH.height();

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
        */
}

void WelcomeGui::build(SLScene* s, SLSceneView* sv)
{
    //push styles at first
    pushStyle();

    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(_screenWidthPix, _screenHeightPix), ImGuiCond_Always);
    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoScrollbar;

    ImGui::Begin("WelcomeGui", nullptr, windowFlags);

    //big text
    {
        ImGui::PushFont(_fontBig);
        ImGui::SetNextWindowPos(ImVec2((float)_screenHeightPix * 0.2, (float)_screenHeightPix * 0.2), ImGuiCond_Always);
        ImGui::BeginChild("ChildBigText", ImVec2(0, 0), false, windowFlags);
        ImGui::Text("ErlebAR");
        ImGui::EndChild();
        ImGui::PopFont();
    }
    //small text
    {
        ImGui::PushFont(_fontSmall);
        ImGui::SetNextWindowPos(ImVec2((float)_screenHeightPix * 0.2, (float)_screenHeightPix * 0.2 + _fontHeightBigDots), ImGuiCond_Always);
        ImGui::BeginChild("ChildSmallText", ImVec2(0, 0), false, windowFlags);
        ImGui::SameLine(_smallFontShift); //I have problems to get the fonts aligned vertically.. this is a hack
        ImGui::Text(_versionStr.c_str());
        ImGui::EndChild();
        ImGui::PopFont();
    }

    ////bfh logo texture
    //{
    //    ImGui::SetNextWindowPos(ImVec2((float)_screenHeightPix * 0.2, (float)_screenHeightPix * 0.2 + _fontHeightBigDots), ImGuiCond_Always);
    //    ImGui::BeginChild("BFHLogo");
    //    ImGui::Image((void*)(intptr_t)_logoBFHTexId, ImVec2(_logoBFHWidth, _logoBFHHeight));
    //    ImGui::EndChild();
    //}

    ////admin ch logo texture
    //{
    //    ImGui::SetNextWindowPos(ImVec2((float)_screenHeightPix * 0.2, (float)_screenHeightPix * 0.2 + _fontHeightBigDots), ImGuiCond_Always);
    //    ImGui::BeginChild("AdminCHLogo");
    //    ImGui::Image((void*)(intptr_t)_logoAdminCHTexId, ImVec2(_logoAdminCHWidth, _logoAdminCHHeight));
    //    ImGui::EndChild();
    //}

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
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));

    const SLVec4f o = BFHColors::OrangePrimary;
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(o.r, o.g, o.b, o.a));
}

void WelcomeGui::popStyle()
{
    ImGui::PopStyleVar(6);
    ImGui::PopStyleColor(1);
}
