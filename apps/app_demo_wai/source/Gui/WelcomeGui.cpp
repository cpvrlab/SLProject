#include <WelcomeGui.h>
#include <ErlebAR.h>
#include <imgui_internal.h>
#include <cv/CVImage.h>

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
    
    //load erlebar logo texture
    std::string launchImageName = texturePath + "erleb-AR_logo_rounded.png";
    if (Utils::fileExists(launchImageName))
    {
        // load texture image
        CVImage launchImage(launchImageName);
        launchImage.flipY();
        _textureLaunchImgW = launchImage.width();
        _textureLaunchImgH = launchImage.height();

        // Create a OpenGL texture identifier
        glGenTextures(1, &_launchImgTexId);
        glBindTexture(GL_TEXTURE_2D, _launchImgTexId);

        // Setup filtering parameters for display
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Upload pixels into texture
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     launchImage.format(),
                     (GLsizei)launchImage.width(),
                     (GLsizei)launchImage.height(),
                     0,
                     launchImage.format(),
                     GL_UNSIGNED_BYTE,
                     (GLvoid*)launchImage.data());
    }
    else
        Utils::log("WelcomeGui", "ErlebARLaunchImage does not exist: %s!", launchImageName.c_str());
    
    
    resize(screenWidthPix, screenHeightPix);
}

WelcomeGui::~WelcomeGui()
{
    if (_launchImgTexId)
    {
        glDeleteTextures(1, &_launchImgTexId);
        _launchImgTexId = 0;
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
}

void WelcomeGui::build(SLScene* s, SLSceneView* sv)
{
    //push styles at first
    pushStyle();

    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(_screenWPix, _screenHPix), ImGuiCond_Always);
    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoScrollbar;

    ImGui::Begin("WelcomeGui", nullptr, windowFlags);

    float imgW, imgH, xPos, yPos;
    if(_screenHPix < _screenWPix)
        imgH = _scaleToSmallerLen * (float)_screenHPix;
    else
        imgH = _scaleToSmallerLen * (float)_screenWPix;
    
    imgW = (float)_textureLaunchImgW / (float)_textureLaunchImgH * imgH;
    xPos = ((float)_screenWPix - imgW) * 0.5f;
    yPos = ((float)_screenHPix - imgH) * 0.5f;
    
    {
        ImGui::SetNextWindowPos(ImVec2(xPos, yPos), ImGuiCond_Always);
        ImGui::BeginChild("BFHLogo", ImVec2(imgW, imgH));
        ImGui::Image((void*)(intptr_t)_launchImgTexId, ImVec2(imgW, imgH));
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

    ImGui::PushStyleColor(ImGuiCol_WindowBg, _resources.style().backgroundColorWelcome);
}

void WelcomeGui::popStyle()
{
    ImGui::PopStyleVar(7);
    ImGui::PopStyleColor(1);
}
