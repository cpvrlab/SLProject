#include "Resources.h"
#include <LogWindow.h>
#include "opencv2/core/persistence.hpp"
#include <sstream>

namespace ErlebAR
{
Fonts::Fonts()
{
    //atlas = new ImFontAtlas();
}

Fonts::~Fonts()
{
    //if (atlas)
    //    delete atlas;
}

void Fonts::load(std::string fontDir, const Style& style, int screenH)
{
    ImGuiContext* context = ImGui::GetCurrentContext();
    if (!context)
        context = ImGui::CreateContext();

    ImFontAtlas* atlas = context->IO.Fonts;
    std::string  ttf   = fontDir + "Roboto-Medium.ttf";
    if (Utils::fileExists(ttf))
    {
        //header bar font
        float headerBarH     = style.headerBarPercH * (float)screenH;
        float headerBarTextH = style.headerBarTextH * (float)headerBarH;
        headerBar            = atlas->AddFontFromFileTTF(ttf.c_str(), headerBarTextH);
        //standard font
        float standardTextH = style.textStandardH * (float)screenH;
        standard            = atlas->AddFontFromFileTTF(ttf.c_str(), standardTextH);
        //heading font
        float headingTextH = style.textHeadingH * (float)screenH;
        heading            = atlas->AddFontFromFileTTF(ttf.c_str(), headingTextH);
        //tiny font
        float tinyTextH = 0.035f * (float)screenH;
        tiny            = atlas->AddFontFromFileTTF(ttf.c_str(), tinyTextH);
        //big font
        float bigTextHPix      = 0.3f * (float)screenH;
        float scale            = 2.0f;
        float bigTextHPixAlloc = bigTextHPix / scale;
        float bigTextH         = 0.035f * (float)screenH;
        big                    = atlas->AddFontFromFileTTF(ttf.c_str(), bigTextHPixAlloc);
        big->Scale             = scale;
        //selection buttons
        int   nButVert  = 6;
        int   buttonH   = (0.6f * (float)screenH - (nButVert - 1) * 0.02f * (float)screenH) / nButVert;
        float selectBtn = buttonH * style.buttonTextH;
        selectBtns      = atlas->AddFontFromFileTTF(ttf.c_str(), selectBtn);
    }
    else
    {
        std ::stringstream ss;
        ss << "Font does not exist: " << ttf;
        Utils::exitMsg("Resources", ss.str().c_str(), __LINE__, __FILE__);
    }
}

Resources::Resources(int         screenWidth,
                     int         screenHeight,
                     std::string writableDir,
                     std::string textureDir,
                     std::string fontDir)
  : _screenW(screenWidth),
    _screenH(screenHeight),
    _writableDir(writableDir)
{
    load(writableDir + "ErlebARResources.json");

    //load textures
    textures.load(textureDir);
    //load fonts
    _fonts.load(fontDir, _style, screenHeight);

    //definition of erlebar locations and areas
    _locations = ErlebAR::defineLocations();

    if (logWinEnabled)
        logWinInit();
}

Resources::~Resources()
{
    save();
    //delete shared textures
    textures.free();
    //delete fonts
}

void Resources::load(std::string resourceFileName)
{
    _fileName = resourceFileName;

    cv::FileStorage fs(resourceFileName, cv::FileStorage::READ);
    if (fs.isOpened())
    {
        if (!fs["developerMode"].empty())
            fs["developerMode"] >> developerMode;
        if (!fs["logWinEnabled"].empty())
            fs["logWinEnabled"] >> logWinEnabled;

        if (!fs["languageId"].empty())
        {
            std::string languageId;
            fs["languageId"] >> languageId;
            if (languageId == stringsGerman.id())
            {
                _currStrings = &stringsGerman;
            }
            else if (languageId == stringsFrench.id())
            {
                _currStrings = &stringsFrench;
            }
            else if (languageId == stringsItalien.id())
            {
                _currStrings = &stringsItalien;
            }
            else
            {
                _currStrings = &stringsEnglish;
            }
        }
    }
    else
    {
        Utils::warnMsg("ErlebAR::Resources", "Could not save resources!", __LINE__, __FILE__);
    }
}

void Resources::save()
{
    cv::FileStorage fs(_fileName, cv::FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "developerMode" << developerMode;
        fs << "logWinEnabled" << logWinEnabled;
        fs << "languageId" << _currStrings->id();
    }
    else
    {
        Utils::warnMsg("ErlebAR::Resources", "Could not save resources!", __LINE__, __FILE__);
    }
}

void Resources::setLanguageGerman()
{
    _currStrings = &stringsGerman;
}
void Resources::setLanguageEnglish()
{
    _currStrings = &stringsEnglish;
}
void Resources::setLanguageFrench()
{
    _currStrings = &stringsFrench;
}
void Resources::setLanguageItalien()
{
    _currStrings = &stringsItalien;
}

void Resources::logWinInit()
{
    Utils::customLog = std::make_unique<LogWindow>(_screenW, _screenH);
}

void Resources::logWinUnInit()
{
    if (Utils::customLog)
        Utils::customLog.release();
}

void Resources::logWinDraw()
{
    if (Utils::customLog)
    {
        LogWindow* log = static_cast<LogWindow*>(Utils::customLog.get());
        log->draw(fonts().tiny, fonts().standard, "Log");
    }
}

Strings::Strings()
{
    _developerNames = "Jan Dellsperger\nLuc Girod\nMichael Göttlicher";
}

StringsEnglish::StringsEnglish()
{
    _id = "English";

    _settings = "Settings";
    _about    = "About";
    _tutorial = "Tutorial";

    _general        = "General";
    _generalContent = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla non scelerisque nisi, in egestas massa. Nulla non lorem nec magna consequat convallis. Integer est ex, pellentesque vitae tristique sit amet, tempor nec ante. Phasellus tristique nulla felis, non malesuada diam convallis vitae. Aliquam enim leo, molestie quis diam in, blandit venenatis neque. Cras imperdiet metus at enim egestas fringilla. Aliquam facilisis purus nisl, eget elementum ligula rutrum et. Sed et tincidunt arcu. Suspendisse interdum et dolor nec facilisis. Vivamus aliquet non dolor sit amet dignissim. Vestibulum fringilla nisi vel ultricies aliquet. Ut sed nibh at ligula posuere luctus. Maecenas turpis tortor, tincidunt a gravida sit amet, tincidunt a purus. Vestibulum vitae mollis est, non blandit massa.\nInteger in felis vestibulum, rhoncus turpis a, iaculis nulla. Sed at sapien sit amet ligula ultrices luctus vel id massa. Quisque sodales aliquet mi, sed elementum purus mattis et. Phasellus sit amet aliquet odio. Nam magna purus, ullamcorper a nibh ac, semper euismod dui. Morbi in ipsum lectus. Pellentesque id rhoncus nibh. Vivamus vulputate egestas volutpat. Morbi at luctus nunc, quis congue sem. Ut luctus ligula libero.\nDonec tempor, mauris vitae faucibus euismod, dolor ante bibendum enim, non molestie orci massa sit amet nisi. Suspendisse non nisi eget mi iaculis lacinia. Mauris tempor leo eu posuere tempor. Nunc quis magna et sem fermentum accumsan. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Nam non interdum lorem. Vivamus feugiat purus in congue ornare. Nulla mi eros, ullamcorper at pharetra vitae, dictum a ipsum. In mollis lorem nulla, eget laoreet tellus luctus sit amet. Etiam ac eros ex. Curabitur id arcu vitae purus pulvinar euismod sed nec lectus. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Proin eu magna magna. Aliquam erat volutpat. Nulla sit amet porta eros.\nEtiam sodales varius pulvinar. Nam venenatis dictum turpis, vitae dignissim enim tincidunt sit amet. Quisque tristique placerat est, vel dapibus enim posuere et. Nulla commodo fermentum maximus. Ut facilisis nisi id turpis varius sodales. Fusce feugiat lobortis facilisis. Fusce sit amet efficitur purus, quis commodo diam. Donec eleifend turpis ligula, a lacinia risus porta eget.\nCras auctor ultrices tempus. Phasellus sed commodo ex, in cursus ex. Nunc ac quam et diam bibendum venenatis eget vel velit. Duis id dui dolor. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; In tempor sollicitudin mauris, eget ornare tortor. Cras vel lacus non sem faucibus accumsan. Vestibulum placerat finibus elit. Integer nisl velit, egestas nec urna malesuada, malesuada dictum dui.";
    _developers     = "Developers";

    _language  = "Language";
    _develMode = "Developer mode";
}

StringsGerman::StringsGerman()
{
    _id = "German";

    _settings = "Einstellungen";
    _about    = "Info";
    _tutorial = "Anleitung";

    _general        = "Allgemein";
    _generalContent = "";
    _developers     = "Entwickler";

    _language  = "Sprache";
    _develMode = "Entwicklermodus";
}

StringsFrench::StringsFrench()
{
    _id = "French";

    _settings = "Paramètres";
    _about    = "À propos";
    _tutorial = "Manuel";

    _general        = "";
    _generalContent = "";
    _developers     = "développeur";

    _language  = "Langue";
    _develMode = "";
}

StringsItalien::StringsItalien()
{
    _id = "Italien";

    _settings = "a";
    _about    = "b";
    _tutorial = "c";

    _general        = "d";
    _generalContent = "e";
    _developers     = "f";

    _language  = "g";
    _develMode = "h";
}

};
