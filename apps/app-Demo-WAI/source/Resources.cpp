#include "Resources.h"
#include <LogWindow.h>
#include "opencv2/core/persistence.hpp"
#include <sstream>

namespace ErlebAR
{
Fonts::Fonts()
{
    _atlas = new ImFontAtlas();
}

Fonts::~Fonts()
{
    if (_atlas)
        delete _atlas;
}

void Fonts::load(std::string fontDir, const Style& style, int screenH, int dpi)
{
    //ImGuiContext* context = ImGui::GetCurrentContext();
    //if (!context)
    //    context = ImGui::CreateContext();

    //ImFontAtlas* atlas = context->IO.Fonts;
    std::string unitRounded      = fontDir + "UnitRoundedPro.otf";
    std::string unitRoundedMedi  = fontDir + "UnitRoundedPro-Medi.otf";
    std::string unitRoundedLight = fontDir + "UnitRoundedPro-Light.otf";
    if (Utils::fileExists(unitRounded) && ::fileExists(unitRoundedMedi))
    {
        //header bar font
        float headerBarH     = style.headerBarPercH * screenH;
        float headerBarTextH = style.headerBarTextH * headerBarH;
        headerBar            = _atlas->AddFontFromFileTTF(unitRoundedMedi.c_str(), headerBarTextH);
        //headerBar            = _atlas->AddFontFromFileTTF(ttf.c_str(), headerBarTextH);
        //standard font
        //float standardTextH = style.textStandardH * (float)screenH;
        float standardTextH = style.textStandardHMM * (float)dpi * 0.0393701f;
        standard            = _atlas->AddFontFromFileTTF(unitRoundedLight.c_str(), standardTextH);
        //heading font
        float headingTextH = style.textHeadingH * (float)screenH;
        heading            = _atlas->AddFontFromFileTTF(unitRoundedMedi.c_str(), headingTextH);
        //tiny font
        float tinyTextH = 0.035f * (float)screenH;
        tiny            = _atlas->AddFontFromFileTTF(unitRoundedLight.c_str(), tinyTextH);
        /*
        //big font
        float bigTextHPix      = 0.3f * (float)screenH;
        float scale            = 2.0f;
        float bigTextHPixAlloc = bigTextHPix / scale;
        float bigTextH         = 0.035f * (float)screenH;
        big                    = _atlas->AddFontFromFileTTF(unitRounded.c_str(), bigTextHPixAlloc);
        big->Scale             = scale;
        */
        //selection buttons
        int   nButVert  = 6;
        int   buttonH   = (int)((0.7f * (float)screenH - (nButVert - 1) * 0.02f * (float)screenH) / nButVert);
        float selectBtn = buttonH * style.buttonTextH;
        selectBtns      = _atlas->AddFontFromFileTTF(unitRoundedMedi.c_str(), selectBtn);
    }
    else
    {
        std ::stringstream ss;
        ss << "Fonts do not exist: " << unitRounded << ", " << unitRoundedMedi;
        Utils::exitMsg("Resources", ss.str().c_str(), __LINE__, __FILE__);
    }
}

Resources::Resources(const DeviceData& deviceData, int screenH, std::string languageId)
  : //_screenW(deviceData.scrWidth()),
    //_screenH(deviceData.scrHeight()),
    _writableDir(deviceData.writableDir())
{
    //load strings first (we need the id for string selection)
    stringsEnglish.load(deviceData.stringsDir() + "StringsEnglish.json");
    stringsGerman.load(deviceData.stringsDir() + "StringsGerman.json");
    stringsFrench.load(deviceData.stringsDir() + "StringsFrench.json");
    stringsItalian.load(deviceData.stringsDir() + "StringsItalian.json");
    //load Resources
    //load(deviceData.writableDir() + "ErlebARResources.json");

    if (languageId == stringsGerman.id())
    {
        _currStrings = &stringsGerman;
    }
    else if (languageId == stringsFrench.id())
    {
        _currStrings = &stringsFrench;
    }
    else if (languageId == stringsItalian.id())
    {
        _currStrings = &stringsItalian;
    }
    else
    {
        _currStrings = &stringsEnglish;
    }

    //load textures
    textures.load(deviceData.textureDir());
    //load fonts
    _fonts.load(deviceData.fontDir(), _style, screenH, deviceData.dpi());

    //definition of erlebar locations and areas
    //_locations = ErlebAR::defineLocations();

    //if (logWinEnabled)
    //    logWinInit();
}

Resources::~Resources()
{
    //save();
    //delete shared textures
    textures.free();
    //delete fonts
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
    _currStrings = &stringsItalian;
}

void loadString(const cv::FileStorage& fs, const std::string& name, std::string& target)
{
    if (!fs[name].empty())
        fs[name] >> target;
    else
        Utils::log("Strings", "Warning: String %s does not exist!", name.c_str());
}

void Strings::load(std::string fileName)
{
    if (Utils::fileExists(fileName))
    {
        cv::FileStorage fs(fileName, cv::FileStorage::READ);
        if (fs.isOpened())
        {
            loadString(fs, "id", _id);
            //selection
            loadString(fs, "settings", _settings);
            loadString(fs, "about", _about);
            loadString(fs, "tutorial", _tutorial);
            //about
            loadString(fs, "general", _general);
            loadString(fs, "generalContent", _generalContent);
            loadString(fs, "developers", _developers);
            loadString(fs, "developerNames", _developerNames);
            //settings
            loadString(fs, "language", _language);
            loadString(fs, "develMode", _develMode);
            //errors
            loadString(fs, "cameraStartError", _cameraStartError);
            //info text
            //bern:
            loadString(fs, "bernInfoHeading1", _bernInfoHeading1);
            loadString(fs, "bernInfoText1", _bernInfoText1);
            loadString(fs, "bernInfoHeading2", _bernInfoHeading2);
            loadString(fs, "bernInfoText2", _bernInfoText2);
            //augst:
            loadString(fs, "augstTempleHillInfoHeading1", _augstTempleHillInfoHeading1);
            loadString(fs, "augstTempleHillInfoText1", _augstTempleHillInfoText1);
            loadString(fs, "augstTempleHillInfoText2", _augstTempleHillInfoText2);
        }
    }
    else
    {
        Utils::log("Strings", "Warning: Strings file does not exist: %s", fileName.c_str());
    }
}

Config::Config(const DeviceData& deviceData)
{
    _screenW = deviceData.scrWidth();
    _screenH = deviceData.scrHeight();

    load(deviceData.writableDir() + "ErlebARResources.json");
    //init resources
    _resources = new Resources(deviceData, deviceData.scrHeight(), _languageId);

    //definition of erlebar locations and areas
    _locations = ErlebAR::defineLocations();

    if (logWinEnabled)
        logWinInit();
}

Config::~Config()
{
    save();

    if (_resources)
        delete _resources;
}

void Config::load(std::string resourceFileName)
{
    _fileName = resourceFileName;

    Utils::log("ErlebAR::Config", "loading config from %s", _fileName.c_str());
    cv::FileStorage fs(resourceFileName, cv::FileStorage::READ);
    if (fs.isOpened())
    {
        if (!fs["developerMode"].empty())
            fs["developerMode"] >> developerMode;
        if (!fs["simulatorMode"].empty())
            fs["simulatorMode"] >> simulatorMode;
        if (!fs["enableUserGuidance"].empty())
            fs["enableUserGuidance"] >> enableUserGuidance;
        if (!fs["logWinEnabled"].empty())
            fs["logWinEnabled"] >> logWinEnabled;

        if (!fs["useGps"].empty())
            fs["useGps"] >> useGps;
        if (!fs["useARCore"].empty())
            fs["useARCore"] >> useARCore;
        if (!fs["useWAISlam"].empty())
            fs["useWAISlam"] >> useWAISlam;

        if (!fs["languageId"].empty())
        {
            std::string languageId;
            fs["languageId"] >> languageId;
        }
    }
    else
    {
        Utils::warnMsg("ErlebAR::Config", "Could not load config!", __LINE__, __FILE__);
    }
}

void Config::save()
{
    Utils::log("ErlebAR::Config", "saving config to %s", _fileName.c_str());
    cv::FileStorage fs(_fileName, cv::FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "developerMode" << developerMode;
        fs << "simulatorMode" << simulatorMode;
        fs << "enableUserGuidance" << enableUserGuidance;
        fs << "logWinEnabled" << logWinEnabled;

        fs << "useGps" << useGps;
        fs << "useARCore" << useARCore;
        fs << "useWAISlam" << useWAISlam;

        fs << "languageId" << _resources->strings().id();
    }
    else
    {
        Utils::warnMsg("ErlebAR::Config", "Could not save config!", __LINE__, __FILE__);
    }
}

void Config::logWinInit()
{
    Utils::customLog = std::make_unique<LogWindow>(_screenW, _screenH, _resources->fonts().tiny, _resources->fonts().standard);
}

void Config::logWinUnInit()
{
    if (Utils::customLog)
        Utils::customLog.release();
}

void Config::logWinDraw()
{
    if (Utils::customLog)
    {
        LogWindow* log = static_cast<LogWindow*>(Utils::customLog.get());
        log->draw("Log");
    }
}

};
