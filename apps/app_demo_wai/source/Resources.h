#ifndef RESOURCES_H
#define RESOURCES_H

#include <string>
#include "ErlebAR.h"
#include <GuiUtils.h>

namespace ErlebAR
{
//-----------------------------------------------------------------------------
// Textures
//-----------------------------------------------------------------------------
class Textures
{
public:
    void load(std::string textureDir)
    {
        texIdBackArrow = loadTexture(textureDir + "back1orange.png", false, false, 1.f);
    }
    void free()
    {
        deleteTexture(texIdBackArrow);
    }

    GLuint texIdBackArrow = 0;
};

//-----------------------------------------------------------------------------
// App appearance
//-----------------------------------------------------------------------------

class Style
{
public:
    ImVec4 transparentColor = {0.f, 0.f, 0.f, 0.f};
    ImVec4 whiteColor       = {1.f, 1.f, 1.f, 1.f};
    //header bar:
    // percental header bar height relative to screen height
    float headerBarPercH = 0.125f;
    // percental header bar text height relative to header bar height
    float headerBarTextH = 0.8f;
    // percental header bar button height relative to header bar height
    float headerBarButtonH = 0.8f;

    ImVec4 headerBarBackgroundColor       = {BFHColors::Gray1Backgr.r,
                                       BFHColors::Gray1Backgr.g,
                                       BFHColors::Gray1Backgr.b,
                                       BFHColors::Gray1Backgr.a};
    ImVec4 headerBarBackgroundTranspColor = {BFHColors::Gray1Backgr.r,
                                             BFHColors::Gray1Backgr.g,
                                             BFHColors::Gray1Backgr.b,
                                             0.5f};

    //ImVec4 headerBarTextColor = {1.f, 1.f, 1.f, 1.f}; //white
    ImVec4 headerBarTextColor = {BFHColors::Orange2Text.r,
                                 BFHColors::Orange2Text.g,
                                 BFHColors::Orange2Text.b,
                                 1.f};

    ImVec4 headerBarBackButtonColor       = headerBarBackgroundColor;
    ImVec4 headerBarBackButtonTranspColor = {0.f, 0.f, 0.f, 0.f};

    //selection gui pressed button color
    ImVec4 headerBarBackButtonPressedColor       = {BFHColors::GrayLogo.r,
                                              BFHColors::GrayLogo.g,
                                              BFHColors::GrayLogo.b,
                                              1.0};
    ImVec4 headerBarBackButtonPressedTranspColor = {1.f, 1.f, 1.f, 0.5f};

    // percental spacing between backbutton text relative to header bar height
    float headerBarSpacingBB2Text = 0.3f;

    //checkbox, slider
    ImVec4 frameBgColor       = headerBarBackgroundTranspColor;
    ImVec4 frameBgActiveColor = headerBarBackgroundTranspColor;

    //buttons:
    // percental button text height relative to button height
    float buttonTextH = 0.7f;
    // percental button rounding relative to screen height
    float buttonRounding = 0.01f;
    //selection gui button color
    ImVec4 buttonColorSelection = {BFHColors::Gray1Backgr.r,
                                   BFHColors::Gray1Backgr.g,
                                   BFHColors::Gray1Backgr.b,
                                   1.0f};
    //selection gui pressed button color
    ImVec4 buttonColorPressedSelection = {BFHColors::GrayLogo.r,
                                          BFHColors::GrayLogo.g,
                                          BFHColors::GrayLogo.b,
                                          0.3f};

    ImVec4 buttonTextColorSelection = {BFHColors::Orange2Text.r,
                                       BFHColors::Orange2Text.g,
                                       BFHColors::Orange2Text.b,
                                       1.f};

    //area pose button
    ImVec4 areaPoseButtonShapeColor = {BFHColors::OrangeGraphic.r,
                                       BFHColors::OrangeGraphic.g,
                                       BFHColors::OrangeGraphic.b,
                                       1.f};

    ImVec4 areaPoseButtonShapeColorPressed = {BFHColors::Gray4Backgr.r,
                                              BFHColors::Gray4Backgr.g,
                                              BFHColors::Gray4Backgr.b,
                                              1.f};
    //the background of the button
    ImVec4 areaPoseButtonColor        = {1.f, 1.f, 1.f, 0.f};
    ImVec4 areaPoseButtonColorPressed = {1.f, 1.f, 1.f, 0.f};
    //percental view triangle width relative to screen heigth
    float areaPoseButtonViewTriangleWidth = 0.1f;
    //percental view triangle length relative to view triangle width
    float areaPoseButtonViewTriangleLength = 1.1f;
    //percental circle radius relative to view triangle width
    float areaPoseButtonCircleRadius = 0.2f;

    //other:
    // text height in mm
    float textStandardHMM = 3.8f;
    // percental heading text height relative to screen height
    float  textHeadingH      = 0.07f;
    ImVec4 textStandardColor = {BFHColors::GrayDark.r,
                                BFHColors::GrayDark.g,
                                BFHColors::GrayDark.b,
                                1.0};

    ImVec4 textHeadingColor = headerBarTextColor;

    ImVec4 backgroundColorPrimary = {BFHColors::Gray5Backgr.r,
                                     BFHColors::Gray5Backgr.g,
                                     BFHColors::Gray5Backgr.b,
                                     BFHColors::Gray5Backgr.a};
    
    ImVec4 backgroundColorWelcome = {BFHColors::Orange2Text.r,
                                     BFHColors::Orange2Text.g,
                                     BFHColors::Orange2Text.b,
                                     BFHColors::Orange2Text.a};

    // percental window padding for content (e.g. about and settings) rel. to screen height
    float windowPaddingContent = 0.03f;
    // percental frame padding for content (e.g. about and settings) rel. to screen height
    float framePaddingContent = 0.02f;
    // percental item spacing for content (e.g. about and settings) rel. to screen height
    float itemSpacingContent = 0.03f;
    //waiting spinner color
    ImVec4 waitingSpinnerMainColor     = whiteColor;
    ImVec4 waitingSpinnerBackDropColor = headerBarBackgroundTranspColor;
};

//-----------------------------------------------------------------------------
// Fonts
//-----------------------------------------------------------------------------
/*!
Every instance of ImGuiWrapper has its own imgui context. But we will share the
fonts using a shared font atlas.
*/
class Fonts
{
public:
    Fonts();
    ~Fonts();

    void load(std::string fontDir, const Style& style, int screenH, int dpi);

    ImFont* headerBar{nullptr};  //font in header bars
    ImFont* standard{nullptr};   //e.g. about, widgets text
    ImFont* heading{nullptr};    //e.g. heading above widgets, about headings
    ImFont* big{nullptr};        //e.g. Welcome screen
    ImFont* tiny{nullptr};       //e.g. log window
    ImFont* selectBtns{nullptr}; //buttons in selection window

    ImFontAtlas* atlas() const { return _atlas; }

private:
    //shared imgui font atlas
    ImFontAtlas* _atlas{nullptr};
};

//-----------------------------------------------------------------------------
// Strings (all visal text in four languages)
//-----------------------------------------------------------------------------
class Strings
{
public:
    virtual ~Strings() {}
    const char* id() const
    {
        return _id.c_str();
    }

    const char* augst() const { return _augst.c_str(); }
    const char* avenches() const { return _avenches.c_str(); }
    const char* bern() const { return _bern.c_str(); }
    const char* biel() const { return _biel.c_str(); }

    const char* settings() const { return _settings.c_str(); }
    const char* about() const { return _about.c_str(); }
    const char* tutorial() const { return _tutorial.c_str(); }
    const char* general() const { return _general.c_str(); }
    const char* generalContent() const { return _generalContent.c_str(); }
    const char* developers() const { return _developers.c_str(); }
    const char* developerNames() const { return _developerNames.c_str(); }

    const char* language() const { return _language.c_str(); }
    const char* develMode() const { return _develMode.c_str(); }

    const char* downloadManager() const { return _download.c_str(); }
    const char* download1() const { return _downloadInfoSize1.c_str(); }
    const char* download2() const { return _downloadInfoSize2.c_str(); }
    const char* downloadButton() const { return _downloadButton.c_str(); }
    const char* downloadSkipButton() const { return _downloadSkipButton.c_str(); }

    const char* cameraStartError() const { return _cameraStartError.c_str(); }

    //info text
    //bern:
    const char* bernInfoHeading1() const { return _bernInfoHeading1.c_str(); }
    const char* bernInfoText1() const { return _bernInfoText1.c_str(); }
    const char* bernInfoHeading2() const { return _bernInfoHeading2.c_str(); }
    const char* bernInfoText2() const { return _bernInfoText2.c_str(); }

    //augst:
    const char* augstTempleHillInfoHeading1() const { return _augstTempleHillInfoHeading1.c_str(); }
    const char* augstTempleHillInfoText1() const { return _augstTempleHillInfoText1.c_str(); }
    const char* augstTempleHillInfoText2() const { return _augstTempleHillInfoText2.c_str(); }
    const char* augstTheaterInfoHeading1() const { return _augstTheaterInfoHeading1.c_str(); }
    const char* augstTheaterInfoText1() const { return _augstTheaterInfoText1.c_str(); }
    const char* augstTheaterInfoText2() const { return _augstTheaterInfoText2.c_str(); }
 
    //avenches:
    const char* avenchesAmphitheaterInfoHeading1() const { return _avenchesAmphitheaterInfoHeading1.c_str(); }
    const char* avenchesAmphitheaterInfoText1() const { return _avenchesAmphitheaterInfoText1.c_str(); }
    const char* avenchesAmphitheaterInfoText2() const { return _avenchesAmphitheaterInfoText2.c_str(); }
    const char* avenchesCigognierInfoHeading1() const { return _avenchesCigognierInfoHeading1.c_str(); }
    const char* avenchesCigognierInfoText1() const { return _avenchesCigognierInfoText1.c_str(); }
    const char* avenchesCigognierInfoText2() const { return _avenchesCigognierInfoText2.c_str(); }
    const char* avenchesTheaterInfoHeading1() const { return _avenchesTheaterInfoHeading1.c_str(); }
    const char* avenchesTheaterInfoText1() const { return _avenchesTheaterInfoText1.c_str(); }
    const char* avenchesTheaterInfoText2() const { return _avenchesTheaterInfoText2.c_str(); }


    //tracking view user guidance
    const char* ugInfoReloc() const { return _ugInfoReloc.c_str(); }
    const char* ugInfoRelocWrongOrient() const { return _ugInfoRelocWrongOrient.c_str(); }
    const char* ugInfoDirArrow() const { return _ugInfoDirArrow.c_str(); }

    void load(std::string fileName);

protected:
    //static void loadString(const cv::FileStorage& fs, const std::string& name, std::string& target);

    std::string _id = "English";

    //selection
    std::string _augst    = "Augst";
    std::string _avenches = "Avenches";
    std::string _bern     = "Bern";
    std::string _biel     = "Biel";
    std::string _settings = "Settings";
    std::string _about    = "About";
    std::string _download = "Download";
    std::string _tutorial = "Tutorial";
    //about
    std::string _general        = "General";
    std::string _generalContent = "Lorem ipsum dolor sit amet";
    std::string _developers     = "Developers";
    std::string _developerNames = "Jan Dellsperger\nLuc Girod\nMichael G�ttlicher";
    //settings
    std::string _language  = "Language";
    std::string _develMode = "Developer mode";
    //errors
    std::string _cameraStartError = "Could not start camera!";
    //download
    std::string _downloadInfoSize1 = "For this location, you need to download";
    std::string _downloadInfoSize2 = ".";
    std::string _downloadSkipButton = "skip";
    std::string _downloadButton = "Download";

    //info text
    //bern:
    std::string _bernInfoHeading1 = "bern heading 1";
    std::string _bernInfoText1    = "bern info 1";
    std::string _bernInfoHeading2 = "bern heading 2";
    std::string _bernInfoText2    = "bern info 2";

    //augst:
    std::string _augstTempleHillInfoHeading1 = "augst temple hill heading 1";
    std::string _augstTempleHillInfoText1    = "augst temple hill text 1";
    std::string _augstTempleHillInfoText2    = "augst temple hill text 2";
    std::string _augstTheaterInfoHeading1    = "augst Theater heading 1";
    std::string _augstTheaterInfoText1       = "augst Theater text 1";
    std::string _augstTheaterInfoText2       = "augst Theater text 2";

    //avenches:
    std::string _avenchesAmphitheaterInfoHeading1 = "avenches amphitheater heading 1";
    std::string _avenchesAmphitheaterInfoText1    = "avenches amphitheater into text 1";
    std::string _avenchesAmphitheaterInfoText2    = "avenches amphitheater into text 2";
    std::string _avenchesCigognierInfoHeading1 = "avenches cigognier heading 1";
    std::string _avenchesCigognierInfoText1    = "avenches cigognier into text 1";
    std::string _avenchesCigognierInfoText2    = "avenches cigognier into text 2";
    std::string _avenchesTheaterInfoHeading1 = "avenches theater heading 1";
    std::string _avenchesTheaterInfoText1    = "avenches theater text 1";
    std::string _avenchesTheaterInfoText2    = "avenches theater text 2";

    //tracking view user guidance
    std::string _ugInfoReloc            = "Trying to relocalize, please move slowly";
    std::string _ugInfoRelocWrongOrient = "You are looking in the wrong direction";
    std::string _ugInfoDirArrow         = "The area is located %.0fm in the direction of the arrow";
};

//-----------------------------------------------------------------------------
// Resources (Strings, Style, shared Textures, shared Fonts)
//-----------------------------------------------------------------------------
class Resources
{
public:
    Resources(const DeviceData& deviceData, int screenH, std::string languageId);
    ~Resources();

    void setLanguageEnglish();
    void setLanguageGerman();
    void setLanguageFrench();
    void setLanguageItalien();

    const Strings& strings() const { return *_currStrings; }
    const Style&   style() { return _style; }
    const Fonts&   fonts() { return _fonts; }

    Textures textures;

    //const std::map<ErlebAR::LocationId, ErlebAR::Location>& locations() { return _locations; }

    const char* stringsEnglishId() const { return stringsEnglish.id(); }
    const char* stringsGermanId() const { return stringsGerman.id(); }
    const char* stringsFrenchId() const { return stringsFrench.id(); }
    const char* stringsItalianId() const { return stringsItalian.id(); }

    void logWinInit();
    void logWinUnInit();
    void logWinDraw();

    //developper helper flags
    bool developerMode      = true;
    bool simulatorMode      = false;
    bool enableUserGuidance = false;
    bool logWinEnabled      = false;

private:
    Strings stringsEnglish;
    Strings stringsGerman;
    Strings stringsFrench;
    Strings stringsItalian;

    //void load(std::string resourceFileName);
    //void save();

    Strings* _currStrings = &stringsEnglish;

    Style _style;
    Fonts _fonts;

    //initialized in function load()
    //std::string _fileName;
    //erlebar locations definition
    //std::map<ErlebAR::LocationId, ErlebAR::Location> _locations;
    //writeable directory, e.g. for logfile
    std::string _writableDir;

    //int _screenW;
    //int _screenH;
};

class Config
{
public:
    Config(const DeviceData& deviceData);
    ~Config();

    void logWinInit();
    void logWinUnInit();
    void logWinDraw();

    void load(std::string resourceFileName);
    void save();

    Resources&                                              resources() { return *_resources; }
    const std::map<ErlebAR::LocationId, ErlebAR::Location>& locations() { return _locations; }

    //developper helper flags
    bool developerMode      = true;
    bool simulatorMode      = false;
    bool enableUserGuidance = false;
    bool logWinEnabled      = false;

    bool useGps     = false;
    bool useARCore  = false;
    bool useWAISlam = false;

private:
    int _screenW;
    int _screenH;

    Resources* _resources = nullptr;

    //initialized in function load()
    std::string _fileName;
    //erlebar locations definition
    std::map<ErlebAR::LocationId, ErlebAR::Location> _locations;

    std::string _languageId;
};

};

#endif //RESOURCES_H
