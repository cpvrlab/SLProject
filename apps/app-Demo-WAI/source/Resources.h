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
        texIdBackArrow = loadTexture(textureDir + "back1white.png", false, false, 1.f);
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
    //header bar:
    // percental header bar height relative to screen height
    float headerBarPercH = 0.1f;
    // percental header bar text height relative to header bar height
    float headerBarTextH = 0.8f;
    // percental header bar button height relative to header bar height
    float headerBarButtonH = 0.8f;

    ImVec4 headerBarBackgroundColor       = {BFHColors::Gray2.r,
                                       BFHColors::Gray2.g,
                                       BFHColors::Gray2.b,
                                       BFHColors::Gray2.a};
    ImVec4 headerBarBackgroundTranspColor = {BFHColors::Gray2.r,
                                             BFHColors::Gray2.g,
                                             BFHColors::Gray2.b,
                                             0.2};

    ImVec4 headerBarTextColor = {1.f, 1.f, 1.f, 1.f}; //white
                                                      //selection gui button color
    ImVec4 headerBarBackButtonColor       = {BFHColors::GrayDark.r,
                                       BFHColors::GrayDark.g,
                                       BFHColors::GrayDark.b,
                                       0.0};
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
    ImVec4 buttonColorSelection = {BFHColors::OrangePrimary.r,
                                   BFHColors::OrangePrimary.g,
                                   BFHColors::OrangePrimary.b,
                                   0.3};
    //selection gui pressed button color
    ImVec4 buttonColorPressedSelection = {BFHColors::GrayLogo.r,
                                          BFHColors::GrayLogo.g,
                                          BFHColors::GrayLogo.b,
                                          0.3};

    //area pose button
    ImVec4 areaPoseButtonShapeColor        = {1.f, 0.f, 0.f, 0.5f};
    ImVec4 areaPoseButtonShapeColorPressed = {1.f, 1.f, 1.f, 1.f};
    ImVec4 areaPoseButtonColor             = {1.f, 1.f, 1.f, 0.f};
    ImVec4 areaPoseButtonColorPressed      = {1.f, 1.f, 1.f, 0.f};
    //percental view triangle width relative to screen heigth
    float areaPoseButtonViewTriangleWidth = 0.1f;
    //percental view triangle length relative to view triangle width
    float areaPoseButtonViewTriangleLength = 1.1f;
    //percental circle radius relative to view triangle width
    float areaPoseButtonCircleRadius = 0.2f;

    //other:
    // percental standard text height relative to screen height
    float textStandardH = 0.05f;
    // percental heading text height relative to screen height
    float  textHeadingH      = 0.07f;
    ImVec4 textStandardColor = {BFHColors::GrayDark.r,
                                BFHColors::GrayDark.g,
                                BFHColors::GrayDark.b,
                                1.0};
    ImVec4 textHeadingColor  = {1.f, 1.f, 1.f, 1.f};

    ImVec4 backgroundColorPrimary = {BFHColors::OrangePrimary.r,
                                     BFHColors::OrangePrimary.g,
                                     BFHColors::OrangePrimary.b,
                                     BFHColors::OrangePrimary.a};
    ImVec4 transparentColor       = {0.f, 0.f, 0.f, 0.f};
    ImVec4 whiteColor             = {1.f, 1.f, 1.f, 1.f};
    // percental window padding for content (e.g. about and settings) rel. to screen height
    float windowPaddingContent = 0.03f;
    // percental frame padding for content (e.g. about and settings) rel. to screen height
    float framePaddingContent = 0.02f;
    // percental item spacing for content (e.g. about and settings) rel. to screen height
    float itemSpacingContent = 0.03f;
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

    void load(std::string fontDir, const Style& style, int screenH);

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
    Strings();
    virtual ~Strings() {}
    const char* id() const { return _id.c_str(); }

    const char* settings() const { return _settings.c_str(); }
    const char* about() const { return _about.c_str(); }
    const char* tutorial() const { return _tutorial.c_str(); }
    const char* general() const { return _general.c_str(); }
    const char* generalContent() const { return _generalContent.c_str(); }
    const char* developers() const { return _developers.c_str(); }
    const char* developerNames() const { return _developerNames.c_str(); }

    const char* language() const { return _language.c_str(); }
    const char* develMode() const { return _develMode.c_str(); }

protected:
    std::string _id;

    //selection
    std::string _settings;
    std::string _about;
    std::string _tutorial;
    //about
    std::string _general;
    std::string _generalContent;
    std::string _developers;
    std::string _developerNames;
    //settings
    std::string _language;
    std::string _develMode;
};

class StringsEnglish : public Strings
{
public:
    StringsEnglish();
};

class StringsGerman : public Strings
{
public:
    StringsGerman();
};

class StringsFrench : public Strings
{
public:
    StringsFrench();
};

class StringsItalien : public Strings
{
public:
    StringsItalien();
};

//-----------------------------------------------------------------------------
// Resources (Strings, Style, shared Textures, shared Fonts)
//-----------------------------------------------------------------------------
class Resources
{
public:
    Resources(int         screenWidth,
              int         screenHeight,
              std::string writableDir,
              std::string textureDir,
              std::string fontDir);
    ~Resources();

    void setLanguageEnglish();
    void setLanguageGerman();
    void setLanguageFrench();
    void setLanguageItalien();

    const Strings& strings() { return *_currStrings; }
    const Style&   style() { return _style; }
    const Fonts&   fonts() { return _fonts; }

    bool developerMode = false;

    StringsEnglish stringsEnglish;
    StringsGerman  stringsGerman;
    StringsFrench  stringsFrench;
    StringsItalien stringsItalien;

    Textures textures;

    const std::map<ErlebAR::LocationId, ErlebAR::Location>& locations() { return _locations; }

    bool logWinEnabled = false;
    void logWinInit();
    void logWinUnInit();
    void logWinDraw();

private:
    void load(std::string resourceFileName);
    void save();

    Strings* _currStrings = &stringsEnglish;

    Style _style;
    Fonts _fonts;

    //initialized in function load()
    std::string _fileName;
    //erlebar locations definition
    std::map<ErlebAR::LocationId, ErlebAR::Location> _locations;
    //writeable directory, e.g. for logfile
    std::string _writableDir;

    int _screenW;
    int _screenH;
};
};

#endif //RESOURCES_H
