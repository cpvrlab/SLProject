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
// Fonts
//-----------------------------------------------------------------------------
//class Fonts
//{
//public:
//    void load(std::string fontDir)
//    {
//    }
//    void free()
//    {
//    }
//
//private:
//    //shared imgui font atlas
//    //ImFontAtlas* fontAtlas = nullptr;
//};

//-----------------------------------------------------------------------------
// App appearance
//-----------------------------------------------------------------------------

class Style
{
public:
    //header bar:
    // percental header bar height relative to screen height
    float headerBarPercH = 0.15f;
    // percental header bar text height relative to header bar height
    float headerBarTextH = 0.6f;
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
    ImVec4 headerBarTextColor             = {1.f, 1.f, 1.f, 1.f}; //white
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
    // percental window padding for content (e.g. about and settings) rel. to screen height
    float windowPaddingContent = 0.03f;
    // percental frame padding for content (e.g. about and settings) rel. to screen height
    float framePaddingContent = 0.02f;
    // percental item spacing for content (e.g. about and settings) rel. to screen height
    float itemSpacingContent = 0.03f;
};

//-----------------------------------------------------------------------------
// Strings (all visal text in four languages)
//-----------------------------------------------------------------------------
class Strings
{
public:
    Strings();
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
    Resources(std::string resourceFileName, std::string textureDir);
    ~Resources();

    void setLanguageEnglish();
    void setLanguageGerman();
    void setLanguageFrench();
    void setLanguageItalien();

    const Strings& strings() { return *_currStrings; }
    const Style&   style() { return _style; }
    //const Fonts&   fonts() { return _fonts; }

    bool developerMode = false;

    StringsEnglish stringsEnglish;
    StringsGerman  stringsGerman;
    StringsFrench  stringsFrench;
    StringsItalien stringsItalien;

    Textures textures;

private:
    void load(std::string resourceFileName);
    void save();

    Strings* _currStrings = &stringsEnglish;

    Style _style;
    //Fonts _fonts;

    //initialized in function load()
    std::string _fileName;
};
};

#endif //RESOURCES_H
