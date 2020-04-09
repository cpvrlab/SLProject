#ifndef ERLEBAR_H
#define ERLEBAR_H

#include <SLVec4.h>
#include <DeviceData.h>
#include <sm/Event.h>
#include <imgui.h>
#include <string>

//bfh colors
namespace BFHColors
{
const SLVec4f GrayPrimary   = {105.f / 255.f, 125.f / 255.f, 145.f / 255.f, 1.f};
const SLVec4f OrangePrimary = {250.f / 255.f, 165.f / 255.f, 0.f / 255.f, 1.f};
const SLVec4f GrayLogo      = {105.f / 255.f, 125.f / 255.f, 145.f / 255.f, 1.f};
const SLVec4f OrangeLogo    = {250.f / 255.f, 19.f / 255.f, 0.f / 255.f, 1.f};
const SLVec4f GrayText      = {75.f / 255.f, 100.f / 255.f, 125.f / 255.f, 1.f};
const SLVec4f Gray1         = {100.f / 255.f, 120.f / 255.f, 139.f / 255.f, 1.f};
const SLVec4f Gray2         = {162.f / 255.f, 174.f / 255.f, 185.f / 255.f, 1.f};
const SLVec4f Gray3         = {193.f / 255.f, 201.f / 255.f, 209.f / 255.f, 1.f};
const SLVec4f Gray4         = {224.f / 255.f, 228.f / 255.f, 232.f / 255.f, 1.f};
const SLVec4f Gray5         = {239.f / 255.f, 241.f / 255.f, 243.f / 255.f, 1.f};
const SLVec4f Orange1Text   = {189.f / 255.f, 126.f / 255.f, 0.f / 255.f, 1.f};
const SLVec4f Orange2Text   = {255.f / 255.f, 203.f / 255.f, 62.f / 255.f, 1.f};
const SLVec4f OrangeGraphic = {250.f / 255.f, 165.f / 255.f, 0.f / 255.f, 1.f};
const SLVec4f GrayDark      = {60.f / 255.f, 60.f / 255.f, 60.f / 255.f, 1.f};
};

namespace ErlebAR
{
//erlebar location
enum class Location
{
    NONE,
    AUGST,
    AVANCHES,
    BIEL,
    CHRISTOFFEL
};

//erlebar area
enum class Area
{
    NONE,
    AUGST_FORUM_MARKER,
    //..
    BIEL_LEUBRINGENBAHN
    //..
};

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
    float  headerBarTextH           = 0.7f;
    ImVec4 headerBarBackgroundColor = {BFHColors::Gray2.r,
                                       BFHColors::Gray2.g,
                                       BFHColors::Gray2.b,
                                       BFHColors::Gray2.a};
    ImVec4 headerBarTextColor       = {1.f, 1.f, 1.f, 1.f}; //white
                                                            //selection gui button color
    ImVec4 headerBarBackButtonColor = {BFHColors::GrayDark.r,
                                       BFHColors::GrayDark.g,
                                       BFHColors::GrayDark.b,
                                       1.0};
    //selection gui pressed button color
    ImVec4 headerBarBackButtonPressedColor = {BFHColors::GrayLogo.r,
                                              BFHColors::GrayLogo.g,
                                              BFHColors::GrayLogo.b,
                                              1.0};
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
    // percental item spacing for content (e.g. about and settings) rel. to screen height
    float itemSpacingContent = 0.03f;
};

//-----------------------------------------------------------------------------
// Language/Text
//-----------------------------------------------------------------------------

class Strings
{
public:
    Strings();
    const char* settings() const { return _settings.c_str(); }
    const char* about() const { return _about.c_str(); }
    const char* tutorial() const { return _tutorial.c_str(); }
    const char* general() const { return _general.c_str(); }
    const char* generalContent() const { return _generalContent.c_str(); }
    const char* developers() const { return _developers.c_str(); }
    const char* developerNames() const { return _developerNames.c_str(); }

protected:
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

class Fonts
{
public:
    ImFontAtlas* fontAtlas = nullptr;
};

class Resources
{
public:
    Resources();
    ~Resources();

    void setLanguageEnglish();
    void setLanguageGerman();
    void setLanguageFrench();
    void setLanguageItalien();

    const Strings& strings() { return *_currStrings; }
    const Style&   style() { return _style; }
    const Fonts&   fonts() { return _fonts; }

private:
    StringsEnglish _stringsEnglish;
    StringsGerman  _stringsGerman;
    StringsFrench  _stringsFrench;
    StringsItalien _stringsItalien;
    Strings*       _currStrings = &_stringsEnglish;

    Style _style;
    Fonts _fonts;
};

}; //namespace ErlebAR

//erlebar app state machine stateIds
enum class StateId
{
    IDLE = 0,
    INIT,
    WELCOME,
    DESTROY,
    SELECTION,

    START_TEST,
    TEST,
    HOLD_TEST,
    RESUME_TEST,

    START_ERLEBAR,
    MAP_VIEW,
    AREA_TRACKING,

    TUTORIAL,
    ABOUT,
    SETTINGS,
    CAMERA_TEST
};

//-----------------------------------------------------------------------------
// EventData
//-----------------------------------------------------------------------------
class InitData : public sm::EventData
{
public:
    InitData(DeviceData deviceData)
      : deviceData(deviceData)
    {
    }
    const DeviceData deviceData;
};

class ErlebarData : public sm::EventData
{
public:
    ErlebarData(ErlebAR::Location location)
      : location(location)
    {
    }
    const ErlebAR::Location location;
};

class AreaData : public sm::EventData
{
public:
    AreaData(ErlebAR::Area area)
      : area(area)
    {
    }
    const ErlebAR::Area area;
};

//-----------------------------------------------------------------------------
// Event
//-----------------------------------------------------------------------------
class InitEvent : public sm::Event
{
public:
    InitEvent(int            scrWidth,
              int            scrHeight,
              float          scr2fbX,
              float          scr2fbY,
              int            dpi,
              AppDirectories dirs)
    {
        enableTransition((unsigned int)StateId::IDLE,
                         (unsigned int)StateId::INIT);

        DeviceData deviceData(scrWidth, scrHeight, scr2fbX, scr2fbY, dpi, dirs);
        _eventData = new InitData(deviceData);
    }
};

class GoBackEvent : public sm::Event
{
public:
    GoBackEvent()
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::MAP_VIEW,
                         (unsigned int)StateId::SELECTION);
        enableTransition((unsigned int)StateId::AREA_TRACKING,
                         (unsigned int)StateId::MAP_VIEW);
        enableTransition((unsigned int)StateId::TUTORIAL,
                         (unsigned int)StateId::SELECTION);
        enableTransition((unsigned int)StateId::ABOUT,
                         (unsigned int)StateId::SELECTION);
        enableTransition((unsigned int)StateId::SETTINGS,
                         (unsigned int)StateId::SELECTION);
        enableTransition((unsigned int)StateId::CAMERA_TEST,
                         (unsigned int)StateId::SELECTION);
        enableTransition((unsigned int)StateId::TEST,
                         (unsigned int)StateId::SELECTION);
    }
};

class DestroyEvent : public sm::Event
{
public:
    DestroyEvent()
    {
        enableTransition((unsigned int)StateId::INIT,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::START_TEST,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::TEST,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::START_ERLEBAR,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::MAP_VIEW,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::AREA_TRACKING,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::TUTORIAL,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::ABOUT,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::CAMERA_TEST,
                         (unsigned int)StateId::DESTROY);
    }
};

class DoneEvent : public sm::Event
{
public:
    DoneEvent()
    {

        enableTransition((unsigned int)StateId::DESTROY,
                         (unsigned int)StateId::IDLE);
        enableTransition((unsigned int)StateId::INIT,
                         (unsigned int)StateId::WELCOME);
        enableTransition((unsigned int)StateId::WELCOME,
                         (unsigned int)StateId::SELECTION);
        enableTransition((unsigned int)StateId::START_ERLEBAR,
                         (unsigned int)StateId::MAP_VIEW);
        enableTransition((unsigned int)StateId::START_TEST,
                         (unsigned int)StateId::TEST);
        enableTransition((unsigned int)StateId::RESUME_TEST,
                         (unsigned int)StateId::TEST);
    }
};

class StartErlebarEvent : public sm::Event
{
public:
    StartErlebarEvent(ErlebAR::Location location)
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::START_ERLEBAR);

        _eventData = new ErlebarData(location);
    }
};

class AreaSelectedEvent : public sm::Event
{
public:
    AreaSelectedEvent(ErlebAR::Area area)
    {
        enableTransition((unsigned int)StateId::MAP_VIEW,
                         (unsigned int)StateId::AREA_TRACKING);

        _eventData = new AreaData(area);
    }
};

class StartTutorialEvent : public sm::Event
{
public:
    StartTutorialEvent()
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::TUTORIAL);
    }
};

class ShowAboutEvent : public sm::Event
{
public:
    ShowAboutEvent()
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::ABOUT);
    }
};

class ShowSettingsEvent : public sm::Event
{
public:
    ShowSettingsEvent()
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::SETTINGS);
    }
};

class StartCameraTestEvent : public sm::Event
{
public:
    StartCameraTestEvent()
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::CAMERA_TEST);
    }
};

class StartTestEvent : public sm::Event
{
public:
    StartTestEvent()
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::START_TEST);
    }
};

class HoldEvent : public sm::Event
{
public:
    HoldEvent()
    {
        enableTransition((unsigned int)StateId::TEST,
                         (unsigned int)StateId::HOLD_TEST);
    }
};

class ResumeEvent : public sm::Event
{
public:
    ResumeEvent()
    {
        enableTransition((unsigned int)StateId::HOLD_TEST,
                         (unsigned int)StateId::RESUME_TEST);
    }
};

#endif //ERLEBAR_H
