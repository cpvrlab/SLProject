#include "ErlebAR.h"

namespace ErlebAR
{

Resources::Resources()
{
    //load fonts
}

Resources::~Resources()
{
    //delete fonts
}

void Resources::setLanguageGerman()
{
    currStrings = &stringsGerman;
}
void Resources::setLanguageEnglish()
{
    currStrings = &stringsEnglish;
}
void Resources::setLanguageFrench()
{
    currStrings = &stringsFrench;
}
void Resources::setLanguageItalien()
{
    currStrings = &stringsItalien;
}

StringsEnglish::StringsEnglish()
{
    _settings   = "Settings";
    _about      = "About";
    _tutorial   = "Tutorial";
    _developers = "Developers";
}

StringsGerman::StringsGerman()
{
    _settings   = "Einstellungen";
    _about      = "Info";
    _tutorial   = "Anleitung";
    _developers = "Entwickler";
}

StringsFrench::StringsFrench()
{
    _settings   = "Paramètres";
    _about      = "À propos";
    _tutorial   = "Manuel";
    _developers = "développeur";
}

StringsItalien::StringsItalien()
{
    _settings = "";
    _tutorial = "";
}

};
