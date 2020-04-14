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
    _currStrings = &_stringsGerman;
}
void Resources::setLanguageEnglish()
{
    _currStrings = &_stringsEnglish;
}
void Resources::setLanguageFrench()
{
    _currStrings = &_stringsFrench;
}
void Resources::setLanguageItalien()
{
    _currStrings = &_stringsItalien;
}

Strings::Strings()
{
    _developerNames = "Jan Dellsperger\nLuc Girod\nMichael Göttlicher";
}

StringsEnglish::StringsEnglish()
{
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
