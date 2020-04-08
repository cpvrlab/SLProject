#include "ErlebAR.h"

namespace ErlebAR
{
void Resources::setLanguageGerman()
{
    instance().currStrings = &instance().stringsGerman;
}
void Resources::setLanguageEnglish()
{
    instance().currStrings = &instance().stringsEnglish;
}
void Resources::setLanguageFrench()
{
    instance().currStrings = &instance().stringsFrench;
}
void Resources::setLanguageItalien()
{
    instance().currStrings = &instance().stringsItalien;
}
};
