#ifndef ERLEBAR_H
#define ERLEBAR_H

enum class AppMode
{
    CAMERA_TEST = 0,
    TEST,
    AUGST,
    AVANCHES,
    CHRISTOFFELTOWER,
    BIEL,
    NONE
};

enum class Location
{
    NONE,
    AUGST,
    AVANCHES,
    BIEL,
    CHRISTOFFEL
};

enum class Area
{
    NONE,
    AUGST_FORUM_MARKER,
    //..
    BIEL_LEUBRINGENBAHN
    //..
};

#endif //ERLEBAR_H
