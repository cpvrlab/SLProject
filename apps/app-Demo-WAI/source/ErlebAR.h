#ifndef ERLEBAR_H
#define ERLEBAR_H

#include <SLVec4.h>

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

#endif //ERLEBAR_H
