#ifndef ERLEBAR_H
#define ERLEBAR_H

#include <string>
#include <map>

#include <imgui.h>
#include <SLVec4.h>

#include <DeviceData.h>

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
enum class LocationId
{
    NONE,
    AUGST,
    AVENCHES,
    BIEL,
    BERN
};

const char* mapLocationIdToName(LocationId id);

//erlebar area
enum class AreaId
{
    NONE,
    //AUGST
    AUGST_TEMPLE_HILL_MARKER,
    AUGST_TEMPLE_HILL_THEATER_BOTTOM,
    //AVENCHES
    AVENCHES_AMPHITHEATER,
    AVENCHES_AMPHITHEATER_ENTRANCE,
    AVENCHES_THEATER,
    AVENCHES_CIGOGNIER,
    AVENCHES_TEMPLE,
    //BERN
    BERN_SBB,
    BERN_MILCHGAESSLI,
    //BIEL
    BIEL_SOUTHWALL,
    BIEL_GERECHTIGKEITSBRUNNEN,
    BIEL_JACOB_ROSINUS,
    BIEL_LEUBRINGENBAHN,
    BIEL_RING
};

const char* mapAreaIdToName(AreaId id);

class Area
{
public:
    //Area(AreaId id, int posXPix, int posYPix, float viewAngle);

    AreaId      id;
    const char* name;
    //x position in pixel (only valid for current map image)
    int xPosPix;
    //y position in pixel (only valid for current map image)
    int yPosPix;
    //view angle in degree
    float viewAngleDeg;
    //map name in erlebAR directory
    std::string slamMapFileName;
};

//location description
class Location
{
public:
    LocationId  id = LocationId::NONE;
    const char* name;
    //name of area map image in erlebAR directory
    std::string areaMapImageFileName;
    //map image display pixel width
    int                    dspPixWidth;
    std::map<AreaId, Area> areas;
    //location center wgs84 (for gps user positionioning in map)
};

//get definition of current locations and areas
const Location defineLocationAugst();
const Location defineLocationAvenches();
const Location defineLocationChristoffel();
const Location defineLocationBiel();

const std::map<LocationId, Location> defineLocations();

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

    TEST_RUNNER,

    LOCATION_MAP,
    AREA_INFO,
    AREA_TRACKING,
    HOLD_TRACKING,

    TUTORIAL,
    ABOUT,
    SETTINGS,
    CAMERA_TEST
};

#endif //ERLEBAR_H
