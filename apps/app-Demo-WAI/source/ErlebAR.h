#ifndef ERLEBAR_H
#define ERLEBAR_H

#include <string>
#include <map>

#include <imgui.h>
#include <SLVec4.h>

#include <DeviceData.h>
#include <FeatureExtractorFactory.h>

//bfh colors
namespace BFHColors
{
//primary colors
const SLVec4f GrayPrimary   = {105.f / 255.f, 125.f / 255.f, 145.f / 255.f, 1.f};
const SLVec4f OrangePrimary = {250.f / 255.f, 165.f / 255.f, 0.f / 255.f, 1.f};
//logo colors
const SLVec4f GrayLogo   = {105.f / 255.f, 125.f / 255.f, 145.f / 255.f, 1.f};
const SLVec4f OrangeLogo = {250.f / 255.f, 195.f / 255.f, 0.f / 255.f, 1.f};
//gray for text
const SLVec4f GrayText = {75.f / 255.f, 100.f / 255.f, 125.f / 255.f, 1.f};
//grays for backgrounds
const SLVec4f Gray1Backgr = {100.f / 255.f, 120.f / 255.f, 139.f / 255.f, 1.f};
const SLVec4f Gray2Backgr = {162.f / 255.f, 174.f / 255.f, 185.f / 255.f, 1.f};
const SLVec4f Gray3Backgr = {193.f / 255.f, 201.f / 255.f, 209.f / 255.f, 1.f};
const SLVec4f Gray4Backgr = {224.f / 255.f, 228.f / 255.f, 232.f / 255.f, 1.f};
const SLVec4f Gray5Backgr = {239.f / 255.f, 241.f / 255.f, 243.f / 255.f, 1.f};
//Orange for text, e.g. title or hover effect
const SLVec4f Orange1TitleOrHover = {189.f / 255.f, 126.f / 255.f, 0.f / 255.f, 1.f};
//Orange for text, e.g. negative text on dark background or hover effect
const SLVec4f Orange2Text = {255.f / 255.f, 203.f / 255.f, 62.f / 255.f, 1.f};
//orange for graphical elements
const SLVec4f OrangeGraphic = {250.f / 255.f, 165.f / 255.f, 0.f / 255.f, 1.f};
//for intranet, for combinartion with secondary colors
const SLVec4f GrayDark = {60.f / 255.f, 60.f / 255.f, 60.f / 255.f, 1.f};

//secondary colors
//green:
const SLVec4f GreenDark  = {85.f / 255.f, 100.f / 255.f, 85.f / 255.f, 1.f};
const SLVec4f GreenMean  = {105.f / 255.f, 150.f / 255.f, 115.f / 255.f, 1.f};
const SLVec4f GreenLight = {140.f / 255.f, 175.f / 255.f, 130.f / 255.f, 1.f};
//blue:
const SLVec4f BlueDark   = {80.f / 255.f, 110.f / 255.f, 150.f / 255.f, 1.f};
const SLVec4f BlueMean   = {105.f / 255.f, 155.f / 255.f, 190.f / 255.f, 1.f};
const SLVec4f BlueLight  = {135.f / 255.f, 185.f / 255.f, 200.f / 255.f, 1.f};
const SLVec4f BlueImgui1 = {0.24f, 0.52f, 0.88f, 1.00f};
//violett
//ocker

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
    NONE = 0,
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
    BIEL_RING,
    BIEL_OFFICE
};

const char* mapAreaIdToName(AreaId id);

class Area
{
public:
    //Area(AreaId id, int posXPix, int posYPix, float viewAngle);

    AreaId      id = AreaId::NONE;
    std::string name;
    //x position in pixel (only valid for current map image)
    int xPosPix = 0;
    //y position in pixel (only valid for current map image)
    int yPosPix = 0;
    //view angle in degree
    float viewAngleDeg = 0.f;
    //map name in erlebAR directory
    std::string slamMapFileName;
    //WaiSlam extractor types
    ExtractorType initializationExtractorType = ExtractorType::ExtractorType_FAST_ORBS_2000;
    ExtractorType relocalizationExtractorType = ExtractorType::ExtractorType_FAST_ORBS_1000;
    ExtractorType trackingExtractorType       = ExtractorType::ExtractorType_FAST_ORBS_1000;
    int           nExtractorLevels            = 2;
    //camera image size
    cv::Size cameraFrameTargetSize = {640, 360};
    //crop the camera image to fit screen aspect ratio
    bool cameraFrameCropToScreen = false;
};

//location description
class Location
{
public:
    LocationId  id = LocationId::NONE;
    std::string name;
    //name of area map image in erlebAR directory
    std::string areaMapImageFileName;
    //top left image corner in WGS84 (lla)
    SLVec3d mapTLLla = {0, 0, 0};
    //bottom right image corner in WGS84 (lla)
    SLVec3d mapBRLla = {0, 0, 0};

    //map image display pixel width
    int                    dspPixWidth;
    std::map<AreaId, Area> areas;
    //location center wgs84 (for gps user positionioning in map)
};

//get definition of current locations and areas
const Location defineLocationAugst();
const Location defineLocationAvenches();
const Location defineLocationBern();
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

    TEST_RUNNER,

    LOCATION_MAP,
    AREA_INFO,
    AREA_TRACKING,
    HOLD_TRACKING,

    TUTORIAL,
    ABOUT,
    SETTINGS,
    CAMERA_TEST,
    SENSOR_TEST
};

#endif //ERLEBAR_H
