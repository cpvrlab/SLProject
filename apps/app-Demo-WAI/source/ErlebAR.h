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
    NONE = 0,
    AUGST,
    AVENCHES,
    BIEL,
    BERN,
    EVILARD
};

const char* mapLocationIdToName(LocationId id);

//erlebar area
enum class AreaId
{
    NONE = 0,
    //AUGST
    AUGST_TEMPLE_HILL,
    AUGST_THEATER_FRONT,
    //AVENCHES
    AVENCHES_AMPHITHEATER,
    AVENCHES_AMPHITHEATER_ENTRANCE,
    AVENCHES_THEATER,
    AVENCHES_CIGOGNIER,
    AVENCHES_TEMPLE,
    //BERN
    BERN_SBB,
    BERN_SBB_ENTRANCE,
    BERN_MILCHGAESSLI,
    BERN_SPITALGASSE,
    BERN_BUBENBERGPLATZ,
    BERN_CHRISTOFFEL_BRIDGE,
    //BIEL
    BIEL_SOUTHWALL,
    BIEL_GERECHTIGKEITSBRUNNEN,
    BIEL_JACOB_ROSINUS,
    BIEL_LEUBRINGENBAHN,
    BIEL_RING,
    BIEL_OFFICE,
    BIEL_BFH,
    //EVILARD
    EVILARD_ROC2,
    EVILARD_FIREFIGHTERS,
    EVILARD_OFFICE
};

const char* mapAreaIdToName(AreaId id);

class Area
{
public:
    Area() {}
    //Use the constructor so we dont miss new parameter
    Area(AreaId        id,
         SLVec3d       llaPos,
         float         viewAngleDeg,
         SLVec3d       modelOrigin,
         std::string   slamMapFileName,
         std::string   relocAlignImage,
         std::string   vocFileName                 = "calibrations/voc_fbow.bin",
         int           vocLayer                    = 2,
         cv::Size      cameraFrameTargetSize       = {640, 360},
         ExtractorType initializationExtractorType = ExtractorType::ExtractorType_FAST_ORBS_2000,
         ExtractorType relocalizationExtractorType = ExtractorType::ExtractorType_FAST_ORBS_1000,
         ExtractorType trackingExtractorType       = ExtractorType::ExtractorType_FAST_ORBS_1000,
         int           nExtractorLevels            = 2)
      : id(id),
        name(mapAreaIdToName(id)),
        llaPos(llaPos),
        viewAngleDeg(viewAngleDeg),
        modelOrigin(modelOrigin),
        slamMapFileName(slamMapFileName),
        relocAlignImage(relocAlignImage),
        vocFileName(vocFileName),
        vocLayer(vocLayer),
        cameraFrameTargetSize(cameraFrameTargetSize),
        initializationExtractorType(initializationExtractorType),
        relocalizationExtractorType(relocalizationExtractorType),
        trackingExtractorType(trackingExtractorType),
        nExtractorLevels(nExtractorLevels)
    {
    }

    AreaId      id = AreaId::NONE;
    std::string name;
    //area position in WGS84 coordinates
    SLVec3d llaPos;
    //view angle on map image in degree
    float viewAngleDeg;
    //origin of 3d model in WGS84 coordinates
    SLVec3d modelOrigin;
    //map name in erlebAR directory
    std::string slamMapFileName;
    //image of point of view that is shown during user guidance
    std::string relocAlignImage;
    //vocabulary file name
    std::string vocFileName;
    int         vocLayer;
    //camera image size
    cv::Size cameraFrameTargetSize;
    //extractor types
    ExtractorType initializationExtractorType;
    ExtractorType relocalizationExtractorType;
    ExtractorType trackingExtractorType;
    //number of pyramid levels
    int nExtractorLevels;
};

//location description
class Location
{
public:
    Location() {}
    Location(LocationId  id,
             std::string areaMapImageFileName,
             SLVec3d     mapTLLla,
             SLVec3d     mapBRLla,
             int         dspPixWidth,
             std::string geoTiffFileName)
      : id(id),
        name(mapLocationIdToName(id)),
        areaMapImageFileName(areaMapImageFileName),
        mapTLLla(mapTLLla),
        mapBRLla(mapBRLla),
        dspPixWidth(dspPixWidth),
        geoTiffFileName(geoTiffFileName)
    {
    }

    LocationId  id = LocationId::NONE;
    std::string name;
    //name of area map image in erlebAR directory
    std::string areaMapImageFileName;
    //top left image corner in WGS84 (lla)
    SLVec3d mapTLLla = {0, 0, 0};
    //bottom right image corner in WGS84 (lla)
    SLVec3d mapBRLla = {0, 0, 0};
    //map image display pixel width to define standard zoom
    int dspPixWidth;
    //geo tiff file name
    std::string geoTiffFileName;

    std::map<AreaId, Area> areas;
};

//get definition of current locations and areas
const Location defineLocationAugst();
const Location defineLocationAvenches();
const Location defineLocationBern();
const Location defineLocationBiel();
const Location defineLocationEvilard();

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
