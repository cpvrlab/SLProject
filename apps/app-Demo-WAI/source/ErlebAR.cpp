#include "ErlebAR.h"

namespace ErlebAR
{

const char* mapLocationIdToName(LocationId id)
{
    switch (id)
    {
        case LocationId::NONE:
            return "Undefined location";
        case LocationId::AUGST:
            return "Augst";
        case LocationId::AVENCHES:
            return "Avenches";
        case LocationId::BIEL:
            return "Biel";
        case LocationId::BERN:
            return "Bern";
        case LocationId::EVILARD:
            return "Evilard";
        default:
            return "Missing id to name mapping!";
    }
}

const char* mapAreaIdToName(AreaId id)
{
    switch (id)
    {
        case AreaId::NONE:
            return "Undefined area";
        //augst
        case AreaId::AUGST_TEMPLE_HILL:
            return "Temple";
        case AreaId::AUGST_THEATER_FRONT:
            return "Theater";
        case AreaId::AUGST_THEATER_SIDE:
            return "Theater-Side";
        case AreaId::AUGST_THEATER_RANKS:
            return "Theater-Ranks";
        //avenches
        case AreaId::AVENCHES_AMPHITHEATER:
            return "Amphitheater";
        case AreaId::AVENCHES_AMPHITHEATER_ENTRANCE:
            return "Amphitheater-Entrance";
        case AreaId::AVENCHES_CIGOGNIER:
            return "Cigognier";
        case AreaId::AVENCHES_THEATER:
            return "Theater";
            //christoffel
        case AreaId::BERN_MILCHGAESSLI:
            return "Milchgaessli";
        case AreaId::BERN_SBB:
            return "Sbb";
        case AreaId::BERN_SBB_ENTRANCE:
            return "Sbb Entrance";
        case AreaId::BERN_BUBENBERGPLATZ:
            return "Bubenbergplatz";
        case AreaId::BERN_SPITALGASSE:
            return "Spitalgasse";
        case AreaId::BERN_CHRISTOFFEL_BRIDGE:
            return "Bridge Christoffeltower";
        //biel
        case AreaId::BIEL_BFH:
            return "BFH";
        case AreaId::BIEL_GERECHTIGKEITSBRUNNEN:
            return "Gerechtigkeitsbrunnen";
        case AreaId::BIEL_JACOB_ROSINUS:
            return "Jacob-Rosinus";
        case AreaId::BIEL_LEUBRINGENBAHN:
            return "Leubringenbahn";
        case AreaId::BIEL_RING:
            return "Ring";
        case AreaId::BIEL_SOUTHWALL:
            return "Southwall";
        case AreaId::BIEL_OFFICE:
            return "Office";
        //evilard
        case AreaId::EVILARD_ROC2:
            return "Roc 2";
        case AreaId::EVILARD_FIREFIGHTERS:
            return "Firefighters";
        case AreaId::EVILARD_OFFICE:
            return "Office";
        default:
            return "Missing id to name mapping!";
    }
}

const Location defineLocationAugst()
{
    Location loc(LocationId::AUGST,
                 "locations/augst/locationMapImgAugst.jpg",
                 {47.53443, 7.71890, 282.6},
                 {47.53194, 7.72524, 282.6},
                 2000,
                 "models/augst/DTM-Theater-Tempel-WGS84.tif",
                 "https://pallas.ti.bfh.ch/erlebar/models/augst.zip",
                 "augst");

    {
        Area area(AreaId::AUGST_TEMPLE_HILL,                                                                                    //id
                  {47.53315, 7.72089, 293.2},                                                                                   //llaPos
                  120,                                                                                                          //map viewing angle
                  {47.53319, 7.72207, 282.6},                                                                                   //modelOrigin
                  "locations/augst/templeHill-marker/maps/map_20201119-131033_Augst_templeHill-marker_FAST-ORBS-1000_2.waimap", //slamMapFileName
                  "dummy.jpg",                                                                                                  //relocAlignImage,
                  "calibrations/voc_fbow.bin",                                                                                  //vocFileName,
                  2,                                                                                                            //vocLayer,
                  //{640, 360},                                                                                                   //cameraFrameTargetSize
                  ExtractorType::ExtractorType_FAST_ORBS_2000,                                                                  //initializationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000,                                                                  //relocalizationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000,                                                                  //trackingExtractorType
                  2);                                                                                                           //nExtractorLevels
        loc.areas[area.id] = area;
    }
    {
        Area area(AreaId::AUGST_THEATER_FRONT,                 //id
                  {47.53309, 7.72159, 285.3},                  //llaPos
                  -18,                                         //map viewing angle
                  {47.53319, 7.72207, 282.6},                  //modelOrigin
                  "",                                          //slamMapFileName
                  "dummy.jpg",                                 //relocAlignImage,
                  "calibrations/voc_fbow.bin",                 //vocFileName,
                  2,                                           //vocLayer,
                  //{640, 360},                                  //cameraFrameTargetSize
                  ExtractorType::ExtractorType_FAST_ORBS_2000, //initializationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000, //relocalizationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000, //trackingExtractorType
                  2);
        loc.areas[area.id] = area;
    }
    {
        Area area(AreaId::AUGST_THEATER_RANKS,                 //id
                  {47.53298, 7.72217, 290.9},                  //llaPos
                  200,                                         //map viewing angle
                  {47.53319, 7.72207, 282.6},                  //modelOrigin
                  "",                                          //slamMapFileName
                  "dummy.jpg",                                 //relocAlignImage,
                  "calibrations/voc_fbow.bin",                 //vocFileName,
                  2,                                           //vocLayer,
                  //{640, 360},                                  //cameraFrameTargetSize
                  ExtractorType::ExtractorType_FAST_ORBS_2000, //initializationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000, //relocalizationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000, //trackingExtractorType
                  2);
        loc.areas[area.id] = area;
    }
    {
        Area area(AreaId::AUGST_THEATER_SIDE,                  //id
                  {47.53273, 7.72179, 286.4},                  //llaPos
                  260,                                         //map viewing angle
                  {47.53319, 7.72207, 282.6},                  //modelOrigin
                  "",                                          //slamMapFileName
                  "dummy.jpg",                                 //relocAlignImage,
                  "calibrations/voc_fbow.bin",                 //vocFileName,
                  2,                                           //vocLayer,
                  //{640, 360},                                                                                                   //cameraFrameTargetSize
                  ExtractorType::ExtractorType_FAST_ORBS_2000, //initializationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000, //relocalizationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000, //trackingExtractorType
                  2);
        loc.areas[area.id] = area;
    }
    return loc;
}
const Location defineLocationAvenches()
{
    Location loc(LocationId::AVENCHES,
                 "locations/avenches/locationMapImgAvenches.jpg",
                 {46.88264, 7.04148, 455.0},
                 {46.87954, 7.04983, 455.0},
                 2046,
                 "models/avenches/DTM-Aventicum-WGS84.tif",
                 "https://pallas.ti.bfh.ch/erlebar/models/avenches.zip",
                 "avenches");

    {
        Area area(AreaId::AVENCHES_AMPHITHEATER_ENTRANCE,                                                                                          //id
                  {46.88120, 7.04368, 461.43},                                                                                                     //llaPos
                  170,                                                                                                                             //map viewing angle
                  {46.88102, 7.04263, 461.43},                                                                                                     //modelOrigin
                  "locations/avenches/amphitheaterEntrance/maps/stable-map_20200914-185624_avenches_amphitheaterEntrance_FAST-ORBS-2000_2.waimap", //slamMapFileName
                  "locations/avenches/amphitheaterEntrance/amphitheaterEntrance-reloc-align-img.jpg",                                              //relocAlignImage,
                  "calibrations/voc_fbow.bin",                                                                                                     //vocFileName,
                  2,                                                                                                                               //vocLayer,
                  //{640, 360},                                                                                                                      //cameraFrameTargetSize
                  ExtractorType::ExtractorType_FAST_ORBS_2000,                                                                                     //initializationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000,                                                                                     //relocalizationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000,                                                                                     //trackingExtractorType
                  2);
        loc.areas[area.id] = area;
    }
    {
        Area area(AreaId::AVENCHES_AMPHITHEATER,                                                                                    //id
                  {46.88102, 7.04263, 461.43},                                                                                      //llaPos
                  -18,                                                                                                              //map viewing angle
                  {46.88102, 7.04263, 461.43},                                                                                      //modelOrigin
                  "locations/avenches/amphitheater/maps/release-map_20200930-160451_avenches_amphitheater_FAST-ORBS-2000_2.waimap", //slamMapFileName
                  "locations/avenches/amphitheater/amphitheater-reloc-align-img.jpg",                                               //relocAlignImage,
                  "calibrations/voc_fbow.bin",                                                                                      //vocFileName,
                  2,                                                                                                                //vocLayer,
                  //{640, 360},                                                                                                       //cameraFrameTargetSize
                  ExtractorType::ExtractorType_FAST_ORBS_2000,                                                                      //initializationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000,                                                                      //relocalizationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000,                                                                      //trackingExtractorType
                  2);
        loc.areas[area.id] = area;
    }
    {
        Area area(AreaId::AVENCHES_CIGOGNIER,                                                                                     //id
                  {46.88146, 7.04645, 450.95},                                                                                    //llaPos
                  -140,                                                                                                           //map viewing angle
                  {46.88146, 7.04645, 450.95},                                                                                    //modelOrigin
                  "locations/avenches/cigonier-marker/maps/map_20201007-215134_Avenches_Cigonier-marker_FAST-ORBS-2000_2.waimap", //slamMapFileName
                  "dummy.jpg",                                                                                                    //relocAlignImage,
                  "calibrations/voc_fbow.bin",                                                                                    //vocFileName,
                  2,                                                                                                              //vocLayer,
                  //{640, 360},                                                                                                     //cameraFrameTargetSize
                  ExtractorType::ExtractorType_FAST_ORBS_2000,                                                                    //initializationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000,                                                                    //relocalizationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000,                                                                    //trackingExtractorType
                  2);
        loc.areas[area.id] = area;
    }
    {
        Area area(AreaId::AVENCHES_THEATER,                                                                               //id
                  {46.88029, 7.04880, 454.95},                                                                            //llaPos
                  50,                                                                                                     //map viewing angle
                  {46.88029, 7.04876, 454.95},                                                                            //modelOrigin
                  "locations/avenches/theater/maps/release-map_20200930-154707_avenches_theater_FAST-ORBS-2000_2.waimap", //slamMapFileName
                  "locations/avenches/theater/theater-reloc-align-img.jpg",                                               //relocAlignImage,
                  "calibrations/voc_fbow.bin",                                                                            //vocFileName,
                  2,                                                                                                      //vocLayer,
                  //{640, 360},                                                                                             //cameraFrameTargetSize
                  ExtractorType::ExtractorType_FAST_ORBS_2000,                                                            //initializationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000,                                                            //relocalizationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000,                                                            //trackingExtractorType
                  2);
        loc.areas[area.id] = area;
    }
    return loc;
}
const Location defineLocationBern()
{
    Location loc(LocationId::BERN,
                 "locations/bern/locationMapImgBern.jpg",
                 {46.94885, 7.43808, 542.0},
                 {46.94701, 7.44290, 542.0},
                 2080,
                 "models/bern/DEM-Bern-2600_1199-WGS84.tif",
                 "https://pallas.ti.bfh.ch/erlebar/models/bern.zip",
                 "bern");

    {
        Area area(AreaId::BERN_MILCHGAESSLI,                                                                                    //id
                  {46.94839, 7.43973, 541.2},                                                                                   //llaPos
                  60,                                                                                                           //map viewing angle
                  {46.947629, 7.440754, 542.2},                                                                                 //modelOrigin
                  "locations/bern/milchgaessli/maps/DEVELOPMENT-map_20201118-111850_bern_milchgaessli_FAST-ORBS-1000_2.waimap", //slamMapFileName
                  "dummy.jpg",                                                                                                  //relocAlignImage,
                  "calibrations/voc_fbow.bin",                                                                                  //vocFileName,
                  2,                                                                                                            //vocLayer,
                  //{640, 360},                                                                                                   //cameraFrameTargetSize
                  ExtractorType::ExtractorType_FAST_ORBS_2000,                                                                  //initializationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000,                                                                  //relocalizationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000,                                                                  //trackingExtractorType
                  2);
        loc.areas[area.id] = area;
    }

    {
        Area area(AreaId::BERN_SPITALGASSE,                    //id
                  {46.94783, 7.44089, 542.1},                  //llaPos 46.94785, 7.44099     46.9478469, 7.4409524
                  140,                                         //map viewing angle
                  {46.94762, 7.44074, 542.2},                  //modelOrigin 46.94762, 7.44074       46.947726, 7.4409616
                  "",                                          //slamMapFileName
                  "dummy.jpg",                                 //relocAlignImage,
                  "calibrations/voc_fbow.bin",                 //vocFileName,
                  2,                                           //vocLayer,
                  //{640, 360},                                  //cameraFrameTargetSize
                  ExtractorType::ExtractorType_FAST_ORBS_2000, //initializationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000, //relocalizationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000, //trackingExtractorType
                  2);
        loc.areas[area.id] = area;
    }

    {
        Area area(AreaId::BERN_BUBENBERGPLATZ,                 //id
                  {46.94754, 7.43904, 542.2},                  //llaPos
                  0,                                           //map viewing angle
                  {46.947629, 7.440754, 542.2},                //modelOrigin
                  "",                                          //slamMapFileName
                  "dummy.jpg",                                 //relocAlignImage,
                  "calibrations/voc_fbow.bin",                 //vocFileName,
                  2,                                           //vocLayer,
                  //{640, 360},                                  //cameraFrameTargetSize
                  ExtractorType::ExtractorType_FAST_ORBS_2000, //initializationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000, //relocalizationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000, //trackingExtractorType
                  2);
        loc.areas[area.id] = area;
    }

    {
        Area area(AreaId::BERN_SBB,                            //id
                  {46.94837, 7.44041, 541.4},                  //llaPos
                  110,                                         //map viewing angle
                  {46.947629, 7.440754, 542.2},                //modelOrigin
                  "",                                          //slamMapFileName
                  "dummy.jpg",                                 //relocAlignImage,
                  "calibrations/voc_fbow.bin",                 //vocFileName,
                  2,                                           //vocLayer,
                  //{640, 360},                                  //cameraFrameTargetSize
                  ExtractorType::ExtractorType_FAST_ORBS_2000, //initializationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000, //relocalizationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000, //trackingExtractorType
                  2);
        loc.areas[area.id] = area;
    }

    {
        Area area(AreaId::BERN_SBB_ENTRANCE,                   //id
                  {46.94848, 7.44012, 541.1},                  //llaPos
                  60,                                          //map viewing angle
                  {46.947629, 7.440754, 542.2},                //modelOrigin
                  "",                                          //slamMapFileName
                  "dummy.jpg",                                 //relocAlignImage,
                  "calibrations/voc_fbow.bin",                 //vocFileName,
                  2,                                           //vocLayer,
                  //{640, 360},                                  //cameraFrameTargetSize
                  ExtractorType::ExtractorType_FAST_ORBS_2000, //initializationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000, //relocalizationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000, //trackingExtractorType
                  2);
        loc.areas[area.id] = area;
    }

    {
        Area area(AreaId::BERN_CHRISTOFFEL_BRIDGE,             //id
                  {46.94749, 7.44003, 542.5},                  //llaPos
                  0,                                           //map viewing angle
                  {46.947629, 7.440754, 542.2},                //modelOrigin
                  "",                                          //slamMapFileName
                  "dummy.jpg",                                 //relocAlignImage,
                  "calibrations/voc_fbow.bin",                 //vocFileName,
                  2,                                           //vocLayer,
                  //{640, 360},                                  //cameraFrameTargetSize
                  ExtractorType::ExtractorType_FAST_ORBS_2000, //initializationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000, //relocalizationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000, //trackingExtractorType
                  2);
        loc.areas[area.id] = area;
    }

    return loc;
}

const Location defineLocationBiel()
{
    Location loc(LocationId::BIEL,
                 "locations/biel/locationMapImgBiel.jpg",
                 {47.14290, 7.24225, 506.3},
                 {47.14060, 7.24693, 434.3},
                 1600,
                 "models/biel/DEM_Biel-BFH_WGS84.tif",
                 "https://pallas.ti.bfh.ch/erlebar/models/biel.zip",
                 "biel");

    {
        Area area(AreaId::BIEL_BFH,                                                                              //id
                  {47.14262, 7.24313, 488.3},                                                                    //llaPos
                  60,                                                                                            //map viewing angle
                  {47.14271, 7.24337, 487.0},                                                                    //modelOrigin
                  "locations/biel/bfh/maps/DEVELOPMENT-map_20201116-165837_biel_office_FAST-ORBS-2000_2.waimap", //slamMapFileName
                  "dummy.jpg",                                                                                   //relocAlignImage,
                  "calibrations/voc_fbow.bin",                                                                   //vocFileName,
                  2,                                                                                             //vocLayer,
                  //{640, 360},                                                                                    //cameraFrameTargetSize
                  ExtractorType::ExtractorType_FAST_ORBS_2000,                                                   //initializationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000,                                                   //relocalizationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000,                                                   //trackingExtractorType
                  2);
        loc.areas[area.id] = area;
    }
    return loc;
}

const Location defineLocationEvilard()
{
    Location loc(LocationId::EVILARD,
                 "locations/evilard/locationMapImgEvilard.jpg",
                 {47.14954, 7.23246, 741.0},
                 {47.14778, 7.23661, 696.0},
                 2000,
                 "",
                 "",
                 "evilard");

    {
        Area area(AreaId::EVILARD_ROC2,                                                                               //id
                  {47.14888, 7.23343, 727.3},                                                                         //llaPos
                  90.f,                                                                                               //map viewing angle
                  {47.14888, 7.23343, 727.3},                                                                         //modelOrigin
                  "locations/evilard/roc2/maps/DEVELOPMENT-map_20200918-163220_evilard_roc2_FAST-ORBS-3000_2.waimap", //slamMapFileName
                  "locations/evilard/roc2/relocAlignImg.jpg",                                                         //relocAlignImage,
                  "calibrations/voc_fbow.bin",                                                                        //vocFileName,
                  2,                                                                                                  //vocLayer,
                  //{640, 360},                                                                                         //cameraFrameTargetSize
                  ExtractorType::ExtractorType_FAST_ORBS_2000,                                                        //initializationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000,                                                        //relocalizationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000,                                                        //trackingExtractorType
                  2);
        loc.areas[area.id] = area;
    }
    {
        Area area(AreaId::EVILARD_FIREFIGHTERS,                                                                                       //id
                  {47.14867, 7.23291, 725.9},                                                                                         //llaPos
                  -150.f,                                                                                                             //map viewing angle
                  {47.14888, 7.23343, 727.3},                                                                                         //modelOrigin
                  "locations/evilard/firefighters/maps/DEVELOPMENT-map_20200918-100317_evilard_firefighters_FAST-ORBS-2000_2.waimap", //slamMapFileName
                  "locations/evilard/firefighters/relocAlignImg.jpg",                                                                 //relocAlignImage,
                  "calibrations/voc_fbow.bin",                                                                                        //vocFileName,
                  2,                                                                                                                  //vocLayer,
                  //{640, 360},                                                                                                         //cameraFrameTargetSize
                  ExtractorType::ExtractorType_FAST_ORBS_2000,                                                                        //initializationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000,                                                                        //relocalizationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000,                                                                        //trackingExtractorType
                  2);
        loc.areas[area.id] = area;
    }
    {
        Area area(AreaId::EVILARD_OFFICE,                                 //id
                  {47.14886, 7.23322, 726.9},                             //llaPos
                  -150.f,                                                 //map viewing angle
                  {47.14886, 7.23322, 726.9},                             //modelOrigin
                  "locations/evilard/office/maps/DEVELOPMENT-map.waimap", //slamMapFileName
                  "locations/evilard/office/relocAlignImg.jpg",           //relocAlignImage,
                  "calibrations/voc_fbow.bin",                            //vocFileName,
                  2,                                                      //vocLayer,
                  //{640, 480},                                             //cameraFrameTargetSize
                  ExtractorType::ExtractorType_FAST_ORBS_2000,            //initializationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000,            //relocalizationExtractorType
                  ExtractorType::ExtractorType_FAST_ORBS_1000,            //trackingExtractorType
                  2);
        loc.areas[area.id] = area;
    }
    return loc;
}

const std::map<LocationId, Location> defineLocations()
{
    std::map<LocationId, Location> locations;
    locations[LocationId::AUGST]    = defineLocationAugst();
    locations[LocationId::AVENCHES] = defineLocationAvenches();
    locations[LocationId::BERN]     = defineLocationBern();
    locations[LocationId::BIEL]     = defineLocationBiel();
    locations[LocationId::EVILARD]  = defineLocationEvilard();

    return locations;
}
};
