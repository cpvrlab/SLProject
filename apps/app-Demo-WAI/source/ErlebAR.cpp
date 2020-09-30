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
        case AreaId::AUGST_TEMPLE_HILL_MARKER:
            return "templeHill-marker";
        case AreaId::AUGST_TEMPLE_HILL_THEATER_BOTTOM:
            return "templeHillTheaterBottom";
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
        //biel
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
        default:
            return "Missing id to name mapping!";
    }
}

const Location defineLocationAugst()
{
    Location loc;
    loc.id                   = LocationId::AUGST;
    loc.name                 = mapLocationIdToName(loc.id);
    loc.areaMapImageFileName = "locations/augst/locationMapImgAugst.jpg";
    loc.dspPixWidth          = 800;
    {
        Area area;
        area.id              = AreaId::AUGST_TEMPLE_HILL_MARKER;
        area.name            = mapAreaIdToName(area.id);
        area.xPosPix         = 496;
        area.yPosPix         = 566;
        area.viewAngleDeg    = 120;
        area.slamMapFileName = "locations/augst/templeHill-marker/maps/map_20200812-114906_augst_templeHill-marker_FAST-ORBS-3000_2.waimap";
        loc.areas[area.id]   = area;
    }
    {
        Area area;
        area.id              = AreaId::AUGST_TEMPLE_HILL_THEATER_BOTTOM;
        area.name            = mapAreaIdToName(area.id);
        area.xPosPix         = 627;
        area.yPosPix         = 606;
        area.viewAngleDeg    = -18;
        area.slamMapFileName = "locations/augst/templeHillTheater/maps/map_20200819-154204_augst_templeHillTheater_FAST-ORBS-3000_2.waimap";
        loc.areas[area.id]   = area;
    }
    return loc;
}
const Location defineLocationAvenches()
{
    Location loc;
    loc.id                   = LocationId::AVENCHES;
    loc.name                 = mapLocationIdToName(loc.id);
    loc.areaMapImageFileName = "locations/avenches/locationMapImgAvenches.jpg";
    loc.dspPixWidth          = 2046;
    {
        Area area;
        area.id              = AreaId::AVENCHES_AMPHITHEATER_ENTRANCE;
        area.name            = mapAreaIdToName(area.id);
        area.xPosPix         = 520;
        area.yPosPix         = 530;
        area.viewAngleDeg    = 170;
        area.vocFileName     = "locations/avenches/amphitheater/amphitheater_voc.bin";
        area.slamMapFileName = "locations/avenches/amphitheaterEntrance/maps/realease-map_20201001-010003_Avenches_amphitheaterEntrance_FAST-ORBS-2000_2.waimap";
        area.relocAlignImage = "locations/avenches/amphitheaterEntrance/amphitheaterEntrance-reloc-align-img.jpg";
        area.vocLayer        = 4;
        loc.areas[area.id]   = area;
    }
    {
        Area area;
        area.id              = AreaId::AVENCHES_AMPHITHEATER;
        area.name            = mapAreaIdToName(area.id);
        area.xPosPix         = 290;
        area.yPosPix         = 580;
        area.viewAngleDeg    = -18;
        area.vocFileName     = "locations/avenches/amphitheater/amphitheater_voc.bin";
        area.slamMapFileName = "locations/avenches/amphitheater/maps/release-map_20200930-160451_avenches_amphitheater_FAST-ORBS-2000_2.waimap";
        area.relocAlignImage = "locations/avenches/amphitheater/amphitheater-reloc-align-img.jpg";
        area.vocLayer        = 4;
        loc.areas[area.id]   = area;
    }
    {
        Area area;
        area.id              = AreaId::AVENCHES_CIGOGNIER;
        area.name            = mapAreaIdToName(area.id);
        area.xPosPix         = 1250;
        area.yPosPix         = 450;
        area.viewAngleDeg    = -140;
        area.slamMapFileName = "locations/avenches/cigonier-marker/maps/DEVELOPMENT-map_20200529-162110_avenches_cigonier-marker_FAST_ORBS_2000.json";
        area.vocLayer        = 4;
        loc.areas[area.id]   = area;
    }
    {
        Area area;
        area.id              = AreaId::AVENCHES_THEATER;
        area.name            = mapAreaIdToName(area.id);
        area.xPosPix         = 1800;
        area.yPosPix         = 850;
        area.viewAngleDeg    = 50;
        area.vocFileName     = "locations/avenches/theater/theater_voc.bin";
        area.slamMapFileName = "locations/avenches/theater/maps/release-map_20200930-154707_avenches_theater_FAST-ORBS-2000_2.waimap";
        area.relocAlignImage = "locations/avenches/theater/theater-reloc-align-img.jpg";
        area.vocLayer        = 4;
        loc.areas[area.id]   = area;
    }

    return loc;
}
const Location defineLocationChristoffel()
{
    Location loc;
    loc.id                   = LocationId::BERN;
    loc.name                 = mapLocationIdToName(loc.id);
    loc.areaMapImageFileName = "locations/bern/locationMapImgBern.jpg";
    loc.dspPixWidth          = 2080;
    {
        Area area;
        area.id           = AreaId::BERN_MILCHGAESSLI;
        area.name         = mapAreaIdToName(area.id);
        area.xPosPix      = 740;
        area.yPosPix      = 294;
        area.viewAngleDeg = 60;
        area.slamMapFileName = "locations/bern/milchgaessli/maps/DEVELOPMENT-map_20200702-173422_christoffel_milchgaessli_FAST-ORBS-2000_2.json";
        //area.slamMapFileName = "locations/bern/milchgaessli/maps/orig-DEVELOPMENT-map_20200811-152001_bern_milchgaessli_FAST-ORBS-3000_2.waimap";
        loc.areas[area.id]   = area;
    }
    return loc;
}

const Location defineLocationBiel()
{
    Location loc;
    loc.id                   = LocationId::BIEL;
    loc.name                 = mapLocationIdToName(loc.id);
    loc.areaMapImageFileName = "locations/biel/locationMapImgBiel.jpg";
    loc.dspPixWidth          = 800;
    {
        Area area;
        area.id            = AreaId::BIEL_GERECHTIGKEITSBRUNNEN;
        area.name          = mapAreaIdToName(area.id);
        area.xPosPix       = 606;
        area.yPosPix       = 277;
        area.viewAngleDeg  = 10.f;
        loc.areas[area.id] = area;
    }
    {
        Area area;
        area.id            = AreaId::BIEL_JACOB_ROSINUS;
        area.name          = mapAreaIdToName(area.id);
        area.xPosPix       = 1387;
        area.yPosPix       = 730;
        area.viewAngleDeg  = 25.f;
        loc.areas[area.id] = area;
    }
    {
        Area area;
        area.id            = AreaId::BIEL_LEUBRINGENBAHN;
        area.name          = mapAreaIdToName(area.id);
        area.xPosPix       = 606;
        area.yPosPix       = 50;
        area.viewAngleDeg  = 60.f;
        loc.areas[area.id] = area;
    }
    {
        Area area;
        area.id           = AreaId::BIEL_RING;
        area.name         = mapAreaIdToName(area.id);
        area.xPosPix      = 200;
        area.yPosPix      = 200;
        area.viewAngleDeg = 110.f;
        //area.slamMapFileName       = "locations/biel/ring/maps/DEVELOPMENT-map_20200814-130443_biel_ring_FAST-ORBS-1000_2.json.gz";
        area.slamMapFileName       = "locations/biel/ring/maps/DEVELOPMENT-map_20200814-130443_biel_ring_FAST-ORBS-1000_2.waimap";
        area.cameraFrameTargetSize = {640, 480};
        loc.areas[area.id]         = area;
    }
    {
        Area area;
        area.id            = AreaId::BIEL_SOUTHWALL;
        area.name          = mapAreaIdToName(area.id);
        area.xPosPix       = 250;
        area.yPosPix       = 250;
        area.viewAngleDeg  = 270.f;
        loc.areas[area.id] = area;
    }
    {
        Area area;
        area.id                      = AreaId::BIEL_OFFICE;
        area.name                    = mapAreaIdToName(area.id);
        area.xPosPix                 = 322;
        area.yPosPix                 = 238;
        area.viewAngleDeg            = 20.f;
        area.slamMapFileName         = "locations/biel/office/maps/DEVELOPMENT-map_20200909-152037_biel_office_FAST-ORBS-2000_2.waimap";
        area.relocAlignImage         = "locations/biel/office/office-reloc-align-img.jpg";
        area.cameraFrameTargetSize   = {640, 360};
        area.cameraFrameCropToScreen = false;
        loc.areas[area.id]           = area;
    }
    return loc;
}

const std::map<LocationId, Location> defineLocations()
{
    std::map<LocationId, Location> locations;
    locations[LocationId::AUGST]    = defineLocationAugst();
    locations[LocationId::AVENCHES] = defineLocationAvenches();
    locations[LocationId::BERN]     = defineLocationChristoffel();
    locations[LocationId::BIEL]     = defineLocationBiel();

    return locations;
}
};
