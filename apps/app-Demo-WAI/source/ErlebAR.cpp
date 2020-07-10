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
        case LocationId::CHRISTOFFEL:
            return "Christoffelturm";
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
            return "templeHillTheater";
        //avenches
        case AreaId::AVENCHES_AMPHITHEATER:
            return "Amphitheater";
        case AreaId::AVENCHES_AMPHITHEATER_ENTRANCE:
            return "Amphitheater-Entrance";
        case AreaId::AVENCHES_CIGOGNIER:
            return "cigognier";
        case AreaId::AVENCHES_TEMPLE:
            return "Temple";
        case AreaId::AVENCHES_THEATER:
            return "Theater";
        //christoffel
        case AreaId::CHRISTOFFEL_SBB:
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
        area.slamMapFileName = "locations/augst/templeHill-marker/maps/final_marker-map_20200526-142338_augst_templeHill-marker_FAST_ORBS_1000.json";
        loc.areas[area.id]   = area;
    }
    {
        Area area;
        area.id              = AreaId::AUGST_TEMPLE_HILL_THEATER_BOTTOM;
        area.name            = mapAreaIdToName(area.id);
        area.xPosPix         = 627;
        area.yPosPix         = 606;
        area.viewAngleDeg    = -18;
        area.slamMapFileName = "locations/augst/templeHillTheaterBottom/maps/DEVELOPMENT-map_20200528-090204_augst_templeHillTheaterBottom_FAST_ORBS_1000.json";
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
    loc.dspPixWidth          = 2000;
    {
        Area area;
        area.id              = AreaId::AVENCHES_AMPHITHEATER_ENTRANCE;
        area.name            = mapAreaIdToName(area.id);
        area.xPosPix         = 820;
        area.yPosPix         = 660;
        area.viewAngleDeg    = 170;
        area.slamMapFileName = "location/avenches/amphitheaterEntrance/maps/stable-map_20200710-170350_avenches_amphitheaterEntrance_FAST-ORBS-2000_2.json";
        loc.areas[area.id]   = area;
    }
    {
        Area area;
        area.id              = AreaId::AVENCHES_AMPHITHEATER;
        area.name            = mapAreaIdToName(area.id);
        area.xPosPix         = 472;
        area.yPosPix         = 736;
        area.viewAngleDeg    = -18;
        area.slamMapFileName = "location/avenches/amphitheater/stable-map_20200710-151049_avenches_amphitheater_FAST-ORBS-2000_2.json";
        loc.areas[area.id]   = area;
    }
    {
        Area area;
        area.id              = AreaId::AVENCHES_CIGOGNIER;
        area.name            = mapAreaIdToName(area.id);
        area.xPosPix         = 1760;
        area.yPosPix         = 554;
        area.viewAngleDeg    = -140;
        area.slamMapFileName = "locations/avenches/cigonier-marker/maps/DEVELOPMENT-map_20200529-162110_avenches_cigonier-marker_FAST_ORBS_2000.json";
        loc.areas[area.id]   = area;
    }
    {
        Area area;
        area.id              = AreaId::AVENCHES_THEATER;
        area.name            = mapAreaIdToName(area.id);
        area.xPosPix         = 2463;
        area.yPosPix         = 1132;
        area.viewAngleDeg    = 50;
        area.slamMapFileName = "locations/avenches/theater/maps/DEVELOPMENT-map_20200710-182227_avenches_theater_FAST-ORBS-2000_2.json";
        loc.areas[area.id]   = area;
    }

    return loc;
}
const Location defineLocationChristoffel()
{
    Location loc;
    loc.id                   = LocationId::CHRISTOFFEL;
    loc.name                 = mapLocationIdToName(loc.id);
    loc.areaMapImageFileName = "locations/christoffel/locationMapImgChristoffel.jpg";
    loc.dspPixWidth          = 2080;
    {
        Area area;
        area.id            = AreaId::CHRISTOFFEL_MILCHGAESSLI;
        area.name          = mapAreaIdToName(area.id);
        area.xPosPix       = 740;
        area.yPosPix       = 294;
        area.viewAngleDeg  = 60;
        loc.areas[area.id] = area;
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
        area.id            = AreaId::BIEL_RING;
        area.name          = mapAreaIdToName(area.id);
        area.xPosPix       = 200;
        area.yPosPix       = 200;
        area.viewAngleDeg  = 110.f;
        loc.areas[area.id] = area;
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
    return loc;
}

const std::map<LocationId, Location> defineLocations()
{
    std::map<LocationId, Location> locations;
    locations[LocationId::AUGST]       = defineLocationAugst();
    locations[LocationId::AVENCHES]    = defineLocationAvenches();
    locations[LocationId::CHRISTOFFEL] = defineLocationChristoffel();
    locations[LocationId::BIEL]        = defineLocationBiel();

    return locations;
}
};
