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
        case AreaId::AVENCHES_CYGOGNIER:
            return "cygognier";
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
    loc.dspPixWidth          = 800;
    {
        Area area;
        area.id              = AreaId::AVENCHES_AMPHITHEATER_ENTRANCE;
        area.name            = mapAreaIdToName(area.id);
        area.xPosPix         = 878;
        area.yPosPix         = 666;
        area.viewAngleDeg    = 170;
        area.slamMapFileName = "location/avenches/amphitheaterEntrance/maps/DEVELOPMENT-map_20200528-184748_avenches_amphitheaterEntrance_FAST_ORBS_1000.json";
        loc.areas[area.id]   = area;
    }
    {
        Area area;
        area.id              = AreaId::AVENCHES_AMPHITHEATER;
        area.name            = mapAreaIdToName(area.id);
        area.xPosPix         = 520;
        area.yPosPix         = 742;
        area.viewAngleDeg    = -18;
        area.slamMapFileName = "location/avenches/amphitheater/maps/DEVELOPMENT-map_20200608-130203_avenches_amphitheater_FAST_BRIEF_1000.json";
        loc.areas[area.id]   = area;
    }
    {
        Area area;
        area.id              = AreaId::AVENCHES_TEMPLE;
        area.name            = mapAreaIdToName(area.id);
        area.xPosPix         = 1337;
        area.yPosPix         = 36;
        area.viewAngleDeg    = 100;
        area.slamMapFileName = "locations/avenches/temple/maps/DEVELOPMENT-map_20200531-143950_avenches_temple_FAST_ORBS_2000.json";
        loc.areas[area.id]   = area;
    }
    {
        Area area;
        area.id              = AreaId::AVENCHES_CYGOGNIER;
        area.name            = mapAreaIdToName(area.id);
        area.xPosPix         = 1817;
        area.yPosPix         = 560;
        area.viewAngleDeg    = -140;
        area.slamMapFileName = "locations/avenches/cigonier-marker/maps/DEVELOPMENT-map_20200529-162110_avenches_cygonier-marker_FAST_ORBS_2000.json";
        loc.areas[area.id]   = area;
    }
    {
        Area area;
        area.id              = AreaId::AVENCHES_THEATER;
        area.name            = mapAreaIdToName(area.id);
        area.xPosPix         = 2521;
        area.yPosPix         = 1138;
        area.viewAngleDeg    = 50;
        area.slamMapFileName = "locations/avenches/theater-marker/maps/DEVELOPMENT-map_20200602-202250_avenches_theater-marker_FAST_ORBS_1000.json";
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
    loc.dspPixWidth          = 800;
    {
        Area area;
        area.id            = AreaId::CHRISTOFFEL_SBB;
        area.name          = mapAreaIdToName(area.id);
        area.xPosPix       = 50;
        area.yPosPix       = 50;
        area.viewAngleDeg  = 0;
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
