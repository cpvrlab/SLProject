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
    }
}

const std::map<LocationId, Location> defineLocations()
{
    std::map<LocationId, Location> locations;
    //location augst
    {}
    //location avenches
    {}
    //location christoffel
    {}
    //location biel
    {
        Location loc;
        loc.id                   = LocationId::BIEL;
        loc.areaMapImageFileName = "areaMapImgs/areaMapImgBiel.jpg";
        loc.name                 = mapLocationIdToName(loc.id);
        {
            Area area;
            area.id            = AreaId::BIEL_GERECHTIGKEITSBRUNNEN;
            area.xPosPix       = 50;
            area.yPosPix       = 50;
            area.viewAngleDeg  = 0;
            loc.areas[area.id] = area;
        }
        {
            Area area;
            area.id            = AreaId::BIEL_JACOB_ROSINUS;
            area.xPosPix       = 100;
            area.yPosPix       = 100;
            area.viewAngleDeg  = 0;
            loc.areas[area.id] = area;
        }
        {
            Area area;
            area.id            = AreaId::BIEL_LEUBRINGENBAHN;
            area.xPosPix       = 150;
            area.yPosPix       = 150;
            area.viewAngleDeg  = 0;
            loc.areas[area.id] = area;
        }
        {
            Area area;
            area.id            = AreaId::BIEL_RING;
            area.xPosPix       = 200;
            area.yPosPix       = 200;
            area.viewAngleDeg  = 0;
            loc.areas[area.id] = area;
        }
        {
            Area area;
            area.id            = AreaId::BIEL_SOUTHWALL;
            area.xPosPix       = 250;
            area.yPosPix       = 250;
            area.viewAngleDeg  = 0;
            loc.areas[area.id] = area;
        }
        locations[LocationId::BIEL] = loc;
    }

    return locations;
}
};
