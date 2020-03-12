#ifndef LOCATION_MAP_STATE_H
#define LOCATION_MAP_STATE_H

#include <states/State.h>
#include <ErlebAR.h>

class LocationMapState : public State
{
public:
    bool update() override;

    Area getTargetArea() const { return _selectedArea; }

protected:
    void doStart() override;

    Area _selectedArea;
};

class BielMapState : public LocationMapState
{
public:
};

class AugstMapState : public LocationMapState
{
public:
};

class AvanchesMapState : public LocationMapState
{
public:
};

class ChristoffelMapState : public LocationMapState
{
public:
};

#endif //LOCATION_MAP_STATE_H
