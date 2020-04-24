#ifndef ERLEBAR_EVENTS_H
#define ERLEBAR_EVENTS_H

#include "ErlebAR.h"
#include <sm/Event.h>

//-----------------------------------------------------------------------------
// EventData
//-----------------------------------------------------------------------------
class InitEventData : public sm::EventData
{
public:
    InitEventData(DeviceData deviceData)
      : deviceData(deviceData)
    {
    }
    const DeviceData deviceData;
};

class ErlebarEventData : public sm::EventData
{
public:
    ErlebarEventData(ErlebAR::LocationId location)
      : location(location)
    {
    }
    const ErlebAR::LocationId location;
};

class AreaEventData : public sm::EventData
{
public:
    AreaEventData(ErlebAR::LocationId locId, ErlebAR::AreaId areaId)
      : locId(locId),
        areaId(areaId)
    {
    }
    const ErlebAR::LocationId locId;
    const ErlebAR::AreaId     areaId;
};

//-----------------------------------------------------------------------------
// Events
//-----------------------------------------------------------------------------
class InitEvent : public sm::Event
{
public:
    InitEvent(int            scrWidth,
              int            scrHeight,
              float          scr2fbX,
              float          scr2fbY,
              int            dpi,
              AppDirectories dirs)
    {
        enableTransition((unsigned int)StateId::IDLE,
                         (unsigned int)StateId::INIT);

        DeviceData deviceData(scrWidth, scrHeight, scr2fbX, scr2fbY, dpi, dirs);
        _eventData = new InitEventData(deviceData);
    }
};

class GoBackEvent : public sm::Event
{
public:
    GoBackEvent()
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::LOCATION_MAP,
                         (unsigned int)StateId::SELECTION);
        enableTransition((unsigned int)StateId::AREA_TRACKING,
                         (unsigned int)StateId::LOCATION_MAP);
        enableTransition((unsigned int)StateId::TUTORIAL,
                         (unsigned int)StateId::SELECTION);
        enableTransition((unsigned int)StateId::ABOUT,
                         (unsigned int)StateId::SELECTION);
        enableTransition((unsigned int)StateId::SETTINGS,
                         (unsigned int)StateId::SELECTION);
        enableTransition((unsigned int)StateId::CAMERA_TEST,
                         (unsigned int)StateId::SELECTION);
        enableTransition((unsigned int)StateId::TEST,
                         (unsigned int)StateId::SELECTION);
        enableTransition((unsigned int)StateId::AREA_INFO,
                         (unsigned int)StateId::LOCATION_MAP);
    }
};

class DestroyEvent : public sm::Event
{
public:
    DestroyEvent()
    {
        enableTransition((unsigned int)StateId::INIT,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::START_TEST,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::TEST,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::LOCATION_MAP,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::AREA_TRACKING,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::TUTORIAL,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::ABOUT,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::CAMERA_TEST,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::AREA_INFO,
                         (unsigned int)StateId::DESTROY);
    }
};

class DoneEvent : public sm::Event
{
public:
    DoneEvent()
    {

        enableTransition((unsigned int)StateId::DESTROY,
                         (unsigned int)StateId::IDLE);
        enableTransition((unsigned int)StateId::INIT,
                         (unsigned int)StateId::WELCOME);
        enableTransition((unsigned int)StateId::WELCOME,
                         (unsigned int)StateId::SELECTION);
        enableTransition((unsigned int)StateId::START_TEST,
                         (unsigned int)StateId::TEST);
        enableTransition((unsigned int)StateId::RESUME_TEST,
                         (unsigned int)StateId::TEST);
        enableTransition((unsigned int)StateId::AREA_INFO,
                         (unsigned int)StateId::AREA_TRACKING);
    }
};

class StartErlebarEvent : public sm::Event
{
public:
    StartErlebarEvent(ErlebAR::LocationId location)
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::LOCATION_MAP);

        _eventData = new ErlebarEventData(location);
    }
};

class AreaSelectedEvent : public sm::Event
{
public:
    AreaSelectedEvent(ErlebAR::LocationId locId, ErlebAR::AreaId areaId)
    {
        enableTransition((unsigned int)StateId::LOCATION_MAP,
                         (unsigned int)StateId::AREA_INFO);

        _eventData = new AreaEventData(locId, areaId);
    }
};

class StartTutorialEvent : public sm::Event
{
public:
    StartTutorialEvent()
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::TUTORIAL);
    }
};

class ShowAboutEvent : public sm::Event
{
public:
    ShowAboutEvent()
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::ABOUT);
    }
};

class ShowSettingsEvent : public sm::Event
{
public:
    ShowSettingsEvent()
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::SETTINGS);
    }
};

class StartCameraTestEvent : public sm::Event
{
public:
    StartCameraTestEvent()
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::CAMERA_TEST);
    }
};

class StartTestEvent : public sm::Event
{
public:
    StartTestEvent()
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::START_TEST);
    }
};

class HoldEvent : public sm::Event
{
public:
    HoldEvent()
    {
        enableTransition((unsigned int)StateId::TEST,
                         (unsigned int)StateId::HOLD_TEST);
        enableTransition((unsigned int)StateId::AREA_INFO,
                         (unsigned int)StateId::HOLD_TRACKING);
        enableTransition((unsigned int)StateId::AREA_TRACKING,
                         (unsigned int)StateId::HOLD_TRACKING);
    }
};

class ResumeEvent : public sm::Event
{
public:
    ResumeEvent()
    {
        enableTransition((unsigned int)StateId::HOLD_TEST,
                         (unsigned int)StateId::RESUME_TEST);
        enableTransition((unsigned int)StateId::HOLD_TRACKING,
                         (unsigned int)StateId::AREA_INFO);
    }
};

#endif // !ERLEBAR_EVENTS_H
