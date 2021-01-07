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

class DownloadEventData : public sm::EventData
{
public:
    DownloadEventData(ErlebAR::LocationId location)
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
    InitEvent(std::string        senderInfo,
              int                scrWidth,
              int                scrHeight,
              int                dpi,
              const std::string& dataDir,
              const std::string& writableDir)
      : sm::Event("InitEvent", senderInfo)
    {
        enableTransition((unsigned int)StateId::IDLE,
                         (unsigned int)StateId::INIT);

        DeviceData deviceData(scrWidth, scrHeight, dpi, dataDir, writableDir);
        _eventData = new InitEventData(deviceData);
    }
};

class GoBackEvent : public sm::Event
{
public:
    GoBackEvent(std::string senderInfo)
      : sm::Event("GoBackEvent", senderInfo)
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::DOWNLOAD,
                         (unsigned int)StateId::SELECTION);
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
        enableTransition((unsigned int)StateId::TEST_RUNNER,
                         (unsigned int)StateId::SELECTION);
        enableTransition((unsigned int)StateId::AREA_INFO,
                         (unsigned int)StateId::LOCATION_MAP);
        enableTransition((unsigned int)StateId::SENSOR_TEST,
                         (unsigned int)StateId::SELECTION);
    }
};

class DestroyEvent : public sm::Event
{
public:
    DestroyEvent(std::string senderInfo)
      : sm::Event("DestroyEvent", senderInfo)
    {
        enableTransition((unsigned int)StateId::WELCOME,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::INIT,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::DOWNLOAD,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::START_TEST,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::TEST,
                         (unsigned int)StateId::DESTROY);
        enableTransition((unsigned int)StateId::TEST_RUNNER,
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
    DoneEvent(std::string senderInfo)
      : sm::Event("DoneEvent", senderInfo)
    {

        enableTransition((unsigned int)StateId::DESTROY,
                         (unsigned int)StateId::IDLE);
        enableTransition((unsigned int)StateId::INIT,
                         (unsigned int)StateId::WELCOME);
        enableTransition((unsigned int)StateId::WELCOME,
                         (unsigned int)StateId::SELECTION);
        enableTransition((unsigned int)StateId::DOWNLOAD,
                         (unsigned int)StateId::LOCATION_MAP);
        enableTransition((unsigned int)StateId::START_TEST,
                         (unsigned int)StateId::TEST);
        enableTransition((unsigned int)StateId::AREA_INFO,
                         (unsigned int)StateId::AREA_TRACKING);
        enableTransition((unsigned int)StateId::TEST_RUNNER,
                         (unsigned int)StateId::SELECTION);
    }
};

class StartDownloadEvent : public sm::Event
{
public:
    StartDownloadEvent(std::string senderInfo, ErlebAR::LocationId location)
      : sm::Event("StartDownloadEvent", senderInfo)
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::DOWNLOAD);

        _eventData = new DownloadEventData(location);
    }
};

class StartErlebarEvent : public sm::Event
{
public:
    StartErlebarEvent(std::string senderInfo, ErlebAR::LocationId location)
      : sm::Event("StartErlebarEvent", senderInfo)
    {
        enableTransition((unsigned int)StateId::DOWNLOAD,
                         (unsigned int)StateId::LOCATION_MAP);

        _eventData = new ErlebarEventData(location);
    }
};

class AreaSelectedEvent : public sm::Event
{
public:
    AreaSelectedEvent(std::string senderInfo, ErlebAR::LocationId locId, ErlebAR::AreaId areaId)
      : sm::Event("AreaSelectedEvent", senderInfo)
    {
        enableTransition((unsigned int)StateId::LOCATION_MAP,
                         (unsigned int)StateId::AREA_INFO);

        _eventData = new AreaEventData(locId, areaId);
    }
};

class StartTutorialEvent : public sm::Event
{
public:
    StartTutorialEvent(std::string senderInfo)
      : sm::Event("StartTutorialEvent", senderInfo)
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::TUTORIAL);
    }
};

class ShowAboutEvent : public sm::Event
{
public:
    ShowAboutEvent(std::string senderInfo)
      : sm::Event("ShowAboutEvent", senderInfo)
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::ABOUT);
    }
};

class ShowSettingsEvent : public sm::Event
{
public:
    ShowSettingsEvent(std::string senderInfo)
      : sm::Event("ShowSettingsEvent", senderInfo)
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::SETTINGS);
    }
};

class StartCameraTestEvent : public sm::Event
{
public:
    StartCameraTestEvent(std::string senderInfo)
      : sm::Event("StartCameraTestEvent", senderInfo)
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::CAMERA_TEST);
    }
};

class StartSensorTestEvent : public sm::Event
{
public:
    StartSensorTestEvent(std::string senderInfo)
      : sm::Event("StartSensorTestEvent", senderInfo)
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::SENSOR_TEST);
    }
};

class StartTestEvent : public sm::Event
{
public:
    StartTestEvent(std::string senderInfo)
      : sm::Event("StartTestEvent", senderInfo)
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::START_TEST);
    }
};

class StartTestRunnerEvent : public sm::Event
{
public:
    StartTestRunnerEvent(std::string senderInfo)
      : sm::Event("StartTestRunnerEvent", senderInfo)
    {
        enableTransition((unsigned int)StateId::SELECTION,
                         (unsigned int)StateId::TEST_RUNNER);
    }
};

class HoldEvent : public sm::Event
{
public:
    HoldEvent(std::string senderInfo)
      : sm::Event("HoldEvent", senderInfo)
    {
        enableTransition((unsigned int)StateId::AREA_INFO,
                         (unsigned int)StateId::HOLD_TRACKING);
        enableTransition((unsigned int)StateId::AREA_TRACKING,
                         (unsigned int)StateId::HOLD_TRACKING);
    }
};

class ResumeEvent : public sm::Event
{
public:
    ResumeEvent(std::string senderInfo)
      : sm::Event("ResumeEvent", senderInfo)
    {
        enableTransition((unsigned int)StateId::HOLD_TRACKING,
                         (unsigned int)StateId::AREA_INFO);
    }
};

#endif // !ERLEBAR_EVENTS_H
