#ifndef WAI_EVENT_H
#define WAI_EVENT_H

#include <SlamParams.h>
#include <SLTransformNode.h>

enum WAIEventType
{
    WAIEventType_None,
    WAIEventType_StartOrbSlam,
    WAIEventType_SaveMap,
    WAIEventType_AutoCalibration,
    WAIEventType_VideoControl,
    WAIEventType_VideoRecording,
    WAIEventType_MapNodeTransform,
    WAIEventType_DownloadCalibrationFiles,
    WAIEventType_AdjustTransparency,
    WAIEventType_EnterEditMode,
    WAIEventType_EnterEditMapPointMode
};

enum MapPointEditorEnum
{
    MapPointEditor_None,
    MapPointEditor_SaveInMap,
    MapPointEditor_Quit,
    MapPointEditor_EnterEditMode,
    MapPointEditor_SelectSingleVideo
};

struct WAIEvent
{
    WAIEventType type;
};

struct WAIEventStartOrbSlam : WAIEvent
{
    WAIEventStartOrbSlam() { type = WAIEventType_StartOrbSlam; }

    SlamParams params;
};

struct WAIEventSaveMap : WAIEvent
{
    WAIEventSaveMap() { type = WAIEventType_SaveMap; }

    std::string location;
    std::string area;
    std::string marker;
};

struct WAIEventAutoCalibration : WAIEvent
{
    WAIEventAutoCalibration() { type = WAIEventType_AutoCalibration; }
    bool tryCalibrate;
    bool restoreOriginalCalibration;
    bool useGuessCalibration;
};

struct WAIEventVideoControl : WAIEvent
{
    WAIEventVideoControl() { type = WAIEventType_VideoControl; }

    bool pauseVideo;
    int  videoCursorMoveIndex;
};

struct WAIEventVideoRecording : WAIEvent
{
    WAIEventVideoRecording() { type = WAIEventType_VideoRecording; }

    std::string filename;
};

struct WAIEventMapNodeTransform : WAIEvent
{
    WAIEventMapNodeTransform() { type = WAIEventType_MapNodeTransform; }

    SLTransformSpace tSpace;
    SLVec3f          rotation;
    SLVec3f          translation;
    float            scale;
};

struct WAIEventDownloadCalibrationFiles : WAIEvent
{
    WAIEventDownloadCalibrationFiles() { type = WAIEventType_DownloadCalibrationFiles; }
};

struct WAIEventAdjustTransparency : WAIEvent
{
    WAIEventAdjustTransparency() { type = WAIEventType_AdjustTransparency; }

    float kt;
};

struct WAIEventEnterEditMode : WAIEvent
{
    WAIEventEnterEditMode()
    {
        type      = WAIEventType_EnterEditMode;
        saveToMap = false;
    }

    SLNodeEditMode editMode;
    bool saveToMap;
};

struct WAIEventEnterEditMapPointMode : WAIEvent
{
    WAIEventEnterEditMapPointMode()
    {
        type   = WAIEventType_EnterEditMapPointMode;
        action = MapPointEditor_None;
        vid = 0;
    }
    MapPointEditorEnum action;
    std::vector<int> * kFVidMatching;
    int vid;
};

#endif //WAI_EVENT_H
