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
    WAIEventType_EditMap
};

enum MapPointEditorEnum
{
    MapPointEditor_None,
    MapPointEditor_SaveMap,
    MapPointEditor_SaveMapRaw,
    MapPointEditor_SaveMapBinary,
    MapPointEditor_ApplyToMapPoints,
    MapPointEditor_Quit,
    MapPointEditor_EnterEditMode,
    MapPointEditor_SelectSingleVideo,
    MapPointEditor_SelectNMatched,
    MapPointEditor_SelectAllPoints,
    MapPointEditor_LoadMatching,
    MapPointEditor_KeyFrameMode
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

struct WAIEventEditMap : WAIEvent
{
    WAIEventEditMap()
    {
        type     = WAIEventType_EditMap;
        action   = MapPointEditor_None;
        editMode = NodeEditMode_None;
    }
    MapPointEditorEnum action;
    SLNodeEditMode     editMode;
    std::vector<int>*  kFVidMatching;
    std::vector<bool>  vid;
    std::vector<bool>  nmatches;
    bool               b;
};

#endif //WAI_EVENT_H
