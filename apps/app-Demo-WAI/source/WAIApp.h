//#############################################################################
//  File:      WAISceneView.h
//  Purpose:   Node transform test application that demonstrates all transform
//             possibilities of SLNode
//  Author:    Marc Wacker
//  Date:      July 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef APP_WAI_SCENE_VIEW
#define APP_WAI_SCENE_VIEW

#include <vector>
#include "AppWAIScene.h"

#include <CVCalibration.h>
#include <WAIAutoCalibration.h>
#include <AppDirectories.h>
#include <WAIModeOrbSlam2.h>
#include <AppDemoWaiGui.h>
#include <SLInputEventInterface.h>
#include <WAISlam.h>
#include <SENSCamera.h>
#include <SENSVideoStream.h>
#include <GLSLextractor.h>
#include <FeatureExtractorFactory.h>

class SLMaterial;
class SLPoints;
class SLNode;
class AppDemoGuiError;
class AppDemoGuiSlamLoad;

struct ExtractorIds
{
    int trackingExtractorId;
    int initializationExtractorId;
    int markerExtractorId;
};

struct SlamParams
{
    //returns true if loading was successful. Otherwise there may have been no file.
    bool load(std::string fileName)
    {
        cv::FileStorage fs;
        try
        {
            fs.open(fileName, cv::FileStorage::READ);
            if (fs.isOpened())
            {
                if (!fs["videoFile"].empty())
                    fs["videoFile"] >> videoFile;
                if (!fs["mapFile"].empty())
                    fs["mapFile"] >> mapFile;
                if (!fs["calibrationFile"].empty())
                    fs["calibrationFile"] >> calibrationFile;
                if (!fs["vocabularyFile"].empty())
                    fs["vocabularyFile"] >> vocabularyFile;
                if (!fs["markerFile"].empty())
                    fs["markerFile"] >> markerFile;
                if (!fs["location"].empty())
                    fs["location"] >> location;
                if (!fs["area"].empty())
                    fs["area"] >> area;

                if (!fs["cullRedundantPerc"].empty())
                    fs["cullRedundantPerc"] >> params.cullRedundantPerc;
                if (!fs["fixOldKfs"].empty())
                    fs["fixOldKfs"] >> params.fixOldKfs;
                if (!fs["onlyTracking"].empty())
                    fs["onlyTracking"] >> params.onlyTracking;
                if (!fs["retainImg"].empty())
                    fs["retainImg"] >> params.retainImg;
                if (!fs["serial"].empty())
                    fs["serial"] >> params.serial;
                if (!fs["trackOptFlow"].empty())
                    fs["trackOptFlow"] >> params.trackOptFlow;

                if (!fs["initializationExtractorId"].empty())
                    fs["initializationExtractorId"] >> extractorIds.initializationExtractorId;
                if (!fs["markerExtractorId"].empty())
                    fs["markerExtractorId"] >> extractorIds.markerExtractorId;
                if (!fs["trackingExtractorId"].empty())
                    fs["trackingExtractorId"] >> extractorIds.trackingExtractorId;

                fs.release();
                Utils::log("WAIApp", "SlamParams loaded from %s", fileName.c_str());

                return true;
            }
        }
        catch (...)
        {
            Utils::log("WAIApp", "SlamParams: Parsing of file failed: %s", fileName.c_str());
        }

        return false;
    }

    void save(std::string fileName)
    {
        cv::FileStorage fs(fileName, cv::FileStorage::WRITE);

        if (!fs.isOpened())
        {
            Utils::log("WAIApp", "SlamParams: Failed to open file for writing: %s", fileName.c_str());
            return;
        }

        fs << "videoFile" << videoFile;
        fs << "mapFile" << mapFile;
        fs << "calibrationFile" << calibrationFile;
        fs << "vocabularyFile" << vocabularyFile;
        fs << "markerFile" << markerFile;
        fs << "location" << location;
        fs << "area" << area;

        fs << "cullRedundantPerc" << params.cullRedundantPerc;
        fs << "fixOldKfs" << params.fixOldKfs;
        fs << "onlyTracking" << params.onlyTracking;
        fs << "retainImg" << params.retainImg;
        fs << "serial" << params.serial;
        fs << "trackOptFlow" << params.trackOptFlow;

        fs << "initializationExtractorId" << extractorIds.initializationExtractorId;
        fs << "markerExtractorId" << extractorIds.markerExtractorId;
        fs << "trackingExtractorId" << extractorIds.trackingExtractorId;

        fs.release();
        Utils::log("WAIApp", "SlamParams saved to %s", fileName.c_str());
    }

    std::string               videoFile;
    std::string               mapFile;
    std::string               calibrationFile;
    std::string               vocabularyFile;
    std::string               markerFile;
    std::string               location;
    std::string               area;
    WAI::ModeOrbSlam2::Params params;
    ExtractorIds              extractorIds;
};

enum WAIEventType
{
    WAIEventType_None,
    WAIEventType_StartOrbSlam,
    WAIEventType_SaveMap,
    WAIEventType_VideoControl,
    WAIEventType_VideoRecording,
    WAIEventType_MapNodeTransform,
    WAIEventType_DownloadCalibrationFiles,
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

//-----------------------------------------------------------------------------
class WAIApp : public SLInputEventInterface
{
public:
    WAIApp();
    ~WAIApp();
    //call load to correctly initialize wai app
    int  load(int            scrWidth,
              int            scrHeight,
              float          scr2fbX,
              float          scr2fbY,
              int            dpi,
              AppDirectories dirs);
    void setCamera(SENSCamera* camera);
    //try to load last slam configuration, else open loading dialog
    void loadSlam();

    //call update to update the frame, wai and visualization
    bool update();
    void close();

    //initialize wai orb slam with transferred parameters
    void startOrbSlam(SlamParams slamParams);
    void showErrorMsg(std::string msg);

    //todo: replace when we are independent of SLApplication
    std::string name();
    void        setDeviceParameter(const std::string& parameter,
                                   std::string        value);

    //sensor stuff (todo: move out of waiapp?)
    void setRotationQuat(float quatX,
                         float quatY,
                         float quatZ,
                         float quatW);
    void setLocationLLA(float latitudeDEG,
                        float longitudeDEG,
                        float altitudeM,
                        float accuracyM);
    bool usesRotationSensor();
    bool usesLocationSensor();

    //set path for external writable directory for mobile devices
    //todo: is this still needed?
    void                   initExternalDataDirectory(std::string path);
    const SENSVideoStream* getVideoFileStream() const { return _videoFileStream.get(); }
    const CVCalibration&   getCalibration() const { return _calibration; }
    const cv::Size&        getFrameSize() const { return _videoFrameSize; }

    WAISlam* mode()
    {
        return _mode;
    }

    std::string videoDir;
    std::string mapDir;

private:
    bool updateTracking(SENSFramePtr frame);
    bool initSLProject(int scrWidth, int scrHeight, float scr2fbX, float scr2fbY, int dpi);
    void loadWAISceneView(SLScene* s, SLSceneView* sv, std::string location, std::string area);

    void setupGUI(std::string appName, std::string configDir, int dotsPerInch);
    void setupDefaultErlebARDirTo(std::string dir);
    //!download all remote files to transferred directory
    void downloadCalibrationFilesTo(std::string dir);
    //bool checkCalibration(const std::string& calibDir, const std::string& calibFileName);
    bool updateSceneViews();

    void updateTrackingVisualization(const bool iKnowWhereIAm, cv::Mat& imgRGB);
    void renderMapPoints(std::string                      name,
                         const std::vector<WAIMapPoint*>& pts,
                         SLNode*&                         node,
                         SLPoints*&                       mesh,
                         SLMaterial*&                     material);
    void renderKeyframes();
    void renderGraphs();
    void saveMap(std::string location, std::string area, std::string marker);
    void transformMapNode(SLTransformSpace tSpace,
                          SLVec3f          rotation,
                          SLVec3f          translation,
                          float            scale);
    // video writer
    void saveVideo(std::string filename);
    void saveGPSData(std::string videofile);

    void handleEvents();

    //get new frame from live video or video file stream
    SENSFramePtr updateVideoOrCamera();

    //todo: we dont need a pointer
    AppWAIScene _waiScene;
    //WAI::ModeOrbSlam2*           _mode;
    WAISlam*     _mode       = nullptr;
    SLSceneView* _sv         = nullptr;
    SLGLTexture* _videoImage = nullptr;

    SlamParams     _currentSlamParams;
    AppDirectories _dirs;
    std::string    _calibDir;

    //sensor stuff
    ofstream _gpsDataStream;
    SLQuat4f _lastKnowPoseQuaternion;
    SLQuat4f _IMUQuaternion;

    //load function has been called
    bool _loaded = false;

    cv::VideoWriter*                 _videoWriter = nullptr;
    std::unique_ptr<SENSVideoStream> _videoFileStream;
    SENSCamera*                      _camera = nullptr;

    cv::Size2i _videoFrameSize;

    std::unique_ptr<AppDemoWaiGui>      _gui;
    std::shared_ptr<AppDemoGuiError>    _errorDial;
    std::shared_ptr<AppDemoGuiSlamLoad> _guiSlamLoad;
    int                                 _lastFrameIdx;
    cv::Mat                             _undistortedLastFrame[2];
    bool                                _doubleBufferedOutput;

    // video controls
    bool _pauseVideo           = false;
    int  _videoCursorMoveIndex = 0;

    // event queue
    std::queue<WAIEvent*> _eventQueue;

    CVCalibration _calibration     = {CVCameraType::FRONTFACING, ""};
    bool          _showUndistorted = true;

    FeatureExtractorFactory      _featureExtractorFactory;
    std::unique_ptr<KPextractor> _trackingExtractor;
    std::unique_ptr<KPextractor> _initializationExtractor;
    std::unique_ptr<KPextractor> _markerExtractor;
};

#endif
