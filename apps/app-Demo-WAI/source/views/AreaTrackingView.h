#ifndef AREA_TRACKING_VIEW_H
#define AREA_TRACKING_VIEW_H

#include <string>
#include <thread>
#include <SLInputManager.h>
#include <SLSceneView.h>
#include <AreaTrackingGui.h>
#include <ErlebAR.h>
#include <scenes/AppWAIScene.h>
#include <FeatureExtractorFactory.h>
#include <ImageBuffer.h>
#include <WAISlam.h>
#include <sens/SENSCalibration.h>
#include <WAIOrbVocabulary.h>
#include <sens/SENSFrame.h>
#include <sens/SENSGps.h>
#include <sens/SENSOrientation.h>
#include <sens/SENSCvCamera.h>
#include <AsyncWorker.h>
#include <scenes/UserGuidanceScene.h>
#include <UserGuidance.h>

#include <SLLightSpot.h>
#include <SLArrow.h>
#include <SLCoordAxis.h>
#include <sens/SENSSimHelper.h>
#include <SLDeviceLocation.h>
#include <sens/SENSARCore.h>

class SENSCamera;
class MapLoader;

class AreaTrackingView : public SLSceneView
{
public:
    AreaTrackingView(sm::EventHandler&  eventHandler,
                     SLInputManager&    inputManager,
                     const ImGuiEngine& imGuiEngine,
                     ErlebAR::Config&   config,
                     SENSCamera*        camera,
                     SENSGps*           gps,
                     SENSOrientation*   orientation,
                     SENSARCore*        arcore,
                     const DeviceData&  deviceData);
    ~AreaTrackingView();

    void initArea(ErlebAR::LocationId locId, ErlebAR::AreaId areaId);
    bool update();
    //call when view becomes visible
    void onShow();
    void onHide();

    void resume();
    void hold();

    static std::unique_ptr<WAIMap> tryLoadMap(const std::string& erlebARDir,
                                              const std::string& slamMapFileName,
                                              WAIOrbVocabulary*  voc,
                                              cv::Mat&           mapNodeOm);

    SENSSimHelper* getSimHelper() { return _simHelper.get(); }

private:
    virtual SLbool onMouseDown(SLMouseButton button, SLint scrX, SLint scrY, SLKey mod);
    virtual SLbool onMouseMove(SLint x, SLint y);

    bool updateGPSARCore(SENSFramePtr& frame);
    bool updateGPSWAISlamARCore(SENSFramePtr& frame);

    void    updateSceneCameraFov();
    void    updateVideoImage(SENSFrame& frame, VideoBackgroundCamera* videoBackground);
    void    updateTrackingVisualization(const bool iKnowWhereIAm, SENSFrame& frame);
    void    initDeviceLocation(const ErlebAR::Location& location, const ErlebAR::Area& area);
    void    initSlam(const ErlebAR::Area& area);
    void    initWaiSlam(const cv::Mat& mapNodeOm, std::unique_ptr<WAIMap> waiMap);
    bool    startCamera(const cv::Size& trackImgSize);
    void    onCameraParamsChanged();
    SLMat4f calcCameraPoseGpsOrientationBased();
    cv::Mat convertCameraPoseToWaiCamExtrinisc(SLMat4f& wTc);

    AreaTrackingGui   _gui;
    AppWAIScene       _waiScene;
    UserGuidanceScene _userGuidanceScene;

    std::map<ErlebAR::LocationId, ErlebAR::Location> _locations;

    std::unique_ptr<SENSCvCamera> _camera;
    SENSGps*                      _gps         = nullptr;
    SENSOrientation*              _orientation = nullptr;
    SENSARCore*                   _arcore      = nullptr;

    FeatureExtractorFactory      _featureExtractorFactory;
    std::unique_ptr<KPextractor> _trackingExtractor;
    std::unique_ptr<KPextractor> _initializationExtractor;
    std::unique_ptr<KPextractor> _relocalizationExtractor;
    ImageBuffer                  _imgBuffer;
    SLMat4f                      _transitionMatrix;
    bool                         _hasTransitionMatrix;
    bool                         _hasWaiMap;
    float                        _initTime;

    std::unique_ptr<WAIOrbVocabulary> _voc;
    //wai slam depends on _orbVocabulary and has to be uninitializd first (defined below voc)
    std::unique_ptr<WAISlam> _waiSlam;

#if USE_FBOW
    std::string _vocabularyFileName = "voc_fbow.bin";
#else

    std::string _vocabularyFileName = "ORBvoc.bin";
#endif
    std::string _mapFileName;

    //size with which camera was started last time (needed for a resume call)
    UserGuidance _userGuidance;

    std::unique_ptr<MapLoader> _asyncLoader;

    ErlebAR::Config&    _config;
    ErlebAR::Resources& _resources;
    const DeviceData&   _deviceData;

    ErlebAR::LocationId _locId  = ErlebAR::LocationId::NONE;
    ErlebAR::AreaId     _areaId = ErlebAR::AreaId::NONE;

    std::unique_ptr<SENSSimHelper> _simHelper;

    SLDeviceLocation _devLoc;
    //indicates if intArea finished successfully
    bool _noInitException = false;
};

//! Async loader for vocabulary and maps
class MapLoader : public AsyncWorker
{
public:
    MapLoader(int                vocLayer,
              const std::string& vocFileName,
              const std::string& mapFileDir,
              const std::string& mapFileName)
      : _vocLayer(vocLayer),
        _vocFileName(vocFileName),
        _mapFileDir(mapFileDir),
        _mapFileName(mapFileName)
    {
    }

    void run() override
    {
        //if vocabulary is empty, load it first
        if (Utils::fileExists(_vocFileName))
        {
            Utils::log("MapLoader", "loading voc file from: %s", _vocFileName.c_str());
            _voc = std::make_unique<WAIOrbVocabulary>(_vocLayer);
            _voc->loadFromFile(_vocFileName);

            //load map
            _waiMap = AreaTrackingView::tryLoadMap(_mapFileDir, _mapFileName, _voc.get(), _mapNodeOm);
        }
        else
        {
            std::stringstream ss;
            ss << "MapLoader: vocabulary file does not exist: " << _vocFileName;
            _errorMsg = ss.str();
            _hasError = true;
        }

        //task is ready
        setReady();
    }

    std::unique_ptr<WAIMap>           moveWaiMap() { return std::move(_waiMap); }
    std::unique_ptr<WAIOrbVocabulary> voc() { return std::move(_voc); }
    cv::Mat                           mapNodeOm() { return _mapNodeOm; }

    bool               hasError() const { return _hasError; }
    const std::string& getErrorMsg() const { return _errorMsg; }

private:
    std::unique_ptr<WAIOrbVocabulary> _voc;
    std::string                       _vocFileName;
    int                               _vocLayer;
    std::string                       _mapFileDir;
    std::string                       _mapFileName;

    std::string _errorMsg;
    bool        _hasError = false;

    std::unique_ptr<WAIMap> _waiMap;
    cv::Mat                 _mapNodeOm;
};

#endif //AREA_TRACKING_VIEW_H
