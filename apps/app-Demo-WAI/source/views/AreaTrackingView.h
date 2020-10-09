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
#include <WAISlamTrackPool.h>
#include <sens/SENSCalibration.h>
#include <WAIOrbVocabulary.h>
#include <sens/SENSFrame.h>
#include <sens/SENSGps.h>
#include <sens/SENSOrientation.h>
#include <AsyncWorker.h>
#include <scenes/UserGuidanceScene.h>
#include <UserGuidance.h>

#include <SLLightSpot.h>
#include <SLArrow.h>
#include <SLCoordAxis.h>
#include <sens/SENSSimHelper.h>

class SENSCamera;
class MapLoader;

class AreaTrackingView : public SLSceneView
{
public:
    AreaTrackingView(sm::EventHandler&   eventHandler,
                     SLInputManager&     inputManager,
                     const ImGuiEngine&  imGuiEngine,
                     ErlebAR::Resources& resources,
                     SENSCamera*         camera,
                     SENSGps*            gps,
                     SENSOrientation*    orientation,
                     const DeviceData&   deviceData);
    ~AreaTrackingView();

    bool update();
    //call when view becomes visible
    void onShow()
    {
        _gui.onShow();
        if (_gps)
            _gps->start();
        if (_orientation)
            _orientation->start();

        if (_resources.developerMode && _resources.simulatorMode)
        {
            if(_simHelper)
                _simHelper.reset();
            _simHelper = std::make_unique<SENSSimHelper>(_gps,
                                                         _orientation,
                                                         _camera,
                                                         _deviceData.writableDir() + "SENSSimData",
                                                         std::bind(&AreaTrackingView::onCameraParamsChanged, this));
        }
    }

    void onHide()
    {
        //reset user guidance and run it once
        _userGuidance.reset();

        if (_gps)
            _gps->stop();
        if (_orientation)
            _orientation->stop();
        if (_camera)
            _camera->stop();

        if (_simHelper)
            _simHelper.reset();
    }

    void initArea(ErlebAR::LocationId locId, ErlebAR::AreaId areaId);

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

    void updateSceneCameraFov();
    void updateVideoImage(SENSFrame& frame, VideoBackgroundCamera* videoBackground);
    void updateTrackingVisualization(const bool iKnowWhereIAm, SENSFrame& frame);
    void initSlam();
    bool startCamera(const cv::Size& cameraFrameTargetSize);
    void onCameraParamsChanged();

    AreaTrackingGui   _gui;
    AppWAIScene       _scene;
    UserGuidanceScene _userGuidanceScene;

    std::map<ErlebAR::LocationId, ErlebAR::Location> _locations;

    SENSCamera*      _camera      = nullptr;
    SENSGps*         _gps         = nullptr;
    SENSOrientation* _orientation = nullptr;

    FeatureExtractorFactory      _featureExtractorFactory;
    std::unique_ptr<KPextractor> _trackingExtractor;
    std::unique_ptr<KPextractor> _initializationExtractor;
    std::unique_ptr<KPextractor> _relocalizationExtractor;
    ImageBuffer                  _imgBuffer;
    WAIOrbVocabulary*            _voc = nullptr;

    //wai slam depends on _orbVocabulary and has to be uninitializd first
    std::unique_ptr<WAISlamTrackPool> _waiSlam;

#if USE_FBOW
    std::string _vocabularyFileName = "voc_fbow.bin";
#else

    std::string _vocabularyFileName = "ORBvoc.bin";
#endif
    std::string _mapFileName;

    //size with which camera was started last time (needed for a resume call)
    cv::Size     _cameraFrameResumeSize;
    UserGuidance _userGuidance;

    MapLoader* _asyncLoader = nullptr;

    ErlebAR::Resources& _resources;
    const DeviceData&   _deviceData;

    ErlebAR::LocationId _locId  = ErlebAR::LocationId::NONE;
    ErlebAR::AreaId     _areaId = ErlebAR::AreaId::NONE;

    std::unique_ptr<SENSSimHelper> _simHelper;
};

//! Async loader for vocabulary and maps
class MapLoader : public AsyncWorker
{
public:
    MapLoader(WAIOrbVocabulary*& voc,
              const std::string& vocFileName,
              const std::string& mapFileDir,
              const std::string& mapFileName)
      : _voc(voc),
        _vocFileName(vocFileName),
        _mapFileDir(mapFileDir),
        _mapFileName(mapFileName)
    {
    }

    void run() override
    {
        //if vocabulary is empty, load it first
        if (!_voc && Utils::fileExists(_vocFileName))
        {
            Utils::log("MapLoader", "loading voc file from: %s", _vocFileName.c_str());
            _voc = new WAIOrbVocabulary();
            _voc->loadFromFile(_vocFileName);
        }

        //load map
        _waiMap = AreaTrackingView::tryLoadMap(_mapFileDir, _mapFileName, _voc, _mapNodeOm);

        //task is ready
        setReady();
    }

    std::unique_ptr<WAIMap> moveWaiMap()
    {
        return std::move(_waiMap);
    }

    cv::Mat mapNodeOm() { return _mapNodeOm; }

private:
    WAIOrbVocabulary*& _voc;
    std::string        _vocFileName;
    std::string        _mapFileDir;
    std::string        _mapFileName;

    std::unique_ptr<WAIMap> _waiMap;
    cv::Mat                 _mapNodeOm;
};

#endif //AREA_TRACKING_VIEW_H
