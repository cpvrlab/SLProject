#ifndef AREA_TRACKING_VIEW_H
#define AREA_TRACKING_VIEW_H

#include <string>
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

class SENSCamera;

class AreaTrackingView : public SLSceneView
{
public:
    AreaTrackingView(sm::EventHandler&   eventHandler,
                     SLInputManager&     inputManager,
                     const ImGuiEngine&  imGuiEngine,
                     ErlebAR::Resources& resources,
                     SENSCamera*         camera,
                     const DeviceData&   deviceData);
    ~AreaTrackingView();

    bool update();
    //call when view becomes visible
    void show() { _gui.onShow(); }

    void initArea(ErlebAR::LocationId locId, ErlebAR::AreaId areaId);

    void resume();
    void hold();

private:
    void updateSceneCameraFov();
    void updateTrackingVisualization(const bool iKnowWhereIAm, SENSFrame& frame);

    bool startCamera();

    AreaTrackingGui _gui;
    AppWAIScene     _scene;

    std::map<ErlebAR::LocationId, ErlebAR::Location> _locations;

    SENSCamera* _camera = nullptr;

    FeatureExtractorFactory      _featureExtractorFactory;
    std::unique_ptr<KPextractor> _trackingExtractor;
    std::unique_ptr<KPextractor> _initializationExtractor;
    std::unique_ptr<KPextractor> _relocalizationExtractor;
    ImageBuffer                  _imgBuffer;
    WAIOrbVocabulary*            _voc;

    //wai slam depends on _orbVocabulary and has to be uninitializd first
    std::unique_ptr<WAISlam> _waiSlam;

    //std::unique_ptr<SENSCalibration> _calibration;

    //parameter:
    cv::Size      _cameraFrameTargetSize       = {640, 480};
    ExtractorType _initializationExtractorType = ExtractorType::ExtractorType_FAST_ORBS_2000;
    ExtractorType _relocalizationExtractorType = ExtractorType::ExtractorType_FAST_ORBS_2000;
    ExtractorType _trackingExtractorType       = ExtractorType::ExtractorType_FAST_ORBS_1000;

#if USE_FBOW
    std::string _vocabularyFileName = "voc_fbow.bin";
#else

    std::string _vocabularyFileName = "ORBvoc.bin";
#endif
    std::string _vocabularyDir;
    std::string _erlebARDir;
    std::string _mapFileName;

    //debug visualization
    bool _showKeyPoints        = false;
    bool _showKeyPointsMatched = true;
    bool _showMapPC            = true;
    bool _showMatchesPC        = true;
};

#endif //AREA_TRACKING_VIEW_H
