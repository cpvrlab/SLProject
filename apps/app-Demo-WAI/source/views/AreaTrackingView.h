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
#include <UserGuidance.h>

#include <SLLightSpot.h>
#include <SLArrow.h>
#include <SLCoordAxis.h>

class SENSCamera;
class MapLoader;

class DirectionArrow : public SLNode
{
public:
    DirectionArrow(SLAssetManager& assets, std::string name, SLCamera* cameraNode)
      : SLNode(name)
    {
        SLMaterial* blueMat = new SLMaterial(&assets, "m2", SLCol4f::BLUE * 0.3f, SLCol4f::BLUE, 128, 0.5f, 0.0f, 1.0f);

        //define arrow size in the middle of camera frustum
        float   distance            = (cameraNode->clipFar() - cameraNode->clipNear()) * 0.5f;
        SLVec2i frustumSize         = cameraNode->frustumSizeAtDistance(distance);
        SLfloat length              = (float)frustumSize.x / 10.f;
        SLfloat arrowCylinderRadius = length * 1.5f / 5.f;
        SLfloat headLength          = length * 2.f / 5.f;
        SLfloat headWidth           = length * 3.f / 5.f;
        SLuint  slices              = 20;
        SLNode* arrowNode           = new SLNode(new SLArrow(&assets, arrowCylinderRadius, length, headLength, headWidth, slices, "ArrowMesh", blueMat), "ArrowNode");
        arrowNode->rotate(-90, {0, 1, 0});
        arrowNode->translate(0, 0, -length * 0.5f);

        //coordinate axis
        //SLNode* axisNode = new SLNode(new SLCoordAxis(s), "AxisNode");
        //setup final direction arrow
        //directionArrow->addChild(axisNode);
        addChild(arrowNode);
        translate(0, 0, -distance);
    }

private:
};

class TestScene : public SLScene
{
public:
    TestScene(std::string dataDir)
        : SLScene("TestScene", nullptr),
          _dataDir(dataDir)
    {
        // Create textures and materials
        //SLGLTexture* texC = new SLGLTexture(s, SLApplication::texturePath + "earth1024_C.jpg");
        //SLMaterial*  m1   = new SLMaterial(&_assets, "m1", texC);
        //SLMaterial* blueMat = new SLMaterial(&_assets, "m2", SLCol4f::BLUE * 0.3f, SLCol4f::BLUE, 128, 0.5f, 0.0f, 1.0f);
        
        // Create a scene group node
        SLNode* scene = new SLNode("scene node");

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(&_assets, this, 0.3f);
        light1->translation(0, 0, 5);
        light1->lookAt(0, 0, 0);
        light1->name("light node");
        scene->addChild(light1);

        // Create meshes and nodes
        //SLMesh* rectMesh = new SLRectangle(&_assets, SLVec2f(-5, -5), SLVec2f(5, 5), 25, 25, "rectangle mesh", blueMat);
        //SLNode* rectNode = new SLNode(rectMesh, "rectangle node");
        //rectNode->translation(0, 0, -10);
        //scene->addChild(rectNode);
               
        // Set background color and the root scene node
        //sv->sceneViewCamera()->background().colors(SLCol4f(0.7f, 0.7f, 0.7f), SLCol4f(0.2f, 0.2f, 0.2f));
        camera = new SLCamera();
        camera->translation(0, 0, 0.f);
        camera->lookAt(0, 0, -1);
        //for tracking we have to use the field of view from calibration
        camera->clipNear(0.1f);
        camera->clipFar(1000.0f); // Increase to infinity?
        camera->focalDist(0);
        camera->setInitialState();
        //camera->background().colors(SLCol4f(0.7f, 0.7f, 0.7f), SLCol4f(0.2f, 0.2f, 0.2f));
        
        _videoImage = new SLGLTexture(&_assets, _dataDir + "images/textures/LiveVideoError.png", GL_LINEAR, GL_LINEAR);
        camera->background().texture(_videoImage, false);
        scene->addChild(camera);
        
        _dirArrow = new DirectionArrow(_assets, "DirArrow", camera);
        camera->addChild(_dirArrow);
        
        // pass the scene group as root node
        root3D(scene);
    }
    
    void updateArrowRot(SLMat3f camRarrow)
    {
        SLMat4f cTa;
        cTa.setTranslation(_dirArrow->om().translation());
        cTa.setRotation(camRarrow);
        _dirArrow->om(cTa);
    }
    
    void updateCameraIntrinsics(float cameraFovVDeg)
    {
        camera->fov(cameraFovVDeg);
        // Set camera intrinsics for scene camera frustum. (used in projection->intrinsics mode)
        //std::cout << "cameraMatUndistorted: " << cameraMatUndistorted << std::endl;
        /*
        cameraNode->intrinsics((float)cameraMatUndistorted.at<double>(0, 0),
                               (float)cameraMatUndistorted.at<double>(1, 1),
                               (float)cameraMatUndistorted.at<double>(0, 2),
                               (float)cameraMatUndistorted.at<double>(1, 2));
        */
        //enable projection -> intrinsics mode
        //cameraNode->projection(P_monoIntrinsic);
        camera->projection(P_monoPerspective);
    }
    
    void updateVideoImage(const cv::Mat& image)
    {
        _videoImage->copyVideoImage(image.cols,
                                    image.rows,
                                    CVImage::cv2glPixelFormat(image.type()),
                                    image.data,
                                    image.isContinuous(),
                                    true);
    }
    
    SLCamera* camera;

private:
    std::string _dataDir;
    
    DirectionArrow* _dirArrow;
    SLGLTexture* _videoImage;
    
    SLAssetManager _assets;
};

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
        if(_gps)
            _gps->start();
        if(_orientation)
            _orientation->start();
    }

    void onHide()
    {
        if(_gps)
            _gps->stop();
        if(_orientation)
            _orientation->stop();
    }

    void initArea(ErlebAR::LocationId locId, ErlebAR::AreaId areaId);

    void resume();
    void hold();

    static std::unique_ptr<WAIMap> tryLoadMap(const std::string& erlebARDir,
                                              const std::string& slamMapFileName,
                                              WAIOrbVocabulary*  voc,
                                              cv::Mat&           mapNodeOm);

private:
    virtual SLbool onMouseDown(SLMouseButton button, SLint scrX, SLint scrY, SLKey mod);
    virtual SLbool onMouseMove(SLint x, SLint y);

    void updateSceneCameraFov();
    void updateVideoImage(SENSFrame& frame);
    void updateTrackingVisualization(const bool iKnowWhereIAm, SENSFrame& frame);

    bool startCamera(const cv::Size& cameraFrameTargetSize);

    AreaTrackingGui _gui;
    AppWAIScene     _scene;
    TestScene       _testScene;

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
    std::string _vocabularyDir;
    std::string _erlebARDir;
    std::string _mapFileName;

    //size with which camera was started last time (needed for a resume call)
    cv::Size _cameraFrameResumeSize;
    UserGuidance _userGuidance;

    MapLoader* _asyncLoader = nullptr;

    ErlebAR::Resources& _resources;

    ErlebAR::LocationId _locId  = ErlebAR::LocationId::NONE;
    ErlebAR::AreaId     _areaId = ErlebAR::AreaId::NONE;
    
    bool _userGuidanceMode = false;
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
