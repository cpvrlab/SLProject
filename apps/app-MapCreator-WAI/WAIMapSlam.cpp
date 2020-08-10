#include <WAIMapSlam.h>
#include <AverageTiming.h>
#include <Utils.h>

#define MIN_FRAMES 0
#define MAX_FRAMES 30

WAIMapSlam::WAIMapSlam(const cv::Mat&          intrinsic,
                       const cv::Mat&          distortion,
                       WAIOrbVocabulary*       voc,
                       KPextractor*            extractor,
                       std::unique_ptr<WAIMap> globalMap,
                       WAIMapSlam::Params      p)
{
    _iniData.initializer = nullptr;
    _params = p;

    WAIFrame::nNextId               = 0;
    WAIFrame::mbInitialComputations = true;

    _lastKeyFrameFrameId = 0;
    _lastRelocFrameId    = 0;

    _distortion      = distortion.clone();
    _cameraIntrinsic = intrinsic.clone();
    _voc             = voc;
    _extractor       = extractor;

    if (globalMap == nullptr)
    {
        WAIKeyFrameDB* kfDB = new WAIKeyFrameDB(voc);
        _globalMap          = std::make_unique<WAIMap>(kfDB);
        _state              = WAI::TrackingState_Initializing;

        WAIKeyFrame::nNextId = 0;
        WAIMapPoint::nNextId = 0;
    }
    else
    {
        _globalMap   = std::move(globalMap);
        _state       = WAI::TrackingState_TrackingLost;
        _initialized = true;
    }

    _localMapping = new ORB_SLAM2::LocalMapping(_globalMap.get(), _voc, _params.cullRedundantPerc);
    _loopClosing  = new ORB_SLAM2::LoopClosing(_globalMap.get(), _voc, false, false);

    _localMapping->SetLoopCloser(_loopClosing);
    _loopClosing->SetLocalMapper(_localMapping);

    if (!_params.serial)
    {
        _mappingThreads.push_back(new std::thread(&LocalMapping::Run, _localMapping));
        _loopClosingThread = new std::thread(&LoopClosing::Run, _loopClosing);
    }

    _iniData.initializer = nullptr;
    _cameraExtrinsic     = cv::Mat::eye(4, 4, CV_32F);

    _lastFrame = WAIFrame();
}

WAIMapSlam::~WAIMapSlam()
{
    if (!_params.serial)
    {
        _localMapping->RequestFinish();
        _loopClosing->RequestFinish();

        // Wait until all thread have effectively stopped
        if (_processNewKeyFrameThread)
            _processNewKeyFrameThread->join();

        for (std::thread* t : _mappingThreads) { t->join(); }

        if (_loopClosingThread)
            _loopClosingThread->join();
    }

    delete _localMapping;
    delete _loopClosing;
}

void WAIMapSlam::reset()
{
    if (!_params.serial)
    {
        _localMapping->RequestReset();
        _loopClosing->RequestReset();
    }
    else
    {
        _localMapping->Reset();
        _loopClosing->reset();
    }

    _globalMap->clear();
    _localMap.keyFrames.clear();
    _localMap.mapPoints.clear();
    _localMap.refKF = nullptr;

    _lastKeyFrameFrameId = 0;
    _lastRelocFrameId    = 0;

    WAIKeyFrame::nNextId            = 0;
    WAIFrame::nNextId               = 0;
    WAIFrame::mbInitialComputations = true;
    WAIMapPoint::nNextId            = 0;
    _state                          = WAI::TrackingState_Initializing;
}

void WAIMapSlam::changeIntrinsic(cv::Mat intrinsic, cv::Mat distortion)
{
    _cameraIntrinsic = intrinsic;
    _distortion      = distortion;
}

void WAIMapSlam::createFrame(WAIFrame& frame, cv::Mat& imageGray)
{
    frame = WAIFrame(imageGray, 0.0, _extractor, _cameraIntrinsic, _distortion, _voc, _retainImg);
}

void WAIMapSlam::updatePose(WAIFrame& frame)
{
    switch (_state)
    {
        case WAI::TrackingState_Initializing: {

            if (initialize(_iniData, frame, _voc, _localMap))
            {
                if (genInitialMap(_globalMap.get(), _localMapping, _loopClosing, _localMap, _params.serial))
                {
                    _lastKeyFrameFrameId = frame.mnId;
                    _lastRelocFrameId    = 0;
                    _state               = WAI::TrackingState_TrackingOK;
                    _initialized         = true;
                }
            }
        }
        break;
        case WAI::TrackingState_TrackingOK: {
            int inliers;
            if (tracking(_globalMap.get(), _localMap, frame, _lastFrame, _lastRelocFrameId, _velocity, inliers))
            {
                std::unique_lock<std::mutex> lock(_cameraExtrinsicMutex);
                motionModel(frame, _lastFrame, _velocity, _cameraExtrinsic);
                lock.unlock();
                if (_params.serial)
                    serialMapping(_globalMap.get(), _localMap, _localMapping, _loopClosing, frame, inliers, _lastRelocFrameId, _lastKeyFrameFrameId);
                else
                    mapping(_globalMap.get(), _localMap, _localMapping, frame, inliers, _lastRelocFrameId, _lastKeyFrameFrameId);

                _infoMatchedInliners = inliers;
            }
            else
            {
                _state = WAI::TrackingState_TrackingLost;
            }
        }
        break;
        case WAI::TrackingState_TrackingLost: {
            int inliers;
            if (relocalization(frame, _globalMap.get(), _localMap, inliers))
            {
                _lastRelocFrameId    = frame.mnId;
                _velocity            = cv::Mat();
                _state               = WAI::TrackingState_TrackingOK;
                _infoMatchedInliners = inliers;
            }
        }
        break;
    }

    _lastFrame = WAIFrame(frame);
}

void WAIMapSlam::updatePose2(WAIFrame& frame)
{
    switch (_state)
    {
        case WAI::TrackingState_TrackingOK: {
            int inliers;
            if (tracking(_globalMap.get(), _localMap, frame, _lastFrame, _lastRelocFrameId, _velocity, inliers))
            {
                std::unique_lock<std::mutex> lock(_cameraExtrinsicMutex);
                motionModel(frame, _lastFrame, _velocity, _cameraExtrinsic);
                lock.unlock();
                if (_params.serial)
                    strictSerialMapping(_globalMap.get(), _localMap, _localMapping, _loopClosing, frame, inliers, _lastRelocFrameId, _lastKeyFrameFrameId);
                else
                    strictMapping(_globalMap.get(), _localMap, _localMapping, frame, inliers, _lastRelocFrameId, _lastKeyFrameFrameId);

                _infoMatchedInliners = inliers;
            }
            else
            {
                _state = WAI::TrackingState_TrackingLost;
            }
        }
        break;
        case WAI::TrackingState_TrackingLost: {
            int inliers;
            if (relocalization(frame, _globalMap.get(), _localMap, inliers))
            {
                _lastRelocFrameId    = frame.mnId;
                _velocity            = cv::Mat();
                _state               = WAI::TrackingState_TrackingOK;
                _infoMatchedInliners = inliers;
            }
        }
        break;
    }

    _lastFrame = WAIFrame(frame);
}

WAIFrame WAIMapSlam::getLastFrame()
{
    std::unique_lock<std::mutex> lock(_lastFrameMutex);
    return _lastFrame;
}

WAIFrame* WAIMapSlam::getLastFramePtr()
{
    return &_lastFrame;
}

bool WAIMapSlam::update(cv::Mat& imageGray)
{
    WAIFrame frame;
    createFrame(frame, imageGray);
    updatePose(frame);
    return isTracking();
}

bool WAIMapSlam::update2(cv::Mat& imageGray)
{
    WAIFrame frame;
    createFrame(frame, imageGray);
    updatePose2(frame);
    return isTracking();
}

void WAIMapSlam::drawInfo(cv::Mat& imageRGB,
                       float    scale,
                       bool     showInitLine,
                       bool     showKeyPoints,
                       bool     showKeyPointsMatched)
{
    std::unique_lock<std::mutex> lock(_lastFrameMutex);
    std::unique_lock<std::mutex> lock2(_stateMutex);
    if (_state == WAI::TrackingState_Initializing)
    {
        if (showInitLine)
            drawInitInfo(_iniData, _lastFrame, imageRGB, scale);
    }
    else if (_state == WAI::TrackingState_TrackingOK)
    {
        if (showKeyPoints)
            drawKeyPointInfo(_lastFrame, imageRGB, scale);
        if (showKeyPointsMatched)
            drawKeyPointMatches(_lastFrame, imageRGB, scale);
    }
    else if (_state == WAI::TrackingState_TrackingLost)
    {
        if (showKeyPoints)
            drawKeyPointInfo(_lastFrame, imageRGB, scale);
    }
}

cv::Mat WAIMapSlam::getPose()
{
    std::unique_lock<std::mutex> lock(_cameraExtrinsicMutex);
    return _cameraExtrinsic;
}

void WAIMapSlam::requestStateIdle()
{
    if (!_params.serial)
    {
        std::unique_lock<std::mutex> guard(_mutexStates);
        _localMapping->RequestPause();
        while (!_localMapping->isPaused())
        {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    _state = WAI::TrackingState_Idle;
}

bool WAIMapSlam::isTracking()
{
    std::unique_lock<std::mutex> guard(_mutexStates);
    return _state == WAI::TrackingState_TrackingOK;
}

void WAIMapSlam::setMap(std::unique_ptr<WAIMap> globalMap)
{
    requestStateIdle();
    reset();
    _globalMap   = std::move(globalMap);
    _initialized = true;
}

std::string WAIMapSlam::getLoopCloseStatus()
{
    return _loopClosing->getStatusString();
}

int WAIMapSlam::getLoopCloseCount()
{
    return _globalMap->getNumLoopClosings();
}

int WAIMapSlam::getKeyFramesInLoopCloseQueueCount()
{
    return _loopClosing->numOfKfsInQueue();
}
