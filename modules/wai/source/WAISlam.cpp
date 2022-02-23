#include <WAISlam.h>
#include <AverageTiming.h>
#include <Utils.h>

#define MIN_FRAMES 0
#define MAX_FRAMES 30
//#define MULTI_MAPPING_THREADS 1
#define MULTI_THREAD_FRAME_PROCESSING 1

#define LOG_WAISLAM_WARN(...) Utils::log("WAISlam", __VA_ARGS__);
#define LOG_WAISLAM_INFO(...) Utils::log("WAISlam", __VA_ARGS__);
#define LOG_WAISLAM_DEBUG(...) Utils::log("WAISlam", __VA_ARGS__);

//-----------------------------------------------------------------------------
WAISlam::WAISlam(const cv::Mat&          intrinsic,
                 const cv::Mat&          distortion,
                 WAIOrbVocabulary*       voc,
                 KPextractor*            iniExtractor,
                 KPextractor*            relocExtractor,
                 KPextractor*            extractor,
                 std::unique_ptr<WAIMap> globalMap,
                 WAISlam::Params         params)
{
    _iniData.initializer = nullptr;
    _params              = params;

    WAIFrame::nNextId               = 0;
    WAIFrame::mbInitialComputations = true;

    _lastKeyFrameFrameId = 0;
    _lastRelocFrameId    = 0;

    _distortion      = distortion.clone();
    _cameraIntrinsic = intrinsic.clone();
    _voc             = voc;

    _extractor      = extractor;
    _relocExtractor = relocExtractor;
    _iniExtractor   = iniExtractor;

    if (_iniExtractor == nullptr)
        _iniExtractor = _extractor;
    if (_relocExtractor == nullptr)
        _relocExtractor = _extractor;

    if (globalMap == nullptr)
    {
        WAIKeyFrameDB* kfDB = new WAIKeyFrameDB(voc);
        _globalMap          = std::make_unique<WAIMap>(kfDB);
        _state              = WAITrackingState::Initializing;

        WAIKeyFrame::nNextId = 0;
        WAIMapPoint::nNextId = 0;
    }
    else
    {
        _globalMap   = std::move(globalMap);
        _state       = WAITrackingState::TrackingLost;
        _initialized = true;
    }

    _localMapping = new ORB_SLAM2::LocalMapping(_globalMap.get(),
                                                _voc,
                                                params.cullRedundantPerc);
    _loopClosing  = new ORB_SLAM2::LoopClosing(_globalMap.get(),
                                              _voc,
                                              _params.minCommonWordFactor,
                                              false,
                                              false);

    _localMapping->SetLoopCloser(_loopClosing);
    _loopClosing->SetLocalMapper(_localMapping);

    if (!_params.onlyTracking && !_params.serial)
    {
        _mappingThreads.push_back(new std::thread(&LocalMapping::Run, _localMapping));
        _loopClosingThread = new std::thread(&LoopClosing::Run, _loopClosing);
    }

#if MULTI_THREAD_FRAME_PROCESSING
    _poseUpdateThread = new std::thread(updatePoseThread, this);
    _isFinish         = false;
    _isStop           = false;
    _requestFinish    = false;
#endif

    _iniData.initializer = nullptr;
    _cameraExtrinsic     = cv::Mat::eye(4, 4, CV_32F);

    _lastFrame = WAIFrame();
}
//-----------------------------------------------------------------------------
WAISlam::~WAISlam()
{
    if (!_params.serial)
    {
        _localMapping->RequestFinish();
        _loopClosing->RequestFinish();
    }

    // Wait until all thread have effectively stopped
    if (_processNewKeyFrameThread)
        _processNewKeyFrameThread->join();

    for (std::thread* t : _mappingThreads) { t->join(); }

    if (_loopClosingThread)
        _loopClosingThread->join();

#if MULTI_THREAD_FRAME_PROCESSING
    requestFinish();
    _poseUpdateThread->join();
    delete _poseUpdateThread;
    _poseUpdateThread = nullptr;
#endif

    delete _localMapping;
    delete _loopClosing;
}
//-----------------------------------------------------------------------------
void WAISlam::reset()
{
    if (!_params.serial)
    {
        _localMapping->RequestReset();
        _loopClosing->RequestReset();
    }

#if MULTI_THREAD_FRAME_PROCESSING
    requestFinish();
    _poseUpdateThread->join();
#endif

    _globalMap->clear();
    _localMap.keyFrames.clear();
    _localMap.mapPoints.clear();
    _localMap.refKF = nullptr;

    _lastKeyFrameFrameId = 0;
    _lastRelocFrameId    = 0;

#if MULTI_THREAD_FRAME_PROCESSING
    _poseUpdateThread = new std::thread(updatePoseThread, this);
    _isFinish         = false;
    _isStop           = false;
    _requestFinish    = false;
#endif

    WAIKeyFrame::nNextId            = 0;
    WAIFrame::nNextId               = 0;
    WAIFrame::mbInitialComputations = true;
    WAIMapPoint::nNextId            = 0;
    _state                          = WAITrackingState::Initializing;
}
//-----------------------------------------------------------------------------
void WAISlam::changeIntrinsic(cv::Mat intrinsic, cv::Mat distortion)
{
    _cameraIntrinsic = intrinsic;
    _distortion      = distortion;
}
//-----------------------------------------------------------------------------
void WAISlam::createFrame(WAIFrame& frame, cv::Mat& imageGray)
{
    switch (getTrackingState())
    {
        case WAITrackingState::Initializing:
            frame = WAIFrame(imageGray,
                             0.0,
                             _iniExtractor,
                             _cameraIntrinsic,
                             _distortion,
                             _voc,
                             _params.retainImg);
            break;
        case WAITrackingState::TrackingLost:
        case WAITrackingState::TrackingStart:
            frame = WAIFrame(imageGray,
                             0.0,
                             _relocExtractor,
                             _cameraIntrinsic,
                             _distortion,
                             _voc,
                             _params.retainImg);
            break;
        default:
            frame = WAIFrame(imageGray,
                             0.0,
                             _extractor,
                             _cameraIntrinsic,
                             _distortion,
                             _voc,
                             _params.retainImg);
    }
}
//-----------------------------------------------------------------------------
/* Separate Pose update thread */
void WAISlam::flushQueue()
{
    std::unique_lock<std::mutex> lock(_lastFrameMutex);
    while (!_framesQueue.empty())
    {
        _framesQueue.pop();
    }
}
//-----------------------------------------------------------------------------
void WAISlam::updateState(WAITrackingState state)
{
    std::unique_lock<std::mutex> lock(_mutexStates);
    _state = state;
}
//-----------------------------------------------------------------------------
int WAISlam::getNextFrame(WAIFrame& frame)
{
    int                          nbFrameInQueue;
    std::unique_lock<std::mutex> lock(_frameQueueMutex);
    nbFrameInQueue = (int)_framesQueue.size();
    if (nbFrameInQueue == 0)
        return 0;

    frame = _framesQueue.front();
    _framesQueue.pop();
    return nbFrameInQueue;
}
//-----------------------------------------------------------------------------
void WAISlam::requestFinish()
{
    std::unique_lock<std::mutex> lock(_stateMutex);
    _requestFinish = true;
}
//-----------------------------------------------------------------------------
bool WAISlam::finishRequested()
{
    std::unique_lock<std::mutex> lock(_stateMutex);
    return _requestFinish;
}

bool WAISlam::isFinished()
{
    std::unique_lock<std::mutex> lock(_stateMutex);
    return _isFinish;
}
//-----------------------------------------------------------------------------
bool WAISlam::isStop()
{
    std::unique_lock<std::mutex> lock(_stateMutex);
    return _isStop;
}
//-----------------------------------------------------------------------------
void WAISlam::resume()
{
    std::unique_lock<std::mutex> lock(_stateMutex);
    _isStop = false;
    _localMapping->RequestContinue();
    _state = WAITrackingState::TrackingLost;
}
//-----------------------------------------------------------------------------
void WAISlam::updatePoseThread(WAISlam* ptr)
{
    while (1)
    {
        WAIFrame f;
        while (ptr->getNextFrame(f) && !ptr->finishRequested())
        {
            if (ptr->_params.ensureKFIntegration)
                ptr->updatePoseKFIntegration(f);
            else
                ptr->updatePose(f);
        }

        if (ptr->finishRequested())
        {
            std::unique_lock<std::mutex> lock(ptr->_frameQueueMutex);

            while (!ptr->_framesQueue.empty())
                ptr->_framesQueue.pop();

            break;
        }

        while (ptr->isStop() && !ptr->isFinished() && !ptr->finishRequested())
        {
            std::this_thread::sleep_for(25ms);
        }
    }

    std::unique_lock<std::mutex> lock(ptr->_stateMutex);
    ptr->_requestFinish = false;
    ptr->_isFinish      = true;
}
//-----------------------------------------------------------------------------
void WAISlam::updatePose(WAIFrame& frame)
{
    std::unique_lock<std::mutex> guard(_mutexStates);

    switch (_state)
    {
        case WAITrackingState::Initializing:
        {
#if 0
            bool ok = oldInitialize(frame, _iniData, _globalMap.get(), _localMap, _localMapping, _loopClosing, _voc);
            if (ok)
            {
                _lastKeyFrameFrameId = frame.mnId;
                _lastRelocFrameId    = 0;
                _state               = WAI::TrackingOK;
                _initialized         = true;
            }
#else
            if (initialize(_iniData, frame, _voc, _localMap))
            {
                if (genInitialMap(_globalMap.get(), _localMapping, _loopClosing, _localMap))
                {
                    _localMapping->InsertKeyFrame(_localMap.keyFrames[0]);
                    _localMapping->InsertKeyFrame(_localMap.keyFrames[1]);
                    _lastKeyFrameFrameId = frame.mnId;
                    _lastRelocFrameId    = 0;
                    _state               = WAITrackingState::TrackingOK;
                    _initialized         = true;
                    if (_params.serial)
                    {
                        _localMapping->RunOnce();
                        _localMapping->RunOnce();
                    }
                }
            }
#endif
        }
        break;
        case WAITrackingState::TrackingStart:
        {
            _relocFrameCounter++;
            if (_relocFrameCounter > 30)
                _state = WAITrackingState::TrackingOK;
        }
        case WAITrackingState::TrackingOK:
        {
            int inliers;
            if (tracking(_globalMap.get(),
                         _localMap,
                         frame,
                         _lastFrame,
                         (int)_lastRelocFrameId,
                         _velocity,
                         inliers))
            {
                std::unique_lock<std::mutex> lock(_cameraExtrinsicMutex);
                motionModel(frame,
                            _lastFrame,
                            _velocity,
                            _cameraExtrinsic);
                lock.unlock();
                mapping(_globalMap.get(),
                        _localMap,
                        _localMapping,
                        frame,
                        inliers,
                        _lastRelocFrameId,
                        _lastKeyFrameFrameId);
                if (_params.serial)
                {
                    _localMapping->RunOnce();
                    _loopClosing->RunOnce();
                }
                _infoMatchedInliners = inliers;
            }
            else
            {
                _state = WAITrackingState::TrackingLost;
            }
        }
        break;
        case WAITrackingState::TrackingLost:
        {
            int inliers;
            if (relocalization(frame,
                               _globalMap.get(),
                               _localMap,
                               _params.minCommonWordFactor,
                               inliers,
                               _params.minAccScoreFilter))
            {
                _relocFrameCounter = 0;
                _lastRelocFrameId  = frame.mnId;
                _velocity          = cv::Mat();
                _state             = WAITrackingState::TrackingStart;
            }
        }
        break;
    }

    std::unique_lock<std::mutex> lock(_lastFrameMutex);
    _lastFrame = WAIFrame(frame);
}
//-----------------------------------------------------------------------------
void WAISlam::updatePoseKFIntegration(WAIFrame& frame)
{
    switch (_state)
    {
        case WAITrackingState::TrackingOK:
        {
            int inliers;
            if (tracking(_globalMap.get(),
                         _localMap,
                         frame,
                         _lastFrame,
                         (int)_lastRelocFrameId,
                         _velocity,
                         inliers))
            {
                std::unique_lock<std::mutex> lock(_cameraExtrinsicMutex);
                motionModel(frame,
                            _lastFrame,
                            _velocity,
                            _cameraExtrinsic);
                lock.unlock();
                strictMapping(_globalMap.get(),
                              _localMap,
                              _localMapping,
                              frame,
                              inliers,
                              _lastRelocFrameId,
                              _lastKeyFrameFrameId);
                if (_params.serial)
                {
                    _localMapping->RunOnce();
                    _loopClosing->RunOnce();
                }
                _infoMatchedInliners = inliers;
            }
            else
            {
                _state = WAITrackingState::TrackingLost;
            }
        }
        break;
        case WAITrackingState::TrackingLost:
        {
            int inliers;
            if (relocalization(frame,
                               _globalMap.get(),
                               _localMap,
                               _params.minCommonWordFactor,
                               inliers,
                               _params.minAccScoreFilter))
            {
                _lastRelocFrameId    = frame.mnId;
                _velocity            = cv::Mat();
                _state               = WAITrackingState::TrackingOK;
                _infoMatchedInliners = inliers;
            }
        }
        break;
    }

    std::unique_lock<std::mutex> lock(_lastFrameMutex);
    _lastFrame = WAIFrame(frame);
}
//-----------------------------------------------------------------------------
WAIFrame WAISlam::getLastFrame()
{
    std::unique_lock<std::mutex> lock(_lastFrameMutex);
    return _lastFrame;
}
//-----------------------------------------------------------------------------
WAIFrame* WAISlam::getLastFramePtr()
{
    std::unique_lock<std::mutex> lock(_lastFrameMutex);
    return &_lastFrame;
}
//-----------------------------------------------------------------------------
bool WAISlam::update(cv::Mat& imageGray)
{
    WAIFrame frame;
    createFrame(frame, imageGray);

#if MULTI_THREAD_FRAME_PROCESSING
    std::unique_lock<std::mutex> lock(_frameQueueMutex);
    _framesQueue.push(frame);
#else
    if (_params.ensureKFIntegration)
        updatePoseKFIntegration(frame);
    else
        updatePose(frame);
#endif
    return isTracking();
}
//-----------------------------------------------------------------------------
void WAISlam::drawInfo(cv::Mat& imageBGR,
                       float    scale,
                       bool     showInitLine,
                       bool     showKeyPoints,
                       bool     showKeyPointsMatched)
{
    std::unique_lock<std::mutex> lock(_lastFrameMutex);
    std::unique_lock<std::mutex> lock2(_stateMutex);
    if (_state == WAITrackingState::Initializing)
    {
        if (showInitLine)
            drawInitInfo(_iniData, _lastFrame, imageBGR, scale);
    }
    else if (_state == WAITrackingState::TrackingOK)
    {
        if (showKeyPoints)
            drawKeyPointInfo(_lastFrame, imageBGR, scale);
        if (showKeyPointsMatched)
            drawKeyPointMatches(_lastFrame, imageBGR, scale);
    }
    else if (_state == WAITrackingState::TrackingLost)
    {
        if (showKeyPoints)
            drawKeyPointInfo(_lastFrame, imageBGR, scale);
    }
}
//-----------------------------------------------------------------------------
std::vector<WAIMapPoint*> WAISlam::getMatchedMapPoints(WAIFrame* frame)
{
    std::vector<WAIMapPoint*> result;

    for (int i = 0; i < frame->N; i++)
    {
        if (frame->mvpMapPoints[i])
        {
            if (frame->mvpMapPoints[i]->Observations() > 0)
                result.push_back(frame->mvpMapPoints[i]);
        }
    }

    return result;
}
//-----------------------------------------------------------------------------
int WAISlam::getMatchedCorrespondances(WAIFrame*                            frame,
                                       std::pair<std::vector<cv::Point2f>,
                                                 std::vector<cv::Point3f>>& matching)
{
    for (int i = 0; i < frame->N; i++)
    {
        WAIMapPoint* mp = frame->mvpMapPoints[i];
        if (mp)
        {
            if (mp->isFixed())
            {
                WAI::V3 v = mp->worldPosVec();
                matching.first.push_back(frame->mvKeysUn[i].pt);
                matching.second.push_back(cv::Point3f(v.x, v.y, v.z));
            }
        }
    }
    return (int)matching.first.size();
}
//-----------------------------------------------------------------------------
cv::Mat WAISlam::getPose()
{
    std::unique_lock<std::mutex> lock(_cameraExtrinsicMutex);
    return _cameraExtrinsic;
}
//-----------------------------------------------------------------------------
void WAISlam::setCamExrinsicGuess(cv::Mat extrinsicGuess)
{
    std::unique_lock<std::mutex> lock(_cameraExtrinsicGuessMutex);
    _cameraExtrinsicGuess = extrinsicGuess;
}
//-----------------------------------------------------------------------------
void WAISlam::requestStateIdle()
{
    if (!(_params.onlyTracking || _params.serial))
    {
        std::unique_lock<std::mutex> guard(_mutexStates);
        _localMapping->RequestPause();
        while (!_localMapping->isPaused())
        {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        std::cout << "localMapping is stopped" << std::endl;
    }

    _state = WAITrackingState::Idle;
}
//-----------------------------------------------------------------------------
bool WAISlam::hasStateIdle()
{
    std::unique_lock<std::mutex> guard(_mutexStates);
    return (_state == WAITrackingState::Idle);
}
//-----------------------------------------------------------------------------
bool WAISlam::isTracking()
{
    std::unique_lock<std::mutex> guard(_mutexStates);
    return _state == WAITrackingState::TrackingOK;
}
//-----------------------------------------------------------------------------
bool WAISlam::retainImage()
{
    return false;
}
//-----------------------------------------------------------------------------
void WAISlam::transformCoords(cv::Mat transform)
{
    if (_loopClosingThread != nullptr)
    {
        _localMapping->RequestPause();
        while (!_localMapping->isPaused() && !_localMapping->isFinished())
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    WAIMap* map = _globalMap.get();

    map->transform(transform);

    _initialized = true;
    _localMapping->RequestContinue();
}
//-----------------------------------------------------------------------------
void WAISlam::setMap(std::unique_ptr<WAIMap> globalMap)
{
    requestStateIdle();
    reset();
    _globalMap   = std::move(globalMap);
    _initialized = true;
    resume();
}
//-----------------------------------------------------------------------------
int WAISlam::getMapPointMatchesCount() const
{
    return _infoMatchedInliners;
}
//-----------------------------------------------------------------------------
std::string WAISlam::getLoopCloseStatus()
{
    return _loopClosing->getStatusString();
}
//-----------------------------------------------------------------------------
int WAISlam::getLoopCloseCount()
{
    return _globalMap->getNumLoopClosings();
}
//-----------------------------------------------------------------------------
int WAISlam::getKeyFramesInLoopCloseQueueCount()
{
    return _loopClosing->numOfKfsInQueue();
}
//-----------------------------------------------------------------------------
