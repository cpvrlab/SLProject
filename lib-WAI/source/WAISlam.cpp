#include <WAISlam.h>
#include <WAIModeOrbSlam2.h>
#include <AverageTiming.h>
#include <Utils.h>
#include <fbow.h>

#define MIN_FRAMES 0
#define MAX_FRAMES 30
//#define MULTI_MAPPING_THREADS 1
#define MULTI_THREAD_FRAME_PROCESSING 1

#define LOG_WAISLAM_WARN(...) Utils::log("WAISlam", __VA_ARGS__);
#define LOG_WAISLAM_INFO(...) Utils::log("WAISlam", __VA_ARGS__);
#define LOG_WAISLAM_DEBUG(...) Utils::log("WAISlam", __VA_ARGS__);

WAISlam::WAISlam(const cv::Mat& intrinsic,
                 const cv::Mat& distortion,
                 fbow::Vocabulary* voc,
                 KPextractor*   iniExtractor,
                 KPextractor*   extractor,
                 WAIMap*        globalMap,
                 bool           trackingOnly,
                 bool           serial,
                 bool           retainImg,
                 float          cullRedundantPerc)
{
    _iniData.initializer = nullptr;
    _serial              = serial;
    _trackingOnly        = trackingOnly;
    _retainImg           = retainImg;

    WAIFrame::nNextId               = 0;
    WAIFrame::mbInitialComputations = true;

    _lastKeyFrameFrameId = 0;
    _lastRelocFrameId    = 0;

    _distortion      = distortion.clone();
    _cameraIntrinsic = intrinsic.clone();
    _voc             = voc;
    _extractor       = extractor;
    _iniExtractor    = iniExtractor;

    if (globalMap == nullptr)
    {
        WAIKeyFrameDB* kfDB = new WAIKeyFrameDB(*voc);
        _globalMap          = new WAIMap(kfDB);
        _state              = WAI::TrackingState_Initializing;

        WAIKeyFrame::nNextId = 0;
        WAIMapPoint::nNextId = 0;
    }
    else
    {
        _globalMap   = globalMap;
        _state       = WAI::TrackingState_TrackingLost;
        _initialized = true;
    }

    _localMapping = new ORB_SLAM2::LocalMapping(_globalMap, 1, _voc, cullRedundantPerc);
    _loopClosing  = new ORB_SLAM2::LoopClosing(_globalMap, _voc, false, false);

    _localMapping->SetLoopCloser(_loopClosing);
    _loopClosing->SetLocalMapper(_localMapping);

    if (!_serial)
    {
#if MULTI_MAPPING_THREADS
        _processNewKeyFrameThread = new std::thread(&LocalMapping::ProcessKeyFrames, _localMapping);
        _mappingThreads.push_back(_localMapping->AddLocalBAThread());
#else
        _mappingThreads.push_back(new std::thread(&LocalMapping::Run, _localMapping));
#endif
        _loopClosingThread = new std::thread(&LoopClosing::Run, _loopClosing);

#if MULTI_THREAD_FRAME_PROCESSING
        _poseUpdateThread = new std::thread(updatePoseThread, this);
        _isFinish         = false;
        _isStop           = false;
        _requestFinish    = false;
#endif
    }

    _iniData.initializer = nullptr;
    _cameraExtrinsic     = cv::Mat::eye(4, 4, CV_32F);

    _lastFrame = WAIFrame();
}

WAISlam::~WAISlam()
{
    if (!_serial)
    {
        _localMapping->RequestFinish();
        _loopClosing->RequestFinish();

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
    }

    delete _localMapping;
    delete _loopClosing;
}

void WAISlam::reset()
{
    std::cout << "WAISlam reset" << std::endl;
    if (!_serial)
    {
        std::cout << "Request Reset" << std::endl;
        _localMapping->RequestReset();
        _loopClosing->RequestReset();

#if MULTI_THREAD_FRAME_PROCESSING
        requestFinish();
        _poseUpdateThread->join();
#endif
    }
    else
    {
        _localMapping->reset();
        _loopClosing->reset();
    }

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
    _state                          = WAI::TrackingState_Initializing;
}

void WAISlam::createFrame(WAIFrame& frame, cv::Mat& imageGray)
{
    if (getTrackingState() == WAI::TrackingState_Initializing)
        frame = WAIFrame(imageGray, 0.0, _iniExtractor, _cameraIntrinsic, _distortion, _voc, _retainImg);
    else
        frame = WAIFrame(imageGray, 0.0, _extractor, _cameraIntrinsic, _distortion, _voc, _retainImg);
}

/* Separate Pose update thread */

void WAISlam::flushQueue()
{
    std::unique_lock<std::mutex> lock(_lastFrameMutex);
    while (!_framesQueue.empty())
    {
        _framesQueue.pop();
    }
}

void WAISlam::updateState(WAI::TrackingState state)
{
    std::unique_lock<std::mutex> lock(_mutexStates);
    _state = state;
}

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

void WAISlam::requestFinish()
{
    std::unique_lock<std::mutex> lock(_stateMutex);
    _requestFinish = true;
}

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

bool WAISlam::isStop()
{
    std::unique_lock<std::mutex> lock(_stateMutex);
    return _isStop;
}

void WAISlam::resume()
{
    std::unique_lock<std::mutex> lock(_stateMutex);
    _isStop = false;
    _localMapping->Release();
    _state = WAI::TrackingState_TrackingLost;
}

void WAISlam::updatePoseThread(WAISlam* ptr)
{
    while (1)
    {
        WAIFrame f;
        while (ptr->getNextFrame(f) && !ptr->finishRequested())
            ptr->updatePose(f);

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

void WAISlam::updatePose(WAIFrame& frame)
{
    std::unique_lock<std::mutex> guard(_mutexStates);

    switch (_state)
    {
        case WAI::TrackingState_Initializing: {
#if 0
            bool ok = oldInitialize(frame, _iniData, _globalMap, _localMap, _localMapping, _loopClosing, _voc, 100, _lastKeyFrameFrameId);
            if (ok)
            {
                _lastKeyFrameFrameId = frame.mnId;
                _lastRelocFrameId    = 0;
                _state               = TrackingState_TrackingOK;
                _initialized         = true;
            }
#else
            if (initialize(_iniData, frame, _voc, _localMap, 100, _lastKeyFrameFrameId))
            {
                if (genInitialMap(_globalMap, _localMapping, _loopClosing, _localMap, _serial))
                {
                    _lastKeyFrameFrameId = frame.mnId;
                    _lastRelocFrameId    = 0;
                    _state               = WAI::TrackingState_TrackingOK;
                    _initialized         = true;
                }
            }
#endif
        }
        break;
        case WAI::TrackingState_TrackingOK: {
            int inliers;
            if (tracking(_globalMap, _localMap, frame, _lastFrame, _lastRelocFrameId, _velocity, inliers))
            {
                std::unique_lock<std::mutex> lock(_cameraExtrinsicMutex);
                motionModel(frame, _lastFrame, _velocity, _cameraExtrinsic);
                lock.unlock();

                if (_serial)
                    serialMapping(_globalMap, _localMap, _localMapping, _loopClosing, frame, inliers, _lastRelocFrameId, _lastKeyFrameFrameId);
                else
                    mapping(_globalMap, _localMap, _localMapping, frame, inliers, _lastRelocFrameId, _lastKeyFrameFrameId);

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
            if (relocalization(frame, _globalMap, _localMap, inliers))
            {
                _lastRelocFrameId = frame.mnId;

                std::unique_lock<std::mutex> lock(_cameraExtrinsicMutex);
                motionModel(frame, _lastFrame, _velocity, _cameraExtrinsic);
                lock.unlock();

                if (_serial)
                    serialMapping(_globalMap, _localMap, _localMapping, _loopClosing, frame, inliers, _lastRelocFrameId, _lastKeyFrameFrameId);
                else
                    mapping(_globalMap, _localMap, _localMapping, frame, inliers, _lastRelocFrameId, _lastKeyFrameFrameId);

                _infoMatchedInliners = inliers;
                _state               = WAI::TrackingState_TrackingOK;
            }
        }
        break;
    }

    std::unique_lock<std::mutex> lock(_lastFrameMutex);
    _lastFrame = WAIFrame(frame);
}

WAIFrame WAISlam::getLastFrame()
{
    std::unique_lock<std::mutex> lock(_lastFrameMutex);
    return _lastFrame;
}

WAIFrame* WAISlam::getLastFramePtr()
{
    return &_lastFrame;
}

bool WAISlam::update(cv::Mat& imageGray)
{
    WAIFrame frame;
    createFrame(frame, imageGray);

#if MULTI_THREAD_FRAME_PROCESSING
    if (!_serial)
    {
        std::unique_lock<std::mutex> lock(_frameQueueMutex);
        _framesQueue.push(frame);
    }
    else
        updatePose(frame);
#else
    updatePose(frame);
#endif
    return isTracking();
}

void WAISlam::drawInfo(cv::Mat& imageRGB,
                       bool     showInitLine,
                       bool     showKeyPoints,
                       bool     showKeyPointsMatched)
{
    std::unique_lock<std::mutex> lock(_lastFrameMutex);
    std::unique_lock<std::mutex> lock2(_stateMutex);
    if (_state == WAI::TrackingState_Initializing)
    {
        if (showInitLine)
            drawInitInfo(_iniData, _lastFrame, imageRGB);
    }
    else if (_state == WAI::TrackingState_TrackingOK)
    {
        if (showKeyPoints)
            drawKeyPointInfo(_lastFrame, imageRGB);
        if (showKeyPointsMatched)
            drawKeyPointMatches(_lastFrame, imageRGB);
    }
    else if (_state == WAI::TrackingState_TrackingLost)
    {
        if (showKeyPoints)
            drawKeyPointInfo(_lastFrame, imageRGB);
    }
}


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

int WAISlam::getMatchedCorrespondances(WAIFrame* frame, std::pair<std::vector<cv::Point2f>, std::vector<cv::Point3f>> &matching)
{
    for (int i = 0; i < frame->N; i++)
    {
        WAIMapPoint* mp = frame->mvpMapPoints[i];
        if (mp)
        {
            if (!mp->isBad() && mp->Observations() > 0 && mp->isFixed())
            {
                WAI::V3   v = mp->worldPosVec();
                matching.first.push_back(frame->mvKeysUn[i].pt);
                matching.second.push_back(cv::Point3f(v.x, v.y, v.z));
            }
        }
    }
    return (int)matching.first.size();
}

cv::Mat WAISlam::getPose()
{
    std::unique_lock<std::mutex> lock(_cameraExtrinsicMutex);
    return _cameraExtrinsic;
}

void WAISlam::requestStateIdle()
{
    if (!_serial)
    {
        std::unique_lock<std::mutex> guard(_mutexStates);
        _localMapping->RequestStop();
        while (!_localMapping->isStopped())
        {
            std::cout << "localMapping is not yet stopped" << std::endl;
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        std::cout << "localMapping is stopped" << std::endl;
    }

    _state = WAI::TrackingState_Idle;
}

bool WAISlam::hasStateIdle()
{
    std::unique_lock<std::mutex> guard(_mutexStates);
    return (_state == WAI::TrackingState_Idle);
}

bool WAISlam::isTracking()
{
    std::unique_lock<std::mutex> guard(_mutexStates);
    return _state == WAI::TrackingState_TrackingOK;
}

bool WAISlam::retainImage()
{
    return false;
}

void WAISlam::setMap(WAIMap* globalMap)
{
    requestStateIdle();
    reset();
    _globalMap   = globalMap;
    _initialized = true;
    resume();
}

int WAISlam::getMapPointMatchesCount() const
{
    return _infoMatchedInliners;
}

std::string WAISlam::getLoopCloseStatus()
{
    return _loopClosing->getStatusString();
}

int WAISlam::getLoopCloseCount()
{
    return _globalMap->getNumLoopClosings();
}

int WAISlam::getKeyFramesInLoopCloseQueueCount()
{
    return _loopClosing->numOfKfsInQueue();
}
