#include "WAISlamTrackPool.h"

//-----------------------------------------------------------------------------
WAISlamTrackPool::WAISlamTrackPool(const cv::Mat&           intrinsic,
                                   const cv::Mat&           distortion,
                                   WAIOrbVocabulary*        voc,
                                   KPextractor*             iniExtractor,
                                   KPextractor*             relocExtractor,
                                   KPextractor*             extractor,
                                   std::unique_ptr<WAIMap>  globalMap,
                                   WAISlamTrackPool::Params params)
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
                                              false,
                                              false);

    _localMapping->SetLoopCloser(_loopClosing);
    _loopClosing->SetLocalMapper(_localMapping);

    if (!_params.onlyTracking && !_params.serial)
    {
        _mappingThreads.push_back(new std::thread(&LocalMapping::Run, _localMapping));
        _loopClosingThread = new std::thread(&LoopClosing::Run, _loopClosing);
    }

    _iniData.initializer = nullptr;
    _cameraExtrinsic     = cv::Mat::eye(4, 4, CV_32F);

    _lastFrame = WAIFrame();
}
//-----------------------------------------------------------------------------
WAISlamTrackPool::~WAISlamTrackPool()
{
    if (!_params.serial)
    {
        _localMapping->RequestFinish();
        _loopClosing->RequestFinish();
    }

    // Wait until all thread have effectively stopped
    if (_processNewKeyFrameThread)
        _processNewKeyFrameThread->join();

    for (std::thread* t : _mappingThreads)
    {
        t->join();
    }

    if (_loopClosingThread)
        _loopClosingThread->join();

    delete _localMapping;
    delete _loopClosing;
}
//-----------------------------------------------------------------------------
bool WAISlamTrackPool::update(cv::Mat& imageGray)
{
    WAIFrame frame;
    createFrame(frame, imageGray);

    // if (_params.ensureKFIntegration)
    //     updatePoseKFIntegration(frame);
    // else
    updatePose(frame);

    return (_state == WAITrackingState::TrackingOK);
}
//-----------------------------------------------------------------------------
void WAISlamTrackPool::createFrame(WAIFrame& frame, cv::Mat& imageGray)
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
void WAISlamTrackPool::updatePose(WAIFrame& frame)
{
    switch (_state)
    {
        case WAITrackingState::Initializing:
        {
            if (initialize(_iniData, frame, _voc, _localMap))
            {
                if (genInitialMap(_globalMap.get(),
                                  _localMapping,
                                  _loopClosing,
                                  _localMap))
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
        }
        break;

        case WAITrackingState::TrackingStart:
        {
            _relocFrameCounter++;
            if (_relocFrameCounter > 0)
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
                motionModel(frame, _lastFrame, _velocity, _cameraExtrinsic);
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
            if (relocalization(frame, _globalMap.get(), _localMap, _params.minCommonWordFactor, inliers, _params.minAccScoreFilter))
            {
                _relocFrameCounter = 0;
                _lastRelocFrameId  = frame.mnId;
                _velocity          = cv::Mat();
                _state             = WAITrackingState::TrackingStart;
            }
        }
        break;
    }

    _lastFrame = WAIFrame(frame);
}
//-----------------------------------------------------------------------------
void WAISlamTrackPool::changeIntrinsic(cv::Mat intrinsic, cv::Mat distortion)
{
    _cameraIntrinsic = intrinsic;
    _distortion      = distortion;
}
//-----------------------------------------------------------------------------
cv::Mat WAISlamTrackPool::getPose()
{
    std::unique_lock<std::mutex> lock(_cameraExtrinsicMutex);
    return _cameraExtrinsic;
}
//-----------------------------------------------------------------------------
void WAISlamTrackPool::drawInfo(cv::Mat& imageBGR,
                                float    scale,
                                bool     showInitLine,
                                bool     showKeyPoints,
                                bool     showKeyPointsMatched)
{
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
std::vector<WAIMapPoint*> WAISlamTrackPool::getMatchedMapPoints()
{
    std::vector<WAIMapPoint*> result;

    for (int i = 0; i < _lastFrame.N; i++)
    {
        if (_lastFrame.mvpMapPoints[i])
        {
            if (_lastFrame.mvpMapPoints[i]->Observations() > 0)
                result.push_back(_lastFrame.mvpMapPoints[i]);
        }
    }

    return result;
}
//-----------------------------------------------------------------------------
