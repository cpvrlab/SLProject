//#############################################################################
//  File:      SLCVTrackedMapping.cpp
//  Author:    Michael Goettlicher
//  Date:      March 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLCVMapTracking.h>
#include <SLCVKeyFrameDB.h>
#include <SLCVMap.h>
#include <SLCVMapNode.h>

//-----------------------------------------------------------------------------
SLCVMapTracking::SLCVMapTracking(SLCVKeyFrameDB* keyFrameDB, SLCVMap* map, 
    SLCVMapNode* mapNode, bool serial)
    : mpKeyFrameDatabase(keyFrameDB),
    _map(map),
    _mapNode(mapNode),
    _serial(serial),
    sm(this, serial)
{
}
//-----------------------------------------------------------------------------
SLCVMapTracking::SLCVMapTracking(SLCVMapNode* mapNode, bool serial)
    : _mapNode(mapNode),
    _serial(serial),
    sm(this, serial)
{
}
//-----------------------------------------------------------------------------
void SLCVMapTracking::track()
{
    //apply state transitions
    sm.stateTransition();

    switch (sm.state())
    {
    case SLCVTrackingStateMachine::INITIALIZING:
        initialize();
        break;
    case SLCVTrackingStateMachine::IDLE:
        idle();
        break;
    case SLCVTrackingStateMachine::TRACKING_OK:
        //todo: divide relocalization and tracking
    case SLCVTrackingStateMachine::TRACKING_LOST:
        //relocalize or track 3d points
        track3DPts();
        break;
    }
}
//-----------------------------------------------------------------------------
void SLCVMapTracking::idle()
{
    if(!serial())
        std::this_thread::sleep_for(std::chrono::seconds(1));
}
//-----------------------------------------------------------------------------
int SLCVMapTracking::getNMapMatches()
{
  std::lock_guard<std::mutex> guard(_nMapMatchesLock);
  return mnMatchesInliers;
}
//-----------------------------------------------------------------------------
int SLCVMapTracking::getNumKeyFrames()
{
  std::lock_guard<std::mutex> guard(_mapLock); 
  return _map->KeyFramesInMap();
}
//-----------------------------------------------------------------------------
float SLCVMapTracking::poseDifference()
{
  std::lock_guard<std::mutex> guard(_poseDiffLock);
  return _poseDifference;
}
//-----------------------------------------------------------------------------
float SLCVMapTracking::meanReprojectionError()
{
  std::lock_guard<std::mutex> guard(_meanProjErrorLock);
  return _meanReprojectionError;
}
//-----------------------------------------------------------------------------
int SLCVMapTracking::mapPointsCount()
{
  std::lock_guard<std::mutex> guard(_mapLock);
  if (_map)
    return _map->MapPointsInMap();
  else
    return 0;
}
//-----------------------------------------------------------------------------
string SLCVMapTracking::getPrintableState()
{
    return sm.getPrintableState();
}
//-----------------------------------------------------------------------------
string SLCVMapTracking::getPrintableType()
{
    switch(trackingType)
    {
    case TrackingType_MotionModel:
        return "Motion Model";
    case TrackingType_OptFlow:
        return "Optical Flow";
    case TrackingType_ORBSLAM:
        return "ORB-SLAM";
    case TrackingType_None:
    default:
        return "None";
    }
}
//-----------------------------------------------------------------------------
void SLCVMapTracking::calculateMeanReprojectionError()
{
    //calculation of mean reprojection error
    double reprojectionError = 0.0;
    int n = 0;

    //current frame extrinsic
    const cv::Mat Rcw = mCurrentFrame.GetRotationCW();
    const cv::Mat tcw = mCurrentFrame.GetTranslationCW();
    //current frame intrinsics
    const float fx = mCurrentFrame.fx;
    const float fy = mCurrentFrame.fy;
    const float cx = mCurrentFrame.cx;
    const float cy = mCurrentFrame.cy;

    for (size_t i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i]) 
       {
            if (!mCurrentFrame.mvbOutlier[i])
            {
                if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                {
                    // 3D in absolute coordinates
                    cv::Mat Pw = mCurrentFrame.mvpMapPoints[i]->GetWorldPos();
                    // 3D in camera coordinates
                    const cv::Mat Pc = Rcw*Pw + tcw;
                    const float &PcX = Pc.at<float>(0);
                    const float &PcY = Pc.at<float>(1);
                    const float &PcZ = Pc.at<float>(2);

                    // Check positive depth
                    if (PcZ<0.0f)
                        continue;

                    // Project in image and check it is not outside
                    const float invz = 1.0f / PcZ;
                    const float u = fx*PcX*invz + cx;
                    const float v = fy*PcY*invz + cy;

                    SLCVPoint2f ptProj(u, v);
                    //Use distorted points because we have to undistort the image later
                    const auto& ptImg = mCurrentFrame.mvKeysUn[i].pt;

                    ////draw projected point
                    //cv::rectangle(image,
                    //    cv::Rect(ptProj.x - 3, ptProj.y - 3, 7, 7),
                    //    Scalar(255, 0, 0));

                    reprojectionError += norm(SLCVMat(ptImg), SLCVMat(ptProj));
                    n++;
                }
            }
        }
    }

    {
      std::lock_guard<std::mutex> guard(_meanProjErrorLock);
      if (n > 0)
      {
	_meanReprojectionError = reprojectionError / n;
      }
      else
      {
	_meanReprojectionError = -1;
      }
    }
}
//-----------------------------------------------------------------------------
void SLCVMapTracking::calculatePoseDifference()
{
  std::lock_guard<std::mutex> guard(_poseDiffLock);
  //calculation of L2 norm of the difference between the last and the current camera pose
  if (!mLastFrame.mTcw.empty() && !mCurrentFrame.mTcw.empty())
    _poseDifference = norm(mLastFrame.mTcw - mCurrentFrame.mTcw);
  else
    _poseDifference = -1.0;
}
//-----------------------------------------------------------------------------
void SLCVMapTracking::decorateVideoWithKeyPoints(cv::Mat& image)
{
    //show rectangle for all keypoints in current image
    if (_showKeyPoints)
    {
        for (size_t i = 0; i < mCurrentFrame.N; i++)
        {
            //Use distorted points because we have to undistort the image later
            //const auto& pt = mCurrentFrame.mvKeys[i].pt;
            const auto& pt = mCurrentFrame.mvKeys[i].pt;
            cv::rectangle(image,
                cv::Rect(pt.x - 3, pt.y - 3, 7, 7),
                cv::Scalar(0, 0, 255));
        }
    }
}
//-----------------------------------------------------------------------------
void SLCVMapTracking::decorateVideoWithKeyPointMatches(cv::Mat& image)
{
    //show rectangle for key points in video that where matched to map points
    if (_showKeyPointsMatched)
    {
        for (size_t i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                if (!mCurrentFrame.mvbOutlier[i])
                {
                    if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    {
                        //Use distorted points because we have to undistort the image later
                        const auto& pt = mCurrentFrame.mvKeys[i].pt;
                        cv::rectangle(image,
                            cv::Rect(pt.x - 3, pt.y - 3, 7, 7),
                            cv::Scalar(0, 255, 0));
                    }
                }
            }
        }
    }
}
//-----------------------------------------------------------------------------
void SLCVMapTracking::decorateScene()
{
    //update scene with all mappoints and keyframes (these only change after mapping of bundle adjustment)
    if (_mapHasChanged)
    {
        std::lock_guard<std::mutex> guard(_mapLock);
        _mapHasChanged = false;
        //update scene
        auto mapPts = _map->GetAllMapPoints();
        _mapNode->updateMapPoints(mapPts);
        auto mapKfs = _map->GetAllKeyFrames();
        _mapNode->updateKeyFrames(mapKfs);
    }

    //update hide flags of map points and keyframes
    _mapNode->setHideMapPoints(!_showMapPC);
    _mapNode->setHideKeyFrames(!_showKeyFrames);
    //set flag, that images should be rendered as keyframe backgrounds
    _mapNode->renderKfBackground(_renderKfBackground);
    //allow keyframes as active cameras to watch through them
    _mapNode->allowAsActiveCam(_allowKfsAsActiveCam);

    //-------------------------------------------------------------------------
    //decorate scene with mappoints that were matched to keypoints in current frame
    if (sm.state() == SLCVTrackingStateMachine::TRACKING_OK && _showMatchesPC)
    {
        //find map point matches
        std::vector<SLCVMapPoint*> mapPointMatches;
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                if (!mCurrentFrame.mvbOutlier[i])
                {
                    if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                        mapPointMatches.push_back(mCurrentFrame.mvpMapPoints[i]);
                }
            }
        }

        //update scene
        _mapNode->updateMapPointsMatched(mapPointMatches);
    }
    else
    {
        //remove point cloud
        _mapNode->removeMapPointsMatched();
    }

    //-------------------------------------------------------------------------
    //decorate scene with mappoints of local map
    if (sm.state() == SLCVTrackingStateMachine::TRACKING_OK && _showLocalMapPC)
        _mapNode->updateMapPointsLocal(mvpLocalMapPoints);
    else
        _mapNode->removeMapPointsLocal();
}
//-----------------------------------------------------------------------------
void SLCVMapTracking::decorateSceneAndVideo(cv::Mat& image)
{
    //calculation of mean reprojection error of all matches
    calculateMeanReprojectionError();
    //calculate pose difference
    calculatePoseDifference();
    //show rectangle for all keypoints in current image
    decorateVideoWithKeyPoints(image);
    //show rectangle for key points in video that where matched to map points
    decorateVideoWithKeyPointMatches(image);
    //decorate scene with matched map points, local map points and matched map points
    decorateScene();
}
