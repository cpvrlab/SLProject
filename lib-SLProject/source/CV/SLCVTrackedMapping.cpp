//#############################################################################
//  File:      SLCVTrackedAruco.cpp
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers

/*
The OpenCV library version 3.1 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracked
for a good top down information.
*/
#include <SLApplication.h>
#include <SLSceneView.h>
#include <SLCVTrackedMapping.h>
#include <OrbSlam/Initializer.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;

//-----------------------------------------------------------------------------
SLCVTrackedMapping::SLCVTrackedMapping(SLNode* node, ORBVocabulary* vocabulary, 
    SLCVKeyFrameDB* keyFrameDB, SLCVMap* map)
    : SLCVTracked(node),
    mpVocabulary(vocabulary),
    mpKeyFrameDatabase(keyFrameDB),
    _map(map)
{
    //instantiate Orb extractor
    _extractor = new ORBextractor(1500, 1.44f, 4, 30, 20);
}
//-----------------------------------------------------------------------------
SLbool SLCVTrackedMapping::track(SLCVMat imageGray,
                               SLCVMat imageRgb,
                               SLCVCalibration* calib,
                               SLbool drawDetection,
                               SLSceneView* sv)
{
    //store reference to current color image for decoration
    _img = imageRgb;

    // Current Frame
    double timestamp = 0.0; //todo
    if (_currentState != IDLE) {
        mCurrentFrame = SLCVFrame(imageGray, timestamp, _extractor,
            calib->cameraMat(), calib->distortion(), mpVocabulary);

        if ( true /*_showKeyPoints*/)
        {
            for (size_t i = 0; i < mCurrentFrame.N; i++)
            {
                //Use distorted points because we have to undistort the image later
                const auto& pt = mCurrentFrame.mvKeys[i].pt;
                cv::rectangle(_img,
                    cv::Rect(pt.x - 3, pt.y - 3, 7, 7),
                    cv::Scalar(0, 0, 255));
            }
        }
    }

    switch (_currentState)
    {
    case IDLE:
        break;
    case INITIALIZE:
        initialize();
        break;
    case TRACK_VO:
        trackVO();
        break;
    case TRACK_3DPTS:
        track3DPts();
        break;
    case TRACK_OPTICAL_FLOW:
        trackOpticalFlow();
        break;
    }

    return false;
}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::initialize()
{
    if (!mpInitializer)
    {
        // Set Reference Frame
        if (mCurrentFrame.mvKeys.size()>100)
        {
            mInitialFrame = SLCVFrame(mCurrentFrame);
            mLastFrame = SLCVFrame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for (size_t i = 0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

            if (mpInitializer)
                delete mpInitializer;

            mpInitializer = new ORB_SLAM2::Initializer(mCurrentFrame, 1.0, 200);

            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

            return;
        }
    }
    else
    {
        // Try to initialize
        if ((int)mCurrentFrame.mvKeys.size() <= 100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9, true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

        // Check if there are enough correspondences
        if (nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        //ghm1: decorate image with tracked matches
        for (unsigned int i = 0; i<mvIniMatches.size(); i++)
        {
            if (mvIniMatches[i] >= 0)
            {
                cv::line(_img, mInitialFrame.mvKeys[i].pt, mCurrentFrame.mvKeys[mvIniMatches[i]].pt,
                    cv::Scalar(0, 255, 0));
            }
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for (size_t i = 0, iend = mvIniMatches.size(); i<iend; i++)
            {
                if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                {
                    mvIniMatches[i] = -1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
            Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
            tcw.copyTo(Tcw.rowRange(0, 3).col(3));
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::trackVO()
{

}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::track3DPts()
{

}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::trackOpticalFlow()
{

}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::CreateInitialMapMonocular()
{
    // Create KeyFrames
    SLCVKeyFrame* pKFini = new SLCVKeyFrame(mInitialFrame, _map, mpKeyFrameDatabase);
    SLCVKeyFrame* pKFcur = new SLCVKeyFrame(mCurrentFrame, _map, mpKeyFrameDatabase);

    pKFini->ComputeBoW( mpVocabulary );
    pKFcur->ComputeBoW( mpVocabulary );

    //// Insert KFs in the map
    //_map->AddKeyFrame(pKFini);
    //_map->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for (size_t i = 0; i<mvIniMatches.size(); i++)
    {
        if (mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        SLCVMapPoint* pMP = new SLCVMapPoint(worldPos, pKFcur/*, _map*/);

        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        _map->mapPoints().push_back(pMP);
    }

    //// Update Connections
    //pKFini->UpdateConnections();
    //pKFcur->UpdateConnections();

    //// Bundle Adjustment
    //cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    //Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

    //// Set median depth to 1
    //float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    //float invMedianDepth = 1.0f / medianDepth;

    //if (medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    //{
    //    cout << "Wrong initialization, reseting..." << endl;
    //    Reset();
    //    return;
    //}

    //// Scale initial baseline
    //cv::Mat Tc2w = pKFcur->GetPose();
    //Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3)*invMedianDepth;
    //pKFcur->SetPose(Tc2w);

    //// Scale points
    //vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    //for (size_t iMP = 0; iMP<vpAllMapPoints.size(); iMP++)
    //{
    //    if (vpAllMapPoints[iMP])
    //    {
    //        MapPoint* pMP = vpAllMapPoints[iMP];
    //        pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
    //    }
    //}

    //mpLocalMapper->InsertKeyFrame(pKFini);
    //mpLocalMapper->InsertKeyFrame(pKFcur);

    //mCurrentFrame.SetPose(pKFcur->GetPose());
    //mnLastKeyFrameId = mCurrentFrame.mnId;
    //mpLastKeyFrame = pKFcur;

    //mvpLocalKeyFrames.push_back(pKFcur);
    //mvpLocalKeyFrames.push_back(pKFini);
    //mvpLocalMapPoints = mpMap->GetAllMapPoints();
    //mpReferenceKF = pKFcur;
    //mCurrentFrame.mpReferenceKF = pKFcur;

    //mLastFrame = Frame(mCurrentFrame);

    //mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    //mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    //mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    //mState = OK;
}