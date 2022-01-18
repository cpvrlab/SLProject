//#############################################################################
//  File:      TrackingSystemArucoCube.cpp
//  Date:      December 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <TrackingSystemArucoCube.h>

#include <app/AppPenTracking.h>
#include <AppDemo.h>

//-----------------------------------------------------------------------------
bool TrackingSystemArucoCube::track(CVCaptureProvider* provider)
{
    CVTracked* tracker = AppPenTracking::instance().trackers().at(provider);
    if (!tracker || !AppPenTracking::instance().trackedNode) return false;

    /*
    if (typeid(*tracker) == typeid(CVTrackedArucoCube) && CVTrackedAruco::paramsLoaded)
    {
        optimizeTracking();
    }
    */

    CVCamera* ac        = &provider->camera();
    bool      foundPose = tracker->track(provider->lastFrameGray(),
                                    provider->lastFrameBGR(),
                                    &ac->calibration);
    if (!foundPose) return false;

    _multiTracker.recordCurrentPose(tracker, &ac->calibration);
    return true;
}
//-----------------------------------------------------------------------------
void TrackingSystemArucoCube::finalizeTracking()
{
    _multiTracker.combine();
}
//-----------------------------------------------------------------------------
CVMatx44f TrackingSystemArucoCube::worldMatrix()
{
    return _multiTracker.averageWorldMatrix();
}
//-----------------------------------------------------------------------------
void TrackingSystemArucoCube::calibrate(CVCaptureProvider* provider)
{
    AppPenTrackingCalibrator::calcExtrinsicParams(provider);
    AppDemo::scene->onLoad(AppDemo::scene, AppDemo::sceneViews[0], SID_VideoCalibrateMain);
}
//-----------------------------------------------------------------------------
bool TrackingSystemArucoCube::isAcceptedProvider(CVCaptureProvider* provider)
{
    return typeid(*provider) == typeid(CVCaptureProviderStandard) ||
           typeid(*provider) == typeid(CVCaptureProviderIDSPeak);
}
//-----------------------------------------------------------------------------
void TrackingSystemArucoCube::optimizeTracking()
{
    //    CVTrackedArucoCube* trackedArucoCube = (CVTrackedArucoCube*)tracker;
    //    CVRect              lastRoi          = CVRect(trackedArucoCube->_roi);
    //    trackedArucoCube->_roi               = CVRect(0, 0, 0, 0);
    //
    //    CVMat  imgGray;
    //    CVRect adapterRoi;
    //    if (lastRoi.empty())
    //    {
    //        imgGray    = CVCapture::instance()->lastFrameGray;
    //        adapterRoi = CVRect(0, 0, imgGray.cols, imgGray.rows);
    //    }
    //    else
    //    {
    //        adapterRoi        = CVRect(lastRoi.x - 100, lastRoi.y - 100, lastRoi.width + 200, lastRoi.height + 200);
    //        adapterRoi.x      = max(adapterRoi.x, 0);
    //        adapterRoi.y      = max(adapterRoi.y, 0);
    //        adapterRoi.width  = min(adapterRoi.x + adapterRoi.width, CVCapture::instance()->lastFrameGray.cols) - adapterRoi.x;
    //        adapterRoi.height = min(adapterRoi.y + adapterRoi.height, CVCapture::instance()->lastFrameGray.rows) - adapterRoi.y;
    //
    //        imgGray = CVCapture::instance()->lastFrameGray(adapterRoi);
    //
    //        cv::rectangle(CVCapture::instance()->lastFrame,
    //                      adapterRoi,
    //                      cv::Scalar(0, 255, 0),
    //                      2);
    //    }
    //
    //    CVVVPoint2f      corners, rejected;
    //    std::vector<int> arucoIDs;
    //    cv::aruco::detectMarkers(imgGray,
    //                             CVTrackedAruco::params.dictionary,
    //                             corners,
    //                             arucoIDs,
    //                             CVTrackedAruco::params.arucoParams,
    //                             rejected);
    //
    //    if (!corners.empty())
    //    {
    //        float minX = 10000, minY = 10000, maxX = -10000, maxY = -10000;
    //
    //        for (auto& i : corners)
    //        {
    //            for (const auto& corner : i)
    //            {
    //                if (corner.x < minX) minX = corner.x;
    //                if (corner.x > maxX) maxX = corner.x;
    //
    //                if (corner.y < minY) minY = corner.y;
    //                if (corner.y > maxY) maxY = corner.y;
    //            }
    //        }
    //
    //        minX += adapterRoi.x;
    //        maxX += adapterRoi.x;
    //        minY += adapterRoi.y;
    //        maxY += adapterRoi.y;
    //
    //        int roiX      = (int)minX - 100;
    //        int roiY      = (int)minY - 100;
    //        int roiWidth  = (int)(maxX - minX) + 200;
    //        int roiHeight = (int)(maxY - minY) + 200;
    //
    //        CVRect roi             = CVRect(roiX, roiY, roiWidth, roiHeight);
    //        roi.x                  = max(roi.x, 0);
    //        roi.y                  = max(roi.y, 0);
    //        roi.width              = min(roi.x + roi.width, CVCapture::instance()->lastFrame.cols) - roi.x;
    //        roi.height             = min(roi.y + roi.height, CVCapture::instance()->lastFrame.rows) - roi.y;
    //        trackedArucoCube->_roi = roi;
    //
    //        cv::rectangle(CVCapture::instance()->lastFrame,
    //                      roi,
    //                      cv::Scalar(255, 0, 0),
    //                      2);
    //    }
}
//-----------------------------------------------------------------------------