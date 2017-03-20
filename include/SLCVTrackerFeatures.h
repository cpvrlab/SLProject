//#############################################################################
//  File:      SLCVTrackerAruco.cpp
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVTrackerFeatures_H
#define SLCVTrackerFeatures_H

/*
The OpenCV library version 3.1 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracker
for a good top down information.
*/

#include <SLCV.h>
#include <SLCVTracker.h>
#include <SLNode.h>
using namespace cv;


class SLCVTrackerFeatures : public SLCVTracker
{
    public:
        SLCVTrackerFeatures         (SLNode* node);
        ~SLCVTrackerFeatures        () {;}
        SLbool  track               (SLCVMat imageGray,
                                     SLCVMat image,
                                     SLCVCalibration* calib,
                                     SLSceneView* sv);

    private:
        static SLVMat4f         objectViewMats; //!< object view matrices
        Ptr<ORB>                _detector;
        Ptr<DescriptorMatcher>  _matcher;
        Mat                     _lastFrameDescriptors;
        SLCVMat                 _lastFrameGray;
        SLCVVKeyPoint           _lastFrameKeypoints;
        SLCVVPoint3f            _points3d_model;
        double                  fx, fy, cx, cy;

        void load2dReferenceFeatures();
        void draw2DPoints(cv::Mat image, std::vector<cv::Point2f> &list_points, cv::Scalar color);
        void drawPose(cv::Mat rotVec, cv::Mat transVec, cv::Mat K, cv::Mat dist, cv::Mat image);
        inline SLCVVKeyPoint extractFeatures(Mat &imageGray);
        inline Mat describeFeatures(Mat &imageGray, SLCVVKeyPoint &keypoints);
        inline vector<DMatch> matchFeatures(Mat &descriptors);
        inline vector<Point2f> trackFeatures(SLCVVKeyPoint& keypoints, vector<DMatch> &matches);
};
//-----------------------------------------------------------------------------
#endif // SLCVTrackerFeatures_H
