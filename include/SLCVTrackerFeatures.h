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
        SLfloat                 _fx, _fy, _cx, _cy;
        Mat                     _cam, _distortion;
        Mat                     _rMatrix, _tMatrix, _pMatrix;
        vector<Point3f>         _model;

        void load2dReferenceFeatures();
        inline SLCVVKeyPoint extractFeatures(const Mat &imageGray);
        inline Mat describeFeatures(const Mat &imageGray, SLCVVKeyPoint &keypoints);
        inline vector<DMatch> matchFeatures(const Mat &descriptors);
        inline vector<Point2f> trackFeatures(const SLCVVKeyPoint &keypoints, const vector<DMatch> &matches);
        inline void draw2DPoints(Mat image, const vector<Point2f> &list_points, Scalar color);
        inline void initCameraMat(SLCVCalibration *calib);
        inline void drawObject(const Mat &image);
        inline Point2f backproject3DPoint(const Point3f &point3d);
        inline void calcPMatrix();
};
//-----------------------------------------------------------------------------
#endif // SLCVTrackerFeatures_H
