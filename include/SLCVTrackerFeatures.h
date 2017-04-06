//#############################################################################
//  File:      SLCVTrackerFeatures.h
//  Author:    Pascal Zingg, Timon Tschanz
//  Date:      Spring 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This softwareis provide under the GNU General Public License
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
#include <SLCVRaulMurOrb.h>
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
        SLCVRaulMurOrb*         _detector;
        Ptr<ORB>                _descriptor;
        Ptr<DescriptorMatcher>  _matcher;
        SLfloat                 _fx, _fy, _cx, _cy;
        SLMat4f                 _pose;
        SLCVCalibration         *_calib;
        int                     frameCount;

        struct prev {
            SLCVMat             image;
            SLCVMat             imageGray;
            vector<Point2f>     points2D;
            vector<DMatch>      matches;
            Mat                 rvec;
            Mat                 tvec;
            bool                foundPose;
        } _prev;

        struct map {
            vector<Point3f>     model;
            SLCVMat             frameGray;
            SLCVVKeyPoint       keypoints;
            Mat                 descriptors;
        } _map;

        void loadModelPoints();
        SLCVVKeyPoint getKeypoints(const Mat &imageGray);
        Mat getDescriptors(const Mat &imageGray, SLCVVKeyPoint &keypoints);
        vector<DMatch> getFeatureMatches(const Mat &descriptors);
        bool calculatePose(const Mat &imageGray, const SLCVVKeyPoint &keypoints, vector<DMatch> &matches, vector<DMatch> &inliers, vector<Point2f> &inlierPoints, Mat &rvec, Mat &tvec, bool extrinsicGuess=false);
        bool trackWithOptFlow(SLCVMat &previousFrame, vector<Point2f> &previousPoints, SLCVMat &actualFrame, vector<Point2f> &predPoints, Mat &rvec, Mat &tvec);
        bool solvePnP(vector<Point3f> &modelPoints, vector<Point2f> &framePoints, bool guessExtrinsic, Mat &rvec, Mat &tvec, vector<unsigned char> &inliersMask);
};
//-----------------------------------------------------------------------------
#endif // SLCVTrackerFeatures_H
