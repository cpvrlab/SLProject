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
        ~SLCVTrackerFeatures        ();
        SLbool  track               (SLCVMat imageGray,
                                     SLCVMat image,
                                     SLCVCalibration* calib,
                                     SLSceneView* sv);

private:
        static SLVMat4f         objectViewMats; //!< object view SLCVMatrices
        SLCVRaulMurOrb*         _detector;
        Ptr<ORB>                _descriptor;
        Ptr<DescriptorMatcher>  _matcher;
        vector<Point2f>         _inlierPoints2D;
        vector<Point3f>         _inlierPoints3D;
        SLfloat                 _fx, _fy, _cx, _cy;
        SLMat4f                 _pose;
        SLCVCalibration         *_calib;
        int                     frameCount, reposePatchSize;

        struct prev {
            SLCVMat             image;
            SLCVMat             imageGray;
            vector<Point2f>     points2D;
            vector<Point3f>     points3D;
            vector<DMatch>      SLCVMatches;
            SLCVMat             rvec;
            SLCVMat             tvec;
            bool                foundPose;
            float               reprojectionError;
        } _prev;

        struct map {
            vector<Point3f>     model;
            SLCVMat             frameGray;
            SLCVMat             imgDrawing;
            SLCVVKeyPoint       keypoints;
            SLCVMat             descriptors;
            SLCVVKeyPoint       bboxModelKeypoints;
        } _map;

        void loadModelPoints();

        SLCVVKeyPoint getKeypoints(const SLCVMat &imageGray);

        SLCVMat getDescriptors(const SLCVMat &imageGray, SLCVVKeyPoint &keypoints);

        void getKeypointsAndDescriptors(const SLCVMat &imageGray, SLCVVKeyPoint &keypoints, SLCVMat &descriptors);

        vector<DMatch> getFeatureMatches(const SLCVMat &descriptors);

        bool calculatePose(const SLCVMat &imageVideo, vector<KeyPoint> &keypoints, vector<DMatch> &matches,
            vector<DMatch> &inliers, Mat &rvec, SLCVMat &tvec, bool extrinsicGuess, const SLCVMat& descriptors);

        bool solvePnP(vector<Point3f> &modelPoints, vector<Point2f> &framePoints, bool guessExtrinsic,
            SLCVMat &rvec, SLCVMat &tvec, vector<unsigned char> &inliersMask);

        bool optimizePose(const SLCVMat &imageVideo, vector<KeyPoint> &keypoints, const SLCVMat &descriptors,
            vector<DMatch> &matches, SLCVMat &rvec, SLCVMat &tvec, float reprojectionError=0);

        bool trackWithOptFlow(Mat &previousFrame, vector<Point2f> &previousPoints, Mat &actualFrame,
            Mat &rvec, Mat &tvec, SLCVMat &frame);
};
//-----------------------------------------------------------------------------
#endif // SLCVTrackerFeatures_H
