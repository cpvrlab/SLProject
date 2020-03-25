//#############################################################################
//  File:      CVFeatureManager.cpp
//  Purpose:   OpenCV Detector Describer Wrapper
//  Author:    Marcus Hudritsch
//  Date:      Spring 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant APP_USES_CVCAPTURE.
All classes that use OpenCV begin with CV.
See also the class docs for CVCapture, CVCalibration and CVTracked
for a good top down information.
*/

#include <CVFeatureManager.h>
#include <CVRaulMurOrb.h>
#include <Utils.h>
#include <algorithm> // std::max

using namespace cv;
using namespace std;

//-----------------------------------------------------------------------------
CVFeatureManager::CVFeatureManager()
{
    createDetectorDescriptor(DDT_RAUL_RAUL);
}
//-----------------------------------------------------------------------------
CVFeatureManager::~CVFeatureManager()
{
}
//-----------------------------------------------------------------------------
//! Creates a detector and decriptor to the passed type
void CVFeatureManager::createDetectorDescriptor(CVDetectDescribeType type)
{
    switch (type)
    {
        case DDT_FAST_BRIEF:
            _detector   = FastFeatureDetector::create(30, true, FastFeatureDetector::TYPE_9_16);
            _descriptor = xfeatures2d::BriefDescriptorExtractor::create(32, true);
            break;
        case DDT_ORB_ORB:
            _detector   = ORB::create(200, 1.44f, 3, 31, 0, 2, ORB::HARRIS_SCORE, 31, 30);
            _descriptor = _detector;
            break;
        case DDT_RAUL_RAUL:
            _detector   = new CVRaulMurOrb(1500, 1.44f, 4, 30, 20);
            _descriptor = _detector;
            break;
        case DDT_SURF_SURF:
            _detector   = xfeatures2d::SURF::create(100, 2, 2, false, false);
            _descriptor = _detector;
            break;
        case DDT_SIFT_SIFT:
            _detector = xfeatures2d::SIFT::create(300, 2, 0.04, 10, 1.6);
            ;
            _descriptor = _detector;
            break;
        default:
            Utils::exitMsg("SLProject", "Unknown detector-descriptor type.", __LINE__, __FILE__);
    }

    _type = type;
}
//-----------------------------------------------------------------------------
//! Sets the detector and decriptor to the passed ones
void CVFeatureManager::setDetectorDescriptor(CVDetectDescribeType type,
                                             cv::Ptr<CVFeature2D> detector,
                                             cv::Ptr<CVFeature2D> descriptor)
{
    _type       = type;
    _detector   = detector;
    _descriptor = descriptor;
}
//-----------------------------------------------------------------------------
void CVFeatureManager::detectAndDescribe(CVInputArray  image,
                                         CVVKeyPoint&  keypoints,
                                         CVOutputArray descriptors,
                                         CVInputArray  mask)
{
    assert(_detector && "CVFeatureManager::detectAndDescribe: No detector!");
    assert(_descriptor && "CVFeatureManager::detectAndDescribe: No descriptor!");

    if (_detector == _descriptor)
        _detector->detectAndCompute(image, mask, keypoints, descriptors);
    else
    {
        _detector->detect(image, keypoints, mask);
        _descriptor->compute(image, keypoints, descriptors);
    }
}
//-----------------------------------------------------------------------------
