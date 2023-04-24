//#############################################################################
//  File:      CVFeatureManager.h
//  Date:      Autumn 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CVFEATUREMANAGER_H
#define CVFEATUREMANAGER_H

#include <CVTypedefs.h>

//-----------------------------------------------------------------------------
//! OpenCV feature type
enum CVFeatureType
{
    FT_SIFT, //!<
    FT_SURF, //!<
    FT_ORB   //!<
};
//-----------------------------------------------------------------------------
//! Feature detector-decriptor types
enum CVDetectDescribeType
{
#ifndef __EMSCRIPTEN__
    DDT_FAST_BRIEF,
#endif
    DDT_RAUL_RAUL,
    DDT_ORB_ORB,
#ifndef __EMSCRIPTEN__
    DDT_SURF_SURF,
#endif
    DDT_SIFT_SIFT
};
//-----------------------------------------------------------------------------
//! Wrapper class around OpenCV feature detector & describer
class CVFeatureManager
{
public:
    CVFeatureManager();
    ~CVFeatureManager();

    void detect(CVInputArray image,
                CVVKeyPoint& keypoints,
                CVInputArray mask = cv::noArray());

    void describe(CVInputArray  image,
                  CVVKeyPoint&  keypoints,
                  CVOutputArray descriptors);

    void detectAndDescribe(CVInputArray  image,
                           CVVKeyPoint&  keypoints,
                           CVOutputArray descriptors,
                           CVInputArray  mask = cv::noArray());

    void createDetectorDescriptor(CVDetectDescribeType detectDescribeType);

    void setDetectorDescriptor(CVDetectDescribeType detectDescribeType,
                               cv::Ptr<CVFeature2D> detector,
                               cv::Ptr<CVFeature2D> descriptor);
    // Getter
    CVDetectDescribeType type() { return _type; }

private:
    CVDetectDescribeType _type;       //!< Type of detector-descriptor pair
    cv::Ptr<CVFeature2D> _detector;   //!< CV smart pointer to the OpenCV feature detector
    cv::Ptr<CVFeature2D> _descriptor; //!< CV smart pointer to the OpenCV descriptor extractor
};
//-----------------------------------------------------------------------------
#endif // CVFEATUREMANAGER_H
