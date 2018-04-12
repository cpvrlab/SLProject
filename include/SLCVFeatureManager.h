//#############################################################################
//  File:      SLCVFeatureManager.h
//  Author:    Marcus Hudritsch
//  Date:      Autumn 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVFEATUREMANAGER_H
#define SLCVFEATUREMANAGER_H

#include <SLEnums.h>
#include <SLCV.h>

//-----------------------------------------------------------------------------
//! Wrapper class around OpenCV feature detector & describer
class SLCVFeatureManager
{
    public:
                    SLCVFeatureManager      ();
                   ~SLCVFeatureManager      ();

        void        detect                  (SLCVInputArray image,
                                             SLCVVKeyPoint &keypoints,
                                             SLCVInputArray mask = cv::noArray());

        void        describe                (SLCVInputArray  image,
                                             SLCVVKeyPoint&  keypoints,
                                             SLCVOutputArray descriptors);

        void        detectAndDescribe       (SLCVInputArray  image,
                                             SLCVVKeyPoint&  keypoints,
                                             SLCVOutputArray descriptors,
                                             SLCVInputArray  mask=cv::noArray());

        void        createDetectorDescriptor(SLCVDetectDescribeType detectDescribeType);

        void        setDetectorDescriptor   (SLCVDetectDescribeType detectDescribeType,
                                             cv::Ptr<SLCVFeature2D> detector,
                                             cv::Ptr<SLCVFeature2D> descriptor);
        // Getter
        SLCVDetectDescribeType  type         () {return _type;}

    private:
        SLCVDetectDescribeType  _type;          //!< Type of detector-descriptor pair
        cv::Ptr<SLCVFeature2D>  _detector;      //!< CV smart pointer to the OpenCV feature detector
        cv::Ptr<SLCVFeature2D>  _descriptor;    //!< CV smart pointer to the OpenCV descriptor extractor
};
//-----------------------------------------------------------------------------
#endif // SLCVDETECTOR_H
