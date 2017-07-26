//#############################################################################
//  File:      SLCVDetector.h
//  Author:    Marcus Hudritsch
//  Date:      Autumn 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVDETECTOR_H
#define SLCVDETECTOR_H

#include <SLCV.h>

//-----------------------------------------------------------------------------
class SLCVDetector
{
    private:
        cv::Ptr<cv::FeatureDetector> _detector;

    public:
                            SLCVDetector(SLCVDetectorType type,
                                         SLbool force=false);

        SLbool              forced;
        SLCVDetectorType    type;

        void                detect      (SLCVInputArray image,
                                         SLCVVKeyPoint &keypoints,
                                         SLCVInputArray mask = cv::noArray());

        void setDetector(cv::Ptr<cv::FeatureDetector> detector) { _detector = detector; }
};
//-----------------------------------------------------------------------------
#endif // SLCVDETECTOR_H
