//#############################################################################
//  File:      SLCVTrackedMapping.cpp
//  Author:    Michael Goettlicher
//  Date:      March 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVTrackedMapping_H
#define SLCVTrackedMapping_H

/*
The OpenCV library version 3.1 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracked
for a good top down information.
*/

#include <SLCV.h>
#include <SLCVTracked.h>
#include <SLNode.h>

namespace ORB_SLAM2 {
    class Initializer;
}

//-----------------------------------------------------------------------------

class SLCVTrackedMapping : public SLCVTracked
{
    public:
        enum TrackingStates { IDLE, INITIALIZE, TRACK_VO, TRACK_3DPTS, TRACK_OPTICAL_FLOW };

                SLCVTrackedMapping    (SLNode* node, ORBVocabulary* vocabulary);
               ~SLCVTrackedMapping    () {}

        SLbool  track               (SLCVMat imageGray,
                                     SLCVMat imageRgb,
                                     SLCVCalibration* calib,
                                     SLbool drawDetection,
                                     SLSceneView* sv);

        void setState(TrackingStates state) { _currentState = state; }
    private:
        // Map initialization for monocular
        void CreateInitialMapMonocular();

        //! initialization routine
        void initialize();
        void trackVO();
        void track3DPts();
        void trackOpticalFlow();

        //! states, that we try to make a new key frame out of the next frame
        bool _addKeyframe;
        //! current tracking state
        TrackingStates _currentState = IDLE;

        // ORB vocabulary used for place recognition and feature matching.
        ORBVocabulary* mpVocabulary;
        // Current Frame
        SLCVFrame mCurrentFrame;

        // Initialization Variables (Monocular)
        //std::vector<int> mvIniLastMatches;
        std::vector<int> mvIniMatches;
        std::vector<cv::Point2f> mvbPrevMatched;
        std::vector<cv::Point3f> mvIniP3D;
        SLCVFrame mInitialFrame;

        //Last Frame, KeyFrame and Relocalisation Info
        //KeyFrame* mpLastKeyFrame;
        SLCVFrame mLastFrame;

        //extractor instance
        ORB_SLAM2::ORBextractor* _extractor = NULL;
        // Initalization (only for monocular)
        Initializer* mpInitializer = NULL;

        SLCVMat _img;
};
//-----------------------------------------------------------------------------
#endif // SLCVTrackedMapping_H
