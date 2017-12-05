//#############################################################################
//  File:      SLCVTrackedRaulMur.h
//  Author:    Michael Göttlicher
//  Date:      Dez 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This softwareis provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVTRACKERRAULMUR_H
#define SLCVTRACKERRAULMUR_H

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
#include <SLCVFrame.h>
#include <SLCVKeyFrameDB.h>

using namespace cv;

//-----------------------------------------------------------------------------
//! SLCVTrackedRaulMur is the main part of the AR Christoffelturm scene
/*! 
*/
class SLCVTrackedRaulMur : public SLCVTracked
{
public:
    SLCVTrackedRaulMur(SLNode *node, ORBVocabulary* vocabulary,
        SLCVKeyFrameDB* keyFrameDB);
    ~SLCVTrackedRaulMur();
    SLbool track(SLCVMat imageGray,
        SLCVMat image,
        SLCVCalibration* calib,
        SLbool drawDetection,
        SLSceneView* sv);

protected:
    bool Relocalization();

private:
    // ORB vocabulary used for place recognition and feature matching.
    ORBVocabulary* mpVocabulary;

    // KeyFrame database for place recognition (relocalization and loop detection).
    SLCVKeyFrameDB* mpKeyFrameDatabase;

    // Current Frame
    SLCVFrame mCurrentFrame;

    //extractor instance
    ORB_SLAM2::ORBextractor* _extractor = NULL;
};
//-----------------------------------------------------------------------------
#endif //SLCVTRACKERRAULMUR_H