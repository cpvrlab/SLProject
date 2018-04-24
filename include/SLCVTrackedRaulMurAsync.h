//#############################################################################
//  File:      SLCVTrackedRaulMurAsync.h
//  Author:    Michael Goettlicher
//  Date:      Apr 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVTRACKERRAULMURASYNC_H
#define SLCVTRACKERRAULMURASYNC_H

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
#include <SLCVMap.h>
#include <SLTrackingInfosInterface.h>
#include <SLCVMapTracking.h>
#include <SLCVStateEstimator.h>
#include <SLCVOrbTracking.h>

class SLCVMapNode;

using namespace cv;

//-----------------------------------------------------------------------------
class SLCVTrackedRaulMurAsync : public SLCVTracked
{
public:
  SLCVTrackedRaulMurAsync(SLNode *node, ORBVocabulary* vocabulary,
			  SLCVKeyFrameDB* keyFrameDB, SLCVMap* map, SLCVMapNode* mapNode = NULL );
  SLbool track(SLCVMat imageGray,
	       SLCVMat image,
	       SLCVCalibration* calib,
	       SLbool drawDetection,
	       SLSceneView* sv);
  SLCVOrbTracking* orbTracking();
  
private:
    SLint                   _frameCount=0;    //!< NO. of frames since process start

    SLCVStateEstimator _stateEstimator;
    SLCVOrbTracking _orbTracking;
};
//-----------------------------------------------------------------------------
#endif
